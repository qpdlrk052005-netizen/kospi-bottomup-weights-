"""Stage 1: firm forecasters F^(k) and their cross-fitted outputs (rev4 spec section 3).

NN (poster Figure 1): trading GRU (+ financial GRU when quarterly data exist) -> fuse -> GAT on the firm graph
-> firm vector h_i; market GRU -> m_t; shared head([h_i, m_t]) -> r_hat_i (next-day firm return, %).
Lin: pooled ridge on [x_i, CapPct_i] (lambda = 1000, intercept unpenalised).

F^(k) uses only labels realised before 1 Jan k; early stopping on the last 20% of those days; it predicts only the
days of year k. Output: rev4/results/oof/{nn|lin}/k{year}_s{seed}.npz

  python -X utf8 rev4/code/r4_stage1.py --model nn --years 2016-2021 --seeds 0,1,2,3,4
"""
import argparse
from copy import deepcopy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r4_common import (RESULTS, FinancialSequences, GraphProvider, NeuralBatches, TEST_YEARS, load_data4, log_to,
                       year_days)

HIDDEN, MARKET_HIDDEN, HEADS, DROPOUT = 32, 16, 4, 0.1
LR, WEIGHT_DECAY, BATCH_DAYS = 1e-3, 1e-4, 16
SUB_DAYS = 8          # memory only: gradient accumulation inside each 16-day batch (see accumulate)
ES_FRACTION, PATIENCE = 0.2, 2
RIDGE_LAMBDA = 1000.0
OOF = RESULTS / "oof"


# ------------------------------------------------------------------ model
class GraphAttention(nn.Module):
    def __init__(self, hidden=HIDDEN, heads=HEADS):
        super().__init__()
        self.heads = heads
        self.project = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.a_src = nn.Parameter(torch.empty(heads, hidden // heads))
        self.a_dst = nn.Parameter(torch.empty(heads, hidden // heads))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edges):
        source, target = edges
        n = x.shape[0]
        z = self.project(F.dropout(x, DROPOUT, self.training)).reshape(n, self.heads, -1)
        logits = F.leaky_relu((z * self.a_src).sum(-1)[source] + (z * self.a_dst).sum(-1)[target], 0.2)
        idx = target[:, None].expand(-1, self.heads)
        with torch.no_grad():
            peak = logits.new_full((n, self.heads), -torch.inf).scatter_reduce_(0, idx, logits, "amax")
        num = (logits - peak[target]).exp()
        den = logits.new_zeros((n, self.heads)).index_add_(0, target, num)
        att = F.dropout(num / den[target], DROPOUT, self.training)
        agg = torch.zeros_like(z).index_add_(0, target, z[source] * att[..., None])
        return self.norm(x + self.out(F.elu(agg.reshape(n, -1))))


class FirmForecaster(nn.Module):
    def __init__(self, p, m, q=None):
        super().__init__()
        self.trade = nn.GRU(p + 2, HIDDEN, batch_first=True)
        self.fin = nn.GRU(q + 1, HIDDEN, batch_first=True) if q else None
        self.fuse = nn.Sequential(nn.Linear(HIDDEN * (2 if q else 1), HIDDEN), nn.GELU())
        self.graph = GraphAttention()
        self.market = nn.GRU(m, MARKET_HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HIDDEN + MARKET_HIDDEN, HIDDEN), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, b):
        h = self.trade(b["history"])[1][-1]
        if self.fin is not None:
            h = torch.cat([h, self.fin(b["fin_seq"])[1][-1]], -1)
        h = self.graph(self.fuse(h), b["edges"])
        m = self.market(b["market_history"])[1][-1]
        r = self.head(torch.cat([h, m[b["node_day"]]], -1)).squeeze(-1)
        return r, h, m


class Batches4(NeuralBatches):
    """rev2 batches + node->day index, flattened firm targets and (optional) financial sequences."""

    def __call__(self, days):
        b = super().__call__(days)
        lengths = torch.as_tensor(b["lengths"], device=self.device)
        b["node_day"] = torch.repeat_interleave(torch.arange(len(lengths), device=self.device), lengths)
        b["target"] = b["aux"][b["valid"]]
        b["target_ok"] = b["aux_valid"][b["valid"]]
        if getattr(self.folded, "fin", None) is not None:
            days = np.asarray(days)
            codes = np.concatenate([self.data.members[d] for d in days])
            node_days = np.repeat(days, [len(self.data.members[d]) for d in days])
            b["fin_seq"] = torch.as_tensor(FinancialSequences.gather(self.folded.fin, node_days, codes),
                                           device=self.device)
        return b


def firm_loss(r, b, lo, hi):
    y = b["target"].clamp(lo, hi)
    ok = b["target_ok"].float()
    err = (r - y).square() * ok
    n_days = len(b["lengths"])
    s = err.new_zeros(n_days).index_add_(0, b["node_day"], err)
    n = err.new_zeros(n_days).index_add_(0, b["node_day"], ok)
    present = n > 0
    return (s[present] / n[present]).mean()


def accumulate(model, batches, chunk, lo, hi, opt, sub_days):
    """Gradient of the 16-day batch loss built from sub-batches. The loss is a mean over days and days do not interact
    (graph edges and the market GRU are per day), so weighting each sub-batch mean by its share of days gives exactly
    the full-batch gradient; only the memory peak changes (the 16-day backward needed 1.93 GB on 2026 days)."""
    opt.zero_grad(set_to_none=True)
    for j in range(0, len(chunk), sub_days):
        part = chunk[j:j + sub_days]
        b = batches(part)
        loss = firm_loss(model(b)[0], b, lo, hi) * (len(part) / len(chunk))
        loss.backward()
        del b, loss


def target_bounds(data, days):
    y = data.firm_y[days][data.universe[days]]
    return [float(v) for v in np.nanquantile(y[np.isfinite(y)], [0.01, 0.99])]


def train_nn(data, fold, train_days, seed, device, epochs, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    order = np.sort(train_days)
    n_es = max(1, int(round(len(order) * ES_FRACTION)))
    fit_days, es_days = order[:-n_es], order[-n_es:]
    lo, hi = target_bounds(data, fit_days)
    batches = Batches4(data, fold, device, GraphProvider(data))
    q = fold.fin["z"].shape[1] if fold.fin is not None else None
    model = FirmForecaster(fold.stock.shape[-1], fold.market.shape[-1], q).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    best, state, bad, history = np.inf, None, 0, []
    sub_days = SUB_DAYS
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(fit_days)
        for i in range(0, len(perm), BATCH_DAYS):
            chunk = perm[i:i + BATCH_DAYS]
            while True:
                try:
                    accumulate(model, batches, chunk, lo, hi, opt, sub_days)
                    break
                except torch.OutOfMemoryError:
                    if sub_days == 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    sub_days //= 2
                    log(f"  CUDA memory: gradient accumulation over {sub_days}-day sub-batches from now on")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            es = float(np.mean([firm_loss(model(b)[0], b, lo, hi).item()
                                for b in (batches(es_days[i:i + BATCH_DAYS]) for i in range(0, len(es_days), BATCH_DAYS))]))
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if str(device).startswith("cuda") else 0.0
        history.append({"epoch": epoch, "es_loss": es, "seconds": round(time.time() - t0, 1), "peak_gb": round(peak, 2)})
        log(f"  seed {seed} epoch {epoch}: early-stop loss {es:.5f} ({history[-1]['seconds']} s, peak {peak:.2f} GB)")
        if es < best - 1e-7:
            best, state, bad = es, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, batches, {"best_es_loss": best, "history": history, "target_bounds": [lo, hi],
                            "fit_days": int(len(fit_days)), "es_days": int(len(es_days)), "graph": batches.graphs.kind}


@torch.no_grad()
def predict_nn(model, batches, days):
    r_all, h_all, m_all, lengths = [], [], [], []
    for i in range(0, len(days), BATCH_DAYS):
        b = batches(days[i:i + BATCH_DAYS])
        r, h, m = model(b)
        r_all.append(r.float().cpu().numpy())
        h_all.append(h.half().cpu().numpy())
        m_all.append(m.half().cpu().numpy())
        lengths += b["lengths"]
    return {"r": np.concatenate(r_all), "h": np.concatenate(h_all), "m": np.concatenate(m_all),
            "lengths": np.asarray(lengths)}


# ------------------------------------------------------------------ linear forecaster
def lin_rows(data, fold, day):
    ids = data.members[day]
    return ids, np.concatenate([np.ones((len(ids), 1)), fold.stock[day, ids], fold.cap_pct[day, ids, None]], axis=1)


def train_lin(data, fold, train_days):
    lo, hi = target_bounds(data, train_days)
    k = fold.stock.shape[-1] + 2
    a, c = np.zeros((k, k)), np.zeros(k)
    for day in train_days:
        ids, z = lin_rows(data, fold, day)
        y = data.firm_y[day, ids]
        ok = np.isfinite(y)
        a += z[ok].T @ z[ok]
        c += z[ok].T @ np.clip(y[ok], lo, hi)
    penalty = np.r_[0.0, np.full(k - 1, RIDGE_LAMBDA)]
    return np.linalg.solve(a + np.diag(penalty), c), {"target_bounds": [lo, hi]}


def predict_lin(data, fold, beta, days):
    r, x, lengths = [], [], []
    for day in days:
        ids, z = lin_rows(data, fold, day)
        r.append((z @ beta).astype(np.float32))
        x.append(fold.stock[day, ids].astype(np.float16))
        lengths.append(len(ids))
    return {"r": np.concatenate(r), "x": np.concatenate(x), "lengths": np.asarray(lengths)}


# ------------------------------------------------------------------ driver
def parse_years(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in text.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("nn", "lin"), required=True)
    ap.add_argument("--years", default="2016-2021")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-test-years", action="store_true",
                    help="needed for k >= 2022; run only after results/plan_locked.json exists")
    ap.add_argument("--precompute-test", action="store_true",
                    help="train test-year forecasters before the lock with the frozen stage-1 setting; only the "
                         "forecasts are stored, no test performance is computed (stage 2 --phase test still needs the lock)")
    ap.add_argument("--max-train-days", type=int, default=0, help="smoke test only: keep the latest N training days")
    ap.add_argument("--out", default=str(OOF))
    args = ap.parse_args()
    years, seeds = parse_years(args.years), [int(s) for s in args.seeds.split(",")]
    if any(y in TEST_YEARS for y in years) and not args.precompute_test:
        if not args.allow_test_years or not (RESULTS / "plan_locked.json").exists():
            raise SystemExit("Test-year forecasters need --allow-test-years and a locked plan (or --precompute-test).")
    from pathlib import Path
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "stage1.log")
    data, gate = load_data4()
    log(f"inputs: {json.dumps(gate['inputs'])}")
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        # Windows driver 545 falls back to system RAM when the 4 GB card is full instead of failing, so the caching
        # allocator never frees its cache: the 2026-11-20 run reached 13.6 GB committed RAM, swapped, and slowed per
        # day (0.032 -> 0.043 s and rising). A cap makes PyTorch free cached blocks and retry inside VRAM.
        torch.cuda.set_per_process_memory_fraction(0.85)
    for seed in (seeds if args.model == "nn" else [0]):     # seed-major: a finished seed covers every year
        for k in years:
            path = out / f"k{k}_s{seed}.npz"
            if path.exists():
                continue
            train_days = data.expanding(k)[0]
            if args.max_train_days:
                train_days = np.sort(train_days)[-args.max_train_days:]
            days = year_days(data, k)
            t0 = time.time()
            fold = data.scale(train_days)
            log(f"{args.model} k={k} seed={seed}: {len(train_days)} training days, {len(days)} forecast days")
            if args.model == "nn":
                model, batches, info = train_nn(data, fold, train_days, seed, args.device, args.epochs, log)
                pred = predict_nn(model, batches, days)
                del model, batches
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                beta, info = train_lin(data, fold, train_days)
                pred = predict_lin(data, fold, beta, days)
            info.update({"k": k, "seed": seed, "train_days": int(len(train_days)), "seconds": round(time.time() - t0, 1),
                         "smoke_max_train_days": args.max_train_days, "inputs": gate["inputs"]})
            np.savez_compressed(path, days=days, info=json.dumps(info), **pred)
            log(f"saved {path.name} in {info['seconds']} s")
            del fold


if __name__ == "__main__":
    main()
