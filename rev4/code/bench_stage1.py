"""Competing firm forecasters for the firm-level superiority evaluation (2026-11-21, after the lock; exploratory).

Pre-test years only: forecasters F^(k), k = 2016..2021, trained on labels before 1 Jan k (same cross-fitting as
r4_stage1). Test years (2022+) are refused. Output: <out>/<model>/k{year}_s{seed}.npz (days, r, lengths), the same
node order as rev4/results/oof (data.members[day]).

Shared protocol (identical to our NN unless stated):
  information  daily firm features (41) + CapPct + universe indicator + latest quarterly ratios (15) + present flag
               at every step of the look-back window; market features (60) where the model has a market input
  target       next-day firm return (%), clipped at the training 1%/99% quantiles; loss = mean over days of the
               day's mean squared error (master_csz: MASTER's own label, daily z-score after dropping 2.5% each tail)
  optimiser    AdamW, weight decay 1e-4, 16-day batches, gradient clip 1.0, early stopping on the last 20% of the
               training days, patience 2, at most 20 epochs (ours: 10)
  learning rate  searched per fold over (1e-3, 3e-4, 1e-4) with seed 0 by the early-stopping loss (ours: fixed 1e-3,
               never tuned) -> the comparison favours the competitors
Models
  mlp     current-day features -> 64 -> 32 -> 1 (no history, no market)
  gru     GRU(32) over the 60-day window -> head (our trading branch alone: no graph, no financial GRU, no market)
  dtml    Yoo et al. (KDD 2021), re-implemented from the paper (no official code): attention LSTM per stock and for
          the market, multi-level context c_u + beta*c_mkt (beta 0.1), data-axis transformer across the day's stocks
          (4 heads), tanh(h + MLP(h)) -> linear; window 10, hidden 64
  master  Li et al. (AAAI 2024), official architecture (github SJTU-DMTai/MASTER): market-guided feature gate
          (softmax temperature beta 2, the CSI800 value), linear embedding + sinusoidal PE, intra-stock attention
          (4 heads, unscaled as in the code), inter-stock attention per step (2 heads), temporal attention, linear
          decoder; d_model 256, dropout 0.5, look-back 8; bf16 autocast (speed only)
  master_csz  MASTER with its own label normalisation (cross-sectional z-score, 2.5% tails dropped)
  lgb     LightGBM on the current-day features (pooled), early stopping on the same last-20% days

  python -u -X utf8 rev4/code/bench_stage1.py --models mlp,gru,dtml,master,master_csz,lgb --years 2016-2021 --seeds 0,1,2
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r4_common import RESULTS, TEST_YEARS, load_data4, log_to, year_days
from r4_stage1 import parse_years, target_bounds

OUT = RESULTS / "bench" / "oof"
ES_FRACTION, PATIENCE, BATCH_DAYS, WD, MAX_EPOCHS = 0.2, 2, 16, 1e-4, 20
LR_GRID = (1e-3, 3e-4, 1e-4)
LOOKBACK = {"mlp": 1, "gru": 60, "dtml": 10, "master": 8, "master_csz": 8}
SUB_DAYS = {"mlp": 16, "gru": 8, "dtml": 8, "master": 8, "master_csz": 8}


# ------------------------------------------------------------------ batches built on the GPU
class GPUBatcher:
    """Same content as rev2 NeuralBatches (scaled firm features zeroed outside the universe, CapPct, indicator) plus the
    latest quarterly ratios at every step; built by indexing tensors kept on the device."""

    def __init__(self, data, fold, device, T):
        self.T, self.dev = T, device
        self.stock = torch.as_tensor(fold.stock, device=device)
        self.univ = torch.as_tensor(data.universe, device=device)
        self.cap = torch.as_tensor(fold.cap_pct, device=device)
        self.market = torch.as_tensor(fold.market, device=device)
        self.y = torch.as_tensor(data.firm_y, device=device)
        width = max(len(m) for m in data.members)
        M = np.full((len(data.members), width), -1, np.int64)
        for d, m in enumerate(data.members):
            M[d, :len(m)] = m
        self.M = torch.as_tensor(M, device=device)
        self.fz = self.fl = None
        if fold.fin is not None:
            self.fz = torch.as_tensor(fold.fin["z"], device=device)
            self.fl = torch.as_tensor(fold.fin["latest"].astype(np.int64), device=device)
        self.n_features = fold.stock.shape[-1] + 2 + (self.fz.shape[1] + 1 if self.fz is not None else 0)
        self.n_market = fold.market.shape[-1]

    def __call__(self, days):
        days = torch.as_tensor(np.asarray(days, np.int64), device=self.dev)
        M = self.M[days]
        valid = M >= 0
        width = int(valid.sum(1).max())
        M, valid = M[:, :width], valid[:, :width]
        node_day = torch.nonzero(valid)[:, 0]
        code = M[valid]
        steps = torch.arange(-self.T + 1, 1, device=self.dev)
        win = days[node_day, None] + steps
        c = code[:, None].expand(-1, self.T)
        inu = self.univ[win, c]
        parts = [self.stock[win, c], (self.cap[win, c] * inu)[..., None], inu[..., None].float()]
        if self.fz is not None:
            li = self.fl[win, c]
            ok = li >= 0
            parts += [self.fz[li.clamp(min=0)] * ok[..., None], ok[..., None].float()]
        y = self.y[days[node_day], code]
        ok = torch.isfinite(y)
        mwin = days[:, None] + steps
        return {"hist": torch.cat(parts, -1), "node_day": node_day, "valid": valid, "n_days": len(days),
                "mhist": self.market[mwin], "mlast": self.market[days], "target": torch.nan_to_num(y),
                "target_ok": ok}


def pad(flat, valid):
    out = flat.new_zeros((*valid.shape, *flat.shape[1:]))
    out[valid] = flat
    return out


# ------------------------------------------------------------------ models
class MLP(nn.Module):
    def __init__(self, f, m, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, b):
        return self.net(b["hist"][:, -1]).squeeze(-1)


class GRUOnly(nn.Module):
    def __init__(self, f, m, hidden=32, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(f, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, b):
        return self.head(self.gru(b["hist"])[1][-1]).squeeze(-1)


class AttnLSTM(nn.Module):
    """DTML attentive context: tanh feature transform -> LSTM -> attention with the last hidden state as query ->
    context normalisation (layer norm)."""

    def __init__(self, f, h):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.lstm = nn.LSTM(h, h, batch_first=True)
        self.norm = nn.LayerNorm(h)

    def forward(self, x):
        hs = self.lstm(torch.tanh(self.inp(x)))[0]
        a = torch.softmax((hs * hs[:, -1:]).sum(-1), dim=1)
        return self.norm((a[..., None] * hs).sum(1))


class DTML(nn.Module):
    def __init__(self, f, m, h=64, heads=4, beta=0.1, dropout=0.1):
        super().__init__()
        self.stock, self.mkt, self.beta = AttnLSTM(f, h), AttnLSTM(m, h), beta
        self.att = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.mlp = nn.Sequential(nn.Linear(h, 4 * h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * h, h))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(h, 1)

    def forward(self, b):
        ctx = self.stock(b["hist"]) + self.beta * self.mkt(b["mhist"])[b["node_day"]]
        P, valid = pad(ctx, b["valid"]), b["valid"]
        a = self.att(P, P, P, key_padding_mask=~valid, need_weights=False)[0]
        h = self.norm(P + self.drop(a))
        h = torch.tanh(h + self.mlp(h))
        return self.out(self.drop(h))[valid].squeeze(-1)


class MasterBlock(nn.Module):
    """TAttention / SAttention of the official MASTER code: norm1 -> per-head attention (Q, K, V without bias, no output
    projection, dropout on the attention matrix) -> residual on the normalised input -> norm2 -> + FFN."""

    def __init__(self, d, nhead, dropout, scaled):
        super().__init__()
        self.nhead, self.scaled = nhead, scaled
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att_drop = nn.ModuleList(nn.Dropout(dropout) for _ in range(nhead))
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d), nn.Dropout(dropout))

    def forward(self, x, key_mask=None):          # x (B, L, d); key_mask (B, L) True = valid key
        x = self.norm1(x)
        B, L, d = x.shape
        dim = d // self.nhead                     # d_model 256 splits evenly, so contiguous slices = standard heads
        q, k, v = (t.reshape(B, L, self.nhead, dim).transpose(1, 2) for t in (self.q(x), self.k(x), self.v(x)))
        mask = key_mask[:, None, None, :] if key_mask is not None else None
        p = self.att_drop[0].p if self.training else 0.0
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p,
                                             scale=1 / math.sqrt(dim) if self.scaled else 1.0)
        xt = self.norm2(x + att.transpose(1, 2).reshape(B, L, d))
        return xt + self.ffn(xt)


class Master(nn.Module):
    def __init__(self, f, m, d=256, t_nhead=4, s_nhead=2, dropout=0.5, beta=2.0, max_len=100):
        super().__init__()
        self.gate, self.beta, self.f = nn.Linear(m, f), beta, f
        self.embed = nn.Linear(f, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.intra = MasterBlock(d, t_nhead, dropout, scaled=False)
        self.inter = MasterBlock(d, s_nhead, dropout, scaled=True)
        self.temporal = nn.Linear(d, d, bias=False)
        self.decoder = nn.Linear(d, 1)

    def forward(self, b):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b["hist"].is_cuda):   # speed only (1.6x)
            return self._forward(b).float()

    def _forward(self, b):
        x = b["hist"]
        g = self.f * torch.softmax(self.gate(b["mlast"]) / self.beta, -1)
        x = x * g[b["node_day"]][:, None, :]
        z = self.embed(x) + self.pe[:x.shape[1]]
        z = self.intra(z)                                         # (n, T, d)
        valid = b["valid"]
        P = pad(z, valid)                                         # (D, W, T, d)
        D, W, T, d = P.shape
        S = P.permute(0, 2, 1, 3).reshape(D * T, W, d)
        mask = valid[:, None, :].expand(D, T, W).reshape(D * T, W)
        S = self.inter(S, mask).reshape(D, T, W, d).permute(0, 2, 1, 3)
        z = S[valid]                                              # (n, T, d)
        h = self.temporal(z)
        lam = torch.softmax((h * h[:, -1:]).sum(-1), dim=1)
        return self.decoder((lam[..., None] * z).sum(1)).squeeze(-1)


MODELS = {"mlp": MLP, "gru": GRUOnly, "dtml": DTML, "master": Master, "master_csz": Master}


# ------------------------------------------------------------------ training
def day_loss(r, b, lo, hi, csz=False):
    y, ok = b["target"].clamp(lo, hi), b["target_ok"]
    nd = b["node_day"]
    if csz:                                                       # MASTER label: drop 2.5% tails, daily z-score
        keep = torch.zeros_like(ok)
        yz = torch.zeros_like(y)
        for d in range(b["n_days"]):
            idx = torch.nonzero((nd == d) & ok)[:, 0]
            if len(idx) < 10:
                continue
            k = int(0.025 * len(idx))
            order = idx[torch.argsort(y[idx])]
            mid = order[k:len(order) - k] if k > 0 else order
            v = y[mid]
            yz[mid] = (v - v.mean()) / (v.std() + 1e-8)
            keep[mid] = True
        y, ok = yz, keep
    err = (r - y).square() * ok.float()
    s = err.new_zeros(b["n_days"]).index_add_(0, nd, err)
    n = err.new_zeros(b["n_days"]).index_add_(0, nd, ok.float())
    present = n > 0
    return (s[present] / n[present]).mean()


def train_one(name, batcher, fit_days, es_days, lo, hi, lr, seed, device, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MODELS[name](batcher.n_features, batcher.n_market).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    csz = name == "master_csz"
    sub = SUB_DAYS[name]
    best, state, bad, hist = np.inf, None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(fit_days)
        for i in range(0, len(perm), BATCH_DAYS):
            chunk = perm[i:i + BATCH_DAYS]
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    for j in range(0, len(chunk), sub):
                        part = chunk[j:j + sub]
                        b = batcher(part)
                        (day_loss(model(b), b, lo, hi, csz) * (len(part) / len(chunk))).backward()
                        del b
                    break
                except torch.OutOfMemoryError:
                    if sub == 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    sub //= 2
                    log(f"  CUDA memory: {sub}-day sub-batches from now on")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            es = float(np.mean([day_loss(model(b), b, lo, hi, csz).item()
                                for b in (batcher(es_days[i:i + sub]) for i in range(0, len(es_days), sub))]))
        if not np.isfinite(es):
            log(f"  {name} lr {lr:g} seed {seed} epoch {epoch}: non-finite early-stop loss, stop")
            break
        hist.append({"epoch": epoch, "es_loss": es, "seconds": round(time.time() - t0, 1)})
        log(f"  {name} lr {lr:g} seed {seed} epoch {epoch}: early-stop loss {es:.5f} ({hist[-1]['seconds']} s)")
        if es < best - 1e-7:
            best, state, bad = es, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state is None:
        return None, np.inf, hist
    model.load_state_dict(state)
    model.eval()
    return model, best, hist


@torch.no_grad()
def predict(model, batcher, days, sub):
    out = []
    for i in range(0, len(days), sub):
        out.append(model(batcher(days[i:i + sub])).float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ LightGBM
def lgb_rows(data, fold, days):
    xs, ys, lens = [], [], []
    fin = fold.fin
    for d in days:
        ids = data.members[d]
        parts = [fold.stock[d, ids], fold.cap_pct[d, ids, None]]
        if fin is not None:
            li = fin["latest"][d, ids]
            ok = li >= 0
            parts += [fin["z"][np.maximum(li, 0)] * ok[:, None], ok[:, None].astype(np.float32)]
        xs.append(np.concatenate(parts, 1).astype(np.float32))
        ys.append(data.firm_y[d, ids])
        lens.append(len(ids))
    return np.concatenate(xs), np.concatenate(ys), np.asarray(lens)


def run_lgb(data, fold, fit_days, es_days, days, lo, hi, seed):
    import lightgbm as lgb
    xf, yf, _ = lgb_rows(data, fold, fit_days)
    xe, ye, _ = lgb_rows(data, fold, es_days)
    kf, ke = np.isfinite(yf), np.isfinite(ye)
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "seed": seed, "verbose": -1, "num_threads": 6}
    booster = lgb.train(params, lgb.Dataset(xf[kf], np.clip(yf[kf], lo, hi)), num_boost_round=2000,
                        valid_sets=[lgb.Dataset(xe[ke], np.clip(ye[ke], lo, hi))],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    xt, _, lens = lgb_rows(data, fold, days)
    return booster.predict(xt, num_iteration=booster.best_iteration).astype(np.float32), lens, \
        {"best_iteration": int(booster.best_iteration), "es_l2": float(booster.best_score["valid_0"]["l2"])}


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,gru,dtml,master,master_csz,lgb")
    ap.add_argument("--years", default="2016-2021")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-train-days", type=int, default=0, help="smoke test only")
    ap.add_argument("--allow-test-years", action="store_true",
                    help="extension to 2022-2026, decided 2026-11-21 13:45 after the pre-test evaluation; metrics and "
                         "decision rule unchanged (BENCH_PROTOCOL_KO.md 5-1)")
    args = ap.parse_args()
    years, seeds = parse_years(args.years), [int(s) for s in args.seeds.split(",")]
    if any(y in TEST_YEARS for y in years) and not args.allow_test_years:
        raise SystemExit("Pre-test years only (2016-2021) unless --allow-test-years is given.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "bench_stage1.log")
    data, _ = load_data4()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.85)
    for name in args.models.split(","):
        (out / name).mkdir(exist_ok=True)
        for k in years:
            todo = [s for s in seeds if not (out / name / f"k{k}_s{s}.npz").exists()]
            if not todo:
                continue
            train_days = np.sort(data.expanding(k)[0])
            if args.max_train_days:
                train_days = train_days[-args.max_train_days:]
            days = year_days(data, k)
            n_es = max(1, int(round(len(train_days) * ES_FRACTION)))
            fit_days, es_days = train_days[:-n_es], train_days[-n_es:]
            lo, hi = target_bounds(data, fit_days)
            fold = data.scale(train_days)
            log(f"{name} k={k}: {len(train_days)} training days, {len(days)} forecast days, seeds {todo}")
            if name == "lgb":
                for s in todo:
                    t0 = time.time()
                    r, lens, info = run_lgb(data, fold, fit_days, es_days, days, lo, hi, s)
                    info |= {"k": k, "seed": s, "seconds": round(time.time() - t0, 1)}
                    np.savez_compressed(out / name / f"k{k}_s{s}.npz", days=days, r=r, lengths=lens, info=json.dumps(info))
                    log(f"  saved lgb k{k}_s{s}: {info}")
                del fold
                continue
            batcher = GPUBatcher(data, fold, args.device, LOOKBACK[name])
            lr_file = out / name / f"k{k}_lr.json"
            if lr_file.exists():
                lr = json.loads(lr_file.read_text())["lr"]
            else:                                                  # per-fold learning-rate search, seed 0
                trials = []
                for cand in LR_GRID:
                    t0 = time.time()
                    model, best, hist = train_one(name, batcher, fit_days, es_days, lo, hi, cand, 0, args.device, log)
                    trials.append({"lr": cand, "es_loss": best, "epochs": len(hist), "seconds": round(time.time() - t0, 1)})
                    if model is not None and best <= min(t["es_loss"] for t in trials):
                        best_model = model
                    del model
                lr = min(trials, key=lambda t: t["es_loss"])["lr"]
                lr_file.write_text(json.dumps({"lr": lr, "trials": trials}, indent=1))
                log(f"  {name} k={k}: learning rate {lr:g} ({trials})")
                if 0 in todo:
                    r = predict(best_model, batcher, days, SUB_DAYS[name])
                    lens = np.array([len(data.members[d]) for d in days])
                    np.savez_compressed(out / name / f"k{k}_s0.npz", days=days, r=r, lengths=lens,
                                        info=json.dumps({"k": k, "seed": 0, "lr": lr, "trials": trials}))
                    todo = [s for s in todo if s != 0]
                del best_model
            for s in todo:
                t0 = time.time()
                model, best, hist = train_one(name, batcher, fit_days, es_days, lo, hi, lr, s, args.device, log)
                r = predict(model, batcher, days, SUB_DAYS[name])
                lens = np.array([len(data.members[d]) for d in days])
                np.savez_compressed(out / name / f"k{k}_s{s}.npz", days=days, r=r, lengths=lens,
                                    info=json.dumps({"k": k, "seed": s, "lr": lr, "es_loss": best, "history": hist,
                                                     "seconds": round(time.time() - t0, 1)}))
                log(f"  saved {name} k{k}_s{s} ({round(time.time() - t0, 1)} s)")
                del model
            del batcher, fold
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
