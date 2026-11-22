"""Stage 2: combiners on cross-fitted firm forecasts, nested selection, plan lock, single test, report.

Models (POSTER_RUN_PLAN.md section 2):
  B0, B3R                          baselines
  Lin-cap, Lin-corr, Lin-delta     linear firm forecasts (stage-1 'lin')
  NN-cap, NN-corr, NN-delta-simple, NN-delta   neural firm forecasts (stage-1 'nn', seed mean z_bar, spread sigma)
For evaluation year v: settings (tau, steps, lambda) are chosen by training on cross-fitted years 2016..v-2 and
scoring year v-1; the chosen setting is refit on 2016..v-1 and evaluated once on year v.

  python -X utf8 rev4/code/r4_stage2.py --phase validation     # years 2018-2021
  python -X utf8 rev4/code/r4_stage2.py --phase lock           # pick the best of 7 on validation, freeze the plan
  python -X utf8 rev4/code/r4_stage2.py --phase test           # years 2022-2026, needs the lock
  python -X utf8 rev4/code/r4_stage2.py --phase report         # tests, tables for the poster
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from r4_common import (REV4, RESULTS, ROOT, STAGE2_FIRST_YEAR, TEST_YEARS, VALIDATION_YEARS, b0_b3r, file_sha,
                       load_data4, log_to, save_json, weights_and_kl)

sys.path.insert(0, str(ROOT / "explore_delta"))
from pilot_eval import differential_test  # noqa: E402

TAUS = (0.0, 0.001, 0.01, 0.1)
CORR_LAMBDAS = (10.0, 100.0, 1000.0, 10000.0)
BIL_LR, BIL_STEPS, BIL_EVERY, BIL_PATIENCE = 1e-2, 1000, 10, 200
MLP_LR, MLP_WD, MLP_STEPS, MLP_EVERY, MLP_PATIENCE, MLP_BATCH, MLP_HIDDEN = 1e-3, 1e-4, 1500, 50, 300, 64, 32
CANDIDATES = ("Lin-cap", "Lin-corr", "Lin-delta", "NN-cap", "NN-corr", "NN-delta-simple", "NN-delta")
RQ = ("NN-delta", "NN-cap")
MARKET_CLIP = 5.0


# ------------------------------------------------------------------ assemble cross-fitted forecasts
def load_oof(folder, data):
    """year -> {'days','lengths','r' (S,N),'h' (S,N,H)|None,'m' (S,T,M)|None,'x' (N,p)|None, 'seeds'}"""
    out = {}
    for path in sorted(Path(folder).glob("k*_s*.npz")):
        k, s = (int(v) for v in path.stem[1:].split("_s"))
        z = np.load(path, allow_pickle=False)
        entry = out.setdefault(k, {"days": z["days"], "lengths": z["lengths"], "r": {}, "h": {}, "m": {}, "x": None})
        if not np.array_equal(entry["days"], z["days"]) or not np.array_equal(entry["lengths"], z["lengths"]):
            raise ValueError(f"{path.name}: days/universe differ across seeds.")
        expected = np.array([len(data.members[d]) for d in z["days"]])
        if not np.array_equal(expected, z["lengths"]):
            raise ValueError(f"{path.name}: universe sizes do not match the data.")
        entry["r"][s] = z["r"]
        if "h" in z:
            entry["h"][s], entry["m"][s] = z["h"], z["m"]
        if "x" in z:
            entry["x"] = z["x"]
    for k, e in out.items():
        seeds = sorted(e["r"])
        e["seeds"] = seeds
        e["r"] = np.stack([e["r"][s] for s in seeds])
        e["h"] = np.stack([e["h"][s] for s in seeds]) if e["h"] else None
        e["m"] = np.stack([e["m"][s] for s in seeds]) if e["m"] else None
    return out


class Panel:
    """All cross-fitted years of one base forecaster, padded to a common width."""

    def __init__(self, data, oof, years):
        years = [y for y in years if y in oof]
        seeds = {tuple(oof[y]["seeds"]) for y in years}
        if len(seeds) != 1:
            raise ValueError(f"Seed sets differ across years: {seeds}")
        self.seeds = list(seeds.pop())
        blocks = [oof[y] for y in years]
        self.days = np.concatenate([b["days"] for b in blocks])
        self.year = np.concatenate([np.full(len(b["days"]), y) for y, b in zip(years, blocks)])
        lengths = np.concatenate([b["lengths"] for b in blocks])
        T, W = len(self.days), int(lengths.max())
        self.valid = np.arange(W)[None, :] < lengths[:, None]

        def pad(flat, dtype=np.float32):
            o = np.zeros((T, W) + flat.shape[1:], dtype)
            o[self.valid] = flat
            return o

        r = np.concatenate([b["r"] for b in blocks], axis=1).astype(np.float64)
        self.zbar = pad(r.mean(0))
        self.sigma = pad(r.std(0)) if r.shape[0] > 1 else np.zeros((T, W), np.float32)
        cap = np.concatenate([data.cap[d, data.members[d]] / data.cap[d, data.members[d]].sum() for d in self.days])
        self.p = pad(cap)
        self.cap_pct = pad(np.concatenate([data.cap_pct[d, data.members[d]] for d in self.days]))
        self.codes = pad(np.concatenate([data.members[d] for d in self.days]), np.int64)
        self.y = data.y[self.days].astype(np.float32)
        self.s_raw = data.market[self.days].astype(np.float64)
        self.h = (np.stack([pad(np.concatenate([b["h"][i] for b in blocks]), np.float16) for i in range(len(self.seeds))])
                  if blocks[0]["h"] is not None else None)
        self.m = np.concatenate([b["m"] for b in blocks], axis=1) if blocks[0]["m"] is not None else None
        self.x = pad(np.concatenate([b["x"] for b in blocks]), np.float32) if blocks[0]["x"] is not None else None
        self.cap_fc = (self.p * self.zbar).sum(1)
        self.d = (self.zbar - self.cap_fc[:, None]) * self.valid

    def rows(self, years):
        return np.flatnonzero(np.isin(self.year, years))


def scalers(panel, rows):
    """Stage-2 standardisation from the stage-2 training rows only."""
    s = panel.s_raw[rows]
    mu, sd = np.nanmean(s, 0), np.nanstd(s, 0)
    sd = np.where(sd > 0, sd, 1.0)
    v = panel.valid[rows]
    d_sd = float(panel.d[rows][v].std()) or 1.0
    sg_sd = float(panel.sigma[rows][v].std()) or 1.0
    return {"s_mu": mu, "s_sd": sd, "d_sd": d_sd, "sigma_sd": sg_sd}


def tensors(panel, rows, sc, device="cpu"):
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
    s = np.clip(np.nan_to_num((panel.s_raw[rows] - sc["s_mu"]) / sc["s_sd"]), -MARKET_CLIP, MARKET_CLIP)
    valid = t(panel.valid[rows], torch.bool)
    log_p = torch.log(t(panel.p[rows]).clamp_min(1e-30)).masked_fill(~valid, 0.0)
    out = {"rows": rows, "valid": valid, "log_p": log_p, "zbar": t(panel.zbar[rows]), "y": t(panel.y[rows]),
           "s": t(s), "d": t(panel.d[rows] / sc["d_sd"]), "sigma": t(panel.sigma[rows] / sc["sigma_sd"]),
           "cap_pct": t(panel.cap_pct[rows])}
    if panel.x is not None:
        out["x"] = t(panel.x[rows])
    return out


# ------------------------------------------------------------------ combiners
def corr_design(panel, rows, sc):
    s = np.clip(np.nan_to_num((panel.s_raw[rows] - sc["s_mu"]) / sc["s_sd"]), -MARKET_CLIP, MARKET_CLIP)
    v = (panel.p[rows] * panel.d[rows] ** 2).sum(1)
    sg = (panel.p[rows] * panel.sigma[rows]).sum(1)
    return np.column_stack([s, v, v[:, None] * s, sg])


def fit_corr(panel, train, lam):
    sc = scalers(panel, train)
    x = corr_design(panel, train, sc)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    target = panel.y[train] - panel.cap_fc[train]
    beta = np.linalg.solve(z.T @ z + np.diag(np.r_[0.0, np.full(z.shape[1] - 1, lam)]), z.T @ target)

    def predict(rows):
        xr = corr_design(panel, rows, sc)
        return panel.cap_fc[rows] + np.column_stack([np.ones(len(xr)), (xr - mu) / sd]) @ beta
    return predict


class Bilinear(nn.Module):
    def __init__(self, q, m):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(q, 1 + m))

    def forward(self, f, s):
        g = torch.cat([torch.ones_like(s[:, :1]), s], 1)
        return torch.einsum("twq,qk,tk->tw", f, self.theta, g)


def bilinear_features(t, kind):
    if kind == "Lin-delta":
        return torch.cat([t["x"], t["d"][..., None]], -1)
    return torch.stack([t["d"], t["sigma"], t["cap_pct"]], -1)


def forward_bilinear(model, t, kind):
    delta = model(bilinear_features(t, kind), t["s"]).masked_fill(~t["valid"], 0.0)
    w, kl = weights_and_kl(t["log_p"], delta, t["valid"])
    return (w * t["zbar"]).sum(-1), w, kl


def cap_shares(t):
    """Market-cap shares p of a tensor block (0 on padding)."""
    return (t["log_p"].exp() * t["valid"]).numpy()


def fit_bilinear(t_tr, kind, tau, steps, t_va=None):
    """curve rows: (step, validation MSE, validation moved share, last training index MSE). The moved share and the
    training loss separate "cannot move" (optimisation failure) from "moving hurts out of sample" (POSTER_RUN_PLAN 8.2)."""
    torch.manual_seed(0)
    q = bilinear_features(t_tr, kind).shape[-1]
    model = Bilinear(q, t_tr["s"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=BIL_LR)
    curve, best = [], (np.inf, 0)
    p_va = cap_shares(t_va) if t_va is not None else None
    last_train = float("nan")
    for step in range(steps + 1):
        if t_va is not None and step % BIL_EVERY == 0:
            with torch.no_grad():
                pred_t, w_va, _ = forward_bilinear(model, t_va, kind)
            pred = pred_t.numpy().astype(np.float64)
            mse = float(np.square(pred - t_va["y"].numpy()).mean())
            moved = float(0.5 * np.abs(w_va.numpy() - p_va).sum(1).mean())
            curve.append((step, mse, moved, last_train))
            if mse < best[0]:
                best = (mse, step)
            if step - best[1] >= BIL_PATIENCE:
                break
        if step == steps:
            break
        opt.zero_grad()
        pred, _, kl = forward_bilinear(model, t_tr, kind)
        index_mse = (t_tr["y"] - pred).square().mean()
        (index_mse + tau * kl.mean()).backward()
        last_train = float(index_mse.detach())
        opt.step()
    return model, curve


class SeedMLPs(nn.Module):
    """One weight MLP per stage-1 seed: delta = g([h_i^s, m_t^s, d_i, sigma_i]) (Figure 1, stage 2)."""

    def __init__(self, seeds, in_dim, hidden=MLP_HIDDEN):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(seeds, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(seeds, hidden))
        self.w2 = nn.Parameter(torch.empty(seeds, hidden))
        self.b2 = nn.Parameter(torch.zeros(seeds))
        for s in range(seeds):
            nn.init.kaiming_uniform_(self.w1.data[s].T, a=5 ** 0.5)
        nn.init.normal_(self.w2, std=0.01)
        self.drop = nn.Dropout(0.1)

    def forward(self, h, m, d, sigma):
        # h (S,B,W,H) m (S,B,M) d,sigma (B,W) -> delta (S,B,W)
        S, B, W, _ = h.shape
        x = torch.cat([h, m[:, :, None, :].expand(S, B, W, m.shape[-1]),
                       d[None, ..., None].expand(S, B, W, 1), sigma[None, ..., None].expand(S, B, W, 1)], -1)
        z = torch.relu(torch.einsum("sbwi,sih->sbwh", self.drop(x), self.w1) + self.b1[:, None, None, :])
        return torch.einsum("sbwh,sh->sbw", self.drop(z), self.w2) + self.b2[:, None, None]


def mlp_batch(panel, t, idx, device):
    rows = t["rows"][idx]
    h = torch.as_tensor(panel.h[:, rows], device=device).float()
    m = torch.as_tensor(panel.m[:, rows], device=device).float()
    pick = lambda k: t[k][idx].to(device)  # noqa: E731
    return h, m, pick("d"), pick("sigma"), pick("valid"), pick("log_p"), pick("zbar"), pick("y")


def mlp_forward(model, panel, t, idx, device):
    h, m, d, sg, valid, log_p, zbar, y = mlp_batch(panel, t, idx, device)
    delta = model(h, m, d, sg).masked_fill(~valid[None], 0.0)
    S = delta.shape[0]
    w, kl = weights_and_kl(log_p[None].expand(S, -1, -1), delta, valid[None].expand(S, -1, -1))
    pred = (w * zbar[None]).sum(-1)              # (S,B): one index forecast per seed MLP
    return pred, w, kl, y


@torch.no_grad()
def mlp_predict(model, panel, t, device):
    model.eval()
    preds, ws = [], []
    for i in range(0, len(t["rows"]), MLP_BATCH):
        idx = torch.arange(i, min(i + MLP_BATCH, len(t["rows"])))
        pred, w, _, _ = mlp_forward(model, panel, t, idx, device)
        preds.append(pred.mean(0).cpu().numpy())
        ws.append(w.mean(0).cpu().numpy())
    return np.concatenate(preds).astype(np.float64), np.concatenate(ws)


def fit_mlp(panel, t_tr, tau, steps, device, t_va=None):
    torch.manual_seed(0)
    S = panel.h.shape[0]
    model = SeedMLPs(S, panel.h.shape[-1] + panel.m.shape[-1] + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    gen = torch.Generator().manual_seed(0)
    n = len(t_tr["rows"])
    curve, best, perm, pos = [], (np.inf, 0), torch.randperm(n, generator=gen), 0
    p_va = cap_shares(t_va) if t_va is not None else None
    run_loss, run_n = 0.0, 0
    for step in range(steps + 1):
        if t_va is not None and step % MLP_EVERY == 0:
            # curve rows: (step, validation MSE, validation moved share, mean training index MSE since last check)
            pred, w_va = mlp_predict(model, panel, t_va, device)
            mse = float(np.square(pred - t_va["y"].numpy()).mean())
            moved = float(0.5 * np.abs(w_va - p_va).sum(1).mean())
            curve.append((step, mse, moved, run_loss / run_n if run_n else float("nan")))
            run_loss, run_n = 0.0, 0
            if mse < best[0]:
                best = (mse, step)
            if step - best[1] >= MLP_PATIENCE:
                break
        if step == steps:
            break
        model.train()
        if pos + MLP_BATCH > n:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + MLP_BATCH]
        pos += MLP_BATCH
        pred, _, kl, y = mlp_forward(model, panel, t_tr, idx, device)
        index_mse = (y[None] - pred).square().mean()
        loss = index_mse + tau * kl.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if t_va is not None:
            run_loss += float(index_mse.detach())
            run_n += 1
    return model, curve


# ------------------------------------------------------------------ nested evaluation of one year
def diagnostics(panel, rows, w, data, year, model):
    p, valid = panel.p[rows], panel.valid[rows]
    moved = 0.5 * np.abs(w - p).sum(1)
    eff = 1.0 / np.square(w).sum(1)
    eff_cap = 1.0 / np.square(p).sum(1)
    dec = np.clip((panel.cap_pct[rows] * 10).astype(int), 0, 9)
    shift = np.stack([((w - p) * (dec == k) * valid).sum(1) for k in range(10)], 1)
    frame = pd.DataFrame({"date": data.dates[panel.days[rows]], "year": year, "model": model, "moved": moved,
                          "eff_n": eff, "eff_n_cap": eff_cap})
    frame[[f"shift_size_d{k + 1}" for k in range(10)]] = shift
    if data.sector is not None:
        sec = data.sector.sector[panel.days[rows][:, None], panel.codes[rows]]
        for j, name in enumerate(data.sector.names):
            frame[f"shift_sector_{name}"] = ((w - p) * (sec == j) * valid).sum(1)
    return frame


def evaluate_year(panel, base, v, data, device, log):
    inner_tr = panel.rows(range(STAGE2_FIRST_YEAR, v - 1))
    inner_va = panel.rows([v - 1])
    outer_tr = panel.rows(range(STAGE2_FIRST_YEAR, v))
    ev = panel.rows([v])
    if min(len(inner_tr), len(inner_va), len(ev)) == 0:
        raise ValueError(f"{base}: cross-fitted years missing for evaluation year {v}.")
    preds, settings, diags = {f"{base}-cap": panel.cap_fc[ev].astype(np.float64)}, {}, []

    # direct correction
    scores = {lam: float(np.square(fit_corr(panel, inner_tr, lam)(inner_va) - panel.y[inner_va]).mean())
              for lam in CORR_LAMBDAS}
    lam = min(scores, key=lambda k: (scores[k], -k))
    preds[f"{base}-corr"] = fit_corr(panel, outer_tr, lam)(ev)
    settings[f"{base}-corr"] = {"lambda": lam, "inner_mse": scores}

    # bilinear delta (Lin-delta on linear forecasts, NN-delta-simple on neural forecasts)
    kind = "Lin-delta" if base == "Lin" else "NN-delta-simple"
    sc_in, sc_out = scalers(panel, inner_tr), scalers(panel, outer_tr)
    t_itr, t_iva = tensors(panel, inner_tr, sc_in), tensors(panel, inner_va, sc_in)
    t_otr, t_ev = tensors(panel, outer_tr, sc_out), tensors(panel, ev, sc_out)
    inner, curves = {}, {}
    for tau in TAUS:
        _, curve = fit_bilinear(t_itr, kind, tau, BIL_STEPS, t_iva)
        curves[str(tau)] = curve
        inner[tau] = min(curve, key=lambda c: (c[1], c[0]))
        log(f"    {kind} inner tau={tau}: best step {inner[tau][0]}, mse {inner[tau][1]:.4f} (step 0: {curve[0][1]:.4f}), "
            f"moved at best {inner[tau][2]:.4f}, max moved {max(c[2] for c in curve):.4f}")
    tau = min(TAUS, key=lambda k: (inner[k][1], -k))
    model, _ = fit_bilinear(t_otr, kind, tau, inner[tau][0])
    with torch.no_grad():
        pred, w, _ = forward_bilinear(model, t_ev, kind)
    preds[kind] = pred.numpy().astype(np.float64)
    settings[kind] = {"tau": tau, "steps": inner[tau][0], "inner": {str(k): v for k, v in inner.items()},
                      "curves": curves}
    # readable weight rule: delta_i = f_i' Theta [1, s_t] on standardised inputs; rows = firm signals, columns = market
    # state. Stored for the interpretation figure (which signals move weight, in which market states).
    try:
        rows_names = (list(data.features) + ["d"]) if kind == "Lin-delta" else ["d", "sigma", "CapPct"]
        settings[kind]["theta"] = model.theta.detach().cpu().numpy().round(6).tolist()
        settings[kind]["theta_rows"] = rows_names
        settings[kind]["theta_cols"] = ["1"] + list(data.market_names)
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must never stop the run
        log(f"    could not store {kind} coefficients: {exc}")
    diags.append(diagnostics(panel, ev, w.numpy(), data, v, kind))

    # weight MLP (neural base only)
    if base == "NN" and panel.h is not None:
        t_itr_g = {**t_itr}
        inner, curves = {}, {}
        for tau in TAUS:
            _, curve = fit_mlp(panel, t_itr_g, tau, MLP_STEPS, device, t_iva)
            curves[str(tau)] = curve
            inner[tau] = min(curve, key=lambda c: (c[1], c[0]))
            log(f"    NN-delta inner tau={tau}: best step {inner[tau][0]}, mse {inner[tau][1]:.4f} "
                f"(step 0: {curve[0][1]:.4f}), moved at best {inner[tau][2]:.4f}, max moved {max(c[2] for c in curve):.4f}, "
                f"train mse first/last {curve[1][3] if len(curve) > 1 else float('nan'):.4f}/{curve[-1][3]:.4f}")
        tau = min(TAUS, key=lambda k: (inner[k][1], -k))
        model, _ = fit_mlp(panel, t_otr, tau, inner[tau][0], device)
        pred, w = mlp_predict(model, panel, t_ev, device)
        preds["NN-delta"] = pred
        settings["NN-delta"] = {"tau": tau, "steps": inner[tau][0], "inner": {str(k): v for k, v in inner.items()},
                                "curves": curves}
        diags.append(diagnostics(panel, ev, w, data, v, "NN-delta"))
    return panel.days[ev], preds, settings, diags


# ------------------------------------------------------------------ phases
def run_years(years, phase, args, log):
    data, gate = load_data4()
    out = Path(args.out) / phase
    out.mkdir(parents=True, exist_ok=True)
    oof_root = Path(args.oof)
    panels = {}
    for base, folder in (("Lin", oof_root / "lin"), ("NN", oof_root / "nn")):
        if folder.exists() and any(folder.glob("k*_s*.npz")):
            panels[base] = Panel(data, load_oof(folder, data), range(STAGE2_FIRST_YEAR, max(years) + 1))
            log(f"{base}: seeds {panels[base].seeds}, cross-fitted years {sorted(set(panels[base].year))}")
    frames, all_settings, all_diags = [], {}, []
    for v in years:
        t0 = time.time()
        days, b0, b3r = b0_b3r(data, v)
        rec = pd.DataFrame({"date": data.dates[days], "year": v, "actual_pct": data.y[days], "B0": b0, "B3R": b3r})
        for base, panel in panels.items():
            ev_days, preds, settings, diags = evaluate_year(panel, base, v, data, args.device, log)
            if not np.array_equal(ev_days, days):
                raise ValueError(f"{base} {v}: evaluation days differ from the baseline days.")
            for name, pred in preds.items():
                rec[name] = pred
            all_settings[f"{base}_{v}"] = settings
            all_diags += diags
        frames.append(rec)
        mses = {c: float(np.square(rec[c] - rec.actual_pct).mean()) for c in rec.columns if c not in ("date", "year", "actual_pct")}
        log(f"{phase} {v} done in {time.time() - t0:.0f} s" + ("" if phase == "test" else f": {json.dumps(mses)}"))
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(out / "predictions.csv", index=False)
    if all_diags:
        pd.concat(all_diags, ignore_index=True).to_csv(out / "weight_diagnostics.csv", index=False)
    save_json({"settings": all_settings, "inputs": gate["inputs"]}, out / "settings.json")
    return pred


def fold_scores(pred, models):
    rows = []
    for v, g in pred.groupby("year"):
        b0 = np.square(g.actual_pct - g.B0).mean()
        for m in models:
            if m in g:
                mse = np.square(g.actual_pct - g[m]).mean()
                rows.append({"year": v, "model": m, "mse": mse, "gain_vs_b0": (b0 - mse) / b0})
    return pd.DataFrame(rows)


def lock(args, log):
    pred = pd.read_csv(Path(args.out) / "validation" / "predictions.csv")
    scores = fold_scores(pred, ("B0", "B3R") + CANDIDATES)
    summary = scores.groupby("model").agg(score=("gain_vs_b0", "mean"), folds_positive=("gain_vs_b0", lambda s: int((s > 0).sum())))
    present = [m for m in CANDIDATES if m in summary.index]
    # ties within 1e-6 (e.g. a delta model that chose 0 steps equals its cap model) go to the simpler model,
    # i.e. the one listed first in CANDIDATES
    best = max(present, key=lambda m: (round(float(summary.at[m, "score"]), 6), -present.index(m)))
    code = {p.name: file_sha(p) for p in sorted((REV4 / "code").glob("r4_*.py"))}
    oof_info = {}
    for base in ("lin", "nn"):
        files = sorted((Path(args.oof) / base).glob("k*_s*.npz"))
        oof_info[base] = {f.name: file_sha(f) for f in files}
    vlog = RESULTS / "VALIDATION_LOG.md"           # one "### " heading per configuration tried (POSTER_RUN_PLAN 8)
    tried = max(1, sum(line.startswith("### ") for line in vlog.read_text(encoding="utf-8").splitlines())
                if vlog.exists() else 0)
    plan = {"locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "best_model": best,
            "configurations_tried": tried,
            "research_question_test": {"model": RQ[0], "benchmark": RQ[1], "test": "Clark-West"},
            "validation_scores": summary.reset_index().to_dict(orient="records"), "candidates": list(CANDIDATES),
            "code_sha": code, "validation_oof_sha": oof_info,
            "rule": "best = highest mean fold-normalised MSE gain over B0 across 2018-2021 (ties within 1e-6 go "
                    "to the simpler model); RQ fixed as NN-delta vs NN-cap"}
    save_json(plan, plan_path(args))
    scores.to_csv(Path(args.out) / "validation" / "fold_scores.csv", index=False)
    log(f"locked: best = {best}\n{summary.to_string()}")


def plan_path(args):
    return Path(args.out).parent / "plan_locked.json"


def oracle_room(args, log):
    """Validation-year room for re-weighting, measured on the same cross-fitted forecasts the models use:
    oracle gain over the cap-weighted sum when all weight (m = 1) or 10% of it may move, next to the honest gains
    of the delta models. Feeds poster Figure 2A (replaces the earlier-study numbers)."""
    from oracle import exact_shift
    out = Path(args.out) / "validation"
    pred = pd.read_csv(out / "predictions.csv")
    years = sorted(pred.year.unique())
    data, _ = load_data4()
    rows = []
    for base, folder, deltas in (("Lin", Path(args.oof) / "lin", ("Lin-delta",)),
                                 ("NN", Path(args.oof) / "nn", ("NN-delta-simple", "NN-delta"))):
        if not folder.exists() or not any(folder.glob("k*_s*.npz")):
            continue
        panel = Panel(data, load_oof(folder, data), range(STAGE2_FIRST_YEAR, max(years) + 1))
        for v in years:
            r = panel.rows([v])
            cap, y = panel.cap_fc[r].astype(np.float64), panel.y[r].astype(np.float64)
            mse_cap = float(np.square(y - cap).mean())
            row = {"base": base, "year": v, "mse_cap": mse_cap}
            for m, tag in ((1.0, "oracle_all"), (0.1, "oracle_10pct")):
                down, up = exact_shift(panel.zbar[r], panel.p[r], panel.valid[r], m)
                row[f"gain_{tag}"] = (mse_cap - float(np.square(y - (cap + np.clip(y - cap, down, up))).mean())) / mse_cap
            g = pred.loc[pred.year == v]
            for d in deltas:
                if d in g:
                    row[f"gain_{d}"] = (mse_cap - float(np.square(g.actual_pct - g[d]).mean())) / mse_cap
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "oracle_room.csv", index=False)
    log("oracle room (validation, gain vs the same forecasts' cap-weighted sum):\n"
        + frame.groupby("base").mean(numeric_only=True).drop(columns="year").to_string())


def firm_r2(args, log):
    """Firm-level out-of-sample R2 of the cross-fitted firm forecasts by year and base (poster conclusion: is firm
    predictability the bottleneck?). Pooled over firm-days, against a zero forecast (Gu, Kelly & Xiu 2020 convention).
    Run in the report phase only, i.e. after the lock."""
    data, _ = load_data4()
    rows = []
    for base in ("lin", "nn"):
        folder = Path(args.oof) / base
        if not folder.exists() or not any(folder.glob("k*_s*.npz")):
            continue
        oof = load_oof(folder, data)
        panel = Panel(data, oof, sorted(oof))
        for v in sorted(oof):
            r = panel.rows([v])
            actual = data.firm_y[panel.days[r][:, None], panel.codes[r]]
            ok = panel.valid[r] & np.isfinite(actual)
            err = np.square(actual - panel.zbar[r])[ok].sum()
            rows.append({"base": base.upper() if base == "nn" else "Lin", "year": v,
                         "split": "test" if v in TEST_YEARS else ("validation" if v in VALIDATION_YEARS else "stage-2 training"),
                         "firm_r2_pct": 100 * (1 - err / np.square(actual[ok]).sum()), "firm_days": int(ok.sum())})
    frame = pd.DataFrame(rows)
    log("firm-level R2 (pooled, vs zero forecast):\n" + frame.to_string())
    return frame


def capture_rates(args, phase, pred, log):
    """Share of the oracle gain (same z_bar, same mean moved weight as NN-delta) captured by NN-delta over NN-cap."""
    from oracle import exact_shift
    diag_path = Path(args.out) / phase / "weight_diagnostics.csv"
    if "NN-delta" not in pred or not diag_path.exists():
        return pd.DataFrame()
    diag = pd.read_csv(diag_path)
    data, _ = load_data4()
    years = sorted(pred.year.unique())
    panel = Panel(data, load_oof(Path(args.oof) / "nn", data), range(STAGE2_FIRST_YEAR, max(years) + 1))
    rows_out = []
    for v in years:
        rows = panel.rows([v])
        moved = float(diag.loc[(diag.model == "NN-delta") & (diag.year == v), "moved"].mean())
        down, up = exact_shift(panel.zbar[rows], panel.p[rows], panel.valid[rows], moved)
        cap, y = panel.cap_fc[rows], panel.y[rows]
        oracle = cap + np.clip(y - cap, down, up)
        g = pred.loc[pred.year == v]
        mse = {m: float(np.square(g.actual_pct - g[m]).mean()) for m in ("NN-cap", "NN-delta")}
        mse_oracle = float(np.square(y - oracle).mean())
        room = mse["NN-cap"] - mse_oracle
        rows_out.append({"year": v, "moved_mean": moved, "mse_cap": mse["NN-cap"], "mse_delta": mse["NN-delta"],
                         "mse_oracle": mse_oracle, "oracle_gain_pct": 100 * room / mse["NN-cap"],
                         "capture": (mse["NN-cap"] - mse["NN-delta"]) / room if room > 0 else np.nan})
    frame = pd.DataFrame(rows_out)
    log("capture:\n" + frame.to_string())
    return frame


def report(args, log, phase="test"):
    plan = json.loads(plan_path(args).read_text(encoding="utf-8"))
    pred = pd.read_csv(Path(args.out) / phase / "predictions.csv")
    best = plan["best_model"]
    tests = []
    # B3R is the cap-weighted WLS ridge (firm rows, cap-share row weights, index target; r3_core.head_moments).
    # NN-delta vs B3R added 2026-11-20 21:4x before any stage-2 result existed (user: the WLS ridge is the baseline to beat)
    for model, bench, kind in ((RQ[0], RQ[1], "Clark–West"), (RQ[0], "NN-corr", "Diebold–Mariano"),
                               (RQ[0], "B0", "Clark–West"), (RQ[0], "B3R", "Diebold–Mariano"),
                               (best, "B0", "Clark–West"), (best, "B3R", "Diebold–Mariano")):
        if model in pred and bench in pred:
            tests.append({"role": "research question" if model == RQ[0] else "best model",
                          **differential_test(pred, model, bench, kind)})
    by_year = fold_scores(pred, ("B0", "B3R") + CANDIDATES)
    overall = []
    b0 = np.square(pred.actual_pct - pred.B0).sum()
    val_score = {r["model"]: r["score"] for r in plan.get("validation_scores", [])}
    for m in ("B0", "B3R") + CANDIDATES:
        if m in pred:
            sse = np.square(pred.actual_pct - pred[m]).sum()
            # poster Figure 3 columns: validation score, test MSE, OOS R2, one-sided Clark-West p vs B0 for every model
            p_b0 = np.nan if m == "B0" else float(differential_test(pred, m, "B0", "Clark–West")["p_value"])
            overall.append({"model": m, "validation_score_pct": 100 * val_score[m] if m in val_score else np.nan,
                            "mse": sse / len(pred), "oos_r2_vs_b0_pct": 100 * (1 - sse / b0),
                            "p_clark_west_vs_b0": p_b0, "best_validated": m == best})
    out = Path(args.out) / phase
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    by_year.to_csv(out / "by_year.csv", index=False)
    pd.DataFrame(overall).to_csv(out / "overall.csv", index=False)
    capture_rates(args, phase, pred, log).to_csv(out / "capture.csv", index=False)
    try:
        firm_r2(args, log).to_csv(out / "firm_r2.csv", index=False)
    except Exception as exc:  # noqa: BLE001 -- the poster table must not be lost to a diagnostic
        log(f"firm-level R2 failed: {exc}")
    log("report written:\n" + pd.DataFrame(overall).to_string() + "\n" + pd.DataFrame(tests).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("validation", "lock", "test", "report"), required=True)
    ap.add_argument("--years", default=None, help="override evaluation years (smoke tests only)")
    ap.add_argument("--oof", default=str(RESULTS / "oof"))
    ap.add_argument("--out", default=str(RESULTS / "stage2"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-phase", default="test", help="smoke tests only: 'validation'")
    args = ap.parse_args()
    log = log_to(Path(args.out) / "stage2.log")
    torch.set_num_threads(4)
    if args.phase == "validation":
        years = [int(y) for y in args.years.split(",")] if args.years else list(VALIDATION_YEARS)
        run_years(years, "validation", args, log)
        oracle_room(args, log)
    elif args.phase == "lock":
        lock(args, log)
    elif args.phase == "test":
        if not plan_path(args).exists():
            raise SystemExit("No locked plan: run --phase lock first.")
        run_years(list(TEST_YEARS), "test", args, log)
    else:
        report(args, log, args.report_phase)


if __name__ == "__main__":
    main()
