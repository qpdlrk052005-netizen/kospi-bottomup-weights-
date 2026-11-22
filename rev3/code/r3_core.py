"""Revision-3 core: preprocessing fix, folds, B3R head, delta models (linear and neural)."""

from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

REV2_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(REV2_CODE))
from pilot_data import DailyData, Fold, MARKET_FEATURES, TEST_YEARS, checks  # noqa: E402,F401
from pilot_fit import BATCH, LR, WEIGHT_DECAY, evaluate, set_seed  # noqa: E402,F401
from pilot_neural import SETTINGS, DynamicModel, NeuralBatches  # noqa: E402,F401

VALIDATION_YEARS = (2018, 2019, 2020, 2021)
LAMBDA_GRID = (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0)
TAU_GRID = (0.0, 0.001, 0.01, 0.1, 1.0)
GATE_MOVED_SHARE = 0.01
LINEAR_LR, LINEAR_STEPS, LINEAR_EVERY = 1e-2, 1000, 10
NEURAL_MAX_EPOCHS, NEURAL_PATIENCE = 20, 3
SEEDS = (0, 1, 2, 3, 4)
NEURAL_SETTING = "A-i_LSTM_aux0"
COMPLEXITY = ("L-S", "L-A", "N-A")
MAIN_CANDIDATES = ("L-A", "N-A")


# ---------------------------------------------------------------- data

class DailyData3(DailyData):
    """Revision-2 data with one change (D1): discrete indicator features (at most 3 distinct training values,
    e.g. Up in {0, 1} and LimitHit in {-1, 0, +1}) are neither winsorised nor divided by their SD: they are only
    centred, so one category step stays one unit (LimitHit's SD is about 0.04, so z-scoring would turn a limit
    day into an outlier of about +-23)."""

    def scale(self, training, prepare_history=True):
        x = self.stock[training][self.universe[training]]
        binary = np.array([len(np.unique(col[np.isfinite(col)])) <= 3 for col in x.T])
        lo, hi = np.nanquantile(x, [0.01, 0.99], axis=0)
        lo[binary], hi[binary] = -np.inf, np.inf
        clipped = np.clip(x, lo, hi)
        mu = np.nanmean(clipped, axis=0)
        sigma = np.nanstd(clipped, axis=0)
        dead = [f for f, s in zip(self.features, sigma, strict=True) if not s > 0]
        if dead:
            raise ValueError(f"Zero-variance firm features after preprocessing: {dead}")
        sigma = np.where(binary, 1.0, sigma)
        del x, clipped
        z = np.clip(self.stock, lo.astype(np.float32), hi.astype(np.float32))
        z -= mu.astype(np.float32)
        z /= sigma.astype(np.float32)
        z[~np.isfinite(z)] = 0
        if prepare_history:
            z[~self.universe] = 0
        s_train = self.market[training]
        s_mu, s_sigma = np.nanmean(s_train, axis=0), np.nanstd(s_train, axis=0)
        s_sigma = np.where(s_sigma > 0, s_sigma, 1)
        s = np.nan_to_num((self.market - s_mu) / s_sigma).astype(np.float32)
        import pandas as pd
        return Fold(
            training, z, s, self.cap_pct,
            pd.DataFrame({"feature": self.features, "binary_not_winsorised": binary,
                          "p01": lo, "p99": hi, "mean": mu, "std": sigma}),
            pd.DataFrame({"feature": MARKET_FEATURES, "mean": s_mu, "std": s_sigma}),
        )


def load_data(root):
    market, gate = checks(root / "daily_dataset", root / "inputs" / "market_indices.csv")
    if not gate["gate_pass"]:
        raise SystemExit("Data gate failed.")
    return DailyData3(root / "daily_dataset", root / "inputs" / "market_indices.csv", market), gate


def dense(data, fold, days):
    ids = [data.members[d] for d in days]
    n = np.array([len(i) for i in ids])
    width, p = int(n.max()), fold.stock.shape[-1]
    x = np.zeros((len(days), width, p), np.float32)
    log_cap = np.full((len(days), width), -np.inf, np.float32)
    cap_pct = np.zeros((len(days), width), np.float32)
    for i, (day, codes) in enumerate(zip(days, ids, strict=True)):
        x[i, :len(codes)] = fold.stock[day, codes]
        cap = data.cap[day, codes]
        log_cap[i, :len(codes)] = np.log(cap / cap.sum())
        cap_pct[i, :len(codes)] = fold.cap_pct[day, codes]
    return {"days": np.asarray(days), "x": x, "log_cap": log_cap, "cap_pct": cap_pct,
            "valid": np.arange(width)[None, :] < n[:, None],
            "s": fold.market[days].astype(np.float32), "y": data.y[days].astype(np.float64)}


# ---------------------------------------------------------------- B3 / B3R closed form

def head_moments(d):
    """Gram matrix of loss (2.x): sum z z' + sum diag(0, S_t, 0), with z_t = [1, cap-weighted x, s]."""
    w = np.exp(d["log_cap"].astype(np.float64))
    x = d["x"].astype(np.float64)
    xbar = np.einsum("tn,tnp->tp", w, x)
    centered = x - xbar[:, None, :]
    s_sum = np.einsum("tn,tnp,tnq->pq", w, centered, centered)
    z = np.concatenate([np.ones((len(x), 1)), xbar, d["s"].astype(np.float64)], axis=1)
    p = x.shape[-1]
    a = z.T @ z
    a[1:1 + p, 1:1 + p] += s_sum
    return a, z.T @ d["y"], p


def solve_head(moments, lam):
    a, c, p = moments
    penalty = np.r_[0.0, np.full(len(c) - 1, lam)]
    b = np.linalg.lstsq(a + np.diag(penalty), c, rcond=None)[0]
    return {"alpha": float(b[0]), "beta": b[1:1 + p], "gamma": b[1 + p:], "lambda": float(lam)}


def head_predict(head, d, delta=None):
    logits = d["log_cap"].astype(np.float64) + (0.0 if delta is None else delta)
    logits = np.where(d["valid"], logits, -np.inf)
    w = np.exp(logits - logits.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return head["alpha"] + d["s"].astype(np.float64) @ head["gamma"] + (w * (d["x"].astype(np.float64) @ head["beta"])).sum(1)


# ---------------------------------------------------------------- linear delta (head frozen at B3R)

class LinearDelta(nn.Module):
    """L-S: delta_i = theta'[x_i, CapPct_i].  L-A: delta_i = [x_i, CapPct_i]' Theta [1, s_t]."""

    def __init__(self, q, m, kind):
        super().__init__()
        self.kind = kind
        self.theta = nn.Parameter(torch.zeros(q, 1 if kind == "L-S" else 1 + m))

    def forward(self, f, s):
        if self.kind == "L-S":
            return (f @ self.theta).squeeze(-1)
        g = torch.cat([torch.ones_like(s[:, :1]), s], dim=1)
        return torch.einsum("tnq,qk,tk->tn", f, self.theta, g)


def frozen_tensors(d, head):
    """Precompute everything the frozen head needs: base_t = alpha + gamma's_t and partial_{t,i} = beta'x_{t,i}."""
    valid = torch.from_numpy(d["valid"])
    return {
        "f": torch.from_numpy(np.concatenate([d["x"], d["cap_pct"][..., None]], axis=-1)),
        "s": torch.from_numpy(d["s"]),
        "log_cap": torch.from_numpy(d["log_cap"]).masked_fill(~valid, 0.0),
        "valid": valid,
        "base": torch.from_numpy((head["alpha"] + d["s"].astype(np.float64) @ head["gamma"]).astype(np.float32)),
        "partial": torch.from_numpy((d["x"].astype(np.float64) @ head["beta"]).astype(np.float32)),
        "y": torch.from_numpy(d["y"].astype(np.float32)),
    }


def weights_and_kl(log_cap_safe, delta, valid):
    log_w = torch.log_softmax((log_cap_safe + delta).masked_fill(~valid, -torch.inf), dim=-1)
    log_w_safe = log_w.masked_fill(~valid, 0.0)
    w = log_w_safe.exp() * valid
    kl = (w * (log_w_safe - log_cap_safe)).sum(-1)
    return w, kl


def linear_forward(tilt, t):
    delta = tilt(t["f"], t["s"]).masked_fill(~t["valid"], 0.0)
    w, kl = weights_and_kl(t["log_cap"], delta, t["valid"])
    return t["base"] + (w * t["partial"]).sum(-1), w, kl, delta


LINEAR_PATIENCE_STEPS = 200


def fit_linear(t_train, kind, tau, steps, t_val=None, every=LINEAR_EVERY, lr=LINEAR_LR):
    """Full-batch Adam on delta only (deterministic). With t_val: validation predictions every `every` steps,
    stopping once validation MSE has not improved for LINEAR_PATIENCE_STEPS steps."""
    torch.manual_seed(0)
    tilt = LinearDelta(t_train["f"].shape[-1], t_train["s"].shape[-1], kind)
    optimizer = torch.optim.Adam(tilt.parameters(), lr=lr)
    curve, best = [], (np.inf, 0)
    for step in range(steps + 1):
        if t_val is not None and step % every == 0:
            with torch.no_grad():
                forecast, w, _, _ = linear_forward(tilt, t_val)
                cap = t_val["log_cap"].exp() * t_val["valid"]
                moved = float((0.5 * (w - cap).abs().sum(-1)).mean())
                forecast = forecast.numpy().astype(np.float64)
            curve.append((step, forecast, moved))
            mse = float(np.square(forecast - t_val["y"].numpy()).mean())
            if mse < best[0]:
                best = (mse, step)
            if step - best[1] >= LINEAR_PATIENCE_STEPS:
                break
        if step == steps:
            break
        optimizer.zero_grad()
        forecast, _, kl, _ = linear_forward(tilt, t_train)
        ((t_train["y"] - forecast).square().mean() + tau * kl.mean()).backward()
        optimizer.step()
    return tilt, curve


# ---------------------------------------------------------------- neural delta (head frozen at B3R)

class NeuralDelta(DynamicModel):
    """Revision-2 main-model structure; head fixed at B3R, last delta layer N(0, 0.01^2), KL concentration penalty."""

    def __init__(self, setting, firm_count, head, tau):
        super().__init__(setting, firm_count, head["alpha"])
        self.tau = float(tau)
        with torch.no_grad():
            self.head.alpha.fill_(head["alpha"])
            self.head.beta.weight.copy_(torch.tensor(head["beta"], dtype=torch.float32)[None])
            self.head.gamma.weight.copy_(torch.tensor(head["gamma"], dtype=torch.float32)[None])
            self.weight_score.weight.normal_(0.0, 0.01)
        for p in self.head.parameters():
            p.requires_grad_(False)

    def objective(self, out, batch):
        mse = (batch["y"] - out["prediction"]).square().mean()
        if not self.tau:
            return mse
        valid = batch["valid"]
        _, kl = weights_and_kl(batch["log_cap"].masked_fill(~valid, 0.0), out["delta"].masked_fill(~valid, 0.0), valid)
        return mse + self.tau * kl.mean()


def neural_setting():
    return {s.name: s for s in SETTINGS}[NEURAL_SETTING]


def train_neural(data, batches, head, tau, seed, train, validation=None, epochs=NEURAL_MAX_EPOCHS, log=print):
    """AdamW on the delta path. With validation: early stopping (patience 3) and per-epoch validation predictions."""
    set_seed(seed)
    model = NeuralDelta(neural_setting(), len(data.features), head, tau).to(batches.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(seed)
    history, val_preds, best = [], [], (np.inf, None, None)
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        order, total = rng.permutation(train), 0.0
        for i in range(0, len(order), BATCH):
            batch = batches(order[i:i + BATCH])
            optimizer.zero_grad(set_to_none=True)
            loss = model.objective(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach().item() * len(batch["lengths"])
        row = {"epoch": epoch, "train_loss": total / len(train), "val_mse": np.nan}
        if validation is not None:
            forecast = evaluate(model, batches, validation, BATCH)["prediction"]
            val_preds.append(forecast)
            row["val_mse"] = float(np.square(forecast - data.y[validation]).mean())
            if row["val_mse"] < best[0]:
                best = (row["val_mse"], epoch, deepcopy(model.state_dict()))
        row["seconds"] = time.monotonic() - started
        history.append(row)
        log(f"    epoch={epoch:02d} train_loss={row['train_loss']:.6f} val_mse={row['val_mse']:.6f} s={row['seconds']:.1f}")
        if validation is not None and epoch - best[1] >= NEURAL_PATIENCE:
            break
    if validation is not None:
        model.load_state_dict(best[2])
    return model, history, (best[1] if validation is not None else epochs), val_preds


def weight_summary(weights, deltas, log_caps):
    """Per-day gate statistics from lists of per-day arrays (valid firms only)."""
    rows = []
    for w, delta, log_cap in zip(weights, deltas, log_caps, strict=True):
        c = np.exp(log_cap)
        top = np.argsort(-c, kind="stable")[:10]
        centered = delta - c @ delta
        order = np.argsort(c, kind="stable")
        rows.append({
            "moved_share": 0.5 * np.abs(w - c).sum(), "effective_firms": 1 / (w @ w), "effective_firms_cap": 1 / (c @ c),
            "kl": float(np.sum(w * (np.log(np.maximum(w, 1e-300)) - log_cap))),
            "top10_weight": w[top].sum(), "top10_capweight": c[top].sum(),
            **{f"delta_decile{k}": float(centered[part].mean())
               for k, part in enumerate(np.array_split(order, 10), start=1)},
        })
    return rows
