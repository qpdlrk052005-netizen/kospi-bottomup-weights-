"""Firm-level superiority evaluation of our stage-1 NN against competing forecasters (2026-11-21, after the lock;
exploratory, pre-test years 2016-2021 only). The metric set and decision rule below were written before any competitor
result was seen (BENCH_PROTOCOL_KO.md).

Forecasts compared (all cross-fitted: F^(k) trained on labels before year k, predicting year k)
  ours         our NN (trading GRU + financial GRU + GAT + market GRU), seed mean of seeds 0-2 (5-seed mean reported too)
  lin          pre-registered pooled ridge (rev4 Lin)
  anomaly      IC-weighted composite of 16 known signals; weights = mean daily rank IC on the fold's training days
  mlp, gru, dtml, master, master_csz, lgb   bench_stage1.py (seed mean of seeds 0-2)

Returns
  r_oc   next-open -> next-close (tradable entry; PRIMARY)       r_cc  close -> next close (the training target)
  r_oc5  next-open -> close of t+5 (inside the same year)
Metrics per model (daily cross-sections, then averaged over days)
  RankIC (Spearman), RankICIR = mean/sd, IC (Pearson), ICIR, share of days with RankIC > 0, yearly RankIC
  long-short top/bottom decile (equal weight), annualised Sharpe; top-50 long-only excess over the equal-weight
  universe: AR (% per year) and IR, gross and net of 20 bp one-way cost (turnover-based); DTML-style direction ACC / MCC
  (above vs below the day's median return); firm-level MSE on daily-winsorised returns (1%/99%)
Tests vs ours (paired by day, Newey-West lag 5)
  dRankIC = RankIC(ours) - RankIC(C): mean, t, one-sided p (H1: ours better), Holm over competitors
  encompassing: daily OLS of rank(r) on rank(ours) and rank(C) -> both slopes, NW t
  anomaly-adjusted: daily OLS of rank(r) on rank(model) + ranks of the 16 signals -> model slope, NW t
Decision rule (fixed in advance): ours is "superior to C" in a market if dRankIC(r_oc) > 0 with Holm-adjusted
one-sided p < 0.05 over 2016-2021. Firm-level superiority is "demonstrated" only if that holds against every
competitor in KOSPI and is replicated in KOSDAQ (same sign, one-sided p < 0.05 each).

  python -u -X utf8 rev4/code/bench_eval.py --market kospi
  (KOSDAQ: set R4_DATA_ROOT, R4_PROCESSED, R4_GRAPH_DIR to the kosdaq folders and pass --market kosdaq --base <dir>)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from r4_common import RESULTS, load_data4

YEARS = range(2016, 2022)
VAL_YEARS = range(2018, 2022)
SIGNALS = ["LogMarcap", "LogBM", "Mom12_1", "Change", "Ret5", "Vol20", "LogIdioVol60", "MAX20", "Amihud", "Turnover",
           "Range", "CloseLoc", "Gap", "ForeignNetBuy", "InstNetBuy", "IndivNetBuy"]
COMPETITORS = ["lin", "anomaly", "lgb", "mlp", "gru", "dtml", "master", "master_csz"]
COST_BP, TOPK = 20.0, 50


def nw_t(x, lags=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    u, n = x - x.mean(), len(x)
    s = u @ u / n + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n for k in range(1, lags + 1))
    return x.mean() / np.sqrt(s / n)


def load_preds(folder, data, seeds=None):
    """Dense (days, codes) array of seed-mean forecasts; NaN where no forecast."""
    Z = np.full(data.cap.shape, np.nan, np.float32)
    used = set()
    for k in YEARS:
        files = sorted(Path(folder).glob(f"k{k}_s*.npz"))
        if seeds is not None:
            files = [f for f in files if int(f.stem.split("_s")[1]) in seeds]
        if not files:
            raise FileNotFoundError(f"{folder}: no forecasts for {k}")
        rs, days = [], None
        for f in files:
            z = np.load(f, allow_pickle=False)
            if days is None:
                days, lengths = z["days"], z["lengths"]
            elif not np.array_equal(days, z["days"]):
                raise ValueError(f"{f}: days differ across seeds")
            rs.append(z["r"])
            used.add(int(f.stem.split("_s")[1]))
        r = np.mean(rs, axis=0)
        expected = np.array([len(data.members[d]) for d in days])
        if not np.array_equal(expected, lengths):
            raise ValueError(f"{folder} k{k}: universe sizes differ from the data")
        off = np.r_[0, np.cumsum(lengths)]
        for i, d in enumerate(days):
            Z[d, data.members[d]] = r[off[i]:off[i + 1]]
    return Z, sorted(used)


def rank_pct(x):
    return (stats.rankdata(x) - 0.5) / len(x)


def anomaly_composite(data, raw):
    """For each fold k: weights = mean daily rank IC of each signal with r_cc over the fold's training days."""
    Z = np.full(data.cap.shape, np.nan, np.float32)
    ic_cache = {}
    weights = {}
    all_train = np.unique(np.concatenate([data.expanding(k)[0] for k in YEARS]))
    for d in all_train:
        ids = data.members[d]
        y = data.firm_y[d, ids]
        row = []
        for s in SIGNALS:
            x = raw[s][d, ids]
            ok = np.isfinite(x) & np.isfinite(y)
            row.append(stats.spearmanr(x[ok], y[ok])[0] if ok.sum() > 20 else np.nan)
        ic_cache[d] = row
    for k in YEARS:
        train, test = data.expanding(k)
        with np.errstate(all="ignore"):
            w = np.nanmean(np.array([ic_cache[d] for d in train], float), axis=0)
        w = np.nan_to_num(w)                     # a signal without training coverage gets weight 0
        weights[k] = dict(zip(SIGNALS, w.round(4).tolist()))
        for d in test:
            ids = data.members[d]
            acc = np.zeros(len(ids))
            for s, wj in zip(SIGNALS, w):
                x = raw[s][d, ids]
                rk = np.full(len(ids), 0.5)
                ok = np.isfinite(x)
                if ok.sum() > 1:
                    rk[ok] = rank_pct(x[ok])
                acc += wj * (rk - 0.5)
            Z[d, ids] = acc
    return Z, weights


def daily_cs(Zs, R, days, data):
    """Per day and model: RankIC, IC, decile long-short, top-k excess + turnover, direction hits."""
    out = {m: {k: [] for k in ("rank_ic", "ic", "ls", "topk", "topk_turn", "tp", "tn", "fp", "fn", "se", "n")}
           for m in Zs}
    prev_top = {m: None for m in Zs}
    for d in days:
        ids = data.members[d]
        r = R[d, ids]
        base = np.isfinite(r)
        for m, Z in Zs.items():
            z = Z[d, ids]
            ok = base & np.isfinite(z)
            o = out[m]
            if ok.sum() < 50:
                for k in o:
                    o[k].append(0 if k in ("tp", "tn", "fp", "fn") else np.nan)
                prev_top[m] = None
                continue
            zz, rr, codes = z[ok], r[ok], ids[ok]
            o["rank_ic"].append(stats.spearmanr(zz, rr)[0])
            o["ic"].append(np.corrcoef(zz, rr)[0, 1])
            n = len(zz)
            order = np.argsort(zz, kind="stable")
            q = max(1, n // 10)
            o["ls"].append(rr[order[-q:]].mean() - rr[order[:q]].mean())
            top = set(codes[order[-TOPK:]])
            o["topk"].append(rr[order[-TOPK:]].mean() - rr.mean())
            o["topk_turn"].append(np.nan if prev_top[m] is None else 1 - len(top & prev_top[m]) / TOPK)
            prev_top[m] = top
            up_true = rr > np.median(rr)
            up_pred = zz > np.median(zz)
            o["tp"].append(int((up_true & up_pred).sum()))
            o["tn"].append(int((~up_true & ~up_pred).sum()))
            o["fp"].append(int((~up_true & up_pred).sum()))
            o["fn"].append(int((up_true & ~up_pred).sum()))
            lo, hi = np.quantile(rr, [0.01, 0.99])
            o["se"].append(np.mean((zz - np.clip(rr, lo, hi)) ** 2))
            o["n"].append(n)
    return {m: {k: np.asarray(v, float) for k, v in o.items()} for m, o in out.items()}


def summarise(cs, years_of_day, sel):
    rows = []
    for m, o in cs.items():
        ric, ic = o["rank_ic"][sel], o["ic"][sel]
        tk = o["topk"][sel]
        turn = np.nan_to_num(o["topk_turn"][sel], nan=1.0)
        net = tk - turn * 2 * COST_BP / 100
        ls = o["ls"][sel]
        tp, tn, fp, fn = (o[k][sel].sum() for k in ("tp", "tn", "fp", "fn"))
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        rows.append({"model": m, "RankIC": np.nanmean(ric), "RankICIR": np.nanmean(ric) / np.nanstd(ric),
                     "RankIC_t": nw_t(ric), "RankIC>0 %": 100 * np.nanmean(ric > 0),
                     "IC": np.nanmean(ic), "ICIR": np.nanmean(ic) / np.nanstd(ic),
                     "LS decile %/day": np.nanmean(ls), "LS Sharpe": np.sqrt(252) * np.nanmean(ls) / np.nanstd(ls),
                     f"Top{TOPK} AR %": 252 * np.nanmean(tk), f"Top{TOPK} IR": np.sqrt(252) * np.nanmean(tk) / np.nanstd(tk),
                     f"Top{TOPK} turnover": np.nanmean(o["topk_turn"][sel]),
                     f"Top{TOPK} AR net %": 252 * np.nanmean(net), f"Top{TOPK} IR net": np.sqrt(252) * np.nanmean(net) / np.nanstd(net),
                     "ACC %": 100 * (tp + tn) / (tp + tn + fp + fn), "MCC": mcc,
                     "MSE (winsorised)": np.nanmean(o["se"][sel])})
    return pd.DataFrame(rows)


def by_year(cs, years_of_day):
    rows = {}
    for m, o in cs.items():
        rows[m] = {y: np.nanmean(o["rank_ic"][years_of_day == y]) for y in YEARS}
    return pd.DataFrame(rows).T


def paired_tests(cs, ref, sel, names):
    rows = []
    for m in names:
        dlt = cs[ref]["rank_ic"][sel] - cs[m]["rank_ic"][sel]
        t = nw_t(dlt)
        rows.append({"competitor": m, "dRankIC": np.nanmean(dlt), "t": t, "p_one_sided": 1 - stats.norm.cdf(t),
                     "days_ours_better %": 100 * np.nanmean(dlt > 0)})
    df = pd.DataFrame(rows)
    # Holm step-down on the one-sided p-values
    order = np.argsort(df.p_one_sided.values)
    adj, running = np.empty(len(df)), 0.0
    for i, j in enumerate(order):
        running = max(running, (len(df) - i) * df.p_one_sided.values[j])
        adj[j] = min(1.0, running)
    df["p_holm"] = adj
    return df


def regressions(Zs, R, days, data, raw, ref, names):
    """Encompassing (ours + C) and anomaly-adjusted (model + 16 signals) daily rank regressions."""
    enc = {m: [] for m in names}
    adj = {m: [] for m in [ref] + names}
    for d in days:
        ids = data.members[d]
        r = R[d, ids]
        sig = []
        for s in SIGNALS:
            x = raw[s][d, ids]
            rk = np.full(len(ids), 0.5)
            ok = np.isfinite(x)
            if ok.sum() > 1:
                rk[ok] = rank_pct(x[ok])
            sig.append(rk - 0.5)
        sig = np.stack(sig, 1)
        base = np.isfinite(r) & np.isfinite(Zs[ref][d, ids])
        for m in [ref] + names:
            ok = base & np.isfinite(Zs[m][d, ids])
            if ok.sum() < 100:
                if m != ref:
                    enc[m].append((np.nan, np.nan))
                adj[m].append(np.nan)
                continue
            y = rank_pct(r[ok]) - 0.5
            zm = rank_pct(Zs[m][d, ids][ok]) - 0.5
            X = np.column_stack([np.ones(ok.sum()), zm, sig[ok]])
            adj[m].append(np.linalg.lstsq(X, y, rcond=None)[0][1])
            if m != ref:
                zo = rank_pct(Zs[ref][d, ids][ok]) - 0.5
                X2 = np.column_stack([np.ones(ok.sum()), zo, zm])
                b = np.linalg.lstsq(X2, y, rcond=None)[0]
                enc[m].append((b[1], b[2]))
    return {m: np.asarray(v, float) for m, v in enc.items()}, {m: np.asarray(v, float) for m, v in adj.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="kospi")
    ap.add_argument("--base", default=str(RESULTS), help="folder holding oof/{nn,lin} and bench/oof/<model>")
    ap.add_argument("--competitors", default=",".join(COMPETITORS))
    ap.add_argument("--years", default="2016-2021", help="2016-2026 = extension to the test years (decided 13:45)")
    args = ap.parse_args()
    global YEARS
    a, b = (int(v) for v in args.years.split("-"))
    YEARS = range(a, b + 1)
    base = Path(args.base)
    out = base / "bench" / f"eval_{args.market}"
    out.mkdir(parents=True, exist_ok=True)
    data, _ = load_data4()
    F = list(data.features)
    raw = {s: data.stock[:, :, F.index(s)].astype(np.float64) for s in set(SIGNALS + ["Change", "Gap"]) if s in F}
    missing = [s for s in SIGNALS if s not in raw]
    if missing:                                  # e.g. a feature dropped by the coverage rule in another market
        print(f"signals missing from the data, left out: {missing}")
        SIGNALS[:] = [s for s in SIGNALS if s in raw]
    days = np.concatenate([data.expanding(k)[1] for k in YEARS])
    years_of_day = data.years[days]
    # returns: next open -> next close, close -> next close, next open -> close of t+5 (same year only)
    chg, gap = raw["Change"], raw["Gap"]
    R_cc = data.firm_y.astype(np.float64)
    R_oc = np.full(data.cap.shape, np.nan)
    R_oc5 = np.full(data.cap.shape, np.nan)
    last = len(data.dates) - 1
    for d in days:
        if d + 1 > last:
            continue
        R_oc[d] = 100 * ((1 + chg[d + 1]) / (1 + gap[d + 1]) - 1)
        if d + 5 <= last and data.years[d + 5] == data.years[d]:
            g = (1 + chg[d + 1]) / (1 + gap[d + 1])
            for j in range(2, 6):
                g = g * (1 + chg[d + j])
            R_oc5[d] = 100 * (g - 1)

    comps = [c for c in args.competitors.split(",") if c]
    Zs, seeds_used = {}, {}
    Zs["ours"], seeds_used["ours"] = load_preds(base / "oof" / "nn", data, seeds={0, 1, 2})
    Zs["ours_5seed"], seeds_used["ours_5seed"] = load_preds(base / "oof" / "nn", data)
    weights = None
    for c in comps:
        if c == "lin":
            Zs[c], seeds_used[c] = load_preds(base / "oof" / "lin", data)
        elif c == "anomaly":
            Zs[c], weights = anomaly_composite(data, raw)
            seeds_used[c] = []
        else:
            folder = base / "bench" / "oof" / c
            if not folder.exists() or not list(folder.glob(f"k{max(YEARS)}_s*.npz")):
                print(f"skip {c}: forecasts not finished")
                continue
            Zs[c], seeds_used[c] = load_preds(folder, data, seeds={0, 1, 2})
    comps = [c for c in comps if c in Zs]
    print("seeds used:", seeds_used, flush=True)
    periods = {"2016_2021": years_of_day <= 2021, "2018_2021": np.isin(years_of_day, list(VAL_YEARS))}
    if max(YEARS) >= 2022:
        periods |= {"2022_2026": years_of_day >= 2022, "2016_2026": np.ones(len(days), bool)}
    sel_all = periods["2016_2021"]
    report = {"market": args.market, "seeds": seeds_used, "anomaly_weights": weights, "days": int(len(days)),
              "signals": list(SIGNALS), "firm_features": F}
    for rname, R in (("open_entry", R_oc), ("close", R_cc), ("open_entry_5d", R_oc5)):
        cs = daily_cs(Zs, R, days, data)
        for pname, sel in periods.items():
            summarise(cs, years_of_day, sel).to_csv(out / f"summary_{rname}_{pname}.csv", index=False)
            paired_tests(cs, "ours", sel, comps).to_csv(out / f"paired_{rname}_{pname}.csv", index=False)
        by_year(cs, years_of_day).to_csv(out / f"rankic_by_year_{rname}.csv")
        if rname == "open_entry":
            for pname, sel in periods.items():
                tests = paired_tests(cs, "ours", sel, comps)
                print(f"--- open entry, {pname}")
                print(summarise(cs, years_of_day, sel).round(4).to_string(index=False))
                print(tests.round(4).to_string(index=False), flush=True)
                report[f"paired_{pname}"] = tests.to_dict("records")
            report["primary"] = report["paired_2016_2021"]
    enc, adj = regressions(Zs, R_oc, days, data, raw, "ours", [c for c in comps])
    rows = []
    for m, v in enc.items():
        rows.append({"competitor": m, "slope_ours": np.nanmean(v[:, 0]), "t_ours": nw_t(v[:, 0]),
                     "slope_comp": np.nanmean(v[:, 1]), "t_comp": nw_t(v[:, 1])})
    pd.DataFrame(rows).to_csv(out / "encompassing_open_entry.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    rows = [{"model": m, "slope_after_16_signals": np.nanmean(v), "t": nw_t(v)} for m, v in adj.items()]
    pd.DataFrame(rows).to_csv(out / "anomaly_adjusted_open_entry.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # single-seed spread (RankIC open entry, 2016-2021)
    rows = []
    for name, folder, seeds in [("ours", base / "oof" / "nn", range(5))] + \
            [(c, base / "bench" / "oof" / c, range(3)) for c in comps if c not in ("lin", "anomaly")]:
        for s in seeds:
            try:
                Z1, _ = load_preds(folder, data, seeds={s})
            except FileNotFoundError:
                continue
            cs1 = daily_cs({name: Z1}, R_oc, days, data)
            rows.append({"model": name, "seed": s, "RankIC": np.nanmean(cs1[name]["rank_ic"])})
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(out / "single_seed_rankic_open_entry.csv", index=False)
    print(seed_df.groupby("model").RankIC.agg(["mean", "std", "min", "max", "count"]).round(4).to_string())
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
