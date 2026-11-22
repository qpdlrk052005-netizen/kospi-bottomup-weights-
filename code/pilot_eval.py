"""Prediction metrics, loss-differential tests, and per-day weight diagnostics."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def r2(y, forecast, reference):
    denominator = np.square(y - reference).sum()
    return float(1 - np.square(y - forecast).sum() / denominator) if denominator else np.nan


def metrics(frame, by_year=False):
    keys = ["model", "seed"] + (["year"] if by_year else [])
    output = []
    for key, g in frame.groupby(keys, sort=True):
        y, prediction, reference = (g[col].to_numpy() for col in ("actual_pct", "prediction_pct", "b0_pct"))
        actual_up, actual_down = y > 0, y < 0
        output.append({
            **dict(zip(keys, key, strict=True)),
            "n": len(g), "mse_pct_points_sq": np.square(y - prediction).mean(),
            "oos_r2": r2(y, prediction, reference),
            "direction_accuracy": (np.sign(y) == np.sign(prediction)).mean(),
            "actual_up_days": int(actual_up.sum()),
            "up_hit_rate": float((prediction[actual_up] > 0).mean()) if actual_up.any() else np.nan,
            "actual_down_days": int(actual_down.sum()),
            "down_hit_rate": float((prediction[actual_down] < 0).mean()) if actual_down.any() else np.nan,
            "up_forecast_rate": (prediction > 0).mean(), "up_actual_rate": (y > 0).mean(),
        })
    return pd.DataFrame(output)


def direction_hit_rates(frame):
    """Class-conditional direction hits and confusion counts, overall and by year."""
    output = []
    for (model, seed), model_rows in frame.groupby(["model", "seed"], sort=True):
        periods = [("overall", model_rows)]
        periods.extend((str(year), group) for year, group in model_rows.groupby("year", sort=True))
        for period, group in periods:
            actual = group.actual_pct.to_numpy(dtype=float)
            forecast = group.prediction_pct.to_numpy(dtype=float)
            actual_up, actual_down, actual_flat = actual > 0, actual < 0, actual == 0
            forecast_up, forecast_down, forecast_flat = forecast > 0, forecast < 0, forecast == 0
            up_n, down_n = int(actual_up.sum()), int(actual_down.sum())
            up_correct = int((actual_up & forecast_up).sum())
            down_correct = int((actual_down & forecast_down).sum())
            output.append({
                "model": model,
                "seed": str(seed),
                "period": period,
                "n_days": len(group),
                "actual_up_days": up_n,
                "up_correct": up_correct,
                "up_hit_rate": up_correct / up_n if up_n else np.nan,
                "actual_down_days": down_n,
                "down_correct": down_correct,
                "down_hit_rate": down_correct / down_n if down_n else np.nan,
                "actual_flat_days": int(actual_flat.sum()),
                "actual_up_predicted_down": int((actual_up & forecast_down).sum()),
                "actual_up_predicted_flat": int((actual_up & forecast_flat).sum()),
                "actual_down_predicted_up": int((actual_down & forecast_up).sum()),
                "actual_down_predicted_flat": int((actual_down & forecast_flat).sum()),
            })
    return pd.DataFrame(output)


def prediction_frame(data, days, forecast, reference, model, seed="ensemble"):
    return pd.DataFrame({
        "date": data.dates[days], "target_date": data.next_dates[days], "year": data.years[days],
        "model": model, "seed": str(seed), "actual_pct": data.y[days],
        "prediction_pct": forecast, "b0_pct": reference,
    })


def differential_test(wide, model, benchmark, kind, lag=5):
    raw = np.square(wide["actual_pct"] - wide[benchmark]) - np.square(wide["actual_pct"] - wide[model])
    scores = raw.to_numpy(dtype=float)
    if kind == "Clark–West":
        scores = scores + np.square(wide[benchmark] - wide[model]).to_numpy()
    n = len(scores)
    centered = scores - scores.mean()
    long_var = centered @ centered / n
    for k in range(1, min(lag + 1, n)):
        long_var += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    error = np.sqrt(long_var / n) if long_var > 0 else np.nan
    statistic = scores.mean() / error
    return {
        "model": model, "benchmark": benchmark, "test": kind, "sample_days": n,
        "raw_mse_benchmark_minus_model": raw.mean(), "mean_test_differential": scores.mean(),
        "nw_lag": lag, "statistic": statistic,
        "p_value": norm.sf(statistic) if kind == "Clark–West" else 2 * norm.sf(abs(statistic)),
        "p_value_sides": 1 if kind == "Clark–West" else 2,
        "raw_mse_gain_95ci_low": raw.mean() - 1.96 * error if kind != "Clark–West" else np.nan,
        "raw_mse_gain_95ci_high": raw.mean() + 1.96 * error if kind != "Clark–West" else np.nan,
    }


def comparison_tests(predictions, main_model=None):
    ensemble = predictions.loc[predictions.seed.eq("ensemble")]
    wide = ensemble.pivot(index="date", columns="model", values="prediction_pct").sort_index()
    wide["actual_pct"] = ensemble.groupby("date").actual_pct.first()
    comparisons = [("B6", "B3", "Diebold–Mariano")]
    if main_model is not None:
        comparisons.extend((main_model, b, "Clark–West") for b in ("B3", "B0"))
        comparisons.extend((main_model, b, "Diebold–Mariano") for b in ("B2", "B4", "B5", "B6"))
    return pd.DataFrame([differential_test(wide, *test) for test in comparisons])


def diag_weights(data, days, weights, model, split, seed="ensemble", deltas=None):
    diag, by_decile = [], []
    last_day, last_ids, last_weight = -2, None, None
    for j, (day, w) in enumerate(zip(days, weights, strict=True)):
        ids = data.members[day]
        by_cap = np.argsort(data.cap[day, ids], kind="stable")
        previous_l1 = np.nan
        if last_day == day - 1:
            _, prior, current = np.intersect1d(last_ids, ids, assume_unique=True, return_indices=True)
            previous_l1 = np.abs(w[current] - last_weight[prior]).sum()
        row = {
            "date": data.dates[day], "year": data.years[day], "model": model, "seed": str(seed),
            "split": split, "n_firms": len(ids), "effective_firms": 1 / (w @ w),
            "small_cap_weight": w[by_cap[:len(ids) // 2]].sum(),
            "max_weight": w.max(), "weight_change_l1": previous_l1,
        }
        if deltas is not None:
            delta = deltas[j]
            for name, pct in zip(("p01", "p05", "p50", "p95", "p99"), np.quantile(delta, [0.01, 0.05, 0.50, 0.95, 0.99]), strict=True):
                row[f"delta_{name}"] = pct
            for decile, subset in enumerate(np.array_split(by_cap, 10), start=1):
                by_decile.append({
                    "date": data.dates[day], "year": data.years[day], "model": model,
                    "seed": str(seed), "split": split, "market_cap_decile": decile,
                    "count_firms": len(subset), "delta_sum": float(delta[subset].sum()),
                })
        diag.append(row)
        last_day, last_ids, last_weight = day, ids, w
    return pd.DataFrame(diag), pd.DataFrame(by_decile)


def combine_deciles(frames):
    combined = pd.concat(frames, ignore_index=True)
    group = ["year", "model", "seed", "split", "market_cap_decile"]
    combined = combined.groupby(group, as_index=False)[["count_firms", "delta_sum"]].sum()
    combined["mean_delta"] = combined.delta_sum / combined.count_firms
    return combined.drop(columns="delta_sum")
