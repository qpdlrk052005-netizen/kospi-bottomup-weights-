"""Data gates, chronological splits, and fold-local preprocessing for the daily pilot."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

START = np.datetime64("2015-01-02")
END = np.datetime64("2026-10-30")
TEST_YEARS = range(2022, 2027)
FIRM_FEATURES = (
    "Change", "Vol20", "VolumeRatio", "Disp20", "Disp60", "RSI14",
    "LogMarcap", "Up", "Ret5", "Mom12_1", "High52", "Issuance",
    "Range", "CloseLoc", "Gap", "LimitHit", "Turnover", "Amihud",
    "ForeignNetBuy", "InstNetBuy", "IndivNetBuy",
)
MARKET_FEATURES = (
    "IndexReturn", "IndexVol20", "IndexMom20", "IndexMom60",
    "AvgVar", "AvgCorr", "Breadth",
)
LOOKBACK = 60


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checks(folder, index_file):
    market = pd.read_parquet(folder / "daily_market.parquet").sort_values("Date")
    validation = pd.read_csv(folder / "audit_daily_index_replication.csv").set_index("Decade")
    corr = validation["Corr"].reindex(["2010s", "2020s"])
    pilot = market.loc[market.Date.between(pd.Timestamp(START), pd.Timestamp(END))]
    expected = pd.read_csv(index_file, usecols=["date"], parse_dates=["date"]).date
    expected = expected.loc[expected.between(pd.Timestamp(START), pd.Timestamp(END))]
    actual = pd.DatetimeIndex(pilot.Date)
    complete = (
        actual.is_unique
        and actual.equals(pd.DatetimeIndex(expected))
        and len(pilot) > 0
        and np.isfinite(pilot.NextDayIndexReturn.iloc[:-1]).all()
        and pd.isna(pilot.NextDayIndexReturn.iloc[-1])
    )
    counts = pilot.NValid
    min_day = pilot.loc[counts.eq(counts.min()), "Date"].dt.strftime("%Y-%m-%d").tolist()
    report = {
        "gate_pass": bool(corr.ge(0.98).all() and complete and counts.ge(500).all()),
        "replication_corr": {key: float(val) for key, val in corr.items()},
        "tracking_std_pct_points": {
            key: float(validation.at[key, "TrackingStdPctPt"]) for key in corr.index
        },
        "target_gate": bool(complete),
        "index_sessions_including_final": len(pilot),
        "index_targets_present": int(pilot.NextDayIndexReturn.notna().sum()),
        "missing_target_dates": pilot.loc[pilot.NextDayIndexReturn.isna(), "Date"].dt.strftime("%Y-%m-%d").tolist(),
        "valid_firm_median": float(counts.median()),
        "valid_firm_min": int(counts.min()),
        "valid_firm_min_dates": min_day,
    }
    return market, report


@dataclass
class Fold:
    dates: np.ndarray
    stock: np.ndarray
    market: np.ndarray
    cap_pct: np.ndarray
    firm_stats: pd.DataFrame
    market_stats: pd.DataFrame


class DailyData:
    def __init__(self, folder, index_file, market):
        folder = Path(folder)
        market = market.set_index("Date")
        first = market.index.get_loc(pd.Timestamp(START))
        market = market.iloc[first - LOOKBACK + 1:]
        self.dates = market.index.to_numpy(dtype="datetime64[D]")
        self.years = self.dates.astype("datetime64[Y]").astype(int) + 1970
        self.next_dates = np.r_[self.dates[1:], np.datetime64("NaT", "D")]
        self.next_years = np.r_[self.years[1:], -1]
        self.market = market[list(MARKET_FEATURES)].to_numpy(dtype=np.float64)
        self.y = 100 * market.NextDayIndexReturn.to_numpy(dtype=np.float64)
        self.labeled = np.isfinite(self.y) & (self.dates >= START)
        self.paths = (folder / "daily_firms.parquet", folder / "daily_market.parquet", Path(index_file))

        available = pq.ParquetFile(self.paths[0]).schema_arrow.names
        columns = [c for c in FIRM_FEATURES if c in available]
        firms = pd.read_parquet(
            self.paths[0], columns=["Date", "Code", "Marcap", "Valid", "NextDayReturn", *columns],
            filters=[("Date", ">=", market.index[0])],
        )
        day_id = market.index.get_indexer(firms.Date)
        self.codes = pd.Index(sorted(firms.Code.unique()), name="Code")
        code_id = self.codes.get_indexer(firms.Code)
        if (day_id < 0).any() or firms.duplicated(["Date", "Code"]).any():
            raise ValueError("Firm rows must match unique market-date and code pairs.")
        shape = (len(market), len(self.codes))
        self.valid = np.zeros(shape, dtype=bool)
        self.valid[day_id, code_id] = firms.Valid.to_numpy(dtype=bool)
        valid_count = np.cumsum(self.valid, axis=0, dtype=np.int16)
        valid_count[LOOKBACK:] -= valid_count[:-LOOKBACK].copy()
        self.universe = self.valid & (valid_count >= 40)
        active = self.universe[day_id, code_id] & (self.dates[day_id] >= START)
        raw = firms[columns].to_numpy(dtype=np.float64, copy=True)
        raw[~np.isfinite(raw)] = np.nan
        miss = dict(zip(columns, np.isnan(raw[active]).mean(axis=0).tolist(), strict=True))
        initial = active & (self.next_dates[day_id] <= np.datetime64("2019-12-31"))
        initial_miss = dict(zip(columns, np.isnan(raw[initial]).mean(axis=0).tolist(), strict=True))
        self.feature_coverage = pd.DataFrame({
            "feature": FIRM_FEATURES,
            "missing_share": [miss.get(c, 1.0) for c in FIRM_FEATURES],
            "present_in_parquet": [c in available for c in FIRM_FEATURES],
            "selection_train_missing_share": [initial_miss.get(c, 1.0) for c in FIRM_FEATURES],
        })
        self.feature_coverage["used"] = self.feature_coverage.missing_share.le(0.05)
        self.features = self.feature_coverage.loc[self.feature_coverage.used, "feature"].tolist()
        if len(self.features) < 10:
            raise ValueError(f"Only {len(self.features)} firm features satisfy the pilot coverage rule.")
        raw = raw[:, [columns.index(c) for c in self.features]]
        if "Amihud" in self.features:
            k = self.features.index("Amihud")
            raw[:, k] = np.log(raw[:, k] + 1e-8)
        self.stock = np.full((*shape, len(self.features)), np.nan, dtype=np.float32)
        self.stock[day_id, code_id] = raw
        self.cap = np.zeros(shape, dtype=np.float64)
        self.cap[day_id, code_id] = firms.Marcap.to_numpy(dtype=np.float64)
        if not np.all(self.cap[self.universe] > 0):
            raise ValueError("All eligible firms must have positive current capitalization.")
        self.cap_pct = np.zeros(shape, dtype=np.float32)
        for day, ids in enumerate(map(np.flatnonzero, self.universe)):
            if len(ids) == 1:
                self.cap_pct[day, ids[0]] = 1.0
            elif len(ids) > 1:
                ranks = rankdata(-self.cap[day, ids], method="average")
                self.cap_pct[day, ids] = 1 - (ranks - 1) / (len(ids) - 1)
        self.firm_y = np.full(shape, np.nan, dtype=np.float32)
        self.firm_y[day_id, code_id] = 100 * firms.NextDayReturn.to_numpy(dtype=np.float32)
        self.members = [np.flatnonzero(mask) for mask in self.universe]

    def selection(self):
        train = self.labeled & (self.next_dates <= np.datetime64("2019-12-31"))
        validation = self.labeled & (self.years >= 2020) & (self.next_dates <= np.datetime64("2021-12-31"))
        return np.flatnonzero(train), np.flatnonzero(validation)

    def expanding(self, year, include_test_year_end=False):
        train = self.labeled & (self.next_dates < np.datetime64(f"{year}-01-01"))
        test = self.labeled & (self.years == year)
        if not include_test_year_end:
            test &= self.next_years == year
        return np.flatnonzero(train), np.flatnonzero(test)

    def ar(self, days):
        return 100 * self.market[days[:, None] - np.arange(5), 0]

    def scale(self, training, prepare_history=True):
        x = self.stock[training][self.universe[training]]
        lo, hi = np.nanquantile(x, [0.01, 0.99], axis=0)
        clipped = np.clip(x, lo, hi)
        mu = np.nanmean(clipped, axis=0)
        sigma = np.nanstd(clipped, axis=0)
        sigma = np.where(sigma > 0, sigma, 1)
        del x, clipped
        z = np.clip(self.stock, lo.astype(np.float32), hi.astype(np.float32))
        z -= mu.astype(np.float32)
        z /= sigma.astype(np.float32)
        z[~np.isfinite(z)] = 0
        if prepare_history:
            z[~self.universe] = 0

        s_train = self.market[training]
        s_mu = np.nanmean(s_train, axis=0)
        s_sigma = np.nanstd(s_train, axis=0)
        s_sigma = np.where(s_sigma > 0, s_sigma, 1)
        s = np.nan_to_num((self.market - s_mu) / s_sigma).astype(np.float32)
        return Fold(
            training, z, s, self.cap_pct,
            pd.DataFrame({"feature": self.features, "p01": lo, "p99": hi, "mean": mu, "std": sigma}),
            pd.DataFrame({"feature": MARKET_FEATURES, "mean": s_mu, "std": s_sigma}),
        )

    def locked_inputs(self):
        return {
            "firm_features": self.features,
            "dropped_firm_features": self.feature_coverage.loc[~self.feature_coverage.used, "feature"].tolist(),
            "market_features": list(MARKET_FEATURES),
            "file_sha256": {p.name: sha256(p) for p in self.paths},
            "split_rule": (
                "Within one calendar year for test folds; expanding training uses all "
                "labels realized before January 1 of its test year."
            ),
        }
