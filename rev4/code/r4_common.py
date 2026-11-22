"""Rev4 common layer: the rev2/rev3 daily dataset plus optional new inputs, folds, baselines and paths.

New inputs are picked up automatically when the processed files exist (written by r4_ingest.py):
  rev4/data/processed/firm_daily_extra.parquet    Date, Code, <daily firm features, already transformed>
  rev4/data/processed/market_daily_extra.parquet  Date, <daily market features, already transformed>
  rev4/data/processed/firm_quarterly.parquet      Code, FiscalPeriodEnd, AvailableFrom, <accounting ratios>
  rev4/data/processed/relations.parquet           ValidFrom, Code, Sector
Without them the pipeline runs on the existing 18 firm and 7 market features with the correlation graph.
Scaling follows PREPROCESSING_SPEC.md: training-window statistics only, firm winsorisation 1%/99%, discrete
indicators centred only, market z-scores clipped at +-5, missing values filled with 0 after scaling.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV4 = HERE.parent
ROOT = REV4.parent
sys.path.insert(0, str(ROOT / "rev3" / "code"))
sys.path.insert(0, str(ROOT / "code"))
from r3_core import DailyData3, dense, head_moments, head_predict, solve_head, weights_and_kl  # noqa: E402,F401
from pilot_data import LOOKBACK, MARKET_FEATURES, checks  # noqa: E402,F401
from pilot_neural import CorrelationGraphs, NeuralBatches  # noqa: E402,F401

PROCESSED = Path(os.environ.get("R4_PROCESSED", REV4 / "data" / "processed"))
RESULTS = REV4 / "results"
VALIDATION_YEARS = (2018, 2019, 2020, 2021)
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
OOF_YEARS = tuple(range(2016, 2027))
STAGE2_FIRST_YEAR = 2016
MARKET_CLIP = 5.0
FIN_QUARTERS = 8
FIN_STALE_DAYS = 243          # 8 months: PREPROCESSING_SPEC 3.2
SECTOR_LEADERS = 10           # each firm links to the 10 largest same-industry firms (Hou 2007)
# Validation fix 2026-11-20 22:30 (POSTER_RUN_PLAN 8.3, VALIDATION_LOG attempt 2), adopted on principle before rerun:
# a feature observed on fewer than half of the training days, or constant over them, is switched off for that window.
# A late-starting input acts as a time dummy and is standardised on a few months, so the forecast year sees values far
# outside training (2017: MktLogBM |z|>4 on 55% of days; NN seed-0 firm forecasts averaged -0.5%/day). The rule changes
# only the 2016-2020 windows; 2021-2026 windows are untouched (checked before adoption).
COVERAGE_MIN = 0.5
TIED_INDICATORS = {"FinMissing": "EP", "EPneg": "EP", "FilingDay": "LogBM", "LogDaysSinceFiling": "LogBM"}
GRAPH_DIGEST_DIR = Path(os.environ.get("R4_GRAPH_DIR", ROOT / "results_daily_2015" / "correlation_graphs"))
DATA_ROOT = Path(os.environ.get("R4_DATA_ROOT", ROOT))      # KOSDAQ replication: the kosdaq/ folder


def log_to(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as h:
            h.write(line + "\n")
    return log


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@dataclass
class Fold4:
    dates: np.ndarray
    stock: np.ndarray
    market: np.ndarray
    cap_pct: np.ndarray
    firm_stats: pd.DataFrame
    market_stats: pd.DataFrame
    fin: object = None


class DailyData4(DailyData3):
    """rev3 data (same day/code indexing as rev2, so the rev2 correlation-graph cache stays valid) + new inputs."""

    def __init__(self, folder, index_file, market, extra_firm=None, extra_market=None):
        super().__init__(folder, index_file, market)
        self.market_names = list(MARKET_FEATURES)
        self.extra_firm_names, self.extra_market_names = [], []
        if extra_market is not None:
            self._add_market(extra_market)
        if extra_firm is not None:
            self._add_firm(extra_firm)
        self.fin = None
        self.sector = None

    def _day_index(self, dates):
        return pd.DatetimeIndex(self.dates.astype("datetime64[ns]")).get_indexer(pd.to_datetime(dates))

    def _add_market(self, frame):
        cols = [c for c in frame.columns if c != "Date"]
        idx = pd.DatetimeIndex(self.dates.astype("datetime64[ns]"))
        aligned = frame.assign(Date=pd.to_datetime(frame.Date)).set_index("Date").reindex(idx)[cols]
        self.market = np.concatenate([self.market, aligned.to_numpy(np.float64)], axis=1)
        self.market_names += cols
        self.extra_market_names = cols

    def _add_firm(self, frame):
        cols = [c for c in frame.columns if c not in ("Date", "Code")]
        day = self._day_index(frame.Date)
        code = self.codes.get_indexer(frame.Code.astype(str).str.zfill(6))
        keep = (day >= 0) & (code >= 0)
        extra = np.full((*self.stock.shape[:2], len(cols)), np.nan, np.float32)
        extra[day[keep], code[keep]] = frame.loc[keep, cols].to_numpy(np.float32)
        self.stock = np.concatenate([self.stock, extra], axis=2)
        self.features = list(self.features) + cols
        self.extra_firm_names = cols

    def scale(self, training, prepare_history=True):
        x = self.stock[training][self.universe[training]]
        n_unique = np.array([len(np.unique(col[np.isfinite(col)])) for col in x.T])
        discrete = n_unique <= 3
        uni = self.universe[training]
        day_cov = np.array([(np.isfinite(self.stock[training, :, f]) & uni).any(1).mean()
                            for f in range(self.stock.shape[-1])])
        lo, hi = np.nanquantile(x, [0.01, 0.99], axis=0)
        lo[discrete], hi[discrete] = -np.inf, np.inf
        clipped = np.clip(x, lo, hi)
        with np.errstate(all="ignore"):
            mu, sigma = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        # A feature without variation in this training window (e.g. a short-sale-ban flag before 2020, or a block
        # whose data start later) carries no information here: it is switched off (z = 0) for this window only.
        # rev3 raised an error instead; that was right for the fixed 18 features but not for staggered new inputs.
        off = ~np.isfinite(mu) | (~discrete & ~(sigma > 0))
        off |= (day_cov < COVERAGE_MIN) | (n_unique <= 1)              # coverage and constant rules (see top)
        names = list(self.features)
        for ind, src in TIED_INDICATORS.items():                    # missing/filing indicators follow their block
            if ind in names and src in names and off[names.index(src)]:
                off[names.index(ind)] = True
        if off.any():
            print(f"[scale] switched off in this window: {[f for f, o in zip(self.features, off) if o]}", flush=True)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sigma = np.where(discrete | off, 1.0, sigma)
        del x, clipped
        z = np.clip(self.stock, lo.astype(np.float32), hi.astype(np.float32))
        z -= mu.astype(np.float32)
        z /= sigma.astype(np.float32)
        z[..., off] = 0
        z[~np.isfinite(z)] = 0
        if prepare_history:
            z[~self.universe] = 0
        s_train = self.market[training]
        with np.errstate(all="ignore"):
            s_mu, s_sigma = np.nanmean(s_train, axis=0), np.nanstd(s_train, axis=0)
        s_off = (np.isfinite(s_train).mean(0) < COVERAGE_MIN) | ~(s_sigma > 0)
        if s_off.any():
            print(f"[scale] market switched off in this window: {[f for f, o in zip(self.market_names, s_off) if o]}",
                  flush=True)
        s_sigma = np.where(s_sigma > 0, s_sigma, 1)
        s = np.clip(np.nan_to_num((self.market - s_mu) / s_sigma), -MARKET_CLIP, MARKET_CLIP).astype(np.float32)
        s[:, s_off] = 0
        fin = self.fin.scaled(training) if self.fin is not None else None
        return Fold4(training, z, s, self.cap_pct,
                     pd.DataFrame({"feature": self.features, "discrete_centred_only": discrete, "switched_off": off,
                                   "p01": lo, "p99": hi, "mean": mu, "std": sigma}),
                     pd.DataFrame({"feature": self.market_names, "mean": s_mu, "std": s_sigma}), fin)


class FinancialSequences:
    """Quarterly accounting ratios as the last FIN_QUARTERS quarters usable on each day (AvailableFrom <= day)."""

    def __init__(self, table, data):
        table = table.copy()
        table["Code"] = table.Code.astype(str).str.zfill(6)
        table["AvailableFrom"] = pd.to_datetime(table.AvailableFrom)
        table["FiscalPeriodEnd"] = pd.to_datetime(table.FiscalPeriodEnd)
        table = table.loc[table.Code.isin(data.codes)].sort_values(["Code", "AvailableFrom", "FiscalPeriodEnd"])
        table = table.drop_duplicates(["Code", "FiscalPeriodEnd"], keep="first").reset_index(drop=True)
        self.names = [c for c in table.columns if c not in ("Code", "FiscalPeriodEnd", "AvailableFrom")]
        self.values = table[self.names].to_numpy(np.float64)
        dates = data.dates.astype("datetime64[ns]")
        self.avail_day = np.searchsorted(dates, table.AvailableFrom.to_numpy("datetime64[ns]"), side="left")
        code_id = data.codes.get_indexer(table.Code)
        self.code_start = np.full(len(data.codes), -1, np.int64)
        latest = np.full((len(dates), len(data.codes)), -1, np.int32)
        for c in np.unique(code_id):
            rows = np.flatnonzero(code_id == c)
            self.code_start[c] = rows[0]
            for j, r in enumerate(rows):
                stop = self.avail_day[rows[j + 1]] if j + 1 < len(rows) else len(dates)
                if self.avail_day[r] < len(dates):
                    latest[self.avail_day[r]:stop, c] = r
        fiscal = table.FiscalPeriodEnd.to_numpy("datetime64[ns]")
        age = (dates[:, None] - fiscal[np.maximum(latest, 0)]).astype("timedelta64[D]").astype(np.int64)
        latest[(latest >= 0) & (age > FIN_STALE_DAYS)] = -1
        self.latest = latest
        self.row_code = code_id

    @classmethod
    def load(cls, path, data):
        return cls(pd.read_parquet(path), data)

    def scaled(self, training):
        training = np.asarray(training)
        last_training_day = int(np.max(training))
        used = self.avail_day <= last_training_day
        coverage = float((self.latest[training] >= 0).any(1).mean())      # share of training days with any filing
        if used.any() and coverage < COVERAGE_MIN:
            print(f"[scale] financial sequences switched off in this window (filings on {coverage:.0%} of days)",
                  flush=True)
            used = np.zeros_like(used)
        if not used.any():
            # no quarter filed before the window ends: F(2016) trains on 2014-10..2015-12, the first DART statement
            # (FY2015 annual) is filed 2016-03. Switch the block off, masks included, so the model never sees a
            # "present" flag at forecast time that it never saw in training (crashed the real run 2026-11-20 20:47)
            print("[scale] financial sequences switched off in this window (no filings yet)", flush=True)
            return {"z": np.zeros(self.values.shape, np.float32), "latest": np.full_like(self.latest, -1),
                    "code_start": self.code_start, "row_code": self.row_code}
        x = self.values[used]
        lo, hi = np.nanquantile(x, [0.01, 0.99], axis=0)
        clipped = np.clip(x, lo, hi)
        mu, sd = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        z = np.nan_to_num((np.clip(self.values, lo, hi) - mu) / sd).astype(np.float32)
        return {"z": z, "latest": self.latest, "code_start": self.code_start, "row_code": self.row_code}

    @staticmethod
    def gather(fin, days, codes):
        """(n, FIN_QUARTERS, q + 1): oldest-to-newest quarters, last channel = present mask."""
        latest = fin["latest"][days, codes]
        start = fin["code_start"][codes]
        out = np.zeros((len(codes), FIN_QUARTERS, fin["z"].shape[1] + 1), np.float32)
        for k in range(FIN_QUARTERS):
            row = latest - (FIN_QUARTERS - 1 - k)
            ok = (latest >= 0) & (row >= start)
            out[ok, k, :-1] = fin["z"][row[ok]]
            out[ok, k, -1] = 1.0
        return out


class SectorGraph:
    """Same-industry edges: each firm links to the SECTOR_LEADERS largest firms of its industry that day."""

    def __init__(self, table, data):
        table = table.copy()
        table["Code"] = table.Code.astype(str).str.zfill(6)
        table["ValidFrom"] = pd.to_datetime(table.ValidFrom)
        sectors = pd.Index(sorted(table.Sector.dropna().unique()))
        dates = data.dates.astype("datetime64[ns]")
        self.sector = np.full((len(dates), len(data.codes)), -1, np.int16)
        for valid_from, snap in table.sort_values("ValidFrom").groupby("ValidFrom"):
            start = int(np.searchsorted(dates, np.datetime64(valid_from, "ns"), side="right"))   # next session
            codes = data.codes.get_indexer(snap.Code)
            ok = codes >= 0
            row = np.full(len(data.codes), -1, np.int16)
            row[codes[ok]] = sectors.get_indexer(snap.Sector[ok])
            self.sector[start:, :] = row
        self.data = data
        self.names = list(sectors)

    @classmethod
    def load(cls, path, data):
        return cls(pd.read_parquet(path), data)

    def edges(self, day):
        ids = self.data.members[day]
        sec = self.sector[day, ids]
        cap = self.data.cap[day, ids]
        src, dst = [np.arange(len(ids))], [np.arange(len(ids))]
        for s in np.unique(sec[sec >= 0]):
            members = np.flatnonzero(sec == s)
            leaders = members[np.argsort(-cap[members])[:SECTOR_LEADERS]]
            a = np.repeat(members, len(leaders))
            b = np.tile(leaders, len(members))
            keep = a != b
            src += [a[keep], b[keep]]
            dst += [b[keep], a[keep]]
        pairs = np.unique(np.stack([np.concatenate(src), np.concatenate(dst)]), axis=1)
        return pairs.astype(np.int32)


class GraphProvider:
    """Union of the trailing correlation kNN graph (rev2 cache) and the industry graph when available."""

    def __init__(self, data, use_correlation=True, use_sector=True):
        roots = [p for p in GRAPH_DIGEST_DIR.iterdir() if p.is_dir()] if GRAPH_DIGEST_DIR.exists() else []
        self.corr = CorrelationGraphs(data, roots[0]) if (use_correlation and roots) else None
        self.sector = data.sector if use_sector else None
        if self.corr is None and self.sector is None:
            raise ValueError("No firm graph available.")
        self.kind = "+".join(k for k, v in (("correlation", self.corr), ("industry", self.sector)) if v is not None)

    def __getitem__(self, day):
        parts = []
        if self.corr is not None:
            parts.append(np.asarray(self.corr[day]))
        if self.sector is not None:
            parts.append(self.sector.edges(day))
        edges = np.concatenate(parts, axis=1)
        return np.unique(edges, axis=1) if len(parts) > 1 else edges


def load_data4(root=DATA_ROOT):
    market, gate = checks(root / "daily_dataset", root / "inputs" / "market_indices.csv")
    if not gate["gate_pass"]:
        raise SystemExit("Data gate failed.")
    ef, em = PROCESSED / "firm_daily_extra.parquet", PROCESSED / "market_daily_extra.parquet"
    data = DailyData4(root / "daily_dataset", root / "inputs" / "market_indices.csv", market,
                      pd.read_parquet(ef) if ef.exists() else None, pd.read_parquet(em) if em.exists() else None)
    fq, rel = PROCESSED / "firm_quarterly.parquet", PROCESSED / "relations.parquet"
    if fq.exists():
        data.fin = FinancialSequences.load(fq, data)
    if rel.exists():
        data.sector = SectorGraph.load(rel, data)
    gate["inputs"] = {
        "firm_features": list(data.features), "market_features": list(data.market_names),
        "financial_sequence": data.fin.names if data.fin is not None else None,
        "industry_graph": data.sector is not None,
        "files": {p.name: file_sha(p) for p in (ef, em, fq, rel) if p.exists()},
    }
    return data, gate


def year_days(data, year):
    """Evaluation days of a calendar year: labelled, target realised in the same year (rev2/rev3 rule)."""
    return data.expanding(year)[1]


def b0_b3r(data, year, lam=1000.0):
    """B0 (mean of training index returns) and B3R (cap-weighted features + ridge) for one evaluation year."""
    train, test = data.expanding(year)
    fold = data.scale(train)
    head = solve_head(head_moments(dense(data, fold, train)), lam)
    b3r = head_predict(head, dense(data, fold, test))
    b0 = np.full(len(test), data.y[train].mean())
    return test, b0, b3r


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                    encoding="utf-8")
