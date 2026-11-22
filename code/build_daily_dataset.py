"""Build the daily KOSPI dataset for next-day index prediction with learned firm weights.

Inputs
  --daily        krx_1995_2026_daily.parquet from build_panel.py --daily (adjusted prices, 7 base features)
  --raw          folder of marcap-YYYY.parquet (same source as build_panel.py), for open/high/low
                 and trading value. Optional; without it the OHLC and value features stay missing.
  --fundamentals krx_fundamentals_annual.csv (TS2000 annual statements, exploratory)
  --flows        investor_flows_kospi.csv from download_investor_flows.py (optional)

Outputs (in --output)
  daily_firms.parquet      one row per KOSPI common-share proxy and session
  daily_market.parquet     one row per session, with the next-day KOSPI return target
  feature_coverage.csv     per feature: first year with >= 50% coverage of valid rows, overall share
  audit_daily_index_replication.csv  by decade: correlation and tracking error between the
                           cap-weighted return of valid firms (previous-close weights) and the KOSPI

Rows are firms on a day, columns are firm features in their original units (no
per-day ranks): a weighted average of a price-scaled ratio over firms is the same
ratio for the weighted basket. Standardise with training-period statistics inside the
experiment, never here. A feature whose coverage starts after --min-start is dropped
from the outputs and listed as such in feature_coverage.csv.

Firm features (see FEATURE_NOTES for definitions and sources)
  base     Change Vol20 VolumeRatio Disp20 Disp60 RSI14 LogMarcap   (build_panel.py)
  price    Up Ret5 Mom12_1 High52 Issuance                           (adjusted daily prices)
  ohlc     Range CloseLoc Gap LimitHit Turnover Amihud              (--raw)
  value    BM EP SP CFP                                              (statement / issuer cap)
  slow     ROE ROA OperProf GrossProf AssetGrowth SalesGrowth Leverage Accruals CashToAssets
  flow     ForeignNetBuy InstNetBuy IndivNetBuy                      (net buy value / cap)
Market variables (daily_market.parquet)
  IndexReturn IndexVol20 IndexMom20 IndexMom60 AvgVar AvgCorr Breadth
Masks
  Valid     common-share proxy, traded that day, all 7 base features finite, cap > 0
  FinValid  BM, EP, SP finite;  FlowValid  foreign net buy observed
Statement timing follows panel.attach_fundamentals: a fiscal period ending in month f
is used from the last session of month f + --fs-lag until replaced, at most 12 months.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_daily_targets import daily_targets, listing_reference

BASE = ["Change", "Vol20", "VolumeRatio", "Disp20", "Disp60", "RSI14", "LogMarcap"]
PRICE = ["Up", "Ret5", "Mom12_1", "High52", "Issuance"]
OHLC = ["Range", "CloseLoc", "Gap", "LimitHit", "Turnover", "Amihud"]
FS_PRICE = {"BM": "BookEquity", "EP": "NetIncome", "SP": "Sales", "CFP": "OperatingCashFlow"}
FS_SLOW = ["ROE", "ROA", "OperProf", "GrossProf", "AssetGrowth", "SalesGrowth", "Leverage", "Accruals", "CashToAssets"]
FLOW = {"Foreign": "ForeignNetBuy", "Inst": "InstNetBuy", "Indiv": "IndivNetBuy"}
MARKET = ["IndexReturn", "IndexVol20", "IndexMom20", "IndexMom60", "AvgVar", "AvgCorr", "Breadth"]
DAILY_COLUMNS = ["Date", "Code", "Name", "Market", "Volume", "Marcap", "Close", "CloseRaw", "KOSPI_lvl", "Common", *BASE]
RAW_COLUMNS = ["Date", "Code", "Market", "Open", "High", "Low", "Close", "Amount"]

# KOSPI daily price limits (build_panel.PRICE_LIMITS). LimitHit uses a 0.5%p tolerance
# for tick rounding below the limit price; it is an approximation, not an exchange flag.
KOSPI_LIMITS = (("1995-04-01", 0.06), ("1996-11-25", 0.08), ("1998-03-02", 0.12),
                ("1998-12-07", 0.15), ("2015-06-15", 0.30))
LIMIT_TOLERANCE = 0.005

FEATURE_NOTES = {
    "Change": "daily return t-1 -> t",
    "Vol20": "20-day std of daily log returns; cap-weighted Vol20^2 ~ average variance (Goyal & Santa-Clara 2003)",
    "VolumeRatio": "volume / 20-day mean volume",
    "Disp20": "price / 20-day mean price",
    "Disp60": "price / 60-day mean price",
    "RSI14": "14-day RSI",
    "LogMarcap": "log market cap",
    "Up": "1 if Change > 0 else 0; cap-weighted mean = market breadth",
    "Ret5": "5-day return; short-term reversal",
    "Mom12_1": "return from t-252 to t-21 sessions (Jegadeesh & Titman 1993)",
    "High52": "price / 252-session high (George & Hwang 2004)",
    "Issuance": "log ME growth - log price growth over 252 sessions = net share issuance (Pontiff & Woodgate 2008; aggregate: Baker & Wurgler 2000)",
    "Range": "(high - low) / close, intraday range volatility",
    "CloseLoc": "(close - low) / (high - low), where the close sits in the day's range",
    "Gap": "open / previous close - 1, overnight move",
    "LimitHit": "+1 at the upper price limit, -1 at the lower, else 0 (approximate)",
    "Turnover": "trading value / market cap; cap-weighted mean = market turnover",
    "Amihud": "|Change| / trading value in KRW 100m; aggregate illiquidity (Amihud 2002)",
    "BM": "book equity / issuer cap (aggregate B/M: Kelly & Pruitt 2013)",
    "EP": "net income / issuer cap (earnings yield, Welch & Goyal 2008 predictor set)",
    "SP": "sales / issuer cap, nonfinancials",
    "CFP": "operating cash flow / issuer cap",
    "Accruals": "statement accruals; aggregate accruals predict market returns (Hirshleifer, Hou & Teoh 2009)",
    "ForeignNetBuy": "foreign net buy value / cap",
    "InstNetBuy": "institutional net buy value / cap",
    "IndivNetBuy": "individual net buy value / cap",
}


def kospi_limit(dates):
    limit = np.full(len(dates), np.nan)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    for start, value in KOSPI_LIMITS:
        limit[dates >= np.datetime64(start)] = value
    return limit


def segments(frame):
    """Price-link segments within a code: a missing Change starts a new segment."""
    new = frame.Change.isna() | frame.Code.ne(frame.Code.shift())
    return new.cumsum()


def price_features(rows):
    """Features from adjusted prices; rows sorted by Code, Date (all KOSPI share classes)."""
    seg = segments(rows)
    grouped = lambda s: s.groupby(seg, sort=False)
    close, cap = rows.Close.where(rows.Close > 0), rows.Marcap.where(rows.Marcap > 0)
    lag = lambda s, n: grouped(s).shift(n)
    out = pd.DataFrame(index=rows.index)
    out["Up"] = (rows.Change > 0).astype(float).where(rows.Change.notna())
    out["Ret5"] = close / lag(close, 5) - 1
    out["Mom12_1"] = lag(close, 21) / lag(close, 252) - 1
    high = grouped(close).rolling(252, min_periods=252).max().droplevel(0).reindex(rows.index)
    out["High52"] = close / high
    out["Issuance"] = np.log(cap / lag(cap, 252)) - np.log(close / lag(close, 252))
    return out


def ohlc_features(rows, raw):
    """Same-day ratios from unadjusted open/high/low/close and trading value."""
    keys = rows[["Date", "Code"]].copy()
    if raw is None:
        return pd.DataFrame(np.nan, index=rows.index, columns=OHLC)
    raw = raw.copy()
    raw["Code"] = raw.Code.astype(str).str.zfill(6)
    joined = keys.merge(raw.drop(columns=["Market"], errors="ignore"), on=["Date", "Code"], how="left")
    joined.index = rows.index
    o, h, l, c, amount = (joined[col].where(joined[col] > 0) for col in ["Open", "High", "Low", "Close", "Amount"])
    span = (h - l).where(h >= l)
    out = pd.DataFrame(index=rows.index)
    out["Range"] = span / c
    out["CloseLoc"] = ((c - l) / span).where(span > 0)
    # Previous close in today's price units (after any base-price change) = close / (1 + Change).
    out["Gap"] = o / (c / (1 + rows.Change)) - 1
    limit = kospi_limit(rows.Date)
    out["LimitHit"] = np.select([rows.Change >= limit - LIMIT_TOLERANCE, rows.Change <= -limit + LIMIT_TOLERANCE],
                                [1.0, -1.0], 0.0)
    out.loc[rows.Change.isna(), "LimitHit"] = np.nan
    out["Turnover"] = amount / rows.Marcap.where(rows.Marcap > 0)
    out["Amihud"] = rows.Change.abs() / (amount / 1e8)
    return out


def attach_statements(firms, table, lag):
    """Join lagged annual statements by code; scale by issuer market cap that day."""
    table = table.copy()
    table["Code"] = table.Code.astype(str).str.zfill(6)
    table["available"] = pd.PeriodIndex(table.FiscalPeriodEnd, freq="M") + lag
    table = (table.sort_values(["Code", "available", "FiscalPeriodEnd"])
             .drop_duplicates(["Code", "available"], keep="last"))
    table["key"] = table.available.map(lambda p: p.ordinal).astype("int64")

    month = firms.Date.dt.to_period("M")
    last_session = firms.groupby(month).Date.transform("max")
    # Inside the availability month a statement counts only from that month's last session.
    key = month.map(lambda p: p.ordinal).astype("int64") - (firms.Date < last_session).astype("int64")
    left = firms[["Code"]].assign(key=key.to_numpy(), row=np.arange(len(firms)))
    cols = ["Code", "key", "FiscalPeriodEnd", "Financial", *FS_PRICE.values(), *FS_SLOW]
    merged = pd.merge_asof(left.sort_values("key"), table[cols].sort_values("key"),
                           on="key", by="Code", direction="backward", tolerance=11)
    merged = merged.sort_values("row").reset_index(drop=True)

    me_million = (firms.FirmMarcap / 1e6).where(firms.FirmMarcap > 0).to_numpy()
    out = {}
    for name, column in FS_PRICE.items():
        numerator = merged[column]
        if name == "BM":
            numerator = numerator.where(numerator > 0)
        if name == "SP":
            numerator = numerator.where(~merged.Financial.eq(True))
        out[name] = numerator.to_numpy() / me_million
    for name in FS_SLOW:
        out[name] = merged[name].to_numpy()
    out["FinancialFlag"] = merged.Financial.eq(True).astype("int8").to_numpy()
    out["FiscalPeriodEnd"] = merged.FiscalPeriodEnd.to_numpy()
    return pd.DataFrame(out, index=firms.index).replace([np.inf, -np.inf], np.nan)


def attach_flows(firms, flows):
    if flows is None:
        for name in FLOW.values():
            firms[name] = np.nan
        return firms
    flows = flows.copy()
    flows["Code"] = flows.Code.astype(str).str.zfill(6)
    flows["Date"] = pd.to_datetime(flows.Date)
    net = [f"{p}NetValue" for p in FLOW if f"{p}NetValue" in flows]
    firms = firms.merge(flows[["Date", "Code", *net]], on=["Date", "Code"], how="left")
    cap = firms.Marcap.where(firms.Marcap > 0)
    for prefix, name in FLOW.items():
        column = f"{prefix}NetValue"
        firms[name] = firms[column] / cap if column in firms else np.nan
    return firms.drop(columns=net)


def replica_return(firms):
    """Cap-weighted return of valid firms with weights fixed at the previous session's close."""
    rows = firms[["Date", "Code", "CapWeight", "Change"]].sort_values(["Code", "Date"]).reset_index(drop=True)
    calendar = pd.DatetimeIndex(rows.Date.unique()).sort_values()
    pos = calendar.get_indexer(rows.Date)
    same = rows.Code.eq(rows.Code.shift()).to_numpy() & (pos - np.r_[-9, pos[:-1]] == 1)
    prev_w = np.where(same, rows.CapWeight.shift().to_numpy(), np.nan)
    ok = np.isfinite(prev_w) & rows.Change.notna().to_numpy()
    num = pd.Series(np.where(ok, prev_w * rows.Change.to_numpy(), 0.0)).groupby(rows.Date).sum()
    den = pd.Series(np.where(ok, prev_w, 0.0)).groupby(rows.Date).sum()
    return (num / den.where(den > 0)).rename("ReplicaReturn")


def replication_audit(market):
    """Daily index replication check by decade: correlation and tracking error (%p)."""
    data = market.dropna(subset=["IndexReturn", "ReplicaReturn"])
    decade = (data.Date.dt.year // 10 * 10).astype(str) + "s"
    gap = (data.ReplicaReturn - data.IndexReturn) * 100
    return pd.DataFrame({
        "Sessions": data.groupby(decade).size(),
        "Corr": data.groupby(decade).apply(lambda g: g.ReplicaReturn.corr(g.IndexReturn)),
        "TrackingStdPctPt": gap.groupby(decade).std(),
        "MeanGapPctPt": gap.groupby(decade).mean(),
    }).rename_axis("Decade").reset_index()


def market_table(daily, firms):
    """Index level, past index features, cross-sectional market variables and the target."""
    level = daily.groupby("Date").KOSPI_lvl.first().sort_index()
    if level.isna().any():
        raise ValueError("KOSPI level missing on a stock session.")
    ret = level.pct_change()
    out = pd.DataFrame({
        "KOSPI": level,
        "IndexReturn": ret,
        "IndexVol20": ret.rolling(20).std(),
        "IndexMom20": level.pct_change(20),
        "IndexMom60": level.pct_change(60),
        "NextDayIndexReturn": level.shift(-1) / level - 1,   # target: t -> t+1
    })
    valid = firms.loc[firms.Valid]
    by_day = valid.groupby("Date")
    # Goyal & Santa-Clara (2003): equal-weighted average stock variance.
    out["AvgVar"] = by_day.Vol20.apply(lambda s: np.nanmean(s ** 2))
    # Pollet & Wilson (2010): average correlation ~ index variance / (cap-weighted mean stock vol)^2.
    weighted_vol = (valid.CapWeight * valid.Vol20).groupby(valid.Date).sum(min_count=1)
    out["AvgCorr"] = (out.IndexVol20 ** 2) / (weighted_vol ** 2)
    out["Breadth"] = by_day.Up.mean()   # equal-weighted share of advancing stocks
    out = out.join(replica_return(firms))   # audit only, never a model input
    counts = firms.groupby("Date").agg(NValid=("Valid", "sum"), NFinValid=("FinValid", "sum"),
                                       NFlowValid=("FlowValid", "sum"))
    return out.join(counts).reset_index()


def coverage(firms, market, features, min_start):
    """First year in which each feature covers >= 50% of valid rows (market: of sessions)."""
    records = []
    valid = firms.loc[firms.Valid]
    year = valid.Date.dt.year
    for name in features:
        share = valid[name].notna().groupby(year).mean()
        start = share.index[share >= 0.5].min() if (share >= 0.5).any() else None
        records.append({"Feature": name, "Table": "firm", "FirstYear50pct": start,
                        "Share": float(valid[name].notna().mean())})
    myear = market.Date.dt.year
    for name in MARKET:
        share = market[name].notna().groupby(myear).mean()
        start = share.index[share >= 0.5].min() if (share >= 0.5).any() else None
        records.append({"Feature": name, "Table": "market", "FirstYear50pct": start,
                        "Share": float(market[name].notna().mean())})
    table = pd.DataFrame(records)
    table["Kept"] = table.FirstYear50pct.notna() & (table.FirstYear50pct <= min_start)
    table["Note"] = table.Feature.map(FEATURE_NOTES).fillna("")
    return table


def build(daily, raw, fundamentals, flows, reference, lag=6, exit_window=10, min_start=2005):
    daily = daily.copy()
    daily["Code"] = daily.Code.astype(str).str.zfill(6)
    kospi = daily.loc[daily.Market.eq(MARKET_NAME)].sort_values(["Code", "Date"]).reset_index(drop=True)
    kospi["FirmMarcap"] = kospi.Marcap.groupby([kospi.Date, kospi.Code.str[:5]]).transform("sum")

    targets = daily_targets(kospi, reference, exit_window)
    kospi["NextDayReturn"] = targets.NextDayReturn.to_numpy()
    kospi["NextDayStatus"] = targets.NextDayStatus.to_numpy()
    kospi = pd.concat([kospi, price_features(kospi), ohlc_features(kospi, raw)], axis=1)

    firms = kospi.loc[kospi.Common.eq(True)].reset_index(drop=True)
    firms = pd.concat([firms, attach_statements(firms, fundamentals, lag)], axis=1)
    firms = attach_flows(firms, flows)

    firms["Valid"] = (firms.Volume > 0) & np.isfinite(firms[BASE].astype(float)).all(axis=1) & (firms.Marcap > 0)
    firms["FinValid"] = np.isfinite(firms[["BM", "EP", "SP"]].astype(float)).all(axis=1)
    firms["FlowValid"] = np.isfinite(firms.ForeignNetBuy.astype(float))
    valid_cap = firms.Marcap.where(firms.Valid)
    firms["CapWeight"] = valid_cap / valid_cap.groupby(firms.Date).transform("sum")
    firms = firms.replace([np.inf, -np.inf], np.nan)

    market = market_table(daily, firms)
    features = [*BASE, *PRICE, *OHLC, *FS_PRICE, *FS_SLOW, *FLOW.values()]
    cover = coverage(firms, market, features, min_start)
    dropped = set(cover.loc[~cover.Kept, "Feature"])

    keep = ["Date", "Code", "Name", "Marcap", "FirmMarcap", "CapWeight",
            *(f for f in features if f not in dropped), "FinancialFlag", "FiscalPeriodEnd",
            "Valid", "FinValid", "FlowValid", "NextDayReturn", "NextDayStatus"]
    market = market.drop(columns=[c for c in MARKET if c in dropped])
    return firms[keep].sort_values(["Date", "Code"]).reset_index(drop=True), market, cover


RAW_END = "2026-10-31"
MARKET_NAME = "KOSPI"   # --market; the index level column (KOSPI_lvl) then holds that market's index


def read_raw(folder):
    frames = []
    for path in sorted(folder.glob("marcap-*.parquet")):
        frame = pd.read_parquet(path, columns=RAW_COLUMNS)
        frames.append(frame.loc[frame.Market.replace("KOSDAQ GLOBAL", "KOSDAQ").eq(MARKET_NAME)])
    raw = pd.concat(frames, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw.Date)
    return raw.loc[raw.Date.le(RAW_END)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="folder with marcap-YYYY.parquet")
    parser.add_argument("--fundamentals", type=Path, default=Path(__file__).with_name("krx_fundamentals_annual.csv"))
    parser.add_argument("--no-fundamentals", action="store_true",
                        help="Do not attach statement features (BM ... CashToAssets stay missing and are dropped).")
    parser.add_argument("--flows", type=Path, help="investor_flows_kospi.csv (optional)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--since", help="Read only rows on or after this date (e.g. 2013-01-01) to save memory.")
    parser.add_argument("--fs-lag", type=int, default=6)
    parser.add_argument("--exit-window", type=int, default=10)
    parser.add_argument("--min-start", type=int, default=2005,
                        help="Drop a feature whose coverage of valid rows first reaches 50%% after this year.")
    parser.add_argument("--delisting-reference", type=Path, default=Path(__file__).with_name("delisting_reference.csv"))
    parser.add_argument("--market", default="KOSPI", choices=("KOSPI", "KOSDAQ"))
    parser.add_argument("--raw-end", default="2026-10-31")
    args = parser.parse_args()
    global MARKET_NAME, RAW_END
    MARKET_NAME, RAW_END = args.market, args.raw_end

    daily = pd.read_parquet(args.daily, columns=DAILY_COLUMNS)
    daily = daily.loc[daily.Market.eq(MARKET_NAME)]
    if args.since:
        daily = daily.loc[pd.to_datetime(daily.Date) >= pd.Timestamp(args.since)]
    raw = read_raw(args.raw) if args.raw else None
    if raw is not None and args.since:
        raw = raw.loc[raw.Date >= pd.Timestamp(args.since)]
    if args.no_fundamentals:            # no statement file needed: an empty table with the expected columns
        fundamentals = pd.DataFrame(columns=["Code", "FiscalPeriodEnd", "Financial", *FS_PRICE.values(), *FS_SLOW])
    else:
        fundamentals = pd.read_csv(args.fundamentals, dtype={"Code": str}, encoding="utf-8-sig")
    flows = pd.read_csv(args.flows, dtype={"Code": str}) if args.flows else None
    firms, market, cover = build(daily, raw, fundamentals, flows, listing_reference(args.delisting_reference),
                                 args.fs_lag, args.exit_window, args.min_start)

    args.output.mkdir(parents=True, exist_ok=True)
    firms.to_parquet(args.output / "daily_firms.parquet", index=False, compression="zstd")
    market.to_parquet(args.output / "daily_market.parquet", index=False, compression="zstd")
    cover.to_csv(args.output / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    replication = replication_audit(market)
    replication.to_csv(args.output / "audit_daily_index_replication.csv", index=False)

    print(f"Sessions {market.Date.nunique():,} ({market.Date.min().date()} ~ {market.Date.max().date()})")
    print(f"Firm rows {len(firms):,}; valid {int(firms.Valid.sum()):,}")
    print("Daily index replication (ReplicaReturn vs IndexReturn):\n" + replication.to_string(index=False))
    print(f"Valid firms per session: median {market.NValid.median():.0f}, min {market.NValid.min():.0f}")
    print(cover.drop(columns="Note").to_string(index=False))
    kept = [f for f in cover.loc[cover.Kept & cover.Table.eq("firm"), "Feature"]]
    print(firms.loc[firms.Valid, kept].describe(percentiles=[0.01, 0.5, 0.99]).T.to_string())


if __name__ == "__main__":
    main()