"""Build the KRX panel from annual FinanceData marcap Parquets and a daily index CSV."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ["Date", "Code", "Name", "Market", "Close", "Changes", "ChangesRatio", "Volume", "Marcap", "Stocks", "Dept"]
FEATURES = ["Change", "Vol20", "VolumeRatio", "Disp20", "Disp60", "RSI14", "LogMarcap"]
MOMENTUM_FEATURES = ["Momentum1M", "Momentum12_1M"]

# Keep the source index CSV and log the evidence for the calendar alignment.
ALIGNMENT_DATES = ("2000-05-01",)

# Fixed-percentage price limits by date, used for diagnostics.
PRICE_LIMITS = {
    "KOSPI": (("1995-04-01", 0.06), ("1996-11-25", 0.08), ("1998-03-02", 0.12),
              ("1998-12-07", 0.15), ("2015-06-15", 0.30)),
    "KOSDAQ": (("1996-11-01", 0.08), ("1998-05-25", 0.12), ("2005-03-28", 0.15),
               ("2015-06-15", 0.30)),
}
LIMIT_TOLERANCE = 0.001

# The dated security classification refines the name and code proxy.
NON_COMMON_NAME = r"스팩|기업인수목적|맥쿼리인프라|KB발해인프라|선박투자회사|[0-9]+신$"
COMMON_SECURITY_GROUPS = {"주권", "외국주권"}

MONTHLY_COLUMNS = [
    "Date", "Code", "Market", "Close", "Volume", "KOSPI_lvl", *FEATURES,
    *MOMENTUM_FEATURES, "NextMonthReturn", "NextMonthLastTradeReturn", "TargetStatus", "Common",
]
AUDIT_COLUMNS = [
    "Date", "Code", "Name", "Market", "PrevClose", "BasePrice", "CloseRaw", "DailyReturn", "Reason", "ShareUnitAdjusted",
    "CalendarGap", "UnmatchedCalendarBridge", "InvalidPrice", "RatioMismatch", "LimitBreach", "BasePriceAdjusted",
]


def price_limit(dates, markets):
    """Daily price limit in force for each row."""
    limit = np.full(len(dates), np.nan)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    markets = np.asarray(markets, dtype=object)
    for market, schedule in PRICE_LIMITS.items():
        chosen = markets == market
        for start, value in schedule:
            limit[chosen & (dates >= np.datetime64(start))] = value
    return limit


def listing_reference(path):
    data = pd.read_csv(path, dtype={"Symbol": str})
    for col in ("ListingDate", "DelistingDate"):
        data[col] = pd.to_datetime(data[col], errors="coerce")
    data = data.loc[data.Market.isin(["KOSPI", "KOSDAQ"]) & data.DelistingDate.between("1995-05-02", "2026-10-30")].copy()
    data["Code"] = data.Symbol.str.zfill(6)
    if data.Code.duplicated().any():
        raise ValueError("Multiple delisting records for a security code. Check listing spells.")
    return data.set_index("Code")[["ListingDate", "DelistingDate", "Market", "SecuGroup", "Kind"]]


def is_common(rows, official):
    names = rows.Name.fillna("").astype(str)
    reit = names.str.replace("메리츠", "", regex=False).str.contains("리츠", regex=False)
    non_equity = reit | names.str.contains(NON_COMMON_NAME, regex=True)
    non_equity |= rows.Dept.fillna("").str.contains("SPAC", case=False, regex=False)
    proxy = rows.Code.str.endswith("0")
    same_spell = (rows.Market.eq(official.Market.to_numpy())
                  & rows.Date.ge(official.ListingDate.to_numpy())
                  & rows.Date.le(official.DelistingDate.to_numpy()))
    known_class = same_spell & official.Kind.notna().to_numpy() & official.SecuGroup.notna().to_numpy()
    declared_common = official.Kind.eq("보통주") & official.SecuGroup.isin(COMMON_SECURITY_GROUPS)
    proxy = proxy.where(~known_class, declared_common.to_numpy())
    return (proxy & ~non_equity).to_numpy(dtype=bool)


def make_features(rows, index, month_ends, reference, bridges):
    rows = rows.sort_values(["Code", "Date"]).reset_index(drop=True)
    close = rows.Close.to_numpy(dtype=float)
    ref = close - rows.Changes.to_numpy(dtype=float)
    priced = (close > 0) & (ref > 0)
    gross = np.divide(close, ref, out=np.ones(len(rows)), where=priced)
    matches = np.isfinite(gross) & (np.abs((gross - 1) * 100 - rows.ChangesRatio.to_numpy(dtype=float)) <= 0.021)
    valid = priced & matches
    days = index.index.get_indexer(rows.Date)
    if (days < 0).any():
        raise ValueError("Stock data contains dates absent from KOSPI index.")
    same = rows.Code.eq(rows.Code.shift()).to_numpy()
    step = np.r_[False, np.diff(days) == 1]
    prev_close = np.r_[np.nan, close[:-1]]
    bridge = same & step & rows.Date.isin(bridges).to_numpy()
    unmatched_bridge = bridge & (ref != prev_close)
    use_link = same & step & valid & (prev_close > 0) & ~unmatched_bridge
    segment = (~use_link).cumsum()
    log_return = np.log(np.where(use_link, gross, 1.0))
    cumulative = pd.Series(log_return).groupby(segment, sort=False).cumsum()
    base = rows.Close.groupby(segment, sort=False).transform("first")
    adjusted = (base * np.exp(cumulative)).where(rows.Close.gt(0))
    log_return = pd.Series(log_return).where(use_link)
    delta = adjusted.groupby(segment, sort=False).diff()
    shares = rows.Stocks.to_numpy(dtype=float)
    prev_shares = np.r_[np.nan, shares[:-1]]
    share_ratio = np.divide(shares, prev_shares, out=np.ones(len(rows)), where=prev_shares > 0)
    base_ratio = np.divide(ref, prev_close, out=np.ones(len(rows)), where=prev_close > 0)
    # A matching change in listed shares provides evidence of a unit change.
    unit_event = (
        use_link & (shares > 0) & (prev_shares > 0)
        & (np.abs(base_ratio - 1) > 0.02) & (np.abs(share_ratio - 1) > 0.02)
        & np.isclose(base_ratio * share_ratio, 1.0, rtol=0.01, atol=0)
    )
    unit_scale = pd.Series(np.log(np.where(unit_event, share_ratio, 1.0))).groupby(segment, sort=False).cumsum()
    volume = (rows.Volume / np.exp(unit_scale)).where(rows.Close.gt(0))

    def rolling(values, window, statistic="mean"):
        roll = values.groupby(segment, sort=False).rolling(window)
        return getattr(roll, statistic)().droplevel(0).reindex(rows.index)

    gain = rolling(delta.clip(lower=0), 14)
    loss = rolling(-delta.clip(upper=0), 14)
    features = {
        "Change": np.expm1(log_return), "Vol20": rolling(log_return, 20, "std"),
        "VolumeRatio": volume / rolling(volume, 20).where(lambda s: s > 0),
        "Disp20": adjusted / rolling(adjusted, 20), "Disp60": adjusted / rolling(adjusted, 60),
        "RSI14": 100 * gain / (gain + loss).where(lambda s: s > 0),
        "LogMarcap": np.log(rows.Marcap.where(rows.Marcap.gt(0))),
    }

    month_end_days = np.sort(index.index.get_indexer(pd.DatetimeIndex(month_ends)))
    code_id = pd.factorize(rows.Code)[0].astype(np.int64)
    span = len(index) + 1
    key = code_id * span + days
    origin = np.flatnonzero(np.isin(days, month_end_days))
    following = np.searchsorted(month_end_days, days[origin]) + 1
    has_next = following < len(month_end_days)
    next_day = np.where(has_next, month_end_days[np.minimum(following, len(month_end_days) - 1)], -1)
    last = np.searchsorted(key, code_id[origin] * span + next_day, side="right") - 1
    last = np.maximum(last, origin)
    segment_end = np.searchsorted(segment, segment[origin], side="right") - 1
    linked = last <= segment_end
    reached = has_next & (days[last] == next_day)
    price = adjusted.to_numpy()
    positive_pair = (price[origin] > 0) & (price[last] > 0)
    month_ratio = np.divide(price[last], price[origin], out=np.full(len(origin), np.nan), where=positive_pair) - 1
    valid_month = reached & linked & np.isfinite(month_ratio)

    origin_rows = rows.iloc[origin]
    month_ref = reference.reindex(origin_rows.Code)
    official_exit = pd.DatetimeIndex(month_ref.DelistingDate)
    event_day = official_exit.to_numpy(dtype="datetime64[ns]")
    start = origin_rows.Date.to_numpy(dtype="datetime64[ns]")
    deadline = index.index.take(np.maximum(next_day, 0)).to_numpy(dtype="datetime64[ns]")
    same_spell = (origin_rows.Market.to_numpy() == month_ref.Market.to_numpy())
    same_spell &= start >= month_ref.ListingDate.to_numpy(dtype="datetime64[ns]")
    exit_in_month = has_next & same_spell & (event_day > start) & (event_day <= deadline)

    traded = rows.Volume.to_numpy() > 0
    traded_rows = np.flatnonzero(traded & (price > 0))
    event_pos = index.index.searchsorted(official_exit, side="right") - 1
    event_keys = code_id[origin] * span + np.minimum(event_pos, next_day)
    position = np.searchsorted(key[traded_rows], event_keys, side="right") - 1
    terminal = origin.copy()
    found = position >= 0
    terminal[found] = traded_rows[position[found]]
    valid_trade = (
        exit_in_month & found & (terminal > origin) & (terminal <= segment_end)
        & (code_id[terminal] == code_id[origin]) & (price[origin] > 0) & traded[origin]
    )
    last_trade_ratio = np.divide(price[terminal], price[origin], out=np.full(len(origin), np.nan), where=valid_trade) - 1
    valid_trade &= np.isfinite(last_trade_ratio)
    last_trade_ratio[~valid_trade] = np.nan
    valid_month &= ~exit_in_month
    status = np.select(
        [~has_next, valid_trade, exit_in_month, reached & linked & ~positive_pair,
         valid_month, ~linked, last == origin],
        ["final_month", "confirmed_exit_last_trade", "confirmed_exit_unpriced", "invalid_price",
         "month_end", "link_break", "no_observation_after"],
        default="source_gap",
    ).astype(object)
    returns = np.full(len(rows), np.nan)
    returns[origin] = np.where(valid_month, month_ratio, np.nan)
    last_trade = np.full(len(rows), np.nan)
    last_trade[origin] = last_trade_ratio
    target_status = np.full(len(rows), None, dtype=object)
    target_status[origin] = status

    def past_month(lag):
        end_index = following - 1 - lag
        day = month_end_days[np.maximum(end_index, 0)]
        search_key = code_id[origin] * span + day
        match = np.minimum(np.searchsorted(key, search_key), len(key) - 1)
        valid_past = (end_index >= 0) & (key[match] == search_key) & (segment[match] == segment[origin])
        return np.where(valid_past, price[match], np.nan)

    previous = past_month(1)
    momentum = {}
    for feature, numerator, denominator in (
        ("Momentum1M", price[origin], previous),
        ("Momentum12_1M", previous, past_month(12)),
    ):
        values = np.full(len(rows), np.nan)
        values[origin] = numerator / denominator - 1
        momentum[feature] = values

    official = reference.reindex(rows.Code)

    output = rows[["Date", "Code", "Name", "Market", "Volume", "Marcap"]].assign(
        CloseRaw=rows.Close, Close=adjusted, KOSPI_lvl=rows.Date.map(index.kospi),
        KOSDAQ_lvl=rows.Date.map(index.kosdaq), NextMonthReturn=returns,
        NextMonthLastTradeReturn=last_trade, TargetStatus=target_status, Common=is_common(rows, official),
        **features, **momentum,
    )

    # Rows a reviewer should compare with corporate-action disclosures.
    daily_return = np.where(use_link, gross - 1, np.nan)
    limit = price_limit(rows.Date.to_numpy(), rows.Market.to_numpy())
    continuing = same & step
    flags = {
        "CalendarGap": same & ~step,
        "UnmatchedCalendarBridge": unmatched_bridge,
        "InvalidPrice": continuing & ~(priced & (prev_close > 0)),
        "RatioMismatch": continuing & priced & ~matches,
        "LimitBreach": use_link & (np.abs(daily_return) > limit + LIMIT_TOLERANCE),
        "BasePriceAdjusted": use_link & (ref != prev_close),
    }
    reason = np.select(
        list(flags.values()),
        ["calendar_gap", "unmatched_calendar_bridge", "invalid_price", "ratio_mismatch",
         "limit_breach", "base_price_adjusted"],
        default="",
    )
    flagged = reason != ""
    audit = pd.DataFrame({
        "Date": rows.Date[flagged], "Code": rows.Code[flagged], "Name": rows.Name[flagged],
        "Market": rows.Market[flagged], "PrevClose": prev_close[flagged], "BasePrice": ref[flagged],
        "CloseRaw": close[flagged], "DailyReturn": daily_return[flagged], "Reason": reason[flagged],
        "ShareUnitAdjusted": unit_event[flagged],
        **{name: flag[flagged] for name, flag in flags.items()},
    })[AUDIT_COLUMNS]
    return output, audit


def trading_calendar(index, rows, dates):
    """Align to the stock calendar while recording the dated evidence."""
    index = index.loc[index.index >= dates[0]]
    missing = dates.difference(index.index)
    if len(missing):
        raise ValueError(f"Stock dates without index rows: {list(missing)}")
    extra = index.index.difference(dates)
    unknown = extra.difference(pd.DatetimeIndex(ALIGNMENT_DATES))
    if len(unknown):
        raise ValueError(f"Unresolved index-only dates: {list(unknown)}")
    checks = []
    for date in extra:
        before, after = dates[dates < date].max(), dates[dates > date].min()
        cols = ["Code", "Close", "Changes", "Volume"]
        earlier = rows.loc[rows.Date.eq(before), cols].set_index("Code")
        later = rows.loc[rows.Date.eq(after), cols].set_index("Code")
        comparable = earlier.join(later, how="inner", lsuffix="_before", rsuffix="_after")
        base = comparable.Close_after - comparable.Changes_after
        valid = (base.gt(0) & comparable.Close_before.gt(0)
                 & comparable.Volume_before.gt(0) & comparable.Volume_after.gt(0))
        count = int(valid.sum())
        matches = int((valid & base.eq(comparable.Close_before)).sum())
        if not count or matches / count < 0.8:
            raise ValueError(f"Insufficient reference-price evidence at {date.date()}.")
        checks.append({"IndexDate": date, "PreviousStockDate": before, "NextStockDate": after,
                       "KOSPI": index.loc[date, "kospi"], "Compared": count, "Matched": matches})
        print(f"Calendar alignment {date.date()}: {matches:,}/{count:,} reference prices match.", flush=True)
    return index.loc[index.index.isin(dates)], pd.DataFrame(
        checks, columns=["IndexDate", "PreviousStockDate", "NextStockDate", "KOSPI", "Compared", "Matched"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily", action="store_true", help="Also write the full daily Parquet.")
    parser.add_argument(
        "--delisting-reference", type=Path,
        default=Path(__file__).with_name("delisting_reference.csv"),
    )
    parser.add_argument("--first-year", type=int, default=1995, help="first marcap year to read (KOSDAQ build: 2013)")
    parser.add_argument("--end", default="2026-10-31", help="last date (KOSDAQ validation build: 2022-03-31)")
    args = parser.parse_args()
    index = pd.read_csv(args.index, parse_dates=["date"]).set_index("date").sort_index()
    if args.first_year > 1995:
        index = index.loc[f"{args.first_year}-01-01":]
    if index.index.has_duplicates or index.kospi.isna().any():
        raise ValueError("Invalid KOSPI index.")
    # End at the last complete month in the November 2026 source snapshot.
    limit = pd.Timestamp(args.end)
    index = index.loc[:limit]
    files = [args.raw / f"marcap-{year}.parquet" for year in range(args.first_year, limit.year + 1)]
    frames = [pd.read_parquet(path, columns=COLUMNS) for path in files]
    rows = pd.concat(frames, ignore_index=True)
    del frames
    rows = rows.loc[rows.Market.ne("KONEX") & rows.Date.le(limit)].copy()
    rows["Code"] = rows.Code.astype(str).str.zfill(6)
    rows["Market"] = rows.Market.replace("KOSDAQ GLOBAL", "KOSDAQ")
    if rows.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate date/code after standardization.")
    dates = pd.DatetimeIndex(rows.Date.unique()).sort_values()
    index, calendar_checks = trading_calendar(index, rows, dates)
    reference = listing_reference(args.delisting_reference)
    month_ends = index.index.to_series().groupby(index.index.to_period("M")).max()
    args.output.mkdir(parents=True, exist_ok=True)
    calendar_checks.to_csv(args.output / "audit_calendar.csv", index=False)
    full = args.output / "krx_1995_2026_daily.parquet"
    writer = None
    snapshots, audits = [], []
    try:
        buckets = pd.util.hash_pandas_object(rows.Code, index=False) % 32
        for bucket, frame in rows.groupby(buckets, sort=True):
            output, audit = make_features(frame, index, month_ends, reference, calendar_checks.NextStockDate)
            if args.daily:
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(full, table.schema, compression="zstd")
                writer.write_table(table.cast(writer.schema))
            snapshots.append(output.loc[output.Date.isin(month_ends), MONTHLY_COLUMNS])
            audits.append(audit)
            print(f"Processed partition {bucket + 1}/32", flush=True)
    finally:
        if writer is not None:
            writer.close()

    monthly = pd.concat(snapshots, ignore_index=True)
    # The existing loader computes daily index volatility, so retain every
    # index date. Synthetic index rows occur only on non-month-end dates.
    other_days = index.loc[~index.index.isin(month_ends)]
    carrier = pd.DataFrame({"Date": other_days.index, "Code": "__INDEX__", "KOSPI_lvl": other_days.kospi.to_numpy()})
    compact = pd.concat([monthly, carrier], ignore_index=True).sort_values(["Date", "Code"])
    compact["Common"] = compact.Common.eq(True)
    compact.to_parquet(args.output / "krx_1995_2026_model.parquet", index=False, compression="zstd")
    audit = pd.concat(audits, ignore_index=True).sort_values(["Date", "Code"])
    audit.to_csv(args.output / "audit_daily_links.csv", index=False, encoding="utf-8-sig")

    print(f"Daily rows {len(rows):,}; codes {rows.Code.nunique():,}; sessions {len(dates):,}")
    print(rows.groupby("Market").agg(rows=("Code","size"), codes=("Code","nunique"), start=("Date","min"), end=("Date","max")))
    print(f"Months {len(month_ends)}; firm-month rows {len(monthly):,}; valid targets {monthly.NextMonthReturn.notna().sum():,}")
    print(f"Common-share firm-months {int(monthly.Common.sum()):,}")
    summary = monthly.groupby("TargetStatus").NextMonthReturn.agg(
        rows="size", labelled="count",
        above_100pct=lambda s: int((s > 1).sum()), below_minus_50pct=lambda s: int((s < -0.5).sum()),
    )
    print(summary.to_string())
    print(f"Observed last-trade targets: {monthly.NextMonthLastTradeReturn.notna().sum():,}")
    print(audit.Reason.value_counts().to_string())
    print(f"Model rows {len(compact):,}; market dates {compact.Date.nunique():,}")


if __name__ == "__main__":
    main()
