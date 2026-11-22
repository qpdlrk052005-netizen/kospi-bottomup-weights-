"""Build next-trading-day return targets from the daily panel written by build_panel.py --daily.

The rules follow the monthly NextMonthReturn rules in krx_preprocessing_review.md,
shortened to a one-session horizon:

* NextDayReturn at session t is the adjusted price return from t to the next
  market session t+1. It is kept only when the stock has a row at exactly t+1
  and build_panel.py kept the price link on that day (its Change is not NaN).
  A gap in the stock's rows, an invalid price, a ratio mismatch or an
  unmatched calendar bridge leaves the target missing.
* A next-session row with zero volume leaves the target missing (status
  no_trade). This rule has no monthly counterpart: over one day a halted stock
  shows a 0% return that is not an observed trade.
* When the official delisting date falls within the next --exit-window sessions
  (the delisting-liquidation trading period is usually 7 sessions), the main
  target is left missing, as the monthly target is left missing for a month
  with a confirmed exit. The observed return goes to NextDayLastTradeReturn
  when both days had positive volume and the link held.

Usage:
    python build_daily_targets.py --daily panel/krx_1995_2026_daily.parquet \
        --output panel/krx_1995_2026_daily_targets.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

READ_COLUMNS = ["Date", "Code", "Market", "Volume", "Close", "Change"]
STATUS_ORDER = [
    "next_session", "exit_window_last_trade", "exit_window_unpriced", "no_trade",
    "link_break", "suspended", "no_observation_after", "final_day",
]


def listing_reference(path):
    """Same filter as build_panel.listing_reference."""
    data = pd.read_csv(path, dtype={"Symbol": str})
    for col in ("ListingDate", "DelistingDate"):
        data[col] = pd.to_datetime(data[col], errors="coerce")
    data = data.loc[data.Market.isin(["KOSPI", "KOSDAQ"]) & data.DelistingDate.between("1995-05-02", "2026-10-30")].copy()
    data["Code"] = data.Symbol.str.zfill(6)
    if data.Code.duplicated().any():
        raise ValueError("Multiple delisting records for a security code. Check listing spells.")
    return data.set_index("Code")[["ListingDate", "DelistingDate", "Market"]]


def daily_targets(daily, reference, exit_window=10):
    """Return Date, Code, NextDayReturn, NextDayLastTradeReturn, NextDayStatus."""
    rows = daily[READ_COLUMNS].sort_values(["Code", "Date"]).reset_index(drop=True)
    if rows.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate date/code rows in the daily panel.")
    calendar = pd.DatetimeIndex(rows.Date.unique()).sort_values()
    pos = calendar.get_indexer(rows.Date)
    n = len(rows)

    code = rows.Code.to_numpy()
    same_next = np.r_[code[1:] == code[:-1], False]
    pos_next = np.r_[pos[1:], -1]
    at_next_session = same_next & (pos_next == pos + 1)
    change_next = np.r_[rows.Change.to_numpy(dtype=float)[1:], np.nan]
    volume = rows.Volume.to_numpy(dtype=float)
    volume_next = np.r_[volume[1:], np.nan]
    linked = at_next_session & np.isfinite(change_next)
    traded_next = volume_next > 0

    # Official exit within the window, inside the same listing spell.
    ref = reference.reindex(rows.Code)
    delist = pd.DatetimeIndex(ref.DelistingDate)
    delist_pos = calendar.searchsorted(delist, side="left")
    has_exit = ~pd.isna(delist)
    same_spell = has_exit & (rows.Market.to_numpy() == ref.Market.to_numpy())
    same_spell &= rows.Date.to_numpy() >= ref.ListingDate.to_numpy(dtype="datetime64[ns]")
    in_exit_window = (
        same_spell & (delist.to_numpy(dtype="datetime64[ns]") > rows.Date.to_numpy())
        & (delist_pos <= pos + exit_window)
    )

    final_day = pos == len(calendar) - 1
    valid = linked & traded_next
    last_trade_ok = in_exit_window & valid & (volume > 0)

    status = np.select(
        [final_day, in_exit_window & last_trade_ok, in_exit_window, ~same_next,
         ~at_next_session, ~np.isfinite(change_next), ~traded_next, valid],
        ["final_day", "exit_window_last_trade", "exit_window_unpriced", "no_observation_after",
         "suspended", "link_break", "no_trade", "next_session"],
        default="link_break",
    )
    target = np.where(status == "next_session", change_next, np.nan)
    last_trade = np.where(status == "exit_window_last_trade", change_next, np.nan)
    return pd.DataFrame({
        "Date": rows.Date.to_numpy(), "Code": code,
        "NextDayReturn": target, "NextDayLastTradeReturn": last_trade,
        "NextDayStatus": pd.Categorical(status, categories=STATUS_ORDER),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--daily", type=Path, required=True, help="krx_1995_2026_daily.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exit-window", type=int, default=10,
                        help="Sessions before the official delisting date whose next-day return is not a main target.")
    parser.add_argument("--delisting-reference", type=Path,
                        default=Path(__file__).with_name("delisting_reference.csv"))
    args = parser.parse_args()

    daily = pd.read_parquet(args.daily, columns=READ_COLUMNS + ["Common"])
    daily["Code"] = daily.Code.astype(str).str.zfill(6)
    out = daily_targets(daily, listing_reference(args.delisting_reference), args.exit_window)
    out.to_parquet(args.output, index=False, compression="zstd")

    common_kospi = (daily.sort_values(["Code", "Date"]).reset_index(drop=True)
                    .eval("Market == 'KOSPI' and Common == True").to_numpy())
    print(f"Rows {len(out):,}; sessions {out.Date.nunique():,}; codes {out.Code.nunique():,}")
    print(f"Valid NextDayReturn {out.NextDayReturn.notna().sum():,} "
          f"(KOSPI common proxy {out.NextDayReturn.notna().to_numpy()[common_kospi].sum():,})")
    summary = out.groupby("NextDayStatus", observed=False).agg(
        rows=("Code", "size"),
        labelled=("NextDayReturn", "count"),
        last_trade=("NextDayLastTradeReturn", "count"),
        above_30pct=("NextDayReturn", lambda s: int((s > 0.30).sum())),
        below_minus_30pct=("NextDayReturn", lambda s: int((s < -0.30).sum())),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
