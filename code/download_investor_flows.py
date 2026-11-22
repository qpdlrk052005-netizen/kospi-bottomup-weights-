"""Download daily net trading value by investor type for every KOSPI stock from KRX via pykrx.

One request per session and investor: pykrx.stock.get_market_net_purchases_of_equities(
date, date, "KOSPI", investor) returns, for that day, each ticker's sell, buy and net-buy
value. Results are cached per investor and year in --cache so an interrupted run resumes.

Default investors: 외국인 (foreign), 기관합계 (institutions), 개인 (individuals).
Output CSV columns: Date, Code, ForeignNetValue, InstNetValue, IndivNetValue (KRW),
plus the matching buy and sell values. Days on which KRX returns nothing are listed in
--cache/empty_<investor>.txt; the first year with data decides whether a flow feature
has enough history (see build_daily_dataset.py --min-start).

    pip install pykrx
    python download_investor_flows.py --output investor_flows_kospi.csv
"""
import argparse
import time
from pathlib import Path

import pandas as pd

INVESTORS = {"외국인": "Foreign", "기관합계": "Inst", "개인": "Indiv"}
VALUE_COLUMNS = {"매도거래대금": "SellValue", "매수거래대금": "BuyValue", "순매수거래대금": "NetValue"}


def fetch_day(stock, day, investor, retries=4, pause=0.5, market="KOSPI"):
    stamp = day.strftime("%Y%m%d")
    for attempt in range(retries):
        try:
            frame = stock.get_market_net_purchases_of_equities(stamp, stamp, market, investor)
            time.sleep(pause)
            break
        except Exception:  # KRX sometimes drops a request; back off and retry.
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    if frame is None or frame.empty:
        return None
    missing = set(VALUE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Unexpected pykrx columns on {stamp}: {list(frame.columns)}")
    prefix = INVESTORS[investor]
    out = frame[list(VALUE_COLUMNS)].rename(columns={k: prefix + v for k, v in VALUE_COLUMNS.items()})
    out.index = out.index.astype(str).str.zfill(6)
    return out.rename_axis("Code").reset_index().assign(Date=day.date())


def download(stock, sessions, investor, cache, market="KOSPI"):
    prefix = INVESTORS[investor]
    empty_log = cache / f"empty_{prefix}.txt"
    for year, days in sessions.groupby(sessions.dt.year):
        path = cache / f"{prefix}_{year}.csv"
        if path.exists():
            continue
        frames, empty = [], []
        for day in days:
            got = fetch_day(stock, day, investor, market=market)
            if got is None:
                empty.append(day.date().isoformat())
            else:
                frames.append(got)
        columns = ["Code", *(prefix + v for v in VALUE_COLUMNS.values()), "Date"]
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        data.to_csv(path, index=False)
        with empty_log.open("a", encoding="utf-8") as log:
            log.writelines(f"{d}\n" for d in empty)
        print(f"{prefix} {year}: {len(days)} sessions, {len(days) - len(empty)} with data", flush=True)
    files = sorted(cache.glob(f"{prefix}_*.csv"))
    return pd.concat([pd.read_csv(p, dtype={"Code": str}) for p in files], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("market_indices.csv"))
    parser.add_argument("--start", default="1995-05-02")
    parser.add_argument("--end", default="2026-10-30")
    parser.add_argument("--investors", nargs="+", default=list(INVESTORS), choices=list(INVESTORS))
    parser.add_argument("--cache", type=Path, default=Path("investor_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", default="KOSPI", choices=("KOSPI", "KOSDAQ"))
    args = parser.parse_args()

    from pykrx import stock

    sessions = pd.to_datetime(pd.read_csv(args.index, usecols=["date"]).date)
    sessions = sessions[(sessions >= args.start) & (sessions <= args.end)].sort_values()
    args.cache.mkdir(parents=True, exist_ok=True)

    merged = None
    for investor in args.investors:
        data = download(stock, sessions, investor, args.cache, args.market)
        if data.duplicated(["Date", "Code"]).any():
            raise ValueError(f"Duplicate date/code rows for {investor}.")
        merged = data if merged is None else merged.merge(data, on=["Date", "Code"], how="outer")
        print(f"{investor}: first session with data {data.Date.min()}", flush=True)
    merged.sort_values(["Date", "Code"]).to_csv(args.output, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output}")


if __name__ == "__main__":
    main()
