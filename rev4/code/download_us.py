"""Download US market data that needs no API key (FEATURE_COLLECTION_LIST.md R13 and R23).

  rev4/data/raw/global/us_daily.csv      USDate, SPX, NASDAQ, SOX, VIX            (Yahoo: ^GSPC ^IXIC ^SOX ^VIX)
  rev4/data/raw/global/global_extra.csv  USDate, US10Y, US2Y, WTI, DXY, Copper, Gold
        US10Y, US2Y: FRED DGS10, DGS2 (fallback Yahoo ^TNX for 10Y only); WTI, Copper, Gold: Yahoo front-month
        futures CL=F, HG=F, GC=F; DXY: Yahoo DX-Y.NYB
Values are daily closes on US trading dates; the t-1 alignment to Korean dates is done in r4_ingest.py.
"""
import io
from pathlib import Path
import time
import urllib.request

import pandas as pd
import yfinance as yf

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
START, END = "2012-01-01", "2026-10-31"          # END is exclusive in yfinance
INDICES = {"^GSPC": "SPX", "^IXIC": "NASDAQ", "^SOX": "SOX", "^VIX": "VIX"}
EXTRA_YAHOO = {"CL=F": "WTI", "DX-Y.NYB": "DXY", "HG=F": "Copper", "GC=F": "Gold"}


def yahoo_close(tickers):
    frames = {}
    for ticker, name in tickers.items():
        for attempt in range(3):
            try:
                data = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False, threads=False)
                close = data["Close"]
                close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                if close.dropna().empty:
                    raise ValueError("empty")
                frames[name] = close.rename(name)
                break
            except Exception as exc:  # network hiccups: retry, then report
                if attempt == 2:
                    print(f"  {ticker}: failed ({exc})")
                time.sleep(2 * (attempt + 1))
    out = pd.concat(frames.values(), axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def fred(series_id, timeout=30):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}&coed=2026-10-30"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data client)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        frame = pd.read_csv(io.BytesIO(r.read()))
    frame.columns = ["date", series_id]
    frame["date"] = pd.to_datetime(frame.date)
    return pd.to_numeric(frame.set_index("date")[series_id], errors="coerce")


def summary(frame, name):
    print(f"{name}: {len(frame)} rows, {frame.index.min().date()} ~ {frame.index.max().date()}")
    print(frame.notna().mean().round(3).to_string())


def main():
    folder = RAW / "global"
    folder.mkdir(parents=True, exist_ok=True)
    us = yahoo_close(INDICES)
    us.rename_axis("USDate").reset_index().to_csv(folder / "us_daily.csv", index=False)
    summary(us, "us_daily.csv")

    extra = yahoo_close(EXTRA_YAHOO)
    sources = {k: f"Yahoo {t}" for t, k in EXTRA_YAHOO.items()}
    for sid, name in (("DGS10", "US10Y"), ("DGS2", "US2Y")):
        try:
            extra = extra.join(fred(sid).rename(name), how="outer")
            sources[name] = f"FRED {sid}"
        except Exception as exc:
            print(f"  FRED {sid}: failed ({exc})")
    if "US10Y" not in extra:
        tnx = yahoo_close({"^TNX": "US10Y"})
        extra = extra.join(tnx, how="outer")
        sources["US10Y"] = "Yahoo ^TNX (percent)"
    extra = extra.join(yahoo_close({"^IRX": "US3M"}), how="outer")     # 13-week bill, for 10Y-3M if 2Y is missing
    sources["US3M"] = "Yahoo ^IRX (percent)"
    extra = extra.loc[extra.index.isin(us.index)]              # keep US equity trading dates
    extra.rename_axis("USDate").reset_index().to_csv(folder / "global_extra.csv", index=False)
    summary(extra, "global_extra.csv")

    manifest = RAW / "manifest.csv"
    rows = [{"file": "global/us_daily.csv", "source": "Yahoo Finance " + " ".join(INDICES), "downloaded": time.strftime("%Y-%m-%d %H:%M"),
             "query": f"{START}~2026-10-30 daily close", "rows": len(us), "note": "unadjusted index closes"},
            {"file": "global/global_extra.csv", "source": "; ".join(f"{k}: {v}" for k, v in sources.items()),
             "downloaded": time.strftime("%Y-%m-%d %H:%M"), "query": f"{START}~2026-10-30 daily close",
             "rows": len(extra), "note": "futures are front-month continuous (roll jumps possible)"}]
    old = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
    pd.concat([old[~old.get("file", pd.Series(dtype=str)).isin([r["file"] for r in rows])] if len(old) else old,
               pd.DataFrame(rows)], ignore_index=True).to_csv(manifest, index=False)


if __name__ == "__main__":
    main()
