"""Bank of Korea ECOS downloads (FEATURE_COLLECTION_LIST.md R10, R11, R12, R24).

Item codes are looked up by name from StatisticItemList, so a renamed or re-coded item fails loudly instead of
silently pulling the wrong series.
  raw/ecos/rates.csv      Date, KTB3Y, KTB10Y, CD91, CorpAAm3Y          (817Y002 market interest rates, daily)
  raw/ecos/fx.csv         Date, USDKRW                                  (731Y003 15:30 close if found, else 731Y001)
  raw/ecos/base_rate.csv  MeetingDate, BaseRate, Decision               (722Y001 base rate; change dates only)
  raw/macro/monthly.csv   Indicator, Period, ReleaseDate, Value         (--macro; conservative release dates)
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keys import load_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "rev4" / "data" / "raw"
API = "https://ecos.bok.or.kr/api"
START, END = "20120101", "20261030"


def call(service, *parts):
    url = f"{API}/{service}/{os.environ['ECOS_API_KEY']}/json/kr/" + "/".join(urllib.parse.quote(str(p)) for p in parts)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-data-client"}),
                                        timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(0.2)
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def items(stat):
    data = call("StatisticItemList", 1, 2000, stat)
    rows = data.get("StatisticItemList", {}).get("row", [])
    if not rows:
        raise SystemExit(f"ECOS {stat}: no item list ({data.get('RESULT')})")
    return pd.DataFrame(rows)


def find(stat, *keywords, cycle=None):
    it = items(stat)
    hit = it
    for k in keywords:
        hit = hit.loc[hit.ITEM_NAME.str.replace(" ", "").str.contains(k.replace(" ", ""), regex=False)]
    if cycle:
        hit = hit.loc[hit.CYCLE == cycle]
    if hit.empty:
        raise SystemExit(f"ECOS {stat}: no item matching {keywords}; names: {it.ITEM_NAME.unique()[:30].tolist()}")
    return hit.iloc[0]


def series(stat, cycle, item_code, start=START, end=END):
    if cycle == "M":
        start, end = start[:6], end[:6]
    data = call("StatisticSearch", 1, 100000, stat, cycle, start, end, item_code)
    rows = data.get("StatisticSearch", {}).get("row", [])
    if not rows:
        raise SystemExit(f"ECOS {stat}/{item_code}: no data ({data.get('RESULT')})")
    s = pd.Series({r["TIME"]: pd.to_numeric(r["DATA_VALUE"], errors="coerce") for r in rows}).sort_index()
    s.index = pd.to_datetime(s.index, format="%Y%m%d" if cycle == "D" else "%Y%m")
    return s


def daily_block(log):
    (RAW / "ecos").mkdir(parents=True, exist_ok=True)
    picks = {"KTB3Y": ("국고채(3년)",), "KTB10Y": ("국고채(10년)",), "CD91": ("CD(91일)",),
             "CorpAAm3Y": ("회사채(3년", "AA-")}
    rates, names = {}, {}
    for col, kw in picks.items():
        row = find("817Y002", *kw, cycle="D")
        rates[col] = series("817Y002", "D", row.ITEM_CODE)
        names[col] = row.ITEM_NAME
    pd.DataFrame(rates).rename_axis("Date").reset_index().to_csv(RAW / "ecos" / "rates.csv", index=False)
    log(f"rates.csv: {len(rates['KTB3Y'])} days; items {names}")
    try:
        row = find("731Y003", "종가", cycle="D")
        fx, src = series("731Y003", "D", row.ITEM_CODE), f"731Y003 {row.ITEM_NAME}"
    except SystemExit:
        row = find("731Y001", "미국달러", cycle="D")
        fx, src = series("731Y001", "D", row.ITEM_CODE), f"731Y001 {row.ITEM_NAME}"
    fx.rename("USDKRW").rename_axis("Date").reset_index().to_csv(RAW / "ecos" / "fx.csv", index=False)
    log(f"fx.csv: {len(fx)} days from {src}")
    row = find("722Y001", "기준금리")
    cycle = row.CYCLE if row.CYCLE in ("D", "M") else "D"
    br = series("722Y001", cycle, row.ITEM_CODE).dropna()
    change = br.diff()
    events = br[(change != 0) | change.isna()]
    out = pd.DataFrame({"MeetingDate": events.index, "BaseRate": events.values,
                        "Decision": ["start" if pd.isna(c) else ("인상" if c > 0 else "인하")
                                     for c in change[events.index]]})
    out.to_csv(RAW / "ecos" / "base_rate.csv", index=False)
    log(f"base_rate.csv: {len(out)} change dates from 722Y001 {row.ITEM_NAME} ({cycle}); "
        "hold meetings are not in this series (MPCDay marks changes only)")


MACRO = {  # indicator -> (stat code, keywords, transform, release rule: months after period, day of month)
    "CPI_YoY": ("901Y009", ("총지수",), "yoy", (1, 10)),
    "CoreCPI_YoY": ("901Y009", ("농산물및석유류제외",), "yoy", (1, 10)),
    "Unemployment_YoYChg": ("901Y027", ("실업률",), "diff12", (1, 20)),
    "IndustrialProd_YoY": ("901Y033", ("전산업생산지수",), "yoy", (2, 1)),
    "LeadingCycle_m100": ("901Y067", ("선행지수순환변동치",), "minus100", (2, 1)),
    "CCSI_m100": ("511Y002", ("소비자심리지수",), "minus100", (1, 1)),
}


def macro_block(log):
    (RAW / "macro").mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (stat, kw, transform, (lag_m, day)) in MACRO.items():
        try:
            row = find(stat, *kw, cycle="M")
            s = series(stat, "M", row.ITEM_CODE)
        except SystemExit as exc:
            log(f"macro {name}: skipped ({str(exc)[:160]})")
            continue
        v = {"yoy": 100 * (s / s.shift(12) - 1), "diff12": s - s.shift(12), "minus100": s - 100}[transform]
        period = v.index.to_period("M")
        release = (period + lag_m).to_timestamp() + pd.to_timedelta(day - 1, unit="D")
        frames.append(pd.DataFrame({"Indicator": name, "Period": period.astype(str), "ReleaseDate": release.date,
                                    "Value": v.values}).dropna())
        log(f"macro {name}: {stat} {row.ITEM_NAME}, {len(frames[-1])} months")
    if frames:
        pd.concat(frames).to_csv(RAW / "macro" / "monthly.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro", action="store_true")
    args = ap.parse_args()
    if not load_keys()["ECOS_API_KEY"]:
        raise SystemExit("ECOS_API_KEY is empty in the key file.")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        (RAW / "ecos").mkdir(parents=True, exist_ok=True)
        with open(RAW / "ecos" / "download.log", "a", encoding="utf-8") as h:
            h.write(line + "\n")

    daily_block(log)
    if args.macro:
        macro_block(log)


if __name__ == "__main__":
    main()
