"""OpenDART: periodic-report financial statements (R4) and the periodic-filing list (R5) for KOSPI firms.

Firms: the union over years of the N largest KOSPI common stocks by market cap (default N = 300; the top 300 hold about
96% of KOSPI cap), processed largest first so that a stop at the daily call limit still covers most of the index.
Calls: fnlttSinglAcntAll (CFS first, OFS only when CFS has no data) for every year x report, and list.json (pblntf_ty=A)
once per firm. Responses are cached per call, so re-running resumes and never repeats a successful call.

  python -X utf8 rev4/code/download_dart.py --probe          # one firm, checks the key and the earliest year
  python -X utf8 rev4/code/download_dart.py --top 300        # full run (resumable)
  python -X utf8 rev4/code/download_dart.py --assemble       # write raw/dart/financials.csv and filings.csv from cache
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keys import load_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import os  # noqa: E402
DATA_ROOT = Path(os.environ.get("R4_DATA_ROOT", ROOT))            # KOSDAQ build: the kosdaq/ folder
RAW = Path(os.environ.get("R4_DART_RAW", ROOT / "rev4" / "data" / "raw" / "dart"))
CACHE = RAW / "cache"
API = "https://opendart.fss.or.kr/api"
REPORTS = ("11013", "11012", "11014", "11011")          # Q1, H1, Q3, annual
LAST_FILED = {2026: ("11013", "11012")}                 # Q3 2026 and FY2026 are filed after 2026-10-30
PAUSE = 0.12
LAST_YEAR = 2026
FIELDS = ["corp_code", "stock_code", "bsns_year", "reprt_code", "fs_div", "rcept_no", "sj_div", "account_id",
          "account_nm", "thstrm_amount", "thstrm_add_amount", "currency"]


class LimitReached(Exception):
    pass


def get(endpoint, **params):
    params["crtfc_key"] = __import__("os").environ["DART_API_KEY"]
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-data-client"}),
                                        timeout=60) as r:
                body = r.read()
            time.sleep(PAUSE)
            return body
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def get_json(endpoint, **params):
    for attempt in range(5):                     # a truncated body (seen 18:16) is retried, not fatal
        try:
            data = json.loads(get(endpoint, **params).decode("utf-8"))
            break
        except (json.JSONDecodeError, UnicodeDecodeError):
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    status = data.get("status")
    if status == "020":
        raise LimitReached(data.get("message", ""))
    if status in ("010", "011", "012", "901"):
        raise SystemExit(f"OpenDART key problem: status {status} {data.get('message')}")
    return data


def corp_codes():
    path = RAW / "corp_codes.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    z = zipfile.ZipFile(io.BytesIO(get("corpCode.xml")))
    root = ET.fromstring(z.read(z.namelist()[0]))
    rows = [{k: (item.findtext(k) or "").strip() for k in ("corp_code", "corp_name", "stock_code", "modify_date")}
            for item in root.iter("list")]
    frame = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def target_firms(top):
    firms = pd.read_parquet(DATA_ROOT / "daily_dataset" / "daily_firms.parquet", columns=["Date", "Code", "Marcap", "Valid"])
    firms = firms.loc[firms.Valid & (firms.Date >= "2014-01-01") & (firms.Date.dt.year <= LAST_YEAR)]
    firms["Code"] = firms.Code.astype(str).str.zfill(6)
    last = firms.groupby(firms.Date.dt.year).Date.transform("max")
    year_end = firms.loc[firms.Date == last]
    ranked = year_end.assign(rank=year_end.groupby(year_end.Date.dt.year).Marcap.rank(ascending=False))
    chosen = ranked.loc[ranked["rank"] <= top].groupby("Code")["rank"].min().sort_values()
    return chosen.index.tolist()


def map_codes(codes, cc):
    listed = cc.loc[cc.stock_code.str.len() == 6].drop_duplicates("stock_code").set_index("stock_code").corp_code
    ref = pd.read_csv(ROOT / "inputs" / "delisting_reference.csv", dtype={"Symbol": str})
    ref["Symbol"] = ref.Symbol.str.zfill(6)
    names = ref.set_index("Symbol").Name.to_dict()
    by_name = cc.drop_duplicates("corp_name").set_index("corp_name").corp_code
    out, missing = {}, []
    for c in codes:
        if c in listed.index:
            out[c] = listed[c]
        elif names.get(c) in by_name.index:          # delisted firms may have lost their stock code in DART
            out[c] = by_name[names[c]]
        else:
            missing.append(c)
    return out, missing


def fetch_statement(corp, code, year, reprt):
    path = CACHE / f"{corp}_{year}_{reprt}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = {"status": None, "fs_div": None, "list": []}
    for fs in ("CFS", "OFS"):
        data = get_json("fnlttSinglAcntAll.json", corp_code=corp, bsns_year=str(year), reprt_code=reprt, fs_div=fs)
        if data.get("status") == "000":
            result = {"status": "000", "fs_div": fs, "list": data.get("list", [])}
            break
        result["status"] = data.get("status")
    path.write_text(json.dumps({"stock_code": code, **result}, ensure_ascii=False), encoding="utf-8")
    return result


def fetch_filings(corp, code):
    path = CACHE / f"{corp}_filings.json"
    if path.exists():
        return
    rows, page = [], 1
    while True:
        data = get_json("list.json", corp_code=corp, bgn_de="20120101", end_de="20261030", pblntf_ty="A",
                        page_no=str(page), page_count="100")
        rows += data.get("list", []) if data.get("status") == "000" else []
        if data.get("status") != "000" or page >= int(data.get("total_page", 1)):
            break
        page += 1
    path.write_text(json.dumps({"stock_code": code, "list": rows}, ensure_ascii=False), encoding="utf-8")


def run(top, first_year, log):
    CACHE.mkdir(parents=True, exist_ok=True)
    cc = corp_codes()
    codes = target_firms(top)
    mapping, missing = map_codes(codes, cc)
    log(f"{len(codes)} target firms, {len(mapping)} mapped to DART corp codes, {len(missing)} unmapped")
    (RAW / "unmapped_codes.txt").write_text("\n".join(missing), encoding="utf-8")
    calls_before = len(list(CACHE.glob("*.json")))
    try:
        for i, code in enumerate(codes):
            corp = mapping.get(code)
            if corp is None:
                continue
            fetch_filings(corp, code)
            for year in range(first_year, LAST_YEAR + 1):
                for reprt in LAST_FILED.get(year, REPORTS):
                    fetch_statement(corp, code, year, reprt)
            if (i + 1) % 10 == 0:
                log(f"  {i + 1}/{len(codes)} firms done")
    except LimitReached as exc:
        log(f"daily call limit reached ({exc}); re-run tomorrow or with another key to resume")
    log(f"cached responses: {len(list(CACHE.glob('*.json')))} (new this run: {len(list(CACHE.glob('*.json'))) - calls_before})")


def assemble(log):
    rows, filings = [], []
    for path in CACHE.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.stem.endswith("_filings"):
            for item in data["list"]:
                filings.append({"corp_code": item.get("corp_code"), "stock_code": data["stock_code"],
                                "report_nm": item.get("report_nm"), "rcept_no": item.get("rcept_no"),
                                "rcept_dt": item.get("rcept_dt")})
        elif data.get("status") == "000":
            for item in data["list"]:
                rows.append({**{k: item.get(k, "") for k in FIELDS}, "stock_code": data["stock_code"],
                             "fs_div": data["fs_div"]})
    pd.DataFrame(rows, columns=FIELDS).to_csv(RAW / "financials.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filings).to_csv(RAW / "filings.csv", index=False, encoding="utf-8-sig")
    log(f"financials.csv: {len(rows)} rows; filings.csv: {len(filings)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--first-year", type=int, default=2015)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--last-year", type=int, default=2026)
    args = ap.parse_args()
    global LAST_YEAR
    LAST_YEAR = args.last_year
    status = load_keys()
    if not status["DART_API_KEY"]:
        raise SystemExit("DART_API_KEY is empty in the key file.")
    RAW.mkdir(parents=True, exist_ok=True)
    logf = RAW / "download.log"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as h:
            h.write(line + "\n")

    if args.probe:
        cc = corp_codes()
        corp = cc.loc[cc.stock_code == "005930", "corp_code"].iloc[0]
        for year in (2012, 2013, 2014, 2015):
            data = get_json("fnlttSinglAcntAll.json", corp_code=corp, bsns_year=str(year), reprt_code="11011", fs_div="CFS")
            log(f"probe Samsung FY{year}: status {data.get('status')} rows {len(data.get('list', []))}")
        log(f"corp codes: {len(cc)} entries, {int((cc.stock_code.str.len() == 6).sum())} with stock codes")
        return
    if args.assemble:
        assemble(log)
        return
    run(args.top, args.first_year, log)
    assemble(log)


if __name__ == "__main__":
    main()
