"""KRX downloads for rev4 (FEATURE_COLLECTION_LIST.md R3, R5, R6, R8, R15 parts).

  --probe     try KRX OPEN API endpoints (AUTH_KEY) on one date and report what each returns (VKOSPI? futures?)
  --vkospi    VKOSPI daily close from the OPEN API endpoint the probe found      -> raw/krx/vkospi.csv
              (KOSPI200 futures are added after the probe shows the futures fields)
  --flows    stock-level investor net buying via pykrx (needs KRX_ID/KRX_PW)    -> raw/krx/investor_flows_kospi.csv
  --market    market-level investor net buying via pykrx (needs KRX_ID/KRX_PW)   -> raw/krx/market_flows.csv
Keys come from the key file (keys.py); values are never printed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keys import load_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "rev4" / "data" / "raw" / "krx"
OPENAPI = "https://data-dbg.krx.co.kr/svc/apis"
CANDIDATES = ("idx/krx_dd_trd", "idx/kospi_dd_trd", "idx/drvprod_dd_trd", "idx/bon_dd_trd", "idx/idx_bydd_trd",
              "drv/fut_bydd_trd", "drv/eqsfu_stk_bydd_trd", "sto/stk_bydd_trd", "sto/stk_isu_base_info")
START, END = "2014-10-01", "2026-10-30"


def log(msg):
    RAW.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RAW / "download.log", "a", encoding="utf-8") as h:
        h.write(line + "\n")


def openapi(path, basdd, method=None):
    key = os.environ["KRX_AUTH_KEY"]
    last = None
    for m in ((method,) if method else ("GET", "POST")):
        url = f"{OPENAPI}/{path}" + (f"?basDd={basdd}" if m == "GET" else "")
        body = json.dumps({"basDd": basdd}).encode() if m == "POST" else None
        req = urllib.request.Request(url, data=body, method=m, headers={
            "AUTH_KEY": key, "Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": "research-data-client"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return m, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
    return None, {"error": last}


def records(payload):
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return v
    return []


def probe():
    found = {}
    for path in CANDIDATES:
        method, payload = openapi(path, "20240102")
        recs = records(payload)
        names = [str(r.get("IDX_NM") or r.get("ISU_NM") or "") for r in recs]
        vk = [n for n in names if "변동성" in n or "VKOSPI" in n.upper()]
        k2f = [n for n in names if "200" in n and ("선물" in n or "F " in n)]
        log(f"{path}: method={method} records={len(recs)} fields={list(recs[0].keys())[:12] if recs else payload}"
            + (f" VKOSPI-like={vk}" if vk else "") + (f" K200-futures-like={k2f[:3]}" if k2f else ""))
        found[path] = {"method": method, "n": len(recs), "vkospi": vk, "futures": k2f[:5],
                       "fields": list(recs[0].keys()) if recs else []}
    (RAW / "openapi_probe.json").write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    return found


def sessions():
    s = pd.to_datetime(pd.read_csv(ROOT / "inputs" / "market_indices.csv", usecols=["date"]).date)
    return s[(s >= START) & (s <= END)].sort_values()


def vkospi(found):
    hits = [(p, v) for p, v in found.items() if v["vkospi"]]
    if not hits:
        raise SystemExit("The probe found no VKOSPI-like index in the OPEN API endpoints.")
    path, info = hits[0]
    rows = []
    for day in sessions():
        _, payload = openapi(path, day.strftime("%Y%m%d"), info["method"])
        for r in records(payload):
            name = str(r.get("IDX_NM") or "")
            if name in info["vkospi"]:
                rows.append({"Date": day.date(), "Name": name, "VKOSPI": pd.to_numeric(str(r.get("CLSPRC_IDX", "")).replace(",", ""), errors="coerce")})
        time.sleep(0.05)
    frame = pd.DataFrame(rows)
    frame.loc[frame.Name == info["vkospi"][0], ["Date", "VKOSPI"]].to_csv(RAW / "vkospi.csv", index=False)
    log(f"vkospi.csv: {frame.Name.eq(info['vkospi'][0]).sum()} days from {path}")


def futures():
    """KOSPI200 futures (day session): front-month close vs spot -> basis; open interest summed over all contracts
    (a sum avoids the jump at each roll)."""
    num = lambda v: pd.to_numeric(str(v).replace(",", ""), errors="coerce")  # noqa: E731
    rows = []
    import numpy as np
    missed = []
    for day in sessions():
        # outright contracts only: calendar spreads ("SP 2406-2409") carry an empty open interest, which made the
        # first run's OI sum NaN on every day; failed requests (2018 almost entirely in the first run) are retried
        for attempt in range(4):
            _, payload = openapi("drv/fut_bydd_trd", day.strftime("%Y%m%d"), "GET")
            if records(payload):
                break
            time.sleep(2 * (attempt + 1))
        k = [r for r in records(payload) if str(r.get("PROD_NM", "")).strip() == "코스피200 선물"
             and "(주간)" in str(r.get("ISU_NM", "")) and " F " in str(r.get("ISU_NM", ""))
             and str(r.get("MKT_NM", "")).strip() == "정규"]
        if not k:
            missed.append(day.date().isoformat())
            continue
        k.sort(key=lambda r: str(r.get("ISU_NM", "")).split(" F ")[-1][:6])     # YYYYMM: nearest expiry first
        front = next((r for r in k if pd.notna(num(r.get("TDD_CLSPRC"))) and num(r.get("ACC_OPNINT_QTY")) > 0), k[0])
        close = num(front.get("TDD_CLSPRC"))
        rows.append({"Date": day.date(), "K200Spot": num(front.get("SPOT_PRC")),
                     "K200Futures": close if pd.notna(close) else num(front.get("SETL_PRC")),
                     "OpenInterest": float(np.nansum([num(r.get("ACC_OPNINT_QTY")) for r in k])),
                     "FrontContract": front.get("ISU_NM")})
        time.sleep(0.05)
    log(f"futures: {len(missed)} sessions without KOSPI200 futures rows: {missed[:10]}")
    frame = pd.DataFrame(rows)
    frame.to_csv(RAW / "futures_program.csv", index=False)
    log(f"futures_program.csv: {len(frame)} days (KOSPI200 futures; foreign/program net buying not in the OPEN API)")


def flows(market="KOSPI", end=END):
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        raise SystemExit("KRX_ID / KRX_PW are empty in the key file.")
    script = ROOT / "code" / "download_investor_flows.py"
    tag = "" if market == "KOSPI" else "_" + market.lower()
    argv = [str(script), "--index", str(ROOT / "inputs" / "market_indices.csv"), "--start", START, "--end", end,
            "--cache", str(RAW / f"flows_cache{tag}"), "--output", str(RAW / f"investor_flows_{market.lower()}.csv"),
            "--market", market]
    log("investor flows: " + " ".join(argv[1:]))
    # in-process (not a subprocess) so the Windows certificate store injected in main() also covers pykrx;
    # 2026-11-20 18:3x the KRX login failed with CERTIFICATE_VERIFY_FAILED under the certifi bundle
    import runpy
    saved = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = saved


def market():
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        raise SystemExit("KRX_ID / KRX_PW are empty in the key file.")
    from pykrx import stock
    frames = []
    for year in range(int(START[:4]), int(END[:4]) + 1):
        a, b = max(f"{year}0101", START.replace("-", "")), min(f"{year}1231", END.replace("-", ""))
        df = stock.get_market_trading_value_by_date(a, b, "KOSPI")
        if df is None or df.empty:
            log(f"market flows {year}: empty")
            continue
        frames.append(df)
        time.sleep(1)
    data = pd.concat(frames)
    cols = {"외국인합계": "ForeignNet", "외국인": "ForeignNet", "기관합계": "InstNet", "개인": "IndivNet"}
    out = data.rename(columns=cols)[[c for c in ("ForeignNet", "InstNet", "IndivNet") if c in data.rename(columns=cols)]]
    out.index = pd.to_datetime(out.index)
    out.rename_axis("Date").reset_index().to_csv(RAW / "market_flows.csv", index=False)
    log(f"market_flows.csv: {len(out)} days, columns {list(out.columns)} (source columns {list(data.columns)})")


def main():
    ap = argparse.ArgumentParser()
    for flag in ("probe", "vkospi", "futures", "flows", "market"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--flows-market", default="KOSPI", choices=("KOSPI", "KOSDAQ"), help="stock-level flows of this market")
    ap.add_argument("--flows-end", default=END)
    args = ap.parse_args()
    try:                                   # verify TLS against the Windows store (fetches missing intermediates)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    status = load_keys()
    log(f"keys: {{{', '.join(f'{k}: {v}' for k, v in status.items())}}}")
    RAW.mkdir(parents=True, exist_ok=True)
    found = None
    if args.probe or args.vkospi:
        if not status["KRX_AUTH_KEY"]:
            raise SystemExit("KRX_AUTH_KEY is empty in the key file.")
        found = probe()
    if args.vkospi:
        vkospi(found)
    if args.futures:
        futures()
    if args.market:
        market()
    if args.flows:
        flows(args.flows_market, args.flows_end)


if __name__ == "__main__":
    main()
