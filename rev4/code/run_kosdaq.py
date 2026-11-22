"""Unattended KOSDAQ replication of the firm-level competitor evaluation (2026-11-21; pre-test years 2016-2021 only).

Steps (each logged to kosdaq/logs/run_kosdaq.log; a finished step leaves its output and is skipped on re-run)
  1 wait for the KOSDAQ downloads: investor flows (rev4/data/raw/krx/investor_flows_kosdaq.csv), DART financials
  2 rev4 ingest on the KOSDAQ base panel (derived risk, flows, DART value ratios, quarterly ratios, industry snapshot);
    the market block is replaced by the KOSPI one (economy-wide state; the 7 base market features are KOSDAQ's own)
  3 correlation-graph cache for KOSDAQ
  4 wait for the KOSPI GPU competitor run, run the KOSPI evaluation, pick the MASTER variant with the higher KOSPI
    primary metric (rule fixed in BENCH_PROTOCOL_KO.md before the MASTER results were seen)
  5 KOSDAQ forecasters: Lin + LightGBM on CPU, our NN (seeds 0-2) then mlp, gru, dtml, MASTER variant on the GPU
  6 KOSDAQ evaluation
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "rev4" / "code"
KQ = ROOT / "kosdaq"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = KQ / "logs" / "run_kosdaq.log"
ENV = dict(os.environ, R4_DATA_ROOT=str(KQ), R4_RAW=str(KQ / "raw"), R4_PROCESSED=str(KQ / "processed"),
           R4_GRAPH_DIR=str(KQ / "graphs" / "correlation_graphs"), R4_FLOWS_FILE="investor_flows_kosdaq.csv",
           PYTHONUTF8="1")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as h:
        h.write(line + "\n")


def run(args, env=ENV, logname=None, background=False):
    out = open(KQ / "logs" / (logname or "step.log"), "a", encoding="utf-8")
    log("run: " + " ".join(str(a) for a in args[1:]))
    p = subprocess.Popen([PY, "-u", "-X", "utf8", *map(str, args)], cwd=CODE, env=env, stdout=out,
                         stderr=subprocess.STDOUT)
    if background:
        return p
    rc = p.wait()
    log(f"  exit {rc}")
    if rc != 0:
        raise SystemExit(f"step failed: {args[:2]} (see {logname})")
    return p


def scrub(name):
    """Remove pykrx login lines from a log file (they can carry the login id)."""
    path = KQ / "logs" / name
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        path.write_text("\n".join(l for l in lines if "로그인" not in l) + "\n", encoding="utf-8")


def gpu_job_running(tag):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
                         capture_output=True, text=True).stdout
    return any(tag in line for line in out.splitlines())


def main():
    (KQ / "logs").mkdir(parents=True, exist_ok=True)
    flows = ROOT / "rev4" / "data" / "raw" / "krx" / "investor_flows_kosdaq.csv"
    fin = KQ / "raw" / "dart" / "financials.csv"
    log("step 1: waiting for flows and DART")
    while not (flows.exists() and fin.exists()):
        time.sleep(60)
    time.sleep(30)
    (KQ / "raw" / "krx").mkdir(parents=True, exist_ok=True)
    shutil.copy2(flows, KQ / "raw" / "krx" / "investor_flows_kosdaq.csv")
    log(f"step 1 done: flows {flows.stat().st_size >> 20} MB, financials {fin.stat().st_size >> 20} MB")

    if not (KQ / "processed" / "firm_daily_extra.parquet").exists():
        log("step 2: ingest")
        run([CODE / "r4_ingest.py", "--derived", "--calendar"], logname="ingest.log")
        kospi_mk = pd.read_parquet(ROOT / "rev4" / "data" / "processed" / "market_daily_extra.parquet")
        own = pd.read_parquet(KQ / "processed" / "market_daily_extra.parquet")
        own.to_parquet(KQ / "processed" / "market_daily_extra_own.parquet", index=False)
        kospi_mk.to_parquet(KQ / "processed" / "market_daily_extra.parquet", index=False)
        log(f"step 2 done; market block replaced by the KOSPI one ({kospi_mk.shape[1] - 1} columns; own block kept "
            f"as market_daily_extra_own.parquet with {own.shape[1] - 1} columns)")

    if not (KQ / "graphs" / "correlation_graphs").exists() or not any((KQ / "graphs" / "correlation_graphs").iterdir()):
        log("step 3: correlation graphs")
        run(["-c", "import sys; sys.path.insert(0, r'%s'); from r4_common import load_data4, CorrelationGraphs; "
                   "d, g = load_data4(); print(g['inputs']['firm_features']); "
                   "CorrelationGraphs.prepare(d, r'%s', log=print)" % (CODE, KQ / "graphs")], logname="graphs.log")

    log("step 4: waiting for the KOSPI GPU run")
    while gpu_job_running("bench_stage1.py --models mlp,gru,dtml,master,master_csz"):
        time.sleep(120)
    kospi_env = {k: v for k, v in os.environ.items() if not k.startswith("R4_")}
    kospi_env["PYTHONUTF8"] = "1"
    summary = ROOT / "rev4" / "results" / "bench" / "eval_kospi" / "summary_open_entry_2016_2021.csv"
    if not summary.exists():
        run([CODE / "bench_eval.py", "--market", "kospi"], env=kospi_env, logname="eval_kospi.log")
    s = pd.read_csv(summary).set_index("model").RankIC
    variant = "master" if s.get("master", -1) >= s.get("master_csz", -1) else "master_csz"
    log(f"step 4 done: KOSPI RankIC master {s.get('master')}, master_csz {s.get('master_csz')} -> KOSDAQ runs {variant}")

    log("step 5: KOSDAQ forecasters")
    oof, bench = KQ / "results" / "oof", KQ / "results" / "bench" / "oof"
    cpu = [run([CODE / "r4_stage1.py", "--model", "lin", "--years", "2016-2021", "--out", oof], logname="lin.log",
               background=True)]
    cpu.append(run([CODE / "bench_stage1.py", "--models", "lgb", "--years", "2016-2021", "--seeds", "0,1,2",
                    "--device", "cpu", "--out", bench], logname="lgb.log", background=True))
    run([CODE / "r4_stage1.py", "--model", "nn", "--years", "2016-2021", "--seeds", "0,1,2", "--out", oof],
        logname="nn.log")
    run([CODE / "bench_stage1.py", "--models", f"mlp,gru,dtml,{variant}", "--years", "2016-2021", "--seeds", "0,1,2",
         "--out", bench], logname="bench_gpu.log")
    for p in cpu:
        p.wait()
    log("step 5 done")

    log("step 6: KOSDAQ evaluation")
    run([CODE / "bench_eval.py", "--market", "kosdaq", "--base", KQ / "results",
         "--competitors", f"lin,anomaly,lgb,mlp,gru,dtml,{variant}"], logname="eval_kosdaq.log")
    (KQ / "results" / "KOSDAQ_DONE.json").write_text('{"done": true}', encoding="utf-8")
    log("all done")


if __name__ == "__main__":
    main()
