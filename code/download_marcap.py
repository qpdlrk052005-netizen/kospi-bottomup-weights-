"""Download the public annual KRX source files used by build_panel.py."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import shutil
import time
import urllib.request

REF = "d3b703628e5a73112111d8bc43439f950762f67e"


def fetch(path, year):
    name = f"marcap-{year}.parquet"
    dest = path / name
    if dest.exists():
        with dest.open("rb") as f:
            f.seek(-4, 2)
            if f.read() == b"PAR1":
                return
    temp = dest.with_suffix(".part")
    url = f"https://raw.githubusercontent.com/FinanceData/marcap/{REF}/data/{name}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, temp.open("wb") as out:
                shutil.copyfileobj(response, out)
            with temp.open("rb") as f:
                f.seek(-4, 2)
                if f.read() != b"PAR1":
                    raise ValueError(f"Incomplete download for {year}")
            temp.replace(dest)
            print(f"Downloaded {year}", flush=True)
            return
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        finally:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("raw_marcap"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda year: fetch(args.output, year), range(1995, 2027)))


if __name__ == "__main__":
    main()
