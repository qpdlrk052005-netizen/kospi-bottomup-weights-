"""Load API keys and logins from the user's key file into os.environ without ever printing the values.

Default file: %USERPROFILE%\\kospi_keys.txt (override with the R4_KEYS environment variable).
Format: KEY=value per line; lines starting with # are ignored; empty values are skipped.
"""
import os
from pathlib import Path

NAMES = ("KRX_AUTH_KEY", "DART_API_KEY", "ECOS_API_KEY", "KRX_ID", "KRX_PW")


def default_path():
    for p in (os.environ.get("R4_KEYS"), Path.home() / "Downloads" / "key.txt", Path.home() / "kospi_keys.txt"):
        if p and Path(p).exists() and "=" in Path(p).read_text(encoding="utf-8-sig").split("#")[-1] + "=":
            text = Path(p).read_text(encoding="utf-8-sig")
            if any(line.split("=", 1)[1].strip() for line in text.splitlines()
                   if "=" in line and not line.strip().startswith("#")):
                return Path(p)
    return Path.home() / "kospi_keys.txt"


def load_keys(path=None):
    path = Path(path) if path else default_path()
    found = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if name in NAMES and value:
                os.environ[name] = value
                found[name] = True
    return {n: bool(found.get(n) or os.environ.get(n)) for n in NAMES}


if __name__ == "__main__":
    status = load_keys()
    print({k: ("set" if v else "empty") for k, v in status.items()})
