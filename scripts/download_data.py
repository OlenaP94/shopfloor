"""Download, verify and unpack the UCI hydraulic test rig dataset (id 447)."""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
DEST = Path("data/raw/hydraulic")

# Paste the hash printed on the first successful run to pin the download.
EXPECTED_SHA256 = ""

# name -> expected number of columns (sampling rate x 60 s)
EXPECTED_COLUMNS = {
    "PS1": 6000,
    "PS2": 6000,
    "PS3": 6000,
    "PS4": 6000,
    "PS5": 6000,
    "PS6": 6000,
    "EPS1": 6000,
    "FS1": 600,
    "FS2": 600,
    "TS1": 60,
    "TS2": 60,
    "TS3": 60,
    "TS4": 60,
    "VS1": 60,
    "CE": 60,
    "CP": 60,
    "SE": 60,
    "profile": 5,
}
EXPECTED_ROWS = 2205


def download(url: str, target: Path) -> str:
    """Download url to target, printing progress. Return the sha256 hex digest."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, target.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()
    return digest.hexdigest()


def validate(root: Path) -> None:
    """Check that every expected file exists with the right shape."""
    problems: list[str] = []

    for name, columns in EXPECTED_COLUMNS.items():
        path = root / f"{name}.txt"
        if not path.exists():
            problems.append(f"{name}.txt is missing")
            continue

        with path.open() as f:
            first = f.readline().split()
            rows = 1 + sum(1 for _ in f)

        if rows != EXPECTED_ROWS:
            problems.append(f"{name}.txt has {rows} rows, expected {EXPECTED_ROWS}")
        if len(first) != columns:
            problems.append(f"{name}.txt has {len(first)} columns, expected {columns}")

    if problems:
        raise SystemExit("Validation failed:\n  " + "\n  ".join(problems))

    print(f"  validated {len(EXPECTED_COLUMNS)} files, {EXPECTED_ROWS} cycles each")


def main() -> None:
    if DEST.exists() and (DEST / "profile.txt").exists():
        print(f"{DEST} already present — validating only")
        validate(DEST)
        return

    DEST.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "hydraulic.zip"

        print(f"Downloading {URL}")
        checksum = download(URL, archive)
        print(f"  sha256 {checksum}")

        if EXPECTED_SHA256 and checksum != EXPECTED_SHA256:
            raise SystemExit(
                f"Checksum mismatch!\n  expected {EXPECTED_SHA256}\n  got      {checksum}"
            )
        if not EXPECTED_SHA256:
            print("  (EXPECTED_SHA256 is empty — paste the hash above to pin it)")

        print(f"Unpacking into {DEST}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(DEST)

    # some UCI archives nest everything one level deeper
    inner = DEST / "data"
    if inner.is_dir() and (inner / "profile.txt").exists():
        for item in inner.iterdir():
            shutil.move(str(item), DEST)
        inner.rmdir()

    print("Validating")
    validate(DEST)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
