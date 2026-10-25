"""Download the APTOS 2019 diabetic retinopathy dataset from Kaggle.

Fetches train.csv, train_images.zip, test.csv, test_images.zip from the
diabetic-retinopathy-detection competition into data/raw/ and extracts them.

Auth delegates entirely to the Kaggle client (OAuth cache, env vars, or
~/.kaggle). We never read, print, or log any token.

IMPORTANT: check the APTOS 2019 dataset license and terms on Kaggle before
downloading. This repo never commits data.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "diabetic-retinopathy-detection"
RAW_DIR = Path("data/raw")
TRAIN_CSV = RAW_DIR / "train.csv"
TRAIN_IMAGES = RAW_DIR / "train_images"
HTTP_FORBIDDEN = 403


def _handle_download_error(exc: Exception) -> int:
    """Print an actionable message for download/auth failures and exit 1."""
    status = getattr(exc, "status", None)
    if status == HTTP_FORBIDDEN:
        print(
            "ERROR: HTTP 403 downloading APTOS 2019. Accept the competition "
            "rules on the Kaggle website (diabetic-retinopathy-detection), "
            "then retry.",
            file=sys.stderr,
        )
    else:
        print(f"ERROR: Kaggle download failed: {exc}", file=sys.stderr)
        print("Authenticate with: uv run kaggle auth login", file=sys.stderr)
    return 1


def _extract_zip(zip_path: Path) -> bool:
    """Extract *zip_path* next to itself. Return True on success.

    On failure the zip is left in place so it can be retried.
    """
    try:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
    except Exception as exc:  # keep zip on any extraction error
        print(
            f"WARNING: extraction of {zip_path.name} failed ({exc}); keeping zip.", file=sys.stderr
        )
        return False
    return True


def main() -> int:
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # auth failures are expected to surface
        print("Authenticate with: uv run kaggle auth login", file=sys.stderr)
        return _handle_download_error(exc)

    # Skip when outputs already exist.
    if TRAIN_CSV.exists() and TRAIN_IMAGES.is_dir():
        print(f"{RAW_DIR} already contains train.csv and train_images/; skipping download.")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {COMPETITION} files to {RAW_DIR} ...")
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as exc:
        return _handle_download_error(exc)

    # Extract each zip, deleting it only after successful extraction.
    for zip_path in sorted(RAW_DIR.glob("*.zip")):
        if _extract_zip(zip_path):
            zip_path.unlink()

    missing = []
    if not TRAIN_CSV.exists():
        missing.append("train.csv")
    if not TRAIN_IMAGES.is_dir():
        missing.append("train_images/")
    if missing:
        print(f"WARNING: expected outputs not found after download: {missing}", file=sys.stderr)

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
