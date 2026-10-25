"""Cache preprocessed APTOS images to data/processed/.

Reads PNG/JPEG images from data/raw/train_images, runs the preprocessing
pipeline from src.lumoculis.preprocess, and writes uint8 PNG results to
data/processed/. Skips images whose output already exists.

Size defaults to 384 (configurable via --size). Idempotent: safe to re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from lumoculis.preprocess import preprocess

RAW_IMAGES = Path("data/raw/train_images")
PROCESSED_DIR = Path("data/processed")
SUPPORTED = {".png", ".jpg", ".jpeg"}


def process_image(src: Path, dst: Path, size: int) -> bool:
    """Preprocess *src* and write to *dst*. Return True on success."""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        print(f"WARNING: could not read {src}; skipping.", file=sys.stderr)
        return False

    out = preprocess(img, size=size)
    cv2.imwrite(str(dst), out)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess APTOS fundus images.")
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="Output image size (square). Default 384.",
    )
    args = parser.parse_args()

    if not RAW_IMAGES.is_dir():
        print(
            f"ERROR: {RAW_IMAGES} not found. Run scripts/download_data.py first.", file=sys.stderr
        )
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in RAW_IMAGES.iterdir() if p.suffix.lower() in SUPPORTED)
    if not images:
        print(f"No images found in {RAW_IMAGES}.", file=sys.stderr)
        return 1

    done = 0
    skipped = 0
    for src in images:
        dst = PROCESSED_DIR / (src.stem + ".png")
        if dst.exists():
            skipped += 1
            continue
        if process_image(src, dst, args.size):
            done += 1

    print(f"Processed {done} new images, skipped {skipped} already-cached, total {len(images)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
