"""Inspect the real APTOS dataset after preprocessing.

Loads data/raw/train.csv and cached PNGs from data/processed/, builds the
stratified train/val split, and prints:
  - train/val sizes
  - per-class counts in each split
  - class weights (and whether any exceeds ~5x the smallest)
  - one batch's shapes/dtypes/value ranges
  - exact duplicate image count (hash of processed PNG bytes)

Saves a grid of 16 augmented training images to data/inspect_grid.png
(gitignored). If duplicates exist, reports them and proposes keeping them
in the same split.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lumoculis.config import Config
from lumoculis.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    class_counts_and_weights,
    train_val_split,
)

CSV_PATH = Path("data/raw/train.csv")
PROCESSED_DIR = Path("data/processed")
GRID_PATH = Path("data/inspect_grid.png")
WEIGHT_RATIO_WARN = 5.0


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_duplicates(processed_dir: Path, id_codes) -> tuple[int, list[tuple[str, str]]]:
    """Return (duplicate_count, list of (id_code_1, id_code_2) duplicate pairs)."""
    hashes: dict[str, list[str]] = {}
    for id_code in id_codes:
        p = processed_dir / f"{id_code}.png"
        if p.exists():
            hashes.setdefault(_hash_file(p), []).append(str(id_code))
    pairs = []
    dup_count = 0
    for ids in hashes.values():
        if len(ids) > 1:
            dup_count += len(ids) - 1
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((ids[i], ids[j]))
    return dup_count, pairs


def _class_counts(df) -> dict[int, int]:
    return {
        int(cls): int(count) for cls, count in df["diagnosis"].value_counts().sort_index().items()
    }


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run scripts/download_data.py first.", file=sys.stderr)
        return 1
    if not PROCESSED_DIR.is_dir():
        print(
            f"ERROR: {PROCESSED_DIR} not found. Run scripts/preprocess_images.py first.",
            file=sys.stderr,
        )
        return 1

    config = Config(
        image_size=384,
        batch_size=16,
        epochs=10,
        lr=0.0003,
        model_name="efficientnet_b0",
        seed=42,
    )

    train_ds, val_ds = train_val_split(CSV_PATH, PROCESSED_DIR, config)

    print(f"train size: {len(train_ds)}")
    print(f"val size:   {len(val_ds)}")
    print(f"train class counts: {_class_counts(train_ds.df)}")
    print(f"val class counts:   {_class_counts(val_ds.df)}")

    counts, weights = class_counts_and_weights(train_ds.df)
    print(f"class counts:   {dict(counts)}")
    print(f"class weights:  {weights}")
    min_w = min(weights.values())
    max_w = max(weights.values())
    ratio = max_w / min_w if min_w > 0 else float("inf")
    print(f"max/min weight ratio: {ratio:.2f}x")
    if ratio > WEIGHT_RATIO_WARN:
        print("WARNING: a class weight exceeds ~5x the smallest; consider --dampen.")
        _, damp_weights = class_counts_and_weights(train_ds.df, dampen=True)
        print(f"dampened weights: {damp_weights}")

    # Duplicate images by PNG byte hash.
    all_ids = list(train_ds.df["id_code"]) + list(val_ds.df["id_code"])
    dup_count, dup_pairs = _count_duplicates(PROCESSED_DIR, all_ids)
    print(f"exact duplicate images (by PNG hash): {dup_count}")
    if dup_pairs:
        print("Duplicate pairs:")
        for a, b in dup_pairs:
            print(f"  {a} == {b}")
        print("Proposal: keep duplicate images in the same split (train or val) to avoid leakage.")

    # One batch shapes/dtypes/value ranges.
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    images, labels = next(iter(train_loader))
    print(f"batch image shape: {tuple(images.shape)} dtype={images.dtype}")
    print(f"batch label shape: {tuple(labels.shape)} dtype={labels.dtype}")
    print(f"batch image min/max: {images.min().item():.4f} / {images.max().item():.4f}")
    print(f"batch labels: {labels.tolist()}")

    # Grid of 16 augmented training images.
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax in axes.flat:
        img, _ = train_ds[np.random.randint(len(train_ds))]
        # Undo ImageNet normalization for display.
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        disp = img * std + mean
        disp = disp.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(disp)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(GRID_PATH)
    plt.close(fig)
    print(f"Saved augmented image grid to {GRID_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
