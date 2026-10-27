"""Inspect the real APTOS dataset after preprocessing.

Loads data/raw/train.csv and cached PNGs from data/processed/, builds the
group-aware train/val split, and prints:
  - train/val sizes
  - per-class counts in each split
  - class weights (and whether any exceeds ~5x the smallest)
  - exact duplicate groups: unique groups, images involved, cross-split
    groups, conflicting-label groups
  - one batch's shapes/dtypes/value ranges

Saves a grid of 16 augmented training images to data/inspect_grid.png
(gitignored).
"""

from __future__ import annotations

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
    _representative_label,
    class_counts_and_weights,
    find_duplicate_groups,
    train_val_split,
)

CSV_PATH = Path("data/raw/train.csv")
PROCESSED_DIR = Path("data/processed")
GRID_PATH = Path("data/inspect_grid.png")
WEIGHT_RATIO_WARN = 5.0


def _class_counts(df) -> dict[int, int]:
    return {
        int(cls): int(count) for cls, count in df["diagnosis"].value_counts().sort_index().items()
    }


def _report_duplicates(train_ds, val_ds, processed_dir: Path) -> None:
    """Report exact duplicate groups: counts, cross-split, conflicting labels."""
    all_ids = list(train_ds.df["id_code"]) + list(val_ds.df["id_code"])
    id_to_label = {
        str(k): int(v)
        for k, v in zip(train_ds.df["id_code"], train_ds.df["diagnosis"], strict=True)
    } | {str(k): int(v) for k, v in zip(val_ds.df["id_code"], val_ds.df["diagnosis"], strict=True)}
    hash_groups = find_duplicate_groups(processed_dir, all_ids)
    dup_groups = {h: ids for h, ids in hash_groups.items() if len(ids) > 1}

    train_id_set = set(train_ds.df["id_code"].astype(str))
    val_id_set = set(val_ds.df["id_code"].astype(str))

    print(f"exact duplicate groups: {len(dup_groups)}")
    print(f"images in duplicate groups: {sum(len(ids) for ids in dup_groups.values())}")

    cross_split = []
    for h, ids in dup_groups.items():
        in_train = any(gid in train_id_set for gid in ids)
        in_val = any(gid in val_id_set for gid in ids)
        if in_train and in_val:
            cross_split.append((h, ids))
    print(f"duplicate groups crossing train/val: {len(cross_split)}")
    if cross_split:
        print("WARNING: duplicates cross the train/val boundary (leakage!):")
        for h, ids in cross_split:
            print(f"  {h[:16]}...: {ids}")

    conflicting = []
    for h, ids in dup_groups.items():
        labels = [id_to_label[gid] for gid in ids]
        if _representative_label(labels) == -1:
            conflicting.append((h, ids, labels))
    print(f"duplicate groups with conflicting labels: {len(conflicting)}")
    if conflicting:
        for h, ids, labels in conflicting:
            print(f"  {h[:16]}...: ids={ids}, labels={labels}")


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

    train_ds, val_ds, dup_report = train_val_split(CSV_PATH, PROCESSED_DIR, config)

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
        print("WARNING: a class weight exceeds ~5x the smallest; consider dampened weights.")
        _, damp_weights = class_counts_and_weights(train_ds.df, dampen=True)
        print(f"dampened weights: {damp_weights}")

    print(f"duplicate groups (from split): {dup_report['n_dup_groups']}")
    print(f"images in duplicate groups: {dup_report['n_dup_images']}")
    print(f"conflicting groups (from split): {len(dup_report['conflicting_groups'])}")
    print(f"cross-split groups (from split): {len(dup_report['cross_split_groups'])}")

    _report_duplicates(train_ds, val_ds, PROCESSED_DIR)

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
