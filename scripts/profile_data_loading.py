"""Profile data-loading performance for Lumoculis.

Measures each step of __getitem__ (disk read, decode, augmentation,
normalization, device transfer) for both train and val paths, using the same
representative images. Run with:

    uv run python scripts/profile_data_loading.py

This does NOT start training. It profiles the dataset pipeline only.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from lumoculis.config import Config
from lumoculis.data import (
    IMAGENET_MEAN_TENSOR,
    IMAGENET_STD_TENSOR,
    train_transforms,
)

PROCESSED_DIR = Path("data/processed")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_SAMPLES = 20


def _time_getitem_manual(id_codes, transform, label_map) -> dict[str, float]:
    """Manually replicate __getitem__ and time each phase."""
    timings: dict[str, float] = {
        "disk_read": 0.0,
        "cvt_color": 0.0,
        "pil_convert": 0.0,
        "transform": 0.0,
        "to_tensor": 0.0,
        "normalize": 0.0,
        "label_tensor": 0.0,
        "device_transfer": 0.0,
        "total": 0.0,
    }

    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for id_code in id_codes:
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        image = cv2.imread(str(PROCESSED_DIR / f"{id_code}.png"), cv2.IMREAD_COLOR)
        timings["disk_read"] += time.perf_counter() - t1
        if image is None:
            raise FileNotFoundError(f"Could not read {id_code}.png")

        t2 = time.perf_counter()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timings["cvt_color"] += time.perf_counter() - t2

        t3 = time.perf_counter()
        image = Image.fromarray(image)
        timings["pil_convert"] += time.perf_counter() - t3

        t4 = time.perf_counter()
        if transform is not None:
            image = transform(image)
        timings["transform"] += time.perf_counter() - t4

        t5 = time.perf_counter()
        image = transforms.ToTensor()(image)
        timings["to_tensor"] += time.perf_counter() - t5

        t6 = time.perf_counter()
        image = (image - IMAGENET_MEAN_TENSOR) / IMAGENET_STD_TENSOR
        timings["normalize"] += time.perf_counter() - t6

        t7 = time.perf_counter()
        label = torch.tensor(label_map[id_code], dtype=torch.long)
        timings["label_tensor"] += time.perf_counter() - t7

        images.append(image)
        labels.append(label)
        timings["total"] += time.perf_counter() - t0

    # Batch and device transfer.
    t_batch0 = time.perf_counter()
    batch_images = torch.stack(images)
    batch_labels = torch.stack(labels)
    t_batch1 = time.perf_counter()
    timings["batch_assembly"] = t_batch1 - t_batch0

    t_dev0 = time.perf_counter()
    batch_images = batch_images.to(DEVICE)
    batch_labels = batch_labels.to(DEVICE)
    t_dev1 = time.perf_counter()
    timings["device_transfer"] = t_dev1 - t_dev0

    return timings


def _time_dataloader(dataset, n_batches: int) -> dict[str, float]:
    """Time a DataLoader end-to-end including __getitem__ blocking time."""
    from torch.utils.data import DataLoader  # noqa: PLC0415 - local import to keep main() lean

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    total = 0.0
    batches = 0
    for _batch_idx, (images, labels) in enumerate(loader):
        if batches >= n_batches:
            break
        t0 = time.perf_counter()
        _ = images.to(DEVICE)
        _ = labels.to(DEVICE)
        total += time.perf_counter() - t0
        batches += 1

    return {"total_epoch_time": total, "batches": batches}


def main() -> int:
    if not PROCESSED_DIR.is_dir():
        print(f"ERROR: {PROCESSED_DIR} not found. Run scripts/preprocess_images.py first.")
        return 1

    # Pick the first N_SAMPLES images from a CSV if available, else scan dir.
    csv_path = Path("data/raw/train.csv")
    if csv_path.exists():
        import pandas as pd  # noqa: PLC0415 - local import to keep main() lean

        df = pd.read_csv(csv_path)
        id_codes = [str(k) for k in df["id_code"].head(N_SAMPLES)]
        label_map = {str(k): int(v) for k, v in zip(df["id_code"], df["diagnosis"], strict=True)}
    else:
        id_codes = sorted(p.stem for p in PROCESSED_DIR.glob("*.png"))[:N_SAMPLES]
        label_map = dict.fromkeys(id_codes, 0)

    print(f"Profiling {len(id_codes)} images on device: {DEVICE}")
    print(f"Processed dir: {PROCESSED_DIR}")
    print()

    # Train path (with augmentation).
    print("=== Train path (with augmentation) ===")
    train_timings = _time_getitem_manual(id_codes, train_transforms(), label_map)
    for key, val in train_timings.items():
        print(f"  {key:20s}: {val:.4f}s  (per image: {val / len(id_codes) * 1000:.1f} ms)")
    print()

    # Val path (no augmentation).
    print("=== Val path (no augmentation) ===")
    val_timings = _time_getitem_manual(id_codes, None, label_map)
    for key, val in val_timings.items():
        print(f"  {key:20s}: {val:.4f}s  (per image: {val / len(id_codes) * 1000:.1f} ms)")
    print()

    # Difference.
    print("=== Train - Val (augmentation overhead) ===")
    for key in train_timings:
        diff = train_timings[key] - val_timings[key]
        print(f"  {key:20s}: {diff:+.4f}s  (per image: {diff / len(id_codes) * 1000:+.1f} ms)")
    print()

    # DataLoader end-to-end timing via the actual dataset classes.
    print("=== DataLoader end-to-end (num_workers=0) ===")
    config = Config(
        image_size=384,
        batch_size=16,
        epochs=1,
        lr=0.0003,
        model_name="efficientnet_b0",
        seed=42,
        data_dir="data/raw",
        processed_dir=str(PROCESSED_DIR),
        checkpoint_dir="checkpoints",
        mlflow_dir="mlruns",
        num_workers=0,
    )

    from lumoculis.data import train_val_split  # noqa: PLC0415 - local import to keep main() lean

    train_ds, val_ds, _ = train_val_split(
        csv_path if csv_path.exists() else PROCESSED_DIR / "train.csv",
        PROCESSED_DIR,
        config,
        val_size=0.2,
    )

    train_dl = _time_dataloader(train_ds, n_batches=3)
    val_dl = _time_dataloader(val_ds, n_batches=3)

    print(
        f"  Train: {train_dl['total_epoch_time']:.4f}s for {train_dl['batches']} batches "
        f"({train_dl['total_epoch_time'] / train_dl['batches'] * 1000:.1f} ms/batch)"
    )
    print(
        f"  Val:   {val_dl['total_epoch_time']:.4f}s for {val_dl['batches']} batches "
        f"({val_dl['total_epoch_time'] / val_dl['batches'] * 1000:.1f} ms/batch)"
    )
    print()

    print("=== Recommendations ===")
    print("  - If disk_read dominates: images are large or disk is slow; consider")
    print("    caching decoded tensors to disk as .pt files.")
    print("  - If transform dominates: augmentation is expensive; reduce rotation")
    print("    degrees or disable ColorJitter for smoke tests.")
    print("  - If device_transfer dominates: MPS synchronization issue; consider")
    print("    running on CPU for smoke tests (--no-pretrained still works).")
    print("  - num_workers=0 means all loading is sequential; try num_workers=2")
    print("    in configs/baseline.yaml for full runs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
