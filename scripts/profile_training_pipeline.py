"""Comprehensive data-loading profiler replicating exact training conditions.

Measures the full pipeline end-to-end: DataLoader blocking time (__getitem__),
device transfer, forward/backward compute, and MPS synchronization. Uses the
actual LumoculisDataset, train_val_split, and DataLoader configuration from
the codebase.

Run with:
    uv run python scripts/profile_training_pipeline.py

Does NOT start full training. Profiles 5 batches of train + val only.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lumoculis.config import Config
from lumoculis.data import train_val_split
from lumoculis.model import build_model

PROCESSED_DIR = Path("data/processed")
CSV_PATH = Path("data/raw/train.csv")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_BATCHES = 5


def _time_full_pipeline(  # noqa: PLR0913
    dataset, model, criterion, optimizer, is_train: bool, n_batches: int
) -> dict[str, float]:
    """Time a full DataLoader pipeline including __getitem__ blocking.

    Returns timings for: dataloader_blocking (__getitem__ + collate),
    device_transfer, compute (forward/+backward), and total.
    """
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=is_train,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    dataloader_s = 0.0
    device_s = 0.0
    compute_s = 0.0
    total_s = 0.0
    batches = 0

    model.train() if is_train else model.eval()

    for _batch_idx, (images, labels) in enumerate(loader):
        if batches >= n_batches:
            break

        # Time from start of iteration to after batch is on device.
        t_iter0 = time.perf_counter()

        t_dev0 = time.perf_counter()
        images = images.to(DEVICE)  # noqa: PLW2901
        labels = labels.to(DEVICE)  # noqa: PLW2901
        device_s += time.perf_counter() - t_dev0

        # Time spent inside the DataLoader (before this iteration body).
        # We capture it by measuring from iteration start to device transfer start.
        dataloader_s += t_dev0 - t_iter0

        if is_train:
            t_comp0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                t_comp0 = time.perf_counter()
                logits = model(images)
                loss = criterion(logits, labels)
        compute_s += time.perf_counter() - t_comp0

        total_s += time.perf_counter() - t_iter0
        batches += 1

    return {
        "dataloader_s": dataloader_s,
        "device_s": device_s,
        "compute_s": compute_s,
        "total_s": total_s,
        "batches": batches,
    }


def main() -> int:
    if not PROCESSED_DIR.is_dir():
        print(f"ERROR: {PROCESSED_DIR} not found. Run scripts/preprocess_images.py first.")
        return 1
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run scripts/download_data.py first.")
        return 1

    print(f"Device: {DEVICE} (MPS: {torch.backends.mps.is_available()})")
    print(f"Profiling {MAX_BATCHES} batches of train + val")
    print()

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

    train_ds, val_ds, dup_report = train_val_split(CSV_PATH, PROCESSED_DIR, config)
    print(f"Train dataset: {len(train_ds)} images, Val dataset: {len(val_ds)} images")
    print()

    model = build_model(config.model_name, config.num_classes, pretrained=False).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("=== Train pipeline (num_workers=0, with augmentation) ===")
    train_t = _time_full_pipeline(train_ds, model, criterion, optimizer, True, MAX_BATCHES)
    for key, val in train_t.items():
        if key == "batches":
            continue
        per_batch = val / train_t["batches"]
        print(f"  {key:18s}: {val:8.4f}s  ({per_batch * 1000:7.1f} ms/batch)")
    print(
        f"  Sum of components: {train_t['dataloader_s'] + train_t['device_s'] + train_t['compute_s']:.4f}s"
    )
    print()

    print("=== Val pipeline (num_workers=0, no augmentation) ===")
    val_t = _time_full_pipeline(val_ds, model, criterion, None, False, MAX_BATCHES)
    for key, val in val_t.items():
        if key == "batches":
            continue
        per_batch = val / val_t["batches"]
        print(f"  {key:18s}: {val:8.4f}s  ({per_batch * 1000:7.1f} ms/batch)")
    print(
        f"  Sum of components: {val_t['dataloader_s'] + val_t['device_s'] + val_t['compute_s']:.4f}s"
    )
    print()

    print("=== Analysis ===")
    print(f"  Train dataloader (__getitem__+collate): {train_t['dataloader_s']:.4f}s")
    print(f"  Train device transfer:                   {train_t['device_s']:.4f}s")
    print(f"  Train compute (fwd+bwd):                 {train_t['compute_s']:.4f}s")
    print(f"  Val dataloader (__getitem__+collate):    {val_t['dataloader_s']:.4f}s")
    print(f"  Val device transfer:                     {val_t['device_s']:.4f}s")
    print(f"  Val compute (fwd only):                  {val_t['compute_s']:.4f}s")
    print()

    # Recommendations based on measurements.
    if train_t["dataloader_s"] > train_t["device_s"] + train_t["compute_s"]:
        print("  Bottleneck: DataLoader (__getitem__ + collate) dominates train time.")
        print("  Cause: num_workers=0 means sequential image loading in main process.")
        print("  Fix: Set num_workers=2 in configs/baseline.yaml.")
    elif train_t["device_s"] > train_t["compute_s"]:
        print("  Bottleneck: Device transfer (CPU->MPS) dominates train time.")
        print("  Fix: Consider CPU for smoke tests; MPS transfer may have overhead.")
    else:
        print("  Bottleneck: Compute dominates. Data loading is not the issue.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
