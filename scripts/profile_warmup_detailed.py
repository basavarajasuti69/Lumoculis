#!/usr/bin/env python3
"""Detailed warmup profiling - diagnostic only, no production changes."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumoculis.config import Config
from lumoculis.data import train_val_split
from lumoculis.model import build_model
from lumoculis.train import _build_dataloaders, pick_device, seed_everything


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def main() -> int:  # noqa: PLR0915, PLR0912
    print("Starting warmup detailed profiling...", flush=True)
    timings = {}

    config = Config.from_yaml("configs/baseline.yaml")
    device = pick_device()
    seed_everything(config.seed)
    print(f"Device: {device}", flush=True)

    # Setup dataset/loaders
    print("Loading dataset...", flush=True)
    t_ds0 = time.perf_counter()
    csv_path = Path(config.data_dir) / "train.csv"
    train_ds, val_ds, dup_report = train_val_split(csv_path, config.processed_dir, config)
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, config)
    timings["dataset_setup_s"] = time.perf_counter() - t_ds0
    print(f"Dataset setup: {timings['dataset_setup_s']:.3f}s", flush=True)

    # Model
    print("Building model...", flush=True)
    t_m0 = time.perf_counter()
    model = build_model(config.model_name, config.num_classes, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if device.type == "mps":
        _sync_mps()
    timings["model_setup_s"] = time.perf_counter() - t_m0
    print(f"Model setup: {timings['model_setup_s']:.3f}s", flush=True)

    # Get first batch
    print("Retrieving first batch from DataLoader...", flush=True)
    t_fb0 = time.perf_counter()
    it1 = iter(train_loader)
    batch1 = next(it1)
    images1, labels1 = batch1
    t_fb1 = time.perf_counter()
    timings["first_batch_retrieval_s"] = t_fb1 - t_fb0
    print(f"First batch retrieval: {timings['first_batch_retrieval_s']:.3f}s", flush=True)

    # Transfer first batch
    print("Transferring first batch to device...", flush=True)
    t_ft0 = time.perf_counter()
    images1_t = images1.to(device)
    labels1_t = labels1.to(device)
    if device.type == "mps":
        _sync_mps()
    t_ft1 = time.perf_counter()
    timings["first_batch_transfer_s"] = t_ft1 - t_ft0
    print(f"First batch transfer: {timings['first_batch_transfer_s']:.3f}s", flush=True)

    # First forward
    print("First forward pass...", flush=True)
    t_ff0 = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits1 = model(images1_t)
    if device.type == "mps":
        _sync_mps()
    t_ff1 = time.perf_counter()
    timings["first_forward_s"] = t_ff1 - t_ff0
    print(f"First forward: {timings['first_forward_s']:.3f}s", flush=True)

    # Loss
    t_fl0 = time.perf_counter()
    loss1 = criterion(logits1, labels1_t)
    if device.type == "mps":
        _sync_mps()
    t_fl1 = time.perf_counter()
    timings["first_loss_s"] = t_fl1 - t_fl0
    print(f"First loss: {timings['first_loss_s']:.3f}s", flush=True)

    # Backward
    print("First backward pass...", flush=True)
    t_fbwd0 = time.perf_counter()
    loss1.backward()
    if device.type == "mps":
        _sync_mps()
    t_fbwd1 = time.perf_counter()
    timings["first_backward_s"] = t_fbwd1 - t_fbwd0
    print(f"First backward: {timings['first_backward_s']:.3f}s", flush=True)

    # Optimizer step
    print("First optimizer step...", flush=True)
    t_fopt0 = time.perf_counter()
    optimizer.step()
    if device.type == "mps":
        _sync_mps()
    t_fopt1 = time.perf_counter()
    timings["first_optimizer_step_s"] = t_fopt1 - t_fopt0
    print(f"First optimizer step: {timings['first_optimizer_step_s']:.3f}s", flush=True)

    timings["first_iteration_total_s"] = time.perf_counter() - t_fb0
    print(
        f"First iteration total (from retrieval): {timings['first_iteration_total_s']:.3f}s",
        flush=True,
    )

    del images1, labels1, images1_t, labels1_t, logits1, loss1

    # Subsequent batch
    print("\nRetrieving subsequent batch...", flush=True)
    t_sb0 = time.perf_counter()
    batch2 = next(it1)
    images2, labels2 = batch2
    t_sb1 = time.perf_counter()
    timings["subsequent_batch_retrieval_s"] = t_sb1 - t_sb0
    print(f"Subsequent batch retrieval: {timings['subsequent_batch_retrieval_s']:.3f}s", flush=True)

    t_st0 = time.perf_counter()
    images2_t = images2.to(device)
    labels2_t = labels2.to(device)
    if device.type == "mps":
        _sync_mps()
    t_st1 = time.perf_counter()
    timings["subsequent_batch_transfer_s"] = t_st1 - t_st0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_sf0 = time.perf_counter()
    logits2 = model(images2_t)
    if device.type == "mps":
        _sync_mps()
    t_sf1 = time.perf_counter()
    timings["subsequent_forward_s"] = t_sf1 - t_sf0

    t_sl0 = time.perf_counter()
    loss2 = criterion(logits2, labels2_t)
    if device.type == "mps":
        _sync_mps()
    t_sl1 = time.perf_counter()
    timings["subsequent_loss_s"] = t_sl1 - t_sl0

    t_sbwd0 = time.perf_counter()
    loss2.backward()
    if device.type == "mps":
        _sync_mps()
    t_sbwd1 = time.perf_counter()
    timings["subsequent_backward_s"] = t_sbwd1 - t_sbwd0

    t_sopt0 = time.perf_counter()
    optimizer.step()
    if device.type == "mps":
        _sync_mps()
    t_sopt1 = time.perf_counter()
    timings["subsequent_optimizer_step_s"] = t_sopt1 - t_sopt0
    timings["subsequent_iteration_total_s"] = time.perf_counter() - t_sb0

    print(
        f"Subsequent: retrieval={timings['subsequent_batch_retrieval_s']:.3f}s transfer={timings['subsequent_batch_transfer_s']:.3f}s fwd={timings['subsequent_forward_s']:.3f}s loss={timings['subsequent_loss_s']:.3f}s bwd={timings['subsequent_backward_s']:.3f}s opt={timings['subsequent_optimizer_step_s']:.3f}s total={timings['subsequent_iteration_total_s']:.3f}s",
        flush=True,
    )

    del images2, labels2, images2_t, labels2_t, logits2, loss2

    # Synthetic comparison
    print("\n=== Synthetic input comparison ===", flush=True)
    try:
        x_syn = torch.randn(config.batch_size, 3, config.image_size, config.image_size)
        y_syn = torch.randint(0, config.num_classes, (config.batch_size,))
        t_syn0 = time.perf_counter()
        x_syn_t = x_syn.to(device)
        y_syn_t = y_syn.to(device)
        if device.type == "mps":
            _sync_mps()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits_syn = model(x_syn_t)
        if device.type == "mps":
            _sync_mps()
        loss_syn = criterion(logits_syn, y_syn_t)
        if device.type == "mps":
            _sync_mps()
        loss_syn.backward()
        if device.type == "mps":
            _sync_mps()
        optimizer.step()
        if device.type == "mps":
            _sync_mps()
        timings["synthetic_step_total_s"] = time.perf_counter() - t_syn0
        print(f"Synthetic step total: {timings['synthetic_step_total_s']:.3f}s", flush=True)
        del x_syn, y_syn, x_syn_t, y_syn_t, logits_syn, loss_syn
    except Exception as e:
        timings["synthetic_step_err"] = str(e)
        print(f"Synthetic err: {e}", flush=True)

    print("\n=== Summary ===", flush=True)
    for k in sorted(timings.keys()):
        print(f"{k}: {timings[k]}")

    print(
        "\nNote: Timings with MPS syncs include synchronization overhead; not all syncs necessary for correctness.",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
