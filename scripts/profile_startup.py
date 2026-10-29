#!/usr/bin/env python3
"""Profile startup stages before first training batch. Does not modify production code."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumoculis.config import Config
from lumoculis.data import train_val_split
from lumoculis.train import _build_dataloaders, _setup_training, pick_device, seed_everything


def main() -> int:
    timings = {}
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    config = Config.from_yaml("configs/baseline.yaml")
    timings["config_load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    device = pick_device()
    seed_everything(config.seed)
    try:
        mlflow.end_run()
    except Exception:
        pass
    timings["device_seed_mlflow_init_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    csv_path = Path(config.data_dir) / "train.csv"
    train_ds, val_ds, dup_report = train_val_split(csv_path, config.processed_dir, config)
    timings["dataset_split_hashing_s"] = time.perf_counter() - t0
    timings["dup_groups"] = dup_report.get("n_dup_groups")
    timings["dup_images"] = dup_report.get("n_dup_images")

    t0 = time.perf_counter()
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, config)
    timings["dataloader_init_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    epochs = config.epochs
    model, criterion, optimizer, scheduler, class_weights = _setup_training(
        config, device, train_ds, pretrained=False, epochs=epochs
    )
    if device.type == "mps":
        torch.mps.synchronize()
    timings["model_opt_loss_init_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        first_batch = next(iter(train_loader))
        images_fb, labels_fb = first_batch
        images_fb = images_fb.to(device)
        labels_fb = labels_fb.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits_fb = model(images_fb)
        loss_fb = criterion(logits_fb, labels_fb)
        loss_fb.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        timings["first_batch_warmup_s"] = time.perf_counter() - t0
        del images_fb, labels_fb, logits_fb, loss_fb
    except Exception as e:
        timings["first_batch_warmup_s"] = time.perf_counter() - t0
        timings["first_batch_warmup_err"] = str(e)

    t0 = time.perf_counter()
    try:
        it = iter(train_loader)
        batch = next(it)
        images_b, labels_b = batch
        images_b = images_b.to(device)
        labels_b = labels_b.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        t_data = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits_b = model(images_b)
        if device.type == "mps":
            torch.mps.synchronize()
        t_fwd = time.perf_counter()
        loss_b = criterion(logits_b, labels_b)
        if device.type == "mps":
            torch.mps.synchronize()
        t_loss = time.perf_counter()
        loss_b.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        t_bwd = time.perf_counter()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        t_opt = time.perf_counter()
        timings["first_train_batch_total_s"] = t_opt - t0
        timings["first_train_batch_transfer_s"] = t_data - t0
        timings["first_train_batch_forward_s"] = t_fwd - t_data
        timings["first_train_batch_loss_s"] = t_loss - t_fwd
        timings["first_train_batch_backward_s"] = t_bwd - t_loss
        timings["first_train_batch_optimizer_s"] = t_opt - t_bwd
        del images_b, labels_b, logits_b, loss_b
    except Exception as e:
        timings["first_train_batch_total_s"] = time.perf_counter() - t0
        timings["first_train_batch_err"] = str(e)

    t0 = time.perf_counter()
    try:
        vit = iter(val_loader)
        vbatch = next(vit)
        vimages, vlabels = vbatch
        vimages = vimages.to(device)
        vlabels = vlabels.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        model.eval()
        with torch.no_grad():
            vlogits = model(vimages)
            if device.type == "mps":
                torch.mps.synchronize()
        timings["first_val_batch_total_s"] = time.perf_counter() - t0
        del vimages, vlabels, vlogits
    except Exception as e:
        timings["first_val_batch_total_s"] = time.perf_counter() - t0
        timings["first_val_batch_err"] = str(e)

    timings["total_init_before_first_train_batch_s"] = time.perf_counter() - t_total_start

    for k in sorted(timings.keys()):
        print(f"{k}: {timings[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
