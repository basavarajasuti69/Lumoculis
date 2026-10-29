#!/usr/bin/env python3
"""Controlled CPU vs MPS benchmark for Lumoculis EfficientNet-B0.

This script does NOT use the production training CLI, does NOT process the
APTOS dataset, does NOT download pretrained weights, does NOT save checkpoints
or MLflow artifacts, and does NOT modify production files.

It uses the existing model factory (lumoculis.model.build_model) with random
initialization and synthetic tensors to isolate compute from I/O/startup.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

import torch
from torch import nn

from lumoculis.model import build_model


def _sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_bench_for_device(  # noqa: PLR0912, PLR0915, PLR0913
    device: torch.device,
    batch_size: int,
    train_steps: int,
    val_steps: int,
    num_classes: int,
    image_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "train_steps": train_steps,
        "val_steps": val_steps,
        "num_classes": num_classes,
        "image_size": image_size,
        "errors": [],
    }

    try:
        gc.collect()
        _clear_cuda()
        if device.type == "mps":
            gc.collect()

        # Model initialization (random init, no pretrained weights)
        t0 = time.perf_counter()
        model = build_model("efficientnet_b0", num_classes=num_classes, pretrained=False)
        model = model.to(device)
        if device.type == "mps":
            _sync_mps()
        t_model_init = time.perf_counter() - t0
        result["model_init_s"] = t_model_init

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Generate synthetic tensors on CPU (avoid including generation in compute)
        # Keep them on CPU until transfer timing
        x_cpu = torch.randn(batch_size, 3, image_size, image_size)
        y_cpu = torch.randint(0, num_classes, (batch_size,))

        # Warmup
        t_warm0 = time.perf_counter()
        try:
            x_w = x_cpu.to(device)
            y_w = y_cpu.to(device)
            if device.type == "mps":
                _sync_mps()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits_w = model(x_w)
            if device.type == "mps":
                _sync_mps()
            loss_w = criterion(logits_w, y_w)
            if device.type == "mps":
                _sync_mps()
            loss_w.backward()
            if device.type == "mps":
                _sync_mps()
            optimizer.step()
            if device.type == "mps":
                _sync_mps()
        except Exception as e:
            result["errors"].append(f"warmup_error: {e}")
            raise
        t_warm1 = time.perf_counter()
        result["warmup_s"] = t_warm1 - t_warm0

        # Clear warmup tensors
        del x_w, y_w, logits_w, loss_w
        gc.collect()
        if device.type == "mps":
            gc.collect()

        train_step_times = []
        forward_times = []
        loss_times = []
        backward_times = []
        opt_times = []
        transfer_times = []

        model.train()
        for _step in range(train_steps):
            step_t0 = time.perf_counter()
            # Transfer
            t_tr0 = time.perf_counter()
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            if device.type == "mps":
                _sync_mps()
            t_tr1 = time.perf_counter()
            transfer_times.append(t_tr1 - t_tr0)

            # Forward
            t_f0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if device.type == "mps":
                _sync_mps()
            t_f1 = time.perf_counter()
            forward_times.append(t_f1 - t_f0)

            # Loss
            t_l0 = time.perf_counter()
            loss = criterion(logits, y)
            if device.type == "mps":
                _sync_mps()
            t_l1 = time.perf_counter()
            loss_times.append(t_l1 - t_l0)

            # Backward
            t_b0 = time.perf_counter()
            loss.backward()
            if device.type == "mps":
                _sync_mps()
            t_b1 = time.perf_counter()
            backward_times.append(t_b1 - t_b0)

            # Optimizer
            t_o0 = time.perf_counter()
            optimizer.step()
            if device.type == "mps":
                _sync_mps()
            t_o1 = time.perf_counter()
            opt_times.append(t_o1 - t_o0)

            step_t1 = time.perf_counter()
            train_step_times.append(step_t1 - step_t0)

            del x, y, logits, loss

        # Validation
        val_forward_times = []
        val_transfer_times = []
        model.eval()
        with torch.no_grad():
            for _step in range(val_steps):
                t_vtr0 = time.perf_counter()
                x_v = x_cpu.to(device)
                y_v = y_cpu.to(device)
                if device.type == "mps":
                    _sync_mps()
                t_vtr1 = time.perf_counter()
                val_transfer_times.append(t_vtr1 - t_vtr0)

                t_vf0 = time.perf_counter()
                logits_v = model(x_v)
                if device.type == "mps":
                    _sync_mps()
                t_vf1 = time.perf_counter()
                val_forward_times.append(t_vf1 - t_vf0)

                del x_v, y_v, logits_v

        result["train_step_times"] = train_step_times
        result["avg_train_step_s"] = (
            sum(train_step_times) / len(train_step_times) if train_step_times else 0.0
        )
        result["transfer_times"] = transfer_times
        result["avg_transfer_s"] = (
            sum(transfer_times) / len(transfer_times) if transfer_times else 0.0
        )
        result["forward_times"] = forward_times
        result["avg_forward_s"] = sum(forward_times) / len(forward_times) if forward_times else 0.0
        result["loss_times"] = loss_times
        result["avg_loss_s"] = sum(loss_times) / len(loss_times) if loss_times else 0.0
        result["backward_times"] = backward_times
        result["avg_backward_s"] = (
            sum(backward_times) / len(backward_times) if backward_times else 0.0
        )
        result["opt_times"] = opt_times
        result["avg_opt_s"] = sum(opt_times) / len(opt_times) if opt_times else 0.0
        result["val_transfer_times"] = val_transfer_times
        result["avg_val_transfer_s"] = (
            sum(val_transfer_times) / len(val_transfer_times) if val_transfer_times else 0.0
        )
        result["val_forward_times"] = val_forward_times
        result["avg_val_forward_s"] = (
            sum(val_forward_times) / len(val_forward_times) if val_forward_times else 0.0
        )

        # Cleanup
        del model, criterion, optimizer, x_cpu, y_cpu
        gc.collect()
        if device.type == "mps":
            gc.collect()
        _clear_cuda()

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or (
            "mps" in str(e).lower() and "memory" in str(e).lower()
        ):
            result["errors"].append(f"OOM_OR_MEMORY_ERROR: {e}")
        else:
            result["errors"].append(f"RUNTIME_ERROR: {e}")
        gc.collect()
        if device.type == "mps":
            gc.collect()
        _clear_cuda()
    except Exception as e:
        result["errors"].append(f"ERROR: {e}")
        gc.collect()
        if device.type == "mps":
            gc.collect()
        _clear_cuda()

    return result


def main() -> int:  # noqa: PLR0915, PLR0912
    parser = argparse.ArgumentParser(description="CPU vs MPS benchmark for Lumoculis")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10, help="Training steps per device")
    parser.add_argument("--val-steps", type=int, default=5, help="Validation steps per device")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    print("=== Lumoculis CPU vs MPS Benchmark (synthetic, no dataset/pretrained/checkpoints) ===")
    print(
        f"Batch size: {args.batch_size}, image: {args.image_size}x{args.image_size}, classes: {args.num_classes}"
    )
    print(f"Train steps: {args.steps}, Val steps: {args.val_steps}")
    print()

    results = []

    # CPU
    print("Running on CPU...")
    cpu_res = _run_bench_for_device(
        torch.device("cpu"),
        args.batch_size,
        args.steps,
        args.val_steps,
        args.num_classes,
        args.image_size,
    )
    results.append(cpu_res)
    print("CPU done.")
    print()

    # MPS
    if torch.backends.mps.is_available():
        print("Running on MPS...")
        try:
            mps_res = _run_bench_for_device(
                torch.device("mps"),
                args.batch_size,
                args.steps,
                args.val_steps,
                args.num_classes,
                args.image_size,
            )
            results.append(mps_res)
            print("MPS done.")
        except Exception as e:
            print(f"MPS benchmark failed: {e}")
            results.append({"device": "mps", "errors": [str(e)]})
    else:
        print("MPS not available, skipping.")
        results.append({"device": "mps", "skipped": True, "reason": "MPS not available"})

    print()
    print("=== Results ===")
    for r in results:
        print(f"\nDevice: {r['device']}")
        if r.get("skipped"):
            print(f"  Skipped: {r.get('reason')}")
            continue
        if r.get("errors"):
            print(f"  Errors: {r['errors']}")
            # Still print what we have if partial
        if "model_init_s" in r:
            print(f"  Model init: {r['model_init_s']:.4f}s")
        if "warmup_s" in r:
            print(f"  Warmup (separate): {r['warmup_s']:.4f}s")
        if "avg_transfer_s" in r:
            print(f"  Avg input transfer (train): {r['avg_transfer_s']:.4f}s")
        if "avg_forward_s" in r:
            print(f"  Avg forward (train): {r['avg_forward_s']:.4f}s")
        if "avg_loss_s" in r:
            print(f"  Avg loss (train): {r['avg_loss_s']:.4f}s")
        if "avg_backward_s" in r:
            print(f"  Avg backward (train): {r['avg_backward_s']:.4f}s")
        if "avg_opt_s" in r:
            print(f"  Avg optimizer step (train): {r['avg_opt_s']:.4f}s")
        if "avg_train_step_s" in r:
            print(f"  Avg steady-state train step: {r['avg_train_step_s']:.4f}s")
        if "avg_val_transfer_s" in r:
            print(f"  Avg input transfer (val): {r['avg_val_transfer_s']:.4f}s")
        if "avg_val_forward_s" in r:
            print(f"  Avg forward (val): {r['avg_val_forward_s']:.4f}s")
        if "train_step_times" in r:
            print(f"  Per-step train times: {[f'{t:.4f}' for t in r['train_step_times']]}")
        if "val_forward_times" in r:
            print(f"  Per-step val forward times: {[f'{t:.4f}' for t in r['val_forward_times']]}")

    print("\n=== Notes ===")
    print(
        "- Synthetic tensors only; excludes dataset hashing, image loading, MLflow, checkpoint saves"
    )
    print("- MPS synced before/after timed ops where applicable")
    print("- Warmup excluded from steady-state averages")
    print("- Tensor generation not included in compute timings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
