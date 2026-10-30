#!/usr/bin/env python3
"""Diagnostic for MPS slowness - read only, small tests, timeouts."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TIMEOUT_SEC = 300


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_op(name, fn, timeout_sec=TIMEOUT_SEC):
    print(f"START: {name}", flush=True)
    t0 = time.perf_counter()
    try:
        res = fn()
        if torch.backends.mps.is_available():
            _sync_mps()
        t1 = time.perf_counter()
        print(f"DONE: {name} in {t1 - t0:.3f}s", flush=True)
        return res, t1 - t0, None
    except Exception as e:
        t1 = time.perf_counter()
        print(f"ERR: {name} after {t1 - t0:.3f}s: {e}", flush=True)
        return None, t1 - t0, e


def main() -> int:  # noqa: PLR0915
    print("=== MPS Diagnostic ===", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)
    print(f"MPS built: {torch.backends.mps.is_built()}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    try:
        import psutil  # noqa: PLC0415

        p = psutil.Process()
        print(f"Process RSS: {p.memory_info().rss / 1024**3:.2f} GB", flush=True)
    except Exception as e:
        print(f"Memory info unavailable: {e}", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Basic tensor ops
    results = {}

    def alloc_small():
        x = torch.randn(1000, 1000)
        return x

    _, t, e = timed_op("alloc_cpu_small", alloc_small)
    results["alloc_cpu_small"] = (t, e)

    if device.type == "mps":

        def alloc_mps():
            x = torch.randn(1000, 1000).to(device)
            return x

        _, t, e = timed_op("alloc_mps_small", alloc_mps)
        results["alloc_mps_small"] = (t, e)

        def elemwise():
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = x + y * 2.0
            return z

        _, t, e = timed_op("elemwise_mps", elemwise)
        results["elemwise_mps"] = (t, e)

        def matmul():
            x = torch.randn(512, 512).to(device)
            y = torch.randn(512, 512).to(device)
            z = torch.matmul(x, y)
            return z

        _, t, e = timed_op("matmul_mps_512", matmul)
        results["matmul_mps_512"] = (t, e)

    # Small NN
    print("\n=== Small NN test ===", flush=True)
    if device.type == "mps":

        def small_nn():
            model = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            x = torch.randn(32, 1024).to(device)
            y = torch.randint(0, 10, (32,)).to(device)
            crit = nn.CrossEntropyLoss()
            # first
            model.train()
            opt.zero_grad()
            out = model(x)
            _sync_mps()
            loss = crit(out, y)
            _sync_mps()
            loss.backward()
            _sync_mps()
            opt.step()
            _sync_mps()
            return model

        model_res, t, e = timed_op("small_nn_first", small_nn)
        results["small_nn_first"] = (t, e)

        if model_res is not None and e is None:

            def small_nn_sub():
                model = model_res
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                x = torch.randn(32, 1024).to(device)
                y = torch.randint(0, 10, (32,)).to(device)
                crit = nn.CrossEntropyLoss()
                model.train()
                opt.zero_grad()
                out = model(x)
                _sync_mps()
                loss = crit(out, y)
                _sync_mps()
                loss.backward()
                _sync_mps()
                opt.step()
                _sync_mps()

            _, t, e = timed_op("small_nn_subsequent", small_nn_sub)
            results["small_nn_subsequent"] = (t, e)

    # EfficientNet-B0 if safe
    print("\n=== EfficientNet-B0 test ===", flush=True)
    if device.type == "mps":
        # Check if previous tests were reasonable
        if (
            results.get("small_nn_first", (999,))[0] < 60
            and results.get("matmul_mps_512", (999,))[0] < 30
        ):
            from lumoculis.model import build_model  # noqa: PLC0415

            def effnet_test():
                model = build_model("efficientnet_b0", 5, pretrained=False).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                crit = nn.CrossEntropyLoss()
                x = torch.randn(1, 3, 384, 384).to(device)  # batch size 1
                y = torch.randint(0, 5, (1,)).to(device)
                model.train()
                opt.zero_grad()
                out = model(x)
                _sync_mps()
                loss = crit(out, y)
                _sync_mps()
                loss.backward()
                _sync_mps()
                opt.step()
                _sync_mps()
                return model

            model_res, t, e = timed_op("effnet_b0_b1_first", effnet_test)
            results["effnet_b0_b1_first"] = (t, e)

            if model_res is not None and e is None:

                def effnet_sub():
                    model = model_res
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    crit = nn.CrossEntropyLoss()
                    x = torch.randn(1, 3, 384, 384).to(device)
                    y = torch.randint(0, 5, (1,)).to(device)
                    model.train()
                    opt.zero_grad()
                    out = model(x)
                    _sync_mps()
                    loss = crit(out, y)
                    _sync_mps()
                    loss.backward()
                    _sync_mps()
                    opt.step()
                    _sync_mps()

                _, t, e = timed_op("effnet_b0_b1_sub", effnet_sub)
                results["effnet_b0_b1_sub"] = (t, e)
        else:
            print("SKIPPING EfficientNet-B0 due to slow earlier tests", flush=True)
            results["effnet_skipped"] = (0, "slow earlier tests")

    print("\n=== Summary ===", flush=True)
    for k, v in results.items():
        print(f"{k}: {v[0]:.3f}s, err={v[1]}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
