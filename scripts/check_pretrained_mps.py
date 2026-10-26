"""Manual check: download pretrained EfficientNet-B0 weights and run a forward pass on MPS.

Run: uv run python scripts/check_pretrained_mps.py

This confirms timm can fetch pretrained weights and that the model runs on the
Apple GPU (mps). Falls back to CPU if MPS is unavailable.
"""

from __future__ import annotations

import torch

from lumoculis.model import build_model, count_parameters


def main() -> int:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=True)
    model.to(device)
    model.eval()

    print(f"Trainable parameters: {count_parameters(model):,}")

    with torch.no_grad():
        x = torch.randn(1, 3, 384, 384, device=device)
        logits = model(x)
    print(f"Forward pass logits shape: {tuple(logits.shape)} on {logits.device}")
    print("Pretrained weights downloaded and forward pass OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
