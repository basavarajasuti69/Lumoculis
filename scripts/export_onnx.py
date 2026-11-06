"""Export the Lumoculis EfficientNet-B0 checkpoint to ONNX.

This is a development/export utility. It does not modify the checkpoint or
production inference code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lumoculis.inference import _load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Lumoculis checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="checkpoints/expC_weighted_sampling_20261101_172044/best.pt")
    parser.add_argument("--output", default="models/best.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model, metadata = _load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 384, 384)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {output} ({output.stat().st_size} bytes)")
    print(f"Checkpoint metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
