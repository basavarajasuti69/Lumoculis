"""Command-line interface for Lumoculis inference.

Runs DR grade prediction on a single retinal image using a trained checkpoint,
with optional quality gating, temperature-scaling calibration, and Grad-CAM.

Usage:

    lumoculis-infer \\
        --image data/processed/001.png \\
        --checkpoint checkpoints/run_name/best.pt \\
        --output report.json \\
        --gradcam --gradcam-output gradcam.png

    lumoculis-infer \\
        --image data/processed/001.png \\
        --checkpoint checkpoints/run_name/best.pt \\
        --run-quality-gate --quality-required \\
        --temperature calibration/temperature.json

Does not require retraining. Does not fit calibration parameters during
inference. Quality checking is optional and configurable.
"""

from __future__ import annotations

import argparse
import json
import sys

from lumoculis.inference import (
    RESEARCH_DISCLAIMER,
    InferenceConfig,
    run_inference,
)
from lumoculis.quality import QualityConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DR grade inference on a retinal fundus image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a single retinal image file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint (.pt) produced by lumoculis.train.",
    )
    parser.add_argument(
        "--output",
        help="Path to write the JSON prediction report. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Input image size for the model (default: 384).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Torch device (default: auto, prefers MPS if available).",
    )

    # Quality gate options.
    quality = parser.add_argument_group("quality gate")
    quality.add_argument(
        "--run-quality-gate",
        action="store_true",
        help="Run image-quality assessment before inference.",
    )
    quality.add_argument(
        "--quality-required",
        action="store_true",
        help="Block inference if quality check does not pass.",
    )
    quality.add_argument(
        "--min-width",
        type=int,
        help="Minimum acceptable image width (pixels).",
    )
    quality.add_argument(
        "--min-height",
        type=int,
        help="Minimum acceptable image height (pixels).",
    )
    quality.add_argument(
        "--blur-min",
        type=float,
        help="Minimum Laplacian variance for blur check.",
    )
    quality.add_argument(
        "--fov-min",
        type=float,
        help="Minimum retinal content fraction (0-1).",
    )
    quality.add_argument(
        "--no-resolution-hard-failure",
        action="store_true",
        help="Treat resolution below minimum as a warning, not a hard failure.",
    )

    # Calibration options.
    calibration = parser.add_argument_group("calibration")
    calibration.add_argument(
        "--temperature",
        help="Path to a previously fitted temperature JSON (temperature.json).",
    )

    # Grad-CAM options.
    gradcam = parser.add_argument_group("grad-cam")
    gradcam.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate a Grad-CAM visualization.",
    )
    gradcam.add_argument(
        "--gradcam-target-class",
        type=int,
        default=None,
        help="Target class for Grad-CAM (default: predicted class).",
    )
    gradcam.add_argument(
        "--gradcam-output",
        help="Path to save the Grad-CAM visualization image.",
    )

    return parser.parse_args(argv)


def _build_quality_config(args: argparse.Namespace) -> QualityConfig | None:
    """Build a QualityConfig from CLI args, or None if no overrides."""
    kwargs: dict = {}
    if args.min_width is not None:
        kwargs["min_width"] = args.min_width
    if args.min_height is not None:
        kwargs["min_height"] = args.min_height
    if args.blur_min is not None:
        kwargs["blur_laplacian_var_min"] = args.blur_min
    if args.fov_min is not None:
        kwargs["fov_min_content_fraction"] = args.fov_min
    if args.no_resolution_hard_failure:
        kwargs["resolution_hard_failure"] = False
    return QualityConfig(**kwargs) if kwargs else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    quality_config = _build_quality_config(args)

    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
        run_quality_gate=args.run_quality_gate,
        quality_required=args.quality_required,
        quality_config=quality_config,
        temperature_path=args.temperature,
        gradcam=args.gradcam,
        gradcam_target_class=args.gradcam_target_class,
        gradcam_output=args.gradcam_output,
        report_output=args.output,
    )

    try:
        report = run_inference(args.image, config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: inference failed: {exc}", file=sys.stderr)
        return 1

    report_json = report.to_dict()
    report_json["disclaimer"] = RESEARCH_DISCLAIMER

    if args.output is not None:
        print(f"Prediction report saved to {args.output}")
    else:
        print(json.dumps(report_json, indent=2))

    # Exit non-zero if quality blocked inference.
    if report.quality_blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
