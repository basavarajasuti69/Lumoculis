"""Command-line interface for retinal image-quality assessment.

Evaluates one or more fundus images for technical quality (resolution,
exposure, blur, field of view) and writes a JSON report.

Usage:

    lumoculis-quality --image data/processed/001.png --output report.json
    lumoculis-quality --dir data/processed/ --recursive --output report.json
    lumoculis-quality --dir data/processed/ --min-width 300 --min-height 300

Does not make disease predictions. Results are separate from grading model
outputs. Images with status "fail" or "warning" should not be passed to the
grading model without review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumoculis.quality import (
    QualityConfig,
    assess_quality,
    process_directory,
    summarize,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess retinal image quality for Lumoculis screening support."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        help="Path to a single image file to assess.",
    )
    source.add_argument(
        "--dir",
        help="Path to a directory of images to assess.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when --dir is used.",
    )
    parser.add_argument(
        "--output",
        help="Path to write the JSON report. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        help="Minimum acceptable image width (pixels).",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        help="Minimum acceptable image height (pixels).",
    )
    parser.add_argument(
        "--blur-min",
        type=float,
        help="Minimum Laplacian variance for blur check.",
    )
    parser.add_argument(
        "--fov-min",
        type=float,
        help="Minimum retinal content fraction (0-1).",
    )
    parser.add_argument(
        "--no-resolution-hard-failure",
        action="store_true",
        help="Treat resolution below minimum as a warning, not a hard failure.",
    )

    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> QualityConfig:
    """Build a QualityConfig from CLI arguments, overriding only provided values."""
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
    return QualityConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _build_config(args)

    if args.image is not None:
        result = assess_quality(args.image, config=config)
        report = summarize([result])
    else:
        assert args.dir is not None
        results = process_directory(args.dir, config=config, recursive=args.recursive)
        report = summarize(results)

    report_json = json.dumps(report, indent=2)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json, encoding="utf-8")
        print(f"Wrote quality report to {output_path}")
    else:
        print(report_json)

    # Exit non-zero if any image failed, so this can be used in CI/scripts.
    if report["fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
