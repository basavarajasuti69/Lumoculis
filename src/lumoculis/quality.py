"""Retinal image-quality gate for Lumoculis.

Heuristic image-quality checks for fundus photographs. All thresholds are
configurable; none are claimed to be clinically validated. This module does
NOT make disease predictions and does not modify training data or splits.

Hard failures (status "fail") indicate the image should not be passed to the
grading model without review. Warnings (status "warning") indicate marginal
quality that may still be usable but should be flagged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

GRAYSCALE_NDIM = 2
CHANNEL_LAST_RGB = 3


@dataclass(frozen=True)
class QualityConfig:
    """Configuration thresholds for image-quality assessment.

    All values are defaults and should be tuned per camera / dataset. None
    are clinically validated.
    """

    # Resolution (pixels). Images below either min width or min height fail
    # the resolution check (hard failure by default).
    min_width: int = 224
    min_height: int = 224

    # Underexposure: fraction of pixels below ``dark_pixel_level`` that
    # triggers the check, and the mean-intensity fallback.
    dark_pixel_level: int = 15
    dark_pixel_fraction: float = 0.85  # >85% of pixels very dark -> underexposed
    low_mean_level: int = 25  # mean intensity below this -> underexposed

    # Overexposure: fraction of pixels at or above ``saturated_level``.
    saturated_level: int = 250
    saturated_fraction: float = 0.30  # >30% saturated -> overexposed

    # Blur: variance of the Laplacian on the grayscale image. Below this is
    # considered blurry. Limitations: sensitive to image scale and content;
    # not a substitute for clinical focus assessment.
    blur_laplacian_var_min: float = 100.0

    # Retinal field of view: after cropping near-black borders, the fraction
    # of remaining pixels that are non-dark. Below this suggests little
    # visible retinal content. Dark borders alone are NOT treated as failure.
    fov_min_content_fraction: float = 0.10

    # Whether resolution below minimum is a hard failure or a warning.
    resolution_hard_failure: bool = True


@dataclass
class CheckResult:
    """Result of a single quality check."""

    name: str
    status: str  # "pass", "warning", or "fail"
    measured: Any
    threshold: Any
    reason: str
    hard_failure: bool = False


@dataclass
class QualityResult:
    """Structured result of image-quality assessment."""

    status: str  # "pass", "warning", or "fail"
    checks: list[CheckResult]
    image_width: int | None
    image_height: int | None
    decoded: bool
    reasons: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "status": self.status,
            "decoded": self.decoded,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "reasons": self.reasons,
            "thresholds": self.thresholds,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "measured": c.measured,
                    "threshold": c.threshold,
                    "reason": c.reason,
                    "hard_failure": c.hard_failure,
                }
                for c in self.checks
            ],
        }


def _status_from_checks(checks: list[CheckResult]) -> str:
    """Aggregate check statuses into an overall status.

    Any hard failure -> "fail". Else any warning/fail -> "warning". Else
    "pass".
    """
    if any(c.hard_failure and c.status == "fail" for c in checks):
        return "fail"
    if any(c.status in ("warning", "fail") for c in checks):
        return "warning"
    return "pass"


def _load_image_safely(image: np.ndarray | str | Path) -> tuple[np.ndarray | None, str]:
    """Load an image from path or use an array directly.

    Returns (image_or_none, decode_status). Never raises on corrupt input.
    """
    result_img: np.ndarray | None = None
    status = "ok"

    if isinstance(image, str | Path):
        path = Path(image)
        if not path.exists():
            status = f"file_not_found:{path}"
        else:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            except Exception:  # never crash on corrupt input
                status = "read_exception"
                img = None
            if img is None:
                status = "decode_failed"
            elif img.size == 0:
                status = "empty_array"
            else:
                result_img = img
    else:
        # Already an array.
        arr = np.asarray(image)
        if arr.size == 0:
            status = "empty_array"
        elif arr.ndim not in (GRAYSCALE_NDIM, CHANNEL_LAST_RGB):
            status = f"unsupported_ndim:{arr.ndim}"
        else:
            result_img = arr

    return result_img, status


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR uint8 image to grayscale."""
    if image.ndim == GRAYSCALE_NDIM:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _check_resolution(image: np.ndarray, cfg: QualityConfig) -> CheckResult:
    h, w = image.shape[:2]
    ok_w = w >= cfg.min_width
    ok_h = h >= cfg.min_height
    status = "pass" if (ok_w and ok_h) else "fail"
    reason = (
        f"Image {w}x{h} meets minimum {cfg.min_width}x{cfg.min_height}."
        if status == "pass"
        else f"Image {w}x{h} below minimum {cfg.min_width}x{cfg.min_height}."
    )
    return CheckResult(
        name="resolution",
        status=status,
        measured={"width": w, "height": h},
        threshold={
            "min_width": cfg.min_width,
            "min_height": cfg.min_height,
        },
        reason=reason,
        hard_failure=cfg.resolution_hard_failure,
    )


def _check_underexposure(image: np.ndarray, cfg: QualityConfig) -> CheckResult:
    gray = _to_grayscale(image)
    mean = float(gray.mean())
    dark_frac = float((gray < cfg.dark_pixel_level).mean())
    status = "pass"
    reasons = []
    if dark_frac > cfg.dark_pixel_fraction:
        status = "fail"
        reasons.append(f"{dark_frac:.3f} of pixels below level {cfg.dark_pixel_level}")
    elif mean < cfg.low_mean_level:
        status = "warning"
        reasons.append(f"mean intensity {mean:.1f} below {cfg.low_mean_level}")
    else:
        reasons.append(f"mean {mean:.1f}, dark fraction {dark_frac:.3f} within range")
    measured = {
        "mean_intensity": round(mean, 3),
        "dark_pixel_fraction": round(dark_frac, 4),
    }
    return CheckResult(
        name="underexposure",
        status=status,
        measured=measured,
        threshold={
            "dark_pixel_level": cfg.dark_pixel_level,
            "dark_pixel_fraction": cfg.dark_pixel_fraction,
            "low_mean_level": cfg.low_mean_level,
        },
        reason="; ".join(reasons),
        hard_failure=False,
    )


def _check_overexposure(image: np.ndarray, cfg: QualityConfig) -> CheckResult:
    gray = _to_grayscale(image)
    sat_frac = float((gray >= cfg.saturated_level).mean())
    status = "fail" if sat_frac > cfg.saturated_fraction else "pass"
    reason = (
        f"{sat_frac:.3f} of pixels saturated (>= {cfg.saturated_level}); "
        f"threshold {cfg.saturated_fraction:.2f}"
    )
    return CheckResult(
        name="overexposure",
        status=status,
        measured={"saturated_fraction": round(sat_frac, 4)},
        threshold={
            "saturated_level": cfg.saturated_level,
            "saturated_fraction": cfg.saturated_fraction,
        },
        reason=reason,
        hard_failure=False,
    )


def _check_blur(image: np.ndarray, cfg: QualityConfig) -> CheckResult:
    gray = _to_grayscale(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    status = "fail" if var < cfg.blur_laplacian_var_min else "pass"
    reason = (
        f"Laplacian variance {var:.2f} below {cfg.blur_laplacian_var_min:.2f}"
        if status == "fail"
        else f"Laplacian variance {var:.2f} above {cfg.blur_laplacian_var_min:.2f}"
    )
    return CheckResult(
        name="blur",
        status=status,
        measured={"laplacian_variance": round(var, 3)},
        threshold={"blur_laplacian_var_min": cfg.blur_laplacian_var_min},
        reason=reason,
        hard_failure=False,
    )


def _check_retinal_fov(image: np.ndarray, cfg: QualityConfig) -> CheckResult:
    """Estimate visible retinal content after removing near-black borders.

    Uses crop_black_borders-like logic: finds the bounding box of non-dark
    pixels and measures how much of the full frame it occupies. Naturally
    dark borders are expected and are not themselves a failure; only an
    extremely small content region triggers the check.
    """
    gray = _to_grayscale(image)
    mask = gray > cfg.dark_pixel_level
    if not mask.any():
        # Entirely dark image: content fraction is 0. Compare against
        # threshold instead of hard-failing.
        frac = 0.0
    else:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        content_area = int((rmax - rmin + 1) * (cmax - cmin + 1))
        total_area = int(gray.shape[0] * gray.shape[1])
        frac = content_area / total_area if total_area else 0.0
    status = "fail" if frac < cfg.fov_min_content_fraction else "pass"
    reason = (
        f"Retinal content fraction {frac:.3f} below {cfg.fov_min_content_fraction:.2f}"
        if status == "fail"
        else f"Retinal content fraction {frac:.3f} above {cfg.fov_min_content_fraction:.2f}"
    )
    return CheckResult(
        name="retinal_fov",
        status=status,
        measured={"content_fraction": round(frac, 4)},
        threshold={"fov_min_content_fraction": cfg.fov_min_content_fraction},
        reason=reason,
        hard_failure=False,
    )


def assess_quality(
    image: np.ndarray | str | Path,
    config: QualityConfig | None = None,
) -> QualityResult:
    """Assess the quality of a retinal image.

    Parameters
    ----------
    image : np.ndarray | str | Path
        A uint8 HWC image array (BGR or grayscale), or a path to an image
        file. Corrupt/unreadable files are handled gracefully and result in
        a "fail" status rather than an exception.
    config : QualityConfig | None
        Threshold configuration. Defaults are used when omitted.

    Returns
    -------
    QualityResult
        Structured result with overall status, per-check results, measured
        values, thresholds, and human-readable reasons.
    """
    cfg = config or QualityConfig()

    img, decode_status = _load_image_safely(image)

    width: int | None = None
    height: int | None = None
    decoded = decode_status == "ok"
    checks: list[CheckResult] = []
    reasons: list[str] = []

    if not decoded:
        checks.append(
            CheckResult(
                name="image_validity",
                status="fail",
                measured=None,
                threshold={"readable": True},
                reason=f"Could not decode image: {decode_status}",
                hard_failure=True,
            )
        )
        reasons.append(f"Image not decodable: {decode_status}")
        return QualityResult(
            status="fail",
            checks=checks,
            image_width=width,
            image_height=height,
            decoded=False,
            reasons=reasons,
            thresholds=asdict(cfg),
        )

    # Array loaded successfully.
    height, width = img.shape[:2]
    checks.append(_check_resolution(img, cfg))
    checks.append(_check_underexposure(img, cfg))
    checks.append(_check_overexposure(img, cfg))
    checks.append(_check_blur(img, cfg))
    checks.append(_check_retinal_fov(img, cfg))

    reasons = [c.reason for c in checks if c.status != "pass"]

    status = _status_from_checks(checks)
    return QualityResult(
        status=status,
        checks=checks,
        image_width=width,
        image_height=height,
        decoded=True,
        reasons=reasons,
        thresholds=asdict(cfg),
    )


def quality_gate(
    image: np.ndarray | str | Path,
    config: QualityConfig | None = None,
) -> tuple[bool, QualityResult]:
    """Convenience wrapper returning (should_proceed, result).

    should_proceed is True when the image passed all quality checks
    (status "pass"). False when status is "warning" or "fail".

    This is the entry point to call before inference: if should_proceed is
    False, the image should not be passed to the grading model without
    review.
    """
    result = assess_quality(image, config)
    return result.status == "pass", result


def process_image_path(
    path: str | Path,
    config: QualityConfig | None = None,
) -> QualityResult:
    """Assess a single image file path. Wrapper around assess_quality."""
    return assess_quality(path, config)


def process_directory(
    directory: str | Path,
    config: QualityConfig | None = None,
    recursive: bool = False,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> list[QualityResult]:
    """Assess every image file in a directory.

    Returns a list of QualityResult, one per image file found. Does not
    recurse unless ``recursive=True``.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    results: list[QualityResult] = []
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in extensions:
            results.append(assess_quality(path, config))
    return results


def summarize(results: list[QualityResult]) -> dict[str, Any]:
    """Summarize a list of QualityResult into counts and statuses."""
    counts = {"pass": 0, "warning": 0, "fail": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "total": len(results),
        "pass": counts["pass"],
        "warning": counts["warning"],
        "fail": counts["fail"],
        "results": [r.to_dict() for r in results],
    }
