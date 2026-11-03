"""Tests for src.lumoculis.quality using synthetic images (no real data)."""

from __future__ import annotations

import json

import cv2
import numpy as np

from lumoculis.quality import (
    CheckResult,
    QualityConfig,
    QualityResult,
    assess_quality,
    process_directory,
    quality_gate,
    summarize,
)

IMG_SIZE = 256
LOW_RES_SIZE = 64
BLUR_LAPLACIAN_MIN = 30.0
CONFIG_MIN_WIDTH = 224
CONFIG_MIN_HEIGHT = 224
CONFIG_BLUR_MIN = 30.0
CONFIG_FOV_FRACTION = 0.10
NUM_IMAGES = 3


def _synthetic_fundus(size: int = IMG_SIZE) -> np.ndarray:
    """A synthetic BGR image resembling a retinal photograph: a bright disc
    (optic disc / vessels) on a gray background with dark borders."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = [40, 40, 40]  # gray background
    # Bright circular "disc" region in the center.
    cx, cy, r = size // 2, size // 2, size // 4
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[mask] = [180, 140, 100]
    return img


def _make_blurred(img: np.ndarray, ksize: int = 31) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def test_valid_image_passes() -> None:
    img = _synthetic_fundus()
    result = assess_quality(img)
    assert result.status == "pass"
    assert result.decoded is True
    assert result.image_width == IMG_SIZE
    assert result.image_height == IMG_SIZE


def test_corrupt_file_never_crashes(tmp_path) -> None:
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not a real image file")
    result = assess_quality(bad)
    assert result.status == "fail"
    assert result.decoded is False
    assert len(result.reasons) > 0


def test_missing_file_never_crashes(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.png"
    result = assess_quality(missing)
    assert result.status == "fail"
    assert result.decoded is False


def test_low_resolution_fails() -> None:
    img = np.zeros((LOW_RES_SIZE, LOW_RES_SIZE, 3), dtype=np.uint8)
    img[:] = 100
    result = assess_quality(img)
    assert result.status == "fail"
    res_check = next(c for c in result.checks if c.name == "resolution")
    assert res_check.status == "fail"
    assert res_check.hard_failure is True


def test_low_resolution_warning_when_configured() -> None:
    img = np.zeros((LOW_RES_SIZE, LOW_RES_SIZE, 3), dtype=np.uint8)
    img[:] = 100
    cfg = QualityConfig(resolution_hard_failure=False)
    result = assess_quality(img, config=cfg)
    assert result.status == "warning"
    res_check = next(c for c in result.checks if c.name == "resolution")
    assert res_check.hard_failure is False


def test_very_dark_image_fails() -> None:
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    result = assess_quality(img)
    # Extreme underexposure is a quality warning (not a hard technical failure),
    # so overall status is "warning"; quality_gate will still block it.
    assert result.status == "warning"
    dark_check = next(c for c in result.checks if c.name == "underexposure")
    assert dark_check.status == "fail"


def test_very_bright_image_fails() -> None:
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=np.uint8)
    result = assess_quality(img)
    # Extreme overexposure is a quality warning, so overall status is "warning".
    assert result.status == "warning"
    over_check = next(c for c in result.checks if c.name == "overexposure")
    assert over_check.status == "fail"


def test_blurred_image_fails() -> None:
    img = _synthetic_fundus()
    blurred = _make_blurred(img, ksize=41)
    result = assess_quality(blurred)
    blur_check = next(c for c in result.checks if c.name == "blur")
    # With heavy blur the Laplacian variance should drop below threshold.
    assert blur_check.measured["laplacian_variance"] < BLUR_LAPLACIAN_MIN


def test_sharp_vs_blurry_distinguished() -> None:
    """Sharp images should score above threshold; blurred below."""
    sharp = _synthetic_fundus()
    blurred = _make_blurred(sharp, ksize=41)
    sharp_result = assess_quality(sharp)
    blurred_result = assess_quality(blurred)
    sharp_blur = next(c for c in sharp_result.checks if c.name == "blur")
    blurred_blur = next(c for c in blurred_result.checks if c.name == "blur")
    assert sharp_blur.measured["laplacian_variance"] > BLUR_LAPLACIAN_MIN
    assert blurred_blur.measured["laplacian_variance"] < BLUR_LAPLACIAN_MIN
    assert sharp_blur.status == "pass"
    assert blurred_blur.status == "fail"


def test_dark_borders_around_visible_content_passes() -> None:
    """A visible retinal region with naturally dark borders should pass."""
    size = IMG_SIZE
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Textured retinal-like content in the center, leaving dark borders.
    content = np.zeros((size, size), dtype=bool)
    content[32:224, 32:224] = True
    img[content] = [40, 40, 40]
    cx, cy, r = size // 2, size // 2, size // 6
    yy, xx = np.ogrid[:size, :size]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[content & disk] = [180, 140, 100]
    result = assess_quality(img)
    assert result.status == "pass"
    fov_check = next(c for c in result.checks if c.name == "retinal_fov")
    assert fov_check.status == "pass"


def test_configurable_thresholds() -> None:
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 10, dtype=np.uint8)
    # Lenient config: no check should fail.
    cfg = QualityConfig(
        dark_pixel_fraction=1.0,
        low_mean_level=0,
        blur_laplacian_var_min=0.0,
        fov_min_content_fraction=0.0,
        saturated_fraction=1.0,
    )
    result = assess_quality(img, config=cfg)
    assert result.status == "pass"



def test_blur_standardized_across_resolutions() -> None:
    """Blur assessment should be stable for the same content at different sizes."""
    for size in (256, 384, 512):
        img = _synthetic_fundus(size=size)
        result = assess_quality(img)
        blur_check = next(c for c in result.checks if c.name == "blur")
        assert blur_check.measured["standardized_size"] == 256  # noqa: PLR2004
        assert blur_check.status == "pass"


def test_blur_with_black_borders_uses_retinal_roi() -> None:
    """Black borders must not dominate the blur measurement."""
    size = IMG_SIZE
    img = np.zeros((size, size, 3), dtype=np.uint8)
    content = np.zeros((size, size), dtype=bool)
    content[32:224, 32:224] = True
    img[content] = [40, 40, 40]
    cx, cy, r = size // 2, size // 2, size // 6
    yy, xx = np.ogrid[:size, :size]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[content & disk] = [180, 140, 100]
    result = assess_quality(img)
    blur_check = next(c for c in result.checks if c.name == "blur")
    assert blur_check.status == "pass"


def test_blur_empty_image_does_not_crash() -> None:
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    result = assess_quality(img)
    assert result.decoded is False
    assert result.status == "fail"


def test_blur_grayscale_input() -> None:
    gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gray[:] = 100
    result = assess_quality(gray)
    blur_check = next(c for c in result.checks if c.name == "blur")
    assert "laplacian_variance" in blur_check.measured


def test_json_serialization() -> None:
    img = _synthetic_fundus()
    result = assess_quality(img)
    data = result.to_dict()
    assert "status" in data
    assert "checks" in data
    assert "thresholds" in data
    assert "image_width" in data
    assert "image_height" in data
    assert "decoded" in data
    assert "reasons" in data
    # Must round-trip through JSON.
    json_str = json.dumps(data)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert loaded["status"] == "pass"


def test_quality_returns_bool() -> None:
    img = _synthetic_fundus()
    proceed, result = quality_gate(img)
    assert proceed is True
    assert isinstance(result, QualityResult)

    dark = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    proceed, result = quality_gate(dark)
    assert proceed is False


def test_process_directory_counts(tmp_path) -> None:
    # Create a few synthetic images in a directory.
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(NUM_IMAGES):
        img = _synthetic_fundus()
        path = img_dir / f"img_{i}.png"
        cv2.imwrite(str(path), img)
    # Add a non-image file that should be ignored.
    (img_dir / "notes.txt").write_text("hello")

    results = process_directory(img_dir)
    assert len(results) == NUM_IMAGES
    summary = summarize(results)
    assert summary["total"] == NUM_IMAGES
    assert summary["pass"] == NUM_IMAGES
    assert summary["fail"] == 0


def test_process_directory_recursive(tmp_path) -> None:
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True)
    img = _synthetic_fundus()
    cv2.imwrite(str(sub / "deep.png"), img)
    results = process_directory(root, recursive=True)
    assert len(results) == 1
    assert results[0].status == "pass"


def test_check_result_dataclass() -> None:
    c = CheckResult(
        name="resolution",
        status="pass",
        measured={"width": IMG_SIZE, "height": IMG_SIZE},
        threshold={"min_width": CONFIG_MIN_WIDTH, "min_height": CONFIG_MIN_HEIGHT},
        reason="ok",
        hard_failure=True,
    )
    assert c.name == "resolution"
    assert c.status == "pass"
    assert c.hard_failure is True


def test_quality_config_defaults() -> None:
    cfg = QualityConfig()
    assert cfg.min_width == CONFIG_MIN_WIDTH
    assert cfg.min_height == CONFIG_MIN_HEIGHT
    assert cfg.blur_laplacian_var_min == CONFIG_BLUR_MIN
    assert cfg.fov_min_content_fraction == CONFIG_FOV_FRACTION


def test_grayscale_image_handled() -> None:
    img = np.full((IMG_SIZE, IMG_SIZE), 100, dtype=np.uint8)
    img[64:192, 64:192] = 180  # add variation so blur check passes
    result = assess_quality(img)
    assert result.decoded is True
    assert result.status == "pass"


def test_empty_array_fails() -> None:
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    result = assess_quality(img)
    assert result.status == "fail"
    assert result.decoded is False
