"""Integration tests for src.lumoculis.inference using synthetic models and images.

No pretrained weights or datasets are downloaded. All models are randomly
initialized and all images are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from lumoculis.calibration import TemperatureScaling
from lumoculis.inference import (
    NUM_CLASSES,
    REFERRABLE_THRESHOLD_GRADE,
    InferenceConfig,
    PredictionReport,
    run_inference,
)
from lumoculis.model import build_model

TEST_IMAGE_SIZE = 128
PROB_SUM_TOLERANCE = 1e-5
DEFAULT_IMAGE_SIZE = 384


def _save_checkpoint(
    path: Path,
    model_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    image_size: int = 64,
) -> None:
    """Build a tiny model and save a checkpoint in the training format."""
    model = build_model(model_name, num_classes, pretrained=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": model_name,
            "num_classes": num_classes,
            "image_size": image_size,
            "seed": 42,
        },
        "epoch": 1,
        "val_qwk": 0.5,
    }
    torch.save(checkpoint, path)


def _synthetic_image(size: int = 128) -> np.ndarray:
    """A small synthetic BGR image resembling a fundus photo."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = 80
    cx, cy, r = size // 2, size // 2, size // 6
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[mask] = [180, 140, 100]
    return img


def _write_image(path: Path, size: int = 128) -> None:
    cv2.imwrite(str(path), _synthetic_image(size))


def test_valid_image_inference(tmp_path) -> None:
    """End-to-end inference on a valid synthetic image."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    report = run_inference(img_path, config)

    assert isinstance(report, PredictionReport)
    assert report.image_width == TEST_IMAGE_SIZE
    assert report.image_height == TEST_IMAGE_SIZE
    assert 0 <= report.predicted_class < NUM_CLASSES
    assert report.predicted_grade == report.predicted_class
    assert isinstance(report.referable, bool)
    assert 0.0 <= report.referable_probability <= 1.0
    assert len(report.raw_probabilities) == NUM_CLASSES
    assert np.all(np.isfinite(report.raw_probabilities))
    assert abs(sum(report.raw_probabilities) - 1.0) < PROB_SUM_TOLERANCE
    assert report.calibrated_probabilities is None
    assert report.temperature is None
    assert report.temperature_applied is False
    assert report.quality is None
    assert report.quality_blocked is False
    assert report.disclaimer  # non-empty disclaimer


def test_probability_normalization(tmp_path) -> None:
    """Raw probabilities sum to 1 and are finite."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    report = run_inference(img_path, config)

    probs = np.array(report.raw_probabilities)
    assert np.allclose(probs.sum(), 1.0, atol=1e-5)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_missing_checkpoint_raises(tmp_path) -> None:
    """Inference on a missing checkpoint raises FileNotFoundError."""
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=tmp_path / "nonexistent.pt",
        image_size=64,
        device="cpu",
    )
    with pytest.raises(FileNotFoundError):
        run_inference(img_path, config)


def test_corrupt_image_raises(tmp_path) -> None:
    """Inference on a corrupt image file raises ValueError."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not an image")

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    with pytest.raises(ValueError):
        run_inference(bad, config)


def test_incompatible_checkpoint_raises(tmp_path) -> None:
    """A checkpoint missing model_state_dict raises ValueError."""
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"foo": "bar"}, bad_ckpt)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=bad_ckpt,
        image_size=64,
        device="cpu",
    )
    with pytest.raises(ValueError):
        run_inference(img_path, config)


def test_calibration_loading(tmp_path) -> None:
    """Loading a fitted temperature scales probabilities."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    # Save a fitted temperature.
    ts = TemperatureScaling(num_classes=NUM_CLASSES)
    ts.temperature = 1.5
    ts.fitted = True
    ts.calibration_loss = 0.42
    temp_path = tmp_path / "temperature.json"
    ts.save(temp_path)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
        temperature_path=temp_path,
    )
    report = run_inference(img_path, config)

    assert report.temperature_applied is True
    assert report.temperature == pytest.approx(1.5)
    assert report.calibrated_probabilities is not None
    assert len(report.calibrated_probabilities) == NUM_CLASSES
    assert np.allclose(np.sum(report.calibrated_probabilities), 1.0, atol=1e-5)
    assert np.all(np.isfinite(report.calibrated_probabilities))


def test_missing_temperature_file_silent_none(tmp_path) -> None:
    """A missing temperature file results in no calibration (not an error)."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
        temperature_path=tmp_path / "nonexistent.json",
    )
    report = run_inference(img_path, config)

    assert report.temperature_applied is False
    assert report.temperature is None
    assert report.calibrated_probabilities is None


def test_quality_gate_pass(tmp_path) -> None:
    """Quality gate with a good image does not block inference."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=256)  # large enough to pass resolution

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
        run_quality_gate=True,
        quality_required=False,
    )
    report = run_inference(img_path, config)

    assert report.quality is not None
    assert report.quality_blocked is False


def test_quality_gate_blocks_low_resolution(tmp_path) -> None:
    """Quality gate with required=true blocks low-resolution images."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=64)  # below default min 224

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
        run_quality_gate=True,
        quality_required=True,
    )
    report = run_inference(img_path, config)

    assert report.quality is not None
    assert report.quality_blocked is True


def test_quality_gate_warning_does_not_block(tmp_path) -> None:
    """quality_required=false does not block even on quality failure."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=64)  # below default min 224

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
        run_quality_gate=True,
        quality_required=False,
    )
    report = run_inference(img_path, config)

    assert report.quality_blocked is False


def test_gradcam_output(tmp_path) -> None:
    """Grad-CAM produces a saved visualization for a supported model."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)
    gradcam_path = tmp_path / "gradcam.png"

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
        gradcam=True,
        gradcam_output=gradcam_path,
    )
    report = run_inference(img_path, config)

    assert report.gradcam_saved is True
    assert report.gradcam_output_path is not None
    assert Path(report.gradcam_output_path).exists()


def test_json_report_serialization(tmp_path) -> None:
    """The prediction report serializes to valid JSON."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    report = run_inference(img_path, config)

    report_path = tmp_path / "report.json"
    report.save(report_path)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "status" not in data  # quality is None when not run
    assert "predicted_class" in data
    assert "raw_probabilities" in data
    assert "disclaimer" in data
    assert "research prototype" in data["disclaimer"].lower()
    assert len(data["raw_probabilities"]) == NUM_CLASSES


def test_referable_flag_grade_2_plus(tmp_path) -> None:
    """Referable flag is True when predicted grade >= 2."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    report = run_inference(img_path, config)

    expected_referable = report.predicted_class >= REFERRABLE_THRESHOLD_GRADE
    assert report.referable == expected_referable


def test_inference_config_defaults() -> None:
    """InferenceConfig has sensible defaults."""
    config = InferenceConfig(checkpoint_path="dummy.pt")
    assert config.image_size == DEFAULT_IMAGE_SIZE
    assert config.num_classes == NUM_CLASSES
    assert config.device == "mps"
    assert config.run_quality_gate is False
    assert config.quality_required is False
    assert config.temperature_path is None
    assert config.gradcam is False


def test_report_to_dict_contains_expected_keys(tmp_path) -> None:
    """The report dict contains all expected top-level keys."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)
    img_path = tmp_path / "image.png"
    _write_image(img_path, size=128)

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    report = run_inference(img_path, config)

    d = report.to_dict()
    expected_keys = {
        "image_path",
        "image_width",
        "image_height",
        "predicted_class",
        "predicted_grade",
        "referable",
        "referable_probability",
        "raw_probabilities",
        "calibrated_probabilities",
        "temperature",
        "temperature_applied",
        "quality_blocked",
        "quality",
        "gradcam_saved",
        "gradcam_output_path",
        "model_checkpoint",
        "model_architecture",
        "disclaimer",
    }
    assert set(d.keys()) == expected_keys


def test_inference_on_grayscale_image(tmp_path) -> None:
    """A grayscale image array is handled correctly."""
    ckpt = tmp_path / "model.pt"
    _save_checkpoint(ckpt, image_size=64)

    img = np.full((128, 128), 100, dtype=np.uint8)
    img[50:78, 50:78] = 180

    config = InferenceConfig(
        checkpoint_path=ckpt,
        image_size=64,
        device="cpu",
    )
    report = run_inference(img, config)

    assert 0 <= report.predicted_class < NUM_CLASSES
    assert len(report.raw_probabilities) == NUM_CLASSES
