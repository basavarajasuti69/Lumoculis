"""Tests for the FastAPI evaluation backend (api/main.py).

These tests use lightweight mocks so they do not load the real PyTorch
checkpoint or run model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from lumoculis.api import main as api_main
from lumoculis.quality import QualityResult

OK = 200
BAD_REQUEST = 400
UNPROCESSABLE_ENTITY = 422
EXPECTED_IMAGE_SIZE = 384
NUM_CLASSES = 5
EXPECTED_CLASS = 3
EXPECTED_REFERRABLE_PROB = 0.55
PROB_TOLERANCE = 1e-6


@dataclass
class _DummyModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def zero_grad(self, set_to_none=True):
        return self

    def forward(self, x):
        return torch.tensor([[0.05, 0.10, 0.25, 0.30, 0.30]])


def _dummy_metadata() -> dict[str, Any]:
    return {
        "epoch": 0,
        "val_qwk": 0.5,
        "config": {
            "image_size": 384,
            "model_name": "dummy",
            "num_classes": 5,
        },
    }


def _patch_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "CHECKPOINT_PATH", "dummy.pt")
    monkeypatch.setattr(api_main, "CALIBRATION_PATH", None)
    monkeypatch.setattr(api_main, "DEVICE", "cpu")
    monkeypatch.setattr(api_main, "_load_checkpoint", lambda path, device: (_DummyModel(), _dummy_metadata()))
    monkeypatch.setattr(api_main, "_load_temperature", lambda path, num_classes: None)
    monkeypatch.setattr(api_main, "_pick_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        api_main,
        "_preprocess_image",
        lambda image, size, device: (
            torch.zeros(1, 3, size, size),
            np.zeros((size, size, 3), dtype=np.uint8),
            size,
            size,
        ),
    )
    monkeypatch.setattr(
        api_main,
        "_compute_predictions",
        lambda model, input_tensor, device, temperature_scaling: (
            np.array([0.05, 0.10, 0.25, 0.30, 0.30]),
            None,
            0.55,
            3,
            True,
        ),
    )
    monkeypatch.setattr(
        api_main,
        "_run_quality_assessment",
        lambda image, config: (
            QualityResult(
                status="pass",
                checks=[],
                image_width=int(getattr(image, "shape", (384, 384, 3))[0]),
                image_height=int(getattr(image, "shape", (384, 384, 3))[1]),
                decoded=True,
                reasons=[],
                thresholds={},
            ),
            False,
        ),
    )
    gradcam_dir = Path(api_main.GRADCAM_DIR.name)
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    dummy_gradcam = gradcam_dir / "dummy.png"
    dummy_gradcam.touch()
    monkeypatch.setattr(
        api_main,
        "_run_gradcam_visualization",
        lambda model, input_tensor, rgb_image, config, predicted_class: (True, str(dummy_gradcam)),
    )


def _make_png_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return encoded.tobytes()


def test_health_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    with TestClient(api_main.app) as client:
        response = client.get("/health")
        assert response.status_code == OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["checkpoint_configured"] is True


def test_model_info(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    with TestClient(api_main.app) as client:
        response = client.get("/model-info")
        assert response.status_code == OK
        data = response.json()
        assert data["model_loaded"] is True
        assert data["architecture"] == "dummy"
        assert data["num_classes"] == NUM_CLASSES
        assert data["calibration_loaded"] is False


def test_predict_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _make_png_bytes(), "image/png")},
            data={"run_quality_gate": "false", "quality_required": "false", "gradcam": "false"},
        )
        assert response.status_code == OK, response.text
        data = response.json()
        assert data["predicted_class"] == EXPECTED_CLASS
        assert data["predicted_grade"] == EXPECTED_CLASS
        assert data["referable"] is True
        assert abs(data["referable_probability"] - EXPECTED_REFERRABLE_PROB) < PROB_TOLERANCE
        assert len(data["raw_probabilities"]) == NUM_CLASSES
        assert data["quality"] is None
        assert data["quality_blocked"] is False
        assert data["gradcam_saved"] is None
        assert data["gradcam_url"] is None
        assert data["model_checkpoint"] == "dummy.pt"
        assert data["image_width"] == EXPECTED_IMAGE_SIZE
        assert data["image_height"] == EXPECTED_IMAGE_SIZE
        assert "Lumoculis is a research prototype" in data["disclaimer"]


def test_predict_missing_file() -> None:
    _patch_api(pytest.MonkeyPatch())
    with TestClient(api_main.app) as client:
        response = client.post("/predict", data={"run_quality_gate": "false"})
        assert response.status_code == UNPROCESSABLE_ENTITY


def test_predict_unsupported_extension() -> None:
    _patch_api(pytest.MonkeyPatch())
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.txt", b"not an image", "text/plain")},
            data={"run_quality_gate": "false"},
        )
        assert response.status_code == BAD_REQUEST


def test_predict_corrupt_image() -> None:
    _patch_api(pytest.MonkeyPatch())
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", b"not a png", "image/png")},
            data={"run_quality_gate": "false"},
        )
        assert response.status_code == BAD_REQUEST


def test_predict_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    with TestClient(api_main.app) as client:
        too_large = b"0" * (int(api_main.MAX_UPLOAD_MB * 1024 * 1024) + 1)
        response = client.post(
            "/predict",
            files={"file": ("large.png", too_large, "image/png")},
            data={"run_quality_gate": "false"},
        )
        assert response.status_code == BAD_REQUEST


def test_predict_quality_required_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_run_quality_assessment",
        lambda image, config: (
            QualityResult(
                status="fail",
                checks=[],
                image_width=64,
                image_height=64,
                decoded=True,
                reasons=["blur"],
                thresholds={},
            ),
            True,
        ),
    )
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _make_png_bytes(), "image/png")},
            data={"run_quality_gate": "true", "quality_required": "true", "gradcam": "false"},
        )
        assert response.status_code == BAD_REQUEST, response.text
        data = response.json()
        detail = data.get("detail", data)
        assert detail["error"] == "quality_gate_failed"
        assert detail["quality"]["status"] == "fail"


def test_predict_quality_pass_included(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _make_png_bytes(), "image/png")},
            data={"run_quality_gate": "true", "quality_required": "false", "gradcam": "false"},
        )
        assert response.status_code == OK, response.text
        data = response.json()
        assert data["quality"] is not None
        assert data["quality"]["status"] == "pass"
        assert data["quality_blocked"] is False


def test_predict_quality_warning_allowed_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_run_quality_assessment",
        lambda image, config: (
            QualityResult(
                status="warning",
                checks=[],
                image_width=64,
                image_height=64,
                decoded=True,
                reasons=["blur"],
                thresholds={},
            ),
            False,
        ),
    )
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _make_png_bytes(), "image/png")},
            data={"run_quality_gate": "true", "quality_required": "false", "gradcam": "false"},
        )
        assert response.status_code == OK, response.text
        data = response.json()
        assert data["quality"] is not None
        assert data["quality"]["status"] == "warning"
        assert data["quality_blocked"] is False
        assert data["predicted_class"] == EXPECTED_CLASS


def test_predict_quality_warning_blocked_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_run_quality_assessment",
        lambda image, config: (
            QualityResult(
                status="warning",
                checks=[],
                image_width=64,
                image_height=64,
                decoded=True,
                reasons=["blur"],
                thresholds={},
            ),
            True,
        ),
    )
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _make_png_bytes(), "image/png")},
            data={"run_quality_gate": "true", "quality_required": "true", "gradcam": "false"},
        )
        assert response.status_code == BAD_REQUEST, response.text
        data = response.json()
        detail = data.get("detail", data)
        assert detail["error"] == "quality_gate_failed"
        assert detail["quality"]["status"] == "warning"

def test_predict_gradcam(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch)
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _make_png_bytes(), "image/png")},
            data={"run_quality_gate": "false", "quality_required": "false", "gradcam": "true"},
        )
        assert response.status_code == OK, response.text
        data = response.json()
        assert data["gradcam_saved"] is True
        assert data["gradcam_url"] is not None
        gradcam_response = client.get(data["gradcam_url"])
        assert gradcam_response.status_code == OK


def test_model_info_no_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "CHECKPOINT_PATH", "")
    monkeypatch.setattr(api_main, "model", None)
    monkeypatch.setattr(api_main, "model_metadata", None)
    monkeypatch.setattr(api_main, "temperature_scaling", None)
    monkeypatch.setattr(api_main, "model_error", "no_checkpoint_configured")
    with TestClient(api_main.app) as client:
        response = client.get("/model-info")
        assert response.status_code == OK
        assert response.json()["model_loaded"] is False
        assert response.json()["error"] == "no_checkpoint_configured"
