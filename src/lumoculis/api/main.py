"""FastAPI backend for Lumoculis diabetic retinopathy grading.

This service reuses the existing ML inference pipeline in
``src/lumoculis/`` without modifying model code, training code, or experiment
artifacts. It exposes a small HTTP API for image upload, prediction, optional
quality gating, optional Grad-CAM visualization, and JSON report generation.

Endpoints:
- GET /health
- GET /model-info
- POST /predict
- GET /gradcam/{filename}
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from lumoculis.inference import (
    RESEARCH_DISCLAIMER,
    InferenceConfig,
    _compute_predictions,
    _load_checkpoint,
    _load_temperature,
    _pick_device,
    _preprocess_image,
    _run_gradcam_visualization,
    _run_quality_assessment,
)
from lumoculis.quality import QualityConfig

LOGGER = logging.getLogger("lumoculis.api")

DEFAULT_MAX_UPLOAD_MB = 10.0
DEFAULT_DEVICE = "auto"
DEFAULT_CORS_ORIGINS = "http://localhost:3000"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CHECKPOINT_PATH = os.environ.get("LUMOCULIS_CHECKPOINT") or ""
if not CHECKPOINT_PATH or not Path(CHECKPOINT_PATH).exists():
    default_checkpoint = Path("checkpoints/expC_weighted_sampling_20261101_172044/best.pt")
    if default_checkpoint.exists():
        CHECKPOINT_PATH = str(default_checkpoint)
    else:
        CHECKPOINT_PATH = ""

CALIBRATION_PATH = os.environ.get("LUMOCULIS_CALIBRATION") or None
DEVICE = os.environ.get("LUMOCULIS_DEVICE", DEFAULT_DEVICE)
MAX_UPLOAD_MB = float(os.environ.get("LUMOCULIS_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("LUMOCULIS_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

GRADCAM_DIR = tempfile.TemporaryDirectory(prefix="lumoculis-gradcam-")
GRADCAM_DIR.__enter__()

model: Any = None
model_metadata: dict[str, Any] | None = None
temperature_scaling: Any = None
model_error: str | None = None


class QualityCheckResult(BaseModel):
    name: str
    status: str
    measured: Any
    threshold: Any
    reason: str
    hard_failure: bool = False

    model_config = ConfigDict(from_attributes=True)


class QualityResultModel(BaseModel):
    status: str
    decoded: bool
    image_width: int | None
    image_height: int | None
    reasons: list[str]
    thresholds: dict[str, Any]
    checks: list[QualityCheckResult]

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_grade: int
    referable: bool
    referable_probability: float
    raw_probabilities: list[float]
    calibrated_probabilities: list[float] | None = None
    temperature: float | None = None
    temperature_applied: bool = False
    quality: QualityResultModel | None = None
    quality_blocked: bool = False
    gradcam_saved: bool | None = None
    gradcam_url: str | None = None
    model_checkpoint: str
    model_architecture: str | None = None
    image_width: int
    image_height: int
    filename: str | None = None
    disclaimer: str = RESEARCH_DISCLAIMER

    model_config = ConfigDict(from_attributes=True)


class ModelInfo(BaseModel):
    model_loaded: bool
    checkpoint_configured: bool
    checkpoint_path: str | None = None
    device: str
    architecture: str | None = None
    num_classes: int | None = None
    image_size: int | None = None
    epoch: int | None = None
    val_qwk: float | None = None
    calibration_loaded: bool = False
    error: str | None = None
    disclaimer: str = RESEARCH_DISCLAIMER

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_configured: bool
    device: str
    version: str = "0.1.0"
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_device(value: str | None) -> str:
    value = (value or DEVICE).strip().lower()
    if value not in {"cpu", "mps", "auto"}:
        raise HTTPException(status_code=400, detail="device must be one of cpu, mps, or auto")
    return value


def _validate_upload_file(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file extension '{extension}'; use {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
        )
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"file too large; maximum {MAX_UPLOAD_MB:g} MB",
        )
    return content


def _decode_image(content: bytes, filename: str) -> np.ndarray:
    array = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="could not decode image file")
    return img


def _serialize_quality_result(result: Any) -> QualityResultModel:
    data = result.to_dict() if hasattr(result, "to_dict") else {}
    return QualityResultModel(
        status=data.get("status", "unknown"),
        decoded=bool(data.get("decoded", False)),
        image_width=data.get("image_width"),
        image_height=data.get("image_height"),
        reasons=list(data.get("reasons", [])),
        thresholds=data.get("thresholds", {}),
        checks=[
            QualityCheckResult(
                name=c.get("name", ""),
                status=c.get("status", ""),
                measured=c.get("measured"),
                threshold=c.get("threshold"),
                reason=c.get("reason", ""),
                hard_failure=bool(c.get("hard_failure", False)),
            )
            for c in data.get("checks", [])
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_metadata, temperature_scaling, model_error  # noqa: PLW0603
    model = None
    model_metadata = None
    temperature_scaling = None
    model_error = None

    if not CHECKPOINT_PATH:
        model_error = "no_checkpoint_configured"
        LOGGER.warning("LUMOCULIS_CHECKPOINT is not set or the default checkpoint is missing")
        yield
        return

    try:
        device = _pick_device(DEVICE)
        loaded_model, metadata = _load_checkpoint(CHECKPOINT_PATH, device)
        model = loaded_model
        model_metadata = metadata
        num_classes = int(metadata.get("config", {}).get("num_classes", 5))
        temperature_scaling = _load_temperature(CALIBRATION_PATH, num_classes)
        LOGGER.info("Lumoculis model loaded from %s", CHECKPOINT_PATH)
    except Exception as exc:  # keep startup resilient; /health reports degraded status
        model_error = "model_initialization_failed"
        LOGGER.exception("Failed to load Lumoculis checkpoint: %s", exc)
    yield
    GRADCAM_DIR.cleanup()


app = FastAPI(
    title="Lumoculis DR Grading API",
    version="0.1.0",
    description="Research prototype API for diabetic retinopathy grading. Not a medical device.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/gradcam", StaticFiles(directory=GRADCAM_DIR.name), name="gradcam")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model_error == "no_checkpoint_configured":
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            checkpoint_configured=False,
            device=DEVICE,
            error="no_checkpoint_configured",
        )
    if model_error:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            checkpoint_configured=bool(CHECKPOINT_PATH),
            device=DEVICE,
            error="model_initialization_failed",
        )
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        checkpoint_configured=bool(CHECKPOINT_PATH),
        device=DEVICE,
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if model_error == "no_checkpoint_configured":
        return ModelInfo(
            model_loaded=False,
            checkpoint_configured=False,
            device=DEVICE,
            error="no_checkpoint_configured",
        )
    if model_error:
        return ModelInfo(
            model_loaded=False,
            checkpoint_configured=bool(CHECKPOINT_PATH),
            device=DEVICE,
            error="model_initialization_failed",
        )
    if model is None or model_metadata is None:
        return ModelInfo(
            model_loaded=False,
            checkpoint_configured=bool(CHECKPOINT_PATH),
            device=DEVICE,
            error="model_not_loaded",
        )
    config_metadata = model_metadata.get("config", {}) if isinstance(model_metadata, dict) else {}
    return ModelInfo(
        model_loaded=True,
        checkpoint_configured=True,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        architecture=config_metadata.get("model_name"),
        num_classes=config_metadata.get("num_classes"),
        image_size=config_metadata.get("image_size"),
        epoch=model_metadata.get("epoch"),
        val_qwk=model_metadata.get("val_qwk"),
        calibration_loaded=temperature_scaling is not None,
    )


@dataclass(frozen=True)
class _PredictionParams:
    img: np.ndarray
    filename: str | None
    device_name: str
    run_quality_gate: bool
    quality_required: bool
    gradcam: bool


def _make_inference_config(
    *,
    device_name: str,
    run_quality_gate: bool,
    quality_required: bool,
    gradcam: bool,
    gradcam_output: str | None,
) -> InferenceConfig:
    return InferenceConfig(
        checkpoint_path=CHECKPOINT_PATH,
        image_size=384,
        num_classes=5,
        device=device_name,
        run_quality_gate=run_quality_gate,
        quality_required=quality_required,
        quality_config=QualityConfig(),
        temperature_path=CALIBRATION_PATH,
        gradcam=gradcam,
        gradcam_output=gradcam_output,
    )


def _run_prediction(params: _PredictionParams) -> PredictionResponse:
    img = params.img
    filename = params.filename
    device_name = params.device_name
    run_quality_gate = params.run_quality_gate
    quality_required = params.quality_required
    gradcam = params.gradcam

    if model is None:
        raise HTTPException(status_code=503, detail="model is not available")

    device = _pick_device(device_name)

    input_tensor, rgb_image, width, height = _preprocess_image(img, 384, device)

    quality_result = None
    quality_blocked = False
    if run_quality_gate:
        quality_result, quality_blocked = _run_quality_assessment(
            img,
            _make_inference_config(
                device_name=device_name,
                run_quality_gate=True,
                quality_required=quality_required,
                gradcam=False,
                gradcam_output=None,
            ),
        )
        if quality_required and quality_blocked:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "quality_gate_failed",
                    "quality": _serialize_quality_result(quality_result).model_dump(),
                    "disclaimer": RESEARCH_DISCLAIMER,
                },
            )

    gradcam_path: str | None = None
    if gradcam:
        gradcam_path = str(Path(GRADCAM_DIR.name) / f"{uuid.uuid4().hex}.png")

    try:
        raw_probs_1d, calibrated_probs_1d, referable_prob, predicted_class, referable = (
            _compute_predictions(model, input_tensor, device, temperature_scaling)
        )
    except Exception as exc:  # keep inference errors out of client tracebacks
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail="inference_failed") from exc

    gradcam_saved = None
    gradcam_url = None
    if gradcam and gradcam_path:
        try:
            saved, output_path = _run_gradcam_visualization(
                model,
                input_tensor,
                rgb_image,
                _make_inference_config(
                    device_name=device_name,
                    run_quality_gate=False,
                    quality_required=False,
                    gradcam=True,
                    gradcam_output=gradcam_path,
                ),
                predicted_class,
            )
            gradcam_saved = saved
            if saved and output_path:
                output_path = Path(output_path)
                if output_path.exists():
                    gradcam_url = f"/gradcam/{output_path.name}"
        except Exception:  # Grad-CAM is optional
            LOGGER.exception("Grad-CAM failed")
            gradcam_saved = False

    temperature = (
        float(temperature_scaling.temperature)
        if (temperature_scaling is not None and getattr(temperature_scaling, "fitted", False))
        else None
    )
    temperature_applied = temperature is not None

    return PredictionResponse(
        predicted_class=int(predicted_class),
        predicted_grade=int(predicted_class),
        referable=bool(referable),
        referable_probability=float(referable_prob),
        raw_probabilities=[float(p) for p in raw_probs_1d],
        calibrated_probabilities=(
            [float(p) for p in calibrated_probs_1d] if calibrated_probs_1d is not None else None
        ),
        temperature=temperature,
        temperature_applied=temperature_applied,
        quality=_serialize_quality_result(quality_result) if quality_result is not None else None,
        quality_blocked=bool(quality_blocked),
        gradcam_saved=gradcam_saved,
        gradcam_url=gradcam_url,
        model_checkpoint=CHECKPOINT_PATH,
        model_architecture=(model_metadata.get("config", {}).get("model_name") if model_metadata else None),
        image_width=int(width),
        image_height=int(height),
        filename=filename,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    run_quality_gate: str | None = Form(default="false"),
    quality_required: str | None = Form(default="false"),
    gradcam: str | None = Form(default="false"),
    device: str | None = Form(default=None),
) -> PredictionResponse:
    content = _validate_upload_file(file)
    img = _decode_image(content, file.filename)
    device_name = _parse_device(device)

    return _run_prediction(
        _PredictionParams(
            img=img,
            filename=file.filename,
            device_name=device_name,
            run_quality_gate=_parse_bool(run_quality_gate, False),
            quality_required=_parse_bool(quality_required, False),
            gradcam=_parse_bool(gradcam, False),
        )
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "disclaimer": RESEARCH_DISCLAIMER},
    )
