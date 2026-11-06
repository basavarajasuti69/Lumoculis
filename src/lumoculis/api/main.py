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

import io
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

RESEARCH_DISCLAIMER = (
    "Lumoculis is a research prototype, not a medical device. Predictions are "
    "not a diagnosis and are not a substitute for qualified clinical "
    "assessment. Image-quality thresholds are heuristic and not clinically "
    "validated. Grad-CAM highlights model-relevant regions but does not prove "
    "disease location or causality. Calibration improves probability reliability "
    "only to the extent supported by appropriate held-out data; it does not "
    "establish clinical reliability. APTOS 2019 does not provide patient IDs, "
    "and grouping exact duplicate images does not guarantee patient-level "
    "independence. The system has not been clinically validated."
)

BACKEND = os.environ.get("LUMOCULIS_BACKEND", "pytorch").strip().lower()
if BACKEND not in {"pytorch", "onnx"}:
    raise RuntimeError("LUMOCULIS_BACKEND must be 'pytorch' or 'onnx'")

if BACKEND == "pytorch":

    from lumoculis.inference import (
        RESEARCH_DISCLAIMER as _PT_RESEARCH_DISCLAIMER,
    )
    from lumoculis.inference import (
        _compute_predictions,
        _load_checkpoint,
        _load_temperature,
        _pick_device,
        _preprocess_image,
        _run_gradcam_visualization,
        _run_quality_assessment,
    )

    # Keep the shared disclaimer identical to the PyTorch implementation.
    RESEARCH_DISCLAIMER = _PT_RESEARCH_DISCLAIMER
else:
    _compute_predictions = None
    _load_checkpoint = None
    _load_temperature = None
    _pick_device = None
    _preprocess_image = None
    _run_gradcam_visualization = None
    _run_quality_assessment = None

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
ONNX_MODEL_PATH = os.environ.get("LUMOCULIS_ONNX_MODEL") or "models/best.onnx"
DEVICE = os.environ.get("LUMOCULIS_DEVICE", DEFAULT_DEVICE)
MAX_UPLOAD_MB = float(os.environ.get("LUMOCULIS_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)))
MAX_IMAGE_PIXELS = int(os.environ.get("LUMOCULIS_MAX_IMAGE_PIXELS", "12500000"))
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




def _image_dimensions(content: bytes) -> tuple[int, int] | None:
    """Return (width, height) from image headers without full decoding when possible."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _decode_image(content: bytes, filename: str) -> np.ndarray:
    dimensions = _image_dimensions(content)
    if dimensions is not None:
        width, height = dimensions
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "image_too_large",
                    "message": (
                        f"Image dimensions {width}x{height} "
                        f"({width * height} pixels) exceed the maximum of "
                        f"{MAX_IMAGE_PIXELS} pixels."
                    ),
                },
            )

    array = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="could not decode image file")

    if img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "image_too_large",
                "message": (
                    f"Image dimensions {img.shape[1]}x{img.shape[0]} "
                    f"({img.shape[0] * img.shape[1]} pixels) exceed the maximum of "
                    f"{MAX_IMAGE_PIXELS} pixels."
                ),
            },
        )
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
async def lifespan(app: FastAPI):  # noqa: PLR0915
    global model, model_metadata, temperature_scaling, model_error  # noqa: PLW0603
    model = None
    model_metadata = None
    temperature_scaling = None
    model_error = None

    if BACKEND == "onnx":
        if not ONNX_MODEL_PATH or not Path(ONNX_MODEL_PATH).exists():
            model_error = "no_onnx_model_configured"
            LOGGER.warning("LUMOCULIS_ONNX_MODEL is not set or the ONNX model is missing")
            yield
            return
        try:
            from lumoculis.onnx_inference import load_onnx_model, load_temperature  # noqa: PLC0415

            loaded_model = load_onnx_model(ONNX_MODEL_PATH)
            model = loaded_model
            model_metadata = {"config": {"model_name": "efficientnet_b0", "num_classes": 5}, "epoch": None, "val_qwk": None}
            temperature_scaling = load_temperature(CALIBRATION_PATH, 5)
            import gc  # noqa: PLC0415

            gc.collect()
            try:
                import ctypes  # noqa: PLC0415

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass
            LOGGER.info("Lumoculis ONNX model loaded from %s", ONNX_MODEL_PATH)
        except Exception as exc:
            model_error = "model_initialization_failed"
            LOGGER.exception("Failed to load ONNX model: %s", exc)
        yield
        GRADCAM_DIR.cleanup()
        return

    if not CHECKPOINT_PATH:
        model_error = "no_checkpoint_configured"
        LOGGER.warning("LUMOCULIS_CHECKPOINT is not set or the default checkpoint is missing")
        yield
        return

    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        device = _pick_device(DEVICE)
        loaded_model, metadata = _load_checkpoint(CHECKPOINT_PATH, device)
        model = loaded_model
        model_metadata = metadata
        num_classes = int(metadata.get("config", {}).get("num_classes", 5))
        temperature_scaling = _load_temperature(CALIBRATION_PATH, num_classes)
        import gc  # noqa: PLC0415

        gc.collect()
        try:
            import ctypes  # noqa: PLC0415

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        LOGGER.info("Lumoculis model loaded from %s", CHECKPOINT_PATH)
    except Exception as exc:
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
    if model_error in {"no_checkpoint_configured", "no_onnx_model_configured"}:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            checkpoint_configured=False,
            device=DEVICE,
            error=model_error,
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
    if model_error in {"no_checkpoint_configured", "no_onnx_model_configured"}:
        return ModelInfo(
            model_loaded=False,
            checkpoint_configured=False,
            device=DEVICE,
            error=model_error,
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
        checkpoint_path=(ONNX_MODEL_PATH if BACKEND == "onnx" else CHECKPOINT_PATH),
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


def _run_quality_assessment(image, config):
    """Compatibility wrapper for the PyTorch quality gate."""
    from lumoculis.inference import _run_quality_assessment as _pt_quality  # noqa: PLC0415

    return _pt_quality(image, config)


def _make_inference_config(
    *,
    device_name: str,
    run_quality_gate: bool,
    quality_required: bool,
    gradcam: bool,
    gradcam_output: str | None,
):
    from lumoculis.inference import InferenceConfig  # noqa: PLC0415
    from lumoculis.quality import QualityConfig  # noqa: PLC0415

    return InferenceConfig(
        checkpoint_path=(ONNX_MODEL_PATH if BACKEND == "onnx" else CHECKPOINT_PATH),
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


def _run_prediction(params: _PredictionParams) -> PredictionResponse:  # noqa: PLR0912, PLR0915
    img = params.img
    filename = params.filename
    device_name = params.device_name
    run_quality_gate = params.run_quality_gate
    quality_required = params.quality_required
    gradcam = params.gradcam

    if model is None:
        raise HTTPException(status_code=503, detail="model is not available")

    if BACKEND == "onnx":
        if gradcam:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "gradcam_not_available",
                    "message": "Grad-CAM is not available with the ONNX backend.",
                },
            )

        from lumoculis.onnx_inference import predict, prepare_input  # noqa: PLC0415
        from lumoculis.quality import QualityConfig, assess_quality  # noqa: PLC0415

        quality_result = None
        quality_blocked = False
        if run_quality_gate:
            quality_result = assess_quality(img, QualityConfig())
            if quality_required and quality_result.status != "pass":
                quality_blocked = True
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "quality_gate_failed",
                        "quality": _serialize_quality_result(quality_result).model_dump(),
                        "disclaimer": RESEARCH_DISCLAIMER,
                    },
                )

        input_array, rgb_image, width, height = prepare_input(img, 384)
        img = None

        try:
            raw_probs_1d, calibrated_probs_1d, referable_prob, predicted_class, referable = predict(
                model, input_array, temperature_scaling
            )
        except Exception as exc:
            LOGGER.exception("ONNX inference failed")
            raise HTTPException(status_code=500, detail="inference_failed") from exc

        input_array = None
        rgb_image = None

        import gc  # noqa: PLC0415

        gc.collect()
        try:
            import ctypes  # noqa: PLC0415

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

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
            gradcam_saved=None,
            gradcam_url=None,
            model_checkpoint=ONNX_MODEL_PATH,
            model_architecture=(model_metadata.get("config", {}).get("model_name") if model_metadata else None),
            image_width=int(width),
            image_height=int(height),
            filename=filename,
        )

    # PyTorch backend.
    from lumoculis.inference import _pick_device  # noqa: PLC0415

    device = _pick_device(device_name)

    input_tensor, rgb_image, width, height = _preprocess_image(img, 384, device)
    img = None  # release the original decoded image

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

    if not gradcam:
        input_tensor = None
        rgb_image = None

    import gc  # noqa: PLC0415

    gc.collect()
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    if not gradcam:
        img = None
        rgb_image = None

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
