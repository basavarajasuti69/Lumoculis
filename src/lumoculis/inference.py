"""Inference workflow for Lumoculis.

Integrates model loading, image preprocessing, optional quality gating,
optional temperature-scaling calibration, and optional Grad-CAM
visualization into a single reproducible inference path for DR grade
prediction (grades 0-4).

This is a research-support tool, not a medical device. It does not diagnose
disease, and image-quality thresholds are heuristic, not clinically validated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lumoculis.calibration import TemperatureScaling
    from lumoculis.quality import QualityConfig, QualityResult

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from lumoculis.config import Config
from lumoculis.model import build_model
from lumoculis.preprocess import IMAGENET_MEAN, IMAGENET_STD, load_image, preprocess

NUM_CLASSES = 5
IMAGE_SIZE = 384
REFERRABLE_THRESHOLD_GRADE = 2  # grade >= 2 is "referable DR"

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


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for the inference workflow."""

    checkpoint_path: str | Path
    image_size: int = IMAGE_SIZE
    num_classes: int = NUM_CLASSES
    device: str = "mps"  # "mps", "cpu", or "auto"
    quality_config: QualityConfig | None = None
    run_quality_gate: bool = False
    quality_required: bool = False  # if True, blocked images fail inference
    temperature_path: str | Path | None = None
    gradcam: bool = False
    gradcam_target_class: int | None = None
    gradcam_output: str | Path | None = None
    report_output: str | Path | None = None


@dataclass
class PredictionReport:
    """Structured prediction report."""

    image_path: str | None
    image_width: int | None
    image_height: int | None
    predicted_class: int
    predicted_grade: int
    referable: bool  # grade >= 2
    referable_probability: float  # P(grade >= 2)
    raw_probabilities: list[float]  # softmax of raw logits, length num_classes
    calibrated_probabilities: list[float] | None
    temperature: float | None
    temperature_applied: bool
    quality: QualityResult | None
    quality_blocked: bool
    gradcam_saved: bool | None
    gradcam_output_path: str | None
    model_checkpoint: str
    model_architecture: str
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "predicted_class": self.predicted_class,
            "predicted_grade": self.predicted_grade,
            "referable": self.referable,
            "referable_probability": round(self.referable_probability, 6),
            "raw_probabilities": [round(p, 6) for p in self.raw_probabilities],
            "calibrated_probabilities": (
                [round(p, 6) for p in self.calibrated_probabilities]
                if self.calibrated_probabilities is not None
                else None
            ),
            "temperature": round(self.temperature, 6) if self.temperature is not None else None,
            "temperature_applied": self.temperature_applied,
            "quality_blocked": self.quality_blocked,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "gradcam_saved": self.gradcam_saved,
            "gradcam_output_path": (
                str(self.gradcam_output_path) if self.gradcam_output_path else None
            ),
            "model_checkpoint": self.model_checkpoint,
            "model_architecture": self.model_architecture,
            "disclaimer": self.disclaimer,
        }

    def save(self, path: str | Path) -> None:
        """Save the report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def _pick_device(device: str) -> torch.device:
    """Select torch device from a user-specified string."""
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a model from a training checkpoint.

    Returns (model, checkpoint_metadata). The model is built from the config
    stored in the checkpoint (falling back to defaults) and its state dict is
    loaded. Raises FileNotFoundError / ValueError with clear messages on
    missing or incompatible checkpoints.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:  # surface a clear error
        raise ValueError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a dict; expected keys "
            "'model_state_dict' and 'config'."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'model_state_dict' key.")

    # Rebuild config from checkpoint, with defaults for missing keys.
    raw_config = checkpoint.get("config", {})
    if isinstance(raw_config, dict):
        config = Config(
            image_size=int(raw_config.get("image_size", IMAGE_SIZE)),
            batch_size=1,
            epochs=1,
            lr=0.0,
            model_name=str(raw_config.get("model_name", "efficientnet_b0")),
            seed=int(raw_config.get("seed", 42)),
            num_classes=int(raw_config.get("num_classes", NUM_CLASSES)),
        )
    else:
        config = Config(
            image_size=IMAGE_SIZE,
            batch_size=1,
            epochs=1,
            lr=0.0,
            model_name="efficientnet_b0",
            seed=42,
            num_classes=NUM_CLASSES,
        )

    model = build_model(config.model_name, config.num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "val_qwk": checkpoint.get("val_qwk"),
        "config": {
            "image_size": config.image_size,
            "model_name": config.model_name,
            "num_classes": config.num_classes,
        },
    }
    del checkpoint
    return model, metadata


def _load_temperature(
    temperature_path: str | Path | None,
    num_classes: int,
) -> TemperatureScaling | None:
    """Load a fitted TemperatureScaling from JSON, or return None.

    Does NOT fit any parameters; only loads previously saved metadata.
    Fabricates nothing: returns None if the file is missing or unreadable.
    """
    if temperature_path is None:
        return None
    path = Path(temperature_path)
    if not path.exists():
        return None
    try:
        from lumoculis.calibration import TemperatureScaling  # noqa: PLC0415

        return TemperatureScaling.load(path)
    except Exception:  # do not fabricate calibration metadata
        return None


def _preprocess_image(
    image: np.ndarray | str | Path,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, int, int]:
    """Preprocess an image for model input.

    Uses the project's existing preprocessing conventions (crop black borders,
    resize, Ben Graham) followed by ToTensor + ImageNet normalization, matching
    the training dataset pipeline in data.py.

    Returns (input_tensor, uint8_rgb_image, width, height).
    Raises ValueError if the image cannot be loaded or is invalid.
    """
    # Accept either a file path or an in-memory array.
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = load_image(image)
        if img is None:
            raise ValueError(f"Could not load image: {image}")

    height, width = img.shape[:2]

    # Preprocess with project conventions: crop, resize, Ben Graham.
    processed = preprocess(img, size=image_size)

    # Convert BGR (OpenCV) -> RGB (PIL/torchvision convention).
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # ToTensor + ImageNet normalization, same as data.py __getitem__.
    # Avoid importing torchvision here to keep the API process smaller.
    tensor = torch.from_numpy(processed_rgb).permute(2, 0, 1).contiguous()
    tensor = tensor.float().div(255.0)
    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t

    input_tensor = tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    return input_tensor, processed_rgb, width, height


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _predict_probs(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run the model and return raw class probabilities (shape (1, num_classes))."""
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def _resolve_model_and_temperature(
    config: InferenceConfig,
    device: torch.device,
) -> tuple[nn.Module, str, int, TemperatureScaling | None]:
    """Load the model and optional temperature scaling from config.

    Returns (model, model_name, num_classes, temperature_scaling).
    """
    model, ckpt_metadata = _load_checkpoint(config.checkpoint_path, device)
    model_name = ckpt_metadata["config"]["model_name"]
    num_classes = ckpt_metadata["config"]["num_classes"]

    temperature_scaling = _load_temperature(config.temperature_path, num_classes)
    return model, model_name, num_classes, temperature_scaling


def _run_quality_assessment(
    image: np.ndarray | str | Path,
    config: InferenceConfig,
) -> tuple[QualityResult | None, bool]:
    """Run the optional quality gate.

    Returns (quality_result, quality_blocked).
    """
    quality_result: QualityResult | None = None
    quality_blocked = False
    if config.run_quality_gate:
        from lumoculis.quality import QualityConfig, assess_quality  # noqa: PLC0415

        q_config = config.quality_config or QualityConfig()
        quality_result = assess_quality(image, config=q_config)
        if config.quality_required and quality_result.status != "pass":
            quality_blocked = True
    return quality_result, quality_blocked


def _compute_predictions(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    temperature_scaling: TemperatureScaling | None,
) -> tuple[np.ndarray, np.ndarray | None, float, int, bool]:
    """Run the model and compute raw and calibrated predictions.

    Returns (raw_probs_1d, calibrated_probs_1d_or_None, referable_prob,
             predicted_class, referable).
    """
    raw_probs = _predict_probs(model, input_tensor, device)
    raw_probs_1d = raw_probs[0]
    referable_prob_raw = float(raw_probs_1d[REFERRABLE_THRESHOLD_GRADE:].sum())

    calibrated_probs_1d: np.ndarray | None = None
    temperature_applied = temperature_scaling is not None and temperature_scaling.fitted
    if temperature_applied:
        model.eval()
        with torch.inference_mode():
            logits = model(input_tensor)
        logits_np = logits.cpu().numpy()
        calibrated_probs = temperature_scaling.apply(logits_np)
        calibrated_probs_1d = calibrated_probs[0]

    if calibrated_probs_1d is not None:
        probs_for_pred = calibrated_probs_1d
        referable_prob = float(calibrated_probs_1d[REFERRABLE_THRESHOLD_GRADE:].sum())
    else:
        probs_for_pred = raw_probs_1d
        referable_prob = referable_prob_raw

    predicted_class = int(probs_for_pred.argmax())
    referable = bool(predicted_class >= REFERRABLE_THRESHOLD_GRADE)

    return raw_probs_1d, calibrated_probs_1d, referable_prob, predicted_class, referable


def _run_gradcam_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    config: InferenceConfig,
    predicted_class: int,
) -> tuple[bool | None, str | Path | None]:
    """Run optional Grad-CAM visualization.

    Returns (gradcam_saved, gradcam_output_path).
    """
    gradcam_saved: bool | None = None
    gradcam_output_path: str | Path | None = None
    if config.gradcam:
        try:
            from lumoculis.explain import GradCAM, save_gradcam_visualization  # noqa: PLC0415

            with GradCAM(model, target_layer=None) as gradcam:
                heatmap, target_class = gradcam.compute_heatmap(
                    input_tensor,
                    target_class=config.gradcam_target_class,
                )
            if config.gradcam_output is not None:
                output_path = Path(config.gradcam_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_gradcam_visualization(
                    original_image=rgb_image,
                    heatmap=heatmap,
                    output_path=output_path,
                    predicted_class=predicted_class,
                    target_class=target_class,
                )
                gradcam_saved = True
                gradcam_output_path = output_path
            else:
                gradcam_saved = False
        except Exception:  # Grad-CAM is optional
            gradcam_saved = False
    return gradcam_saved, gradcam_output_path


def run_inference(
    image: np.ndarray | str | Path,
    config: InferenceConfig,
) -> PredictionReport:
    """Run the full inference workflow on a single image.

    Parameters
    ----------
    image : np.ndarray | str | Path
        A uint8 HWC image array (BGR or grayscale) or a path to an image file.
    config : InferenceConfig
        Inference configuration.

    Returns
    -------
    PredictionReport
        Structured report with raw and calibrated probabilities, quality
        assessment (if requested), and optional Grad-CAM output.
    """
    device = _pick_device(config.device)

    model, model_name, _num_classes, temperature_scaling = _resolve_model_and_temperature(
        config, device
    )

    input_tensor, rgb_image, width, height = _preprocess_image(image, config.image_size, device)

    quality_result, quality_blocked = _run_quality_assessment(image, config)

    (
        raw_probs_1d,
        calibrated_probs_1d,
        referable_prob,
        predicted_class,
        referable,
    ) = _compute_predictions(model, input_tensor, device, temperature_scaling)

    temperature = (
        temperature_scaling.temperature
        if (temperature_scaling is not None and temperature_scaling.fitted)
        else None
    )
    temperature_applied = temperature is not None

    gradcam_saved, gradcam_output_path = _run_gradcam_visualization(
        model, input_tensor, rgb_image, config, predicted_class
    )

    return PredictionReport(
        image_path=str(image) if isinstance(image, str | Path) else None,
        image_width=width,
        image_height=height,
        predicted_class=predicted_class,
        predicted_grade=predicted_class,
        referable=referable,
        referable_probability=referable_prob,
        raw_probabilities=[float(p) for p in raw_probs_1d],
        calibrated_probabilities=(
            [float(p) for p in calibrated_probs_1d] if calibrated_probs_1d is not None else None
        ),
        temperature=temperature,
        temperature_applied=temperature_applied,
        quality=quality_result,
        quality_blocked=quality_blocked,
        gradcam_saved=gradcam_saved,
        gradcam_output_path=gradcam_output_path,
        model_checkpoint=str(config.checkpoint_path),
        model_architecture=model_name,
        disclaimer=RESEARCH_DISCLAIMER,
    )


def run_inference_from_paths(
    image_path: str | Path,
    checkpoint_path: str | Path,
    config: InferenceConfig | None = None,
    **kwargs: Any,
) -> PredictionReport:
    """Convenience wrapper for path-based inference.

    Accepts an optional :class:`InferenceConfig`. Additional keyword arguments
    are passed to the :class:`InferenceConfig` constructor when no config is
    supplied.
    """
    if config is None:
        config = InferenceConfig(
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
    else:
        config.checkpoint_path = checkpoint_path

    report = run_inference(image_path, config)
    if config.report_output is not None:
        report.save(config.report_output)
    return report
