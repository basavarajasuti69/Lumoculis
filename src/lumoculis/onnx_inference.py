"""ONNX Runtime inference backend for Lumoculis.

This module provides a low-memory alternative to the PyTorch inference path.
It preserves the existing preprocessing, normalization, probability and
referable-DR semantics. Grad-CAM is not available in this backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from lumoculis.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess

NUM_CLASSES = 5
IMAGE_SIZE = 384
REFERRABLE_THRESHOLD_GRADE = 2


def load_onnx_model(path: str | Path) -> ort.InferenceSession:
    """Load an ONNX model with constrained CPU threading."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if model_path.stat().st_size == 0:
        raise ValueError(f"ONNX model is empty: {model_path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def prepare_input(image: np.ndarray, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Preprocess an image exactly as the PyTorch path does."""
    height, width = image.shape[:2]
    processed = preprocess(image, size=image_size)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    array = processed_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    array = (array - mean) / std
    return array[np.newaxis, ...], processed_rgb, width, height


def predict(
    session: ort.InferenceSession,
    input_array: np.ndarray,
    temperature_scaling: Any = None,
) -> tuple[np.ndarray, np.ndarray | None, float, int, bool]:
    """Run ONNX inference and return the same tuple as _compute_predictions."""
    logits = session.run(["logits"], {"input": input_array})[0]
    logits = np.asarray(logits, dtype=np.float64)

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    raw_probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    raw_probs_1d = raw_probs[0]
    referable_prob_raw = float(raw_probs_1d[REFERRABLE_THRESHOLD_GRADE:].sum())

    calibrated_probs_1d: np.ndarray | None = None
    temperature_applied = temperature_scaling is not None and getattr(temperature_scaling, "fitted", False)
    if temperature_applied:
        calibrated_probs = temperature_scaling.apply(logits)
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


def load_temperature(path: str | None, num_classes: int = NUM_CLASSES):
    """Load temperature scaling if available."""
    if not path:
        return None
    try:
        from lumoculis.calibration import TemperatureScaling  # noqa: PLC0415

        temperature_path = Path(path)
        if not temperature_path.exists():
            return None
        return TemperatureScaling.load(temperature_path)
    except Exception:
        return None
