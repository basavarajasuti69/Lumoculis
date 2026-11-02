"""Lumoculis Streamlit web application.

Provides a web interface for diabetic retinopathy (DR) grade inference using
the existing Lumoculis inference, quality-gate, calibration, and Grad-CAM
APIs. This is a research prototype, not a medical device.

Configuration is provided through environment variables:

- LUMOCULIS_CHECKPOINT: path to a training checkpoint (.pt).
- LUMOCULIS_CALIBRATION: path to a fitted temperature JSON (optional).
- LUMOCULIS_ENABLE_QUALITY_GATE: "1"/"true" to enable quality gating.
- LUMOCULIS_QUALITY_REQUIRED: "1"/"true" to block inference on quality failure.
- LUMOCULIS_DEVICE: "cpu", "mps", or "auto" (default "cpu").
- LUMOCULIS_MAX_UPLOAD_MB: maximum upload size in MB (default 10).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from torch import nn
from torchvision import transforms

from lumoculis.calibration import TemperatureScaling
from lumoculis.data import IMAGENET_MEAN, IMAGENET_STD
from lumoculis.explain import GradCAM, save_gradcam_visualization
from lumoculis.inference import (
    NUM_CLASSES,
    REFERRABLE_THRESHOLD_GRADE,
    RESEARCH_DISCLAIMER,
    _load_checkpoint,
)
from lumoculis.preprocess import preprocess
from lumoculis.quality import QualityConfig, QualityResult, assess_quality

GRADE_NAMES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_UPLOAD_MB = 10
DEFAULT_IMAGE_SIZE = 384
TEMP_FILE_PREFIX = "lumoculis_gradcam_"


@dataclass
class PredictionContext:
    """Context for a single inference run."""

    raw_probs: np.ndarray
    calibrated_probs: np.ndarray | None
    temperature: float | None
    metadata: dict | None
    predicted_class: int
    referable: bool
    referable_prob: float
    quality_result: QualityResult | None
    quality_blocked: bool
    effective_checkpoint: str
    image_width: int | None = None
    image_height: int | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable."""
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@st.cache_resource(show_spinner="Loading model checkpoint...")
def _load_model_cached(
    checkpoint_path: str,
    device: str,
) -> tuple[nn.Module, dict | None, str | None]:
    """Load a model from a checkpoint, cached by Streamlit.

    Returns (model, metadata_or_none, error_message_or_none).
    """
    try:
        dev = torch.device(device)
        model, metadata = _load_checkpoint(checkpoint_path, dev)
        return model, metadata, None
    except FileNotFoundError as exc:
        return None, None, f"Checkpoint not found: {exc}"
    except ValueError as exc:
        return None, None, f"Incompatible checkpoint: {exc}"
    except Exception as exc:
        return None, None, f"Failed to load checkpoint: {exc}"


@st.cache_resource(show_spinner=False)
def _load_temperature_cached(
    calibration_path: str | None,
    num_classes: int,
) -> tuple[TemperatureScaling | None, str | None]:
    """Load a fitted temperature scaling from JSON, cached by Streamlit.

    Returns (temperature_scaling_or_none, error_message_or_none).
    """
    if not calibration_path:
        return None, None
    path = Path(calibration_path)
    if not path.exists():
        return None, f"Calibration file not found: {calibration_path}"
    try:
        ts = TemperatureScaling.load(path)
        return ts, None
    except Exception as exc:
        return None, f"Could not load calibration: {exc}"


def _decode_uploaded(uploaded_file) -> np.ndarray | None:
    """Decode an uploaded Streamlit file into a BGR numpy array."""
    try:
        file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _preprocess_for_inference(
    img: np.ndarray,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, int, int] | None:
    """Preprocess an image for model input.

    Returns (input_tensor, rgb_image, width, height) or None on failure.
    """
    try:
        height, width = img.shape[:2]
        processed = preprocess(img, size=image_size)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        tensor = transforms.ToTensor()(processed_rgb)
        mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t
        input_tensor = tensor.unsqueeze(0).to(device)

        return input_tensor, processed_rgb, width, height
    except Exception:
        return None


def _predict_probs(
    model: nn.Module,
    input_tensor: torch.Tensor,
    temperature_scaling: TemperatureScaling | None,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Run the model and return raw and calibrated probabilities.

    Returns (raw_probs_1d, calibrated_probs_1d_or_none, temperature_or_none).
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    calibrated_probs_1d = None
    temperature = None
    if temperature_scaling is not None and temperature_scaling.fitted:
        temperature = temperature_scaling.temperature
        calibrated = temperature_scaling.apply(logits.cpu().numpy())
        calibrated_probs_1d = calibrated[0]

    return raw_probs, calibrated_probs_1d, temperature


def _run_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    output_path: Path,
) -> tuple[bool, str | None]:
    """Run optional Grad-CAM visualization.

    Returns (success, output_path_or_none).
    """
    try:
        with GradCAM(model, target_layer=None) as gradcam:
            heatmap, target_class_used = gradcam.compute_heatmap(
                input_tensor,
                target_class=None,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_gradcam_visualization(
            original_image=rgb_image,
            heatmap=heatmap,
            output_path=output_path,
            predicted_class=target_class_used,
            target_class=target_class_used,
        )
        return True, str(output_path)
    except Exception:
        return False, None


def _display_quality_result(quality: QualityResult | None) -> None:
    """Display a quality-gate result in the Streamlit UI."""
    if quality is None:
        return

    if quality.status == "pass":
        st.success(f"Quality check: **PASS** ({quality.image_width}x{quality.image_height})")
    elif quality.status == "warning":
        st.warning(f"Quality check: **WARNING** ({quality.image_width}x{quality.image_height})")
    else:
        st.error(f"Quality check: **FAIL** ({quality.image_width}x{quality.image_height})")

    with st.expander("Quality check details", expanded=False):
        for check in quality.checks:
            icon = "✅" if check.status == "pass" else "⚠️" if check.status == "warning" else "❌"
            st.write(
                f"{icon} **{check.name}**: {check.reason} "
                f"(measured={check.measured}, threshold={check.threshold})"
            )
        if quality.reasons:
            st.write("**Reasons:** " + "; ".join(quality.reasons))


def _render_sidebar() -> dict:
    """Render the sidebar configuration and return user overrides."""
    with st.sidebar:
        st.header("Configuration")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value=os.environ.get("LUMOCULIS_CHECKPOINT", ""),
            key="checkpoint_path_input",
            help="Set LUMOCULIS_CHECKPOINT env var to configure.",
        )
        calibration_path = st.text_input(
            "Calibration file (optional)",
            value=os.environ.get("LUMOCULIS_CALIBRATION", ""),
            key="calibration_path_input",
            help="Set LUMOCULIS_CALIBRATION env var.",
        )
        enable_quality_gate = st.checkbox(
            "Enable quality gate",
            value=_env_bool("LUMOCULIS_ENABLE_QUALITY_GATE", False),
            key="enable_quality_gate_input",
            help="Set LUMOCULIS_ENABLE_QUALITY_GATE=1.",
        )
        quality_required = st.checkbox(
            "Require quality pass to infer",
            value=_env_bool("LUMOCULIS_QUALITY_REQUIRED", False),
            key="quality_required_input",
            help="Set LUMOCULIS_QUALITY_REQUIRED=1.",
        )
        device = st.selectbox(
            "Device",
            options=["cpu", "mps", "auto"],
            index=["cpu", "mps", "auto"].index(
                os.environ.get("LUMOCULIS_DEVICE", DEFAULT_DEVICE)
                if os.environ.get("LUMOCULIS_DEVICE", DEFAULT_DEVICE) in ("cpu", "mps", "auto")
                else 0
            ),
            key="device_input",
        )
        max_upload_mb = st.number_input(
            "Max upload size (MB)",
            min_value=1,
            max_value=100,
            value=int(_env_float("LUMOCULIS_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)),
            key="max_upload_mb_input",
        )

    return {
        "checkpoint_path": checkpoint_path,
        "calibration_path": calibration_path,
        "enable_quality_gate": enable_quality_gate,
        "quality_required": quality_required,
        "device": device,
        "max_upload_mb": max_upload_mb,
    }


def _render_disclaimer() -> None:
    """Render the research-prototype disclaimer."""
    st.error(
        "⚠️ **Research prototype — not a medical device.** Predictions are not a "
        "diagnosis and are not a substitute for qualified clinical assessment. "
        "Image-quality thresholds are heuristic and not clinically validated. "
        "Grad-CAM highlights model-relevant regions but does not prove disease "
        "location or causality. The system has not been clinically validated."
    )


def _render_prediction_results(ctx: PredictionContext) -> dict:
    """Render prediction results and return the report dict."""
    st.subheader("Prediction results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted grade", f"Grade {ctx.predicted_class}")
        st.metric("Referable DR (grade ≥ 2)", "Yes" if ctx.referable else "No")
        st.metric("P(referable)", f"{ctx.referable_prob:.4f}")

    with col2:
        st.metric(
            "Model",
            ctx.metadata["config"]["model_name"] if ctx.metadata else "unknown",
        )
        if ctx.temperature is not None:
            st.metric("Temperature applied", f"{ctx.temperature:.4f}")
        else:
            st.metric("Calibration", "Not applied")

    # Probability chart.
    st.write("### Class probabilities")
    prob_data = []
    for i, name in enumerate(GRADE_NAMES):
        raw_p = float(ctx.raw_probs[i])
        cal_p = float(ctx.calibrated_probs[i]) if ctx.calibrated_probs is not None else None
        prob_data.append({
            "Grade": name,
            "Raw probability": raw_p,
            "Calibrated probability": cal_p if cal_p is not None else raw_p,
        })
    st.bar_chart(
        {row["Grade"]: row["Raw probability"] for row in prob_data},
        use_container_width=True,
    )

    # Show calibrated probabilities separately when available.
    if ctx.calibrated_probs is not None:
        st.write("### Calibrated probabilities (temperature scaling applied)")
        st.bar_chart(
            {row["Grade"]: row["Calibrated probability"] for row in prob_data},
            use_container_width=True,
        )

    report = {
        "predicted_class": ctx.predicted_class,
        "predicted_grade": ctx.predicted_class,
        "referable": ctx.referable,
        "referable_probability": round(ctx.referable_prob, 6),
        "raw_probabilities": [round(float(p), 6) for p in ctx.raw_probs],
        "calibrated_probabilities": (
            [round(float(p), 6) for p in ctx.calibrated_probs]
            if ctx.calibrated_probs is not None
            else None
        ),
        "temperature": round(ctx.temperature, 6) if ctx.temperature is not None else None,
        "temperature_applied": ctx.temperature is not None,
        "quality_blocked": ctx.quality_blocked,
        "quality": (ctx.quality_result.to_dict() if ctx.quality_result is not None else None),
        "model_checkpoint": ctx.effective_checkpoint,
        "model_architecture": (ctx.metadata["config"]["model_name"] if ctx.metadata else "unknown"),
        "image_width": ctx.image_width,
        "image_height": ctx.image_height,
        "disclaimer": RESEARCH_DISCLAIMER,
    }
    return report


def _handle_upload(uploaded_file, max_upload_mb: float):
    """Handle uploaded file validation and decoding."""
    uploaded_file.seek(0, 2)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    if file_size_mb > max_upload_mb:
        st.error(f"File too large: {file_size_mb:.1f} MB exceeds limit of {max_upload_mb:.0f} MB.")
        return None
    img = _decode_uploaded(uploaded_file)
    if img is None:
        st.error("Could not decode the uploaded image. Please upload a valid PNG or JPEG file.")
        return None
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded image",
        use_container_width=True,
    )
    return img




def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lumo-bg: #181818;
            --lumo-surface: #303030;
            --lumo-accent: #DA291C;
            --lumo-accent-active: #B01E0A;
            --lumo-text: #FFFFFF;
            --lumo-muted: #969696;
            --lumo-dim: #666666;
            --lumo-border: #303030;
            --lumo-info: #4C98B9;
        }
        .stApp { background: var(--lumo-bg); color: var(--lumo-text); }
        .block-container { max-width: 1280px; padding-top: 2rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        .lumo-hero { padding: 6rem 0 4rem; }
        .lumo-eyebrow { text-transform: uppercase; letter-spacing: 1.1px; color: var(--lumo-muted); font-size: 0.75rem; }
        .lumo-hero-title { font-size: clamp(2.5rem, 6vw, 5rem); font-weight: 500; line-height: 1.05; }
        .lumo-card { border: 1px solid var(--lumo-border); padding: 1.5rem; background: var(--lumo-surface); }
        .stButton>button { border-radius: 0; text-transform: uppercase; letter-spacing: 1.4px; background: var(--lumo-accent); color: #fff; min-height: 48px; }
        .stButton>button:hover { background: var(--lumo-accent-active); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_overview() -> None:
    st.markdown('<div class="lumo-hero"><div class="lumo-eyebrow">AI-POWERED RETINAL IMAGE ANALYSIS</div><h1 class="lumo-hero-title">See Retinal Images Through the Lens of AI.</h1><p>Explore a deep-learning research prototype that classifies retinal fundus images into five diabetic retinopathy severity grades.</p></div>', unsafe_allow_html=True)
    st.markdown('<span style="border:1px solid #303030;padding:4px 8px;font-size:0.75rem;">RESEARCH PROTOTYPE</span>', unsafe_allow_html=True)
    st.markdown("### Technical highlights")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="lumo-card"><strong>FIVE-CLASS CLASSIFICATION</strong><br>Predicts a diabetic retinopathy severity grade from 0 to 4.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="lumo-card"><strong>EXPLAINABLE AI</strong><br>Displays Grad-CAM visualizations when supported.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="lumo-card"><strong>IMAGE QUALITY</strong><br>Provides heuristic image-quality assessment when supported.</div>', unsafe_allow_html=True)
    c4.markdown('<div class="lumo-card"><strong>MODEL EVALUATION</strong><br>Presents available research metrics and evaluation information.</div>', unsafe_allow_html=True)
    st.markdown("### How it works")
    st.markdown("01 — UPLOAD: Select a supported retinal fundus image.")
    st.markdown("02 — ANALYZE: Run the existing deep-learning inference pipeline.")
    st.markdown("03 — REVIEW: Inspect the predicted grade, probability distribution, and supported visualizations.")
    st.markdown("### Responsible AI")
    st.warning("Lumoculis is an experimental research prototype. It is not clinically validated and must not replace professional medical assessment.")


def _render_model_info() -> None:
    st.subheader("Model information")
    st.write("Architecture: EfficientNet-B0")
    st.write("Framework: PyTorch")
    st.write("Dataset: APTOS 2019")
    st.write("Output classes: 5 (grades 0-4)")
    st.write("Referable DR threshold: grade >= 2")
    st.write("Grad-CAM: supported")
    st.write("Probability calibration: supported")
    st.write("Known limitations: research prototype; not clinically validated.")


def _render_about() -> None:
    st.subheader("About")
    st.write("Lumoculis is a research prototype for AI-assisted diabetic retinopathy screening support.")
    st.write(RESEARCH_DISCLAIMER)
    st.caption("© 2026 Lumoculis. Research prototype, not a medical device.")


def _run_analysis_app(config: dict) -> None:  # noqa: PLR0911, PLR0912, PLR0915
    """Run the existing image analysis workflow."""

    st.set_page_config(
        page_title="Lumoculis - DR Screening Support",
        page_icon="👁️",
        layout="wide",
    )

    st.title("Lumoculis")
    st.caption("AI-assisted diabetic retinopathy screening support (research prototype)")

    _render_disclaimer()
    config = _render_sidebar()
    st.session_state["device"] = config["device"]

    st.subheader("Upload a retinal fundus image")
    uploaded_file = st.file_uploader(
        "Choose a PNG or JPEG image",
        type=["png", "jpg", "jpeg"],
        help=f"Maximum file size: {config['max_upload_mb']:.0f} MB.",
    )
    if uploaded_file is None:
        return

    img = _handle_upload(uploaded_file, config["max_upload_mb"])
    if img is None:
        return

    effective_checkpoint = config["checkpoint_path"]
    if not effective_checkpoint:
        st.info(
            "No model checkpoint configured. Set the `LUMOCULIS_CHECKPOINT` "
            "environment variable to a `.pt` checkpoint produced by "
            "`lumoculis.train`, or enter a path in the sidebar."
        )
        return
    if not Path(effective_checkpoint).exists():
        st.error(f"Checkpoint not found: {effective_checkpoint}")
        return

    model, metadata, load_error = _load_model_cached(effective_checkpoint, config["device"])
    if load_error:
        st.error(load_error)
        return

    effective_calibration = config["calibration_path"] or ""
    temperature_scaling, cal_error = _load_temperature_cached(
        effective_calibration or None, NUM_CLASSES
    )
    if cal_error:
        st.warning(cal_error)

    dev = torch.device(config["device"])
    preprocessed = _preprocess_for_inference(img, DEFAULT_IMAGE_SIZE, dev)
    if preprocessed is None:
        st.error("Could not preprocess the uploaded image.")
        return
    input_tensor, rgb_image, width, height = preprocessed

    quality_result = None
    quality_blocked = False
    if config["enable_quality_gate"]:
        q_config = QualityConfig()
        quality_result = assess_quality(img, config=q_config)
        if config["quality_required"] and quality_result.status != "pass":
            quality_blocked = True

    _display_quality_result(quality_result)
    if quality_blocked:
        st.error(
            "Inference blocked: image did not pass the required quality gate. "
            "Review the image or adjust quality-gate settings."
        )
        return

    try:
        raw_probs, calibrated_probs, temperature = _predict_probs(
            model, input_tensor, temperature_scaling
        )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    probs_for_pred = calibrated_probs if calibrated_probs is not None else raw_probs
    predicted_class = int(probs_for_pred.argmax())
    referable = bool(predicted_class >= REFERRABLE_THRESHOLD_GRADE)
    referable_prob = float(probs_for_pred[REFERRABLE_THRESHOLD_GRADE:].sum())

    context = PredictionContext(
        raw_probs=raw_probs,
        calibrated_probs=calibrated_probs,
        temperature=temperature,
        metadata=metadata,
        predicted_class=predicted_class,
        referable=referable,
        referable_prob=referable_prob,
        quality_result=quality_result,
        quality_blocked=quality_blocked,
        effective_checkpoint=effective_checkpoint,
        image_width=width,
        image_height=height,
    )

    report = _render_prediction_results(context)

    if st.checkbox("Generate Grad-CAM visualization", value=False, key="gradcam_checkbox"):
        with tempfile.NamedTemporaryFile(
            prefix=TEMP_FILE_PREFIX, suffix=".png", delete=False
        ) as tmp:
            gradcam_path = Path(tmp.name)
        gradcam_ok, gradcam_saved = _run_gradcam(model, input_tensor, rgb_image, gradcam_path)
        if gradcam_ok and gradcam_saved:
            st.image(
                gradcam_saved,
                caption="Grad-CAM visualization (model sensitivity, not medical proof)",
                use_container_width=True,
            )
        else:
            st.warning("Grad-CAM visualization could not be generated for this model.")

    st.download_button(
        label="Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="lumoculis_prediction_report.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="Lumoculis - DR Screening Support",
        page_icon="👁️",
        layout="wide",
    )
    _inject_css()

    tabs = st.tabs(["Overview", "Analyze Image", "Model Information", "About"])
    with tabs[0]:
        _render_overview()
    with tabs[1]:
        config = _render_sidebar()
        st.session_state["device"] = config["device"]
        _run_analysis_app(config)
    with tabs[2]:
        _render_model_info()
    with tabs[3]:
        _render_about()


if __name__ == "__main__":
    main()
