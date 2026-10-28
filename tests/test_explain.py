"""Tests for src.lumoculis.explain using synthetic data and pretrained=False."""

from __future__ import annotations

import numpy as np
import torch

from lumoculis.explain import (
    GradCAM,
    find_final_conv_layer,
    overlay_heatmap,
    save_gradcam_visualization,
)
from lumoculis.model import build_model


def _synthetic_image_tensor(size: int = 64) -> torch.Tensor:
    """Return a random single-image tensor of shape (1, 3, size, size)."""
    return torch.randn(1, 3, size, size)


def _synthetic_rgb_image(size: int = 64) -> np.ndarray:
    """Return a random uint8 RGB image of shape (size, size, 3)."""
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def test_find_final_conv_layer_returns_name() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    name = find_final_conv_layer(model)
    assert isinstance(name, str)
    assert len(name) > 0


def test_gradcam_output_shape_and_finite() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    model.eval()
    input_tensor = _synthetic_image_tensor(size=64)

    with GradCAM(model) as cam:
        heatmap, target = cam.compute_heatmap(input_tensor, target_class=2)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2  # noqa: PLR2004
    assert heatmap.shape == (64, 64)
    assert np.all(np.isfinite(heatmap))
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert target == 2  # noqa: PLR2004


def test_gradcam_default_target_is_predicted() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    model.eval()
    input_tensor = _synthetic_image_tensor(size=64)

    with GradCAM(model) as cam:
        heatmap, target = cam.compute_heatmap(input_tensor, target_class=None)

    logits = model(input_tensor)
    predicted = int(logits.argmax(dim=1).item())
    assert target == predicted
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (64, 64)


def test_gradcam_valid_target_classes() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    model.eval()
    input_tensor = _synthetic_image_tensor(size=64)

    for target_class in range(5):
        with GradCAM(model) as cam:
            heatmap, target = cam.compute_heatmap(input_tensor, target_class=target_class)
        assert target == target_class
        assert heatmap.shape == (64, 64)
        assert np.all(np.isfinite(heatmap))


def test_overlay_heatmap_output_shape_and_dtype() -> None:
    image = _synthetic_rgb_image(size=100)
    heatmap = np.random.rand(50, 50).astype(np.float32)
    overlay = overlay_heatmap(image, heatmap, alpha=0.6)

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == image.shape  # (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_save_gradcam_visualization_creates_file(tmp_path) -> None:
    image = _synthetic_rgb_image(size=64)
    heatmap = np.random.rand(64, 64).astype(np.float32)
    output_path = tmp_path / "gradcam.png"

    save_gradcam_visualization(
        original_image=image,
        heatmap=heatmap,
        output_path=output_path,
        predicted_class=2,
        target_class=2,
        class_names=["0", "1", "2", "3", "4"],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_gradcam_hooks_removed_after_context() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    model.eval()
    input_tensor = _synthetic_image_tensor(size=64)

    with GradCAM(model) as cam:
        cam.compute_heatmap(input_tensor, target_class=0)
        assert len(cam._handles) == 2  # noqa: PLR2004

    # Hooks should be removed after exiting the context.
    assert len(cam._handles) == 0
    assert cam._activations is None
    assert cam._gradients is None
