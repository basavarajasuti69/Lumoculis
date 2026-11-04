"""Grad-CAM explainability for Lumoculis.

Generates class-specific heatmaps using gradients and activations from the
final convolutional layer of a classification model. Overlays heatmaps on
original retinal images.

IMPORTANT: Grad-CAM visualizes model sensitivity (which pixels influenced
the prediction). It is NOT proof that the highlighted region is medically
causal or correct, and it is not a diagnostic tool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def find_final_conv_layer(model: nn.Module) -> str:
    """Find the name of the final Conv2d layer in *model*.

    Iterates through all named modules and returns the last Conv2d found.
    For EfficientNet-B0, this is typically ``conv_head``.
    """
    conv_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError(
            "No Conv2d layer found in model; Grad-CAM requires a convolutional backbone."
        )
    return conv_layers[-1]


class GradCAM:
    """Grad-CAM visualizer for classification models.

    Captures activations and gradients from a target convolutional layer,
    computes class-specific importance weights, and produces a heatmap
    overlay on the input image.

    Supports classification models only (not ordinal-regression models).
    """

    def __init__(self, model: nn.Module, target_layer: str | None = None) -> None:
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

        if target_layer is None:
            target_layer = find_final_conv_layer(model)
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        layer = dict(self.model.named_modules())[self.target_layer]

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles.append(layer.register_forward_hook(forward_hook))
        self._handles.append(layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        """Remove all registered hooks and clear captured tensors."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._activations = None
        self._gradients = None

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._remove_hooks()

    def compute_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Compute Grad-CAM heatmap for *input_tensor* and *target_class*.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Single image tensor of shape (1, C, H, W), already on the model's
            device and normalized as expected by the model.
        target_class : int | None
            Target class index. If None, uses the model's predicted class.

        Returns
        -------
        heatmap : np.ndarray
            Normalized heatmap of shape (H, W) with values in [0, 1].
        predicted_class : int
            The class used as the target (predicted if target_class is None).
        """
        self._activations = None
        self._gradients = None

        input_tensor = input_tensor.to(self.device)

        # Forward pass.
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward pass: gradient of target class score w.r.t. activations.
        self.model.zero_grad(set_to_none=True)
        score = logits[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations or gradients.")

        # Activations: (1, C, H, W) -> (C, H, W)
        activations = self._activations[0]
        # Gradients: (1, C, H, W) -> (C, H, W)
        gradients = self._gradients[0]

        # Global average pool gradients over spatial dimensions -> (C,)
        weights = gradients.mean(dim=(1, 2))

        # Weighted sum of activations -> (H, W)
        heatmap = (weights[:, None, None] * activations).sum(dim=0)

        # ReLU: only positive contributions.
        heatmap = torch.relu(heatmap)

        # Normalize to [0, 1].
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.detach().cpu().numpy()

        # Resize heatmap back to the input image spatial dimensions so the
        # overlay aligns with the original image.
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        if heatmap_np.shape != (input_h, input_w):
            import cv2  # noqa: PLC0415

            heatmap_np = cv2.resize(heatmap_np, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            # Re-normalize after resize to keep values in [0, 1].
            max_val = heatmap_np.max()
            if max_val > 0:
                heatmap_np = heatmap_np / max_val

        return heatmap_np, target_class


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    colormap: int | None = None,
) -> np.ndarray:
    """Overlay *heatmap* on *image*.

    Parameters
    ----------
    image : np.ndarray
        Original image as uint8 HWC RGB, shape (H, W, 3).
    heatmap : np.ndarray
        Normalized heatmap of shape (H, W) with values in [0, 1].
    alpha : float
        Blending factor for the heatmap overlay (0 = image only,
        1 = heatmap only).
    colormap : int
        OpenCV colormap to apply to the heatmap.

    Returns
    -------
    overlay : np.ndarray
        uint8 HWC RGB image with heatmap overlaid.
    """
    if colormap is None:
        import cv2  # noqa: PLC0415

        colormap = cv2.COLORMAP_JET
    import cv2  # noqa: PLC0415

    heatmap_resized = cv2.resize(
        heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (1.0 - alpha) * image.astype(np.float32) + alpha * heatmap_color.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def save_gradcam_visualization(  # noqa: PLR0913, PLR0917
    original_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    predicted_class: int,
    target_class: int,
    class_names: list[str] | None = None,
) -> None:
    """Save a side-by-side Grad-CAM visualization.

    The figure shows the original image, the heatmap, and the overlay.
    A caption explicitly states that this is a model-sensitivity
    visualization, not a medical diagnosis.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = overlay_heatmap(original_image, heatmap)

    if class_names is None:
        class_names = [str(i) for i in range(5)]
    target_name = (
        class_names[target_class] if target_class < len(class_names) else str(target_class)
    )
    pred_name = (
        class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
    )

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM (target: {target_name})")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (predicted: {pred_name})")
    axes[2].axis("off")

    fig.suptitle(
        "Grad-CAM: model-sensitivity visualization. "
        "Highlighted regions are NOT medical proof of disease location.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
