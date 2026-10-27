"""Model factory for Lumoculis.

Builds a timm image classifier (efficientnet_b0 by default) with a head that
outputs raw logits for DR grades 0-4. No softmax is applied inside the model
so the caller owns the loss / temperature-scaling logic.

Also provides an ordinal-regression-style head that treats DR grades as
ordered categories (0 < 1 < 2 < 3 < 4). The ordinal head outputs 4 logits,
one per cumulative threshold P(grade > k) for k = 0..3, using binary
cross-entropy on each threshold. This respects the ordering of DR grades
without assuming equal spacing between adjacent grades.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

NUM_SPATIAL_DIMS = 2
SPATIAL_DIMS_THRESHOLD = NUM_SPATIAL_DIMS


def build_model(
    name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """Build a timm classifier with a *num_classes* logit head.

    Returns a plain ``nn.Module`` whose forward pass outputs logits of shape
    ``(batch, num_classes)``. The pretrained backbone weights come from timm
    (downloaded on first use when ``pretrained=True``).
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_ordinal_model(
    name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """Build a timm classifier with an ordinal-regression head.

    The backbone is the same as :func:`build_model`, but the final head
    outputs ``num_classes - 1`` logits instead of ``num_classes``. Each logit
    corresponds to a cumulative threshold: logit[k] models P(grade > k) for
    k = 0, 1, ..., num_classes - 2.

    Returns a plain ``nn.Module`` whose forward pass outputs logits of shape
    ``(batch, num_classes - 1)``. Use :func:`ordinal_logits_to_probs` to
    convert these to per-class probabilities, and
    :func:`ordinal_cross_entropy_loss` for the training loss.
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,  # remove the default head; we add our own below
    )

    # Get the backbone's feature dimension.
    if hasattr(model, "num_features"):
        in_features = model.num_features
    elif hasattr(model, "head"):
        in_features = model.head.in_features
    else:
        in_features = model.classifier.in_features

    # Replace the classification head with our ordinal head.
    model.reset_classifier(0)  # remove existing head
    ordinal_head = nn.Linear(in_features, num_classes - 1)
    model.ordinal_head = ordinal_head

    def forward(x: torch.Tensor) -> torch.Tensor:
        features = model.forward_features(x)
        # Global average pooling if needed (timm models return pooled features
        # from forward_features when global_pool is set, but be safe).
        if features.dim() > SPATIAL_DIMS_THRESHOLD:  # (N, C, H, W) -> (N, C)
            features = features.mean(dim=(-2, -1))
        return model.ordinal_head(features)

    model.forward = forward  # type: ignore[method-assign]
    return model


def ordinal_logits_to_probs(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Convert ordinal logits (batch, num_classes - 1) to class probabilities (batch, num_classes).

    Uses the cumulative link: P(grade > k) = sigmoid(logit[k]).
    Then:
      P(grade = 0) = 1 - P(grade > 0)
      P(grade = k) = P(grade > k-1) - P(grade > k)  for 1 <= k <= num_classes - 2
      P(grade = num_classes - 1) = P(grade > num_classes - 2)
    """
    n_classes_minus_1 = logits.shape[-1]
    n_classes = n_classes_minus_1 + 1

    cumulative = torch.sigmoid(logits)  # P(grade > k) for k = 0..n-2
    probs = torch.empty(logits.shape[0], n_classes, device=logits.device)

    probs[:, 0] = 1.0 - cumulative[:, 0]
    for k in range(1, n_classes - 1):
        probs[:, k] = cumulative[:, k - 1] - cumulative[:, k]
    probs[:, n_classes - 1] = cumulative[:, n_classes - 2]

    # Clamp to [0, 1] and renormalize to handle float error.
    probs = probs.clamp(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs


def ordinal_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Binary cross-entropy loss for ordinal cumulative thresholds.

    For each sample with grade g, all thresholds k < g should be "grade > k"
    (positive), and all thresholds k >= g should be "not grade > k" (negative).
    """
    n_classes_minus_1 = num_classes - 1
    batch_size = targets.shape[0]

    # Build binary targets: for each threshold k, is grade > k?
    # Shape: (batch, n_classes_minus_1)
    threshold_targets = torch.zeros(
        batch_size, n_classes_minus_1, device=logits.device, dtype=logits.dtype
    )
    for k in range(n_classes_minus_1):
        threshold_targets[:, k] = (targets > k).to(logits.dtype)

    return nn.functional.binary_cross_entropy_with_logits(logits, threshold_targets)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
