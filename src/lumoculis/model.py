"""Model factory for Lumoculis.

Builds a timm image classifier (efficientnet_b0 by default) with a head that
outputs raw logits for DR grades 0-4. No softmax is applied inside the model
so the caller owns the loss / temperature-scaling logic.
"""

from __future__ import annotations

import timm
from torch import nn


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


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
