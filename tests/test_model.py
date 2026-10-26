"""Tests for src.lumoculis.model using pretrained=False (offline, fast)."""

from __future__ import annotations

import torch

from lumoculis.model import build_model, count_parameters


def test_output_shape_and_num_classes() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 384, 384))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 5)


def test_num_classes_respected() -> None:
    for num_classes in (3, 5, 10):
        model = build_model(name="efficientnet_b0", num_classes=num_classes, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 224, 224))
        assert logits.shape == (1, num_classes)


def test_count_parameters_positive() -> None:
    model = build_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    n = count_parameters(model)
    assert isinstance(n, int)
    assert n > 0
