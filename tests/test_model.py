"""Tests for src.lumoculis.model using pretrained=False (offline, fast)."""

from __future__ import annotations

import torch

from lumoculis.model import (
    build_model,
    build_ordinal_model,
    count_parameters,
    ordinal_cross_entropy_loss,
    ordinal_logits_to_probs,
)


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


def test_ordinal_model_output_shape() -> None:
    model = build_ordinal_model(name="efficientnet_b0", num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4)  # num_classes - 1 = 4 thresholds


def test_ordinal_logits_to_probs() -> None:
    logits = torch.randn(4, 4)
    probs = ordinal_logits_to_probs(logits)
    assert probs.shape == (4, 5)
    # Probabilities sum to 1 per sample
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    # All probabilities in [0, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


def test_ordinal_cross_entropy_loss() -> None:
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = ordinal_cross_entropy_loss(logits, targets, num_classes=5)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad is not None
