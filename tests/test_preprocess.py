"""Tests for src.lumoculis.preprocess using small synthetic images (no real data)."""

from __future__ import annotations

import numpy as np

from lumoculis.preprocess import ben_graham, crop_black_borders, preprocess, resize


def _synthetic_fundus() -> np.ndarray:
    """A small BGR image with a bright disc on a black background."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [200, 150, 100]
    return img


def test_preprocess_output_shape_and_dtype() -> None:
    img = _synthetic_fundus()
    out = preprocess(img, size=64, sigma=10.0)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_preprocess_default_size() -> None:
    img = _synthetic_fundus()
    out = preprocess(img)
    assert out.shape == (384, 384, 3)
    assert out.dtype == np.uint8


def test_crop_black_borders_returns_original_when_all_dark() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out = crop_black_borders(img)
    assert np.array_equal(out, img)


def test_resize_output_shape_and_dtype() -> None:
    img = _synthetic_fundus()
    out = resize(img, 32)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_ben_graham_output_shape_and_dtype() -> None:
    img = _synthetic_fundus()
    out = ben_graham(img, sigma=10.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
