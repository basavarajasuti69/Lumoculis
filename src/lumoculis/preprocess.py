"""Image preprocessing for Lumoculis.

Pure, typed functions. Crop black borders, resize to a target size, and apply
Ben Graham contrast enhancement (Gaussian blur sigma ~10 with addWeighted).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GRAYSCALE_NDIM = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_black_borders(image: np.ndarray, tol: int = 10) -> np.ndarray:
    """Crop near-black borders from *image*.

    Returns the image unchanged if no non-dark region is found. *tol* is the
    per-channel intensity threshold below which a pixel counts as "black".
    """
    if image.ndim == GRAYSCALE_NDIM:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = gray > tol
    if not mask.any():
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image[rmin : rmax + 1, cmin : cmax + 1]


def resize(image: np.ndarray, size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize *image* to (size, size)."""
    return cv2.resize(image, (size, size), interpolation=interpolation)


def ben_graham(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Ben Graham contrast enhancement.

    ``4 * image - 4 * GaussianBlur(image, sigma) + 128`` emphasizes local
    contrast (vessels, lesions) while suppressing slow illumination changes.
    Works on uint8 BGR/grayscale images.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 4.0, blurred, -4.0, 128.0)


def preprocess(image: np.ndarray, size: int = 384, sigma: float = 10.0) -> np.ndarray:
    """Run the full pipeline: crop black borders, resize, Ben Graham.

    Returns a uint8 BGR image of shape (size, size, 3).
    """
    cropped = crop_black_borders(image)
    resized = resize(cropped, size)
    return ben_graham(resized, sigma=sigma)


def load_image(path: str | Path) -> np.ndarray | None:
    """Load an image with OpenCV, returning None if it cannot be read."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


__all__ = [
    "ben_graham",
    "crop_black_borders",
    "load_image",
    "preprocess",
    "resize",
]
