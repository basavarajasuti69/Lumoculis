"""Tests for src.lumoculis.data using synthetic data in temp directories (no real data)."""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pandas as pd
import torch

from lumoculis.config import Config
from lumoculis.data import (
    LumoculisDataset,
    class_counts_and_weights,
    train_val_split,
)

EXPECTED_TRAIN = 30
EXPECTED_VAL = 10
CLASS_ZERO_COUNT = 3


def _make_synthetic_images(tmp_path, id_codes) -> None:
    """Create synthetic PNG images in data/processed/ under tmp_path."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for id_code in id_codes:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(processed_dir / f"{id_code}.png"), img)
    return processed_dir


def _make_csv(tmp_path, id_codes, labels):
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_code", "diagnosis"])
        for id_code, label in zip(id_codes, labels, strict=True):
            writer.writerow([id_code, label])
    return csv_path


def _config(seed: int = 42, image_size: int = 64) -> Config:
    return Config(
        image_size=image_size,
        batch_size=16,
        epochs=1,
        lr=0.001,
        model_name="efficientnet_b0",
        seed=seed,
    )


def test_dataset_output_shape_and_dtype(tmp_path):
    id_codes = ["img0", "img1", "img2"]
    labels = [0, 1, 2]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    ds = LumoculisDataset.from_csv(csv_path, processed_dir, image_size=64)
    image, label = ds[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert label.dtype == torch.long


def test_dataset_rgb_order(tmp_path):
    """Verify BGR->RGB conversion: channel 0 should be the originally-reddest channel."""
    id_codes = ["img0"]
    labels = [0]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    # Known BGR image: blue=100, green=150, red=200
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 100  # B
    img[:, :, 1] = 150  # G
    img[:, :, 2] = 200  # R
    cv2.imwrite(str(processed_dir / "img0.png"), img)

    ds = LumoculisDataset.from_csv(csv_path, processed_dir, image_size=10)
    image, _ = ds[0]

    # After BGR->RGB, ToTensor, and ImageNet normalization:
    # Original RGB: R=200, G=150, B=100 -> after /255: R~0.784, G~0.588, B~0.392
    # After normalize: channel 0 (R) > channel 1 (G) > channel 2 (B)
    c0 = image[0, 0, 0].item()
    c1 = image[1, 0, 0].item()
    c2 = image[2, 0, 0].item()
    assert c0 > c1 > c2


def test_train_val_split_stratified(tmp_path):
    id_codes = [f"img{i}" for i in range(40)]
    labels = [i % 4 for i in range(40)]  # 10 per class 0-3
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    train_ds, val_ds = train_val_split(csv_path, processed_dir, _config(), val_size=0.25)

    assert len(train_ds) == EXPECTED_TRAIN
    assert len(val_ds) == EXPECTED_VAL

    train_labels = [int(train_ds.df.iloc[i]["diagnosis"]) for i in range(len(train_ds))]
    val_labels = [int(val_ds.df.iloc[i]["diagnosis"]) for i in range(len(val_ds))]
    for cls in range(4):
        assert cls in train_labels, f"Class {cls} missing from train"
        assert cls in val_labels, f"Class {cls} missing from val"


def test_train_val_split_reproducible(tmp_path):
    id_codes = [f"img{i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    train_ds_1, val_ds_1 = train_val_split(csv_path, processed_dir, _config(seed=42), val_size=0.25)
    train_ds_2, val_ds_2 = train_val_split(csv_path, processed_dir, _config(seed=42), val_size=0.25)

    train_ids_1 = [str(train_ds_1.df.iloc[i]["id_code"]) for i in range(len(train_ds_1))]
    train_ids_2 = [str(train_ds_2.df.iloc[i]["id_code"]) for i in range(len(train_ds_2))]
    assert train_ids_1 == train_ids_2

    val_ids_1 = [str(val_ds_1.df.iloc[i]["id_code"]) for i in range(len(val_ds_1))]
    val_ids_2 = [str(val_ds_2.df.iloc[i]["id_code"]) for i in range(len(val_ds_2))]
    assert val_ids_1 == val_ids_2


def test_class_counts_and_weights():
    df = pd.DataFrame({
        "id_code": ["a", "b", "c", "d", "e"],
        "diagnosis": [0, 0, 0, 1, 2],  # class 0: 3, class 1: 1, class 2: 1
    })
    counts, weights = class_counts_and_weights(df)

    assert counts.loc[0] == CLASS_ZERO_COUNT
    assert counts.loc[1] == 1
    assert counts.loc[2] == 1

    # Most common class gets the lowest weight
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]
    assert weights[1] == weights[2]


def test_class_counts_and_weights_dampen():
    df = pd.DataFrame({
        "id_code": ["a", "b", "c", "d", "e"],
        "diagnosis": [0, 0, 0, 1, 2],  # class 0: 3, class 1: 1, class 2: 1
    })
    _, weights = class_counts_and_weights(df)
    _, damp_weights = class_counts_and_weights(df, dampen=True)

    # Dampening reduces the spread between most and least common class weights
    spread_plain = max(weights.values()) / min(weights.values())
    spread_damp = max(damp_weights.values()) / min(damp_weights.values())
    assert spread_damp < spread_plain


def test_missing_image_warns(tmp_path, capsys):
    id_codes = ["img0", "img1"]
    labels = [0, 1]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, ["img0"])  # img1 missing

    _ = LumoculisDataset.from_csv(csv_path, processed_dir, image_size=64)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "img1" in captured.out


def test_val_has_no_augmentation_transform(tmp_path):
    id_codes = ["img0", "img1", "img2", "img3"]
    labels = [0, 0, 1, 1]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    _, val_ds = train_val_split(csv_path, processed_dir, _config(), val_size=0.5)
    assert val_ds.transform is None
