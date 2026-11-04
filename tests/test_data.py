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
    _representative_label,
    class_counts_and_weights,
    find_duplicate_groups,
    train_val_split,
    train_val_test_split,
    train_val_test_split_preserving_validation,
)

EXPECTED_TRAIN = 32
EXPECTED_VAL = 8
CLASS_ZERO_COUNT = 3
NUM_CLASSES = 5


def _make_synthetic_images(tmp_path, id_codes, duplicate_of=None) -> None:
    """Create synthetic PNG images in data/processed/ under tmp_path.

    If ``duplicate_of`` is a dict {id_code: source_id_code}, the image for
    id_code is an exact copy of the source image (same bytes, same hash).
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    images = {}
    for id_code in id_codes:
        if duplicate_of and id_code in duplicate_of:
            img = images[duplicate_of[id_code]]
        else:
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            images[id_code] = img
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

    train_ds, val_ds, _ = train_val_split(csv_path, processed_dir, _config(), val_size=0.25)

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

    train_ds_1, val_ds_1, _ = train_val_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.25
    )
    train_ds_2, val_ds_2, _ = train_val_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.25
    )

    train_ids_1 = [str(train_ds_1.df.iloc[i]["id_code"]) for i in range(len(train_ds_1))]
    train_ids_2 = [str(train_ds_2.df.iloc[i]["id_code"]) for i in range(len(train_ds_2))]
    assert train_ids_1 == train_ids_2

    val_ids_1 = [str(val_ds_1.df.iloc[i]["id_code"]) for i in range(len(val_ds_1))]
    val_ids_2 = [str(val_ds_2.df.iloc[i]["id_code"]) for i in range(len(val_ds_2))]
    assert val_ids_1 == val_ids_2


def test_duplicates_never_cross_split(tmp_path):
    """Exact duplicate images must never appear in both train and val."""
    id_codes = [f"img{i}" for i in range(20)]
    # Make img0..img3 exact duplicates of each other, img4..img7 duplicates of each other
    duplicate_of = {}
    for i in range(1, 4):
        duplicate_of[f"img{i}"] = "img0"
    for i in range(5, 8):
        duplicate_of[f"img{i}"] = "img4"
    labels = [i % 4 for i in range(20)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)

    train_ds, val_ds, dup_report = train_val_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.25
    )

    train_ids = set(train_ds.df["id_code"].astype(str))
    val_ids = set(val_ds.df["id_code"].astype(str))

    # No id_code appears in both splits
    overlap = train_ids & val_ids
    assert not overlap, f"id_codes in both train and val: {overlap}"

    # No duplicate group crosses the boundary
    for h, ids in dup_report["cross_split_groups"]:
        raise AssertionError(f"duplicate group crosses split: {h} -> {ids}")

    # Every duplicate group is entirely in train or entirely in val
    hash_groups = find_duplicate_groups(processed_dir, id_codes)
    for h, ids in hash_groups.items():
        if len(ids) > 1:
            in_train = all(gid in train_ids for gid in ids)
            in_val = all(gid in val_ids for gid in ids)
            assert in_train or in_val, f"duplicate group {h} split across train/val: {ids}"


def test_conflicting_labels_detected(tmp_path):
    """Duplicate groups with conflicting diagnosis labels are reported."""
    id_codes = [f"img{i}" for i in range(12)]
    # img0 and img1 are exact duplicates but have different labels (0 vs 1)
    duplicate_of = {"img1": "img0"}
    labels = [0, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 3]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)

    _train_ds, _val_ds, dup_report = train_val_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.25
    )

    # The conflicting group (img0, img1) should be in conflicting_groups
    conflicting = dup_report["conflicting_groups"]
    assert len(conflicting) >= 1, f"expected at least 1 conflicting group, got {conflicting}"
    found = False
    for _h, ids in conflicting:
        if set(ids) == {"img0", "img1"}:
            found = True
    assert found, f"conflicting group {{img0, img1}} not in {conflicting}"


def test_duplicate_group_reproducible(tmp_path):
    """Duplicate grouping is reproducible for a fixed seed."""
    id_codes = [f"img{i}" for i in range(16)]
    duplicate_of = {f"img{i}": "img0" for i in range(1, 5)}  # img0..img4 are duplicates
    labels = [i % 4 for i in range(16)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)

    _, _, report1 = train_val_split(csv_path, processed_dir, _config(seed=42), val_size=0.25)
    _, _, report2 = train_val_split(csv_path, processed_dir, _config(seed=42), val_size=0.25)

    assert report1["n_dup_groups"] == report2["n_dup_groups"]
    assert report1["n_dup_images"] == report2["n_dup_images"]

    train_ids_1 = set(
        train_val_split(csv_path, processed_dir, _config(seed=42), val_size=0.25)[0].df["id_code"]
    )
    train_ids_2 = set(
        train_val_split(csv_path, processed_dir, _config(seed=42), val_size=0.25)[0].df["id_code"]
    )
    assert train_ids_1 == train_ids_2


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

    _, val_ds, _ = train_val_split(csv_path, processed_dir, _config(), val_size=0.5)
    assert val_ds.transform is None


def test_representative_label_conflicting():
    """_representative_label returns -1 for conflicting groups."""
    assert _representative_label([0, 0, 0]) == 0
    assert _representative_label([1, 1, 2]) == 1
    assert _representative_label([0, 1]) == -1  # tie -> conflicting
    assert _representative_label([0, 1, 2]) == -1  # all different -> conflicting



def test_train_val_test_split_reproducible(tmp_path):
    """The same seed must produce the same three splits."""
    id_codes = [f"img{i}" for i in range(50)]
    labels = [i % 5 for i in range(50)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    first = train_val_test_split(
        csv_path, processed_dir, _config(seed=42)
    )
    second = train_val_test_split(
        csv_path, processed_dir, _config(seed=42)
    )

    for first_ds, second_ds in zip(first[:3], second[:3], strict=True):
        first_ids = list(first_ds.df["id_code"].astype(str))
        second_ids = list(second_ds.df["id_code"].astype(str))
        assert first_ids == second_ids


def test_train_val_test_split_has_disjoint_ids(tmp_path):
    """No image ID may appear in more than one split."""
    id_codes = [f"img{i}" for i in range(50)]
    labels = [i % 5 for i in range(50)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    train_ds, val_ds, test_ds, report = train_val_test_split(
        csv_path, processed_dir, _config(seed=42)
    )

    train_ids = set(train_ds.df["id_code"].astype(str))
    val_ids = set(val_ds.df["id_code"].astype(str))
    test_ids = set(test_ds.df["id_code"].astype(str))

    assert train_ids
    assert val_ids
    assert test_ids
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    assert len(train_ids | val_ids | test_ids) == len(id_codes)
    assert report["cross_split_groups"] == []


def test_train_val_test_split_keeps_exact_duplicates_together(tmp_path):
    """Exact duplicate images must stay within a single split."""
    id_codes = [f"img{i}" for i in range(50)]
    labels = [i % 5 for i in range(50)]

    # img0 and img5 both belong to class 0 and have identical image content.
    duplicate_of = {"img5": "img0"}
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(
        tmp_path, id_codes, duplicate_of=duplicate_of
    )

    train_ds, val_ds, test_ds, report = train_val_test_split(
        csv_path, processed_dir, _config(seed=42)
    )

    split_by_id = {}
    for split_name, dataset in (
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ):
        for image_id in dataset.df["id_code"].astype(str):
            split_by_id[image_id] = split_name

    assert split_by_id["img0"] == split_by_id["img5"]
    assert report["cross_split_groups"] == []


def test_preserving_validation_keeps_historical_val(tmp_path):
    id_codes = [f"img{i}" for i in range(50)]
    labels = [i % 5 for i in range(50)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    hist_train_ds, hist_val_ds, _ = train_val_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.2
    )
    new_train_ds, new_val_ds, new_test_ds, _report = (
        train_val_test_split_preserving_validation(
            csv_path, processed_dir, _config(seed=42), val_size=0.2, test_size=0.15
        )
    )

    hist_val_ids = set(hist_val_ds.df["id_code"].astype(str))
    new_val_ids = set(new_val_ds.df["id_code"].astype(str))
    new_train_ids = set(new_train_ds.df["id_code"].astype(str))
    new_test_ids = set(new_test_ds.df["id_code"].astype(str))
    hist_train_ids = set(hist_train_ds.df["id_code"].astype(str))

    assert new_val_ids == hist_val_ids
    assert new_test_ids <= hist_train_ids
    assert not (new_train_ids & new_val_ids)
    assert not (new_train_ids & new_test_ids)
    assert not (new_val_ids & new_test_ids)
    assert new_train_ids | new_val_ids | new_test_ids == set(id_codes)


def test_three_way_split_all_ids_assigned(tmp_path):
    id_codes = [f"img{i}" for i in range(60)]
    labels = [i % 5 for i in range(60)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    train_ds, val_ds, test_ds, report = train_val_test_split(
        csv_path, processed_dir, _config(seed=7), val_size=0.2, test_size=0.2
    )
    train_ids = set(train_ds.df["id_code"].astype(str))
    val_ids = set(val_ds.df["id_code"].astype(str))
    test_ids = set(test_ds.df["id_code"].astype(str))

    assert train_ids | val_ids | test_ids == set(id_codes)
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert report["cross_split_groups"] == []


def test_three_way_rare_class_allocation(tmp_path):
    # Class 0 has 3 groups, class 1 has 2 groups, class 2 has 1 group.
    id_codes = []
    labels = []
    # three singleton-ish groups for class 0 by making duplicates
    duplicate_of = {}
    base = 0
    for cls in range(3):
        for g in range([3, 2, 1][cls]):
            idx = len(id_codes)
            code = f"img{idx}"
            id_codes.append(code)
            labels.append(cls)
            if g > 0:
                duplicate_of[code] = id_codes[base]
        base = len(id_codes)
    # pad with enough unique images for other classes
    for i in range(20):
        id_codes.append(f"img{len(id_codes)}")
        labels.append((i % 2) + 3)

    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)

    train_ds, val_ds, test_ds, _report = train_val_test_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.2, test_size=0.2
    )
    train_ids = set(train_ds.df["id_code"].astype(str))
    val_ids = set(val_ds.df["id_code"].astype(str))
    test_ids = set(test_ds.df["id_code"].astype(str))
    assert train_ids | val_ids | test_ids == set(id_codes)
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_three_way_conflicting_labels_reported(tmp_path):
    id_codes = [f"img{i}" for i in range(20)]
    duplicate_of = {"img1": "img0"}
    labels = [0, 1] + [i % 5 for i in range(2, 20)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)

    _train_ds, _val_ds, _test_ds, report = train_val_test_split(
        csv_path, processed_dir, _config(seed=42), val_size=0.2, test_size=0.2
    )
    assert any(set(ids) == {"img0", "img1"} for _, ids in report["conflicting_groups"])


def test_three_way_different_seed_changes_split(tmp_path):
    id_codes = [f"img{i}" for i in range(60)]
    labels = [i % 5 for i in range(60)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)

    a = train_val_test_split(csv_path, processed_dir, _config(seed=42), val_size=0.2, test_size=0.2)
    b = train_val_test_split(csv_path, processed_dir, _config(seed=7), val_size=0.2, test_size=0.2)
    a_ids = [list(ds.df["id_code"].astype(str)) for ds in a[:3]]
    b_ids = [list(ds.df["id_code"].astype(str)) for ds in b[:3]]
    assert a_ids != b_ids
