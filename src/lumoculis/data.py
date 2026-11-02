"""Dataset and stratified train/val split for Lumoculis.

Reads data/raw/train.csv (id_code, diagnosis) and loads cached PNGs from
data/processed/{id_code}.png. Returns (image_tensor, label) torch tensors
with RGB channel order, float32, ImageNet normalization.

Augmentation (flips, small rotations, light brightness/contrast) is applied
only to the training set; validation receives no augmentation.

The train/val split is group-aware: exact duplicate images (same SHA-256 of
the processed PNG) are always kept in the same split to avoid leakage. APTOS
provides no patient IDs, so patient-level splitting cannot be guaranteed even
with duplicate grouping. See the "Limitations" section of README.md.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from lumoculis.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Module-level normalization tensors created once and reused in every
# __getitem__ call. Avoids allocating new CPU tensors per image.
IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD).view(3, 1, 1)

MISSING_PREVIEW_LIMIT = 10
DEFAULT_VAL_SIZE = 0.2
DatasetT = TypeVar("DatasetT", bound="LumoculisDataset")


class LumoculisDataset(Dataset):
    """Dataset of preprocessed fundus images and DR grade labels."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processed_dir: str | Path,
        image_size: int = 384,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.image_size = image_size
        self.transform = transform

        self._warn_missing_images()

    def _warn_missing_images(self) -> None:
        """Warn if any image listed in the dataframe is missing from processed_dir."""
        missing = [
            str(id_code)
            for id_code in self.df["id_code"]
            if not (self.processed_dir / f"{id_code}.png").exists()
        ]
        if missing:
            preview = ", ".join(missing[:MISSING_PREVIEW_LIMIT])
            more = (
                f" ... and {len(missing) - MISSING_PREVIEW_LIMIT} more"
                if len(missing) > MISSING_PREVIEW_LIMIT
                else ""
            )
            print(
                "WARNING: "
                f"{len(missing)} image(s) listed in CSV not found in "
                f"{self.processed_dir}: {preview}{more}"
            )

    @classmethod
    def from_csv(
        cls: type[DatasetT],
        csv_path: str | Path,
        processed_dir: str | Path,
        image_size: int = 384,
        transform: transforms.Compose | None = None,
    ) -> DatasetT:
        """Build a dataset by reading *csv_path* (id_code, diagnosis)."""
        df = pd.read_csv(csv_path)
        return cls(df, processed_dir, image_size, transform)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        label = int(row["diagnosis"])

        img_path = self.processed_dir / f"{id_code}.png"
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # OpenCV loads BGR; convert to RGB for the model.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Augmentation (train set only; val transform is None).
        if self.transform is not None:
            image = self.transform(image)

        # To float32 CHW tensor in [0, 1], then ImageNet normalization.
        # Reuse module-level tensors instead of allocating per call.
        image = transforms.ToTensor()(image)
        image = (image - IMAGENET_MEAN_TENSOR) / IMAGENET_STD_TENSOR

        return image, torch.tensor(label, dtype=torch.long)


def train_transforms(augment: str = "light") -> transforms.Compose:
    """Train-only augmentation.

    Parameters
    ----------
    augment : str
        - "none": no augmentation (only ToTensor + normalization in dataset).
        - "light": horizontal/vertical flips, small rotation, light jitter.
        - "moderate": light + larger rotation, stronger jitter, random crop.
    """
    if augment == "none":
        return transforms.Compose([])

    if augment == "light":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    if augment == "moderate":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.RandomResizedCrop(size=(384, 384), scale=(0.85, 1.0)),
        ])

    raise ValueError(f"Unknown augment level: {augment}")


def val_transforms() -> None:
    """Validation receives no augmentation."""
    return None


def find_duplicate_groups(processed_dir: str | Path, id_codes) -> dict[str, list[str]]:
    """Map SHA-256 hash of processed PNG -> list of id_codes sharing that hash.

    Includes singleton groups (one image per hash). Only groups with more than
    one id_code are exact duplicates.
    """
    processed_dir = Path(processed_dir)
    groups: dict[str, list[str]] = {}
    for id_code in id_codes:
        p = processed_dir / f"{id_code}.png"
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            groups.setdefault(h, []).append(str(id_code))
    return groups


def _representative_label(labels: list[int]) -> int:
    """Return the majority diagnosis label for a group.

    Returns -1 if the group is conflicting (no unique majority).
    """
    counts = Counter(labels)
    max_count = max(counts.values())
    modes = [label for label, _count in counts.items() if _count == max_count]
    if len(modes) > 1:
        return -1
    return modes[0]

def train_val_test_split_preserving_validation(
    csv_path: str | Path,
    processed_dir: str | Path,
    config: Config,
    test_size: float = 0.15,
    val_size: float = DEFAULT_VAL_SIZE,
) -> tuple[LumoculisDataset, LumoculisDataset, LumoculisDataset, dict]:
    """Create an independent test set without changing historical validation.

    First reproduce the existing train/validation split. Then select test
    images exclusively from the original training partition. Exact duplicate
    images remain together. APTOS has no patient IDs, so patient-level
    independence cannot be guaranteed.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    # Reproduce the established split used in previous experiments.
    original_train_ds, val_ds, dup_report = train_val_split(
        csv_path,
        processed_dir,
        config,
        val_size=val_size,
    )

    original_train_df = original_train_ds.df.copy()
    original_train_ids = set(
        original_train_df["id_code"].astype(str)
    )
    historical_val_ids = set(
        val_ds.df["id_code"].astype(str)
    )

    # Convert the desired overall test proportion into a proportion of
    # the original training partition.
    test_fraction_of_train = test_size / (1.0 - val_size)

    id_to_label = {
        str(row.id_code): int(row.diagnosis)
        for row in original_train_df.itertuples(index=False)
    }

    # Group exact duplicates within the original training partition only.
    hash_groups = find_duplicate_groups(
        processed_dir,
        list(original_train_ids),
    )

    groups_by_class = defaultdict(list)

    for _image_hash, ids in hash_groups.items():
        labels = [id_to_label[image_id] for image_id in ids]
        representative_label = _representative_label(labels)
        groups_by_class[representative_label].append(ids)

    rng = random.Random(config.seed + 1)  # noqa: S311
    test_ids = set()

    # Select test groups class by class while leaving training examples
    # in every class that has at least two distinct groups.
    for groups in groups_by_class.values():
        rng.shuffle(groups)
        count = len(groups)

        if count <= 1:
            continue

        n_test = max(1, round(count * test_fraction_of_train))
        n_test = min(n_test, count - 1)

        for group in groups[:n_test]:
            test_ids.update(group)

    train_ids = original_train_ids - test_ids

    if not train_ids or not test_ids:
        raise RuntimeError("The new training or test split is empty.")

    if train_ids & test_ids:
        raise RuntimeError("Training and test image IDs overlap.")

    if test_ids & historical_val_ids:
        raise RuntimeError("Test set overlaps historical validation.")

    # Verify that no exact duplicate group crosses the new boundary.
    for ids in hash_groups.values():
        group_ids = set(ids)
        if group_ids & train_ids and group_ids & test_ids:
            raise RuntimeError(
                f"Exact duplicate group crosses train/test: {ids}"
            )

    train_df = (
        original_train_df[
            original_train_df["id_code"].astype(str).isin(train_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        original_train_df[
            original_train_df["id_code"].astype(str).isin(test_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    train_ds = LumoculisDataset(
        train_df,
        processed_dir,
        config.image_size,
        transform=train_transforms(config.augment),
    )

    # val_ds is deliberately returned unchanged.
    test_ds = LumoculisDataset(
        test_df,
        processed_dir,
        config.image_size,
        transform=val_transforms(),
    )

    report = {
        **dup_report,
        "n_original_train_images": len(original_train_df),
        "n_historical_val_images": len(val_ds),
        "n_new_train_images": len(train_ds),
        "n_test_images": len(test_ds),
        "requested_test_fraction_of_dataset": test_size,
        "actual_test_fraction_of_dataset": len(test_ds)
        / (len(original_train_df) + len(val_ds)),
        "test_fraction_of_original_train": test_fraction_of_train,
    }

    return train_ds, val_ds, test_ds, report

def train_val_split(
    csv_path: str | Path,
    processed_dir: str | Path,
    config: Config,
    val_size: float = DEFAULT_VAL_SIZE,
) -> tuple[LumoculisDataset, LumoculisDataset, dict]:
    """Stratified, group-aware train/val split by diagnosis.

    Exact duplicate images (same processed PNG hash) are always kept in the
    same split to avoid leakage. The split is seeded by ``config.seed`` for
    reproducibility.

    Returns (train_ds, val_ds, dup_report) where dup_report contains:
      - n_dup_groups: number of groups with >1 image
      - n_dup_images: total images in duplicate groups
      - conflicting_groups: list of (hash, [id_codes]) with conflicting labels
      - cross_split_groups: list of (hash, [id_codes]) that cross train/val
                            (empty after a group-aware split)

    APTOS has no patient IDs, so patient-level splitting cannot be guaranteed
    even with duplicate grouping.
    """
    df = pd.read_csv(csv_path)
    id_to_label = {str(k): int(v) for k, v in zip(df["id_code"], df["diagnosis"], strict=True)}

    # Group all images by processed PNG hash.
    hash_groups = find_duplicate_groups(processed_dir, list(id_to_label.keys()))

    # Assign each id_code to a group index.
    groups_list: list[list[str]] = []
    id_to_group_idx: dict[str, int] = {}
    for ids in hash_groups.values():
        idx = len(groups_list)
        groups_list.append(ids)
        for gid in ids:
            id_to_group_idx[gid] = idx

    # Representative label and conflict flag per group.
    group_labels: list[int] = []
    group_conflicting: list[bool] = []
    for ids in groups_list:
        labels = [id_to_label[gid] for gid in ids]
        rep = _representative_label(labels)
        group_labels.append(rep)
        group_conflicting.append(rep == -1)

    # Shuffle groups within each representative class, then split proportionally.
    rng = random.Random(config.seed)  # noqa: S311 - seeded for reproducibility, not cryptography
    groups_by_class: dict[int, list[int]] = defaultdict(list)
    for gi, label in enumerate(group_labels):
        groups_by_class[label].append(gi)

    train_group_idxs: list[int] = []
    val_group_idxs: list[int] = []
    for _cls, gidxs in groups_by_class.items():
        rng.shuffle(gidxs)
        n_total = len(gidxs)
        if n_total > 1:
            n_val = max(1, round(n_total * val_size))
            n_val = min(n_val, n_total - 1)  # keep at least one in train
        else:
            n_val = 0
        val_group_idxs.extend(gidxs[:n_val])
        train_group_idxs.extend(gidxs[n_val:])

    # Expand group indices back to id_codes.
    train_ids: list[str] = []
    for gi in train_group_idxs:
        train_ids.extend(groups_list[gi])
    val_ids: list[str] = []
    for gi in val_group_idxs:
        val_ids.extend(groups_list[gi])

    train_df = df[df["id_code"].astype(str).isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["id_code"].astype(str).isin(val_ids)].copy().reset_index(drop=True)

    # Duplicate report.
    dup_groups = {h: ids for h, ids in hash_groups.items() if len(ids) > 1}
    cross_split = []
    for h, ids in dup_groups.items():
        in_train = any(gid in train_ids for gid in ids)
        in_val = any(gid in val_ids for gid in ids)
        if in_train and in_val:
            cross_split.append((h, ids))
    conflicting = [
        (h, ids) for h, ids in dup_groups.items() if group_conflicting[groups_list.index(ids)]
    ]
    dup_report = {
        "n_dup_groups": len(dup_groups),
        "n_dup_images": sum(len(ids) for ids in dup_groups.values()),
        "conflicting_groups": conflicting,
        "cross_split_groups": cross_split,
    }

    train_ds = LumoculisDataset(
        train_df,
        processed_dir,
        config.image_size,
        transform=train_transforms(config.augment),
    )
    val_ds = LumoculisDataset(val_df, processed_dir, config.image_size, transform=val_transforms())

    return train_ds, val_ds, dup_report



def train_val_test_split(  # noqa: PLR0912, PLR0915
    csv_path: str | Path,
    processed_dir: str | Path,
    config: Config,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[LumoculisDataset, LumoculisDataset, LumoculisDataset, dict]:
    """Create reproducible, stratified, duplicate-aware data splits.

    Exact duplicate groups stay in one split. APTOS does not provide patient
    IDs, so patient-level independence cannot be guaranteed.
    """
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")

    df = pd.read_csv(csv_path)
    id_to_label = {
        str(k): int(v)
        for k, v in zip(df["id_code"], df["diagnosis"], strict=True)
    }

    hash_groups = find_duplicate_groups(
        processed_dir, list(id_to_label.keys())
    )
    groups = list(hash_groups.values())
    rng = random.Random(config.seed)  # noqa: S311

    groups_by_class = defaultdict(list)
    for index, ids in enumerate(groups):
        labels = [id_to_label[image_id] for image_id in ids]
        groups_by_class[_representative_label(labels)].append(index)

    train_indices, val_indices, test_indices = [], [], []

    for indices in groups_by_class.values():
        rng.shuffle(indices)
        count = len(indices)

        if count == 1:
            train_indices.extend(indices)
            continue

        if count == 2:  # noqa: PLR2004
            train_indices.append(indices[0])
            test_indices.append(indices[1])
            continue

        n_val = max(1, round(count * val_size))
        n_test = max(1, round(count * test_size))

        while n_val + n_test >= count:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        val_indices.extend(indices[:n_val])
        test_indices.extend(indices[n_val:n_val + n_test])
        train_indices.extend(indices[n_val + n_test:])

    def expand(indices):
        return [
            image_id
            for index in indices
            for image_id in groups[index]
        ]

    train_ids = expand(train_indices)
    val_ids = expand(val_indices)
    test_ids = expand(test_indices)

    train_set, val_set, test_set = (
        set(train_ids), set(val_ids), set(test_ids)
    )

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Image IDs overlap between splits.")

    split_by_id = {
        **dict.fromkeys(train_ids, "train"),
        **dict.fromkeys(val_ids, "val"),
        **dict.fromkeys(test_ids, "test"),
    }

    cross_split_groups = [
        (image_hash, ids)
        for image_hash, ids in hash_groups.items()
        if len({split_by_id.get(image_id) for image_id in ids}) > 1
    ]
    if cross_split_groups:
        raise RuntimeError("Exact duplicate group crosses dataset splits.")

    conflicting_groups = [
        (image_hash, ids)
        for image_hash, ids in hash_groups.items()
        if len(ids) > 1
        and len({id_to_label[image_id] for image_id in ids}) > 1
    ]

    dup_report = {
        "n_dup_groups": sum(len(ids) > 1 for ids in hash_groups.values()),
        "n_dup_images": sum(
            len(ids) for ids in hash_groups.values() if len(ids) > 1
        ),
        "conflicting_groups": conflicting_groups,
        "cross_split_groups": cross_split_groups,
    }

    train_df = df[df["id_code"].astype(str).isin(train_set)].copy()
    val_df = df[df["id_code"].astype(str).isin(val_set)].copy()
    test_df = df[df["id_code"].astype(str).isin(test_set)].copy()

    train_ds = LumoculisDataset(
        train_df, processed_dir, config.image_size,
        transform=train_transforms(config.augment),
    )
    val_ds = LumoculisDataset(
        val_df, processed_dir, config.image_size,
        transform=val_transforms(),
    )
    test_ds = LumoculisDataset(
        test_df, processed_dir, config.image_size,
        transform=val_transforms(),
    )

    return train_ds, val_ds, test_ds, dup_report


def class_counts_and_weights(
    dataframe: pd.DataFrame,
    dampen: bool = False,
) -> tuple[pd.Series, dict[int, float]]:
    """Return class counts and inverse-frequency weights for imbalanced loss.

    Weights are total_samples / (num_classes * class_count), suitable as the
    ``weight`` argument of ``torch.nn.CrossEntropyLoss``. When ``dampen`` is
    True, weights use the square root of inverse frequency so no class is
    overweighted relative to the smallest by more than ~2x.
    """
    counts = dataframe["diagnosis"].value_counts().sort_index()
    total = len(dataframe)
    num_classes = len(counts)
    inverse = total / (num_classes * counts)
    if dampen:
        inverse = np.sqrt(inverse)
    weights = {int(cls): float(w) for cls, w in inverse.items()}
    return counts, weights


def create_weighted_sampler(
    dataframe: pd.DataFrame,
    num_classes: int,
    seed: int | None = None,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler using inverse class frequency."""
    counts, _ = class_counts_and_weights(dataframe, dampen=False)
    # inverse frequency weights per class
    class_weights = {}
    for cls in range(num_classes):
        c = counts.get(cls, 1)
        class_weights[cls] = 1.0 / c if c > 0 else 1.0
    sample_weights = []
    for _, row in dataframe.iterrows():
        cls = int(row["diagnosis"])
        sample_weights.append(class_weights.get(cls, 1.0))
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
