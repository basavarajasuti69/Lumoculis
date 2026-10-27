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
from torch.utils.data import Dataset
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
