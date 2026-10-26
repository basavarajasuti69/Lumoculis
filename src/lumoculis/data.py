"""Dataset and stratified train/val split for Lumoculis.

Reads data/raw/train.csv (id_code, diagnosis) and loads cached PNGs from
data/processed/{id_code}.png. Returns (image_tensor, label) torch tensors
with RGB channel order, float32, ImageNet normalization.

Augmentation (flips, small rotations, light brightness/contrast) is applied
only to the training set; validation receives no augmentation.

APTOS provides no patient IDs, so patient-level train/val splitting cannot be
guaranteed. See the "Limitations" section of README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from lumoculis.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
        image = transforms.ToTensor()(image)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std

        return image, torch.tensor(label, dtype=torch.long)


def train_transforms() -> transforms.Compose:
    """Train-only augmentation: flips, small rotations, light brightness/contrast."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])


def val_transforms() -> None:
    """Validation receives no augmentation."""
    return None


def train_val_split(
    csv_path: str | Path,
    processed_dir: str | Path,
    config: Config,
    val_size: float = DEFAULT_VAL_SIZE,
) -> tuple[LumoculisDataset, LumoculisDataset]:
    """Stratified train/val split by diagnosis.

    The split is seeded by ``config.seed`` for reproducibility. APTOS has no
    patient IDs, so patient-level splitting cannot be guaranteed.
    """
    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["diagnosis"],
        random_state=config.seed,
        shuffle=True,
    )

    train_ds = LumoculisDataset(
        train_df, processed_dir, config.image_size, transform=train_transforms()
    )
    val_ds = LumoculisDataset(val_df, processed_dir, config.image_size, transform=val_transforms())

    return train_ds, val_ds


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
