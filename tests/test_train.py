"""Smoke test for src.lumoculis.train using pretrained=False and synthetic data.

No real data is used; everything is created in a temp directory.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from lumoculis.config import Config
from lumoculis.train import run_training


def _make_synthetic_dataset(tmp_path) -> Config:
    """Create synthetic train.csv + PNGs and return a Config pointing at them."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    id_codes = [f"img{i}" for i in range(25)]
    labels = [i % 5 for i in range(25)]  # 5 samples per class 0-4

    with open(raw_dir / "train.csv", "w") as handle:
        handle.write("id_code,diagnosis\n")
        for id_code, label in zip(id_codes, labels, strict=True):
            handle.write(f"{id_code},{label}\n")

    for id_code in id_codes:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(processed_dir / f"{id_code}.png"), img)

    return Config(
        image_size=64,
        batch_size=4,
        epochs=1,
        lr=0.001,
        model_name="efficientnet_b0",
        seed=42,
        data_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        mlflow_dir=str(tmp_path / "mlruns"),
        num_workers=0,
    )


def test_smoke_training_2_batches_1_epoch(tmp_path) -> None:
    config = _make_synthetic_dataset(tmp_path)
    result = run_training(
        config,
        max_batches=2,
        epochs_override=1,
        pretrained=False,
    )

    assert os.path.exists(result["best_checkpoint_path"])
    assert result["best_qwk"] != float("-inf")
