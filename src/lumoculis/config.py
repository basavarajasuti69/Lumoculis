"""Configuration loading for Lumoculis.

Reads experiment YAML files into a frozen, typed :class:`Config` object with
attribute access. Kept small and explicit so every line can be explained in
an interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Experiment hyperparameters and paths.

    ``frozen=True`` makes the config immutable after loading, so a training
    run cannot accidentally change a hyperparameter mid-way. Directory paths
    are relative to the project root.
    """

    image_size: int
    batch_size: int
    epochs: int
    lr: float
    model_name: str
    seed: int
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    mlflow_dir: str = "mlruns"
    mlflow_experiment: str = "lumoculis"
    num_classes: int = 5
    num_workers: int = 0
    pin_memory: bool = False
    weight_decay: float = 0.05
    label_smoothing: float = 0.0
    augment: str = "light"
    threshold_referable: float = 0.5
    use_weighted_sampler: bool = False
    class_weight_strength: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file and build a :class:`Config`.

        Unknown YAML keys are stored in ``extra`` so adding a new key to an
        experiment file does not break older config readers.
        """
        with open(path, encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        known = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}
        known_values = {k: v for k, v in raw.items() if k in known}
        extra_values = {k: v for k, v in raw.items() if k not in known and k != "extra"}

        return cls(**known_values, extra=extra_values)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style ``.get`` for optional lookups."""
        return getattr(self, key, default)
