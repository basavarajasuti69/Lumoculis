"""Training loop for Lumoculis.

Device selection: mps if available, else cpu. ``PYTORCH_ENABLE_MPS_FALLBACK=1``
is set before torch is imported so MPS ops that lack a Metal implementation
fall back to CPU instead of crashing. No float64, no autocast/fp16 for now.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lumoculis.config import Config
from lumoculis.data import LumoculisDataset, class_counts_and_weights, train_val_split
from lumoculis.metrics import (
    compute_confusion_matrix,
    quadratic_weighted_kappa,
    referable_auroc,
    referable_sensitivity_specificity,
    render_confusion_matrix,
)
from lumoculis.model import build_model, count_parameters


def pick_device() -> torch.device:
    """Return an MPS device if available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dampened_class_weight_tensor(train_df, num_classes: int, device: torch.device) -> torch.Tensor:
    """Build a float32 class-weight vector (sqrt inverse frequency) for CrossEntropyLoss."""
    _counts, weights = class_counts_and_weights(train_df, dampen=True)
    weight_list = [float(weights.get(i, 1.0)) for i in range(num_classes)]
    return torch.tensor(weight_list, dtype=torch.float32, device=device)


def train_epoch(  # noqa: PLR0913
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Run one training epoch; return mean loss over processed batches."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)  # noqa: PLW2901
        labels = labels.to(device)  # noqa: PLW2901

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1
    return running_loss / max(n_batches, 1)


def validate(  # noqa: PLR0913
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    val_ids: np.ndarray | None = None,
    max_batches: int | None = None,
) -> tuple[
    float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
]:
    """Validate.

    Returns (val_loss, qwk, auroc, sensitivity, specificity, cm, y_true,
    y_prob, val_ids).
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)  # noqa: PLW2901
            labels = labels.to(device)  # noqa: PLW2901

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss += float(loss.item())
            n_batches += 1
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    val_loss = running_loss / max(n_batches, 1)
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    auroc = referable_auroc(y_true, y_prob)
    sensitivity, specificity = referable_sensitivity_specificity(y_true, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    # Slice val_ids to match the number of samples actually processed.
    if val_ids is not None:
        val_ids = val_ids[: len(y_true)]

    return val_loss, qwk, auroc, sensitivity, specificity, cm, y_true, y_prob, val_ids


def _default_run_name() -> str:
    """Default run name: UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017


def _build_dataloaders(
    train_ds: LumoculisDataset,
    val_ds: LumoculisDataset,
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from the datasets and config."""
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    return train_loader, val_loader


def _setup_training(
    config: Config,
    device: torch.device,
    train_ds: LumoculisDataset,
    pretrained: bool,
    epochs: int,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build model, loss, optimizer, and scheduler from config."""
    model = build_model(config.model_name, config.num_classes, pretrained=pretrained).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    class_weights = _dampened_class_weight_tensor(train_ds.df, config.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return model, criterion, optimizer, scheduler


def _log_mlflow_params(config: Config, device: torch.device) -> None:
    """Log config parameters and device to MLflow."""
    params = {k: str(v) for k, v in dataclasses.asdict(config).items() if k != "extra"}
    params.update({f"extra.{k}": str(v) for k, v in config.extra.items()})
    params["device"] = str(device)
    mlflow.log_params(params)


def _save_best_epoch(  # noqa: PLR0913
    no_save: bool,
    run_dir: Path,
    model: nn.Module,
    config: Config,
    epoch: int,
    qwk: float,
    cm_fig,
    val_ids: np.ndarray | None,
    y_true: np.ndarray | None,
    y_prob: np.ndarray | None,
) -> None:
    """Save the best checkpoint and validation predictions to disk."""
    if no_save:
        return
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dataclasses.asdict(config),
            "epoch": epoch,
            "val_qwk": qwk,
        },
        run_dir / "best.pt",
    )
    np.savez(
        run_dir / "val_preds.npz",
        val_ids=val_ids,
        y_true=y_true,
        y_prob=y_prob,
    )
    mlflow.log_figure(cm_fig, "confusion_matrix.png")


def _log_mlflow_artifacts(
    no_save: bool,
    run_dir: Path,
    best_checkpoint_path: Path,
) -> None:
    """Log best checkpoint and predictions as MLflow artifacts."""
    if not no_save and best_checkpoint_path.exists():
        mlflow.log_artifact(str(best_checkpoint_path))
        mlflow.log_param("best_checkpoint_path", str(best_checkpoint_path))
        mlflow.log_artifact(str(run_dir / "val_preds.npz"))


def run_training(  # noqa: PLR0913
    config: Config,
    run_name: str | None = None,
    max_batches: int | None = None,
    epochs_override: int | None = None,
    pretrained: bool = True,
    no_save: bool = False,
) -> dict:
    """Run the full training loop.

    Returns a dict with the best validation QWK, the run directory, and the
    best checkpoint path (None when no_save).
    """
    device = pick_device()
    seed_everything(config.seed)
    mlflow.end_run()

    epochs = epochs_override if epochs_override is not None else config.epochs

    csv_path = Path(config.data_dir) / "train.csv"
    train_ds, val_ds = train_val_split(csv_path, config.processed_dir, config)
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, config)

    model, criterion, optimizer, scheduler = _setup_training(
        config, device, train_ds, pretrained, epochs
    )

    # Per-run checkpoint directory: checkpoints/<run_name>/
    run_name = run_name or _default_run_name()
    run_dir = Path(config.checkpoint_dir) / run_name
    if not no_save:
        run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "best.pt"

    mlflow.set_tracking_uri(str(config.mlflow_dir))
    mlflow.set_experiment(config.mlflow_experiment)

    best_qwk = float("-inf")
    best_cm_fig = None
    best_val_ids = None
    best_y_true = None
    best_y_prob = None

    with mlflow.start_run(run_name=run_name):
        _log_mlflow_params(config, device)

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_batches)
            (
                val_loss,
                qwk,
                auroc,
                sensitivity,
                specificity,
                cm,
                y_true,
                y_prob,
                val_ids,
            ) = validate(
                model,
                val_loader,
                criterion,
                device,
                config.num_classes,
                val_ids=val_ds.df["id_code"].values,
                max_batches=max_batches,
            )
            scheduler.step()
            elapsed = time.time() - t0

            print(
                f"epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"qwk={qwk:.4f} auroc={auroc:.4f} "
                f"sens={sensitivity:.4f} spec={specificity:.4f} "
                f"({elapsed:.1f}s)"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "qwk": qwk,
                    "referable_auroc": auroc,
                    "referable_sensitivity": sensitivity,
                    "referable_specificity": specificity,
                    "epoch_time_s": round(elapsed, 3),
                },
                step=epoch,
            )

            if qwk > best_qwk:
                best_qwk = qwk
                best_cm_fig = render_confusion_matrix(cm)
                best_val_ids = val_ids
                best_y_true = y_true
                best_y_prob = y_prob
                _save_best_epoch(
                    no_save,
                    run_dir,
                    model,
                    config,
                    epoch,
                    qwk,
                    best_cm_fig,
                    best_val_ids,
                    best_y_true,
                    best_y_prob,
                )

        _log_mlflow_artifacts(no_save, run_dir, best_checkpoint_path)

    return {
        "best_qwk": best_qwk,
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(best_checkpoint_path) if not no_save else None,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Lumoculis DR model.")
    parser.add_argument(
        "--config", default="configs/baseline.yaml", help="Path to experiment YAML config."
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Limit batches per epoch (smoke test)."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs from config."
    )
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Skip writing checkpoints and MLflow artifacts (still logs metrics).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained backbone weights (default True).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Randomly initialize the backbone (offline, fast).",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    run_training(
        config,
        run_name=args.run_name,
        max_batches=args.max_batches,
        epochs_override=args.epochs,
        pretrained=args.pretrained,
        no_save=args.no_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
