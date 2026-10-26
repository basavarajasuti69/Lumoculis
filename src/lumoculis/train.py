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
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lumoculis.config import Config
from lumoculis.data import class_counts_and_weights, train_val_split
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
    max_batches: int | None = None,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """Validate; return (val_loss, qwk, auroc, sensitivity, specificity, confusion_matrix)."""
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
    return val_loss, qwk, auroc, sensitivity, specificity, cm


def _load_existing_best_qwk(checkpoint_path: Path) -> float:
    """Return the val_qwk stored in an existing best checkpoint, or -inf."""
    if not checkpoint_path.exists():
        return float("-inf")
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
        return float(state.get("val_qwk", float("-inf")))
    except Exception:
        return float("-inf")


def run_training(
    config: Config,
    run_name: str | None = None,
    max_batches: int | None = None,
    epochs_override: int | None = None,
    pretrained: bool = True,
) -> dict:
    """Run the full training loop.

    Returns a dict with the best validation QWK and the best checkpoint path.
    """
    device = pick_device()
    seed_everything(config.seed)

    epochs = epochs_override if epochs_override is not None else config.epochs

    csv_path = Path(config.data_dir) / "train.csv"
    train_ds, val_ds = train_val_split(csv_path, config.processed_dir, config)

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

    model = build_model(config.model_name, config.num_classes, pretrained=pretrained).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    class_weights = _dampened_class_weight_tensor(train_ds.df, config.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"

    mlflow.set_tracking_uri(str(config.mlflow_dir))
    mlflow.set_experiment(config.mlflow_experiment)

    params = {k: str(v) for k, v in dataclasses.asdict(config).items() if k != "extra"}
    params.update({f"extra.{k}": str(v) for k, v in config.extra.items()})

    best_qwk = _load_existing_best_qwk(best_checkpoint_path)
    best_cm_fig = None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_batches)
            val_loss, qwk, auroc, sensitivity, specificity, cm = validate(
                model, val_loader, criterion, device, config.num_classes, max_batches
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
                },
                step=epoch,
            )

            if qwk > best_qwk:
                best_qwk = qwk
                best_cm_fig = render_confusion_matrix(cm)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": dataclasses.asdict(config),
                        "epoch": epoch,
                        "val_qwk": qwk,
                    },
                    best_checkpoint_path,
                )

        if best_cm_fig is not None:
            mlflow.log_figure(best_cm_fig, "confusion_matrix.png")
        mlflow.log_artifact(str(best_checkpoint_path))
        mlflow.log_param("best_checkpoint_path", str(best_checkpoint_path))

    return {"best_qwk": best_qwk, "best_checkpoint_path": str(best_checkpoint_path)}


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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
