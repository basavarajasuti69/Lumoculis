"""Training loop for Lumoculis.

Device selection: mps if available, else cpu. ``PYTORCH_ENABLE_MPS_FALLBACK=1``
is set before torch is imported so MPS ops that lack a Metal implementation
fall back to CPU instead of crashing. No float64, no autocast/fp16 for now.

Timing: every phase is measured with ``time.perf_counter()``. On MPS,
``torch.mps.synchronize()`` is called at timing boundaries so that GPU
execution time is reflected in CPU timings. Synchronization is NOT performed
inside every batch — only at phase boundaries — to avoid serializing the GPU.
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
from lumoculis.data import (
    LumoculisDataset,
    class_counts_and_weights,
    create_weighted_sampler,
    train_val_split,
)
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


def _mps_sync() -> None:
    """Synchronize MPS if available; no-op on CPU."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


MLFLOW_LOG_THRESHOLD_S = 0.1


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
) -> dict[str, float | int | torch.Tensor]:
    """Run one training epoch with detailed per-phase timing.

    Returns a dict with:
      - total_loss: accumulated loss tensor (on device)
      - n_batches: number of batches processed
      - data_retrieval_s: time spent in DataLoader.__next__ (__getitem__ + collate)
      - device_transfer_s: time spent moving batch to device
      - forward_s: time spent in forward pass
      - backward_s: time spent in backward pass
      - optimizer_s: time spent in optimizer step
      - compute_s: total compute (forward + backward + optimizer)

    MPS is synchronized at phase boundaries so GPU execution time is reflected
    in CPU timings. Synchronization is NOT performed inside every batch.
    """
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    n_batches = 0
    data_retrieval_s = 0.0
    device_transfer_s = 0.0
    forward_s = 0.0
    backward_s = 0.0
    optimizer_s = 0.0

    iterator = iter(loader)
    total_batches = max_batches if max_batches is not None else len(loader)
    progress_interval = 25
    t_epoch_start = time.perf_counter()
    for batch_idx in range(total_batches):
        # --- Data retrieval: blocking on DataLoader.__next__ (__getitem__) ---
        t0 = time.perf_counter()
        try:
            images, labels = next(iterator)
        except StopIteration:
            break
        data_retrieval_s += time.perf_counter() - t0
        if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.perf_counter() - t_epoch_start
            print(f"  [train] batch {batch_idx + 1}/{total_batches} elapsed={elapsed:.1f}s")

        # --- Device transfer ---
        t0 = time.perf_counter()
        images = images.to(device)
        labels = labels.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        device_transfer_s += time.perf_counter() - t0

        # --- Forward pass ---
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if device.type == "mps":
            torch.mps.synchronize()
        forward_s += time.perf_counter() - t0

        # --- Loss ---
        loss = criterion(logits, labels)

        # --- Backward pass ---
        t0 = time.perf_counter()
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        backward_s += time.perf_counter() - t0

        # --- Optimizer step ---
        t0 = time.perf_counter()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        optimizer_s += time.perf_counter() - t0

        running_loss = running_loss + loss.detach()
        n_batches += 1

    return {
        "total_loss": running_loss,
        "n_batches": n_batches,
        "data_retrieval_s": data_retrieval_s,
        "device_transfer_s": device_transfer_s,
        "forward_s": forward_s,
        "backward_s": backward_s,
        "optimizer_s": optimizer_s,
        "compute_s": forward_s + backward_s + optimizer_s,
    }


def validate(  # noqa: PLR0913, PLR0915
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    val_ids: np.ndarray | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int | torch.Tensor | np.ndarray | None]:
    """Validate with detailed per-phase timing.

    Returns a dict with val loss, n_batches, timings, metrics, and arrays.
    MPS is synchronized at phase boundaries.
    """
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    n_batches = 0
    data_retrieval_s = 0.0
    device_transfer_s = 0.0
    forward_s = 0.0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    iterator = iter(loader)
    total_batches = max_batches if max_batches is not None else len(loader)
    progress_interval = 25
    t_epoch_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx in range(total_batches):
            # --- Data retrieval ---
            t0 = time.perf_counter()
            try:
                images, labels = next(iterator)
            except StopIteration:
                break
            data_retrieval_s += time.perf_counter() - t0
            if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.perf_counter() - t_epoch_start
                print(f"  [val] batch {batch_idx + 1}/{total_batches} elapsed={elapsed:.1f}s")

            # --- Device transfer ---
            t0 = time.perf_counter()
            images = images.to(device)
            labels = labels.to(device)
            if device.type == "mps":
                torch.mps.synchronize()
            device_transfer_s += time.perf_counter() - t0

            # --- Forward pass ---
            t0 = time.perf_counter()
            logits = model(images)
            if device.type == "mps":
                torch.mps.synchronize()
            forward_s += time.perf_counter() - t0

            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss = running_loss + loss.detach()
            n_batches += 1
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    avg_loss = running_loss / max(n_batches, 1)
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    auroc = referable_auroc(y_true, y_prob)
    sensitivity, specificity = referable_sensitivity_specificity(y_true, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    if val_ids is not None:
        val_ids = val_ids[: len(y_true)]

    return {
        "avg_loss": avg_loss,
        "n_batches": n_batches,
        "data_retrieval_s": data_retrieval_s,
        "device_transfer_s": device_transfer_s,
        "forward_s": forward_s,
        "compute_s": forward_s,
        "qwk": qwk,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "cm": cm,
        "y_true": y_true,
        "y_prob": y_prob,
        "val_ids": val_ids,
    }


def _default_run_name() -> str:
    """Default run name: UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017


def _build_dataloaders(
    train_ds: LumoculisDataset,
    val_ds: LumoculisDataset,
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from the datasets and config."""
    train_sampler = None
    train_shuffle = True
    use_weighted = getattr(config, "use_weighted_sampler", False)
    if use_weighted:
        try:
            train_sampler = create_weighted_sampler(
                train_ds.df, config.num_classes, seed=config.seed
            )
            train_shuffle = False
        except Exception:
            train_sampler = None
            train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle if train_sampler is None else False,
        sampler=train_sampler,
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
) -> tuple[
    nn.Module,
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler._LRScheduler,
    dict[int, float],
]:
    """Build model, loss, optimizer, and scheduler from config.

    Returns (model, criterion, optimizer, scheduler, class_weights).
    """
    model = build_model(config.model_name, config.num_classes, pretrained=pretrained).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    class_weights = _dampened_class_weight_tensor(train_ds.df, config.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights_dict = {i: float(class_weights[i].item()) for i in range(len(class_weights))}
    return model, criterion, optimizer, scheduler, weights_dict


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
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    print(f"  checkpoint+mlflow save: {elapsed:.2f}s")


def _log_mlflow_artifacts(
    no_save: bool,
    run_dir: Path,
    best_checkpoint_path: Path,
) -> None:
    """Log best checkpoint and predictions as MLflow artifacts."""
    if not no_save and best_checkpoint_path.exists():
        t0 = time.perf_counter()
        mlflow.log_artifact(str(best_checkpoint_path))
        mlflow.log_param("best_checkpoint_path", str(best_checkpoint_path))
        mlflow.log_artifact(str(run_dir / "val_preds.npz"))
        elapsed = time.perf_counter() - t0
        print(f"  mlflow artifact upload: {elapsed:.2f}s")


def run_training(  # noqa: PLR0913, PLR0915
    config: Config,
    run_name: str | None = None,
    max_batches: int | None = None,
    epochs_override: int | None = None,
    pretrained: bool = True,
    no_save: bool = False,
) -> dict:
    """Run the full training loop with comprehensive phase timing.

    Returns a dict with the best validation QWK, the run directory, and the
    best checkpoint path (None when no_save).
    """
    # --- Process startup and setup ---
    t_start_total = time.perf_counter()
    device = pick_device()
    seed_everything(config.seed)
    mlflow.end_run()
    print(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")

    epochs = epochs_override if epochs_override is not None else config.epochs

    # --- Dataset and DataLoader initialization ---
    t0 = time.perf_counter()
    csv_path = Path(config.data_dir) / "train.csv"
    train_ds, val_ds, dup_report = train_val_split(csv_path, config.processed_dir, config)
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, config)
    t_dataset_init = time.perf_counter() - t0
    print(f"Dataset + DataLoader init: {t_dataset_init:.2f}s")

    # --- Model initialization ---
    t0 = time.perf_counter()
    model, criterion, optimizer, scheduler, class_weights = _setup_training(
        config, device, train_ds, pretrained, epochs
    )
    t_model_init = time.perf_counter() - t0
    print(f"Model init + device transfer: {t_model_init:.2f}s")

    # --- First training batch (warmup, includes MPS shader compilation) ---
    t0 = time.perf_counter()
    first_batch = next(iter(train_loader))
    images_fb, labels_fb = first_batch
    images_fb = images_fb.to(device)
    labels_fb = labels_fb.to(device)
    if device.type == "mps":
        torch.mps.synchronize()
    t_first_batch_data = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits_fb = model(images_fb)
    loss_fb = criterion(logits_fb, labels_fb)
    loss_fb.backward()
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    t_first_batch_compute = time.perf_counter() - t0
    print(
        f"First batch (warmup): data={t_first_batch_data:.3f}s compute={t_first_batch_compute:.3f}s "
        f"(includes MPS shader compilation)"
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
        mlflow.log_params({f"class_weight/{i}": w for i, w in class_weights.items()})

        for epoch in range(epochs):
            t_epoch0 = time.perf_counter()

            # --- Training ---
            train_result = train_epoch(
                model, train_loader, criterion, optimizer, device, max_batches
            )

            # --- Validation ---
            t_val0 = time.perf_counter()
            val_result = validate(
                model,
                val_loader,
                criterion,
                device,
                config.num_classes,
                val_ids=val_ds.df["id_code"].values,
                max_batches=max_batches,
            )
            t_val_total = time.perf_counter() - t_val0

            scheduler.step()
            t_epoch_total = time.perf_counter() - t_epoch0

            train_loss = float(
                train_result["total_loss"].item() / max(train_result["n_batches"], 1)
            )
            val_loss = float(val_result["avg_loss"].item()) if isinstance(val_result["avg_loss"], torch.Tensor) else float(val_result["avg_loss"])

            print(
                f"epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"qwk={val_result['qwk']:.4f} auroc={val_result['auroc']:.4f} "
                f"sens={val_result['sensitivity']:.4f} spec={val_result['specificity']:.4f} "
                f"({t_epoch_total:.1f}s)"
            )
            print(
                f"  timing: train_batches={train_result['n_batches']} "
                f"val_batches={val_result['n_batches']} "
                f"train_data_retrieval={train_result['data_retrieval_s']:.2f}s "
                f"train_device_transfer={train_result['device_transfer_s']:.2f}s "
                f"train_forward={train_result['forward_s']:.2f}s "
                f"train_backward={train_result['backward_s']:.2f}s "
                f"train_optimizer={train_result['optimizer_s']:.2f}s "
                f"val_data_retrieval={val_result['data_retrieval_s']:.2f}s "
                f"val_device_transfer={val_result['device_transfer_s']:.2f}s "
                f"val_forward={val_result['forward_s']:.2f}s "
                f"max_batches={max_batches if max_batches is not None else 'full'}"
            )
            print(
                f"  epoch breakdown: train={t_epoch_total - t_val_total:.2f}s "
                f"val={t_val_total:.2f}s total={t_epoch_total:.2f}s"
            )

            # --- MLflow metrics ---
            t_mlflow0 = time.perf_counter()
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "qwk": float(val_result["qwk"]),
                    "referable_auroc": float(val_result["auroc"]),
                    "referable_sensitivity": float(val_result["sensitivity"]),
                    "referable_specificity": float(val_result["specificity"]),
                    "epoch_time_s": round(t_epoch_total, 3),
                    "train_data_retrieval_s": round(float(train_result["data_retrieval_s"]), 3),
                    "train_device_transfer_s": round(float(train_result["device_transfer_s"]), 3),
                    "train_forward_s": round(float(train_result["forward_s"]), 3),
                    "train_backward_s": round(float(train_result["backward_s"]), 3),
                    "train_optimizer_s": round(float(train_result["optimizer_s"]), 3),
                    "val_data_retrieval_s": round(float(val_result["data_retrieval_s"]), 3),
                    "val_device_transfer_s": round(float(val_result["device_transfer_s"]), 3),
                    "val_forward_s": round(float(val_result["forward_s"]), 3),
                },
                step=epoch,
            )
            t_mlflow = time.perf_counter() - t_mlflow0
            if t_mlflow > MLFLOW_LOG_THRESHOLD_S:
                print(f"  mlflow log_metrics: {t_mlflow:.2f}s")

            # --- Best checkpoint ---
            if val_result["qwk"] > best_qwk:
                best_qwk = float(val_result["qwk"])
                best_cm_fig = render_confusion_matrix(val_result["cm"])
                best_val_ids = val_result["val_ids"]
                best_y_true = val_result["y_true"]
                best_y_prob = val_result["y_prob"]
                _save_best_epoch(
                    no_save,
                    run_dir,
                    model,
                    config,
                    epoch,
                    best_qwk,
                    best_cm_fig,
                    best_val_ids,
                    best_y_true,
                    best_y_prob,
                )

        _log_mlflow_artifacts(no_save, run_dir, best_checkpoint_path)

    t_total = time.perf_counter() - t_start_total
    print(f"Total wall-clock time: {t_total:.1f}s")

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
