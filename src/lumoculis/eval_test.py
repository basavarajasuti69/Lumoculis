"""Standalone final test-set evaluation for Lumoculis.

This module never trains a model or selects checkpoints. It evaluates an
explicitly provided checkpoint on a test split derived from the historical
training partition while preserving the historical validation IDs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lumoculis.config import Config
from lumoculis.data import (
    LumoculisDataset,
    _representative_label,
    find_duplicate_groups,
    train_transforms,
    train_val_split,
    train_val_test_split_preserving_validation,
    val_transforms,
)
from lumoculis.inference import _load_checkpoint
from lumoculis.metrics import (
    NUM_CLASSES,
    compute_confusion_matrix,
    quadratic_weighted_kappa,
    referable_auroc,
    referable_sensitivity_specificity,
)

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        precision_recall_fscore_support,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"scikit-learn is required for evaluation: {exc}") from exc


def _collect_predictions(model, dataset, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for batch, labels in loader:
            inputs = batch.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())
    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0).astype(int)
    return y_true, y_prob


def run_test_evaluation(  # noqa: PLR0913
    csv_path: str | Path,
    processed_dir: str | Path,
    config: Config,
    checkpoint_path: str | Path,
    output_root: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Historical split for exact validation-ID preservation.
    hist_train_ds, hist_val_ds, _ = train_val_split(
        csv_path, processed_dir, config, val_size=val_size
    )
    hist_val_ids = set(hist_val_ds.df["id_code"].astype(str))

    train_ds, val_ds, test_ds, split_report = train_val_test_split_preserving_validation(
        csv_path, processed_dir, config, val_size=val_size, test_size=test_size
    )

    new_val_ids = set(val_ds.df["id_code"].astype(str))
    train_ids = set(train_ds.df["id_code"].astype(str))
    test_ids = set(test_ds.df["id_code"].astype(str))

    if new_val_ids != hist_val_ids:
        raise RuntimeError("Historical validation IDs were not preserved.")
    if train_ids & new_val_ids or train_ids & test_ids or new_val_ids & test_ids:
        raise RuntimeError("Training, validation, and test IDs are not pairwise disjoint.")

    torch_device = torch.device(device)
    model, metadata = _load_checkpoint(checkpoint_path, torch_device)

    y_true, y_prob = _collect_predictions(model, test_ds, torch_device)
    y_pred = y_prob.argmax(axis=1)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)

    metrics = {
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "referable_auroc": float(referable_auroc(y_true, y_prob)),
        "referable_sensitivity": float(referable_sensitivity_specificity(y_true, y_prob, threshold)[0]),
        "referable_specificity": float(referable_sensitivity_specificity(y_true, y_prob, threshold)[1]),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            str(i): {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "n_test": len(y_true),
        "true_counts": [int(x) for x in np.bincount(y_true, minlength=NUM_CLASSES)],
        "predicted_counts": [int(x) for x in np.bincount(y_pred, minlength=NUM_CLASSES)],
    }

    run_dir = Path(output_root) / f"test_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"  # noqa: UP017
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": "test",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_metadata": metadata,
                "split_report": split_report,
                "historical_val_ids_count": len(hist_val_ids),
                "train_count": len(train_ids),
                "val_count": len(new_val_ids),
                "test_count": len(test_ids),
                "class_distribution_test": metrics["true_counts"],
                "threshold": threshold,
                "seed": config.seed,
                "metrics": metrics,
            },
            handle,
            indent=2,
        )

    with open(run_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grade", "precision", "recall", "f1", "support"])
        for i in range(NUM_CLASSES):
            writer.writerow(
                [
                    i,
                    metrics["per_class"][str(i)]["precision"],
                    metrics["per_class"][str(i)]["recall"],
                    metrics["per_class"][str(i)]["f1"],
                    metrics["per_class"][str(i)]["support"],
                ]
            )

    with open(run_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *list(range(NUM_CLASSES))])
        for i in range(NUM_CLASSES):
            writer.writerow([i, *cm[i].tolist()])

    return {"run_dir": str(run_dir), "metrics": metrics}


def create_holdout_split(  # noqa: PLR0912, PLR0913, PLR0915
    csv_path: str | Path,
    processed_dir: str | Path,
    config: Config,
    historical_ids: set[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
):


    df = pd.read_csv(csv_path)
    id_to_label = {str(row.id_code): int(row.diagnosis) for row in df.itertuples(index=False)}
    all_ids = list(id_to_label.keys())
    if not set(historical_ids).issubset(id_to_label):
        raise ValueError("Historical IDs are absent from the dataset.")

    hash_groups = find_duplicate_groups(processed_dir, all_ids)

    excluded_ids = set(historical_ids)
    intersecting_groups = []
    for ids in hash_groups.values():
        if set(ids) & historical_ids:
            intersecting_groups.append(ids)
            excluded_ids.update(ids)

    eligible_ids = [x for x in all_ids if x not in excluded_ids]

    # Group eligible exact duplicates.
    eligible_groups = []
    for ids in hash_groups.values():
        if set(ids).isdisjoint(excluded_ids):
            eligible_groups.append(ids)
    # singleton images not in hash_groups? find_duplicate_groups returns all? Check: likely returns groups for all ids. Assume yes. If not, add missing as singletons.
    grouped_ids = {x for g in eligible_groups for x in g}
    for x in eligible_ids:
        if x not in grouped_ids:
            eligible_groups.append([x])

    rng = __import__("random").Random(seed)
    groups_by_class = defaultdict(list)
    for index, ids in enumerate(eligible_groups):
        labels = [id_to_label[x] for x in ids]
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
        return [x for index in indices for x in eligible_groups[index]]

    train_ids = expand(train_indices)
    val_ids = expand(val_indices)
    test_ids = expand(test_indices)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Holdout split IDs overlap.")
    if train_set & excluded_ids or val_set & excluded_ids or test_set & excluded_ids:
        raise RuntimeError("Historical IDs leaked into holdout split.")
    if not (train_set | val_set | test_set).issubset(set(eligible_ids)):
        raise RuntimeError("Holdout split contains ineligible images.")
    if len(train_set | val_set | test_set) != len(eligible_ids):
        raise RuntimeError("Not all eligible images were assigned.")

    # duplicate group integrity
    for ids in eligible_groups:
        s = set(ids)
        hits = sum(1 for part in (train_set, val_set, test_set) if s & part)
        if hits > 1:
            raise RuntimeError(f"Duplicate group crossed holdout split: {ids}")

    conflicting_groups = [
        (h, ids) for h, ids in hash_groups.items()
        if len(ids) > 1 and len({id_to_label[x] for x in ids}) > 1
    ]

    report = {
        "n_historical_ids": len(historical_ids),
        "n_excluded_ids": len(excluded_ids),
        "n_intersecting_duplicate_groups": len(intersecting_groups),
        "n_eligible": len(eligible_ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "conflicting_groups": conflicting_groups,
        "cross_split_groups": [],
        "seed": seed,
    }

    train_df = df[df["id_code"].astype(str).isin(train_set)].copy().reset_index(drop=True)
    val_df = df[df["id_code"].astype(str).isin(val_set)].copy().reset_index(drop=True)
    test_df = df[df["id_code"].astype(str).isin(test_set)].copy().reset_index(drop=True)

    train_ds = LumoculisDataset(train_df, processed_dir, config.image_size, transform=train_transforms(config.augment))
    val_ds = LumoculisDataset(val_df, processed_dir, config.image_size, transform=val_transforms())
    test_ds = LumoculisDataset(test_df, processed_dir, config.image_size, transform=val_transforms())

    return train_ds, val_ds, test_ds, report, train_ids, val_ids, test_ids, excluded_ids


def run_holdout_evaluation(  # noqa: PLR0913
    csv_path,
    processed_dir,
    config,
    checkpoint_path,
    historical_ids,
    output_root,
    val_size=0.15,
    test_size=0.15,
    seed=42,
    device="cpu",
    threshold=0.5,
):


    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_ds, val_ds, test_ds, report, train_ids, val_ids, test_ids, excluded_ids = create_holdout_split(
        csv_path, processed_dir, config, set(historical_ids), val_size=val_size, test_size=test_size, seed=seed
    )

    if set(historical_ids) & (set(train_ids) | set(val_ids) | set(test_ids)):
        raise RuntimeError("Historical IDs leaked into holdout split.")

    torch_device = torch.device(device)
    model, metadata = _load_checkpoint(checkpoint_path, torch_device)

    ckpt_config = metadata.get("config", {})
    compatibility = {
        "model_name": {
            "checkpoint": ckpt_config.get("model_name"),
            "config": config.model_name,
            "verified": ckpt_config.get("model_name") == config.model_name if "model_name" in ckpt_config else None,
        },
        "num_classes": {
            "checkpoint": ckpt_config.get("num_classes"),
            "config": config.num_classes,
            "verified": ckpt_config.get("num_classes") == config.num_classes if "num_classes" in ckpt_config else None,
        },
        "image_size": {
            "checkpoint": ckpt_config.get("image_size"),
            "config": config.image_size,
            "verified": ckpt_config.get("image_size") == config.image_size if "image_size" in ckpt_config else None,
        },
    }

    ckpt_config = metadata.get("config", {})
    compatibility = {
        "model_name": {
            "checkpoint": ckpt_config.get("model_name"),
            "config": config.model_name,
            "verified": ckpt_config.get("model_name") == config.model_name if "model_name" in ckpt_config else None,
        },
        "num_classes": {
            "checkpoint": ckpt_config.get("num_classes"),
            "config": config.num_classes,
            "verified": ckpt_config.get("num_classes") == config.num_classes if "num_classes" in ckpt_config else None,
        },
        "image_size": {
            "checkpoint": ckpt_config.get("image_size"),
            "config": config.image_size,
            "verified": ckpt_config.get("image_size") == config.image_size if "image_size" in ckpt_config else None,
        },
    }

    loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    probs_list, labels_list = [], []
    with torch.no_grad():
        for batch, labels in loader:
            inputs = batch.to(torch_device)
            logits = model(inputs)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(labels.numpy())
    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0).astype(int)
    y_pred = y_prob.argmax(axis=1)

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)

    metrics = {
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "referable_auroc": float(referable_auroc(y_true, y_prob)),
        "referable_sensitivity": float(referable_sensitivity_specificity(y_true, y_prob, threshold)[0]),
        "referable_specificity": float(referable_sensitivity_specificity(y_true, y_prob, threshold)[1]),
        "confusion_matrix": cm.tolist(),
        "per_class": {str(i): {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(NUM_CLASSES)},
        "n_test": len(y_true),
        "true_counts": [int(x) for x in np.bincount(y_true, minlength=NUM_CLASSES)],
        "predicted_counts": [int(x) for x in np.bincount(y_pred, minlength=NUM_CLASSES)],
    }

    run_dir = Path(output_root) / f"holdout_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({
            "split": "holdout_test",
            "evaluation_type": "exploratory_holdout",
            "limitation": "Original training IDs are unavailable; test images may have been seen during training.",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata": metadata,
            "split_report": report,
            "checkpoint_compatibility": compatibility,
            "historical_id_count": len(historical_ids),
            "excluded_id_count": len(excluded_ids),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "seed": seed,
            "metrics": metrics,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "excluded_ids": sorted(excluded_ids),
        }, handle, indent=2)

    with open(run_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grade", "precision", "recall", "f1", "support"])
        for i in range(NUM_CLASSES):
            writer.writerow([i, metrics["per_class"][str(i)]["precision"], metrics["per_class"][str(i)]["recall"], metrics["per_class"][str(i)]["f1"], metrics["per_class"][str(i)]["support"]])
    with open(run_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *list(range(NUM_CLASSES))])
        for i in range(NUM_CLASSES):
            writer.writerow([i, *cm[i].tolist()])

    return {"run_dir": str(run_dir), "metrics": metrics}
