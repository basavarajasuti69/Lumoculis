"""Evaluation metrics for Lumoculis.

All functions operate on numpy arrays and return plain Python floats/ints.
Keep the logic explicit and dependency-light so it can be explained line by
line in an interview.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)

matplotlib.use("Agg")
NUM_CLASSES = 5
REFERRABLE_THRESHOLD_GRADE = 2  # grade >= 2 is "referable DR"
MIN_CLASSES_FOR_AUROC = 2


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa between true and predicted DR grades (0-4)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _referable_binary(y_true: np.ndarray) -> np.ndarray:
    """Convert grades 0-4 to a binary referable flag (1 = grade >= 2)."""
    return (y_true >= REFERRABLE_THRESHOLD_GRADE).astype(int)


def _referable_probability(probs: np.ndarray) -> np.ndarray:
    """P(referable) = sum of softmax probabilities for grades 2, 3, 4."""
    return probs[:, REFERRABLE_THRESHOLD_GRADE:].sum(axis=1)


def referable_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """AUROC for referable DR using summed grade 2-4 probability as the score.

    Returns NaN when the validation fold contains only one referable class
    (AUROC is undefined in that case).
    """
    y_bin = _referable_binary(y_true)
    p = _referable_probability(probs)
    if len(np.unique(y_bin)) < MIN_CLASSES_FOR_AUROC:
        return float("nan")
    return float(roc_auc_score(y_bin, p))


def referable_sensitivity_specificity(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """Sensitivity and specificity for referable DR at *threshold*.

    A prediction is "referable" when P(referable) >= threshold.
    """
    y_bin = _referable_binary(y_true)
    p = _referable_probability(probs)
    y_pred = (p >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_bin == 1)).sum())
    tn = int(((y_pred == 0) & (y_bin == 0)).sum())
    fp = int(((y_pred == 1) & (y_bin == 0)).sum())
    fn = int(((y_pred == 0) & (y_bin == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sensitivity), float(specificity)


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Confusion matrix of shape (num_classes, num_classes); rows=true, cols=pred."""
    return sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def render_confusion_matrix(cm: np.ndarray, class_names: Sequence[str] | None = None):
    """Render a confusion matrix as a matplotlib Figure (for MLflow logging)."""
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (DR grades 0-4)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = float(cm.max()) / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig
