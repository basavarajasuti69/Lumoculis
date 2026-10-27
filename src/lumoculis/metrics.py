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


def bootstrap_ci(  # noqa: PLR0913
    metric_fn,
    y_true: np.ndarray,
    y_pred_or_probs: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    is_proba: bool = False,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Resamples (y_true, y_pred_or_probs) pairs with replacement *n_bootstrap*
    times, computes the metric on each resample, and returns the point
    estimate plus the (1 - ci) / 2 and 1 - (1 - ci) / 2 percentiles.

    Parameters
    ----------
    metric_fn : callable
        Function taking (y_true, y_pred_or_probs) and returning a float.
    y_true : np.ndarray
        Ground truth labels.
    y_pred_or_probs : np.ndarray
        Predicted labels or probabilities, aligned with y_true.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95).
    seed : int
        Random seed for reproducibility.
    is_proba : bool
        If True, y_pred_or_probs are probabilities (2D array). If False, they
        are predicted class labels (1D array).

    Returns
    -------
    (point_estimate, ci_lower, ci_upper) : tuple[float, float, float]
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Point estimate on the full data.
    point = metric_fn(y_true, y_pred_or_probs)

    if n_bootstrap <= 1:
        return float(point), float("nan"), float("nan")

    bootstrap_stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_resampled = y_true[idx]
        if is_proba:
            y_pred_resampled = y_pred_or_probs[idx]
        else:
            y_pred_resampled = y_pred_or_probs[idx]
        try:
            stat = metric_fn(y_true_resampled, y_pred_resampled)
        except Exception:  # noqa: S112 - skip resamples where metric is undefined
            continue
        bootstrap_stats.append(float(stat))

    if len(bootstrap_stats) == 0:
        return float(point), float("nan"), float("nan")

    alpha = 1.0 - ci
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


def bootstrap_qwk_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for quadratic weighted kappa."""
    return bootstrap_ci(
        quadratic_weighted_kappa,
        y_true,
        y_pred,
        n_bootstrap,
        ci,
        seed,
        is_proba=False,
    )


def bootstrap_referable_auroc_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for referable DR AUROC."""
    return bootstrap_ci(
        referable_auroc,
        y_true,
        probs,
        n_bootstrap,
        ci,
        seed,
        is_proba=True,
    )


def bootstrap_referable_sens_spec_ci(  # noqa: PLR0913
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Bootstrap CI for referable DR sensitivity and specificity at *threshold*.

    Returns ((sens_point, sens_lo, sens_hi), (spec_point, spec_lo, spec_hi)).
    """

    def _sens(y_t, p):
        s, _ = referable_sensitivity_specificity(y_t, p, threshold=threshold)
        return s

    def _spec(y_t, p):
        _, sp = referable_sensitivity_specificity(y_t, p, threshold=threshold)
        return sp

    sens_ci = bootstrap_ci(_sens, y_true, probs, n_bootstrap, ci, seed, is_proba=True)
    spec_ci = bootstrap_ci(_spec, y_true, probs, n_bootstrap, ci, seed, is_proba=True)
    return sens_ci, spec_ci


def bootstrap_per_class_sensitivity_ci(  # noqa: PLR0913
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> list[tuple[float, float, float]]:
    """Bootstrap CI for per-class sensitivity.

    Returns a list of (point, ci_lower, ci_upper) per class.
    """

    def _class_sens(y_t, y_p, cls: int):
        cm = compute_confusion_matrix(y_t, y_p, num_classes=num_classes)
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results: list[tuple[float, float, float]] = []
    for cls in range(num_classes):

        def _sens_for_cls(y_t, y_p, _cls: int = cls):
            return _class_sens(y_t, y_p, _cls)

        point, lo, hi = bootstrap_ci(
            _sens_for_cls,
            y_true,
            y_pred,
            n_bootstrap,
            ci,
            seed,
            is_proba=False,
        )
        results.append((point, lo, hi))
    return results
