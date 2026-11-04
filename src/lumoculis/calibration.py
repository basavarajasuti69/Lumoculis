"""Probability calibration for Lumoculis.

Provides multiclass Brier score, Expected Calibration Error (ECE) with a
documented equal-width binning strategy, reliability diagrams, binary
calibration for referable DR (grade >= 2), and optional post-training
temperature scaling.

IMPORTANT: Calibration may improve probability reliability but does NOT
necessarily improve classification accuracy or clinical safety. Calibration
parameters must be fit only on a designated calibration/validation subset,
never on the final test set, to avoid data leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lumoculis.metrics import REFERRABLE_THRESHOLD_GRADE, _referable_binary, _referable_probability

NUM_CLASSES = 5
DEFAULT_N_BINS = 10
BIN_STRATEGY = "equal_width"  # document the binning convention used
EXPECTED_PROB_NDIM = 2
BINARY_REFERRABLE_THRESHOLD = 0.5


def _validate_probs_and_labels(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate probability and label arrays.

    Returns the arrays as float64 / int64 numpy arrays.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim != EXPECTED_PROB_NDIM or probs.shape[1] != num_classes:
        raise ValueError(f"probs must have shape (n_samples, {num_classes}), got {probs.shape}")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError(
            f"labels must have shape (n_samples,) with n_samples={probs.shape[0]}, "
            f"got {labels.shape}"
        )
    if probs.shape[0] == 0:
        raise ValueError("probs and labels must not be empty")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs contains NaN or Inf values")
    if not np.all((probs >= 0.0) & (probs <= 1.0)):
        raise ValueError("probs must contain values in [0, 1]")
    if not np.all(np.isfinite(labels)):
        raise ValueError("labels contains NaN or Inf values")
    if not np.all((labels >= 0) & (labels < num_classes)):
        raise ValueError(f"labels must contain integers in [0, {num_classes - 1}]")

    # Each row of probs should sum to ~1 (valid probability distribution).
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each row of probs must sum to 1 (valid probability distribution)")

    return probs, labels


def multiclass_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    """Multiclass Brier score.

    Definition:
        BS = (1 / N) * sum_{i=1}^{N} sum_{c=1}^{C} (p_{ic} - y_{ic})^2

    where p_{ic} is the predicted probability for class c for sample i, and
    y_{ic} is 1 if sample i truly belongs to class c, else 0.

    The score is averaged over both samples and classes. Range is [0, 2];
    lower is better. A perfect classifier (p_{ic} = y_{ic} for all i, c) gets 0.

    This is the same convention as sklearn's
    ``brier_score_loss`` with multiclass one-hot encoding, averaged over all
    N*C terms. Do not mix this convention with per-class averaging when
    comparing experiments.
    """
    probs, labels = _validate_probs_and_labels(probs, labels, num_classes)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0

    return float(np.mean((probs - one_hot) ** 2))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
    n_bins: int = DEFAULT_N_BINS,
    bin_strategy: str = BIN_STRATEGY,
) -> dict:
    """Expected Calibration Error (ECE) using equal-width binning on max confidence.

    Binning strategy:
      - Confidence per sample = max_c p_{ic} (the model's top predicted probability).
      - Predicted class = argmax_c p_{ic}.
      - Samples are sorted into ``n_bins`` equal-width bins spanning [0, 1]
        based on confidence.
      - For each non-empty bin b:
          acc_b = fraction of samples in bin b whose predicted class is correct
          conf_b = mean confidence of samples in bin b
          prop_b = fraction of all samples in bin b
      - ECE = sum_b prop_b * |acc_b - conf_b|

    Returns a dict with:
      - ece: float, the expected calibration error
      - per_bin: list of dicts with bin index, confidence, accuracy, prop,
                 lower/upper bin edges, and sample count
    """
    probs, labels = _validate_probs_and_labels(probs, labels, num_classes)

    if probs.shape[0] == 0:
        return {"ece": float("nan"), "per_bin": []}

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    per_bin = []
    ece = 0.0
    n_total = len(labels)

    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        # Include the right edge for the final bin.
        if b < n_bins - 1:
            in_bin = (confidences >= lo) & (confidences < hi)
        else:
            in_bin = (confidences >= lo) & (confidences <= hi)

        count = int(in_bin.sum())
        if count == 0:
            per_bin.append({
                "bin": b,
                "lower": float(lo),
                "upper": float(hi),
                "count": 0,
                "confidence": float("nan"),
                "accuracy": float("nan"),
                "prop": 0.0,
            })
            continue

        bin_conf = float(confidences[in_bin].mean())
        bin_acc = float(accuracies[in_bin].mean())
        bin_prop = count / n_total
        ece += bin_prop * abs(bin_acc - bin_conf)

        per_bin.append({
            "bin": b,
            "lower": float(lo),
            "upper": float(hi),
            "count": count,
            "confidence": bin_conf,
            "accuracy": bin_acc,
            "prop": bin_prop,
        })

    return {"ece": float(ece), "per_bin": per_bin}


def _binary_referable(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold_grade: int = REFERRABLE_THRESHOLD_GRADE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert multiclass probs/labels to binary referable scores/labels.

    Returns (referable_probs, referable_labels) where
    referable_probs = P(grade >= threshold_grade) = sum of probs for grades
    threshold_grade..num_classes-1, and referable_labels = 1 if true grade
    >= threshold_grade else 0.
    """
    probs, labels = _validate_probs_and_labels(probs, labels)
    ref_probs = _referable_probability(probs)
    ref_labels = _referable_binary(labels)
    return ref_probs, ref_labels


def binary_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold_grade: int = REFERRABLE_THRESHOLD_GRADE,
) -> float:
    """Binary Brier score for referable DR (grade >= threshold_grade).

    BS = (1 / N) * sum_{i=1}^{N} (p_i - y_i)^2
    where p_i = P(referable) for sample i and y_i in {0, 1}.
    Range is [0, 1]; lower is better.
    """
    ref_probs, ref_labels = _binary_referable(probs, labels, threshold_grade)
    return float(np.mean((ref_probs - ref_labels) ** 2))


def binary_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    bin_strategy: str = BIN_STRATEGY,
    threshold_grade: int = REFERRABLE_THRESHOLD_GRADE,
) -> dict:
    """Binary ECE for referable DR using equal-width binning on P(referable).

    Returns the same structure as ``expected_calibration_error`` but with
    binary referable confidence/accuracy.
    """
    ref_probs, ref_labels = _binary_referable(probs, labels, threshold_grade)

    if ref_probs.shape[0] == 0:
        return {"ece": float("nan"), "per_bin": []}

    confidences = ref_probs
    predictions = (ref_probs >= BINARY_REFERRABLE_THRESHOLD).astype(np.int64)
    accuracies = (predictions == ref_labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    per_bin = []
    ece = 0.0
    n_total = len(ref_labels)

    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        if b < n_bins - 1:
            in_bin = (confidences >= lo) & (confidences < hi)
        else:
            in_bin = (confidences >= lo) & (confidences <= hi)

        count = int(in_bin.sum())
        if count == 0:
            per_bin.append({
                "bin": b,
                "lower": float(lo),
                "upper": float(hi),
                "count": 0,
                "confidence": float("nan"),
                "accuracy": float("nan"),
                "prop": 0.0,
            })
            continue

        bin_conf = float(confidences[in_bin].mean())
        bin_acc = float(accuracies[in_bin].mean())
        bin_prop = count / n_total
        ece += bin_prop * abs(bin_acc - bin_conf)

        per_bin.append({
            "bin": b,
            "lower": float(lo),
            "upper": float(hi),
            "count": count,
            "confidence": bin_conf,
            "accuracy": bin_acc,
            "prop": bin_prop,
        })

    return {"ece": float(ece), "per_bin": per_bin}


@dataclass
class TemperatureScaling:
    """Post-training temperature scaling for multiclass probabilities.

    Temperature T > 0 scales logits before softmax:
        p_c = softmax(logits_c / T)

    T == 1 means no scaling. T > 1 softens (lowers confidence); T < 1 sharpens
    (raises confidence). Fit T on a held-out calibration subset using
    L-BFGS-B minimization of cross-entropy on scaled logits.

    Attributes
    ----------
    temperature : float
        The fitted temperature. 1.0 if not yet fit.
    fitted : bool
        Whether temperature has been fit on calibration data.
    num_classes : int
        Number of classes.
    """

    temperature: float = 1.0
    fitted: bool = False
    num_classes: int = NUM_CLASSES
    calibration_loss: float | None = None

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> TemperatureScaling:
        """Fit temperature on calibration logits and labels.

        Parameters
        ----------
        logits : np.ndarray
            Raw model logits of shape (n_samples, num_classes), NOT softmaxed.
        labels : np.ndarray
            True class labels of shape (n_samples,).

        Returns
        -------
        self : TemperatureScaling
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if logits.ndim != EXPECTED_PROB_NDIM or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits must have shape (n_samples, {self.num_classes}), got {logits.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must match logits sample count")
        if logits.shape[0] == 0:
            raise ValueError("logits and labels must not be empty")
        if not np.all(np.isfinite(logits)):
            raise ValueError("logits contains NaN or Inf values")
        if not np.all((labels >= 0) & (labels < self.num_classes)):
            raise ValueError(f"labels must be integers in [0, {self.num_classes - 1}]")

        logits_t = torch.from_numpy(logits)
        labels_t = torch.from_numpy(labels)

        # Optimize log(T) to keep T > 0 (T = exp(log_T)).
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            t = torch.exp(log_t)
            scaled = logits_t / t
            loss = torch.nn.functional.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = float(torch.exp(log_t).item())
        self.fitted = True
        # Record calibration loss at the fitted temperature.
        with torch.no_grad():
            t = torch.exp(log_t)
            scaled = logits_t / t
            self.calibration_loss = float(
                torch.nn.functional.cross_entropy(scaled, labels_t).item()
            )
        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted temperature to logits and return calibrated probs.

        Parameters
        ----------
        logits : np.ndarray
            Raw model logits of shape (n_samples, num_classes).

        Returns
        -------
        probs : np.ndarray
            Calibrated probabilities of shape (n_samples, num_classes).
        """
        if not self.fitted:
            # No scaling applied; return softmax of raw logits.
            t = 1.0
        else:
            t = self.temperature

        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != EXPECTED_PROB_NDIM or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits must have shape (n_samples, {self.num_classes}), got {logits.shape}"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError("logits contains NaN or Inf values")

        # Numerically stable softmax with temperature.
        shifted = logits / t
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    def to_dict(self) -> dict:
        """Serialize temperature and metadata to a plain dict."""
        return {
            "temperature": self.temperature,
            "fitted": self.fitted,
            "num_classes": self.num_classes,
            "calibration_loss": self.calibration_loss,
            "method": "temperature_scaling",
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemperatureScaling:
        """Restore a TemperatureScaling from a dict produced by ``to_dict``."""
        obj = cls(
            temperature=float(data["temperature"]),
            fitted=bool(data["fitted"]),
            num_classes=int(data["num_classes"]),
        )
        obj.calibration_loss = (
            float(data["calibration_loss"]) if data.get("calibration_loss") is not None else None
        )
        return obj

    def save(self, path: str | Path) -> None:
        """Save temperature and metadata to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaling:
        """Load temperature and metadata from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


def plot_reliability_diagram(
    calibration_result: dict,
    output_path: str | Path,
    title: str | None = None,
    bin_strategy: str = BIN_STRATEGY,
) -> None:
    """Render a reliability diagram from an ECE result dict.

    Shows predicted confidence vs observed accuracy per bin, with an ideal
    calibration diagonal and per-bin sample counts annotated.
    Bins with zero samples are omitted from the plot (to avoid misleading
    diagrams on empty bins).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    bins = [b for b in calibration_result["per_bin"] if b["count"] > 0]
    if not bins:
        # Still save an empty diagram with a note.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Ideal calibration")
        ax.text(0.5, 0.5, "No populated bins", ha="center", va="center")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title or "Reliability diagram")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    confidences = [b["confidence"] for b in bins]
    accuracies = [b["accuracy"] for b in bins]
    counts = [b["count"] for b in bins]
    lowers = [b["lower"] for b in bins]
    uppers = [b["upper"] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Ideal calibration diagonal.
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Ideal calibration")

    # Bars: bin width proportional to prop (sample fraction).
    widths = [uppers[i] - lowers[i] for i in range(len(bins))]
    ax.bar(
        confidences,
        accuracies,
        width=widths,
        alpha=0.6,
        label="Observed accuracy",
        edgecolor="black",
        linewidth=0.5,
    )

    # Line connecting bin accuracies.
    ax.plot(confidences, accuracies, "o-", color="C1", label="Model calibration")

    # Annotate sample counts.
    for x, y, c in zip(confidences, accuracies, counts, strict=False):
        ax.annotate(
            f"n={c}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title or f"Reliability diagram (binning: {bin_strategy})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
