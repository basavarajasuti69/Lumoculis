"""Command-line interface for probability calibration evaluation.

Evaluates calibration metrics (Brier score, ECE, reliability diagram) from
saved model outputs (logits or probabilities) and labels. Supports optional
post-training temperature scaling fit on a held-out calibration subset.

Usage:

    lumoculis-calibrate \\
        --logits data/val_logits.npy \\
        --labels data/val_labels.npy \\
        --calibration-fraction 0.3 \\
        --output-dir ./calibration_output \\
        --reliability-diagram reliability.png \\
        --temperature-out temperature.json

Does not require retraining the model. Temperature is fit only on the
designated calibration subset to avoid data leakage.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lumoculis.calibration import (
    DEFAULT_N_BINS,
    EXPECTED_PROB_NDIM,
    NUM_CLASSES,
    TemperatureScaling,
    binary_brier_score,
    binary_ece,
    expected_calibration_error,
    multiclass_brier_score,
    plot_reliability_diagram,
)

MLFLOW_TRACKING_URI = "mlruns"


@dataclass
class CalibrationOutputPaths:
    """Paths for calibration output artifacts."""

    output_dir: Path
    diagram_after: str
    temperature: str


@dataclass
class TemperatureScalingConfig:
    """Configuration for temperature scaling."""

    calibration_fraction: float
    seed: int
    n_bins: int


def _load_array(path: str | Path, name: str) -> np.ndarray:
    """Load a numpy array from .npy or .npz file."""
    path = Path(path)
    if not path.exists():
        print(f"Error: {name} file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 1:
            return data[data.files[0]]
        print(
            f"Error: {name} .npz contains multiple arrays {data.files}; "
            "use --npz-key to specify which array to load.",
            file=sys.stderr,
        )
        sys.exit(1)
    return np.load(path)


def _load_npz(
    path: str | Path,
    probs_key: str | None,
    logits_key: str | None,
    labels_key: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load probs, logits, and labels from an .npz file using specific keys."""
    path = Path(path)
    if not path.exists():
        print(f"Error: NPZ file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(path)

    def _get(key: str | None) -> np.ndarray | None:
        if key is None:
            return None
        if key not in data:
            print(
                f"Error: key '{key}' not found in NPZ; available keys: {list(data.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        return data[key]

    return _get(probs_key), _get(logits_key), _get(labels_key)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate probability calibration from saved model outputs."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--probs",
        help="Path to .npy/.npz with predicted probabilities (shape N x C).",
    )
    source.add_argument(
        "--logits",
        help="Path to .npy/.npz with raw logits (shape N x C). Required for temperature scaling.",
    )
    source.add_argument(
        "--npz",
        help="Path to .npz file containing probs/logits/labels.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to .npy/.npz with true labels (shape N,).",
    )

    parser.add_argument(
        "--npz-probs-key",
        default="y_prob",
        help="Key for probabilities in --npz file (default: y_prob).",
    )
    parser.add_argument(
        "--npz-logits-key",
        default="y_logits",
        help="Key for logits in --npz file (default: y_logits).",
    )
    parser.add_argument(
        "--npz-labels-key",
        default="y_true",
        help="Key for labels in --npz file (default: y_true).",
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.0,
        help="Fraction of data to use for fitting temperature (0.0 = no scaling fit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibration/eval split.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help=f"Number of equal-width bins for ECE (default: {DEFAULT_N_BINS}).",
    )
    parser.add_argument(
        "--output-dir",
        default="calibration_output",
        help="Directory for saved outputs (default: calibration_output).",
    )
    parser.add_argument(
        "--reliability-diagram-before",
        default="reliability_before.png",
        help="Output path for pre-calibration reliability diagram.",
    )
    parser.add_argument(
        "--reliability-diagram-after",
        default="reliability_after.png",
        help="Output path for post-calibration reliability diagram.",
    )
    parser.add_argument(
        "--temperature-out",
        default="temperature.json",
        help="Path to save fitted temperature JSON (default: temperature.json).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of classes (default: {NUM_CLASSES}).",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log calibration results to MLflow.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="lumoculis",
        help="MLflow experiment name (default: lumoculis).",
    )

    return parser.parse_args(argv)


def _split_calibration(
    n: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into calibration and evaluation sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_cal = int(n * calibration_fraction)
    cal_idx = indices[:n_cal]
    eval_idx = indices[n_cal:]
    return cal_idx, eval_idx


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Load probs, logits, and labels from CLI arguments."""
    labels = _load_array(args.labels, "labels")
    labels = np.asarray(labels, dtype=np.int64)

    probs: np.ndarray | None = None
    logits: np.ndarray | None = None

    if args.npz is not None:
        probs_arr, logits_arr, labels_arr = _load_npz(
            args.npz,
            args.npz_probs_key,
            args.npz_logits_key,
            args.npz_labels_key,
        )
        if probs_arr is not None:
            probs = probs_arr
        if logits_arr is not None:
            logits = logits_arr
        if labels_arr is not None:
            labels = labels_arr
    else:
        if args.probs is not None:
            probs = _load_array(args.probs, "probs")
        if args.logits is not None:
            logits = _load_array(args.logits, "logits")

    return probs, logits, labels


def _validate_arrays(
    probs: np.ndarray | None,
    logits: np.ndarray | None,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate and convert probs/logits arrays. Returns validated arrays."""
    for arr, name in ((probs, "probs"), (logits, "logits")):
        if arr is None:
            continue
        converted = np.asarray(arr, dtype=np.float64)
        if converted.ndim != EXPECTED_PROB_NDIM or converted.shape[1] != num_classes:
            print(
                f"Error: {name} must have shape (N, {num_classes}), got {converted.shape}",
                file=sys.stderr,
            )
            sys.exit(1)
        if converted.shape[0] != labels.shape[0]:
            print(
                f"Error: {name} ({converted.shape[0]} samples) and labels "
                f"({labels.shape[0]} samples) have different counts",
                file=sys.stderr,
            )
            sys.exit(1)
    return probs, logits


def _derive_probs_from_logits(
    logits: np.ndarray | None,
    probs: np.ndarray | None,
) -> np.ndarray | None:
    """Derive softmax probabilities from logits if probs are not provided."""
    if probs is None and logits is not None:
        logits_arr = np.asarray(logits, dtype=np.float64)
        shifted = logits_arr - logits_arr.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs


def _run_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    config: TemperatureScalingConfig,
    paths: CalibrationOutputPaths,
) -> tuple[float, bool, dict | None]:
    """Fit temperature on calibration split, apply to eval set, save outputs.

    Returns (temperature, fitted, metrics_after).
    """
    cal_idx, eval_idx = _split_calibration(len(labels), config.calibration_fraction, config.seed)

    ts = TemperatureScaling(num_classes=num_classes)
    ts.fit(logits[cal_idx], labels[cal_idx])
    temperature = ts.temperature

    calibrated_probs = ts.apply(logits[eval_idx])
    eval_labels = labels[eval_idx]

    metrics_after = _compute_metrics(calibrated_probs, eval_labels, num_classes, config.n_bins)
    _print_metrics(
        "Calibration metrics (after temperature scaling)",
        metrics_after,
        temperature,
    )

    # Reliability diagram after.
    after_diagram = paths.output_dir / paths.diagram_after
    plot_reliability_diagram(
        metrics_after["ece"],
        after_diagram,
        title="Reliability diagram (after calibration)",
    )
    print(f"  Saved post-calibration diagram: {after_diagram}")

    # Save temperature.
    temp_path = paths.output_dir / paths.temperature
    ts.save(temp_path)
    print(f"  Saved temperature: {temp_path}")

    return temperature, True, metrics_after


def _compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bins: int,
) -> dict:
    """Compute calibration metrics for probs vs labels."""
    return {
        "brier": multiclass_brier_score(probs, labels, num_classes=num_classes),
        "ece": expected_calibration_error(probs, labels, num_classes=num_classes, n_bins=n_bins),
        "brier_binary": binary_brier_score(probs, labels),
        "ece_binary": binary_ece(probs, labels, n_bins=n_bins),
    }


def _print_metrics(
    title: str,
    metrics: dict,
    temperature: float | None = None,
) -> None:
    """Print calibration metrics to stdout."""
    print(f"=== {title} ===")
    if temperature is not None:
        print(f"  Temperature:            {temperature:.6f}")
    print(f"  Multiclass Brier score: {metrics['brier']:.6f}")
    print(f"  Multiclass ECE:         {metrics['ece']['ece']:.6f}")
    print(f"  Binary Brier score:     {metrics['brier_binary']:.6f}")
    print(f"  Binary ECE:             {metrics['ece_binary']['ece']:.6f}")


def _log_mlflow(
    experiment: str,
    metrics_before: dict,
    metrics_after: dict | None,
    temperature: float | None,
    fitted: bool,
) -> None:
    """Log calibration results to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="calibration_eval"):
        mlflow.log_params({
            "calibration/ece_before": metrics_before["ece"]["ece"],
            "calibration/brier_before": metrics_before["brier"],
            "calibration/brier_binary_before": metrics_before["brier_binary"],
            "calibration/ece_binary_before": metrics_before["ece_binary"]["ece"],
        })
        if metrics_after is not None:
            mlflow.log_params({
                "calibration/ece_after": metrics_after["ece"]["ece"],
                "calibration/brier_after": metrics_after["brier"],
                "calibration/brier_binary_after": metrics_after["brier_binary"],
                "calibration/ece_binary_after": metrics_after["ece_binary"]["ece"],
                "calibration/temperature": temperature if temperature else 1.0,
                "calibration/temperature_fitted": int(fitted),
            })
        mlflow.log_param("calibration/n_bins", DEFAULT_N_BINS)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    probs, logits, labels = _load_inputs(args)
    num_classes = args.num_classes

    probs, logits = _validate_arrays(probs, logits, labels, num_classes)
    probs = _derive_probs_from_logits(logits, probs)

    if probs is None:
        print("Error: must provide either --probs or --logits.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Metrics before calibration ----
    metrics_before = _compute_metrics(probs, labels, num_classes, args.n_bins)
    _print_metrics("Calibration metrics (before temperature scaling)", metrics_before)

    # Reliability diagram before.
    before_diagram = output_dir / args.reliability_diagram_before
    plot_reliability_diagram(
        metrics_before["ece"],
        before_diagram,
        title="Reliability diagram (before calibration)",
    )
    print(f"  Saved pre-calibration diagram: {before_diagram}")

    # ---- Temperature scaling ----
    temperature: float | None = None
    fitted = False
    metrics_after: dict | None = None

    if args.calibration_fraction > 0.0:
        if logits is None:
            print(
                "\nWarning: --calibration-fraction > 0 requires --logits; "
                "skipping temperature scaling.",
                file=sys.stderr,
            )
        else:
            temperature, fitted, metrics_after = _run_temperature_scaling(
                logits,
                labels,
                num_classes,
                TemperatureScalingConfig(
                    calibration_fraction=args.calibration_fraction,
                    seed=args.seed,
                    n_bins=args.n_bins,
                ),
                CalibrationOutputPaths(
                    output_dir=output_dir,
                    diagram_after=args.reliability_diagram_after,
                    temperature=args.temperature_out,
                ),
            )

    # ---- MLflow ----
    if args.mlflow:
        _log_mlflow(
            args.mlflow_experiment,
            metrics_before,
            metrics_after,
            temperature,
            fitted,
        )
        print("  Logged to MLflow.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
