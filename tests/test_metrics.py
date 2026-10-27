"""Tests for src.lumoculis.metrics on tiny hand-made examples with known answers."""

from __future__ import annotations

import matplotlib.figure
import numpy as np

from lumoculis import metrics


def test_quadratic_weighted_kappa_perfect() -> None:
    y = np.array([0, 1, 2, 3, 4])
    assert metrics.quadratic_weighted_kappa(y, y) == 1.0


def test_quadratic_weighted_kappa_all_worst() -> None:
    """Maximally wrong predictions should give the lowest kappa."""
    y_true = np.array([0, 4])
    y_pred = np.array([4, 0])
    kappa = metrics.quadratic_weighted_kappa(y_true, y_pred)
    assert kappa < 0.0


def test_referable_auroc_perfect_separation() -> None:
    y_true = np.array([0, 1, 2, 3, 4])
    # grades 0-1 -> referable prob 0; grades 2-4 -> referable prob 1
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    assert metrics.referable_auroc(y_true, probs) == 1.0


def test_referable_auroc_undefined_returns_nan() -> None:
    y_true = np.array([0, 0, 1, 1])  # no referable cases
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert np.isnan(metrics.referable_auroc(y_true, probs))


def test_referable_sensitivity_specificity_known_counts() -> None:
    y_true = np.array([0, 1, 2, 3, 4])  # non-ref, non-ref, ref, ref, ref
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # ref prob 0 -> pred non-ref (TN)
        [1.0, 0.0, 0.0, 0.0, 0.0],  # ref prob 0 -> pred non-ref (TN)
        [0.0, 0.0, 1.0, 0.0, 0.0],  # ref prob 1 -> pred ref (TP)
        [0.0, 0.0, 1.0, 0.0, 0.0],  # ref prob 1 -> pred ref (TP)
        [0.0, 0.0, 1.0, 0.0, 0.0],  # ref prob 1 -> pred ref (TP)
    ])
    sens, spec = metrics.referable_sensitivity_specificity(y_true, probs, threshold=0.5)
    assert sens == 1.0  # 3/3 referable caught
    assert spec == 1.0  # 2/2 non-referable correctly rejected


def test_referable_sensitivity_specificity_threshold_misses() -> None:
    y_true = np.array([0, 2])
    probs = np.array([
        [0.5, 0.0, 0.5, 0.0, 0.0],  # ref prob 0.5 -> at threshold 0.5 -> pred ref (FP)
        [0.5, 0.0, 0.5, 0.0, 0.0],  # ref prob 0.5 -> pred ref (TP)
    ])
    sens, spec = metrics.referable_sensitivity_specificity(y_true, probs, threshold=0.5)
    assert sens == 1.0
    assert spec == 0.0  # the non-referable case was flagged


def test_confusion_matrix_shape_and_counts() -> None:
    num_classes = 5
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    cm = metrics.compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (num_classes, num_classes)
    # true 0: pred 0 once, pred 1 once
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1
    # true 2: pred 2 once, pred 0 once
    assert cm[2, 2] == 1
    assert cm[2, 0] == 1
    assert int(cm.sum()) == num_classes


def test_render_confusion_matrix_returns_figure() -> None:
    cm = np.array([[5, 0], [0, 5]])
    fig = metrics.render_confusion_matrix(cm, class_names=["non-ref", "ref"])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


def test_bootstrap_qwk_ci_perfect() -> None:
    y_true = np.array([0, 1, 2, 3, 4] * 20)
    y_pred = y_true.copy()
    point, lo, hi = metrics.bootstrap_qwk_ci(y_true, y_pred, n_bootstrap=50, seed=42)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_qwk_ci_worst() -> None:
    y_true = np.array([0, 4])
    y_pred = np.array([4, 0])
    point, lo, hi = metrics.bootstrap_qwk_ci(y_true, y_pred, n_bootstrap=50, seed=42)
    assert point < 0.0
    assert lo <= point <= hi


def test_bootstrap_referable_auroc_ci_perfect() -> None:
    # Non-referable (grades 0,1) -> prob 0; referable (grades 2,3,4) -> prob 1
    y_true = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10)
    probs = np.zeros((50, 5))
    probs[0:20, 0] = 1.0  # grades 0-1 -> all mass on class 0
    probs[20:50, 2] = 1.0  # grades 2-4 -> all mass on class 2
    point, lo, hi = metrics.bootstrap_referable_auroc_ci(y_true, probs, n_bootstrap=50, seed=42)
    assert point == 1.0


def test_bootstrap_referable_sens_spec_ci() -> None:
    # Non-referable (grades 0,1) -> prob 0; referable (grades 2,3,4) -> prob 1
    y_true = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10)
    probs = np.zeros((50, 5))
    probs[0:20, 0] = 1.0  # non-referable
    probs[20:50, 2] = 1.0  # referable
    (s_point, s_lo, s_hi), (sp_point, sp_lo, sp_hi) = metrics.bootstrap_referable_sens_spec_ci(
        y_true, probs, threshold=0.5, n_bootstrap=50, seed=42
    )
    assert s_point == 1.0
    assert sp_point == 1.0
    assert s_lo <= s_point <= s_hi
    assert sp_lo <= sp_point <= sp_hi


def test_bootstrap_per_class_sensitivity_ci() -> None:
    y_true = np.array([0, 1, 2, 3, 4] * 10)
    y_pred = y_true.copy()
    results = metrics.bootstrap_per_class_sensitivity_ci(
        y_true, y_pred, num_classes=metrics.NUM_CLASSES, n_bootstrap=50, seed=42
    )
    assert len(results) == metrics.NUM_CLASSES
    for point, lo, hi in results:
        assert point == 1.0
        assert lo == 1.0
        assert hi == 1.0
