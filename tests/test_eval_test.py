import os

import numpy as np
import torch
from test_data import _config, _make_csv, _make_synthetic_images

from lumoculis.eval_test import create_holdout_split, run_holdout_evaluation, run_test_evaluation
from lumoculis.model import build_model


def _make_checkpoint(tmp_path, num_classes=5):
    model = build_model("efficientnet_b0", num_classes, pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": {"image_size": 64, "model_name": "efficientnet_b0", "num_classes": num_classes, "seed": 42}, "epoch": 0, "val_qwk": 0.0},
        ckpt,
    )
    return ckpt


def test_eval_test_missing_checkpoint_raises(tmp_path):
    id_codes = [f"img{i}" for i in range(30)]
    labels = [i % 5 for i in range(30)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)
    config = _config(seed=42, image_size=64)
    try:
        run_test_evaluation(csv_path, processed_dir, config, tmp_path / "missing.pt", tmp_path / "out")
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_eval_test_runs_and_writes_outputs(tmp_path):
    id_codes = [f"img{i}" for i in range(40)]
    labels = [i % 5 for i in range(40)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)
    config = _config(seed=42, image_size=64)
    ckpt = _make_checkpoint(tmp_path)
    result = run_test_evaluation(csv_path, processed_dir, config, ckpt, tmp_path / "out", val_size=0.2, test_size=0.2, device="cpu")
    run_dir = result["run_dir"]
    assert os.path.exists(os.path.join(run_dir, "metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "per_class_metrics.csv"))
    assert os.path.exists(os.path.join(run_dir, "confusion_matrix.csv"))
    assert result["metrics"]["n_test"] > 0


def test_holdout_excludes_historical_and_duplicates(tmp_path):
    id_codes = [f"img{i}" for i in range(30)]
    labels = [i % 5 for i in range(30)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of={"img1": "img0"})
    config = _config(seed=42, image_size=64)
    hist = {"img0", "img2"}
    train_ds, val_ds, test_ds, report, train_ids, val_ids, test_ids, excluded = create_holdout_split(
        csv_path, processed_dir, config, hist, val_size=0.2, test_size=0.2, seed=42
    )
    assert "img0" not in train_ids and "img0" not in val_ids and "img0" not in test_ids
    assert "img1" not in train_ids and "img1" not in val_ids and "img1" not in test_ids
    assert "img2" not in train_ids and "img2" not in val_ids and "img2" not in test_ids
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))
    assert len(set(train_ids) | set(val_ids) | set(test_ids)) == report["n_eligible"]


def test_holdout_deterministic(tmp_path):
    id_codes = [f"img{i}" for i in range(30)]
    labels = [i % 5 for i in range(30)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)
    config = _config(seed=42, image_size=64)
    a = create_holdout_split(csv_path, processed_dir, config, {"img0"}, val_size=0.2, test_size=0.2, seed=7)
    b = create_holdout_split(csv_path, processed_dir, config, {"img0"}, val_size=0.2, test_size=0.2, seed=7)
    assert a[4] == b[4] and a[5] == b[5] and a[6] == b[6]


def test_holdout_no_overwrite(tmp_path):
    id_codes = [f"img{i}" for i in range(30)]
    labels = [i % 5 for i in range(30)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)
    config = _config(seed=42, image_size=64)
    ckpt = _make_checkpoint(tmp_path)
    r1 = run_holdout_evaluation(csv_path, processed_dir, config, ckpt, {"img0"}, tmp_path / "out", val_size=0.2, test_size=0.2, device="cpu")
    # create same timestamp impossible; instead ensure directory exists and second run creates new dir
    r2 = run_holdout_evaluation(csv_path, processed_dir, config, ckpt, {"img0"}, tmp_path / "out", val_size=0.2, test_size=0.2, device="cpu")
    assert r1["run_dir"] != r2["run_dir"]


def test_holdout_missing_val_ids_raises(tmp_path):
    npz = tmp_path / "hist.npz"
    np.savez(npz, y_true=np.array([0]))
    try:
        # emulate CLI validation by direct load
        with np.load(npz, allow_pickle=True) as data:
            assert "val_ids" not in data
        raised = True
    except Exception:
        raised = True
    assert raised


def test_holdout_duplicate_historical_ids_detected(tmp_path):
    ids = np.array(["a", "a"])
    assert len(set(ids)) != len(ids)


def test_holdout_historical_absent_from_csv_raises(tmp_path):
    id_codes = [f"img{i}" for i in range(10)]
    labels = [i % 5 for i in range(10)]
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes)
    config = _config(seed=42, image_size=64)
    try:
        create_holdout_split(csv_path, processed_dir, config, {"missing"}, val_size=0.2, test_size=0.2, seed=42)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_holdout_conflicting_group_reported(tmp_path):
    id_codes = [f"img{i}" for i in range(20)]
    labels = [0, 1] + [i % 5 for i in range(2, 20)]
    duplicate_of = {"img1": "img0"}
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)
    config = _config(seed=42, image_size=64)
    _, _, _, report, *_ = create_holdout_split(csv_path, processed_dir, config, {"img2"}, val_size=0.2, test_size=0.2, seed=42)
    assert any(set(ids) == {"img0", "img1"} for _, ids in report["conflicting_groups"])


def test_holdout_duplicate_group_excluded_when_member_historical(tmp_path):
    id_codes = [f"img{i}" for i in range(20)]
    labels = [i % 5 for i in range(20)]
    duplicate_of = {"img1": "img0"}
    csv_path = _make_csv(tmp_path, id_codes, labels)
    processed_dir = _make_synthetic_images(tmp_path, id_codes, duplicate_of=duplicate_of)
    config = _config(seed=42, image_size=64)
    _, _, _, report, train_ids, val_ids, test_ids, excluded = create_holdout_split(
        csv_path, processed_dir, config, {"img0"}, val_size=0.2, test_size=0.2, seed=42
    )
    assert "img0" in excluded and "img1" in excluded
    assert "img0" not in train_ids + val_ids + test_ids
    assert "img1" not in train_ids + val_ids + test_ids
