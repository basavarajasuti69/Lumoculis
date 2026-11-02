import argparse
from pathlib import Path

import numpy as np

from lumoculis.config import Config
from lumoculis.eval_test import run_holdout_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--historical-npz", required=True)
    parser.add_argument("--output-root", default="reports/test_evaluations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    checkpoint_path = Path(args.checkpoint)
    npz_path = Path(args.historical_npz)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Historical NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "val_ids" not in data:
            raise ValueError("Historical NPZ is missing 'val_ids'.")
        historical_ids = [str(x) for x in data["val_ids"]]
    if not historical_ids:
        raise ValueError("Historical 'val_ids' array is empty.")
    if len(set(historical_ids)) != len(historical_ids):
        raise ValueError("Historical 'val_ids' contains duplicates.")

    csv_path = Path(config.data_dir) / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    import pandas as pd
    df = pd.read_csv(csv_path)
    available_ids = set(df["id_code"].astype(str))
    missing = [x for x in historical_ids if x not in available_ids]
    if missing:
        raise ValueError(f"Historical IDs not found in dataset: {missing[:10]}")

    result = run_holdout_evaluation(
        csv_path,
        config.processed_dir,
        config,
        checkpoint_path,
        set(historical_ids),
        args.output_root,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        device=args.device,
    )
    print(result["run_dir"])
    print(result["metrics"]["n_test"])


if __name__ == "__main__":
    main()
