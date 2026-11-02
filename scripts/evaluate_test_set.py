import argparse
import pathlib

from lumoculis.config import Config
from lumoculis.eval_test import run_test_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="reports/test_evaluations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    csv_path = pathlib.Path(config.data_dir) / "train.csv"

    result = run_test_evaluation(
        csv_path,
        config.processed_dir,
        config,
        args.checkpoint,
        args.output_root,
        val_size=args.val_size,
        test_size=args.test_size,
        device=args.device,
    )
    print(result["run_dir"])


if __name__ == "__main__":
    main()
