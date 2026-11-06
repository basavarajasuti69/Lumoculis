"""Secure retrieval of a Lumoculis checkpoint from Amazon S3.

This module is intended to run before Uvicorn starts. It downloads the
configured checkpoint to a local path and prints that path to stdout so a
shell wrapper can export it as ``LUMOCULIS_CHECKPOINT``.

No AWS credentials are ever logged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import boto3

DEFAULT_CHECKPOINT_PATH = Path("/tmp/lumoculis/best.pt")  # noqa: S108
DEFAULT_ONNX_PATH = Path("/tmp/lumoculis/best.onnx")  # noqa: S108
DEFAULT_ONNX_KEY = "models/best.onnx"
DEFAULT_BUCKET = "lumoculis-project-basavaraj-2026"
DEFAULT_KEY = "models/best.pt"
DEFAULT_REGION = "ap-southeast-2"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _region() -> str:
    return (
        os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or DEFAULT_REGION
    )


def download_checkpoint() -> Path:
    """Download the checkpoint from S3 if it is not already present.

    Returns the local path to a non-empty checkpoint file.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if not access_key or not secret_key:
        raise RuntimeError("AWS credentials are not configured")

    backend = os.environ.get("LUMOCULIS_BACKEND", "pytorch").strip().lower()
    if backend == "onnx":
        key = os.environ.get("LUMOCULIS_S3_ONNX_KEY", "").strip() or DEFAULT_ONNX_KEY
        destination = Path(
            os.environ.get("LUMOCULIS_ONNX_MODEL", "").strip() or str(DEFAULT_ONNX_PATH)
        )
    else:
        key = os.environ.get("LUMOCULIS_S3_KEY", "").strip() or DEFAULT_KEY
        destination = Path(
            os.environ.get("LUMOCULIS_CHECKPOINT", "").strip() or str(DEFAULT_CHECKPOINT_PATH)
        )

    bucket = os.environ.get("LUMOCULIS_S3_BUCKET", "").strip() or DEFAULT_BUCKET

    if destination.exists() and destination.is_file() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        client: Any = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=_region(),
        )
        client.download_file(bucket, key, str(destination))
    except Exception as exc:  # keep startup errors concise and credential-free
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"Failed to download checkpoint from s3://{bucket}/{key}: {exc}") from exc

    if not destination.exists() or destination.stat().st_size == 0:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError("Downloaded checkpoint is missing or empty")

    if backend == "onnx":
        os.environ["LUMOCULIS_ONNX_MODEL"] = str(destination)
    else:
        os.environ["LUMOCULIS_CHECKPOINT"] = str(destination)
    return destination


def main() -> int:
    try:
        path = download_checkpoint()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
