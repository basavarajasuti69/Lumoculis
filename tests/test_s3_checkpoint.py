"""Tests for secure S3 checkpoint retrieval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lumoculis import s3_checkpoint


def _make_fake_client(download_side_effect=None):
    client = MagicMock()
    if download_side_effect is not None:
        client.download_file.side_effect = download_side_effect
    else:
        def _download(bucket, key, dest):
            Path(dest).write_bytes(b"checkpoint-bytes")
        client.download_file.side_effect = _download
    return client


def test_download_skips_existing_valid_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"existing")
    monkeypatch.setenv("LUMOCULIS_CHECKPOINT", str(dest))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    fake_client = _make_fake_client()
    monkeypatch.setattr(s3_checkpoint.boto3, "client", lambda *a, **k: fake_client)

    result = s3_checkpoint.download_checkpoint()

    assert result == dest
    fake_client.download_file.assert_not_called()


def test_download_downloads_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"
    monkeypatch.setenv("LUMOCULIS_CHECKPOINT", str(dest))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("LUMOCULIS_S3_BUCKET", "bucket")
    monkeypatch.setenv("LUMOCULIS_S3_KEY", "models/best.pt")
    fake_client = _make_fake_client()
    monkeypatch.setattr(s3_checkpoint.boto3, "client", lambda *a, **k: fake_client)

    result = s3_checkpoint.download_checkpoint()

    assert result == dest
    assert dest.read_bytes() == b"checkpoint-bytes"
    fake_client.download_file.assert_called_once_with("bucket", "models/best.pt", str(dest))
    assert s3_checkpoint.os.environ["LUMOCULIS_CHECKPOINT"] == str(dest)


def test_download_fails_without_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"
    monkeypatch.setenv("LUMOCULIS_CHECKPOINT", str(dest))
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    with pytest.raises(RuntimeError, match="AWS credentials"):
        s3_checkpoint.download_checkpoint()


def test_download_fails_on_s3_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"
    monkeypatch.setenv("LUMOCULIS_CHECKPOINT", str(dest))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    fake_client = _make_fake_client(download_side_effect=RuntimeError("boom"))
    monkeypatch.setattr(s3_checkpoint.boto3, "client", lambda *a, **k: fake_client)

    with pytest.raises(RuntimeError, match="Failed to download"):
        s3_checkpoint.download_checkpoint()
    assert not dest.exists()


def test_download_fails_on_empty_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest = tmp_path / "best.pt"

    def _download(bucket, key, path):
        Path(path).write_bytes(b"")

    monkeypatch.setenv("LUMOCULIS_CHECKPOINT", str(dest))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    fake_client = MagicMock()
    fake_client.download_file.side_effect = _download
    monkeypatch.setattr(s3_checkpoint.boto3, "client", lambda *a, **k: fake_client)

    with pytest.raises(RuntimeError, match="missing or empty"):
        s3_checkpoint.download_checkpoint()


def test_region_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    assert s3_checkpoint._region() == "us-east-1"


def test_main_prints_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"existing")
    monkeypatch.setenv("LUMOCULIS_CHECKPOINT", str(dest))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert s3_checkpoint.main() == 0
    assert capsys.readouterr().out.strip() == str(dest)
