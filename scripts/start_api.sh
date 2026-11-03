#!/usr/bin/env bash
set -euo pipefail

export LUMOCULIS_CHECKPOINT="$(python -m lumoculis.s3_checkpoint)"
exec uvicorn lumoculis.api.main:app --host 0.0.0.0 --port "${PORT:-10000}"
