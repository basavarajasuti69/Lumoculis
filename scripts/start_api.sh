#!/usr/bin/env bash
set -euo pipefail

BACKEND="${LUMOCULIS_BACKEND:-pytorch}"

if [ "$BACKEND" = "onnx" ]; then
    export LUMOCULIS_ONNX_MODEL="$(python -m lumoculis.s3_checkpoint)"
else
    export LUMOCULIS_CHECKPOINT="$(python -m lumoculis.s3_checkpoint)"
fi

exec uvicorn lumoculis.api.main:app --host 0.0.0.0 --port "${PORT:-10000}"