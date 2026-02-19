#!/bin/bash
set -e

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    python download_model.py --output api/src/models/v1_0
fi

# Pre-populate HF cache at runtime if HF_MODEL_REPO is set (e.g. when not baked in at build)
if [ -n "$HF_MODEL_REPO" ]; then
    python download_hf_model.py || true
fi

exec uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug