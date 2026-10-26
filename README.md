# Lumoculis

**Lumoculis: AI-assisted diabetic retinopathy screening support.**

Classifies retinal fundus images into diabetic retinopathy (DR) grades 0-4
(APTOS 2019) and flags "referable DR" (grade >= 2).

## Safety disclaimer

This is a screening SUPPORT tool for camps and rural clinics. It is NOT a
diagnostic tool. Do not use the words "diagnosis", "clinical-grade", "FDA", or
any accuracy claim that implies clinical validity. Results are not clinically
validated. Always refer flagged cases to an eye doctor.

## Setup

```bash
# 1. Install uv (https://github.com/astral-sh/uv) if you don't have it.
# 3. Create the environment and install dependencies.
uv sync --dev

# 4. Copy the env example and add your Kaggle credentials.
cp .env.example .env
# Edit .env with KAGGLE_USERNAME and KAGGLE_KEY.

# 5. Download APTOS 2019 (check the dataset license and terms first).
uv run python scripts/download_data.py

# 6. Preprocess images (crops black borders, resizes, Ben Graham contrast).
uv run python scripts/preprocess_images.py

# 7. Train the baseline (smoke test: only 2 batches, no real data needed).
uv run python -m lumoculis.train --config configs/baseline.yaml --max-batches 2 --epochs 1 --no-pretrained --no-save
```

## Layout

- `configs/` — experiment YAML configs
- `src/lumoculis/` — package: config, data, preprocess, model, metrics, train, evaluate, explain, quality, calibrate, summarize, utils
- `api/` — FastAPI serving app
- `app/` — Gradio demo
- `scripts/` — download, preprocess, ONNX export, external evaluation
- `tests/` — pytest suite
- `notebooks/` — EDA only
- `docker/` — CPU-only API Dockerfile
- `.github/workflows/ci.yml` — ruff + pytest on push

## Limitations

See `MODEL_CARD.md` for intended use, out-of-scope use, data, metrics, bias,
and limitations. APTOS has no patient IDs, so patient-level train/val splitting
cannot be guaranteed.
