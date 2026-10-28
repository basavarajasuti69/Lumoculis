# Lumoculis

**Lumoculis: AI-assisted diabetic retinopathy screening support.**

Classifies retinal fundus images into diabetic retinopathy (DR) grades 0-4
(APTOS 2019) and flags "referable DR" (grade >= 2).

## Safety disclaimer

This is a screening SUPPORT tool for camps and rural clinics. It is NOT a
diagnostic tool. Do not use the words "diagnosis", "clinical-grade", "FDA", or
any accuracy claim that implies clinical validity. Results are not clinically
validated. Always refer flagged cases to an eye doctor.

Lumoculis is a research prototype, not a medical device. Its predictions are
not a diagnosis or a substitute for qualified clinical assessment. Image-quality
thresholds are heuristic and not clinically validated. Grad-CAM highlights
model-relevant regions but does not prove disease location or causality.
Calibration improves probability reliability only to the extent supported by
appropriate held-out data; it does not establish clinical reliability. APTOS
2019 does not provide patient IDs in the dataset used here, and grouping exact
duplicate images does not guarantee patient-level independence. The system has
not been clinically validated.

## Setup

```bash
# 1. Install uv (https://github.com/astral-sh/uv) if you don't have it.
# 2. Create the environment and install dependencies.
uv sync --dev

# 3. Copy the env example and add your Kaggle credentials.
cp .env.example .env
# Edit .env with KAGGLE_USERNAME and KAGGLE_KEY.

# 4. Download APTOS 2019 (check the dataset license and terms first).
uv run python scripts/download_data.py

# 5. Preprocess images (crops black borders, resizes, Ben Graham contrast).
uv run python scripts/preprocess_images.py

# 6. Inspect the dataset (class balance, duplicate groups, weight ratios).
uv run python scripts/inspect_data.py

# 7. Train the baseline (smoke test: only 2 batches, no real data needed).
uv run python -m lumoculis.train --config configs/baseline.yaml --max-batches 2 --epochs 1 --no-pretrained --no-save

# 8. Full baseline run (real data, pretrained weights).
uv run python -m lumoculis.train --config configs/baseline.yaml

# Open the MLflow UI to view runs.
uv run mlflow ui --backend-store-uri ./mlruns
```

## Layout

- `configs/` — experiment YAML configs
- `src/lumoculis/` — package: config, data, preprocess, model, metrics, train,
  explain, quality, calibration, inference
- `scripts/` — download, preprocess, ONNX export, external evaluation
- `tests/` — pytest suite
- `docker/` — CPU-only API Dockerfile
- `.github/workflows/ci.yml` — ruff + pytest on push

## Commands

### Training

```bash
# Smoke test (2 batches, no pretrained weights, no saving).
uv run python -m lumoculis.train --config configs/baseline.yaml \
  --max-batches 2 --epochs 1 --no-pretrained --no-save

# Full training run.
uv run python -m lumoculis.train --config configs/baseline.yaml
```

### Image-quality assessment (`lumoculis-quality`)

Assess a single image:

```bash
lumoculis-quality --image data/processed/001.png --output report.json
```

Assess a directory of images:

```bash
lumoculis-quality --dir data/processed/ --recursive --output report.json
```

With custom thresholds:

```bash
lumoculis-quality --image data/processed/001.png \
  --min-width 300 --min-height 300 --blur-min 50.0
```

Treat sub-min resolution as a warning instead of a hard failure:

```bash
lumoculis-quality --image data/processed/001.png --no-resolution-hard-failure
```

### Calibration evaluation (`lumoculis-calibrate`)

Evaluate calibration from saved probabilities:

```bash
lumoculis-calibrate --probs data/val_probs.npy --labels data/val_labels.npy
```

Evaluate from raw logits with temperature scaling fit on a calibration subset:

```bash
lumoculis-calibrate --logits data/val_logits.npy --labels data/val_labels.npy \
  --calibration-fraction 0.3 --output-dir ./calibration_output
```

### Inference (`lumoculis-infer`)

Run DR grade prediction on a single retinal image using a trained checkpoint:

```bash
lumoculis-infer \
  --image data/processed/001.png \
  --checkpoint checkpoints/run_name/best.pt \
  --output report.json
```

With image-quality gating (blocks inference when quality fails):

```bash
lumoculis-infer \
  --image data/processed/001.png \
  --checkpoint checkpoints/run_name/best.pt \
  --run-quality-gate --quality-required \
  --output report.json
```

With temperature-scaling calibration (loads a previously fitted temperature;
does not fit during inference):

```bash
lumoculis-infer \
  --image data/processed/001.png \
  --checkpoint checkpoints/run_name/best.pt \
  --temperature calibration/temperature.json \
  --output report.json
```

With Grad-CAM visualization:

```bash
lumoculis-infer \
  --image data/processed/001.png \
  --checkpoint checkpoints/run_name/best.pt \
  --gradcam --gradcam-output gradcam.png \
  --output report.json
```

### Grad-CAM (Python API)

```python
from lumoculis.explain import GradCAM, save_gradcam_visualization
from lumoculis.model import build_model
import torch

model = build_model("efficientnet_b0", 5, pretrained=False).eval()
# ... load checkpoint state dict ...

with GradCAM(model) as gradcam:
    heatmap, target_class = gradcam.compute_heatmap(input_tensor)
    save_gradcam_visualization(
        original_image=rgb_image,
        heatmap=heatmap,
        output_path="gradcam.png",
        predicted_class=pred_class,
        target_class=target_class,
    )
```

Grad-CAM supports classification models only (not ordinal-regression models).

## Expected outputs

- Checkpoints: `checkpoints/<run_name>/best.pt` (contains model state dict,
  config, epoch, val QWK)
- Validation predictions: `checkpoints/<run_name>/val_preds.npz`
- MLflow runs: `mlruns/`
- Quality reports: JSON with per-image status (pass/warning/fail) and measured
  values
- Calibration outputs: `temperature.json` and reliability diagram PNGs
- Inference reports: JSON with raw and calibrated probabilities, quality
  assessment, and optional Grad-CAM output

## Reproducibility

- Training is seeded by `config.seed` (default 42).
- The train/val split is group-aware by SHA-256 duplicate hash, stratified by
  representative diagnosis label.
- Device selection: MPS (Apple Silicon) if available, otherwise CPU.

## Known limitations

- No pretrained weights are bundled; the first training run downloads them from
  timm (requires network access).
- APTOS 2019 provides no patient IDs, so patient-level train/val splitting
  cannot be guaranteed even with duplicate-image grouping.
- Image-quality thresholds are heuristic and not clinically validated.
- Grad-CAM visualizes model sensitivity, not medical causality.
- Calibration may improve probability reliability but does not necessarily
  improve classification accuracy or clinical safety.
- The system has not been clinically validated. Results are for research and
  screening support only.

See `MODEL_CARD.md` for intended use, out-of-scope use, data, metrics, bias,
and limitations.

### Train/val split trade-offs

The split is **group-aware**: exact duplicate images (same SHA-256 of the
processed PNG) are always kept in the same split (train or val) to prevent
leakage. This slightly reduces the effective number of independent groups
available for stratification, so class proportions in each split are
approximate rather than exact. Grouping exact duplicates does **not** identify
all images from the same patient, so patient-level separation is still not
guaranteed. Use `scripts/inspect_data.py` to review duplicate groups,
cross-split groups, and conflicting labels before training.

### --no-save flag

`uv run python -m lumoculis.train --no-save` skips writing checkpoints,
validation predictions, and MLflow artifacts while still computing and
printing metrics. Useful for smoke tests and CI.

## Streamlit web app (local and EC2)

Lumoculis includes a Streamlit app (`app.py`) for a low-traffic public research demo. It reuses the existing inference, quality, calibration, and Grad-CAM code.

### Run locally

```bash
# From repo root
export LUMOCULIS_CHECKPOINT=/path/to/checkpoint.pt  # required for inference
# Optional
export LUMOCULIS_CALIBRATION=/path/to/temperature.json
export LUMOCULIS_ENABLE_QUALITY_GATE=1
export LUMOCULIS_QUALITY_REQUIRED=0
export LUMOCULIS_DEVICE=cpu
export LUMOCULIS_MAX_UPLOAD_MB=10

uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. The app accepts PNG/JPEG, shows the image, offers optional quality assessment, displays raw and calibrated probabilities (when calibration configured), optional Grad-CAM, JSON report download, and a research-use disclaimer. It does not claim diagnoses. Uploads are not retained permanently by the app; avoid logging sensitive image data.

Notes:
- CPU-only inference. Model is loaded once via caching.
- Do not download pretrained weights at startup. Use a local checkpoint.
- Uploads limited by `LUMOCULIS_MAX_UPLOAD_MB` (default 10 MB).
- Clear errors shown if checkpoint missing/incompatible or image invalid.


## EC2 deployment (Free Tier, budget ₹0 / cost-conscious)

**Do not create AWS resources or share credentials.** Follow these steps manually.

### 1) Prepare GitHub repo
- Push your code (without datasets, `mlruns/`, large checkpoints). Add checkpoints via secure transfer (SCP/S3 with least privilege) or include only small test checkpoints if necessary.
- Ensure `.gitignore` excludes `data/`, `mlruns/`, large `checkpoints/` as appropriate.

### 2) Launch eligible EC2 instance (Ubuntu)
- In AWS Console, launch an Ubuntu Server (e.g. Ubuntu 24.04 LTS). Target **t3.micro/t4g.micro or equivalent Free Tier eligible** instance with ~1 GiB RAM if available; **~2 GiB RAM is recommended** for CPU PyTorch + Streamlit. Check current Free Tier eligibility in your account.
- Use gp2/gp3 EBS volume sized for OS + deps + checkpoint (start small, e.g. 8-20 GB). 
- **If 2 GiB is not available or insufficient**: PyTorch CPU wheels are heavy; with only 1 GiB you may hit OOM on first load. Consider the smallest instance that fits, or reduce image size/model. Document actual RAM/disk usage after testing.

### 3) Restrict SSH
- Create/select a key pair.
- Security group: allow SSH (22) **only from your own IP** (not 0.0.0.0/0). Do not leave SSH open to the internet.
- If exposing Streamlit publicly (see below), only open 8501 if necessary and understand HTTP exposure.

### 4) Install dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl

# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or open new shell
```

### 5) Clone repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd Lumoculis
```

### 6) Configure model checkpoint securely
- Transfer checkpoint to instance via `scp -i <key.pem> checkpoint.pt ubuntu@<ip>:/home/ubuntu/Lumoculis/checkpoints/`
- Set env vars (e.g. in `.env` or systemd unit). Never commit checkpoints.
```bash
export LUMOCULIS_CHECKPOINT=/home/ubuntu/Lumoculis/checkpoints/your_best.pt
export LUMOCULIS_DEVICE=cpu
export LUMOCULIS_MAX_UPLOAD_MB=10
# optional calibration/quality gate
```

### 7) Install Python deps and run as persistent service
```bash
uv sync
```

Create a systemd service (example):
```ini
# /etc/systemd/system/lumoculis.service
[Unit]
Description=Lumoculis Streamlit
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Lumoculis
Environment=PATH=/home/ubuntu/.local/bin:/usr/bin
Environment=LUMOCULIS_CHECKPOINT=/home/ubuntu/Lumoculis/checkpoints/your_best.pt
Environment=LUMOCULIS_DEVICE=cpu
Environment=LUMOCULIS_MAX_UPLOAD_MB=10
ExecStart=/home/ubuntu/.local/bin/uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lumoculis
sudo systemctl status lumoculis
```

### 8) Networking (public access)
- To test publicly: open port **8501** in security group to your IP only (or restrict as much as possible). 
- **HTTP vs HTTPS**: Streamlit over HTTP is unencrypted in transit; avoid sending sensitive data. Adding HTTPS without custom domain/paid cert can be done via a reverse proxy (nginx + Let's Encrypt) on the instance, but Let's Encrypt requires DNS validation and a domain; some approaches need ports 80/443 open. Free options depend on setup - no service is guaranteed free. Consider tunneling (e.g. Cloudflare Tunnel) which may have free tiers but review terms.
- Do **not** open 8501 to 0.0.0.0/0 unless necessary and you understand risks.

### 9) Test inference
```bash
curl http://<instance-ip>:8501  # or open in browser
# Upload test image and verify prediction + JSON download
```

### 10) Logs and restart
```bash
sudo journalctl -u lumoculis -n 50 -f
sudo systemctl restart lumoculis
sudo systemctl stop lumoculis
```

### 11) Stop/terminate and cleanup
- Stop/terminate instance in EC2 console when done.
- Delete attached EBS volumes if not needed.
- Check Elastic IPs (if allocated and not associated) - can incur charges; release if unused.
- Review any snapshots/AMIs/S3 buckets created.
- Verify in billing dashboard.


## Cost safeguards
- **Check Free Tier**: AWS Free Tier page for EC2 (instance hours, EBS, data transfer). Eligibility varies by account/region.
- **Check credits/billing**: Monitor AWS Billing > Bills/Credits. Credits may have expiry.
- **Review charges**: Watch EC2 (hours/GB), EBS (storage/snapshots), public IPv4 (charges may apply in some cases), data transfer out.
- **Billing alerts**: Set up AWS Budgets/Billing alarms (alerts don't stop resources or guarantee ₹0).
- **Clean up**: After testing, terminate instance, release Elastic IP, delete volumes/snapshots. Avoid NAT gateways, load balancers, RDS, etc. No other services required.
- **Note**: "Free Tier" is time/usage limited; do not assume permanent free. Budget ₹0 means you must verify and clean up.

