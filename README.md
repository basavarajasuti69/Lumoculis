# Lumoculis

### AI-Assisted Diabetic Retinopathy Screening Support

Lumoculis is a computer vision research project that classifies retinal fundus images into five diabetic retinopathy (DR) severity grades, from 0 to 4, using the APTOS 2019 dataset. It also flags referable DR cases (grade ≥ 2).

**Live Demo:** [Open Lumoculis](https://theater-planning-shaft-isa.trycloudflare.com)

> **Research disclaimer:** Lumoculis is an experimental research prototype, not a medical device. Its predictions are not diagnoses and must not replace assessment by a qualified eye-care professional. The system has not been clinically validated. Do not use its results to make medical decisions.

## Project Highlights

- **Image classification:** Predicts one of five diabetic retinopathy severity grades.
- **Deep learning:** Uses PyTorch and EfficientNet-B0.
- **Model evaluation:** Includes Quadratic Weighted Kappa (QWK), referable DR AUROC, sensitivity, specificity, and confusion-matrix analysis.
- **Explainability:** Supports Grad-CAM visualizations to explore regions influencing model predictions.
- **Image quality assessment:** Includes image-quality checks before or alongside inference.
- **Probability calibration:** Supports temperature scaling and calibrated probability outputs.
- **Experiment tracking:** Integrates MLflow for recording training experiments.
- **Automated testing:** Includes pytest tests and GitHub Actions CI checks.
- **Cloud deployment:** Runs a Streamlit web application on AWS EC2, with Nginx and a Cloudflare HTTPS tunnel.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, EfficientNet-B0 |
| Computer Vision | Fundus image preprocessing, Grad-CAM |
| Data Processing | NumPy, Pandas |
| Evaluation | QWK, AUROC, sensitivity, specificity, calibration |
| Experiment Tracking | MLflow |
| Web Application | Streamlit |
| Cloud Deployment | AWS EC2, Nginx, Cloudflare Tunnel |
| Testing and CI | pytest, Ruff, GitHub Actions |

## Live Demo

Try the deployed application by opening the link below.

**[Launch Lumoculis](https://theater-planning-shaft-isa.trycloudflare.com)**

The application supports retinal image uploads, model predictions, probability outputs, optional image-quality assessment, optional Grad-CAM visualizations, and JSON report downloads.

**Demo availability:** This uses a temporary Cloudflare Quick Tunnel URL. The link may change if the tunnel is recreated, and availability is not guaranteed.

## Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset.

The model predicts the following severity grades:

| Grade | Description |
|---|---|
| 0 | No diabetic retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative diabetic retinopathy |

Referable DR is defined in this project as grade ≥ 2. Dataset access and use must follow the applicable dataset licence and terms.

## Safety and Limitations

- This is a research prototype, not a clinically validated diagnostic system.
- Predictions must not be used as a substitute for professional medical assessment.
- APTOS 2019 does not provide patient IDs in the dataset used here, so patient-level independence between training and validation data cannot be guaranteed.
- Image-quality thresholds are heuristic and have not been clinically validated.
- Grad-CAM highlights model-relevant regions; it does not establish disease location or causality.
- Probability calibration does not establish clinical reliability.
- Avoid uploading identifiable patient images to the public demo.

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, out-of-scope use, data, metrics, bias, and limitations.
