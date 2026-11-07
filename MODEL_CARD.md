# Model Card — Lumoculis

## Model overview

Lumoculis classifies retinal fundus images into five diabetic-retinopathy (DR) severity grades (0–4) using an EfficientNet-B0 model trained with the APTOS 2019 dataset. The project also defines referable DR as grade 2 or higher.

## Intended use

Research, education and software-development evaluation of retinal-image grading workflows. The system may be used to explore model predictions, image-quality heuristics, probability calibration and Grad-CAM explainability.

## Out-of-scope use

- Not a medical device.
- Not clinically validated.
- Not intended for autonomous diagnosis, triage or treatment decisions.
- Not a substitute for examination or advice from a qualified eye-care professional.
- Not established as safe or effective for deployment in clinics or screening programs.

## Model details

- Architecture: EfficientNet-B0.
- Framework: PyTorch for training and Grad-CAM; ONNX Runtime for the deployed prediction service.
- Number of classes: five.
- Input size: 384 × 384 pixels for the documented checkpoint.
- Referable DR project threshold: grade >= 2.

## Data

APTOS 2019 (Kaggle). The dataset contains retinal fundus images labeled with five DR severity grades. The dataset does not provide patient IDs, so patient-level independence cannot be established from the available identifiers. The repository does not redistribute the dataset; users must follow its access and usage terms.

## Evaluation

The project reports quadratic weighted kappa, accuracy, mean absolute error, balanced accuracy, macro F1, referable-DR AUROC, class-specific recall, sensitivity and specificity.

Validation results and exploratory holdout results are documented in the repository README. The exploratory holdout result must not be interpreted as an independent unbiased estimate because original training IDs were not retained and overlap cannot be ruled out.

## Known limitations and risks

- Training and evaluation are based on APTOS 2019 and may not generalize to other populations, cameras, image protocols or clinical settings.
- Patient-level data separation cannot be verified because patient identifiers are unavailable.
- Exact-duplicate detection does not guarantee detection of near-duplicate images.
- Class imbalance can affect minority-grade performance.
- Image-quality thresholds are heuristic and are not clinically validated.
- Probability calibration does not establish clinical reliability.
- Grad-CAM highlights model-influential regions; it does not establish disease location, pathology or causality.
- External validation on independent datasets and prospective clinical evaluation have not been established.

## Bias and fairness considerations

Performance may vary across imaging devices, acquisition settings, patient populations and disease distributions. The available project evaluation does not establish equitable performance across demographic or clinical subgroups. Such evaluations are required before any clinical use could be considered.

## Responsible use

Treat all outputs as experimental model predictions. Do not use Lumoculis as the sole basis for clinical decisions. Refer individuals to qualified eye-care professionals for medical assessment.
