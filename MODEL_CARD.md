# Model Card — Lumoculis DR Screening Support

## Intended use
Screening support for diabetic retinopathy (DR) in camps and rural clinics.
Classifies retinal fundus images into DR grades 0-4 (APTOS 2019) and flags
referable DR (grade >= 2) for follow-up by an eye doctor.

## Out-of-scope use
Not a diagnostic tool. Not for clinical decision-making without clinician
review. Not FDA-cleared. Not clinically validated.

## Data
APTOS 2019 (Kaggle). Train/val split is stratified by grade but not
patient-level (APTOS provides no patient IDs).

## Metrics
Quadratic weighted kappa, AUROC for referable DR, per-class sensitivity /
specificity at a threshold, confusion matrix, expected calibration error.

## Bias / limitations
Single dataset (APTOS), single source population. External validation on
IDRiD / Messidor-2 pending (Phase 4). Reported metrics are not clinical
accuracy claims.
