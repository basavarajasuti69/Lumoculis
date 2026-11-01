# Lumoculis — Experiment Comparison Report

## 1. Objective

This report compares the baseline model and Experiments A–E for five-class diabetic retinopathy grading using the APTOS 2019 dataset.

The experiments investigate the effects of weighted sampling and different class-weight strengths.

## 2. Experimental Results

All experiments use the same 730-image validation set.

| Metric | Baseline | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| QWK | 0.85218 | 0.74057 | 0.75342 | 0.83767 | 0.87326 | 0.87564 |
| Accuracy | 0.81100 | 0.63425 | 0.68493 | 0.77534 | 0.80411 | 0.80000 |
| MAE | 0.26000 | 0.53425 | 0.52329 | 0.34384 | 0.26301 | 0.26712 |
| Balanced accuracy | 0.55020 | 0.54588 | 0.47745 | 0.65251 | 0.62953 | 0.69845 |
| Macro F1 | 0.54940 | 0.44693 | 0.42804 | 0.62795 | 0.63369 | 0.67084 |
| Referable DR AUROC | 0.97600 | 0.93145 | 0.93940 | 0.97899 | 0.98055 | 0.97976 |
| Grade 3 recall | 0.00000 | 0.50000 | 0.00000 | 0.47368 | 0.26316 | 0.60526 |

Note: Values are reproduced from the recorded experiment evaluations and rounded for display. Small differences may arise from rounding.

## 3. Experiment Configurations

- **Baseline:** Original training configuration.
- **Experiment A:** Weighted sampling with class-weighted loss.
- **Experiment B:** Class-weighted loss without weighted sampling.
- **Experiment C:** Weighted sampling and class-weighted loss, trained for three epochs.
- **Experiment D:** Weighted sampling with reduced class-weight strength of 0.50, trained for three epochs.
- **Experiment E:** Weighted sampling with medium class-weight strength of 0.75, trained for three epochs.

Experiments D and E use the same general training setup, with class-weight strength as the principal configuration difference.

## 4. Key Findings

### Baseline

The baseline achieved the highest accuracy among the six experiments at approximately 81.10%. However, its Grade 3 recall was zero, indicating that it failed to correctly classify any Grade 3 validation examples.

### Experiments A and B

Experiments A and B did not outperform the baseline on overall accuracy, QWK, macro F1, or referable DR AUROC.

Experiment A achieved 50% Grade 3 recall, but its accuracy fell to approximately 63.43%. Experiment B did not correctly classify any Grade 3 examples.

### Experiment C

Experiment C improved balanced accuracy, macro F1, referable DR AUROC, and Grade 3 recall compared with the baseline. However, it had lower accuracy and QWK than the baseline.

### Experiment D

Experiment D achieved a QWK of 0.87326 and accuracy of 80.41%. Its Grade 3 recall was 26.32%, lower than Experiments C and E.

### Experiment E

Experiment E achieved the highest QWK (0.87564), balanced accuracy (0.69845), macro F1 (0.67084), and Grade 3 recall (0.60526) among the six recorded experiments.

Experiment E did not achieve the highest accuracy, lowest MAE, or highest referable DR AUROC. Experiment D performed slightly better on those three metrics.

## 5. Current Candidate Selection

Experiment E is the current leading candidate when prioritizing QWK, balanced accuracy, macro F1, and Grade 3 recall together.

This is a metric-based selection for further research, not proof that Experiment E is universally superior. The preferred model depends on the intended evaluation priorities.

Experiment E also has important limitations:
- Grade 2 recall is 65.50%.
- Only 23 of 38 true Grade 3 examples were correctly classified.
- Some Grade 2 cases were predicted as Grade 3 or Grade 4.
- Multiple experiments have been compared on the same validation set.

## 6. Limitations and Next Steps

1. Evaluate the selected checkpoint on a genuinely independent test set.
2. Investigate Grade 2 and Grade 3 misclassifications.
3. Consider repeated runs with different random seeds to assess variability.
4. Report per-class precision, recall, F1, and confusion matrices alongside aggregate metrics.
5. Avoid claims of clinical readiness or safety based on these experiments alone.

## 7. Reproducibility

Experiment configuration files:

- `configs/experiment_d_sampling_reduced_loss_weights.yaml`
- `configs/experiment_e_sampling_medium_loss_weights.yaml`

Experiment E run:

- `expE_medium_loss_weights_20261101_182349`

Experiment E checkpoint:

- `checkpoints/expE_medium_loss_weights_20261101_182349/best.pt`

Experiment E validation predictions:

- `checkpoints/expE_medium_loss_weights_20261101_182349/val_preds.npz`

The validation predictions for all six experiments are retained under their respective directories in `checkpoints/`.
