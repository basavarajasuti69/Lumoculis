# Experiment E — Evaluation Report

## 1. Configuration

- Model: EfficientNet-B0
- Dataset: APTOS 2019
- Number of classes: 5 (Grades 0–4)
- Input image size: 384 × 384
- Batch size: 2
- Epochs: 3
- Optimizer learning rate: 0.0003
- Weight decay: 0.05
- Weighted sampler: Enabled
- Class weight strength: 0.75
- Augmentation: Light
- Training device: Apple Silicon MPS
- Validation examples: 730

## 2. Overall Metrics

| Metric | Result |
|---|---:|
| Quadratic Weighted Kappa (QWK) | 0.87564 |
| Accuracy | 0.80000 |
| Mean Absolute Error (MAE) | 0.26712 |
| Balanced Accuracy | 0.69845 |
| Macro F1 | 0.67084 |
| Referable DR AUROC | 0.97976 |
| Sensitivity | 0.91525 |
| Specificity | 0.93333 |

## 3. Per-Class Recall

| Grade | True Samples | Predicted Samples | Recall |
|---|---:|---:|---:|
| 0 | 361 | 353 | 0.96122 |
| 1 | 74 | 88 | 0.62162 |
| 2 | 200 | 170 | 0.65500 |
| 3 | 38 | 55 | 0.60526 |
| 4 | 57 | 64 | 0.64912 |

## 4. Confusion Matrix

Rows represent the true grade; columns represent the predicted grade.

| True \ Predicted | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
|---|---:|---:|---:|---:|---:|
| Grade 0 | 347 | 12 | 1 | 0 | 1 |
| Grade 1 | 5 | 46 | 15 | 2 | 6 |
| Grade 2 | 1 | 29 | 131 | 24 | 15 |
| Grade 3 | 0 | 0 | 10 | 23 | 5 |
| Grade 4 | 0 | 1 | 13 | 6 | 37 |

## 5. Grade 3 Analysis

- Correct Grade 3 predictions: 23 out of 38.
- Grade 3 recall: 60.53%.
- Mean predicted probability for Grade 3 among true Grade 3 samples: 0.49454.
- True Grade 3 samples for which Grade 3 had the highest predicted probability: 23 of 38.
- Grade 4 predictions on non-Grade-4 samples: 27.

The model improves detection of Grade 3 cases compared with the baseline, but still misclassifies some Grade 3 images as Grades 2 and 4.

## 6. Interpretation

Experiment E is the current leading candidate among Experiments C, D, and E when considering QWK, balanced accuracy, macro F1, and Grade 3 recall together.

Its performance involves trade-offs. Some Grade 2 cases are classified as Grade 3 or Grade 4. Experiment D has slightly higher accuracy and slightly lower MAE, while Experiment E provides more balanced per-class recall.

## 7. Limitations

- Results are based on a validation set of 730 images.
- Multiple configurations have been compared on the same validation set.
- An independent test set is needed for a less biased final performance estimate.
- These results do not establish clinical safety or clinical readiness.
- Further evaluation is needed before any real-world medical use.

## 8. Reproducibility

Experiment configuration:
`configs/experiment_e_sampling_medium_loss_weights.yaml`

Run identifier:
`expE_medium_loss_weights_20261101_182349`

Checkpoint:
`checkpoints/expE_medium_loss_weights_20261101_182349/best.pt`

Validation predictions:
`checkpoints/expE_medium_loss_weights_20261101_182349/val_preds.npz`
