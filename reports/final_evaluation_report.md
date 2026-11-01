# Final Evaluation Report: Lumoculis Diabetic Retinopathy Grading

## Project overview and objective
Lumoculis is a research prototype for 5-class diabetic retinopathy (DR) grading (grades 0-4) using EfficientNet-B0 on the APTOS 2019 dataset.

## Model, checkpoint, and evaluation configuration
- Checkpoint: checkpoints/expC_weighted_sampling_20261101_172044/best.pt
- Configuration: configs/experiment_c_weighted_sampling_3epochs.yaml
- Model: EfficientNet-B0, 5 classes, image size 384
- Checkpoint compatibility verified: model name, number of classes, and image size all match.

## Dataset and split methodology
- Historical validation IDs (730) and their exact-duplicate groups were excluded.
- Eligible population: 2,932 images.
- New split: train 2,061, validation 439, test 432 (seed 42, duplicate-aware, 70/15/15). These counts describe the newly generated duplicate-aware split only.
- The existing Experiment C checkpoint was evaluated on the holdout test set; these counts do not mean the checkpoint was retrained on the new split.
- Original labels were preserved; conflicting-label groups are reported, not relabeled.
- Train, validation, test, and excluded ID sets are pairwise disjoint; no exact-duplicate group crosses split boundaries.

## Overall metrics
| Metric | Value |
|---|---:|
| QWK | 0.9058 |
| Accuracy | 84.95% |
| MAE | 0.2153 |
| Balanced accuracy | 78.93% |
| Macro F1 | 75.27% |
| Referable DR AUROC | 0.9886 |
| Referable sensitivity | 92.61% |
| Referable specificity | 96.88% |
| Test samples | 432 |

## Per-class metrics
| Grade | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9907 | 0.9815 | 0.9860 | 216 |
| 1 | 0.6604 | 0.8750 | 0.7527 | 40 |
| 2 | 0.8652 | 0.6525 | 0.7440 | 118 |
| 3 | 0.6667 | 0.6087 | 0.6364 | 23 |
| 4 | 0.5273 | 0.8286 | 0.6444 | 35 |

## Confusion matrix
| true\pred | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 212 | 4 | 0 | 0 | 0 |
| 1 | 0 | 35 | 4 | 0 | 1 |
| 2 | 2 | 14 | 77 | 7 | 18 |
| 3 | 0 | 0 | 2 | 14 | 7 |
| 4 | 0 | 0 | 6 | 0 | 29 |

## Key strengths and weaknesses
Strengths: strong overall QWK and accuracy; excellent grade 0 and grade 1 recall; strong referable DR AUROC, sensitivity, and specificity.
Weaknesses: grade 2 recall is only 65.25%; grade 4 precision is low at 52.73%, indicating overprediction of grade 4.

## Error analysis
The confusion matrix shows most errors involve grade 2, grade 3, and grade 4, with substantial confusion between adjacent severe grades. Grade 4 overprediction is the main precision issue.

## Data leakage and duplicate-handling checks
- Historical validation IDs and their exact-duplicate groups were excluded.
- No exact-duplicate group crosses the new split boundaries.
- Conflicting-label groups are documented and not silently relabeled.
- APTOS 2019 lacks patient IDs, so exact-duplicate grouping cannot guarantee patient-level independence.

## Limitations and reproducibility notes
This is an exploratory holdout evaluation, not a definitive independent benchmark or clinical validation. The original training IDs were not saved, so it cannot be proven that the checkpoint never saw any new holdout-test images during training.

## Conclusion and realistic next steps
The model shows promising overall performance but needs improvement in grade 2 recall and grade 4 precision. Future work should address class imbalance, severe-grade confusion, and rigorous independent validation. No clinical readiness claim is made.
