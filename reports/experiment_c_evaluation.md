# Experiment C Evaluation

## Checkpoint metadata
- Path: checkpoints/expC_weighted_sampling_20261101_172044/best.pt
- epoch (0-indexed): 2 (final of 3 epochs)
- val_qwk (stored): 0.8376704301648074
- config: image_size=384, batch_size=2, epochs=3, lr=0.0003, model=efficientnet_b0, seed=42

## Dataset
- Validation examples: 730
- Same validation IDs/labels as baseline, Exp A, Exp B: yes

## Overall metrics
- QWK: 0.83767
- Accuracy: 0.77534
- MAE: 0.34384
- Balanced accuracy: 0.65251
- Macro F1: 0.62795
- Referable DR AUROC (grade>=2): 0.97899
- Sensitivity (thr 0.5): 0.91864
- Specificity (thr 0.5): 0.92184

## Counts
True: [361, 74, 200, 38, 57]
Pred: [362, 69, 154, 40, 105]

## Grade 3 investigation
- recall: 0.47368
- confusion row (true grade3): [0, 0, 10, 18, 10]
- mean predicted prob grade3 among true grade3: 0.37221
- grade3 highest prob among true grade3: 18/38
