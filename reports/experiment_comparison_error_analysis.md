# Error analysis: baseline vs Experiment C

## Alignment
Validation IDs and labels identical across all four experiments (730 samples).

## Per-class recall deltas (C - baseline)

- Grade 0: +0.0166
- Grade 1: -0.0270
- Grade 2: -0.3550
- Grade 3: +0.4737
- Grade 4: +0.4035

## Grade-3 / grade-4 errors

- baseline: grade3 n=38, recall=0.0000, predicted-4 on non-grade4=5
- expA: grade3 n=38, recall=0.5000, predicted-4 on non-grade4=62
- expB: grade3 n=38, recall=0.0000, predicted-4 on non-grade4=99
- expC: grade3 n=38, recall=0.4737, predicted-4 on non-grade4=66

## Confidence (max prob) correct vs incorrect

- baseline: correct=0.8999, incorrect=0.7051, acc=0.8110
- expA: correct=0.9174, incorrect=0.6572, acc=0.6342
- expB: correct=0.8074, incorrect=0.5311, acc=0.6849
- expC: correct=0.8892, incorrect=0.6523, acc=0.7753

## Macro F1 per-grade deltas (C - baseline)

- Grade 0: -0.0010
- Grade 1: -0.0280
- Grade 2: -0.1110
- Grade 3: +0.4615
- Grade 4: +0.0712

## Normalized confusion matrices saved for baseline and Experiment C.
