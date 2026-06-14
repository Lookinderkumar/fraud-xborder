# Model Metrics

| Model             | ROC-AUC | PR-AUC | Brier   | Recall@0.5%FPR | Prec@Top1% |
|-------------------|---------|--------|---------|----------------|------------|
| LR Baseline       |  0.9493 | 0.3332 | 0.07479 |          15.5% |      40.0% |
| XGBoost only      |  0.9918 | 0.8074 | 0.03417 |          70.9% |      95.0% |
| Rules only        |     N/A |    N/A |     N/A |           0.0% |      31.0% |
| Hybrid (R+X)      |  0.9733 | 0.2904 | 0.05261 |           0.0% |      27.5% |
| + Isol. Forest    |  0.9909 | 0.8045 | 0.03735 |          72.5% |      94.0% |

## Targets

| Metric              | Target       | Status          |
|---------------------|--------------|-----------------|
| ROC-AUC             | > 0.92       | 0.9909 [OK]     |
| PR-AUC              | > 0.80       | 0.8045 [OK]     |
| Brier Score         | < 0.05       | 0.037  [OK]     |
| Precision@Top1%     | Maximise     | 94.0%  [OK]     |
| Recall @ FPR=0.5%   | Maximise     | 72.5%  [OK]     |

## Thresholds

| Threshold         | Value  | Notes                                      |
|-------------------|--------|--------------------------------------------|
| FPR-based         | 0.7935 | Recall=72.5%, Precision=76.1% at FPR=0.5% |
| Cost-optimised    | 0.1134 | Minimises FN*250000 + FP*50                |

See `models/threshold.json` for full threshold config.

## Latency and Throughput (stress test, N=500 in-process)

| Metric        | Result    | Target     |
|---------------|-----------|------------|
| p50 latency   | 17.1 ms   | < 1,000 ms |
| p95 latency   | 19.9 ms   | < 2,000 ms |
| p99 latency   | 21.3 ms   | N/A        |
| Throughput    | 57.4 txn/s| > 10 txn/s |

## Cost Model

- cost_FN = EUR 250,000  (average fraud loss per transaction)
- cost_FP = EUR 50  (analyst review cost per false positive)

See `reports/business.md` for full ROI analysis.
