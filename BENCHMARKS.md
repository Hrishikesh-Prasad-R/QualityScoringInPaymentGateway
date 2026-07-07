# DQS Engine — Benchmark Results

## Anomaly Detection Accuracy
*Generated against 1000 labeled VISA transactions (anomaly_rate=0.2, seed=42)*

| Metric | Value |
|--------|-------|
| True Positives | 200 |
| False Positives | 74 |
| True Negatives | 726 |
| False Negatives | 0 |
| **Precision** | **73.0%** |
| **Recall** | **100.0%** |
| **F1-Score** | **84.4%** |
| False Positive Rate | 9.2% |
| Accuracy | 92.6% |
| Detection Time | 356.4ms for 1000 records |

**Resume bullet:**
> "IsolationForest anomaly detector achieved 100% recall at 9% false positive rate across 1,000 labeled VISA transactions (F1=84%)"

