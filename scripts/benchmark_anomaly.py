"""
Task 1: Benchmark Anomaly Detector
===================================
Measures the anomaly detection accuracy against ground truth labels
injected by the data generator (high risk score, velocity fail, etc.)

Ground truth: anomaly_rate=0.20 means 200/1000 records are flagged
as anomalous at the DATA level (high risk score 70-95, velocity=fail,
international transactions). The anomaly detector should catch these.

Run: .venv\\Scripts\\python scripts/benchmark_anomaly.py
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def run_benchmark():
    print("=" * 60)
    print("  ANOMALY DETECTOR BENCHMARK")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1: Generate labeled transactions
    # ----------------------------------------------------------------
    print("\n[1/5] Generating 1,000 labeled transactions (anomaly_rate=0.20)...")
    from src.data_generator import generate_visa_transactions

    N = 1000
    ANOMALY_RATE = 0.20
    SEED = 42

    np.random.seed(SEED)
    n_anomalies = int(N * ANOMALY_RATE)
    anomaly_indices = set(np.random.choice(N, n_anomalies, replace=False))

    transactions = generate_visa_transactions(
        n_transactions=N,
        anomaly_rate=ANOMALY_RATE,
        random_seed=SEED,
    )

    # Ground truth: indices the generator flagged as anomalous
    y_true = [1 if i in anomaly_indices else 0 for i in range(N)]
    print(f"    Total: {N} | Anomalies (ground truth): {sum(y_true)} ({ANOMALY_RATE*100:.0f}%)")

    # ----------------------------------------------------------------
    # STEP 2: Run layers 1-3 to get features_df
    # ----------------------------------------------------------------
    print("\n[2/5] Running Layers 1-3 (Input -> Features)...")
    from src.layers import (
        InputContractLayer,
        InputValidationLayer,
        FeatureExtractionLayer,
    )

    layer1 = InputContractLayer()
    layer1.validate_schema_manifest(use_default=True)

    layer2 = InputValidationLayer(layer1.get_schema())
    result2 = layer2.validate(json_data=transactions)
    dataframe = layer2.get_dataframe()
    print(f"    Records loaded: {len(dataframe)}")

    layer3 = FeatureExtractionLayer()
    layer3.extract_features(dataframe)
    features_df = layer3.get_features()
    print(f"    Features extracted: {len(features_df.columns)}")

    # ----------------------------------------------------------------
    # STEP 3: Run Layer 4.4 Anomaly Detection
    # ----------------------------------------------------------------
    print("\n[3/5] Running Layer 4.4 (Anomaly Detection)...")
    from src.layers import AnomalyDetectionLayer

    t0 = time.perf_counter()
    layer44 = AnomalyDetectionLayer()
    result44 = layer44.detect(features_df)
    t1 = time.perf_counter()

    flagged_indices = set(layer44.get_flagged_indices())
    detection_ms = (t1 - t0) * 1000
    print(f"    Flagged by detector: {len(flagged_indices)}")
    print(f"    Detection time: {detection_ms:.1f}ms")

    # ----------------------------------------------------------------
    # STEP 4: Compute metrics
    # ----------------------------------------------------------------
    print("\n[4/5] Computing benchmark metrics...")

    y_pred = [1 if i in flagged_indices else 0 for i in range(N)]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    accuracy  = (tp + tn) / N

    print(f"\n    BENCHMARK RESULTS")
    print(f"    " + "-" * 40)
    print(f"    True Positives  (TP): {tp}")
    print(f"    False Positives (FP): {fp}")
    print(f"    True Negatives  (TN): {tn}")
    print(f"    False Negatives (FN): {fn}")
    print(f"    " + "-" * 40)
    print(f"    Precision:  {precision*100:5.1f}%")
    print(f"    Recall:     {recall*100:5.1f}%")
    print(f"    F1-Score:   {f1*100:5.1f}%")
    print(f"    FPR:        {fpr*100:5.1f}%")
    print(f"    Accuracy:   {accuracy*100:5.1f}%")

    # ----------------------------------------------------------------
    # STEP 5: Write BENCHMARKS.md
    # ----------------------------------------------------------------
    print("\n[5/5] Writing results to BENCHMARKS.md...")

    benchmarks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "BENCHMARKS.md"
    )

    anomaly_section = f"""# DQS Engine — Benchmark Results

## Anomaly Detection Accuracy
*Generated against {N} labeled VISA transactions (anomaly_rate={ANOMALY_RATE}, seed={SEED})*

| Metric | Value |
|--------|-------|
| True Positives | {tp} |
| False Positives | {fp} |
| True Negatives | {tn} |
| False Negatives | {fn} |
| **Precision** | **{precision*100:.1f}%** |
| **Recall** | **{recall*100:.1f}%** |
| **F1-Score** | **{f1*100:.1f}%** |
| False Positive Rate | {fpr*100:.1f}% |
| Accuracy | {accuracy*100:.1f}% |
| Detection Time | {detection_ms:.1f}ms for {N} records |

**Resume bullet:**
> "IsolationForest anomaly detector achieved {recall*100:.0f}% recall at {fpr*100:.0f}% false positive rate across 1,000 labeled VISA transactions (F1={f1*100:.0f}%)"

"""

    with open(benchmarks_path, "w", encoding="utf-8") as f:
        f.write(anomaly_section)

    print(f"    Written to: {benchmarks_path}")
    print("\n" + "=" * 60)
    print("  TASK 1 COMPLETE")
    print("=" * 60)
    print(f"\n  Copy this to your resume:")
    print(f"  \"IsolationForest anomaly detector achieved {recall*100:.0f}% recall")
    print(f"   at {fpr*100:.0f}% false positive rate across 1,000 labeled")
    print(f"   VISA transactions (F1={f1*100:.0f}%)\"")


if __name__ == "__main__":
    run_benchmark()
