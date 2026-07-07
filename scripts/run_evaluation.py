"""
Task 10: Dynamic Evaluation Dataset Runner
============================================
Loads the generated data/evaluation_visa_dataset.csv, adapts it, runs the DQS pipeline,
compares pipeline action outcomes against the true_anomaly labels, and records
end-to-end metrics to BENCHMARKS.md.

Run: .venv\\Scripts\\python scripts/run_evaluation.py
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.csv_adapter import adapt_csv_to_visa
from src.dqs_engine import DQSEngine
from src.config import Action

def run_evaluation():
    print("=" * 60)
    print("  DQS PIPELINE EVALUATION RUNNER")
    print("=" * 60)

    # 1. Load evaluation dataset
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "evaluation_visa_dataset.csv"
    )
    if not os.path.exists(csv_path):
        print(f"[ERROR] Evaluation dataset not found at {csv_path}. Please run Task 9 script first.")
        return

    print(f"\n[1/4] Loading dataset: {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    # 2. Adapt to VISA format using the adapter
    print("\n[2/4] Adapting CSV fields to standardized DQS schema...")
    transactions, metadata = adapt_csv_to_visa(csv_content)
    print(f"    Loaded {len(transactions)} transactions.")
    print(f"    Schema compliance: {metadata['compliance_score']:.1f}%")

    # Retrieve ground truth labels
    df_raw = pd.read_csv(csv_path)
    ground_truth = df_raw["true_anomaly"].tolist()
    anomaly_reasons = df_raw["anomaly_reason"].tolist()

    # 3. Run transactions through the full DQS pipeline
    print("\n[3/4] Executing DQS pipeline (15 layers, no AI)...")
    engine = DQSEngine(use_ai=False)
    
    # Process batch
    pipeline_result = engine.run(transactions)
    print(f"    Pipeline completed in {pipeline_result.total_duration_ms:.1f}ms")
    print(f"    Processed: {pipeline_result.total_records} | Quality rate: {pipeline_result.quality_rate}%")

    # Match decisions against ground truth
    # Engine decision: Action.SAFE_TO_USE represents normal.
    # Action.REVIEW, Action.ESCALATE, or Action.NO_ACTION represents anomalous detection.
    decisions = getattr(pipeline_result, "decision_report_details", None) or []
    if not decisions and hasattr(engine, "layer_results"):
        # If detail list is empty, fetch decisions from Layer 9 Decision Gate
        decision_layer = engine.layer_results.get(9.0) or engine.layer_results.get(9)
        if decision_layer and hasattr(decision_layer, "details") and "decisions" in decision_layer.details:
            decisions = decision_layer.details["decisions"]
        else:
            # Fallback direct lookup from engine
            decisions = getattr(engine, "decisions", [])

    # Compile predictions
    predictions = []
    for idx, txn in enumerate(transactions):
        # Default prediction is normal (0)
        pred = 0
        
        # Check Layer 9 decision if available
        if idx < len(decisions):
            d = decisions[idx]
            # If decision dict/object indicates anything other than SAFE_TO_USE, it's flagged as anomaly (1)
            action_val = d.action.value if hasattr(d.action, "value") else (d.get("action") if isinstance(d, dict) else "")
            if action_val != Action.SAFE_TO_USE.value:
                pred = 1
        predictions.append(pred)

    # 4. Compute Metrics
    print("\n[4/4] Computing metrics comparing pipeline outcomes against ground truth...")
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # Track metrics per anomaly profile
    profile_stats = {}
    
    for idx in range(len(ground_truth)):
        y_true = ground_truth[idx]
        y_pred = predictions[idx]
        reason = anomaly_reasons[idx]
        
        if y_true == 1 and y_pred == 1:
            tp += 1
            profile_stats[reason] = profile_stats.get(reason, {"total": 0, "detected": 0})
            profile_stats[reason]["total"] += 1
            profile_stats[reason]["detected"] += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
            profile_stats[reason] = profile_stats.get(reason, {"total": 0, "detected": 0})
            profile_stats[reason]["total"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(ground_truth)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n    EVALUATION RESULTS")
    print("    " + "-" * 40)
    print(f"    True Positives  (TP): {tp}")
    print(f"    False Positives (FP): {fp}")
    print(f"    True Negatives  (TN): {tn}")
    print(f"    False Negatives (FN): {fn}")
    print("    " + "-" * 40)
    print(f"    Precision:  {precision*100:5.1f}%")
    print(f"    Recall:     {recall*100:5.1f}%")
    print(f"    F1-Score:   {f1*100:5.1f}%")
    print(f"    FPR:        {fpr*100:5.1f}%")
    print(f"    Accuracy:   {accuracy*100:5.1f}%")

    print("\n    ANOMALY DETECTION BY PROFILE")
    print("    " + "-" * 40)
    for reason, stats in profile_stats.items():
        rate = (stats["detected"] / stats["total"]) * 100
        print(f"    {reason:30s}: {stats['detected']}/{stats['total']} ({rate:.1f}%)")

    # Append report to BENCHMARKS.md
    benchmarks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "BENCHMARKS.md"
    )

    eval_section = f"""
## End-to-End Evaluation Dataset Results
*Evaluated on data/evaluation_visa_dataset.csv (1,000 transactions, 26 injected anomalies)*

### Pipeline Metrics
| Metric | Value |
|--------|-------|
| True Positives (TP) | {tp} |
| False Positives (FP) | {fp} |
| True Negatives (TN) | {tn} |
| False Negatives (FN) | {fn} |
| **Precision** | **{precision*100:.1f}%** |
| **Recall** | **{recall*100:.1f}%** |
| **F1-Score** | **{f1*100:.1f}%** |
| **Accuracy** | **{accuracy*100:.1f}%** |
| **False Positive Rate (FPR)** | **{fpr*100:.1f}%** |

### Detection Rate by Anomaly Profile
| Anomaly Profile | Detected / Total | Rate |
|-----------------|------------------|------|
"""
    for reason, stats in profile_stats.items():
        rate = (stats["detected"] / stats["total"]) * 100
        eval_section += f"| {reason} | {stats['detected']}/{stats['total']} | {rate:.1f}% |\n"

    with open(benchmarks_path, "a", encoding="utf-8") as f:
        f.write(eval_section)

    print(f"\n  Appended metrics to: {benchmarks_path}")
    print("\n" + "=" * 60)
    print("  TASK 10 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
