"""
Phase 4 Demo Script - Test the pipeline end-to-end
"""
import sys
sys.path.insert(0, 'src')

from src.layers import *
from src.data_generator import generate_visa_transactions

def main():
    # Generate test data
    print("=" * 60)
    print("  DATA QUALITY SCORING ENGINE - PHASE 4 DEMO")
    print("=" * 60)
    print()
    
    print("Generating 20 test transactions (10% anomalies)...")
    transactions = generate_visa_transactions(
        n_transactions=20, 
        anomaly_rate=0.1, 
        random_seed=42
    )

    # Run Pipeline Layers 1-3
    print("\n[Layer 1] Input Contract...")
    layer1 = InputContractLayer()
    result1 = layer1.validate_schema_manifest(use_default=True)
    print(f"  Status: {result1.status.value}")

    print("\n[Layer 2] Input Validation...")
    layer2 = InputValidationLayer(layer1.get_schema())
    result2 = layer2.validate(json_data=transactions)
    print(f"  Status: {result2.status.value}, Records: {len(layer2.get_dataframe())}")

    print("\n[Layer 3] Feature Extraction...")
    layer3 = FeatureExtractionLayer()
    result3 = layer3.extract_features(layer2.get_dataframe())
    print(f"  Status: {result3.status.value}, Features: {result3.details['features_extracted']}")

    # Run Layer 4.1
    print("\n[Layer 4.1] Structural Integrity...")
    layer41 = StructuralIntegrityLayer()
    result41 = layer41.validate(layer2.get_dataframe(), layer3.get_features())
    valid = result41.details.get('valid_records', 0)
    rejected = result41.details.get('rejected_records', 0)
    print(f"  Valid: {valid}, Rejected: {rejected}")

    # Run Layer 4.2
    print("\n[Layer 4.2] Field Compliance (7 Dimensions)...")
    layer42 = FieldComplianceLayer()
    result42 = layer42.score(
        layer2.get_dataframe(), 
        layer3.get_features(), 
        layer41.get_valid_indices()
    )
    dqs_mean = result42.details.get('dqs_mean', 0)
    dim_avgs = result42.details.get('dimension_averages', {})
    print(f"  Mean DQS: {dqs_mean:.1f}/100")
    print(f"  Dimensions: {dim_avgs}")

    # Run Layer 4.3
    print("\n[Layer 4.3] Semantic Validation (12 Rules)...")
    layer43 = SemanticValidationLayer()
    result43 = layer43.validate(
        layer2.get_dataframe(), 
        layer3.get_features(), 
        layer41.get_valid_indices()
    )
    sem_rejected = result43.details.get('records_rejected', 0)
    sem_flagged = result43.details.get('records_flagged', 0)
    print(f"  Rejected: {sem_rejected}, Flagged: {sem_flagged}")

    # Run Layer 4.4
    print("\n[Layer 4.4] Anomaly Detection (ML)...")
    layer44 = AnomalyDetectionLayer()
    result44 = layer44.detect(
        layer3.get_features(), 
        layer41.get_valid_indices()
    )
    flagged = result44.details.get('records_flagged', 0)
    high_risk = result44.details.get('high_risk_count', 0)
    levels = result44.details.get('level_counts', {})
    print(f"  Flagged: {flagged}, High Risk: {high_risk}")
    print(f"  Level counts: {levels}")

    # Run Layer 4.5
    print("\n[Layer 4.5] GenAI Summarization...")
    layer45 = GenAISummarizationLayer()
    result45 = layer45.summarize(
        layer2.get_dataframe(),
        layer3.get_features(),
        layer42.get_dqs_dataframe(),
        layer44.get_anomaly_dataframe(),
        layer43.get_validation_results(),
    )
    priority_counts = result45.details.get('priority_counts', {})
    print(f"  Priority counts: {priority_counts}")

    # Print batch report
    print()
    print(layer45.generate_batch_report())
    
    # Print sample summaries
    print("\n" + "=" * 60)
    print("  SAMPLE SUMMARIES")
    print("=" * 60)
    for summary in layer45.get_summaries()[:5]:
        print(f"\n{summary.summary}")
        if summary.key_issues:
            print(f"  Issues: {summary.key_issues[:2]}")
        if summary.recommendations:
            print(f"  Recommendations: {summary.recommendations[:1]}")


if __name__ == "__main__":
    main()
