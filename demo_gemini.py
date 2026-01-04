"""
Gemini API Integration Demo - Test Phase 4 with AI Summaries
"""
import sys
import os
sys.path.insert(0, 'src')

# Set the API key
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

from src.layers import *
from src.data_generator import generate_visa_transactions

def main():
    print("=" * 60)
    print("  PHASE 4 DEMO WITH GEMINI AI")
    print("=" * 60)
    print()
    
    # Generate test data with some anomalies
    print("Generating 10 test transactions (20% anomalies)...")
    transactions = generate_visa_transactions(
        n_transactions=10, 
        anomaly_rate=0.2, 
        random_seed=42
    )

    # Run Layers 1-3
    print("\nRunning Layers 1-3 (Input -> Features)...")
    layer1 = InputContractLayer()
    layer1.validate_schema_manifest(use_default=True)

    layer2 = InputValidationLayer(layer1.get_schema())
    layer2.validate(json_data=transactions)

    layer3 = FeatureExtractionLayer()
    layer3.extract_features(layer2.get_dataframe())
    print(f"  Features extracted: {layer3.get_features().shape}")

    # Run Layer 4.1
    print("\n[Layer 4.1] Structural Integrity...")
    layer41 = StructuralIntegrityLayer()
    result41 = layer41.validate(layer2.get_dataframe(), layer3.get_features())
    print(f"  Valid: {result41.details.get('valid_records', 0)}")

    # Run Layer 4.2
    print("\n[Layer 4.2] Field Compliance...")
    layer42 = FieldComplianceLayer()
    result42 = layer42.score(
        layer2.get_dataframe(), 
        layer3.get_features(), 
        layer41.get_valid_indices()
    )
    print(f"  Mean DQS: {result42.details.get('dqs_mean', 0):.1f}/100")

    # Run Layer 4.3
    print("\n[Layer 4.3] Semantic Validation...")
    layer43 = SemanticValidationLayer()
    result43 = layer43.validate(
        layer2.get_dataframe(), 
        layer3.get_features(), 
        layer41.get_valid_indices()
    )
    print(f"  Flagged: {result43.details.get('records_flagged', 0)}")

    # Run Layer 4.4
    print("\n[Layer 4.4] Anomaly Detection...")
    layer44 = AnomalyDetectionLayer()
    result44 = layer44.detect(
        layer3.get_features(), 
        layer41.get_valid_indices()
    )
    print(f"  Flagged: {result44.details.get('records_flagged', 0)}")
    print(f"  Level counts: {result44.details.get('level_counts', {})}")

    # Run Layer 4.5 WITHOUT AI (template mode)
    print("\n[Layer 4.5] Summarization (Template Mode)...")
    layer45_template = GenAISummarizationLayer(use_ai=False)
    result45_template = layer45_template.summarize(
        layer2.get_dataframe(),
        layer3.get_features(),
        layer42.get_dqs_dataframe(),
        layer44.get_anomaly_dataframe(),
        layer43.get_validation_results(),
    )
    print(f"  Priority counts: {result45_template.details.get('priority_counts', {})}")
    print(f"  AI enabled: {result45_template.details.get('ai_enabled', False)}")

    # Run Layer 4.5 WITH AI (Gemini mode)
    print("\n[Layer 4.5] Summarization (Gemini AI Mode)...")
    layer45_ai = GenAISummarizationLayer(use_ai=True)
    result45_ai = layer45_ai.summarize(
        layer2.get_dataframe(),
        layer3.get_features(),
        layer42.get_dqs_dataframe(),
        layer44.get_anomaly_dataframe(),
        layer43.get_validation_results(),
    )
    print(f"  Priority counts: {result45_ai.details.get('priority_counts', {})}")
    print(f"  AI enabled: {result45_ai.details.get('ai_enabled', False)}")
    print(f"  AI calls made: {result45_ai.details.get('ai_calls_made', 0)}")
    print(f"  AI calls failed: {result45_ai.details.get('ai_calls_failed', 0)}")

    # Compare summaries
    print("\n" + "=" * 60)
    print("  SUMMARY COMPARISON")
    print("=" * 60)
    
    template_summaries = layer45_template.get_summaries()
    ai_summaries = layer45_ai.get_summaries()
    
    for i, (t_sum, ai_sum) in enumerate(zip(template_summaries[:5], ai_summaries[:5])):
        print(f"\n--- Transaction {i+1}: {t_sum.record_id} ---")
        print(f"Priority: {t_sum.priority}")
        print(f"\nTemplate Summary:")
        print(f"  {t_sum.summary}")
        
        if ai_sum.ai_enhanced:
            print(f"\nAI Summary (Gemini):")
            print(f"  {ai_sum.summary}")
        else:
            print(f"\n(AI not used - priority too low or fallback)")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
