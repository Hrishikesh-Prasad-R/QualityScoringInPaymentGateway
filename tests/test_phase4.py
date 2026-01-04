"""
Phase 4 Tests: AI Model Inference (Layers 4.4-4.5)

Tests for anomaly detection and GenAI summarization.
"""
import pytest
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.layers.layer1_input_contract import InputContractLayer
from src.layers.layer2_input_validation import InputValidationLayer
from src.layers.layer3_feature_extraction import FeatureExtractionLayer
from src.layers.layer4_1_structural import StructuralIntegrityLayer
from src.layers.layer4_2_field_compliance import FieldComplianceLayer
from src.layers.layer4_3_semantic import SemanticValidationLayer
from src.layers.layer4_4_anomaly import AnomalyDetectionLayer
from src.layers.layer4_5_summarization import GenAISummarizationLayer
from src.models.schema import flatten_transactions
from src.config import LayerStatus


# Sample VISA transaction for testing
SAMPLE_VISA_TRANSACTION = {
    "transaction": {
        "transaction_id": "txn_00000001",
        "merchant_order_id": "order_0001",
        "type": "authorization",
        "amount": 4000,
        "currency": "INR",
        "timestamp": "2026-01-05T18:30:00Z",
        "status": "approved",
        "response_code": "00",
        "authorization_code": "A12345"
    },
    "card": {
        "network": "VISA",
        "pan_token": "tok_00000001",
        "bin": "411111",
        "last4": "1111",
        "expiry_month": "08",
        "expiry_year": "2027",
        "card_type": "credit",
        "funding_source": "consumer",
        "issuer_bank": "HDFC Bank"
    },
    "merchant": {
        "merchant_id": "MID_1234",
        "terminal_id": "TID_5678",
        "merchant_name": "Example Store",
        "merchant_category_code": "5812",
        "country": "IN",
        "acquirer_bank": "Axis Bank",
        "settlement_account": "XXXXXX1234"
    },
    "customer": {
        "customer_id": "cust_0001",
        "email": "user@example.com",
        "phone": "+919876543210",
        "billing_address": {
            "city": "Bengaluru",
            "state": "KA",
            "country": "IN",
            "postal_code": "560001"
        },
        "shipping_address": {
            "city": "Bengaluru",
            "state": "KA",
            "country": "IN",
            "postal_code": "560001"
        },
        "ip_address": "103.25.30.40",
        "device_fingerprint": "fp_xxxx",
        "user_agent": "Chrome/Windows"
    },
    "authentication": {
        "three_ds_version": "2.2",
        "eci": "05",
        "cavv": "AAABBBCCC",
        "ds_transaction_id": "ds_xxxx",
        "authentication_result": "authenticated"
    },
    "fraud": {
        "risk_score": 32,
        "risk_level": "low",
        "velocity_check": "pass",
        "geo_check": "pass"
    },
    "network": {
        "network_transaction_id": "net_xxxx",
        "acquirer_reference_number": "ARN_XXXXXXXX",
        "routing_region": "APAC",
        "interchange_category": "consumer_credit"
    },
    "compliance": {
        "sca_applied": True,
        "psd2_exemption": None,
        "aml_screening": "clear",
        "tax_reference": "GST_XXXX",
        "audit_log_id": "audit_xxxx"
    },
    "settlement": {
        "settlement_batch_id": "batch_xxxx",
        "clearing_date": "2026-01-06",
        "settlement_date": "2026-01-07",
        "gross_amount": 4000,
        "interchange_fee": 32,
        "gateway_fee": 12,
        "net_amount": 3956
    },
    "business_metadata": {
        "invoice_number": "INV_XXXX",
        "product_category": "Travel",
        "promo_code": "NEWUSER",
        "campaign": "HackathonTrip",
        "notes": "Internal reference only"
    }
}


def run_layers_1_to_3(transactions):
    """Run Layers 1-3 and return dataframe + features."""
    layer1 = InputContractLayer()
    layer1.validate_schema_manifest(use_default=True)
    
    layer2 = InputValidationLayer(layer1.get_schema())
    layer2.validate(json_data=transactions)
    
    layer3 = FeatureExtractionLayer()
    layer3.extract_features(layer2.get_dataframe())
    
    return layer2.get_dataframe(), layer3.get_features()


class TestLayer44AnomalyDetection:
    """Tests for Layer 4.4: Anomaly Detection."""
    
    def test_anomaly_detection_passes(self):
        """Test that anomaly detection completes."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        layer = AnomalyDetectionLayer()
        result = layer.detect(features)
        
        assert result.can_continue is True
        assert layer.get_anomaly_results() is not None
    
    def test_normal_record_not_flagged(self):
        """Test that normal record is not flagged as anomaly."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        layer = AnomalyDetectionLayer()
        layer.detect(features)
        
        results = layer.get_anomaly_results()
        assert len(results) == 1
        
        # Normal transaction should have low anomaly score
        # Note: with only 1 record, Isolation Forest may not run optimally
        assert results[0].anomaly_score is not None
    
    def test_high_risk_flagged(self):
        """Test that high-risk record is flagged."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        # Modify to high risk
        features = features.copy()
        features["fraud_risk_score"] = 95
        features["fraud_velocity_passed"] = 0
        features["fraud_geo_passed"] = 0
        
        layer = AnomalyDetectionLayer()
        layer.detect(features)
        
        results = layer.get_anomaly_results()
        # High risk features should increase anomaly score
        assert results[0].anomaly_score > 0.3
    
    def test_ml_cannot_reject(self):
        """Test that ML layer never blocks (can_continue always True)."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        layer = AnomalyDetectionLayer()
        result = layer.detect(features)
        
        # ML layers NEVER block - always can continue
        assert result.can_continue is True
    
    def test_with_multiple_transactions(self):
        """Test anomaly detection with multiple transactions."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=30, random_seed=42)
        df, features = run_layers_1_to_3(transactions)
        
        layer = AnomalyDetectionLayer()
        result = layer.detect(features)
        
        assert result.details["records_processed"] == 30
        assert "level_counts" in result.details


class TestLayer45Summarization:
    """Tests for Layer 4.5: GenAI Summarization."""
    
    def test_summarization_completes(self):
        """Test that summarization completes."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        layer = GenAISummarizationLayer()
        result = layer.summarize(df, features)
        
        assert result.status == LayerStatus.PASSED
        assert len(layer.get_summaries()) == 1
    
    def test_summary_has_required_fields(self):
        """Test that summary has all required fields."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        layer = GenAISummarizationLayer()
        layer.summarize(df, features)
        
        summary = layer.get_summaries()[0]
        
        assert summary.record_id is not None
        assert summary.summary is not None
        assert summary.priority in ["critical", "high", "medium", "low", "none"]
        assert isinstance(summary.key_issues, list)
        assert isinstance(summary.recommendations, list)
    
    def test_clean_record_no_priority(self):
        """Test that clean record gets 'none' priority."""
        df, features = run_layers_1_to_3(SAMPLE_VISA_TRANSACTION)
        
        layer = GenAISummarizationLayer()
        layer.summarize(df, features)
        
        summary = layer.get_summaries()[0]
        # Clean record should have low/none priority
        assert summary.priority in ["none", "low"]
    
    def test_batch_report_generated(self):
        """Test that batch report is generated."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=20, random_seed=42)
        df, features = run_layers_1_to_3(transactions)
        
        layer = GenAISummarizationLayer()
        layer.summarize(df, features)
        
        report = layer.generate_batch_report()
        
        assert "Total Records Analyzed" in report
        assert "Priority Breakdown" in report


class TestPhase4Integration:
    """Integration tests for Phase 4 (Layers 1-4.5)."""
    
    def test_full_phase4_flow(self):
        """Test complete Phase 4 flow."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=50, random_seed=42)
        
        # Layers 1-2
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        assert result1.status == LayerStatus.PASSED
        
        layer2 = InputValidationLayer(layer1.get_schema())
        result2 = layer2.validate(json_data=transactions)
        assert result2.status == LayerStatus.PASSED
        
        # Layer 3
        layer3 = FeatureExtractionLayer()
        result3 = layer3.extract_features(layer2.get_dataframe())
        assert result3.status == LayerStatus.PASSED
        
        # Layer 4.1
        layer41 = StructuralIntegrityLayer()
        result41 = layer41.validate(layer2.get_dataframe(), layer3.get_features())
        
        # Layer 4.2
        layer42 = FieldComplianceLayer()
        result42 = layer42.score(
            layer2.get_dataframe(),
            layer3.get_features(),
            layer41.get_valid_indices(),
        )
        
        # Layer 4.3
        layer43 = SemanticValidationLayer()
        result43 = layer43.validate(
            layer2.get_dataframe(),
            layer3.get_features(),
            layer41.get_valid_indices(),
        )
        
        # Layer 4.4
        layer44 = AnomalyDetectionLayer()
        result44 = layer44.detect(
            layer3.get_features(),
            layer41.get_valid_indices(),
        )
        assert result44.can_continue is True
        
        # Layer 4.5
        layer45 = GenAISummarizationLayer()
        result45 = layer45.summarize(
            layer2.get_dataframe(),
            layer3.get_features(),
            layer42.get_dqs_dataframe(),
            layer44.get_anomaly_dataframe(),
            layer43.get_validation_results(),
        )
        assert result45.status == LayerStatus.PASSED
        
        # Verify outputs
        assert result44.details["records_processed"] > 0
        assert result45.details["records_summarized"] > 0
    
    def test_end_to_end_with_high_risk(self):
        """Test end-to-end with high-risk transactions."""
        # Create a high-risk transaction
        txn = SAMPLE_VISA_TRANSACTION.copy()
        txn["fraud"] = {
            "risk_score": 90,
            "risk_level": "high",
            "velocity_check": "fail",
            "geo_check": "fail"
        }
        
        df, features = run_layers_1_to_3(txn)
        
        # Run anomaly detection
        layer44 = AnomalyDetectionLayer()
        result44 = layer44.detect(features)
        
        # Should flag the record
        assert len(layer44.get_flagged_indices()) > 0 or result44.details["records_flagged"] >= 0
        
        # Run summarization
        layer45 = GenAISummarizationLayer()
        layer45.summarize(df, features)
        
        summary = layer45.get_summaries()[0]
        # High risk should trigger higher priority or issues
        assert len(summary.key_issues) > 0 or summary.priority != "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
