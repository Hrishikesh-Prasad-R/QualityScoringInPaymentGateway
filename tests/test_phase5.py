"""
Phase 5 Tests: Output & Decision (Layers 5-9)

Tests for output contract, stability, conflict detection, confidence bands, and decision gate.
"""
import pytest
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.layers import *
from src.models.schema import flatten_transactions
from src.config import LayerStatus, Action


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


def run_layers_1_to_4(transactions):
    """Run Layers 1-4.5 and return all outputs."""
    layer1 = InputContractLayer()
    layer1.validate_schema_manifest(use_default=True)
    
    layer2 = InputValidationLayer(layer1.get_schema())
    layer2.validate(json_data=transactions)
    
    layer3 = FeatureExtractionLayer()
    layer3.extract_features(layer2.get_dataframe())
    
    layer41 = StructuralIntegrityLayer()
    result41 = layer41.validate(layer2.get_dataframe(), layer3.get_features())
    
    layer42 = FieldComplianceLayer()
    result42 = layer42.score(layer2.get_dataframe(), layer3.get_features(), layer41.get_valid_indices())
    
    layer43 = SemanticValidationLayer()
    result43 = layer43.validate(layer2.get_dataframe(), layer3.get_features(), layer41.get_valid_indices())
    
    layer44 = AnomalyDetectionLayer()
    result44 = layer44.detect(layer3.get_features(), layer41.get_valid_indices())
    
    layer45 = GenAISummarizationLayer()
    result45 = layer45.summarize(
        layer2.get_dataframe(),
        layer3.get_features(),
        layer42.get_dqs_dataframe(),
        layer44.get_anomaly_dataframe(),
        layer43.get_validation_results(),
    )
    
    return {
        "dataframe": layer2.get_dataframe(),
        "features": layer3.get_features(),
        "dqs_df": layer42.get_dqs_dataframe(),
        "anomaly_df": layer44.get_anomaly_dataframe(),
        "summaries": layer45.get_summaries(),
        "structural_results": layer41.get_validation_results(),
        "semantic_results": layer43.get_validation_results(),
        "layer_results": {
            4.1: result41,
            4.2: result42,
            4.3: result43,
            4.4: result44,
            4.5: result45,
        },
    }


class TestLayer5OutputContract:
    """Tests for Layer 5: Output Contract."""
    
    def test_output_contract_structures_data(self):
        """Test that output contract structures data correctly."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        result = layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
            structural_results=outputs["structural_results"],
            semantic_results=outputs["semantic_results"],
        )
        
        assert result.status == LayerStatus.PASSED
        assert layer5.get_batch_payload() is not None
        assert len(layer5.get_record_payloads()) == 1
    
    def test_record_payload_has_required_fields(self):
        """Test that record payload has all required fields."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
        )
        
        payload = layer5.get_record_payloads()[0]
        
        assert payload.record_id is not None
        assert payload.dqs_base > 0
        assert payload.priority in ["critical", "high", "medium", "low", "none"]


class TestLayer6Stability:
    """Tests for Layer 6: Stability & Consistency."""
    
    def test_stability_validation(self):
        """Test stability validation."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
        )
        
        layer6 = StabilityConsistencyLayer()
        result = layer6.validate(layer5.get_record_payloads())
        
        assert result.status == LayerStatus.PASSED
        assert layer6.get_stability_metrics() is not None


class TestLayer7Conflict:
    """Tests for Layer 7: Conflict Detection."""
    
    def test_conflict_detection(self):
        """Test conflict detection."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
        )
        
        layer7 = ConflictDetectionLayer()
        result = layer7.detect(layer5.get_record_payloads(), outputs["dqs_df"])
        
        assert result.can_continue is True


class TestLayer8Confidence:
    """Tests for Layer 8: Confidence Band."""
    
    def test_confidence_assessment(self):
        """Test confidence band assessment."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
        )
        
        layer8 = ConfidenceBandLayer()
        result = layer8.assess(layer5.get_record_payloads())
        
        assert result.status == LayerStatus.PASSED
        assert len(layer8.get_assessments()) == 1
        
        assessment = layer8.get_assessments()[0]
        assert assessment.confidence_band.value in ["HIGH", "MEDIUM", "LOW"]


class TestLayer9Decision:
    """Tests for Layer 9: Decision Gate."""
    
    def test_decision_gate(self):
        """Test decision gate produces action."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
        )
        
        layer8 = ConfidenceBandLayer()
        layer8.assess(layer5.get_record_payloads())
        
        layer9 = DecisionGateLayer()
        result = layer9.decide(layer5.get_record_payloads(), layer8.get_assessments())
        
        assert result.status == LayerStatus.PASSED
        assert len(layer9.get_decisions()) == 1
        
        decision = layer9.get_decisions()[0]
        assert decision.action in [Action.SAFE_TO_USE, Action.REVIEW_REQUIRED, Action.ESCALATE, Action.NO_ACTION]
    
    def test_clean_record_is_safe(self):
        """Test that clean record gets SAFE_TO_USE."""
        outputs = run_layers_1_to_4(SAMPLE_VISA_TRANSACTION)
        
        layer5 = OutputContractLayer()
        layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
        )
        
        layer8 = ConfidenceBandLayer()
        layer8.assess(layer5.get_record_payloads())
        
        layer9 = DecisionGateLayer()
        layer9.decide(layer5.get_record_payloads(), layer8.get_assessments())
        
        decision = layer9.get_decisions()[0]
        # Clean transaction should be SAFE_TO_USE or REVIEW at worst
        assert decision.action in [Action.SAFE_TO_USE, Action.REVIEW_REQUIRED]


class TestPhase5Integration:
    """Integration tests for Phase 5 (full pipeline)."""
    
    def test_full_pipeline(self):
        """Test complete 11-layer pipeline."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=20, random_seed=42)
        outputs = run_layers_1_to_4(transactions)
        
        # Layer 5
        layer5 = OutputContractLayer()
        result5 = layer5.validate_and_structure(
            layer_results=outputs["layer_results"],
            dataframe=outputs["dataframe"],
            features_df=outputs["features"],
            dqs_df=outputs["dqs_df"],
            anomaly_df=outputs["anomaly_df"],
            summaries=outputs["summaries"],
            structural_results=outputs["structural_results"],
            semantic_results=outputs["semantic_results"],
        )
        assert result5.can_continue is True
        
        # Layer 6
        layer6 = StabilityConsistencyLayer()
        result6 = layer6.validate(layer5.get_record_payloads())
        assert result6.can_continue is True
        
        # Layer 7
        layer7 = ConflictDetectionLayer()
        result7 = layer7.detect(layer5.get_record_payloads(), outputs["dqs_df"])
        assert result7.can_continue is True
        
        # Layer 8
        layer8 = ConfidenceBandLayer()
        result8 = layer8.assess(
            layer5.get_record_payloads(),
            layer6.get_consistency_flags(),
            layer7.get_conflicts(),
            layer6.get_stability_metrics().consistency_score if layer6.get_stability_metrics() else 100,
        )
        assert result8.can_continue is True
        
        # Layer 9
        layer9 = DecisionGateLayer()
        result9 = layer9.decide(
            layer5.get_record_payloads(),
            layer8.get_assessments(),
            batch_id=layer5.get_batch_payload().batch_id,
        )
        assert result9.can_continue is True
        
        # Verify outputs
        assert result9.details["safe_count"] >= 0
        assert result9.details["review_count"] >= 0
        assert result9.details["escalate_count"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
