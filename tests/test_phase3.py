"""
Phase 3 Tests: Model Inference - Deterministic (Layers 4.1-4.3)

Tests for structural integrity, field compliance, and semantic validation.
"""
import pytest
import os
import sys
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.layers.layer1_input_contract import InputContractLayer
from src.layers.layer2_input_validation import InputValidationLayer
from src.layers.layer3_feature_extraction import FeatureExtractionLayer
from src.layers.layer4_1_structural import StructuralIntegrityLayer
from src.layers.layer4_2_field_compliance import FieldComplianceLayer
from src.layers.layer4_3_semantic import SemanticValidationLayer
from src.models.schema import create_default_transaction_schema, flatten_transactions
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


def get_processed_data():
    """Run Layers 1-3 and return dataframe + features."""
    layer1 = InputContractLayer()
    layer1.validate_schema_manifest(use_default=True)
    
    layer2 = InputValidationLayer(layer1.get_schema())
    layer2.validate(json_data=SAMPLE_VISA_TRANSACTION)
    
    layer3 = FeatureExtractionLayer()
    layer3.extract_features(layer2.get_dataframe())
    
    return layer2.get_dataframe(), layer3.get_features()


class TestLayer41StructuralIntegrity:
    """Tests for Layer 4.1: Structural Integrity."""
    
    def test_valid_record_passes(self):
        """Test that valid record passes structural checks."""
        df, features = get_processed_data()
        
        layer = StructuralIntegrityLayer()
        result = layer.validate(df, features)
        
        assert result.status == LayerStatus.PASSED
        assert result.can_continue is True
        assert len(layer.get_rejected_indices()) == 0
    
    def test_duplicate_pk_rejected(self):
        """Test that duplicate primary keys are rejected."""
        # Create duplicate transactions
        txn1 = SAMPLE_VISA_TRANSACTION.copy()
        txn2 = SAMPLE_VISA_TRANSACTION.copy()  # Same ID
        
        flattened = flatten_transactions([txn1, txn2])
        df = pd.DataFrame(flattened)
        df.columns = [col.lower().strip() for col in df.columns]
        
        layer = StructuralIntegrityLayer()
        result = layer.validate(df)
        
        # Should flag one as duplicate
        assert result.details["rejected_records"] >= 1
    
    def test_missing_required_field_rejected(self):
        """Test that missing required field causes rejection."""
        df, features = get_processed_data()
        
        # Remove required field
        df = df.copy()
        df["txn_transaction_id"] = None
        
        layer = StructuralIntegrityLayer()
        result = layer.validate(df)
        
        assert len(layer.get_rejected_indices()) > 0


class TestLayer42FieldCompliance:
    """Tests for Layer 4.2: Field-Level Compliance."""
    
    def test_valid_record_scores_high(self):
        """Test that valid record gets high DQS score."""
        df, features = get_processed_data()
        
        layer = FieldComplianceLayer()
        result = layer.score(df, features)
        
        assert result.status == LayerStatus.PASSED
        assert result.details["dqs_mean"] > 80
    
    def test_all_7_dimensions_scored(self):
        """Test that all 7 dimensions are scored."""
        df, features = get_processed_data()
        
        layer = FieldComplianceLayer()
        layer.score(df, features)
        
        scores_df = layer.get_dqs_dataframe()
        dimension_cols = [col for col in scores_df.columns if col.startswith("dim_")]
        
        assert len(dimension_cols) == 7
        assert "dim_completeness" in scores_df.columns
        assert "dim_accuracy" in scores_df.columns
        assert "dim_validity" in scores_df.columns
        assert "dim_uniqueness" in scores_df.columns
        assert "dim_consistency" in scores_df.columns
        assert "dim_timeliness" in scores_df.columns
        assert "dim_integrity" in scores_df.columns
    
    def test_low_quality_flagged_for_review(self):
        """Test that low quality records are flagged."""
        df, features = get_processed_data()
        
        # Corrupt some data
        features = features.copy()
        features["fraud_risk_score"] = 999  # Invalid
        
        layer = FieldComplianceLayer()
        result = layer.score(df, features)
        
        # Should have some dimension failures
        assert result.details["dqs_mean"] < 100


class TestLayer43SemanticValidation:
    """Tests for Layer 4.3: Semantic Validation."""
    
    def test_valid_record_passes_rules(self):
        """Test that valid record passes business rules."""
        df, features = get_processed_data()
        
        layer = SemanticValidationLayer()
        result = layer.validate(df, features)
        
        assert result.status == LayerStatus.PASSED
        assert result.details["records_rejected"] == 0
    
    def test_negative_amount_rejected(self):
        """Test BR001: Negative amount causes rejection."""
        df, features = get_processed_data()
        
        features = features.copy()
        features["txn_amount"] = -100  # Negative
        
        layer = SemanticValidationLayer()
        result = layer.validate(df, features)
        
        assert len(layer.get_rejected_indices()) > 0
    
    def test_settlement_math_violation(self):
        """Test BR002: Settlement math violation."""
        df, features = get_processed_data()
        
        df = df.copy()
        df["settlement_net_amount"] = 1000  # Wrong value (should be 3956)
        
        layer = SemanticValidationLayer()
        result = layer.validate(df, features)
        
        # Should have critical violations
        validations = layer.get_validation_results()
        has_settlement_violation = any(
            any(v.rule_id == "BR002" for v in r.critical_violations)
            for r in validations
        )
        assert has_settlement_violation
    
    def test_12_rules_evaluated(self):
        """Test that all 12 business rules are evaluated."""
        df, features = get_processed_data()
        
        layer = SemanticValidationLayer()
        result = layer.validate(df, features)
        
        assert result.details["rules_evaluated"] == 12


class TestPhase3Integration:
    """Integration tests for Phase 3 (Layers 1-3 + 4.1-4.3)."""
    
    def test_full_phase3_flow(self):
        """Test complete Phase 3 flow."""
        # Layers 1-2
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        assert result1.status == LayerStatus.PASSED
        
        layer2 = InputValidationLayer(layer1.get_schema())
        result2 = layer2.validate(json_data=SAMPLE_VISA_TRANSACTION)
        assert result2.status == LayerStatus.PASSED
        
        # Layer 3
        layer3 = FeatureExtractionLayer()
        result3 = layer3.extract_features(layer2.get_dataframe())
        assert result3.status == LayerStatus.PASSED
        
        # Layer 4.1
        layer41 = StructuralIntegrityLayer()
        result41 = layer41.validate(layer2.get_dataframe(), layer3.get_features())
        assert result41.can_continue is True
        
        # Layer 4.2
        layer42 = FieldComplianceLayer()
        result42 = layer42.score(
            layer2.get_dataframe(),
            layer3.get_features(),
            layer41.get_valid_indices(),
        )
        assert result42.can_continue is True
        
        # Layer 4.3
        layer43 = SemanticValidationLayer()
        result43 = layer43.validate(
            layer2.get_dataframe(),
            layer3.get_features(),
            layer41.get_valid_indices(),
        )
        assert result43.can_continue is True
    
    def test_with_generated_data(self):
        """Test Phase 3 with generated data."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=50, random_seed=42)
        
        # Layers 1-2
        layer1 = InputContractLayer()
        layer1.validate_schema_manifest(use_default=True)
        
        layer2 = InputValidationLayer(layer1.get_schema())
        layer2.validate(json_data=transactions)
        
        # Layer 3
        layer3 = FeatureExtractionLayer()
        layer3.extract_features(layer2.get_dataframe())
        
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
        
        # Summary stats
        assert result41.details["total_records"] == 50
        assert result42.details["records_scored"] > 0
        assert result43.details["records_validated"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
