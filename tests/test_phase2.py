"""
Phase 2 Tests: Feature Extraction (Layer 3)

Tests for the 35-feature extraction layer working with VISA transaction data.
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
from src.layers.layer3_feature_extraction import FeatureExtractionLayer, REFERENCE_PARAMS
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


def create_test_dataframe():
    """Create a flattened test DataFrame from sample transaction."""
    flattened = flatten_transactions([SAMPLE_VISA_TRANSACTION])
    return pd.DataFrame(flattened)


class TestLayer3FeatureExtraction:
    """Tests for Layer 3: Feature Extraction."""
    
    def test_feature_extraction_passes(self):
        """Test that feature extraction completes successfully."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        result = layer.extract_features(df)
        
        assert result.status == LayerStatus.PASSED
        assert result.can_continue is True
        assert layer.get_features() is not None
    
    def test_all_35_features_extracted(self):
        """Test that all 35 expected features are extracted."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        
        features = layer.get_features()
        assert len(features.columns) == 35
        assert list(features.columns) == layer.FEATURE_NAMES
    
    def test_transaction_features(self):
        """Test transaction feature extraction."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        # Amount should be 4000
        assert features["txn_amount"].iloc[0] == 4000
        
        # Status should be encoded as 0 (approved)
        assert features["txn_status_encoded"].iloc[0] == 0
        
        # Hour should be 18 (from timestamp)
        assert features["txn_hour"].iloc[0] == 18
    
    def test_card_features(self):
        """Test card feature extraction."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        # Network should be VISA (0)
        assert features["card_network_encoded"].iloc[0] == 0
        
        # Card type should be credit (0)
        assert features["card_type_encoded"].iloc[0] == 0
        
        # BIN first 2 should be 41
        assert features["card_bin_first2"].iloc[0] == 41
        
        # Domestic issuer (HDFC) should be 1
        assert features["card_is_domestic_issuer"].iloc[0] == 1
    
    def test_merchant_features(self):
        """Test merchant feature extraction."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        # MCC 5812 -> first 2 = 58 -> restaurant
        assert features["merchant_mcc_first2"].iloc[0] == 58
        
        # Country IN should be domestic
        assert features["merchant_is_domestic"].iloc[0] == 1
        
        # Country risk for IN should be 0.1
        assert features["merchant_country_risk"].iloc[0] == 0.1
    
    def test_customer_features(self):
        """Test customer feature extraction."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        # Has email
        assert features["customer_has_email"].iloc[0] == 1
        
        # Has phone
        assert features["customer_has_phone"].iloc[0] == 1
        
        # Address match (both Bengaluru)
        assert features["customer_address_match"].iloc[0] == 1
        
        # IP is domestic (103.x.x.x)
        assert features["customer_ip_is_domestic"].iloc[0] == 1
    
    def test_fraud_features(self):
        """Test fraud feature extraction."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        # Risk score should be 32
        assert features["fraud_risk_score"].iloc[0] == 32
        
        # Risk level low = 0
        assert features["fraud_risk_level_encoded"].iloc[0] == 0
        
        # Velocity passed
        assert features["fraud_velocity_passed"].iloc[0] == 1
    
    def test_settlement_features(self):
        """Test settlement feature extraction."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        # Fee ratio = (32 + 12) / 4000 = 0.011
        assert abs(features["settlement_fee_ratio"].iloc[0] - 0.011) < 0.001
        
        # Amount match (gross = txn amount)
        assert features["settlement_amount_match"].iloc[0] == 1
        
        # Net ratio = 3956 / 4000 = 0.989
        assert abs(features["settlement_net_ratio"].iloc[0] - 0.989) < 0.001
    
    def test_no_nan_in_features(self):
        """Test that features have no NaN values."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        nan_count = features.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"
    
    def test_feature_stats(self):
        """Test feature statistics retrieval."""
        df = create_test_dataframe()
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        
        stats = layer.get_feature_stats()
        assert stats["total_features"] == 35
        assert stats["row_count"] == 1
        assert "feature_means" in stats


class TestPhase2Integration:
    """Integration tests for Phase 2 (Layers 1 + 2 + 3 together)."""
    
    def test_full_phase2_flow(self):
        """Test complete Phase 2 flow: Schema → Validation → Features."""
        # Layer 1: Contract
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        assert result1.status == LayerStatus.PASSED
        schema = layer1.get_schema()
        
        # Layer 2: Validation
        layer2 = InputValidationLayer(schema)
        result2 = layer2.validate(json_data=SAMPLE_VISA_TRANSACTION)
        assert result2.status == LayerStatus.PASSED
        validated_df = layer2.get_dataframe()
        
        # Layer 3: Feature Extraction
        layer3 = FeatureExtractionLayer()
        result3 = layer3.extract_features(validated_df)
        
        assert result3.status == LayerStatus.PASSED
        assert result3.can_continue is True
        
        features = layer3.get_features()
        assert features is not None
        assert len(features.columns) == 35
    
    def test_multiple_transactions_flow(self):
        """Test Phase 2 with multiple transactions."""
        # Generate multiple transactions
        transactions = []
        for i in range(10):
            txn = SAMPLE_VISA_TRANSACTION.copy()
            txn["transaction"] = SAMPLE_VISA_TRANSACTION["transaction"].copy()
            txn["transaction"]["transaction_id"] = f"txn_{i:08d}"
            txn["transaction"]["amount"] = 1000 * (i + 1)
            transactions.append(txn)
        
        # Layer 1
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        assert result1.status == LayerStatus.PASSED
        
        # Layer 2
        layer2 = InputValidationLayer(layer1.get_schema())
        result2 = layer2.validate(json_data=transactions)
        assert result2.status == LayerStatus.PASSED
        
        # Layer 3
        layer3 = FeatureExtractionLayer()
        result3 = layer3.extract_features(layer2.get_dataframe())
        
        assert result3.status == LayerStatus.PASSED
        features = layer3.get_features()
        assert len(features) == 10
        
        # Verify amount varies
        amounts = features["txn_amount"].tolist()
        assert amounts == [1000 * (i + 1) for i in range(10)]
    
    def test_with_generated_data(self):
        """Test Phase 2 with data from generator."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=50, random_seed=42)
        
        # Layer 1
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        
        # Layer 2
        layer2 = InputValidationLayer(layer1.get_schema())
        result2 = layer2.validate(json_data=transactions)
        
        # Layer 3
        layer3 = FeatureExtractionLayer()
        result3 = layer3.extract_features(layer2.get_dataframe())
        
        assert result3.status == LayerStatus.PASSED
        features = layer3.get_features()
        assert len(features) == 50
        assert len(features.columns) == 35


class TestFeatureRobustness:
    """Test robustness and edge cases."""
    
    def test_missing_optional_fields(self):
        """Test extraction with missing optional fields."""
        txn = SAMPLE_VISA_TRANSACTION.copy()
        # Remove optional sections
        del txn["authentication"]
        del txn["settlement"]
        del txn["business_metadata"]
        
        flattened = flatten_transactions([txn])
        df = pd.DataFrame(flattened)
        
        layer = FeatureExtractionLayer()
        result = layer.extract_features(df)
        
        # Should still pass with defaults
        assert result.can_continue is True
        features = layer.get_features()
        assert len(features.columns) == 35
    
    def test_high_risk_transaction(self):
        """Test feature extraction for high-risk transaction."""
        txn = SAMPLE_VISA_TRANSACTION.copy()
        txn["fraud"] = {
            "risk_score": 85,
            "risk_level": "high",
            "velocity_check": "fail",
            "geo_check": "fail"
        }
        
        flattened = flatten_transactions([txn])
        df = pd.DataFrame(flattened)
        
        layer = FeatureExtractionLayer()
        layer.extract_features(df)
        features = layer.get_features()
        
        assert features["fraud_risk_score"].iloc[0] == 85
        assert features["fraud_risk_level_encoded"].iloc[0] == 2  # high
        assert features["fraud_velocity_passed"].iloc[0] == 0
        assert features["fraud_geo_passed"].iloc[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
