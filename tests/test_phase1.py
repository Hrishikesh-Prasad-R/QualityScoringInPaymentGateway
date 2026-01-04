"""
Phase 1 Tests: Input Contract (Layer 1) and Input Validation (Layer 2)

Updated for the comprehensive VISA transaction schema with nested objects.
"""
import pytest
import os
import sys
import tempfile
import json
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.layers.layer1_input_contract import InputContractLayer
from src.layers.layer2_input_validation import InputValidationLayer
from src.models.schema import (
    SchemaManifest, 
    ColumnDefinition, 
    QualityThresholds, 
    DataType,
    create_default_transaction_schema,
    VisaTransaction,
    parse_visa_transaction,
    flatten_transactions,
)
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
        "ip_address": "103.xxx.xx.xx",
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


class TestVisaTransactionModel:
    """Tests for VisaTransaction Pydantic model."""
    
    def test_parse_sample_transaction(self):
        """Test that sample transaction parses correctly."""
        txn = parse_visa_transaction(SAMPLE_VISA_TRANSACTION)
        
        assert txn.transaction.transaction_id == "txn_00000001"
        assert txn.transaction.amount == 4000
        assert txn.card.network == "VISA"
        assert txn.merchant.country == "IN"
        assert txn.fraud.risk_score == 32
    
    def test_flatten_transaction(self):
        """Test that transaction flattens correctly."""
        txn = parse_visa_transaction(SAMPLE_VISA_TRANSACTION)
        flat = txn.flatten()
        
        assert "txn_transaction_id" in flat
        assert "card_network" in flat
        assert "merchant_country" in flat
        assert "customer_billing_address_city" in flat
        assert flat["txn_amount"] == 4000
        assert flat["fraud_risk_score"] == 32
    
    def test_flatten_multiple_transactions(self):
        """Test flattening multiple transactions."""
        txn2 = SAMPLE_VISA_TRANSACTION.copy()
        txn2["transaction"] = SAMPLE_VISA_TRANSACTION["transaction"].copy()
        txn2["transaction"]["transaction_id"] = "txn_00000002"
        
        flattened = flatten_transactions([SAMPLE_VISA_TRANSACTION, txn2])
        
        assert len(flattened) == 2
        assert flattened[0]["txn_transaction_id"] == "txn_00000001"
        assert flattened[1]["txn_transaction_id"] == "txn_00000002"


class TestLayer1InputContract:
    """Tests for Layer 1: Input Contract validation."""
    
    def test_default_schema_passes(self):
        """Test that default schema validates successfully."""
        layer = InputContractLayer()
        result = layer.validate_schema_manifest(use_default=True)
        
        assert result.status == LayerStatus.PASSED
        assert result.can_continue is True
        assert result.checks_passed == result.checks_performed
        assert layer.get_schema() is not None
    
    def test_no_schema_provided_fails(self):
        """Test that missing schema causes CONTRACT_VIOLATION."""
        layer = InputContractLayer()
        result = layer.validate_schema_manifest()
        
        assert result.status == LayerStatus.FAILED
        assert result.can_continue is False
        assert len(result.issues) > 0
        assert result.issues[0]["code"] == "SCHEMA_NOT_PROVIDED"
    
    def test_valid_json_schema(self):
        """Test that a valid JSON schema file passes."""
        layer = InputContractLayer()
        
        schema_dict = {
            "name": "test_schema",
            "version": "1.0",
            "use_nested_schema": True,
            "required_sections": ["transaction", "card"],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_dict, f)
            temp_path = f.name
        
        try:
            result = layer.validate_schema_manifest(schema_manifest_path=temp_path)
            assert result.status == LayerStatus.PASSED
            assert result.can_continue is True
        finally:
            os.unlink(temp_path)
    
    def test_invalid_json_fails(self):
        """Test that invalid JSON causes failure."""
        layer = InputContractLayer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            result = layer.validate_schema_manifest(schema_manifest_path=temp_path)
            assert result.status == LayerStatus.FAILED
            assert result.issues[0]["code"] == "INVALID_JSON"
        finally:
            os.unlink(temp_path)


class TestLayer2InputValidation:
    """Tests for Layer 2: Input Validation with VISA schema."""
    
    @pytest.fixture
    def schema(self):
        """Get default schema for testing."""
        return create_default_transaction_schema()
    
    def test_single_json_transaction_passes(self, schema):
        """Test that single JSON transaction validates."""
        layer = InputValidationLayer(schema)
        result = layer.validate(json_data=SAMPLE_VISA_TRANSACTION)
        
        assert result.status == LayerStatus.PASSED
        assert result.can_continue is True
        assert layer.get_dataframe() is not None
        assert len(layer.get_dataframe()) == 1
    
    def test_multiple_json_transactions_pass(self, schema):
        """Test that multiple JSON transactions validate."""
        txn2 = SAMPLE_VISA_TRANSACTION.copy()
        txn2["transaction"] = SAMPLE_VISA_TRANSACTION["transaction"].copy()
        txn2["transaction"]["transaction_id"] = "txn_00000002"
        
        layer = InputValidationLayer(schema)
        result = layer.validate(json_data=[SAMPLE_VISA_TRANSACTION, txn2])
        
        assert result.status == LayerStatus.PASSED
        assert len(layer.get_dataframe()) == 2
    
    def test_json_file_validation_passes(self, schema):
        """Test JSON file validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([SAMPLE_VISA_TRANSACTION], f)
            temp_path = f.name
        
        try:
            layer = InputValidationLayer(schema)
            result = layer.validate(file_path=temp_path)
            
            assert result.status == LayerStatus.PASSED
            assert layer.get_file_hash() is not None
        finally:
            os.unlink(temp_path)
    
    def test_flattened_dataframe_has_correct_columns(self, schema):
        """Test that flattened DataFrame has expected columns."""
        layer = InputValidationLayer(schema)
        layer.validate(json_data=SAMPLE_VISA_TRANSACTION)
        
        df = layer.get_dataframe()
        
        # Check key flattened columns exist
        assert "txn_transaction_id" in df.columns
        assert "txn_amount" in df.columns
        assert "card_network" in df.columns
        assert "merchant_country" in df.columns
        assert "fraud_risk_score" in df.columns
    
    def test_file_not_found_fails(self, schema):
        """Test that non-existent file fails."""
        layer = InputValidationLayer(schema)
        result = layer.validate(file_path="/non/existent/path.json")
        
        assert result.status == LayerStatus.FAILED
        assert result.issues[0]["code"] == "FILE_NOT_FOUND"
    
    def test_no_data_provided_fails(self, schema):
        """Test that no data provided fails."""
        layer = InputValidationLayer(schema)
        result = layer.validate()
        
        assert result.status == LayerStatus.FAILED
        assert result.issues[0]["code"] == "NO_DATA_PROVIDED"


class TestPhase1Integration:
    """Integration tests for Phase 1 (Layers 1 + 2 together)."""
    
    def test_full_phase1_flow_json(self):
        """Test complete Phase 1 flow with JSON data."""
        # Step 1: Layer 1 - Contract validation
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        
        assert result1.status == LayerStatus.PASSED
        assert result1.can_continue is True
        
        schema = layer1.get_schema()
        assert schema is not None
        assert schema.use_nested_schema is True
        
        # Step 2: Layer 2 - Data validation with JSON
        layer2 = InputValidationLayer(schema)
        result2 = layer2.validate(json_data=SAMPLE_VISA_TRANSACTION)
        
        assert result2.status == LayerStatus.PASSED
        assert result2.can_continue is True
        
        df = layer2.get_dataframe()
        assert df is not None
        assert len(df) == 1
        assert "txn_transaction_id" in df.columns
    
    def test_full_phase1_flow_multiple_transactions(self):
        """Test Phase 1 with multiple transactions."""
        # Generate multiple transactions
        transactions = []
        for i in range(10):
            txn = SAMPLE_VISA_TRANSACTION.copy()
            txn["transaction"] = SAMPLE_VISA_TRANSACTION["transaction"].copy()
            txn["transaction"]["transaction_id"] = f"txn_{i:08d}"
            transactions.append(txn)
        
        # Layer 1
        layer1 = InputContractLayer()
        result1 = layer1.validate_schema_manifest(use_default=True)
        assert result1.status == LayerStatus.PASSED
        
        # Layer 2
        layer2 = InputValidationLayer(layer1.get_schema())
        result2 = layer2.validate(json_data=transactions)
        
        assert result2.status == LayerStatus.PASSED
        assert len(layer2.get_dataframe()) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
