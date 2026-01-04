"""
Phase 6 Tests: Responsibility & Logging (Layers 10-11)

Tests for responsibility boundary and logging/trace.
"""
import pytest
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.layers import *
from src.layers.layer10_responsibility import ResponsibilityBoundaryLayer, ResponsibilityOwner
from src.layers.layer11_logging import LoggingTraceLayer
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


def run_full_pipeline(transactions):
    """Run complete pipeline through Layer 9."""
    # Layers 1-3
    layer1 = InputContractLayer()
    result1 = layer1.validate_schema_manifest(use_default=True)
    
    layer2 = InputValidationLayer(layer1.get_schema())
    result2 = layer2.validate(json_data=transactions)
    
    layer3 = FeatureExtractionLayer()
    result3 = layer3.extract_features(layer2.get_dataframe())
    
    # Layers 4.1-4.5
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
    
    # Layers 5-9
    layer5 = OutputContractLayer()
    result5 = layer5.validate_and_structure(
        layer_results={4.1: result41, 4.2: result42, 4.3: result43, 4.4: result44, 4.5: result45},
        dataframe=layer2.get_dataframe(),
        features_df=layer3.get_features(),
        dqs_df=layer42.get_dqs_dataframe(),
        anomaly_df=layer44.get_anomaly_dataframe(),
        summaries=layer45.get_summaries(),
        structural_results=layer41.get_validation_results(),
        semantic_results=layer43.get_validation_results(),
    )
    
    layer6 = StabilityConsistencyLayer()
    result6 = layer6.validate(layer5.get_record_payloads())
    
    layer7 = ConflictDetectionLayer()
    result7 = layer7.detect(layer5.get_record_payloads(), layer42.get_dqs_dataframe())
    
    layer8 = ConfidenceBandLayer()
    result8 = layer8.assess(
        layer5.get_record_payloads(),
        layer6.get_consistency_flags(),
        layer7.get_conflicts(),
        layer6.get_stability_metrics().consistency_score if layer6.get_stability_metrics() else 100,
    )
    
    layer9 = DecisionGateLayer()
    result9 = layer9.decide(
        layer5.get_record_payloads(),
        layer8.get_assessments(),
        batch_id=layer5.get_batch_payload().batch_id,
    )
    
    # Collect all layer results
    all_results = {
        1: result1, 2: result2, 3: result3,
        4.1: result41, 4.2: result42, 4.3: result43, 4.4: result44, 4.5: result45,
        5: result5, 6: result6, 7: result7, 8: result8, 9: result9,
    }
    
    return {
        "decisions": layer9.get_decisions(),
        "batch_payload": layer5.get_batch_payload(),
        "layer_results": all_results,
        "total_records": len(layer2.get_dataframe()),
    }


class TestLayer10Responsibility:
    """Tests for Layer 10: Responsibility Boundary."""
    
    def test_responsibility_assignment(self):
        """Test that responsibility is assigned."""
        pipeline = run_full_pipeline(SAMPLE_VISA_TRANSACTION)
        
        layer10 = ResponsibilityBoundaryLayer()
        result = layer10.assign(pipeline["decisions"])
        
        assert result.status == LayerStatus.PASSED
        assert len(layer10.get_assignments()) == 1
    
    def test_safe_decision_is_automated(self):
        """Test that SAFE_TO_USE is automated."""
        pipeline = run_full_pipeline(SAMPLE_VISA_TRANSACTION)
        
        layer10 = ResponsibilityBoundaryLayer()
        layer10.assign(pipeline["decisions"])
        
        assignment = layer10.get_assignments()[0]
        
        # Clean transaction should be SAFE and AUTOMATED
        if pipeline["decisions"][0].action == Action.SAFE_TO_USE:
            assert assignment.owner == ResponsibilityOwner.SYSTEM_AUTOMATED
    
    def test_trace_id_generated(self):
        """Test that trace ID is generated."""
        pipeline = run_full_pipeline(SAMPLE_VISA_TRANSACTION)
        
        layer10 = ResponsibilityBoundaryLayer()
        layer10.assign(pipeline["decisions"])
        
        assignment = layer10.get_assignments()[0]
        assert assignment.trace_id is not None
        assert len(assignment.trace_id) > 0


class TestLayer11Logging:
    """Tests for Layer 11: Logging & Trace."""
    
    def test_logging_creates_traces(self):
        """Test that logging creates traces."""
        pipeline = run_full_pipeline(SAMPLE_VISA_TRANSACTION)
        
        layer10 = ResponsibilityBoundaryLayer()
        layer10.assign(pipeline["decisions"])
        
        layer11 = LoggingTraceLayer()
        result = layer11.log(
            pipeline["layer_results"],
            pipeline["decisions"],
            layer10.get_assignments(),
        )
        
        assert result.status == LayerStatus.PASSED
        assert len(layer11.get_record_traces()) == 1
    
    def test_execution_log_created(self):
        """Test that execution log is created."""
        pipeline = run_full_pipeline(SAMPLE_VISA_TRANSACTION)
        
        layer10 = ResponsibilityBoundaryLayer()
        layer10.assign(pipeline["decisions"])
        
        layer11 = LoggingTraceLayer()
        layer11.start_pipeline()
        layer11.log(
            pipeline["layer_results"],
            pipeline["decisions"],
            layer10.get_assignments(),
        )
        
        log = layer11.get_execution_log()
        assert log is not None
        assert log.processed_records == 1
    
    def test_json_export(self):
        """Test JSON export."""
        pipeline = run_full_pipeline(SAMPLE_VISA_TRANSACTION)
        
        layer10 = ResponsibilityBoundaryLayer()
        layer10.assign(pipeline["decisions"])
        
        layer11 = LoggingTraceLayer()
        layer11.log(
            pipeline["layer_results"],
            pipeline["decisions"],
            layer10.get_assignments(),
        )
        
        json_export = layer11.export_to_json()
        assert "batch_id" in json_export
        assert "execution_id" in json_export


class TestPhase6Integration:
    """Integration tests for Phase 6."""
    
    def test_full_pipeline_11_layers(self):
        """Test complete 11-layer pipeline."""
        from src.data_generator import generate_visa_transactions
        
        transactions = generate_visa_transactions(n_transactions=20, random_seed=42)
        pipeline = run_full_pipeline(transactions)
        
        # Layer 10
        layer10 = ResponsibilityBoundaryLayer()
        result10 = layer10.assign(
            pipeline["decisions"],
            batch_id=pipeline["batch_payload"].batch_id,
        )
        assert result10.status == LayerStatus.PASSED
        
        # Layer 11
        layer11 = LoggingTraceLayer()
        layer11.start_pipeline()
        result11 = layer11.log(
            pipeline["layer_results"],
            pipeline["decisions"],
            layer10.get_assignments(),
            batch_id=pipeline["batch_payload"].batch_id,
            total_records=pipeline["total_records"],
        )
        assert result11.status == LayerStatus.PASSED
        
        # Verify execution log
        log = layer11.get_execution_log()
        assert log.processed_records == 20
        assert log.total_records == 20
        
        # Generate report
        report = layer11.generate_execution_report()
        assert "PIPELINE EXECUTION REPORT" in report
        assert "ACTION SUMMARY" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
