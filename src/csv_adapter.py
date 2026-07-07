"""
CSV Adapter Module

Adapts non-standard CSV data to the required VISA schema format.
Provides graceful fallbacks and quality penalties for missing/incorrect fields.
"""
import csv
import io
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np


# Required fields for full quality score
REQUIRED_FIELDS = {
    "transaction": ["transaction_id", "amount", "currency", "timestamp"],
    "card": ["network", "card_type"],
    "merchant": ["merchant_id", "merchant_category_code", "country"],
    "customer": ["customer_id"],
    "fraud": ["risk_score"],
}

# Field mappings from common CSV column names to our schema
FIELD_MAPPINGS = {
    # Transaction fields
    "transaction_id": ["transaction_id", "txn_id", "id", "order_id", "ref_id", "reference"],
    "amount": ["amount", "txn_amount", "transaction_amount", "value", "total", "price"],
    "currency": ["currency", "currency_code", "ccy"],
    "timestamp": ["timestamp", "date", "datetime", "txn_date", "transaction_date", "created_at"],
    "status": ["status", "txn_status", "transaction_status", "state"],
    "type": ["type", "txn_type", "transaction_type"],
    
    # Card fields
    "network": ["network", "card_network", "scheme"],
    "card_type": ["card_type", "type", "funding_type"],
    "bin": ["bin", "card_bin"],
    "last4": ["last4", "last_4", "card_last4"],
    
    # Merchant fields
    "merchant_id": ["merchant_id", "mid", "store_id", "seller_id"],
    "terminal_id": ["terminal_id", "tid", "pos_id"],
    "merchant_name": ["merchant_name", "store_name", "seller_name", "name"],
    "merchant_category_code": ["mcc", "merchant_category_code", "category_code"],
    "country": ["country", "merchant_country", "location_country"],
    
    # Customer fields
    "customer_id": ["customer_id", "cust_id", "user_id", "buyer_id"],
    "email": ["email", "customer_email", "user_email"],
    "phone": ["phone", "mobile", "customer_phone"],
    "ip_address": ["ip_address", "ip", "client_ip"],
    
    # Fraud fields
    "risk_score": ["risk_score", "fraud_score", "risk"],
    "risk_level": ["risk_level", "fraud_level"],
}


def detect_csv_columns(headers: List[str]) -> Dict[str, str]:
    """
    Detect which CSV columns map to our schema fields.
    Returns mapping of our field names to CSV column names.
    """
    column_mapping = {}
    headers_lower = {h.lower().strip(): h for h in headers}
    
    for our_field, possible_names in FIELD_MAPPINGS.items():
        for name in possible_names:
            if name.lower() in headers_lower:
                column_mapping[our_field] = headers_lower[name.lower()]
                break
    
    return column_mapping


def calculate_schema_compliance(column_mapping: Dict[str, str]) -> Tuple[float, List[str]]:
    """
    Calculate how well the CSV adheres to our schema.
    Returns (compliance_score 0-100, list of missing fields).
    """
    missing_fields = []
    total_required = 0
    found_required = 0
    
    for section, fields in REQUIRED_FIELDS.items():
        for field in fields:
            total_required += 1
            if field in column_mapping:
                found_required += 1
            else:
                missing_fields.append(f"{section}.{field}")
    
    compliance_score = (found_required / total_required * 100) if total_required > 0 else 0
    return compliance_score, missing_fields


def convert_csv_row_to_visa(row: Dict[str, Any], column_mapping: Dict[str, str], row_index: int) -> Dict[str, Any]:
    """
    Convert a single CSV row to our VISA nested schema format.
    Missing fields get default values and penalties are tracked.
    """
    def get_value(field_name: str, default: Any = None) -> Any:
        csv_col = column_mapping.get(field_name)
        if csv_col and csv_col in row:
            val = row[csv_col]
            # Handle empty strings
            if val == "" or val is None:
                return default
            return val
        return default
    
    # Parse amount
    try:
        amount = float(get_value("amount", 0))
    except (ValueError, TypeError):
        amount = 0.0
    
    # Parse timestamp
    timestamp_str = get_value("timestamp", "")
    if timestamp_str:
        try:
            # Try various formats
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
                try:
                    dt = datetime.strptime(str(timestamp_str)[:19], fmt)
                    timestamp_str = dt.isoformat() + "Z"
                    break
                except ValueError:
                    continue
        except Exception:
            timestamp_str = datetime.now().isoformat() + "Z"
    else:
        timestamp_str = datetime.now().isoformat() + "Z"
    
    # Parse risk score
    try:
        risk_score = int(float(get_value("risk_score", 50)))
        risk_score = max(0, min(100, risk_score))
    except (ValueError, TypeError):
        risk_score = 50  # Default medium risk
    
    # Build VISA structure with defaults for missing fields
    txn = {
        "transaction": {
            "transaction_id": get_value("transaction_id", f"csv_row_{row_index:08d}"),
            "merchant_order_id": get_value("order_id", f"order_{row_index:06d}"),
            "type": get_value("type", "authorization"),
            "amount": amount,
            "currency": get_value("currency", "INR"),
            "timestamp": timestamp_str,
            "status": get_value("status", "pending"),
            "response_code": "00",
            "authorization_code": f"CSV{row_index:05d}",
        },
        "card": {
            "network": get_value("network", "UNKNOWN"),
            "pan_token": f"csv_tok_{row_index:08d}",
            "bin": get_value("bin", "000000"),
            "last4": get_value("last4", "0000"),
            "expiry_month": "12",
            "expiry_year": "2030",
            "card_type": get_value("card_type", "unknown"),
            "funding_source": "unknown",
            "issuer_bank": "Unknown Bank",
        },
        "merchant": {
            "merchant_id": get_value("merchant_id", f"CSV_MID_{row_index}"),
            "terminal_id": get_value("terminal_id", f"CSV_TID_{row_index}"),
            "merchant_name": get_value("merchant_name", f"CSV_Merchant_{row_index}"),
            "merchant_category_code": get_value("merchant_category_code", "0000"),
            "country": get_value("country", "XX"),
            "acquirer_bank": "Unknown Bank",
            "settlement_account": "XXXX0000",
        },
        "customer": {
            "customer_id": get_value("customer_id", f"csv_cust_{row_index}"),
            "email": get_value("email"),
            "phone": get_value("phone"),
            "billing_address": {
                "city": "Unknown",
                "state": "XX",
                "country": get_value("country", "XX"),
                "postal_code": "000000",
            },
            "shipping_address": {
                "city": "Unknown",
                "state": "XX",
                "country": get_value("country", "XX"),
                "postal_code": "000000",
            },
            "ip_address": get_value("ip_address", "0.0.0.0"),
            "device_fingerprint": f"csv_fp_{row_index}",
            "user_agent": "CSV Import",
        },
        "authentication": {
            "three_ds_version": "unknown",
            "eci": "00",
            "cavv": None,
            "ds_transaction_id": None,
            "authentication_result": "unknown",
        },
        "fraud": {
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 70 else ("medium" if risk_score > 40 else "low"),
            "velocity_check": "unknown",
            "geo_check": "unknown",
        },
        "network": {
            "network_transaction_id": f"csv_net_{row_index}",
            "acquirer_reference_number": f"CSV_ARN_{row_index}",
            "routing_region": "UNKNOWN",
            "interchange_category": "unknown",
        },
        "compliance": {
            "sca_applied": False,
            "psd2_exemption": None,
            "aml_screening": "unknown",
            "tax_reference": None,
            "audit_log_id": f"csv_audit_{row_index}",
        },
        "settlement": {
            "settlement_batch_id": f"csv_batch_{datetime.now().strftime('%Y%m%d')}",
            "clearing_date": None,
            "settlement_date": None,
            "gross_amount": amount,
            "interchange_fee": round(amount * 0.018, 2),
            "gateway_fee": round(amount * 0.003, 2),
            "net_amount": round(amount * 0.979, 2),
        },
        "business_metadata": {
            "invoice_number": f"CSV_{row_index:06d}",
            "product_category": "Unknown",
            "promo_code": None,
            "campaign": "CSV_Import",
            "notes": "Imported from non-standard CSV format",
        },
        # Mark as CSV import with compliance info
        "_csv_import": {
            "original_row": row_index,
            "is_adapted": True,
        }
    }
    
    return txn


def adapt_csv_to_visa(csv_content: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Adapt CSV content to VISA transaction format.
    
    Returns:
        Tuple of (transactions list, metadata dict with compliance info)
    """
    transactions = []
    
    # Parse CSV
    reader = csv.DictReader(io.StringIO(csv_content))
    headers = reader.fieldnames or []
    rows = list(reader)
    
    # Detect column mappings
    column_mapping = detect_csv_columns(headers)
    
    # Calculate compliance
    compliance_score, missing_fields = calculate_schema_compliance(column_mapping)
    
    # If compliance is low, attempt GenAI mapper to discover missing columns
    ai_mapped = False
    if compliance_score < 80 and os.environ.get("GEMINI_API_KEY"):
        from src.layers.layer1_2_schema_mapper import GenAISchemaMapper
        mapper = GenAISchemaMapper()
        ai_mapping = mapper.map_columns(headers, rows)
        if ai_mapping:
            # Merge mappings, giving priority to Gemini findings for unmapped fields
            for std_field, raw_col in ai_mapping.items():
                if std_field not in column_mapping:
                    column_mapping[std_field] = raw_col
            # Recalculate compliance
            compliance_score, missing_fields = calculate_schema_compliance(column_mapping)
            ai_mapped = True
    
    # Convert rows
    for i, row in enumerate(rows):
        txn = convert_csv_row_to_visa(row, column_mapping, i)
        transactions.append(txn)
    
    metadata = {
        "source": "csv_import",
        "compliance_score": compliance_score,
        "missing_fields": missing_fields,
        "mapped_fields": list(column_mapping.keys()),
        "original_columns": headers,
        "total_rows": len(transactions),
        "is_standard_format": compliance_score >= 80,
        "quality_penalty": max(0, (100 - compliance_score) / 2),  # Up to 50% penalty
        "warnings": [],
        "ai_mapped": ai_mapped
    }
    
    if ai_mapped:
        metadata["warnings"].append(
            f"Dynamic schema alignment active: GenAI successfully mapped raw columns "
            f"to standard fields. Compliance improved to {compliance_score:.0f}%."
        )
    elif compliance_score < 80:
        metadata["warnings"].append(
            f"CSV schema compliance: {compliance_score:.0f}%. "
            f"Missing fields: {', '.join(missing_fields[:5])}{'...' if len(missing_fields) > 5 else ''}. "
            "Please use the standard VISA transaction format for best results."
        )
    
    if compliance_score < 50:
        metadata["warnings"].append(
            "Critical: Less than 50% schema compliance. "
            "Quality scores will be significantly reduced. "
            "Consider reformatting your data to match the required schema."
        )
    
    return transactions, metadata


def adapt_flat_json_to_visa(data: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Adapt flat JSON records to VISA format.
    Handles both already-nested VISA format and flat structures.
    """
    if not data:
        return [], {"source": "json", "compliance_score": 0, "warnings": ["No data provided"]}
    
    # Check if already in VISA format
    first_record = data[0]
    if "transaction" in first_record and isinstance(first_record["transaction"], dict):
        # Already in VISA format
        return data, {
            "source": "json",
            "compliance_score": 100,
            "is_standard_format": True,
            "total_rows": len(data),
            "warnings": []
        }
    
    # Flat format - convert using column mapping logic
    headers = list(first_record.keys())
    column_mapping = detect_csv_columns(headers)
    compliance_score, missing_fields = calculate_schema_compliance(column_mapping)
    
    ai_mapped = False
    if compliance_score < 80 and os.environ.get("GEMINI_API_KEY"):
        from src.layers.layer1_2_schema_mapper import GenAISchemaMapper
        mapper = GenAISchemaMapper()
        ai_mapping = mapper.map_columns(headers, data)
        if ai_mapping:
            for std_field, raw_col in ai_mapping.items():
                if std_field not in column_mapping:
                    column_mapping[std_field] = raw_col
            compliance_score, missing_fields = calculate_schema_compliance(column_mapping)
            ai_mapped = True
            
    transactions = []
    for i, row in enumerate(data):
        txn = convert_csv_row_to_visa(row, column_mapping, i)
        transactions.append(txn)
    
    metadata = {
        "source": "json",
        "compliance_score": compliance_score,
        "missing_fields": missing_fields,
        "mapped_fields": list(column_mapping.keys()),
        "total_rows": len(transactions),
        "is_standard_format": compliance_score >= 80,
        "quality_penalty": max(0, (100 - compliance_score) / 2),
        "warnings": [],
        "ai_mapped": ai_mapped
    }
    
    if ai_mapped:
        metadata["warnings"].append(
            f"Dynamic schema alignment active: GenAI successfully mapped raw keys "
            f"to standard fields. Compliance improved to {compliance_score:.0f}%."
        )
    elif compliance_score < 80:
        metadata["warnings"].append(
            f"JSON schema compliance: {compliance_score:.0f}%. Use nested VISA format for best results."
        )
    
    return transactions, metadata

