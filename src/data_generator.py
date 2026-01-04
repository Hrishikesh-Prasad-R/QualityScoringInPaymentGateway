"""
VISA Transaction Data Generator

Generates sample VISA transaction data following the comprehensive schema:
- Nested structure with all 10 sections
- Realistic distributions
- Injected anomalies for testing
"""
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


def generate_visa_transactions(
    n_transactions: int = 100,
    anomaly_rate: float = 0.05,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate sample VISA transactions with the comprehensive nested structure.
    
    Args:
        n_transactions: Number of transactions to generate
        anomaly_rate: Fraction of anomalies to inject
        random_seed: Random seed for reproducibility
        
    Returns:
        List of transaction dictionaries
    """
    np.random.seed(random_seed)
    
    n_anomalies = int(n_transactions * anomaly_rate)
    anomaly_indices = set(np.random.choice(n_transactions, n_anomalies, replace=False))
    
    transactions = []
    
    mcc_codes = ["5812", "5411", "5541", "5311", "4111", "7011", "5621"]  # Restaurant, Grocery, Gas, Dept Store, Transport, Hotel, Clothing
    risk_levels = ["low", "medium", "high"]
    card_types = ["credit", "debit", "prepaid"]
    networks = ["VISA", "Mastercard", "RuPay"]
    statuses = ["approved", "declined", "pending"]
    countries = ["IN", "US", "GB", "SG"]
    cities = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"]
    banks = ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Bank"]
    
    for i in range(n_transactions):
        is_anomaly = i in anomaly_indices
        
        # Base timestamp
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 90))
        
        # Generate amount (anomalies have extreme values)
        if is_anomaly and np.random.random() < 0.5:
            amount = float(np.random.uniform(100000, 500000))  # Extreme amount
        else:
            amount = float(np.round(np.random.lognormal(7.5, 1.0), 2))
        
        # Settlement calculations
        gross_amount = amount
        interchange_fee = round(amount * 0.018, 2)  # 1.8%
        gateway_fee = round(amount * 0.003, 2)  # 0.3%
        net_amount = round(gross_amount - interchange_fee - gateway_fee, 2)
        
        # Risk score (anomalies have high risk)
        risk_score = int(np.random.uniform(70, 95)) if is_anomaly else int(np.random.uniform(10, 50))
        
        txn = {
            "transaction": {
                "transaction_id": f"txn_{i:08d}",
                "merchant_order_id": f"order_{i:06d}",
                "type": "authorization",
                "amount": amount,
                "currency": "INR",
                "timestamp": base_time.isoformat() + "Z",
                "status": np.random.choice(statuses, p=[0.9, 0.08, 0.02]),
                "response_code": "00" if np.random.random() > 0.1 else "05",
                "authorization_code": f"A{np.random.randint(10000, 99999)}",
            },
            "card": {
                "network": np.random.choice(networks, p=[0.5, 0.3, 0.2]),
                "pan_token": f"tok_{i:08d}",
                "bin": str(np.random.randint(400000, 499999)),
                "last4": str(np.random.randint(1000, 9999)),
                "expiry_month": f"{np.random.randint(1, 13):02d}",
                "expiry_year": str(np.random.randint(2027, 2032)),
                "card_type": np.random.choice(card_types),
                "funding_source": "consumer",
                "issuer_bank": np.random.choice(banks),
            },
            "merchant": {
                "merchant_id": f"MID_{np.random.randint(1000, 9999)}",
                "terminal_id": f"TID_{np.random.randint(1000, 9999)}",
                "merchant_name": f"Merchant_{i}",
                "merchant_category_code": np.random.choice(mcc_codes),
                "country": "IN" if not is_anomaly else np.random.choice(countries),
                "acquirer_bank": np.random.choice(banks),
                "settlement_account": f"XXXX{np.random.randint(1000, 9999)}",
            },
            "customer": {
                "customer_id": f"cust_{np.random.randint(1000, 9999)}",
                "email": f"user{i}@example.com" if np.random.random() > 0.05 else None,
                "phone": f"+91{np.random.randint(7000000, 9999999)}{np.random.randint(1000, 9999)}",
                "billing_address": {
                    "city": np.random.choice(cities),
                    "state": "KA",
                    "country": "IN",
                    "postal_code": str(np.random.randint(500000, 600000)),
                },
                "shipping_address": {
                    "city": np.random.choice(cities),
                    "state": "KA", 
                    "country": "IN",
                    "postal_code": str(np.random.randint(500000, 600000)),
                },
                "ip_address": f"103.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                "device_fingerprint": f"fp_{np.random.randint(10000, 99999)}",
                "user_agent": "Chrome/Windows",
            },
            "authentication": {
                "three_ds_version": "2.2",
                "eci": "05",
                "cavv": f"CAVV{np.random.randint(100000, 999999)}",
                "ds_transaction_id": f"ds_{np.random.randint(10000, 99999)}",
                "authentication_result": "authenticated" if np.random.random() > 0.1 else "failed",
            },
            "fraud": {
                "risk_score": risk_score,
                "risk_level": "high" if risk_score > 70 else ("medium" if risk_score > 40 else "low"),
                "velocity_check": "fail" if is_anomaly else "pass",
                "geo_check": "pass",
            },
            "network": {
                "network_transaction_id": f"net_{np.random.randint(100000, 999999)}",
                "acquirer_reference_number": f"ARN_{np.random.randint(10000000, 99999999)}",
                "routing_region": "APAC",
                "interchange_category": "consumer_credit",
            },
            "compliance": {
                "sca_applied": True,
                "psd2_exemption": None,
                "aml_screening": "clear" if not is_anomaly else "review",
                "tax_reference": f"GST_{np.random.randint(1000, 9999)}",
                "audit_log_id": f"audit_{np.random.randint(10000, 99999)}",
            },
            "settlement": {
                "settlement_batch_id": f"batch_{base_time.strftime('%Y%m%d')}",
                "clearing_date": (base_time + timedelta(days=1)).strftime("%Y-%m-%d"),
                "settlement_date": (base_time + timedelta(days=2)).strftime("%Y-%m-%d"),
                "gross_amount": gross_amount,
                "interchange_fee": interchange_fee,
                "gateway_fee": gateway_fee,
                "net_amount": net_amount,
            },
            "business_metadata": {
                "invoice_number": f"INV_{i:06d}",
                "product_category": np.random.choice(["Electronics", "Clothing", "Food", "Travel", "Entertainment"]),
                "promo_code": "NEWUSER" if np.random.random() < 0.2 else None,
                "campaign": "HackathonDemo",
                "notes": "Test transaction",
            },
        }
        
        transactions.append(txn)
    
    return transactions


def save_transactions_json(
    transactions: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Save transactions to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=2)
    print(f"Generated {len(transactions)} transactions → {output_path}")
    return output_path


def generate_sample_data(output_dir: str = "data") -> str:
    """Generate sample VISA transaction data for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    transactions = generate_visa_transactions(
        n_transactions=100,
        anomaly_rate=0.05,
        random_seed=42,
    )
    
    output_path = os.path.join(output_dir, "sample_visa_transactions.json")
    save_transactions_json(transactions, output_path)
    
    return output_path


if __name__ == "__main__":
    generate_sample_data("data")
