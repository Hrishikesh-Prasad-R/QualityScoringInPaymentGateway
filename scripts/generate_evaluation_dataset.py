"""
Task 9: Generate Realistic Custom CSV Evaluation Dataset
===========================================================
Creates a flattened evaluation dataset with realistic transaction fields,
pre-defined schemas, and injected, documented anomalies with a 'true_anomaly' label.
Saves the CSV to data/evaluation_visa_dataset.csv.

Run: .venv\\Scripts\\python scripts/generate_evaluation_dataset.py
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_evaluation_dataset():
    print("=" * 60)
    print("  GENERATING EVALUATION VISA DATASET")
    print("=" * 60)

    N_RECORDS = 1000
    np.random.seed(42)

    # Base list of valid values
    networks = ["VISA", "Mastercard", "RuPay"]
    card_types = ["Credit", "Debit", "Prepaid"]
    countries = ["IN", "US", "GB", "SG"]
    mcc_codes = ["5812", "5411", "5541", "5311", "4111", "7011", "5621"]
    
    # Pre-generate lists for normal records
    tx_ids = [f"TXN_{100000 + i}" for i in range(N_RECORDS)]
    amounts = np.round(np.random.lognormal(7.2, 0.8, N_RECORDS) + 50, 2)
    currencies = ["INR"] * N_RECORDS
    
    base_time = datetime.now() - timedelta(days=5)
    timestamps = [
        (base_time + timedelta(seconds=int(i * 450) + np.random.randint(-100, 100))).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(N_RECORDS)
    ]
    
    statuses = ["Success"] * N_RECORDS
    txn_types = ["Payment"] * N_RECORDS
    nets = [np.random.choice(networks) for _ in range(N_RECORDS)]
    types = [np.random.choice(card_types) for _ in range(N_RECORDS)]
    
    bins = []
    for net in nets:
        if net == "VISA":
            bins.append(str(np.random.randint(400000, 499999)))
        elif net == "Mastercard":
            bins.append(str(np.random.randint(510000, 559999)))
        else: # RuPay / Other
            bins.append(str(np.random.randint(600000, 699999)))
    last4s = [f"{np.random.randint(1000, 9999)}" for _ in range(N_RECORDS)]
    
    merchants = [f"MID_{np.random.randint(1000, 9999)}" for _ in range(N_RECORDS)]
    merchant_names = [f"Merchant_{i}" for i in range(N_RECORDS)]
    mccs = [np.random.choice(mcc_codes) for _ in range(N_RECORDS)]
    country_vals = ["IN"] * N_RECORDS # Mostly domestic for standard profile
    
    cust_ids = [f"CUST_{1000 + (i % 250)}" for i in range(N_RECORDS)] # 250 unique customers
    emails = [f"user_{cust_ids[i].lower()}@domain.com" for i in range(N_RECORDS)]
    phones = [f"+9198765{i % 100:02d}{np.random.randint(100, 999)}" for i in range(N_RECORDS)]
    ips = [f"192.168.1.{i % 254}" for i in range(N_RECORDS)]
    risk_scores = np.random.randint(5, 35, N_RECORDS) # Normal records are low-risk
    
    true_anomalies = [0] * N_RECORDS
    anomaly_reasons = ["Normal"] * N_RECORDS

    df = pd.DataFrame({
        "transaction_id": tx_ids,
        "amount": amounts,
        "currency": currencies,
        "timestamp": timestamps,
        "status": statuses,
        "type": txn_types,
        "network": nets,
        "card_type": types,
        "bin": bins,
        "last4": last4s,
        "merchant_id": merchants,
        "merchant_name": merchant_names,
        "merchant_category_code": mccs,
        "country": country_vals,
        "customer_id": cust_ids,
        "email": emails,
        "phone": phones,
        "ip_address": ips,
        "risk_score": risk_scores,
        "true_anomaly": true_anomalies,
        "anomaly_reason": anomaly_reasons
    })

    # ========================================================================
    # INJECT INTERESTING, DOCUMENTED ANOMALY PROFILES
    # ========================================================================
    
    # 1. High Velocity Anomaly (5 transactions from same customer in 20 seconds)
    print("  Injecting High Velocity Anomaly profile...")
    vel_cust = "CUST_9999"
    vel_base_time = base_time + timedelta(hours=4)
    for idx in range(10, 15):
        df.loc[idx, "customer_id"] = vel_cust
        df.loc[idx, "timestamp"] = (vel_base_time + timedelta(seconds=(idx - 10) * 3)).strftime("%Y-%m-%d %H:%M:%S")
        df.loc[idx, "amount"] = 8500.00
        df.loc[idx, "true_anomaly"] = 1
        df.loc[idx, "anomaly_reason"] = "Velocity Anomaly"

    # 2. Extreme Transaction Amount Anomaly
    print("  Injecting Extreme Amount Anomaly profile...")
    for idx in range(120, 125):
        df.loc[idx, "amount"] = 450000.00  # Normal average is ~2000
        df.loc[idx, "true_anomaly"] = 1
        df.loc[idx, "anomaly_reason"] = "Extreme Amount"

    # 3. Card Funding Source Mismatch
    print("  Injecting Card Funding Mismatch profile...")
    for idx in range(300, 310):
        # RuPay card with BIN starting with 4 (which belongs exclusively to VISA)
        df.loc[idx, "network"] = "RuPay"
        df.loc[idx, "bin"] = "402611" # Known VISA BIN
        df.loc[idx, "true_anomaly"] = 1
        df.loc[idx, "anomaly_reason"] = "BIN-Network Mismatch"

    # 4. Geographical Mismatch Anomaly (IN to US in 5 minutes)
    print("  Injecting Geo-Mismatch Anomaly profile...")
    geo_cust = "CUST_8888"
    geo_base_time = base_time + timedelta(hours=10)
    
    # Transaction 1: India
    df.loc[450, "customer_id"] = geo_cust
    df.loc[450, "country"] = "IN"
    df.loc[450, "timestamp"] = geo_base_time.strftime("%Y-%m-%d %H:%M:%S")
    df.loc[450, "true_anomaly"] = 0 # Single transaction is normal
    
    # Transaction 2: United States 3 minutes later (impossible travel speed)
    df.loc[451, "customer_id"] = geo_cust
    df.loc[451, "country"] = "US"
    df.loc[451, "timestamp"] = (geo_base_time + timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S")
    df.loc[451, "true_anomaly"] = 1
    df.loc[451, "anomaly_reason"] = "Geo-Mismatch / Impossible Travel"

    # 5. Invalid / Malformed Data Values (schema validity failures)
    print("  Injecting Malformed Field Values profile...")
    for idx in range(600, 605):
        df.loc[idx, "email"] = "malformed_email_at_domain_dot_com"
        df.loc[idx, "merchant_category_code"] = "ABC" # MCC code must be 4 digits
        df.loc[idx, "true_anomaly"] = 1
        df.loc[idx, "anomaly_reason"] = "Malformed Fields"

    # Save to data directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "evaluation_visa_dataset.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n[SUCCESS] Generated dataset: {out_path}")
    print(f"    Total records: {len(df)}")
    print(f"    Anomalies injected: {df['true_anomaly'].sum()} ({df['true_anomaly'].sum()/len(df)*100:.1f}%)")
    print("\n" + "=" * 60)
    print("  TASK 9 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    generate_evaluation_dataset()
