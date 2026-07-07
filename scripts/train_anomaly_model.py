"""
Task 6: Train Anomaly Detection Model Offline
===============================================
Generates training transactions, extracts features, trains StandardScaler
and IsolationForest, and serializes them to src/resources/ for production use.

Run: .venv\\Scripts\\python scripts/train_anomaly_model.py
"""
import sys
import os
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_model():
    print("=" * 60)
    print("  OFFLINE ANOMALY MODEL TRAINING")
    print("=" * 60)

    # 1. Generate training data
    print("\n[1/4] Generating 5,000 training transactions...")
    from src.data_generator import generate_visa_transactions
    transactions = generate_visa_transactions(
        n_transactions=5000,
        anomaly_rate=0.05,  # contamination baseline
        random_seed=42,
    )

    # 2. Extract features
    print("\n[2/4] Extracting features using pipeline layers...")
    from src.layers import (
        InputContractLayer,
        InputValidationLayer,
        FeatureExtractionLayer,
        AnomalyDetectionLayer
    )

    layer1 = InputContractLayer()
    layer1.validate_schema_manifest(use_default=True)

    layer2 = InputValidationLayer(layer1.get_schema())
    layer2.validate(json_data=transactions)
    dataframe = layer2.get_dataframe()

    layer3 = FeatureExtractionLayer()
    layer3.extract_features(dataframe)
    features_df = layer3.get_features()

    # Normalize feature columns
    feat_df = features_df.copy()
    feat_df.columns = [col.lower().strip() for col in feat_df.columns]

    available_features = [f for f in AnomalyDetectionLayer.ANOMALY_FEATURES if f in feat_df.columns]
    print(f"    Selected {len(available_features)} features for training:")
    for f in available_features:
        print(f"      - {f}")

    X = feat_df[available_features].copy().fillna(0)

    # 3. Fit scaler and IsolationForest
    print("\n[3/4] Fitting models...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the same frozen hyperparameters as the anomaly layer
    model = IsolationForest(
        n_estimators=AnomalyDetectionLayer.N_ESTIMATORS,
        contamination=AnomalyDetectionLayer.CONTAMINATION,
        random_state=AnomalyDetectionLayer.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Print summary metrics
    preds = model.predict(X_scaled)
    n_anomalies_detected = sum(1 for p in preds if p == -1)
    print(f"    Model training complete.")
    print(f"    Detected {n_anomalies_detected} anomalies in training set ({n_anomalies_detected/len(X_scaled)*100:.2f}%)")

    # 4. Serialize models to src/resources/
    print("\n[4/4] Serializing models...")
    resources_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "resources"
    )
    os.makedirs(resources_dir, exist_ok=True)

    scaler_path = os.path.join(resources_dir, "anomaly_scaler.pkl")
    model_path = os.path.join(resources_dir, "anomaly_model.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)

    print(f"    Saved scaler -> {scaler_path}")
    print(f"    Saved model  -> {model_path}")
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
