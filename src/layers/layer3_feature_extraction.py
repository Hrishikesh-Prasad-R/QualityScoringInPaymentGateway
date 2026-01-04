"""
Layer 3: Feature / Signal Extraction Layer

Purpose: Transform raw VISA transaction data into analyzable features
Type: 100% Deterministic transformations
Failure Mode: EXTRACTION_ERROR → SAFE_STOP

This layer extracts features from the flattened VISA transaction DataFrame:
- Transaction features (amount stats, type encodings)
- Card features (network, type, BIN category)
- Merchant features (MCC category, country risk)
- Customer features (address consistency, device info)
- Fraud signals (existing scores, checks)
- Settlement features (fee ratios, timing)
- Cross-field interactions and consistency checks

CRITICAL: All transformations use FIXED parameters - no learning during scoring.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import hashlib
import re

from ..config import LayerStatus
from .layer1_input_contract import LayerResult


# ============================================================================
# FIXED REFERENCE PARAMETERS (Frozen - No Learning)
# ============================================================================
REFERENCE_PARAMS = {
    # Amount statistics for z-score
    "amount_mean": 5000.0,
    "amount_std": 15000.0,
    
    # Card network encodings
    "network_encoding": {
        "visa": 0,
        "mastercard": 1,
        "rupay": 2,
        "amex": 3,
        "unknown": 99,
    },
    
    # Card type encodings
    "card_type_encoding": {
        "credit": 0,
        "debit": 1,
        "prepaid": 2,
        "unknown": 99,
    },
    
    # MCC category mapping (first 2 digits)
    "mcc_category": {
        "58": "restaurant",      # 5812 - Restaurants
        "54": "grocery",         # 5411 - Grocery
        "55": "gas_station",     # 5541 - Gas Stations
        "53": "department",      # 5311 - Department Stores
        "41": "transport",       # 4111 - Local Transport
        "70": "hotel",           # 7011 - Hotels
        "56": "clothing",        # 5621 - Clothing
    },
    
    # MCC category encodings
    "mcc_category_encoding": {
        "restaurant": 0,
        "grocery": 1,
        "gas_station": 2,
        "department": 3,
        "transport": 4,
        "hotel": 5,
        "clothing": 6,
        "unknown": 99,
    },
    
    # Country risk scores (0 = low risk, 1 = high risk)
    "country_risk": {
        "in": 0.1,  # India - low risk (home)
        "us": 0.3,  # USA - moderate
        "gb": 0.3,  # UK - moderate
        "sg": 0.2,  # Singapore - low-moderate
        "unknown": 0.8,  # Unknown - high risk
    },
    
    # Transaction status encoding
    "status_encoding": {
        "approved": 0,
        "pending": 1,
        "declined": 2,
        "failed": 3,
        "unknown": 99,
    },
    
    # Risk level encoding
    "risk_level_encoding": {
        "low": 0,
        "medium": 1,
        "high": 2,
        "unknown": 99,
    },
    
    # Authentication result encoding
    "auth_result_encoding": {
        "authenticated": 0,
        "attempted": 1,
        "failed": 2,
        "unknown": 99,
    },
    
    # Expected amount ranges per MCC category
    "mcc_amount_ranges": {
        "restaurant": (100, 10000),
        "grocery": (50, 15000),
        "gas_station": (100, 10000),
        "department": (200, 50000),
        "transport": (50, 5000),
        "hotel": (1000, 100000),
        "clothing": (200, 30000),
        "unknown": (50, 100000),
    },
    
    # BIN ranges for card type inference
    "bin_visa_range": (400000, 499999),
    "bin_mastercard_range": (510000, 559999),
}


class FeatureExtractionLayer:
    """
    Layer 3: Feature Extraction Layer
    
    Extracts deterministic features from validated VISA transaction DataFrame.
    All transformations use fixed reference parameters.
    """
    
    LAYER_ID = 3
    LAYER_NAME = "feature_extraction"
    
    # Complete list of 35 features we extract
    FEATURE_NAMES = [
        # === TRANSACTION FEATURES (1-8) ===
        "txn_amount",
        "txn_amount_log",
        "txn_amount_zscore",
        "txn_amount_percentile",
        "txn_status_encoded",
        "txn_hour",
        "txn_day_of_week",
        "txn_is_weekend",
        
        # === CARD FEATURES (9-14) ===
        "card_network_encoded",
        "card_type_encoded",
        "card_bin_first2",
        "card_expiry_months_remaining",
        "card_is_domestic_issuer",
        "card_bin_category",
        
        # === MERCHANT FEATURES (15-19) ===
        "merchant_mcc_category_encoded",
        "merchant_country_risk",
        "merchant_is_domestic",
        "merchant_mcc_first2",
        "merchant_acquirer_encoded",
        
        # === CUSTOMER FEATURES (20-24) ===
        "customer_has_email",
        "customer_has_phone",
        "customer_address_match",
        "customer_has_device_fp",
        "customer_ip_is_domestic",
        
        # === FRAUD SIGNALS (25-28) ===
        "fraud_risk_score",
        "fraud_risk_level_encoded",
        "fraud_velocity_passed",
        "fraud_geo_passed",
        
        # === AUTHENTICATION FEATURES (29-31) ===
        "auth_3ds_version_numeric",
        "auth_eci_value",
        "auth_result_encoded",
        
        # === SETTLEMENT FEATURES (32-35) ===
        "settlement_fee_ratio",
        "settlement_days_to_clear",
        "settlement_amount_match",
        "settlement_net_ratio",
    ]
    
    def __init__(self):
        self.features_df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.extraction_warnings: List[str] = []
    
    def extract_features(
        self,
        dataframe: pd.DataFrame,
    ) -> LayerResult:
        """
        Extract all features from the validated VISA transaction DataFrame.
        
        Args:
            dataframe: Validated & flattened DataFrame from Layer 2
            
        Returns:
            LayerResult with extraction status
        """
        import time
        start_time = time.time()
        
        issues = []
        warnings = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            df = dataframe.copy()
            self.original_df = df.copy()
            n_rows = len(df)
            
            # Normalize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=df.index)
            
            # ================================================================
            # TRANSACTION FEATURES (1-8)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_transaction_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                issues.append({
                    "type": "EXTRACTION_ERROR",
                    "code": "TRANSACTION_FEATURE_ERROR",
                    "message": f"Failed to extract transaction features: {str(e)}",
                    "severity": "critical",
                })
                return self._create_failed_result(start_time, checks_performed, checks_passed, issues)
            
            # ================================================================
            # CARD FEATURES (9-14)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_card_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                warnings.append(f"Card feature extraction degraded: {str(e)}")
                features = self._fill_card_defaults(features)
            
            # ================================================================
            # MERCHANT FEATURES (15-19)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_merchant_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                warnings.append(f"Merchant feature extraction degraded: {str(e)}")
                features = self._fill_merchant_defaults(features)
            
            # ================================================================
            # CUSTOMER FEATURES (20-24)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_customer_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                warnings.append(f"Customer feature extraction degraded: {str(e)}")
                features = self._fill_customer_defaults(features)
            
            # ================================================================
            # FRAUD SIGNALS (25-28)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_fraud_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                warnings.append(f"Fraud feature extraction degraded: {str(e)}")
                features = self._fill_fraud_defaults(features)
            
            # ================================================================
            # AUTHENTICATION FEATURES (29-31)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_auth_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                warnings.append(f"Auth feature extraction degraded: {str(e)}")
                features = self._fill_auth_defaults(features)
            
            # ================================================================
            # SETTLEMENT FEATURES (32-35)
            # ================================================================
            checks_performed += 1
            try:
                features = self._extract_settlement_features(df, features, warnings)
                checks_passed += 1
            except Exception as e:
                warnings.append(f"Settlement feature extraction degraded: {str(e)}")
                features = self._fill_settlement_defaults(features)
            
            # ================================================================
            # VALIDATION & CLEANUP
            # ================================================================
            checks_performed += 1
            
            # Check for NaN values
            nan_counts = features.isna().sum()
            cols_with_nans = nan_counts[nan_counts > 0]
            if len(cols_with_nans) > 0:
                warnings.append(f"NaN values found in {len(cols_with_nans)} features, filling with defaults")
                features = features.fillna(0)
            
            # Verify all expected features present
            missing_features = set(self.FEATURE_NAMES) - set(features.columns)
            extra_features = set(features.columns) - set(self.FEATURE_NAMES)
            
            if missing_features:
                warnings.append(f"Missing features: {missing_features}")
                for feat in missing_features:
                    features[feat] = 0
            
            # Ensure column order matches FEATURE_NAMES
            features = features[self.FEATURE_NAMES]
            
            checks_passed += 1
            
            # Store features
            self.features_df = features
            self.extraction_warnings = warnings
            
            # Determine status
            if issues:
                status = LayerStatus.FAILED
                can_continue = False
            elif len(warnings) > 5:
                status = LayerStatus.DEGRADED
                can_continue = True
            else:
                status = LayerStatus.PASSED
                can_continue = True
            
            return self._create_result(
                status=status,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=can_continue,
                details={
                    "features_extracted": len(features.columns),
                    "feature_names": list(features.columns),
                    "row_count": n_rows,
                    "warnings_count": len(warnings),
                    "feature_stats": {
                        "txn_amount_mean": float(features["txn_amount"].mean()),
                        "txn_amount_std": float(features["txn_amount"].std()),
                        "fraud_risk_score_mean": float(features["fraud_risk_score"].mean()),
                    },
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "EXTRACTION_ERROR",
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error during feature extraction: {str(e)}",
                "severity": "critical",
            })
            return self._create_failed_result(start_time, checks_performed, checks_passed, issues)
    
    # ========================================================================
    # TRANSACTION FEATURES
    # ========================================================================
    def _extract_transaction_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract transaction-related features."""
        
        # Amount features
        amount_col = self._get_column(df, ["txn_amount", "amount", "transaction_amount"])
        if amount_col:
            features["txn_amount"] = df[amount_col].astype(float)
        else:
            features["txn_amount"] = 0.0
            warnings.append("Amount column not found, defaulting to 0")
        
        features["txn_amount_log"] = np.log10(features["txn_amount"].clip(lower=0.01) + 1)
        features["txn_amount_zscore"] = (
            (features["txn_amount"] - REFERENCE_PARAMS["amount_mean"]) 
            / REFERENCE_PARAMS["amount_std"]
        )
        features["txn_amount_percentile"] = features["txn_amount"].rank(pct=True) * 100
        
        # Status encoding
        status_col = self._get_column(df, ["txn_status", "status", "transaction_status"])
        if status_col:
            features["txn_status_encoded"] = df[status_col].apply(
                lambda x: REFERENCE_PARAMS["status_encoding"].get(
                    str(x).lower().strip(), 99
                )
            )
        else:
            features["txn_status_encoded"] = 0  # Default to approved
        
        # Temporal features
        ts_col = self._get_column(df, ["txn_timestamp", "timestamp", "transaction_timestamp"])
        if ts_col:
            timestamps = pd.to_datetime(df[ts_col], errors="coerce")
            null_ts = timestamps.isna().sum()
            if null_ts > 0:
                warnings.append(f"{null_ts} timestamps could not be parsed")
                timestamps = timestamps.fillna(pd.Timestamp.now())
            
            features["txn_hour"] = timestamps.dt.hour
            features["txn_day_of_week"] = timestamps.dt.dayofweek
            features["txn_is_weekend"] = (features["txn_day_of_week"] >= 5).astype(int)
        else:
            features["txn_hour"] = 12
            features["txn_day_of_week"] = 0
            features["txn_is_weekend"] = 0
            warnings.append("Timestamp column not found, using defaults")
        
        return features
    
    # ========================================================================
    # CARD FEATURES
    # ========================================================================
    def _extract_card_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract card-related features."""
        
        # Network encoding
        network_col = self._get_column(df, ["card_network", "network"])
        if network_col:
            features["card_network_encoded"] = df[network_col].apply(
                lambda x: REFERENCE_PARAMS["network_encoding"].get(
                    str(x).lower().strip(), 99
                )
            )
        else:
            features["card_network_encoded"] = 0  # Default VISA
        
        # Card type encoding
        type_col = self._get_column(df, ["card_card_type", "card_type"])
        if type_col:
            features["card_type_encoded"] = df[type_col].apply(
                lambda x: REFERENCE_PARAMS["card_type_encoding"].get(
                    str(x).lower().strip(), 99
                )
            )
        else:
            features["card_type_encoded"] = 0  # Default credit
        
        # BIN features
        bin_col = self._get_column(df, ["card_bin", "bin"])
        if bin_col:
            features["card_bin_first2"] = df[bin_col].astype(str).str[:2].apply(
                lambda x: int(x) if x.isdigit() else 0
            )
            # BIN category (0=VISA, 1=MC, 2=Other)
            features["card_bin_category"] = df[bin_col].apply(self._categorize_bin)
        else:
            features["card_bin_first2"] = 0
            features["card_bin_category"] = 0
        
        # Expiry months remaining
        exp_month_col = self._get_column(df, ["card_expiry_month", "expiry_month"])
        exp_year_col = self._get_column(df, ["card_expiry_year", "expiry_year"])
        if exp_month_col and exp_year_col:
            try:
                now = datetime.now()
                exp_months = df[exp_month_col].astype(int)
                exp_years = df[exp_year_col].astype(int)
                features["card_expiry_months_remaining"] = (
                    (exp_years - now.year) * 12 + (exp_months - now.month)
                ).clip(lower=0)
            except:
                features["card_expiry_months_remaining"] = 12
        else:
            features["card_expiry_months_remaining"] = 12
        
        # Domestic issuer flag
        issuer_col = self._get_column(df, ["card_issuer_bank", "issuer_bank"])
        if issuer_col:
            indian_banks = ["hdfc", "icici", "sbi", "axis", "kotak", "yes", "idbi", "pnb", "bob"]
            features["card_is_domestic_issuer"] = df[issuer_col].apply(
                lambda x: 1 if any(bank in str(x).lower() for bank in indian_banks) else 0
            )
        else:
            features["card_is_domestic_issuer"] = 1  # Default domestic
        
        return features
    
    def _categorize_bin(self, bin_value) -> int:
        """Categorize BIN into card network (0=VISA, 1=MC, 2=Other)."""
        try:
            bin_int = int(str(bin_value)[:6])
            if 400000 <= bin_int <= 499999:
                return 0  # VISA
            elif 510000 <= bin_int <= 559999 or 222100 <= bin_int <= 272099:
                return 1  # Mastercard
            else:
                return 2  # Other
        except:
            return 2
    
    # ========================================================================
    # MERCHANT FEATURES
    # ========================================================================
    def _extract_merchant_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract merchant-related features."""
        
        # MCC category encoding
        mcc_col = self._get_column(df, ["merchant_merchant_category_code", "merchant_category_code", "mcc"])
        if mcc_col:
            features["merchant_mcc_first2"] = df[mcc_col].astype(str).str[:2]
            features["merchant_mcc_category_encoded"] = features["merchant_mcc_first2"].apply(
                lambda x: REFERENCE_PARAMS["mcc_category_encoding"].get(
                    REFERENCE_PARAMS["mcc_category"].get(x, "unknown"), 99
                )
            )
            # Convert mcc_first2 to numeric
            features["merchant_mcc_first2"] = features["merchant_mcc_first2"].apply(
                lambda x: int(x) if x.isdigit() else 0
            )
        else:
            features["merchant_mcc_first2"] = 0
            features["merchant_mcc_category_encoded"] = 99
        
        # Country risk
        country_col = self._get_column(df, ["merchant_country", "country"])
        if country_col:
            features["merchant_country_risk"] = df[country_col].apply(
                lambda x: REFERENCE_PARAMS["country_risk"].get(
                    str(x).lower().strip(), 0.8
                )
            )
            features["merchant_is_domestic"] = df[country_col].apply(
                lambda x: 1 if str(x).lower().strip() == "in" else 0
            )
        else:
            features["merchant_country_risk"] = 0.1
            features["merchant_is_domestic"] = 1
        
        # Acquirer encoding (simple hash-based)
        acq_col = self._get_column(df, ["merchant_acquirer_bank", "acquirer_bank"])
        if acq_col:
            features["merchant_acquirer_encoded"] = df[acq_col].apply(
                lambda x: hash(str(x).lower()) % 10 if pd.notna(x) else 0
            )
        else:
            features["merchant_acquirer_encoded"] = 0
        
        return features
    
    # ========================================================================
    # CUSTOMER FEATURES
    # ========================================================================
    def _extract_customer_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract customer-related features."""
        
        # Email presence
        email_col = self._get_column(df, ["customer_email", "email"])
        if email_col:
            features["customer_has_email"] = df[email_col].apply(
                lambda x: 1 if pd.notna(x) and "@" in str(x) else 0
            )
        else:
            features["customer_has_email"] = 0
        
        # Phone presence
        phone_col = self._get_column(df, ["customer_phone", "phone"])
        if phone_col:
            features["customer_has_phone"] = df[phone_col].apply(
                lambda x: 1 if pd.notna(x) and len(str(x)) >= 10 else 0
            )
        else:
            features["customer_has_phone"] = 0
        
        # Address match (billing == shipping)
        billing_city = self._get_column(df, ["customer_billing_address_city", "billing_city"])
        shipping_city = self._get_column(df, ["customer_shipping_address_city", "shipping_city"])
        if billing_city and shipping_city:
            features["customer_address_match"] = (
                df[billing_city].astype(str).str.lower() == df[shipping_city].astype(str).str.lower()
            ).astype(int)
        else:
            features["customer_address_match"] = 1  # Assume match
        
        # Device fingerprint presence
        fp_col = self._get_column(df, ["customer_device_fingerprint", "device_fingerprint"])
        if fp_col:
            features["customer_has_device_fp"] = df[fp_col].apply(
                lambda x: 1 if pd.notna(x) and len(str(x)) > 0 else 0
            )
        else:
            features["customer_has_device_fp"] = 0
        
        # IP is domestic (simple check for Indian IP pattern)
        ip_col = self._get_column(df, ["customer_ip_address", "ip_address"])
        if ip_col:
            features["customer_ip_is_domestic"] = df[ip_col].apply(
                lambda x: 1 if pd.notna(x) and str(x).startswith("103.") else 0
            )
        else:
            features["customer_ip_is_domestic"] = 1
        
        return features
    
    # ========================================================================
    # FRAUD FEATURES
    # ========================================================================
    def _extract_fraud_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract fraud-related features."""
        
        # Risk score
        score_col = self._get_column(df, ["fraud_risk_score", "risk_score"])
        if score_col:
            features["fraud_risk_score"] = df[score_col].fillna(0).astype(float).clip(0, 100)
        else:
            features["fraud_risk_score"] = 0
        
        # Risk level encoding
        level_col = self._get_column(df, ["fraud_risk_level", "risk_level"])
        if level_col:
            features["fraud_risk_level_encoded"] = df[level_col].apply(
                lambda x: REFERENCE_PARAMS["risk_level_encoding"].get(
                    str(x).lower().strip(), 99
                )
            )
        else:
            features["fraud_risk_level_encoded"] = 0
        
        # Velocity check
        vel_col = self._get_column(df, ["fraud_velocity_check", "velocity_check"])
        if vel_col:
            features["fraud_velocity_passed"] = df[vel_col].apply(
                lambda x: 1 if str(x).lower().strip() == "pass" else 0
            )
        else:
            features["fraud_velocity_passed"] = 1
        
        # Geo check
        geo_col = self._get_column(df, ["fraud_geo_check", "geo_check"])
        if geo_col:
            features["fraud_geo_passed"] = df[geo_col].apply(
                lambda x: 1 if str(x).lower().strip() == "pass" else 0
            )
        else:
            features["fraud_geo_passed"] = 1
        
        return features
    
    # ========================================================================
    # AUTHENTICATION FEATURES
    # ========================================================================
    def _extract_auth_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract authentication-related features."""
        
        # 3DS version
        ver_col = self._get_column(df, ["auth_three_ds_version", "three_ds_version"])
        if ver_col:
            features["auth_3ds_version_numeric"] = df[ver_col].apply(
                lambda x: float(str(x).replace(".", "")[:2]) / 10 if pd.notna(x) else 0
            )
        else:
            features["auth_3ds_version_numeric"] = 2.0
        
        # ECI value
        eci_col = self._get_column(df, ["auth_eci", "eci"])
        if eci_col:
            features["auth_eci_value"] = df[eci_col].apply(
                lambda x: int(x) if pd.notna(x) and str(x).isdigit() else 0
            )
        else:
            features["auth_eci_value"] = 5
        
        # Auth result
        result_col = self._get_column(df, ["auth_authentication_result", "authentication_result"])
        if result_col:
            features["auth_result_encoded"] = df[result_col].apply(
                lambda x: REFERENCE_PARAMS["auth_result_encoding"].get(
                    str(x).lower().strip(), 99
                )
            )
        else:
            features["auth_result_encoded"] = 0
        
        return features
    
    # ========================================================================
    # SETTLEMENT FEATURES
    # ========================================================================
    def _extract_settlement_features(
        self, df: pd.DataFrame, features: pd.DataFrame, warnings: List[str]
    ) -> pd.DataFrame:
        """Extract settlement-related features."""
        
        # Fee ratio (interchange + gateway) / gross
        gross_col = self._get_column(df, ["settlement_gross_amount", "gross_amount"])
        int_fee_col = self._get_column(df, ["settlement_interchange_fee", "interchange_fee"])
        gw_fee_col = self._get_column(df, ["settlement_gateway_fee", "gateway_fee"])
        net_col = self._get_column(df, ["settlement_net_amount", "net_amount"])
        
        if gross_col and int_fee_col and gw_fee_col:
            gross = df[gross_col].fillna(0).astype(float)
            int_fee = df[int_fee_col].fillna(0).astype(float)
            gw_fee = df[gw_fee_col].fillna(0).astype(float)
            features["settlement_fee_ratio"] = ((int_fee + gw_fee) / gross.clip(lower=1)).clip(0, 1)
        else:
            features["settlement_fee_ratio"] = 0.02  # Default 2%
        
        # Days to clear
        clear_col = self._get_column(df, ["settlement_clearing_date", "clearing_date"])
        ts_col = self._get_column(df, ["txn_timestamp", "timestamp"])
        if clear_col and ts_col:
            try:
                txn_dates = pd.to_datetime(df[ts_col], errors="coerce").dt.date
                clear_dates = pd.to_datetime(df[clear_col], errors="coerce").dt.date
                features["settlement_days_to_clear"] = (
                    (pd.to_datetime(clear_dates) - pd.to_datetime(txn_dates)).dt.days
                ).fillna(1).clip(0, 30)
            except:
                features["settlement_days_to_clear"] = 1
        else:
            features["settlement_days_to_clear"] = 1
        
        # Amount match (gross should equal txn amount)
        txn_amount_col = self._get_column(df, ["txn_amount", "amount"])
        if gross_col and txn_amount_col:
            features["settlement_amount_match"] = (
                np.isclose(df[gross_col].fillna(0).astype(float), 
                          df[txn_amount_col].fillna(0).astype(float), rtol=0.01)
            ).astype(int)
        else:
            features["settlement_amount_match"] = 1
        
        # Net ratio
        if gross_col and net_col:
            gross = df[gross_col].fillna(0).astype(float)
            net = df[net_col].fillna(0).astype(float)
            features["settlement_net_ratio"] = (net / gross.clip(lower=1)).clip(0, 1)
        else:
            features["settlement_net_ratio"] = 0.98  # Default 98%
        
        return features
    
    # ========================================================================
    # DEFAULT FILLERS
    # ========================================================================
    def _fill_card_defaults(self, features: pd.DataFrame) -> pd.DataFrame:
        features["card_network_encoded"] = 0
        features["card_type_encoded"] = 0
        features["card_bin_first2"] = 0
        features["card_expiry_months_remaining"] = 12
        features["card_is_domestic_issuer"] = 1
        features["card_bin_category"] = 0
        return features
    
    def _fill_merchant_defaults(self, features: pd.DataFrame) -> pd.DataFrame:
        features["merchant_mcc_category_encoded"] = 99
        features["merchant_country_risk"] = 0.1
        features["merchant_is_domestic"] = 1
        features["merchant_mcc_first2"] = 0
        features["merchant_acquirer_encoded"] = 0
        return features
    
    def _fill_customer_defaults(self, features: pd.DataFrame) -> pd.DataFrame:
        features["customer_has_email"] = 0
        features["customer_has_phone"] = 0
        features["customer_address_match"] = 1
        features["customer_has_device_fp"] = 0
        features["customer_ip_is_domestic"] = 1
        return features
    
    def _fill_fraud_defaults(self, features: pd.DataFrame) -> pd.DataFrame:
        features["fraud_risk_score"] = 0
        features["fraud_risk_level_encoded"] = 0
        features["fraud_velocity_passed"] = 1
        features["fraud_geo_passed"] = 1
        return features
    
    def _fill_auth_defaults(self, features: pd.DataFrame) -> pd.DataFrame:
        features["auth_3ds_version_numeric"] = 2.0
        features["auth_eci_value"] = 5
        features["auth_result_encoded"] = 0
        return features
    
    def _fill_settlement_defaults(self, features: pd.DataFrame) -> pd.DataFrame:
        features["settlement_fee_ratio"] = 0.02
        features["settlement_days_to_clear"] = 1
        features["settlement_amount_match"] = 1
        features["settlement_net_ratio"] = 0.98
        return features
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    def _get_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _create_result(
        self,
        status: LayerStatus,
        start_time: float,
        checks_performed: int,
        checks_passed: int,
        issues: List[Dict[str, Any]],
        warnings: List[str] = None,
        can_continue: bool = True,
        details: Dict[str, Any] = None,
    ) -> LayerResult:
        """Create a standardized layer result."""
        import time
        return LayerResult(
            layer_id=self.LAYER_ID,
            layer_name=self.LAYER_NAME,
            status=status,
            execution_time_ms=(time.time() - start_time) * 1000,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            issues=issues,
            warnings=warnings or [],
            details=details or {},
            can_continue=can_continue,
        )
    
    def _create_failed_result(
        self,
        start_time: float,
        checks_performed: int,
        checks_passed: int,
        issues: List[Dict[str, Any]],
    ) -> LayerResult:
        """Create a failed layer result."""
        return self._create_result(
            status=LayerStatus.FAILED,
            start_time=start_time,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            issues=issues,
            can_continue=False,
        )
    
    def get_features(self) -> Optional[pd.DataFrame]:
        """Get the extracted features DataFrame."""
        return self.features_df
    
    def get_original_data(self) -> Optional[pd.DataFrame]:
        """Get the original input DataFrame."""
        return self.original_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.FEATURE_NAMES
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted features."""
        if self.features_df is None:
            return {}
        
        return {
            "total_features": len(self.FEATURE_NAMES),
            "row_count": len(self.features_df),
            "feature_means": self.features_df.mean().to_dict(),
            "feature_mins": self.features_df.min().to_dict(),
            "feature_maxs": self.features_df.max().to_dict(),
        }
