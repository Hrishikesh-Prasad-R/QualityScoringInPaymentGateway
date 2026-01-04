"""
Layer 4.4: Anomaly Detection (ML-Informed)

Purpose: Detect statistical outliers and anomalous patterns
Type: ML-INFORMED - CAN ONLY FLAG (not reject)
Failure Mode: ANOMALY_FAILURE → FLAG for review

CRITICAL PRINCIPLE: "ML informs, Rules enforce, Humans decide"
- This layer can ONLY flag records for human review
- It CANNOT reject records on its own
- Uses frozen parameters for reproducibility

Detection Methods:
1. Isolation Forest (ensemble)
2. Statistical outliers (z-score based)
3. Category-specific deviation
4. Velocity anomalies

Output: Anomaly score (0-1) per record + flags
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config import (
    LayerStatus,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_RANDOM_STATE,
    ANOMALY_HIGH_THRESHOLD,
    ANOMALY_MEDIUM_THRESHOLD,
)
from .layer1_input_contract import LayerResult


@dataclass
class AnomalyResult:
    """Anomaly detection result for a single record."""
    record_id: Any
    anomaly_score: float  # 0-1, higher = more anomalous
    is_anomaly: bool
    anomaly_level: str  # "none", "low", "medium", "high"
    flags: List[str] = field(default_factory=list)
    contributing_features: List[Tuple[str, float]] = field(default_factory=list)


class AnomalyDetectionLayer:
    """
    Layer 4.4: Anomaly Detection
    
    Uses ensemble methods to detect anomalous transactions.
    IMPORTANT: Can only FLAG for review, cannot REJECT.
    """
    
    LAYER_ID = 4.4
    LAYER_NAME = "anomaly_detection"
    
    # Features to use for anomaly detection
    ANOMALY_FEATURES = [
        "txn_amount_zscore",
        "txn_amount_percentile",
        "txn_hour",
        "txn_is_weekend",
        "card_type_encoded",
        "merchant_country_risk",
        "merchant_is_domestic",
        "fraud_risk_score",
        "fraud_velocity_passed",
        "fraud_geo_passed",
        "customer_address_match",
        "customer_ip_is_domestic",
        "settlement_fee_ratio",
        "auth_result_encoded",
    ]
    
    # FROZEN PARAMETERS (for reproducibility)
    CONTAMINATION = ISOLATION_FOREST_CONTAMINATION  # Expected anomaly rate
    RANDOM_STATE = ISOLATION_FOREST_RANDOM_STATE  # FROZEN seed
    N_ESTIMATORS = 100  # Number of trees
    
    def __init__(self):
        self.anomaly_results: List[AnomalyResult] = []
        self.flagged_indices: List[int] = []
        self.high_risk_indices: List[int] = []
        self.model_fitted = False
        self.scaler = None
        self.isolation_forest = None
    
    def detect(
        self,
        features_df: pd.DataFrame,
        valid_indices: List[int] = None,
    ) -> LayerResult:
        """
        Detect anomalies in the feature DataFrame.
        
        Args:
            features_df: Features DataFrame from Layer 3
            valid_indices: Indices of records that passed previous layers
            
        Returns:
            LayerResult with anomaly detection results
        """
        import time
        start_time = time.time()
        
        issues = []
        warnings_list = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            feat_df = features_df.copy()
            feat_df.columns = [col.lower().strip() for col in feat_df.columns]
            
            n_records = len(feat_df)
            
            # Filter to valid indices if provided
            if valid_indices is not None:
                process_indices = [i for i in valid_indices if i < len(feat_df)]
            else:
                process_indices = list(range(len(feat_df)))
            
            if len(process_indices) == 0:
                warnings_list.append("No valid records to process")
                return self._create_result(
                    status=LayerStatus.PASSED,
                    start_time=start_time,
                    checks_performed=1,
                    checks_passed=1,
                    issues=issues,
                    warnings=warnings_list,
                    can_continue=True,
                    details={"records_processed": 0},
                )
            
            self.anomaly_results = []
            self.flagged_indices = []
            self.high_risk_indices = []
            
            # ================================================================
            # PREPARE FEATURE MATRIX
            # ================================================================
            checks_performed += 1
            
            # Select available features
            available_features = [f for f in self.ANOMALY_FEATURES if f in feat_df.columns]
            if len(available_features) < 5:
                warnings_list.append(f"Only {len(available_features)} features available for anomaly detection")
            
            # Extract feature matrix for valid indices
            X = feat_df.loc[process_indices, available_features].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            checks_passed += 1
            
            # ================================================================
            # METHOD 1: Isolation Forest
            # ================================================================
            checks_performed += 1
            
            if SKLEARN_AVAILABLE and len(process_indices) >= 10:
                try:
                    # Scale features
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Fit Isolation Forest with FROZEN parameters
                    self.isolation_forest = IsolationForest(
                        n_estimators=self.N_ESTIMATORS,
                        contamination=self.CONTAMINATION,
                        random_state=self.RANDOM_STATE,  # FROZEN for reproducibility
                        n_jobs=-1,
                    )
                    
                    # Get anomaly scores
                    # Note: decision_function returns negative for anomalies
                    raw_scores = self.isolation_forest.fit_predict(X_scaled)
                    decision_scores = self.isolation_forest.decision_function(X_scaled)
                    
                    # Normalize to 0-1 (0 = normal, 1 = anomalous)
                    # decision_function: higher = more normal, so we invert
                    min_score = decision_scores.min()
                    max_score = decision_scores.max()
                    if max_score > min_score:
                        if_scores = 1 - (decision_scores - min_score) / (max_score - min_score)
                    else:
                        if_scores = np.zeros(len(decision_scores))
                    
                    self.model_fitted = True
                    checks_passed += 1
                    
                except Exception as e:
                    warnings_list.append(f"Isolation Forest failed: {str(e)}")
                    if_scores = np.zeros(len(process_indices))
            else:
                if not SKLEARN_AVAILABLE:
                    warnings_list.append("sklearn not available, using statistical methods only")
                else:
                    warnings_list.append("Too few records for Isolation Forest")
                if_scores = np.zeros(len(process_indices))
            
            # ================================================================
            # METHOD 2: Statistical Outliers (z-score based)
            # ================================================================
            checks_performed += 1
            
            stat_scores = self._calculate_statistical_scores(feat_df, process_indices)
            checks_passed += 1
            
            # ================================================================
            # METHOD 3: Rule-Based Flags
            # ================================================================
            checks_performed += 1
            
            rule_flags = self._calculate_rule_flags(feat_df, process_indices)
            checks_passed += 1
            
            # ================================================================
            # COMBINE SCORES (Ensemble)
            # ================================================================
            checks_performed += 1
            
            for i, idx in enumerate(process_indices):
                record_id = feat_df.iloc[idx].get("txn_transaction_id", idx) if "txn_transaction_id" in feat_df.columns else idx
                
                # Combine scores (weighted average)
                if_score = if_scores[i] if i < len(if_scores) else 0
                stat_score = stat_scores.get(idx, 0)
                rule_score = rule_flags.get(idx, {}).get("score", 0)
                
                # Weighted ensemble
                combined_score = (
                    0.5 * if_score +      # Isolation Forest
                    0.3 * stat_score +    # Statistical
                    0.2 * rule_score      # Rule-based
                )
                
                combined_score = min(1.0, max(0.0, combined_score))
                
                # Determine anomaly level
                if combined_score >= ANOMALY_HIGH_THRESHOLD:
                    anomaly_level = "high"
                    is_anomaly = True
                elif combined_score >= ANOMALY_MEDIUM_THRESHOLD:
                    anomaly_level = "medium"
                    is_anomaly = True
                elif combined_score >= 0.25:
                    anomaly_level = "low"
                    is_anomaly = True
                else:
                    anomaly_level = "none"
                    is_anomaly = False
                
                # Get flags
                flags = rule_flags.get(idx, {}).get("flags", [])
                
                # Get contributing features
                contributing = self._get_contributing_features(
                    feat_df.iloc[idx] if isinstance(idx, int) else feat_df.loc[idx],
                    available_features,
                )
                
                self.anomaly_results.append(AnomalyResult(
                    record_id=record_id,
                    anomaly_score=round(combined_score, 4),
                    is_anomaly=is_anomaly,
                    anomaly_level=anomaly_level,
                    flags=flags,
                    contributing_features=contributing[:5],  # Top 5
                ))
                
                if is_anomaly:
                    self.flagged_indices.append(idx)
                    if anomaly_level == "high":
                        self.high_risk_indices.append(idx)
            
            checks_passed += 1
            
            # ================================================================
            # SUMMARY
            # ================================================================
            n_processed = len(process_indices)
            n_flagged = len(self.flagged_indices)
            n_high_risk = len(self.high_risk_indices)
            
            anomaly_scores = [r.anomaly_score for r in self.anomaly_results]
            mean_score = np.mean(anomaly_scores) if anomaly_scores else 0
            
            # Count by level
            level_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
            for r in self.anomaly_results:
                level_counts[r.anomaly_level] += 1
            
            # Determine status
            if n_flagged == 0:
                status = LayerStatus.PASSED
            else:
                status = LayerStatus.DEGRADED  # Flagged records need review
            
            return self._create_result(
                status=status,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings_list,
                can_continue=True,  # ML layers NEVER block
                details={
                    "records_processed": n_processed,
                    "records_flagged": n_flagged,
                    "high_risk_count": n_high_risk,
                    "mean_anomaly_score": round(mean_score, 4),
                    "level_counts": level_counts,
                    "features_used": len(available_features),
                    "model_fitted": self.model_fitted,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "ANOMALY_FAILURE",
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error: {str(e)}",
                "severity": "warning",  # ML failures are warnings, not critical
            })
            return self._create_result(
                status=LayerStatus.DEGRADED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                can_continue=True,  # ML layers NEVER block
            )
    
    def _calculate_statistical_scores(
        self, 
        feat_df: pd.DataFrame, 
        indices: List[int]
    ) -> Dict[int, float]:
        """Calculate statistical anomaly scores based on z-scores."""
        scores = {}
        
        for idx in indices:
            row = feat_df.iloc[idx] if isinstance(idx, int) else feat_df.loc[idx]
            
            anomaly_indicators = []
            
            # Amount z-score anomaly
            amount_zscore = abs(row.get("txn_amount_zscore", 0))
            if amount_zscore > 3:
                anomaly_indicators.append(1.0)
            elif amount_zscore > 2:
                anomaly_indicators.append(0.5)
            else:
                anomaly_indicators.append(0.0)
            
            # Risk score anomaly
            risk_score = row.get("fraud_risk_score", 0)
            if risk_score > 80:
                anomaly_indicators.append(1.0)
            elif risk_score > 60:
                anomaly_indicators.append(0.5)
            else:
                anomaly_indicators.append(0.0)
            
            # Failed checks
            if row.get("fraud_velocity_passed", 1) == 0:
                anomaly_indicators.append(0.8)
            else:
                anomaly_indicators.append(0.0)
            
            if row.get("fraud_geo_passed", 1) == 0:
                anomaly_indicators.append(0.8)
            else:
                anomaly_indicators.append(0.0)
            
            # International transaction
            if row.get("merchant_is_domestic", 1) == 0:
                anomaly_indicators.append(0.3)
            else:
                anomaly_indicators.append(0.0)
            
            scores[idx] = np.mean(anomaly_indicators) if anomaly_indicators else 0
        
        return scores
    
    def _calculate_rule_flags(
        self, 
        feat_df: pd.DataFrame, 
        indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Calculate rule-based anomaly flags."""
        results = {}
        
        for idx in indices:
            row = feat_df.iloc[idx] if isinstance(idx, int) else feat_df.loc[idx]
            flags = []
            score = 0.0
            
            # High amount (top 1%)
            if row.get("txn_amount_percentile", 50) > 99:
                flags.append("EXTREME_AMOUNT")
                score += 0.4
            
            # Very high risk score
            if row.get("fraud_risk_score", 0) > 85:
                flags.append("HIGH_RISK_SCORE")
                score += 0.5
            
            # Failed velocity check
            if row.get("fraud_velocity_passed", 1) == 0:
                flags.append("VELOCITY_FAIL")
                score += 0.3
            
            # Failed geo check
            if row.get("fraud_geo_passed", 1) == 0:
                flags.append("GEO_FAIL")
                score += 0.3
            
            # International with high risk
            if row.get("merchant_is_domestic", 1) == 0 and row.get("fraud_risk_score", 0) > 50:
                flags.append("INTL_HIGH_RISK")
                score += 0.4
            
            # Address mismatch with high amount
            if row.get("customer_address_match", 1) == 0 and row.get("txn_amount_percentile", 50) > 90:
                flags.append("ADDRESS_MISMATCH_HIGH_VALUE")
                score += 0.3
            
            # Auth failed
            if row.get("auth_result_encoded", 0) == 2:  # failed
                flags.append("AUTH_FAILED")
                score += 0.4
            
            # Unusual hour (2-5 AM)
            hour = row.get("txn_hour", 12)
            if 2 <= hour <= 5:
                flags.append("UNUSUAL_HOUR")
                score += 0.2
            
            results[idx] = {
                "flags": flags,
                "score": min(1.0, score),
            }
        
        return results
    
    def _get_contributing_features(
        self, 
        row: pd.Series, 
        features: List[str]
    ) -> List[Tuple[str, float]]:
        """Get features that contribute most to anomaly score."""
        contributions = []
        
        for feat in features:
            value = row.get(feat, 0)
            
            # Calculate deviation from normal
            if feat == "txn_amount_zscore":
                contrib = abs(float(value)) / 3.0  # Normalize by 3 sigma
            elif feat == "fraud_risk_score":
                contrib = float(value) / 100.0
            elif feat in ["fraud_velocity_passed", "fraud_geo_passed", "merchant_is_domestic"]:
                contrib = 1.0 - float(value)  # 0 = bad = high contribution
            elif feat == "auth_result_encoded":
                contrib = float(value) / 2.0  # 0=good, 2=bad
            else:
                contrib = abs(float(value)) / 10.0  # Generic normalization
            
            contributions.append((feat, min(1.0, contrib)))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions
    
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
    
    def get_anomaly_results(self) -> List[AnomalyResult]:
        """Get all anomaly detection results."""
        return self.anomaly_results
    
    def get_flagged_indices(self) -> List[int]:
        """Get indices of flagged records."""
        return self.flagged_indices
    
    def get_high_risk_indices(self) -> List[int]:
        """Get indices of high-risk anomalies."""
        return self.high_risk_indices
    
    def get_anomaly_dataframe(self) -> pd.DataFrame:
        """Get anomaly results as DataFrame."""
        if not self.anomaly_results:
            return pd.DataFrame()
        
        data = []
        for r in self.anomaly_results:
            data.append({
                "record_id": r.record_id,
                "anomaly_score": r.anomaly_score,
                "is_anomaly": r.is_anomaly,
                "anomaly_level": r.anomaly_level,
                "flags": ",".join(r.flags) if r.flags else "",
                "top_contributor": r.contributing_features[0][0] if r.contributing_features else "",
            })
        
        return pd.DataFrame(data)
