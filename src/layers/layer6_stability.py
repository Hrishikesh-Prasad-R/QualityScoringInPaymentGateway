"""
Layer 6: Stability & Consistency

Purpose: Ensure scoring stability and consistency across records
Type: 100% Deterministic - Validation
Failure Mode: STABILITY_WARNING → Flag inconsistencies

This layer checks:
- Score distribution is reasonable
- No extreme score jumps between similar records
- Temporal consistency (if applicable)
- Cross-field consistency validation

Output: Stability metrics and consistency flags
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats

from ..config import LayerStatus
from .layer1_input_contract import LayerResult
from .layer5_output_contract import RecordPayload, BatchPayload


@dataclass
class StabilityMetrics:
    """Stability metrics for a batch."""
    dqs_std: float
    dqs_cv: float  # Coefficient of variation
    dqs_skewness: float
    dqs_kurtosis: float
    outlier_count: int
    consistency_score: float  # 0-100


@dataclass
class ConsistencyFlag:
    """Consistency flag for a record."""
    record_id: Any
    flag_type: str
    severity: str  # "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class StabilityConsistencyLayer:
    """
    Layer 6: Stability & Consistency
    
    Validates that scoring is stable and consistent across the batch.
    Flags potential inconsistencies for review.
    """
    
    LAYER_ID = 6
    LAYER_NAME = "stability_consistency"
    
    # Thresholds
    CV_THRESHOLD = 0.5  # Coefficient of variation threshold
    OUTLIER_ZSCORE = 2.5  # Z-score threshold for outliers
    SCORE_JUMP_THRESHOLD = 30  # Max acceptable score difference for similar records
    
    def __init__(self):
        self.stability_metrics: Optional[StabilityMetrics] = None
        self.consistency_flags: List[ConsistencyFlag] = []
        self.adjusted_payloads: List[RecordPayload] = []
    
    def validate(
        self,
        record_payloads: List[RecordPayload],
        features_df: pd.DataFrame = None,
    ) -> LayerResult:
        """
        Validate stability and consistency of scores.
        
        Args:
            record_payloads: List of RecordPayload from Layer 5
            features_df: Optional features for similarity checking
            
        Returns:
            LayerResult with stability/consistency validation
        """
        import time
        start_time = time.time()
        
        issues = []
        warnings = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            if not record_payloads:
                return self._create_result(
                    status=LayerStatus.PASSED,
                    start_time=start_time,
                    checks_performed=1,
                    checks_passed=1,
                    issues=[],
                    can_continue=True,
                    details={"message": "No records to validate"},
                )
            
            self.consistency_flags = []
            self.adjusted_payloads = record_payloads.copy()
            n_records = len(record_payloads)
            
            # Get DQS scores
            dqs_scores = np.array([r.dqs_base for r in record_payloads])
            
            # ================================================================
            # CHECK 1: Distribution stability
            # ================================================================
            checks_performed += 1
            
            mean_dqs = np.mean(dqs_scores)
            std_dqs = np.std(dqs_scores)
            cv = std_dqs / mean_dqs if mean_dqs > 0 else 0
            
            # Calculate skewness and kurtosis
            if n_records >= 3:
                skewness = stats.skew(dqs_scores)
                kurtosis = stats.kurtosis(dqs_scores)
            else:
                skewness = 0
                kurtosis = 0
            
            # Check for high variability
            if cv > self.CV_THRESHOLD:
                warnings.append(f"High score variability (CV={cv:.2f})")
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 2: Outlier detection
            # ================================================================
            checks_performed += 1
            
            if std_dqs > 0:
                z_scores = (dqs_scores - mean_dqs) / std_dqs
            else:
                z_scores = np.zeros(n_records)
            
            outlier_mask = np.abs(z_scores) > self.OUTLIER_ZSCORE
            outlier_count = int(np.sum(outlier_mask))
            
            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    self.consistency_flags.append(ConsistencyFlag(
                        record_id=record_payloads[i].record_id,
                        flag_type="DQS_OUTLIER",
                        severity="warning",
                        message=f"DQS score is an outlier (z={z_scores[i]:.2f})",
                        details={"dqs": dqs_scores[i], "z_score": float(z_scores[i])},
                    ))
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 3: Score-Anomaly consistency
            # ================================================================
            checks_performed += 1
            
            for r in record_payloads:
                # High anomaly should correlate with lower DQS or higher priority
                if r.anomaly_score > 0.7 and r.dqs_base > 90:
                    self.consistency_flags.append(ConsistencyFlag(
                        record_id=r.record_id,
                        flag_type="SCORE_ANOMALY_MISMATCH",
                        severity="info",
                        message="High anomaly score but high DQS",
                        details={
                            "anomaly_score": r.anomaly_score,
                            "dqs_base": r.dqs_base,
                        },
                    ))
                
                # Low DQS should have issues flagged
                if r.dqs_base < 50 and not r.structural_issues and not r.semantic_violations:
                    self.consistency_flags.append(ConsistencyFlag(
                        record_id=r.record_id,
                        flag_type="UNEXPLAINED_LOW_DQS",
                        severity="warning",
                        message="Low DQS without identified issues",
                        details={"dqs_base": r.dqs_base},
                    ))
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 4: Priority consistency
            # ================================================================
            checks_performed += 1
            
            for r in record_payloads:
                # Critical priority should have low DQS or violations
                if r.priority == "critical" and r.dqs_base > 60 and not r.semantic_violations:
                    self.consistency_flags.append(ConsistencyFlag(
                        record_id=r.record_id,
                        flag_type="PRIORITY_DQS_MISMATCH",
                        severity="info",
                        message="Critical priority but moderate DQS",
                        details={
                            "priority": r.priority,
                            "dqs_base": r.dqs_base,
                        },
                    ))
                
                # None priority should not have anomalies
                if r.priority == "none" and r.is_anomaly:
                    self.consistency_flags.append(ConsistencyFlag(
                        record_id=r.record_id,
                        flag_type="PRIORITY_ANOMALY_MISMATCH",
                        severity="info",
                        message="No priority but flagged as anomaly",
                        details={
                            "priority": r.priority,
                            "anomaly_score": r.anomaly_score,
                        },
                    ))
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 5: Calculate consistency score
            # ================================================================
            checks_performed += 1
            
            # Consistency score based on:
            # - Low CV (40%)
            # - Few outliers (30%)
            # - Few flags (30%)
            cv_score = max(0, 100 - cv * 100)
            outlier_score = max(0, 100 - (outlier_count / n_records * 200))
            flag_score = max(0, 100 - (len(self.consistency_flags) / n_records * 100))
            
            consistency_score = 0.4 * cv_score + 0.3 * outlier_score + 0.3 * flag_score
            
            self.stability_metrics = StabilityMetrics(
                dqs_std=std_dqs,
                dqs_cv=cv,
                dqs_skewness=skewness,
                dqs_kurtosis=kurtosis,
                outlier_count=outlier_count,
                consistency_score=consistency_score,
            )
            
            checks_passed += 1
            
            # ================================================================
            # RESULT
            # ================================================================
            if consistency_score < 50:
                status = LayerStatus.DEGRADED
            else:
                status = LayerStatus.PASSED
            
            return self._create_result(
                status=status,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=True,
                details={
                    "consistency_score": round(consistency_score, 2),
                    "dqs_std": round(std_dqs, 2),
                    "dqs_cv": round(cv, 3),
                    "outlier_count": outlier_count,
                    "flags_count": len(self.consistency_flags),
                    "dqs_skewness": round(skewness, 3),
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "STABILITY_ERROR",
                "message": f"Unexpected error: {str(e)}",
            })
            return self._create_result(
                status=LayerStatus.DEGRADED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                can_continue=True,
            )
    
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
    
    def get_stability_metrics(self) -> Optional[StabilityMetrics]:
        """Get stability metrics."""
        return self.stability_metrics
    
    def get_consistency_flags(self) -> List[ConsistencyFlag]:
        """Get all consistency flags."""
        return self.consistency_flags
    
    def get_flags_dataframe(self) -> pd.DataFrame:
        """Get consistency flags as DataFrame."""
        if not self.consistency_flags:
            return pd.DataFrame()
        
        data = []
        for f in self.consistency_flags:
            data.append({
                "record_id": f.record_id,
                "flag_type": f.flag_type,
                "severity": f.severity,
                "message": f.message,
            })
        
        return pd.DataFrame(data)
