"""
Layer 7: Conflict Detection

Purpose: Detect conflicts between different quality signals
Type: 100% Deterministic - Validation
Failure Mode: CONFLICT_DETECTED → Flag for resolution

This layer detects:
- Rule vs ML conflicts (deterministic says OK, ML flags risk)
- Score vs Priority conflicts
- Cross-layer inconsistencies
- Dimension vs Overall conflicts

Output: Conflict catalog with resolution recommendations
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..config import LayerStatus
from .layer1_input_contract import LayerResult
from .layer5_output_contract import RecordPayload


class ConflictType(Enum):
    """Types of conflicts detected."""
    RULE_ML_CONFLICT = "rule_ml_conflict"
    SCORE_PRIORITY_CONFLICT = "score_priority_conflict"
    DIMENSION_OVERALL_CONFLICT = "dimension_overall_conflict"
    VALIDITY_ANOMALY_CONFLICT = "validity_anomaly_conflict"


class ConflictSeverity(Enum):
    """Severity of conflicts."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Conflict:
    """A detected conflict."""
    record_id: Any
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    signals: Dict[str, Any]  # The conflicting signals
    resolution: str  # Recommended resolution


class ConflictDetectionLayer:
    """
    Layer 7: Conflict Detection
    
    Detects conflicts between different quality signals and
    provides resolution recommendations.
    """
    
    LAYER_ID = 7
    LAYER_NAME = "conflict_detection"
    
    def __init__(self):
        self.conflicts: List[Conflict] = []
        self.conflict_counts: Dict[str, int] = {}
    
    def detect(
        self,
        record_payloads: List[RecordPayload],
        dqs_df: pd.DataFrame = None,
    ) -> LayerResult:
        """
        Detect conflicts between quality signals.
        
        Args:
            record_payloads: List of RecordPayload from Layer 5
            dqs_df: Optional DQS DataFrame with dimension scores
            
        Returns:
            LayerResult with conflict detection results
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
                    details={"message": "No records to check"},
                )
            
            self.conflicts = []
            self.conflict_counts = {ct.value: 0 for ct in ConflictType}
            
            n_records = len(record_payloads)
            
            # ================================================================
            # CHECK 1: Validity vs Anomaly conflicts
            # ================================================================
            checks_performed += 1
            
            for i, r in enumerate(record_payloads):
                # Record is valid but highly anomalous
                if r.is_valid and r.anomaly_score > 0.8:
                    self.conflicts.append(Conflict(
                        record_id=r.record_id,
                        conflict_type=ConflictType.VALIDITY_ANOMALY_CONFLICT,
                        severity=ConflictSeverity.MEDIUM,
                        description="Record passes structural validation but is highly anomalous",
                        signals={
                            "is_valid": r.is_valid,
                            "anomaly_score": r.anomaly_score,
                        },
                        resolution="Review anomaly flags; may indicate new pattern",
                    ))
                    self.conflict_counts[ConflictType.VALIDITY_ANOMALY_CONFLICT.value] += 1
                
                # Record is invalid but not anomalous
                if not r.is_valid and r.anomaly_score < 0.2:
                    self.conflicts.append(Conflict(
                        record_id=r.record_id,
                        conflict_type=ConflictType.VALIDITY_ANOMALY_CONFLICT,
                        severity=ConflictSeverity.LOW,
                        description="Record fails validation but is not flagged as anomaly",
                        signals={
                            "is_valid": r.is_valid,
                            "anomaly_score": r.anomaly_score,
                        },
                        resolution="Structural issues may be data entry errors",
                    ))
                    self.conflict_counts[ConflictType.VALIDITY_ANOMALY_CONFLICT.value] += 1
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 2: Score vs Priority conflicts
            # ================================================================
            checks_performed += 1
            
            for r in record_payloads:
                # High DQS but high priority
                if r.dqs_base > 85 and r.priority in ["critical", "high"]:
                    self.conflicts.append(Conflict(
                        record_id=r.record_id,
                        conflict_type=ConflictType.SCORE_PRIORITY_CONFLICT,
                        severity=ConflictSeverity.MEDIUM,
                        description="High quality score but elevated priority",
                        signals={
                            "dqs_base": r.dqs_base,
                            "priority": r.priority,
                            "anomaly_score": r.anomaly_score,
                        },
                        resolution="Priority likely driven by anomaly; verify flags",
                    ))
                    self.conflict_counts[ConflictType.SCORE_PRIORITY_CONFLICT.value] += 1
                
                # Low DQS but no priority
                if r.dqs_base < 50 and r.priority == "none":
                    self.conflicts.append(Conflict(
                        record_id=r.record_id,
                        conflict_type=ConflictType.SCORE_PRIORITY_CONFLICT,
                        severity=ConflictSeverity.HIGH,
                        description="Low quality score but no priority assigned",
                        signals={
                            "dqs_base": r.dqs_base,
                            "priority": r.priority,
                        },
                        resolution="Escalate for review; priority should be higher",
                    ))
                    self.conflict_counts[ConflictType.SCORE_PRIORITY_CONFLICT.value] += 1
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 3: Rule vs ML conflicts
            # ================================================================
            checks_performed += 1
            
            for r in record_payloads:
                # No semantic violations but high anomaly
                has_violations = len(r.semantic_violations) > 0
                
                if not has_violations and r.is_valid and r.anomaly_score > 0.7:
                    self.conflicts.append(Conflict(
                        record_id=r.record_id,
                        conflict_type=ConflictType.RULE_ML_CONFLICT,
                        severity=ConflictSeverity.MEDIUM,
                        description="Passes all rules but ML flags as anomaly",
                        signals={
                            "semantic_violations": 0,
                            "structural_issues": len(r.structural_issues),
                            "anomaly_score": r.anomaly_score,
                            "anomaly_flags": r.anomaly_flags,
                        },
                        resolution="ML detected pattern not covered by rules; review flags",
                    ))
                    self.conflict_counts[ConflictType.RULE_ML_CONFLICT.value] += 1
                
                # Has violations but low anomaly
                if has_violations and r.anomaly_score < 0.3:
                    self.conflicts.append(Conflict(
                        record_id=r.record_id,
                        conflict_type=ConflictType.RULE_ML_CONFLICT,
                        severity=ConflictSeverity.LOW,
                        description="Rules flag violations but ML shows normal",
                        signals={
                            "semantic_violations": r.semantic_violations,
                            "anomaly_score": r.anomaly_score,
                        },
                        resolution="Rule violations may be systematic; not anomalous",
                    ))
                    self.conflict_counts[ConflictType.RULE_ML_CONFLICT.value] += 1
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 4: Dimension vs Overall conflicts (if DQS data available)
            # ================================================================
            checks_performed += 1
            
            if dqs_df is not None and len(dqs_df) > 0:
                dim_cols = [c for c in dqs_df.columns if c.startswith("dim_")]
                
                for i, r in enumerate(record_payloads):
                    if i >= len(dqs_df):
                        continue
                    
                    dqs_row = dqs_df.iloc[i]
                    overall = dqs_row.get("dqs_base", 100)
                    
                    # Check for single dimension pulling score down
                    for dim_col in dim_cols:
                        dim_score = dqs_row.get(dim_col, 100)
                        dim_name = dim_col.replace("dim_", "")
                        
                        # Dimension much lower than overall
                        if dim_score < 50 and overall > 70:
                            self.conflicts.append(Conflict(
                                record_id=r.record_id,
                                conflict_type=ConflictType.DIMENSION_OVERALL_CONFLICT,
                                severity=ConflictSeverity.LOW,
                                description=f"Low {dim_name} score but overall DQS acceptable",
                                signals={
                                    "dimension": dim_name,
                                    "dimension_score": dim_score,
                                    "overall_dqs": overall,
                                },
                                resolution=f"Address {dim_name} issues specifically",
                            ))
                            self.conflict_counts[ConflictType.DIMENSION_OVERALL_CONFLICT.value] += 1
                            break  # Only flag first dimension conflict per record
            
            checks_passed += 1
            
            # ================================================================
            # RESULT
            # ================================================================
            total_conflicts = len(self.conflicts)
            high_severity = sum(1 for c in self.conflicts if c.severity == ConflictSeverity.HIGH)
            
            if high_severity > 0:
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
                    "total_conflicts": total_conflicts,
                    "high_severity": high_severity,
                    "medium_severity": sum(1 for c in self.conflicts if c.severity == ConflictSeverity.MEDIUM),
                    "low_severity": sum(1 for c in self.conflicts if c.severity == ConflictSeverity.LOW),
                    "conflict_counts": self.conflict_counts,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "CONFLICT_ERROR",
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
    
    def get_conflicts(self) -> List[Conflict]:
        """Get all detected conflicts."""
        return self.conflicts
    
    def get_high_severity_conflicts(self) -> List[Conflict]:
        """Get high severity conflicts."""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.HIGH]
    
    def get_conflicts_dataframe(self) -> pd.DataFrame:
        """Get conflicts as DataFrame."""
        if not self.conflicts:
            return pd.DataFrame()
        
        data = []
        for c in self.conflicts:
            data.append({
                "record_id": c.record_id,
                "conflict_type": c.conflict_type.value,
                "severity": c.severity.value,
                "description": c.description,
                "resolution": c.resolution,
            })
        
        return pd.DataFrame(data)
