"""
Layer 8: Confidence Band

Purpose: Calculate confidence levels for quality assessments
Type: 100% Deterministic - Classification
Failure Mode: CONFIDENCE_ERROR → Default to LOW

This layer:
- Aggregates uncertainty from all layers
- Calculates overall confidence score
- Classifies into confidence bands (HIGH/MEDIUM/LOW)
- Provides confidence-adjusted recommendations

Output: Confidence band assignment per record
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..config import (
    LayerStatus, 
    CONFIDENCE_HIGH_THRESHOLD, 
    CONFIDENCE_MEDIUM_THRESHOLD
)
from .layer1_input_contract import LayerResult
from .layer5_output_contract import RecordPayload
from .layer6_stability import ConsistencyFlag
from .layer7_conflict import Conflict, ConflictSeverity


class ConfidenceBand(Enum):
    """Confidence band classifications."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ConfidenceAssessment:
    """Confidence assessment for a record."""
    record_id: Any
    confidence_score: float  # 0-100
    confidence_band: ConfidenceBand
    contributing_factors: List[str]
    uncertainty_sources: List[str]


class ConfidenceBandLayer:
    """
    Layer 8: Confidence Band
    
    Calculates confidence levels for quality assessments
    based on signal agreement, consistency, and conflicts.
    """
    
    LAYER_ID = 8
    LAYER_NAME = "confidence_band"
    
    # Thresholds (from config)
    HIGH_THRESHOLD = CONFIDENCE_HIGH_THRESHOLD  # 70
    MEDIUM_THRESHOLD = CONFIDENCE_MEDIUM_THRESHOLD  # 30
    
    def __init__(self):
        self.assessments: List[ConfidenceAssessment] = []
        self.batch_confidence: float = 0.0
    
    def assess(
        self,
        record_payloads: List[RecordPayload],
        consistency_flags: List[ConsistencyFlag] = None,
        conflicts: List[Conflict] = None,
        stability_score: float = 100.0,
    ) -> LayerResult:
        """
        Assess confidence for all records.
        
        Args:
            record_payloads: List of RecordPayload from Layer 5
            consistency_flags: Flags from Layer 6
            conflicts: Conflicts from Layer 7
            stability_score: Overall stability score from Layer 6
            
        Returns:
            LayerResult with confidence assessments
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
                    details={"message": "No records to assess"},
                )
            
            self.assessments = []
            n_records = len(record_payloads)
            
            # Build lookup for flags and conflicts
            flags_by_record = {}
            if consistency_flags:
                for f in consistency_flags:
                    rid = f.record_id
                    if rid not in flags_by_record:
                        flags_by_record[rid] = []
                    flags_by_record[rid].append(f)
            
            conflicts_by_record = {}
            if conflicts:
                for c in conflicts:
                    rid = c.record_id
                    if rid not in conflicts_by_record:
                        conflicts_by_record[rid] = []
                    conflicts_by_record[rid].append(c)
            
            # ================================================================
            # ASSESS EACH RECORD
            # ================================================================
            checks_performed += 1
            confidence_scores = []
            
            for r in record_payloads:
                contributing = []
                uncertainty = []
                score = 100.0
                
                # Factor 1: DQS score level (20% weight)
                if r.dqs_base >= 90:
                    contributing.append("High DQS score")
                elif r.dqs_base >= 70:
                    score -= 10
                elif r.dqs_base >= 50:
                    score -= 25
                    uncertainty.append("Moderate DQS")
                else:
                    score -= 40
                    uncertainty.append("Low DQS")
                
                # Factor 2: Anomaly score (20% weight)
                if r.anomaly_score < 0.3:
                    contributing.append("Low anomaly")
                elif r.anomaly_score < 0.5:
                    score -= 10
                elif r.anomaly_score < 0.7:
                    score -= 20
                    uncertainty.append("Elevated anomaly")
                else:
                    score -= 30
                    uncertainty.append("High anomaly")
                
                # Factor 3: No structural issues (15% weight)
                if not r.structural_issues:
                    contributing.append("No structural issues")
                else:
                    score -= 15 * min(len(r.structural_issues), 2)
                    uncertainty.append("Structural issues")
                
                # Factor 4: No semantic violations (15% weight)
                if not r.semantic_violations:
                    contributing.append("No rule violations")
                else:
                    score -= 15 * min(len(r.semantic_violations), 2)
                    uncertainty.append("Rule violations")
                
                # Factor 5: Consistency flags (15% weight)
                record_flags = flags_by_record.get(r.record_id, [])
                if not record_flags:
                    contributing.append("Consistent signals")
                else:
                    score -= 10 * min(len(record_flags), 3)
                    uncertainty.append(f"{len(record_flags)} consistency flags")
                
                # Factor 6: Conflicts (15% weight)
                record_conflicts = conflicts_by_record.get(r.record_id, [])
                high_conflicts = sum(1 for c in record_conflicts if c.severity == ConflictSeverity.HIGH)
                if not record_conflicts:
                    contributing.append("No conflicts")
                else:
                    score -= 5 * len(record_conflicts)
                    score -= 10 * high_conflicts
                    uncertainty.append(f"{len(record_conflicts)} conflicts")
                
                # Apply stability adjustment
                stability_factor = stability_score / 100.0
                score = score * (0.8 + 0.2 * stability_factor)
                
                # Clamp score
                score = max(0, min(100, score))
                
                # Determine band
                if score >= self.HIGH_THRESHOLD:
                    band = ConfidenceBand.HIGH
                elif score >= self.MEDIUM_THRESHOLD:
                    band = ConfidenceBand.MEDIUM
                else:
                    band = ConfidenceBand.LOW
                
                self.assessments.append(ConfidenceAssessment(
                    record_id=r.record_id,
                    confidence_score=round(score, 2),
                    confidence_band=band,
                    contributing_factors=contributing[:5],
                    uncertainty_sources=uncertainty[:5],
                ))
                
                confidence_scores.append(score)
            
            checks_passed += 1
            
            # ================================================================
            # BATCH METRICS
            # ================================================================
            checks_performed += 1
            
            self.batch_confidence = np.mean(confidence_scores)
            
            band_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for a in self.assessments:
                band_counts[a.confidence_band.value] += 1
            
            checks_passed += 1
            
            # ================================================================
            # RESULT
            # ================================================================
            low_confidence_pct = band_counts["LOW"] / n_records * 100
            
            if low_confidence_pct > 50:
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
                    "batch_confidence": round(self.batch_confidence, 2),
                    "band_counts": band_counts,
                    "high_confidence_pct": round(band_counts["HIGH"] / n_records * 100, 1),
                    "low_confidence_pct": round(low_confidence_pct, 1),
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "CONFIDENCE_ERROR",
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
    
    def get_assessments(self) -> List[ConfidenceAssessment]:
        """Get all confidence assessments."""
        return self.assessments
    
    def get_low_confidence_records(self) -> List[ConfidenceAssessment]:
        """Get records with low confidence."""
        return [a for a in self.assessments if a.confidence_band == ConfidenceBand.LOW]
    
    def get_assessments_dataframe(self) -> pd.DataFrame:
        """Get assessments as DataFrame."""
        if not self.assessments:
            return pd.DataFrame()
        
        data = []
        for a in self.assessments:
            data.append({
                "record_id": a.record_id,
                "confidence_score": a.confidence_score,
                "confidence_band": a.confidence_band.value,
                "contributing_factors": "; ".join(a.contributing_factors),
                "uncertainty_sources": "; ".join(a.uncertainty_sources),
            })
        
        return pd.DataFrame(data)
