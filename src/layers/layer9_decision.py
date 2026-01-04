"""
Layer 9: Decision Gate

Purpose: Make final action decisions based on all signals
Type: 100% Deterministic - FINAL DECISION
Failure Mode: DECISION_ERROR → Default to REVIEW_REQUIRED

This is the FINAL layer that:
- Aggregates all quality signals
- Applies decision rules
- Assigns one of 4 actions: SAFE_TO_USE, REVIEW_REQUIRED, ESCALATE, NO_ACTION
- Provides audit trail for decisions

CRITICAL: "Rules enforce, ML informs, Humans decide"
- This layer enforces rules
- ML signals inform but don't override
- Clear escalation paths for human decision
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..config import (
    LayerStatus, 
    Action,
    DQS_CRITICAL_THRESHOLD,
    DQS_BORDERLINE_THRESHOLD,
)
from .layer1_input_contract import LayerResult
from .layer5_output_contract import RecordPayload
from .layer8_confidence import ConfidenceAssessment, ConfidenceBand


@dataclass
class Decision:
    """Final decision for a record."""
    record_id: Any
    action: Action
    dqs_final: float
    confidence_band: str
    
    # Decision reasoning
    primary_reason: str
    supporting_factors: List[str]
    
    # Audit trail
    decision_timestamp: str
    layer_votes: Dict[str, str]  # How each layer "voted"
    
    # Human review info
    requires_human_review: bool
    escalation_reason: str = ""


@dataclass
class BatchDecision:
    """Batch-level decision summary."""
    batch_id: str
    timestamp: str
    total_records: int
    
    # Action counts
    safe_count: int
    review_count: int
    escalate_count: int
    no_action_count: int
    
    # Quality metrics
    overall_quality_rate: float
    requires_human_review: bool
    
    # Individual decisions
    decisions: List[Decision] = field(default_factory=list)


class DecisionGateLayer:
    """
    Layer 9: Decision Gate
    
    Makes final action decisions based on all accumulated signals.
    This is the terminal layer that produces actionable output.
    """
    
    LAYER_ID = 9
    LAYER_NAME = "decision_gate"
    
    def __init__(self):
        self.decisions: List[Decision] = []
        self.batch_decision: Optional[BatchDecision] = None
    
    def decide(
        self,
        record_payloads: List[RecordPayload],
        confidence_assessments: List[ConfidenceAssessment],
        batch_id: str = None,
    ) -> LayerResult:
        """
        Make final decisions for all records.
        
        Args:
            record_payloads: List of RecordPayload from Layer 5
            confidence_assessments: Assessments from Layer 8
            batch_id: Optional batch identifier
            
        Returns:
            LayerResult with final decisions
        """
        import time
        import uuid
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
                    details={"message": "No records to decide"},
                )
            
            self.decisions = []
            n_records = len(record_payloads)
            
            # Build confidence lookup
            confidence_lookup = {}
            for ca in confidence_assessments:
                confidence_lookup[ca.record_id] = ca
            
            # ================================================================
            # MAKE DECISIONS FOR EACH RECORD
            # ================================================================
            checks_performed += 1
            
            for r in record_payloads:
                # Get confidence
                ca = confidence_lookup.get(r.record_id)
                confidence_band = ca.confidence_band.value if ca else "MEDIUM"
                confidence_score = ca.confidence_score if ca else 50.0
                
                # Determine action
                action, reason, factors, escalation = self._determine_action(
                    r, confidence_band, confidence_score
                )
                
                # Build layer votes
                layer_votes = {
                    "L4.1_structural": "PASS" if r.is_valid else "FAIL",
                    "L4.2_dqs": "PASS" if r.dqs_base >= DQS_BORDERLINE_THRESHOLD else "FAIL",
                    "L4.3_semantic": "PASS" if not r.semantic_violations else "FAIL",
                    "L4.4_anomaly": "FLAG" if r.is_anomaly else "PASS",
                    "L8_confidence": confidence_band,
                }
                
                decision = Decision(
                    record_id=r.record_id,
                    action=action,
                    dqs_final=r.dqs_base,
                    confidence_band=confidence_band,
                    primary_reason=reason,
                    supporting_factors=factors,
                    decision_timestamp=datetime.now().isoformat(),
                    layer_votes=layer_votes,
                    requires_human_review=action in [Action.REVIEW_REQUIRED, Action.ESCALATE],
                    escalation_reason=escalation,
                )
                self.decisions.append(decision)
            
            checks_passed += 1
            
            # ================================================================
            # BUILD BATCH DECISION
            # ================================================================
            checks_performed += 1
            
            safe_count = sum(1 for d in self.decisions if d.action == Action.SAFE_TO_USE)
            review_count = sum(1 for d in self.decisions if d.action == Action.REVIEW_REQUIRED)
            escalate_count = sum(1 for d in self.decisions if d.action == Action.ESCALATE)
            no_action_count = sum(1 for d in self.decisions if d.action == Action.NO_ACTION)
            
            quality_rate = safe_count / n_records * 100 if n_records > 0 else 0
            requires_human = review_count > 0 or escalate_count > 0
            
            self.batch_decision = BatchDecision(
                batch_id=batch_id or str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
                total_records=n_records,
                safe_count=safe_count,
                review_count=review_count,
                escalate_count=escalate_count,
                no_action_count=no_action_count,
                overall_quality_rate=quality_rate,
                requires_human_review=requires_human,
                decisions=self.decisions,
            )
            
            checks_passed += 1
            
            # ================================================================
            # RESULT
            # ================================================================
            if escalate_count > 0:
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
                    "batch_id": self.batch_decision.batch_id,
                    "safe_count": safe_count,
                    "review_count": review_count,
                    "escalate_count": escalate_count,
                    "no_action_count": no_action_count,
                    "quality_rate": round(quality_rate, 1),
                    "requires_human_review": requires_human,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "DECISION_ERROR",
                "message": f"Unexpected error: {str(e)}",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                can_continue=False,
            )
    
    def _determine_action(
        self,
        r: RecordPayload,
        confidence_band: str,
        confidence_score: float,
    ) -> Tuple[Action, str, List[str], str]:
        """Determine action for a record."""
        factors = []
        escalation = ""
        
        # ================================================================
        # RULE 1: Invalid records -> NO_ACTION (already rejected)
        # ================================================================
        if not r.is_valid:
            return (
                Action.NO_ACTION,
                "Record failed structural validation",
                ["Rejected in Layer 4.1"],
                "Structural integrity failure",
            )
        
        # ================================================================
        # RULE 2: Critical DQS -> ESCALATE
        # ================================================================
        if r.dqs_base < DQS_CRITICAL_THRESHOLD:
            return (
                Action.ESCALATE,
                f"Critical DQS score ({r.dqs_base:.1f})",
                ["DQS below critical threshold"],
                "Quality score indicates critical data issues",
            )
        
        # ================================================================
        # RULE 3: Semantic violations -> ESCALATE
        # ================================================================
        if r.semantic_violations:
            return (
                Action.ESCALATE,
                "Business rule violations detected",
                r.semantic_violations[:3],
                "Critical business rule violations",
            )
        
        # ================================================================
        # RULE 4: Borderline DQS -> REVIEW_REQUIRED
        # ================================================================
        if r.dqs_base < DQS_BORDERLINE_THRESHOLD:
            factors.append(f"DQS {r.dqs_base:.1f} below borderline")
            if r.is_anomaly:
                factors.append(f"Anomaly score {r.anomaly_score:.2f}")
            return (
                Action.REVIEW_REQUIRED,
                f"Borderline quality score ({r.dqs_base:.1f})",
                factors,
                "",
            )
        
        # ================================================================
        # RULE 5: High anomaly with low confidence -> REVIEW_REQUIRED
        # ================================================================
        if r.is_anomaly and confidence_band != "HIGH":
            factors.append(f"Anomaly detected: {', '.join(r.anomaly_flags[:2])}")
            factors.append(f"Confidence: {confidence_band}")
            return (
                Action.REVIEW_REQUIRED,
                "Anomaly detected with uncertain confidence",
                factors,
                "",
            )
        
        # ================================================================
        # RULE 6: Low confidence -> REVIEW_REQUIRED
        # ================================================================
        if confidence_band == "LOW":
            factors.append(f"Low confidence score ({confidence_score:.1f})")
            return (
                Action.REVIEW_REQUIRED,
                "Low confidence in quality assessment",
                factors,
                "",
            )
        
        # ================================================================
        # RULE 7: High anomaly even with high confidence -> REVIEW
        # ================================================================
        if r.anomaly_score > 0.8:
            factors.append(f"Very high anomaly: {r.anomaly_score:.2f}")
            return (
                Action.REVIEW_REQUIRED,
                "Very high anomaly score requires review",
                factors,
                "",
            )
        
        # ================================================================
        # DEFAULT: SAFE_TO_USE
        # ================================================================
        factors.append(f"DQS: {r.dqs_base:.1f}")
        factors.append(f"Confidence: {confidence_band}")
        if not r.is_anomaly:
            factors.append("No anomalies")
        
        return (
            Action.SAFE_TO_USE,
            "Record passes all quality checks",
            factors,
            "",
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
    
    def get_decisions(self) -> List[Decision]:
        """Get all decisions."""
        return self.decisions
    
    def get_batch_decision(self) -> Optional[BatchDecision]:
        """Get batch decision summary."""
        return self.batch_decision
    
    def get_decisions_dataframe(self) -> pd.DataFrame:
        """Get decisions as DataFrame."""
        if not self.decisions:
            return pd.DataFrame()
        
        data = []
        for d in self.decisions:
            data.append({
                "record_id": d.record_id,
                "action": d.action.value,
                "dqs_final": d.dqs_final,
                "confidence_band": d.confidence_band,
                "primary_reason": d.primary_reason,
                "requires_review": d.requires_human_review,
            })
        
        return pd.DataFrame(data)
    
    def generate_decision_report(self) -> str:
        """Generate a decision summary report."""
        if not self.batch_decision:
            return "No decisions made."
        
        bd = self.batch_decision
        
        report = f"""
============================================================
              FINAL DECISION REPORT
============================================================

Batch ID: {bd.batch_id}
Timestamp: {bd.timestamp}
Total Records: {bd.total_records}

ACTION SUMMARY:
  [OK]  SAFE_TO_USE:     {bd.safe_count:5d}
  [??]  REVIEW_REQUIRED: {bd.review_count:5d}
  [!!]  ESCALATE:        {bd.escalate_count:5d}
  [--]  NO_ACTION:       {bd.no_action_count:5d}

Overall Quality Rate: {bd.overall_quality_rate:.1f}%
Human Review Required: {"YES" if bd.requires_human_review else "NO"}

============================================================
"""
        
        # Add escalation details
        escalated = [d for d in self.decisions if d.action == Action.ESCALATE]
        if escalated:
            report += "\n[!!] ESCALATED RECORDS:\n"
            for d in escalated[:10]:
                report += f"  - {d.record_id}: {d.primary_reason}\n"
        
        # Add review details
        reviews = [d for d in self.decisions if d.action == Action.REVIEW_REQUIRED]
        if reviews:
            report += f"\n[??] RECORDS REQUIRING REVIEW: {len(reviews)}\n"
            for d in reviews[:5]:
                report += f"  - {d.record_id}: {d.primary_reason}\n"
        
        return report
