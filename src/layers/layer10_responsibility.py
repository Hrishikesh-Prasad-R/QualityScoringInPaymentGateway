"""
Layer 10: Responsibility Boundary

Purpose: Track responsibility for decisions and provide audit trail
Type: 100% Deterministic - Tracking
Failure Mode: RESPONSIBILITY_ERROR → Log and continue

This layer:
- Tracks which system component made which decision
- Assigns responsibility labels to each decision
- Creates an audit trail for accountability
- Identifies human vs automated decisions

Output: ResponsibilityAssignment per record with full audit trail
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import LayerStatus, Action
from .layer1_input_contract import LayerResult
from .layer9_decision import Decision


class ResponsibilityOwner(Enum):
    """Who is responsible for the decision."""
    SYSTEM_AUTOMATED = "SYSTEM_AUTOMATED"  # Fully automated decision
    SYSTEM_FLAGGED = "SYSTEM_FLAGGED"      # System flags for human review
    HUMAN_REQUIRED = "HUMAN_REQUIRED"       # Human decision required
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"       # Human overrode system


class DecisionSource(Enum):
    """Source of the decision."""
    DETERMINISTIC_RULES = "DETERMINISTIC_RULES"  # Hard rules (L4.1-4.3)
    ML_INFORMED = "ML_INFORMED"                   # ML contributed (L4.4)
    CONFIDENCE_BASED = "CONFIDENCE_BASED"         # Confidence drove it (L8)
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"   # Conflict needed resolution


@dataclass
class ResponsibilityAssignment:
    """Responsibility assignment for a decision."""
    record_id: Any
    
    # Decision info
    action: Action
    decision_timestamp: str
    
    # Responsibility
    owner: ResponsibilityOwner
    source: DecisionSource
    
    # Audit trail
    layer_contributions: Dict[str, str]  # Layer -> contribution
    determining_factors: List[str]       # What drove the decision
    confidence_level: str                # HIGH/MEDIUM/LOW
    
    # Accountability
    requires_human_sign_off: bool
    escalation_path: str
    review_deadline_hours: int = 24
    
    # Metadata
    batch_id: str = ""
    trace_id: str = ""


@dataclass
class BatchResponsibility:
    """Batch-level responsibility summary."""
    batch_id: str
    timestamp: str
    
    # Counts
    total_decisions: int
    automated_count: int
    human_required_count: int
    
    # Responsibility breakdown
    by_owner: Dict[str, int]
    by_source: Dict[str, int]
    
    # Review requirements
    pending_reviews: int
    average_review_deadline_hours: float
    
    # Individual assignments
    assignments: List[ResponsibilityAssignment] = field(default_factory=list)


class ResponsibilityBoundaryLayer:
    """
    Layer 10: Responsibility Boundary
    
    Tracks and assigns responsibility for each decision,
    creating a complete audit trail.
    """
    
    LAYER_ID = 10
    LAYER_NAME = "responsibility_boundary"
    
    # Review deadlines by action
    REVIEW_DEADLINES = {
        Action.ESCALATE: 4,        # 4 hours for escalated
        Action.REVIEW_REQUIRED: 24, # 24 hours for review
        Action.SAFE_TO_USE: 0,     # No review needed
        Action.NO_ACTION: 0,       # No review needed
    }
    
    def __init__(self):
        self.assignments: List[ResponsibilityAssignment] = []
        self.batch_responsibility: Optional[BatchResponsibility] = None
    
    def assign(
        self,
        decisions: List[Decision],
        batch_id: str = None,
    ) -> LayerResult:
        """
        Assign responsibility for each decision.
        
        Args:
            decisions: List of Decision from Layer 9
            batch_id: Batch identifier
            
        Returns:
            LayerResult with responsibility assignments
        """
        import time
        import uuid
        start_time = time.time()
        
        issues = []
        warnings = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            if not decisions:
                return self._create_result(
                    status=LayerStatus.PASSED,
                    start_time=start_time,
                    checks_performed=1,
                    checks_passed=1,
                    issues=[],
                    can_continue=True,
                    details={"message": "No decisions to assign"},
                )
            
            self.assignments = []
            n_decisions = len(decisions)
            batch_id = batch_id or str(uuid.uuid4())[:8]
            
            # ================================================================
            # ASSIGN RESPONSIBILITY FOR EACH DECISION
            # ================================================================
            checks_performed += 1
            
            for d in decisions:
                # Determine owner
                owner = self._determine_owner(d)
                
                # Determine source
                source = self._determine_source(d)
                
                # Get layer contributions
                contributions = self._get_layer_contributions(d)
                
                # Get determining factors
                factors = d.supporting_factors if d.supporting_factors else [d.primary_reason]
                
                # Get escalation path
                escalation_path = self._get_escalation_path(d)
                
                # Get review deadline
                deadline = self.REVIEW_DEADLINES.get(d.action, 24)
                
                assignment = ResponsibilityAssignment(
                    record_id=d.record_id,
                    action=d.action,
                    decision_timestamp=d.decision_timestamp,
                    owner=owner,
                    source=source,
                    layer_contributions=contributions,
                    determining_factors=factors[:5],
                    confidence_level=d.confidence_band,
                    requires_human_sign_off=d.requires_human_review,
                    escalation_path=escalation_path,
                    review_deadline_hours=deadline,
                    batch_id=batch_id,
                    trace_id=str(uuid.uuid4()),
                )
                self.assignments.append(assignment)
            
            checks_passed += 1
            
            # ================================================================
            # BUILD BATCH SUMMARY
            # ================================================================
            checks_performed += 1
            
            by_owner = {o.value: 0 for o in ResponsibilityOwner}
            by_source = {s.value: 0 for s in DecisionSource}
            
            for a in self.assignments:
                by_owner[a.owner.value] += 1
                by_source[a.source.value] += 1
            
            automated = by_owner[ResponsibilityOwner.SYSTEM_AUTOMATED.value]
            human_required = (
                by_owner[ResponsibilityOwner.HUMAN_REQUIRED.value] +
                by_owner[ResponsibilityOwner.SYSTEM_FLAGGED.value]
            )
            
            pending = sum(1 for a in self.assignments if a.requires_human_sign_off)
            avg_deadline = np.mean([a.review_deadline_hours for a in self.assignments if a.requires_human_sign_off]) if pending > 0 else 0
            
            self.batch_responsibility = BatchResponsibility(
                batch_id=batch_id,
                timestamp=datetime.now().isoformat(),
                total_decisions=n_decisions,
                automated_count=automated,
                human_required_count=human_required,
                by_owner=by_owner,
                by_source=by_source,
                pending_reviews=pending,
                average_review_deadline_hours=avg_deadline,
                assignments=self.assignments,
            )
            
            checks_passed += 1
            
            # ================================================================
            # RESULT
            # ================================================================
            return self._create_result(
                status=LayerStatus.PASSED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=True,
                details={
                    "total_decisions": n_decisions,
                    "automated_count": automated,
                    "human_required_count": human_required,
                    "pending_reviews": pending,
                    "by_owner": by_owner,
                    "by_source": by_source,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "RESPONSIBILITY_ERROR",
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
    
    def _determine_owner(self, d: Decision) -> ResponsibilityOwner:
        """Determine who owns this decision."""
        if d.action == Action.SAFE_TO_USE:
            return ResponsibilityOwner.SYSTEM_AUTOMATED
        elif d.action == Action.NO_ACTION:
            return ResponsibilityOwner.SYSTEM_AUTOMATED
        elif d.action == Action.ESCALATE:
            return ResponsibilityOwner.HUMAN_REQUIRED
        else:  # REVIEW_REQUIRED
            return ResponsibilityOwner.SYSTEM_FLAGGED
    
    def _determine_source(self, d: Decision) -> DecisionSource:
        """Determine the source of the decision."""
        votes = d.layer_votes
        
        # Check if ML informed the decision
        if votes.get("L4.4_anomaly") == "FLAG":
            return DecisionSource.ML_INFORMED
        
        # Check if confidence was the driver
        if votes.get("L8_confidence") == "LOW":
            return DecisionSource.CONFIDENCE_BASED
        
        # Check if rules drove it
        if votes.get("L4.3_semantic") == "FAIL" or votes.get("L4.1_structural") == "FAIL":
            return DecisionSource.DETERMINISTIC_RULES
        
        # Default to deterministic
        return DecisionSource.DETERMINISTIC_RULES
    
    def _get_layer_contributions(self, d: Decision) -> Dict[str, str]:
        """Get contribution from each layer."""
        contributions = {}
        votes = d.layer_votes
        
        if votes.get("L4.1_structural") == "FAIL":
            contributions["L4.1"] = "REJECTED - Structural failure"
        elif votes.get("L4.1_structural") == "PASS":
            contributions["L4.1"] = "PASSED - Valid structure"
        
        if votes.get("L4.2_dqs") == "FAIL":
            contributions["L4.2"] = "FLAGGED - Low DQS"
        elif votes.get("L4.2_dqs") == "PASS":
            contributions["L4.2"] = "PASSED - Acceptable DQS"
        
        if votes.get("L4.3_semantic") == "FAIL":
            contributions["L4.3"] = "REJECTED - Rule violations"
        elif votes.get("L4.3_semantic") == "PASS":
            contributions["L4.3"] = "PASSED - No violations"
        
        if votes.get("L4.4_anomaly") == "FLAG":
            contributions["L4.4"] = "FLAGGED - Anomaly detected"
        elif votes.get("L4.4_anomaly") == "PASS":
            contributions["L4.4"] = "PASSED - No anomaly"
        
        contributions["L8"] = f"Confidence: {votes.get('L8_confidence', 'N/A')}"
        
        return contributions
    
    def _get_escalation_path(self, d: Decision) -> str:
        """Get escalation path for the decision."""
        if d.action == Action.ESCALATE:
            return "Immediate escalation to senior reviewer required"
        elif d.action == Action.REVIEW_REQUIRED:
            return "Route to quality review queue"
        else:
            return "No escalation needed"
    
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
    
    def get_assignments(self) -> List[ResponsibilityAssignment]:
        """Get all responsibility assignments."""
        return self.assignments
    
    def get_batch_responsibility(self) -> Optional[BatchResponsibility]:
        """Get batch responsibility summary."""
        return self.batch_responsibility
    
    def get_assignments_dataframe(self) -> pd.DataFrame:
        """Get assignments as DataFrame."""
        if not self.assignments:
            return pd.DataFrame()
        
        data = []
        for a in self.assignments:
            data.append({
                "record_id": a.record_id,
                "action": a.action.value,
                "owner": a.owner.value,
                "source": a.source.value,
                "confidence": a.confidence_level,
                "requires_review": a.requires_human_sign_off,
                "deadline_hours": a.review_deadline_hours,
                "trace_id": a.trace_id,
            })
        
        return pd.DataFrame(data)
    
    def get_pending_reviews(self) -> List[ResponsibilityAssignment]:
        """Get assignments pending human review."""
        return [a for a in self.assignments if a.requires_human_sign_off]
