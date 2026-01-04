"""
Layer 11: Logging & Trace

Purpose: Comprehensive logging and traceability for the entire pipeline
Type: 100% Deterministic - Logging
Failure Mode: LOG_ERROR → Best effort logging

This layer:
- Creates complete trace logs for each record
- Aggregates all layer results into a unified log
- Provides pipeline execution summary
- Enables debugging and monitoring

Output: TraceLog per record and PipelineExecutionLog for the batch
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..config import LayerStatus, Action
from .layer1_input_contract import LayerResult
from .layer9_decision import Decision
from .layer10_responsibility import ResponsibilityAssignment


@dataclass
class LayerTrace:
    """Trace for a single layer execution."""
    layer_id: float
    layer_name: str
    status: str
    execution_time_ms: float
    checks_performed: int
    checks_passed: int
    issues_count: int
    warnings_count: int


@dataclass
class RecordTrace:
    """Complete trace for a single record."""
    record_id: Any
    trace_id: str
    batch_id: str
    timestamp: str
    
    # Final outcome
    final_action: str
    dqs_score: float
    confidence_band: str
    
    # Responsibility
    owner: str
    requires_review: bool
    
    # Layer traces
    layer_traces: List[LayerTrace]
    
    # Key events
    key_events: List[str]
    
    # Full audit data
    raw_data_hash: str = ""


@dataclass
class PipelineExecutionLog:
    """Complete execution log for the pipeline."""
    batch_id: str
    execution_id: str
    start_time: str
    end_time: str
    total_duration_ms: float
    
    # Record counts
    total_records: int
    processed_records: int
    rejected_records: int
    
    # Layer summary
    layers_executed: int
    layers_passed: int
    layers_failed: int
    
    # Action summary
    action_counts: Dict[str, int]
    
    # Quality metrics
    average_dqs: float
    quality_rate: float
    
    # Issues summary
    total_issues: int
    total_warnings: int
    
    # Individual traces
    record_traces: List[RecordTrace] = field(default_factory=list)


class LoggingTraceLayer:
    """
    Layer 11: Logging & Trace
    
    Provides comprehensive logging and traceability
    for the entire pipeline execution.
    """
    
    LAYER_ID = 11
    LAYER_NAME = "logging_trace"
    
    def __init__(self):
        self.record_traces: List[RecordTrace] = []
        self.execution_log: Optional[PipelineExecutionLog] = None
        self.pipeline_start_time: Optional[datetime] = None
    
    def start_pipeline(self):
        """Mark the start of pipeline execution."""
        self.pipeline_start_time = datetime.now()
    
    def log(
        self,
        layer_results: Dict[float, LayerResult],
        decisions: List[Decision],
        responsibility_assignments: List[ResponsibilityAssignment],
        batch_id: str = None,
        total_records: int = 0,
    ) -> LayerResult:
        """
        Create comprehensive logs for the pipeline execution.
        
        Args:
            layer_results: Dict of layer_id -> LayerResult
            decisions: List of Decision from Layer 9
            responsibility_assignments: Assignments from Layer 10
            batch_id: Batch identifier
            total_records: Total records processed
            
        Returns:
            LayerResult with logging results
        """
        import time
        import uuid
        import hashlib
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
                    details={"message": "No decisions to log"},
                )
            
            self.record_traces = []
            n_records = len(decisions)
            batch_id = batch_id or str(uuid.uuid4())[:8]
            execution_id = str(uuid.uuid4())
            
            # Build responsibility lookup
            resp_lookup = {r.record_id: r for r in responsibility_assignments}
            
            # ================================================================
            # CREATE LAYER TRACES
            # ================================================================
            checks_performed += 1
            
            layer_traces = []
            for lid, result in sorted(layer_results.items()):
                layer_traces.append(LayerTrace(
                    layer_id=lid,
                    layer_name=result.layer_name,
                    status=result.status.value,
                    execution_time_ms=result.execution_time_ms,
                    checks_performed=result.checks_performed,
                    checks_passed=result.checks_passed,
                    issues_count=len(result.issues) if result.issues else 0,
                    warnings_count=len(result.warnings) if result.warnings else 0,
                ))
            
            checks_passed += 1
            
            # ================================================================
            # CREATE RECORD TRACES
            # ================================================================
            checks_performed += 1
            
            for d in decisions:
                resp = resp_lookup.get(d.record_id)
                
                # Generate key events
                key_events = []
                votes = d.layer_votes
                
                if votes.get("L4.1_structural") == "FAIL":
                    key_events.append("Structural validation failed")
                if votes.get("L4.3_semantic") == "FAIL":
                    key_events.append("Semantic rule violation")
                if votes.get("L4.4_anomaly") == "FLAG":
                    key_events.append("Anomaly detected")
                if votes.get("L8_confidence") == "LOW":
                    key_events.append("Low confidence score")
                
                key_events.append(f"Final action: {d.action.value}")
                
                trace = RecordTrace(
                    record_id=d.record_id,
                    trace_id=resp.trace_id if resp else str(uuid.uuid4()),
                    batch_id=batch_id,
                    timestamp=d.decision_timestamp,
                    final_action=d.action.value,
                    dqs_score=d.dqs_final,
                    confidence_band=d.confidence_band,
                    owner=resp.owner.value if resp else "UNKNOWN",
                    requires_review=d.requires_human_review,
                    layer_traces=layer_traces,
                    key_events=key_events,
                    raw_data_hash=hashlib.md5(str(d.record_id).encode()).hexdigest()[:16],
                )
                self.record_traces.append(trace)
            
            checks_passed += 1
            
            # ================================================================
            # CREATE EXECUTION LOG
            # ================================================================
            checks_performed += 1
            
            end_time = datetime.now()
            start_dt = self.pipeline_start_time or end_time
            total_duration = (end_time - start_dt).total_seconds() * 1000
            
            # Layer stats
            layers_passed = sum(1 for t in layer_traces if t.status == "passed")
            layers_failed = sum(1 for t in layer_traces if t.status == "failed")
            
            # Action counts
            action_counts = {a.value: 0 for a in Action}
            for d in decisions:
                action_counts[d.action.value] += 1
            
            # Quality metrics
            dqs_scores = [d.dqs_final for d in decisions]
            avg_dqs = np.mean(dqs_scores) if dqs_scores else 0
            
            safe_count = action_counts.get(Action.SAFE_TO_USE.value, 0)
            quality_rate = safe_count / n_records * 100 if n_records > 0 else 0
            
            # Issue counts
            total_issues = sum(t.issues_count for t in layer_traces)
            total_warnings = sum(t.warnings_count for t in layer_traces)
            
            rejected = action_counts.get(Action.NO_ACTION.value, 0)
            
            self.execution_log = PipelineExecutionLog(
                batch_id=batch_id,
                execution_id=execution_id,
                start_time=start_dt.isoformat(),
                end_time=end_time.isoformat(),
                total_duration_ms=total_duration,
                total_records=total_records or n_records,
                processed_records=n_records,
                rejected_records=rejected,
                layers_executed=len(layer_traces),
                layers_passed=layers_passed,
                layers_failed=layers_failed,
                action_counts=action_counts,
                average_dqs=round(avg_dqs, 2),
                quality_rate=round(quality_rate, 1),
                total_issues=total_issues,
                total_warnings=total_warnings,
                record_traces=self.record_traces,
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
                    "execution_id": execution_id,
                    "batch_id": batch_id,
                    "records_logged": n_records,
                    "layers_logged": len(layer_traces),
                    "total_duration_ms": round(total_duration, 2),
                    "quality_rate": round(quality_rate, 1),
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "LOG_ERROR",
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
    
    def get_record_traces(self) -> List[RecordTrace]:
        """Get all record traces."""
        return self.record_traces
    
    def get_execution_log(self) -> Optional[PipelineExecutionLog]:
        """Get the pipeline execution log."""
        return self.execution_log
    
    def get_traces_dataframe(self) -> pd.DataFrame:
        """Get record traces as DataFrame."""
        if not self.record_traces:
            return pd.DataFrame()
        
        data = []
        for t in self.record_traces:
            data.append({
                "record_id": t.record_id,
                "trace_id": t.trace_id,
                "batch_id": t.batch_id,
                "final_action": t.final_action,
                "dqs_score": t.dqs_score,
                "confidence_band": t.confidence_band,
                "owner": t.owner,
                "requires_review": t.requires_review,
                "key_events": "; ".join(t.key_events),
            })
        
        return pd.DataFrame(data)
    
    def generate_execution_report(self) -> str:
        """Generate a comprehensive execution report."""
        if not self.execution_log:
            return "No execution log available."
        
        log = self.execution_log
        
        report = f"""
============================================================
           PIPELINE EXECUTION REPORT
============================================================

Execution ID: {log.execution_id}
Batch ID: {log.batch_id}
Start Time: {log.start_time}
End Time: {log.end_time}
Total Duration: {log.total_duration_ms:.2f}ms

------------------------------------------------------------
                    RECORD SUMMARY
------------------------------------------------------------
Total Records:     {log.total_records}
Processed:         {log.processed_records}
Rejected:          {log.rejected_records}

------------------------------------------------------------
                    LAYER SUMMARY
------------------------------------------------------------
Layers Executed:   {log.layers_executed}
Layers Passed:     {log.layers_passed}
Layers Failed:     {log.layers_failed}
Total Issues:      {log.total_issues}
Total Warnings:    {log.total_warnings}

------------------------------------------------------------
                    ACTION SUMMARY
------------------------------------------------------------
"""
        for action, count in log.action_counts.items():
            report += f"  {action:20s}: {count}\n"
        
        report += f"""
------------------------------------------------------------
                   QUALITY METRICS
------------------------------------------------------------
Average DQS:       {log.average_dqs}
Quality Rate:      {log.quality_rate}%

============================================================
"""
        
        return report
    
    def export_to_json(self) -> str:
        """Export execution log to JSON."""
        if not self.execution_log:
            return "{}"
        
        log = self.execution_log
        
        # Convert to serializable dict
        data = {
            "batch_id": log.batch_id,
            "execution_id": log.execution_id,
            "start_time": log.start_time,
            "end_time": log.end_time,
            "total_duration_ms": log.total_duration_ms,
            "total_records": log.total_records,
            "processed_records": log.processed_records,
            "rejected_records": log.rejected_records,
            "layers_executed": log.layers_executed,
            "action_counts": log.action_counts,
            "average_dqs": log.average_dqs,
            "quality_rate": log.quality_rate,
            "total_issues": log.total_issues,
            "total_warnings": log.total_warnings,
            "record_count": len(log.record_traces),
        }
        
        return json.dumps(data, indent=2)
