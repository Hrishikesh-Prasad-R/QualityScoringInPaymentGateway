"""
Layer 5: Output Contract

Purpose: Validate and structure the output of all previous layers
Type: 100% Deterministic - Contractual
Failure Mode: CONTRACT_VIOLATION → Error (should not happen)

This layer:
- Validates that all required outputs from L1-L4 are present
- Structures the output into a standardized format
- Ensures output schema compliance

Output: Structured OutputPayload ready for downstream layers
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..config import LayerStatus
from .layer1_input_contract import LayerResult


@dataclass
class RecordPayload:
    """Structured output for a single record."""
    record_id: Any
    
    # Quality Scores
    dqs_base: float  # From Layer 4.2
    semantic_score: float  # From Layer 4.3
    anomaly_score: float  # From Layer 4.4
    
    # Flags
    is_valid: bool  # From Layer 4.1
    is_anomaly: bool  # From Layer 4.4
    priority: str  # From Layer 4.5
    
    # Issues
    structural_issues: List[str] = field(default_factory=list)
    semantic_violations: List[str] = field(default_factory=list)
    anomaly_flags: List[str] = field(default_factory=list)
    
    # Summary
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    layer_results: Dict[str, str] = field(default_factory=dict)


@dataclass
class BatchPayload:
    """Structured output for a batch of records."""
    batch_id: str
    timestamp: str
    total_records: int
    valid_records: int
    rejected_records: int
    flagged_records: int
    
    # Aggregate scores
    mean_dqs: float
    min_dqs: float
    max_dqs: float
    
    # Priority counts
    priority_counts: Dict[str, int] = field(default_factory=dict)
    
    # Individual records
    records: List[RecordPayload] = field(default_factory=list)


class OutputContractLayer:
    """
    Layer 5: Output Contract
    
    Validates and structures output from all previous layers.
    Creates a standardized payload for downstream processing.
    """
    
    LAYER_ID = 5
    LAYER_NAME = "output_contract"
    
    # Required layer outputs
    REQUIRED_LAYERS = [4.1, 4.2, 4.3, 4.4, 4.5]
    
    def __init__(self):
        self.batch_payload: Optional[BatchPayload] = None
        self.record_payloads: List[RecordPayload] = []
    
    def validate_and_structure(
        self,
        layer_results: Dict[float, LayerResult],
        dataframe: pd.DataFrame,
        features_df: pd.DataFrame,
        dqs_df: pd.DataFrame,
        anomaly_df: pd.DataFrame,
        summaries: List[Any],
        structural_results: List[Any] = None,
        semantic_results: List[Any] = None,
    ) -> LayerResult:
        """
        Validate outputs from all layers and create structured payload.
        
        Args:
            layer_results: Dict of layer_id -> LayerResult
            dataframe: Original validated DataFrame
            features_df: Features DataFrame
            dqs_df: DQS scores DataFrame
            anomaly_df: Anomaly detection DataFrame
            summaries: Quality summaries from Layer 4.5
            structural_results: Results from Layer 4.1
            semantic_results: Results from Layer 4.3
            
        Returns:
            LayerResult with structured payload
        """
        import time
        import uuid
        start_time = time.time()
        
        issues = []
        warnings = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            # ================================================================
            # CHECK 1: Verify required layer results
            # ================================================================
            checks_performed += 1
            missing_layers = []
            for layer_id in self.REQUIRED_LAYERS:
                if layer_id not in layer_results:
                    missing_layers.append(layer_id)
            
            if missing_layers:
                warnings.append(f"Missing layer results: {missing_layers}")
            checks_passed += 1
            
            # ================================================================
            # CHECK 2: Verify data consistency
            # ================================================================
            checks_performed += 1
            n_records = len(dataframe)
            
            if len(features_df) != n_records:
                issues.append({
                    "type": "CONTRACT_VIOLATION",
                    "message": f"Features count mismatch: {len(features_df)} vs {n_records}",
                })
            
            if len(dqs_df) > 0 and len(dqs_df) != n_records:
                warnings.append(f"DQS count mismatch: {len(dqs_df)} vs {n_records}")
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 3: Build record payloads
            # ================================================================
            checks_performed += 1
            
            self.record_payloads = []
            
            # Get primary key column
            df = dataframe.copy()
            df.columns = [col.lower().strip() for col in df.columns]
            pk_col = self._get_column(df, ["txn_transaction_id", "transaction_id", "id"])
            
            for idx in range(n_records):
                record_id = df.iloc[idx].get(pk_col, idx) if pk_col else idx
                
                # Get DQS score
                dqs_base = 100.0
                if len(dqs_df) > idx:
                    dqs_base = dqs_df.iloc[idx].get("dqs_base", 100.0)
                
                # Get anomaly data
                anomaly_score = 0.0
                is_anomaly = False
                anomaly_flags = []
                if len(anomaly_df) > idx:
                    anom_row = anomaly_df.iloc[idx]
                    anomaly_score = anom_row.get("anomaly_score", 0.0)
                    is_anomaly = anom_row.get("is_anomaly", False)
                    flags_str = anom_row.get("flags", "")
                    if flags_str:
                        anomaly_flags = flags_str.split(",")
                
                # Get structural data
                is_valid = True
                structural_issues = []
                if structural_results and idx < len(structural_results):
                    sr = structural_results[idx]
                    if hasattr(sr, "is_valid"):
                        is_valid = sr.is_valid
                    if hasattr(sr, "issues"):
                        structural_issues = [i.get("message", str(i)) for i in sr.issues]
                
                # Get semantic data
                semantic_score = 100.0
                semantic_violations = []
                if semantic_results and idx < len(semantic_results):
                    sem = semantic_results[idx]
                    if hasattr(sem, "semantic_score"):
                        semantic_score = sem.semantic_score
                    if hasattr(sem, "critical_violations"):
                        semantic_violations = [v.message for v in sem.critical_violations]
                
                # Get summary data
                priority = "none"
                summary = ""
                recommendations = []
                if summaries and idx < len(summaries):
                    s = summaries[idx]
                    if hasattr(s, "priority"):
                        priority = s.priority
                    if hasattr(s, "summary"):
                        summary = s.summary
                    if hasattr(s, "recommendations"):
                        recommendations = s.recommendations
                
                # Build layer results summary
                layer_status = {}
                for lid, result in layer_results.items():
                    layer_status[f"L{lid}"] = result.status.value
                
                payload = RecordPayload(
                    record_id=record_id,
                    dqs_base=dqs_base,
                    semantic_score=semantic_score,
                    anomaly_score=anomaly_score,
                    is_valid=is_valid,
                    is_anomaly=is_anomaly,
                    priority=priority,
                    structural_issues=structural_issues,
                    semantic_violations=semantic_violations,
                    anomaly_flags=anomaly_flags,
                    summary=summary,
                    recommendations=recommendations,
                    timestamp=datetime.now().isoformat(),
                    layer_results=layer_status,
                )
                self.record_payloads.append(payload)
            
            checks_passed += 1
            
            # ================================================================
            # CHECK 4: Build batch payload
            # ================================================================
            checks_performed += 1
            
            dqs_scores = [r.dqs_base for r in self.record_payloads]
            valid_count = sum(1 for r in self.record_payloads if r.is_valid)
            rejected_count = n_records - valid_count
            flagged_count = sum(1 for r in self.record_payloads if r.is_anomaly)
            
            priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "none": 0}
            for r in self.record_payloads:
                if r.priority in priority_counts:
                    priority_counts[r.priority] += 1
            
            self.batch_payload = BatchPayload(
                batch_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
                total_records=n_records,
                valid_records=valid_count,
                rejected_records=rejected_count,
                flagged_records=flagged_count,
                mean_dqs=np.mean(dqs_scores) if dqs_scores else 0,
                min_dqs=np.min(dqs_scores) if dqs_scores else 0,
                max_dqs=np.max(dqs_scores) if dqs_scores else 0,
                priority_counts=priority_counts,
                records=self.record_payloads,
            )
            
            checks_passed += 1
            
            # ================================================================
            # RESULT
            # ================================================================
            status = LayerStatus.PASSED if not issues else LayerStatus.DEGRADED
            
            return self._create_result(
                status=status,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=True,
                details={
                    "batch_id": self.batch_payload.batch_id,
                    "total_records": n_records,
                    "valid_records": valid_count,
                    "rejected_records": rejected_count,
                    "flagged_records": flagged_count,
                    "mean_dqs": round(self.batch_payload.mean_dqs, 2),
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "CONTRACT_VIOLATION",
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
    
    def _get_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column."""
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
    
    def get_batch_payload(self) -> Optional[BatchPayload]:
        """Get the batch payload."""
        return self.batch_payload
    
    def get_record_payloads(self) -> List[RecordPayload]:
        """Get all record payloads."""
        return self.record_payloads
    
    def get_payload_dataframe(self) -> pd.DataFrame:
        """Get payloads as DataFrame."""
        if not self.record_payloads:
            return pd.DataFrame()
        
        data = []
        for r in self.record_payloads:
            data.append({
                "record_id": r.record_id,
                "dqs_base": r.dqs_base,
                "semantic_score": r.semantic_score,
                "anomaly_score": r.anomaly_score,
                "is_valid": r.is_valid,
                "is_anomaly": r.is_anomaly,
                "priority": r.priority,
                "issues_count": len(r.structural_issues) + len(r.semantic_violations),
                "flags_count": len(r.anomaly_flags),
            })
        
        return pd.DataFrame(data)
