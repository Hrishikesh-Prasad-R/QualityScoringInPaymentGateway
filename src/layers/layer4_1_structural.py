"""
Layer 4.1: Structural Integrity Validation

Purpose: Verify that each record has valid structure
Type: 100% Deterministic - CAN REJECT
Failure Mode: STRUCTURAL_FAILURE → REJECT record

This layer checks:
- Primary key presence and uniqueness
- Required fields are present
- Data types are valid
- No corrupted/malformed data

CRITICAL: This is a GATE layer - records that fail are marked as REJECTED.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import re

from ..config import LayerStatus
from .layer1_input_contract import LayerResult


@dataclass
class RecordValidation:
    """Validation result for a single record."""
    record_id: Any
    is_valid: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    

class StructuralIntegrityLayer:
    """
    Layer 4.1: Structural Integrity Validation
    
    Validates that each record has valid structure.
    Records that fail structural checks are marked for REJECTION.
    """
    
    LAYER_ID = 4.1
    LAYER_NAME = "structural_integrity"
    
    # Required fields that MUST be present and non-null
    REQUIRED_FIELDS = [
        "txn_transaction_id",
        "txn_amount",
        "txn_status",
        "card_network",
        "card_pan_token",
        "merchant_merchant_id",
        "customer_customer_id",
    ]
    
    # Fields with specific type requirements
    TYPE_REQUIREMENTS = {
        "txn_amount": "numeric",
        "txn_hour": "numeric",
        "txn_day_of_week": "numeric",
        "fraud_risk_score": "numeric",
        "settlement_gross_amount": "numeric",
        "settlement_net_amount": "numeric",
    }
    
    # Numeric range validations
    RANGE_VALIDATIONS = {
        "txn_amount": (0.01, 10000000),  # 1 paisa to 1 crore
        "fraud_risk_score": (0, 100),
        "txn_hour": (0, 23),
        "txn_day_of_week": (0, 6),
    }
    
    def __init__(self):
        self.validation_results: List[RecordValidation] = []
        self.rejected_indices: List[int] = []
        self.valid_indices: List[int] = []
    
    def validate(
        self,
        dataframe: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
    ) -> LayerResult:
        """
        Validate structural integrity of all records.
        
        Args:
            dataframe: Original validated DataFrame from Layer 2
            features_df: Features DataFrame from Layer 3 (optional)
            
        Returns:
            LayerResult with validation status and rejection list
        """
        import time
        start_time = time.time()
        
        issues = []
        warnings = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            df = dataframe.copy()
            df.columns = [col.lower().strip() for col in df.columns]
            n_records = len(df)
            
            self.validation_results = []
            self.rejected_indices = []
            self.valid_indices = []
            
            # Get primary key column
            pk_col = self._get_column(df, ["txn_transaction_id", "transaction_id", "id"])
            
            # ================================================================
            # CHECK 1: Primary Key Presence
            # ================================================================
            checks_performed += 1
            if pk_col is None:
                issues.append({
                    "type": "STRUCTURAL_FAILURE",
                    "code": "NO_PRIMARY_KEY",
                    "message": "No primary key column found",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=checks_performed,
                    checks_passed=checks_passed,
                    issues=issues,
                    can_continue=False,
                )
            checks_passed += 1
            
            # ================================================================
            # CHECK 2: Primary Key Uniqueness
            # ================================================================
            checks_performed += 1
            duplicate_mask = df.duplicated(subset=[pk_col], keep='first')
            n_duplicates = duplicate_mask.sum()
            if n_duplicates > 0:
                warnings.append(f"{n_duplicates} duplicate primary keys found - marking as rejected")
                duplicate_indices = df[duplicate_mask].index.tolist()
                for idx in duplicate_indices:
                    self._add_rejection(idx, df.loc[idx, pk_col], "DUPLICATE_PRIMARY_KEY", 
                                      "Duplicate primary key")
            checks_passed += 1
            
            # ================================================================
            # CHECK 3: Per-Record Structural Validation
            # ================================================================
            checks_performed += 1
            
            for idx, row in df.iterrows():
                record_id = row.get(pk_col, idx)
                record_issues = []
                
                # Check required fields
                for field in self.REQUIRED_FIELDS:
                    if field in df.columns:
                        value = row.get(field)
                        if pd.isna(value) or value == "" or value is None:
                            record_issues.append({
                                "type": "MISSING_REQUIRED_FIELD",
                                "field": field,
                                "message": f"Required field '{field}' is missing or null",
                            })
                
                # Check type requirements
                for field, expected_type in self.TYPE_REQUIREMENTS.items():
                    if field in df.columns:
                        value = row.get(field)
                        if pd.notna(value):
                            if expected_type == "numeric":
                                try:
                                    float(value)
                                except (ValueError, TypeError):
                                    record_issues.append({
                                        "type": "INVALID_TYPE",
                                        "field": field,
                                        "message": f"Field '{field}' should be numeric, got: {type(value).__name__}",
                                    })
                
                # Check range validations
                for field, (min_val, max_val) in self.RANGE_VALIDATIONS.items():
                    if field in df.columns:
                        value = row.get(field)
                        if pd.notna(value):
                            try:
                                num_value = float(value)
                                if num_value < min_val or num_value > max_val:
                                    record_issues.append({
                                        "type": "OUT_OF_RANGE",
                                        "field": field,
                                        "message": f"Field '{field}' value {num_value} outside range [{min_val}, {max_val}]",
                                    })
                            except (ValueError, TypeError):
                                pass  # Already caught by type check
                
                # Check for corrupted data patterns
                corrupted = self._check_corruption(row)
                if corrupted:
                    record_issues.extend(corrupted)
                
                # Record validation result
                is_valid = len(record_issues) == 0
                self.validation_results.append(RecordValidation(
                    record_id=record_id,
                    is_valid=is_valid,
                    issues=record_issues,
                ))
                
                if is_valid:
                    self.valid_indices.append(idx)
                else:
                    self.rejected_indices.append(idx)
            
            checks_passed += 1
            
            # ================================================================
            # SUMMARY
            # ================================================================
            n_valid = len(self.valid_indices)
            n_rejected = len(self.rejected_indices)
            rejection_rate = (n_rejected / n_records * 100) if n_records > 0 else 0
            
            if rejection_rate > 50:
                warnings.append(f"High rejection rate: {rejection_rate:.1f}%")
            
            # Determine status
            if n_valid == 0:
                status = LayerStatus.FAILED
                can_continue = False
            elif n_rejected > 0:
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
                    "total_records": n_records,
                    "valid_records": n_valid,
                    "rejected_records": n_rejected,
                    "rejection_rate": round(rejection_rate, 2),
                    "rejected_indices": self.rejected_indices[:100],  # First 100 for brevity
                    "issue_summary": self._summarize_issues(),
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "STRUCTURAL_FAILURE",
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error: {str(e)}",
                "severity": "critical",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                can_continue=False,
            )
    
    def _check_corruption(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Check for corrupted data patterns."""
        issues = []
        
        # Check for obviously invalid transaction IDs
        txn_id = row.get("txn_transaction_id", "")
        if pd.notna(txn_id):
            txn_id_str = str(txn_id)
            # Check for control characters or very short IDs
            if len(txn_id_str) < 3:
                issues.append({
                    "type": "CORRUPTED_DATA",
                    "field": "txn_transaction_id",
                    "message": f"Transaction ID too short: '{txn_id_str}'",
                })
            if re.search(r'[\x00-\x1f]', txn_id_str):
                issues.append({
                    "type": "CORRUPTED_DATA",
                    "field": "txn_transaction_id",
                    "message": "Transaction ID contains control characters",
                })
        
        # Check for negative amounts
        amount = row.get("txn_amount", 0)
        if pd.notna(amount):
            try:
                if float(amount) < 0:
                    issues.append({
                        "type": "CORRUPTED_DATA",
                        "field": "txn_amount",
                        "message": f"Negative amount: {amount}",
                    })
            except:
                pass
        
        return issues
    
    def _add_rejection(self, idx: int, record_id: Any, code: str, message: str):
        """Add a rejection to the list."""
        if idx not in self.rejected_indices:
            self.rejected_indices.append(idx)
            self.validation_results.append(RecordValidation(
                record_id=record_id,
                is_valid=False,
                issues=[{"type": code, "message": message}],
            ))
    
    def _summarize_issues(self) -> Dict[str, int]:
        """Summarize issues by type."""
        summary = {}
        for result in self.validation_results:
            if not result.is_valid:
                for issue in result.issues:
                    issue_type = issue.get("type", "UNKNOWN")
                    summary[issue_type] = summary.get(issue_type, 0) + 1
        return summary
    
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
    
    def get_valid_indices(self) -> List[int]:
        """Get indices of valid records."""
        return self.valid_indices
    
    def get_rejected_indices(self) -> List[int]:
        """Get indices of rejected records."""
        return self.rejected_indices
    
    def get_validation_results(self) -> List[RecordValidation]:
        """Get all validation results."""
        return self.validation_results
