"""
Layer 4.2: Field-Level Compliance Scoring

Purpose: Score data quality across 7 dimensions
Type: 100% Deterministic - CAN REJECT (if below threshold)
Failure Mode: COMPLIANCE_FAILURE → REJECT or FLAG

The 7 Quality Dimensions:
1. Completeness - % of non-null values
2. Accuracy - % of values matching expected patterns
3. Validity - % of values within valid ranges
4. Uniqueness - % of unique values where expected
5. Consistency - % of consistent cross-field relationships
6. Timeliness - % of timely data (not stale)
7. Integrity - % of referential integrity maintained

Output: DQS_base score (0-100) per record
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import re

from ..config import LayerStatus, DIMENSION_WEIGHTS, DEFAULT_QUALITY_THRESHOLDS
from .layer1_input_contract import LayerResult


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension: str
    score: float  # 0-100
    weight: float
    passed_checks: int
    total_checks: int
    issues: List[str] = field(default_factory=list)


@dataclass
class RecordQualityScore:
    """Quality score for a single record."""
    record_id: Any
    dimension_scores: Dict[str, DimensionScore]
    dqs_base: float  # Weighted composite score 0-100
    passes_threshold: bool
    critical_failures: List[str] = field(default_factory=list)


class FieldComplianceLayer:
    """
    Layer 4.2: Field-Level Compliance Scoring
    
    Scores each record across 7 quality dimensions.
    Records below threshold are flagged for rejection.
    """
    
    LAYER_ID = 4.2
    LAYER_NAME = "field_compliance"
    
    # Threshold for rejection (records below this DQS are rejected)
    REJECTION_THRESHOLD = 40.0
    REVIEW_THRESHOLD = 75.0
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS
        self.weights = DIMENSION_WEIGHTS
        self.record_scores: List[RecordQualityScore] = []
        self.rejected_indices: List[int] = []
        self.review_indices: List[int] = []
    
    def score(
        self,
        dataframe: pd.DataFrame,
        features_df: pd.DataFrame,
        valid_indices: List[int] = None,
    ) -> LayerResult:
        """
        Score all records across 7 quality dimensions.
        
        Args:
            dataframe: Original validated DataFrame
            features_df: Features DataFrame from Layer 3
            valid_indices: Indices of records that passed Layer 4.1 (optional)
            
        Returns:
            LayerResult with quality scores
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
            
            feat_df = features_df.copy()
            feat_df.columns = [col.lower().strip() for col in feat_df.columns]
            
            n_records = len(df)
            
            # Filter to valid indices if provided
            if valid_indices is not None:
                process_indices = valid_indices
            else:
                process_indices = list(df.index)
            
            self.record_scores = []
            self.rejected_indices = []
            self.review_indices = []
            dqs_scores = []
            
            # Get primary key column
            pk_col = self._get_column(df, ["txn_transaction_id", "transaction_id", "id"])
            
            # ================================================================
            # SCORE EACH RECORD
            # ================================================================
            checks_performed += 1
            
            for idx in process_indices:
                if idx >= len(df):
                    continue
                    
                row = df.iloc[idx] if isinstance(idx, int) else df.loc[idx]
                feat_row = feat_df.iloc[idx] if isinstance(idx, int) else feat_df.loc[idx]
                record_id = row.get(pk_col, idx) if pk_col else idx
                
                # Calculate dimension scores
                dimension_scores = {}
                
                # 1. Completeness
                completeness = self._score_completeness(row, feat_row)
                dimension_scores["completeness"] = completeness
                
                # 2. Accuracy
                accuracy = self._score_accuracy(row, feat_row)
                dimension_scores["accuracy"] = accuracy
                
                # 3. Validity
                validity = self._score_validity(row, feat_row)
                dimension_scores["validity"] = validity
                
                # 4. Uniqueness (evaluated at dataset level, use proxy here)
                uniqueness = self._score_uniqueness(row, feat_row, df, idx)
                dimension_scores["uniqueness"] = uniqueness
                
                # 5. Consistency
                consistency = self._score_consistency(row, feat_row)
                dimension_scores["consistency"] = consistency
                
                # 6. Timeliness
                timeliness = self._score_timeliness(row, feat_row)
                dimension_scores["timeliness"] = timeliness
                
                # 7. Integrity
                integrity = self._score_integrity(row, feat_row)
                dimension_scores["integrity"] = integrity
                
                # Calculate weighted DQS
                dqs_base = self._calculate_dqs(dimension_scores)
                
                # Check threshold
                passes_threshold = dqs_base >= self.REJECTION_THRESHOLD
                needs_review = dqs_base < self.REVIEW_THRESHOLD
                
                # Collect critical failures
                critical_failures = []
                for dim, score in dimension_scores.items():
                    threshold = self.thresholds.get(dim, 90)
                    if score.score < threshold * 0.5:  # Less than 50% of threshold
                        critical_failures.append(f"{dim}: {score.score:.1f}")
                
                self.record_scores.append(RecordQualityScore(
                    record_id=record_id,
                    dimension_scores=dimension_scores,
                    dqs_base=dqs_base,
                    passes_threshold=passes_threshold,
                    critical_failures=critical_failures,
                ))
                
                dqs_scores.append(dqs_base)
                
                if not passes_threshold:
                    self.rejected_indices.append(idx)
                elif needs_review:
                    self.review_indices.append(idx)
            
            checks_passed += 1
            
            # ================================================================
            # SUMMARY STATISTICS
            # ================================================================
            n_scored = len(dqs_scores)
            n_rejected = len(self.rejected_indices)
            n_review = len(self.review_indices)
            
            mean_dqs = np.mean(dqs_scores) if dqs_scores else 0
            min_dqs = np.min(dqs_scores) if dqs_scores else 0
            max_dqs = np.max(dqs_scores) if dqs_scores else 0
            
            # Dimension averages
            dim_averages = {}
            for dim in ["completeness", "accuracy", "validity", "uniqueness", 
                       "consistency", "timeliness", "integrity"]:
                scores = [r.dimension_scores[dim].score for r in self.record_scores 
                         if dim in r.dimension_scores]
                dim_averages[dim] = np.mean(scores) if scores else 0
            
            # Determine status
            if n_rejected == n_scored and n_scored > 0:
                status = LayerStatus.FAILED
                can_continue = False
            elif n_rejected > 0 or n_review > 0:
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
                    "records_scored": n_scored,
                    "records_rejected": n_rejected,
                    "records_for_review": n_review,
                    "dqs_mean": round(mean_dqs, 2),
                    "dqs_min": round(min_dqs, 2),
                    "dqs_max": round(max_dqs, 2),
                    "dimension_averages": {k: round(v, 2) for k, v in dim_averages.items()},
                    "rejection_threshold": self.REJECTION_THRESHOLD,
                    "review_threshold": self.REVIEW_THRESHOLD,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "COMPLIANCE_FAILURE",
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
    
    # ========================================================================
    # DIMENSION SCORING METHODS
    # ========================================================================
    
    def _score_completeness(self, row: pd.Series, feat_row: pd.Series) -> DimensionScore:
        """Score completeness - % of non-null values in required fields."""
        required_fields = [
            "txn_transaction_id", "txn_amount", "txn_status", "txn_timestamp",
            "card_network", "card_pan_token",
            "merchant_merchant_id", "merchant_merchant_category_code",
            "customer_customer_id",
        ]
        
        present = 0
        total = 0
        issues = []
        
        for field in required_fields:
            if field in row.index:
                total += 1
                value = row.get(field)
                if pd.notna(value) and value != "" and value is not None:
                    present += 1
                else:
                    issues.append(f"Missing: {field}")
        
        score = (present / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="completeness",
            score=score,
            weight=self.weights.get("completeness", 0.2),
            passed_checks=present,
            total_checks=total,
            issues=issues,
        )
    
    def _score_accuracy(self, row: pd.Series, feat_row: pd.Series) -> DimensionScore:
        """Score accuracy - % of values matching expected patterns."""
        checks = []
        issues = []
        
        # Transaction ID format
        txn_id = str(row.get("txn_transaction_id", ""))
        if re.match(r'^[a-zA-Z0-9_-]+$', txn_id) and len(txn_id) >= 5:
            checks.append(True)
        else:
            checks.append(False)
            issues.append("Invalid transaction ID format")
        
        # Card BIN format (6 digits)
        bin_val = str(row.get("card_bin", ""))
        if re.match(r'^\d{6}$', bin_val):
            checks.append(True)
        else:
            checks.append(False)
            issues.append("Invalid BIN format")
        
        # Email format (if present)
        email = row.get("customer_email", "")
        if pd.isna(email) or email == "":
            checks.append(True)  # Optional field
        elif re.match(r'^[^@]+@[^@]+\.[^@]+$', str(email)):
            checks.append(True)
        else:
            checks.append(False)
            issues.append("Invalid email format")
        
        # Country code format (2 letters)
        country = str(row.get("merchant_country", ""))
        if re.match(r'^[A-Z]{2}$', country.upper()):
            checks.append(True)
        else:
            checks.append(False)
            issues.append("Invalid country code format")
        
        # MCC format (4 digits)
        mcc = str(row.get("merchant_merchant_category_code", ""))
        if re.match(r'^\d{4}$', mcc):
            checks.append(True)
        else:
            checks.append(False)
            issues.append("Invalid MCC format")
        
        passed = sum(checks)
        total = len(checks)
        score = (passed / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="accuracy",
            score=score,
            weight=self.weights.get("accuracy", 0.2),
            passed_checks=passed,
            total_checks=total,
            issues=issues,
        )
    
    def _score_validity(self, row: pd.Series, feat_row: pd.Series) -> DimensionScore:
        """Score validity - % of values within valid ranges/enums."""
        checks = []
        issues = []
        
        # Amount validity
        amount = feat_row.get("txn_amount", 0)
        if 0.01 <= amount <= 10000000:
            checks.append(True)
        else:
            checks.append(False)
            issues.append(f"Amount out of range: {amount}")
        
        # Status validity
        status = str(row.get("txn_status", "")).lower()
        if status in ["approved", "declined", "pending", "failed"]:
            checks.append(True)
        else:
            checks.append(False)
            issues.append(f"Invalid status: {status}")
        
        # Card network validity
        network = str(row.get("card_network", "")).lower()
        if network in ["visa", "mastercard", "rupay", "amex", "diners"]:
            checks.append(True)
        else:
            checks.append(False)
            issues.append(f"Invalid card network: {network}")
        
        # Risk score validity
        risk_score = feat_row.get("fraud_risk_score", 0)
        if 0 <= risk_score <= 100:
            checks.append(True)
        else:
            checks.append(False)
            issues.append(f"Invalid risk score: {risk_score}")
        
        # Timestamp validity
        ts = row.get("txn_timestamp", "")
        if pd.notna(ts):
            try:
                parsed = pd.to_datetime(ts)
                # Should be within reasonable range (not in future, not too old)
                now = pd.Timestamp.now()
                if parsed <= now and parsed >= now - pd.Timedelta(days=365):
                    checks.append(True)
                else:
                    checks.append(False)
                    issues.append("Timestamp outside valid range")
            except:
                checks.append(False)
                issues.append("Invalid timestamp format")
        else:
            checks.append(True)  # Will be caught by completeness
        
        passed = sum(checks)
        total = len(checks)
        score = (passed / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="validity",
            score=score,
            weight=self.weights.get("validity", 0.15),
            passed_checks=passed,
            total_checks=total,
            issues=issues,
        )
    
    def _score_uniqueness(self, row: pd.Series, feat_row: pd.Series, 
                         df: pd.DataFrame, idx: int) -> DimensionScore:
        """Score uniqueness - check unique fields are actually unique."""
        checks = []
        issues = []
        
        # Transaction ID should be unique
        txn_id = row.get("txn_transaction_id", "")
        if pd.notna(txn_id):
            # Check if unique in dataset
            duplicates = (df["txn_transaction_id"] == txn_id).sum()
            if duplicates <= 1:
                checks.append(True)
            else:
                checks.append(False)
                issues.append(f"Duplicate transaction ID: {txn_id}")
        else:
            checks.append(True)  # Caught by completeness
        
        passed = sum(checks)
        total = len(checks)
        score = (passed / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="uniqueness",
            score=score,
            weight=self.weights.get("uniqueness", 0.1),
            passed_checks=passed,
            total_checks=total,
            issues=issues,
        )
    
    def _score_consistency(self, row: pd.Series, feat_row: pd.Series) -> DimensionScore:
        """Score consistency - cross-field logical consistency."""
        checks = []
        issues = []
        
        # Settlement amounts should be consistent
        gross = row.get("settlement_gross_amount")
        int_fee = row.get("settlement_interchange_fee")
        gw_fee = row.get("settlement_gateway_fee")
        net = row.get("settlement_net_amount")
        
        if all(pd.notna(x) for x in [gross, int_fee, gw_fee, net]):
            try:
                expected_net = float(gross) - float(int_fee) - float(gw_fee)
                if abs(float(net) - expected_net) < 1:  # Allow 1 unit tolerance
                    checks.append(True)
                else:
                    checks.append(False)
                    issues.append("Net amount inconsistent with gross - fees")
            except:
                checks.append(True)
        else:
            checks.append(True)  # Missing data handled elsewhere
        
        # Address match flag should match actual addresses
        billing_city = str(row.get("customer_billing_address_city", "")).lower()
        shipping_city = str(row.get("customer_shipping_address_city", "")).lower()
        address_match = feat_row.get("customer_address_match", 1)
        
        if billing_city and shipping_city:
            actual_match = 1 if billing_city == shipping_city else 0
            if address_match == actual_match:
                checks.append(True)
            else:
                checks.append(False)
                issues.append("Address match flag inconsistent")
        else:
            checks.append(True)
        
        # Risk level should match risk score
        risk_score = feat_row.get("fraud_risk_score", 0)
        risk_level = feat_row.get("fraud_risk_level_encoded", 0)
        
        expected_level = 2 if risk_score > 70 else (1 if risk_score > 40 else 0)
        if risk_level == expected_level:
            checks.append(True)
        else:
            checks.append(False)
            issues.append(f"Risk level {risk_level} inconsistent with score {risk_score}")
        
        passed = sum(checks)
        total = len(checks)
        score = (passed / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="consistency",
            score=score,
            weight=self.weights.get("consistency", 0.15),
            passed_checks=passed,
            total_checks=total,
            issues=issues,
        )
    
    def _score_timeliness(self, row: pd.Series, feat_row: pd.Series) -> DimensionScore:
        """Score timeliness - data freshness and processing times."""
        checks = []
        issues = []
        
        # Transaction should not be stale (within 90 days)
        ts = row.get("txn_timestamp", "")
        if pd.notna(ts):
            try:
                parsed = pd.to_datetime(ts)
                age_days = (pd.Timestamp.now() - parsed).days
                if age_days <= 90:
                    checks.append(True)
                else:
                    checks.append(False)
                    issues.append(f"Stale transaction: {age_days} days old")
            except:
                checks.append(True)
        else:
            checks.append(True)
        
        # Settlement should happen within reasonable time
        clearing_days = feat_row.get("settlement_days_to_clear", 1)
        if clearing_days <= 7:
            checks.append(True)
        else:
            checks.append(False)
            issues.append(f"Slow clearing: {clearing_days} days")
        
        # Card should not be expired
        months_remaining = feat_row.get("card_expiry_months_remaining", 12)
        if months_remaining >= 0:
            checks.append(True)
        else:
            checks.append(False)
            issues.append("Expired card")
        
        passed = sum(checks)
        total = len(checks)
        score = (passed / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="timeliness",
            score=score,
            weight=self.weights.get("timeliness", 0.1),
            passed_checks=passed,
            total_checks=total,
            issues=issues,
        )
    
    def _score_integrity(self, row: pd.Series, feat_row: pd.Series) -> DimensionScore:
        """Score integrity - referential and logical integrity."""
        checks = []
        issues = []
        
        # Customer ID should be present if customer details exist
        customer_id = row.get("customer_customer_id", "")
        customer_email = row.get("customer_email", "")
        
        if pd.notna(customer_email) and customer_email != "":
            if pd.notna(customer_id) and customer_id != "":
                checks.append(True)
            else:
                checks.append(False)
                issues.append("Customer email without customer ID")
        else:
            checks.append(True)
        
        # Merchant ID should be present if merchant details exist
        merchant_id = row.get("merchant_merchant_id", "")
        merchant_name = row.get("merchant_merchant_name", "")
        
        if pd.notna(merchant_name) and merchant_name != "":
            if pd.notna(merchant_id) and merchant_id != "":
                checks.append(True)
            else:
                checks.append(False)
                issues.append("Merchant name without merchant ID")
        else:
            checks.append(True)
        
        # Network transaction ID should exist for approved transactions
        status = str(row.get("txn_status", "")).lower()
        network_txn_id = row.get("network_network_transaction_id", "")
        
        if status == "approved":
            if pd.notna(network_txn_id) and network_txn_id != "":
                checks.append(True)
            else:
                checks.append(False)
                issues.append("Approved transaction without network ID")
        else:
            checks.append(True)
        
        passed = sum(checks)
        total = len(checks)
        score = (passed / total * 100) if total > 0 else 100
        
        return DimensionScore(
            dimension="integrity",
            score=score,
            weight=self.weights.get("integrity", 0.1),
            passed_checks=passed,
            total_checks=total,
            issues=issues,
        )
    
    def _calculate_dqs(self, dimension_scores: Dict[str, DimensionScore]) -> float:
        """Calculate weighted DQS from dimension scores."""
        total_weight = 0
        weighted_sum = 0
        
        for dim, score in dimension_scores.items():
            weighted_sum += score.score * score.weight
            total_weight += score.weight
        
        return (weighted_sum / total_weight) if total_weight > 0 else 0
    
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
    
    def get_record_scores(self) -> List[RecordQualityScore]:
        """Get all record quality scores."""
        return self.record_scores
    
    def get_dqs_dataframe(self) -> pd.DataFrame:
        """Get DQS scores as a DataFrame."""
        if not self.record_scores:
            return pd.DataFrame()
        
        data = []
        for r in self.record_scores:
            row = {
                "record_id": r.record_id,
                "dqs_base": r.dqs_base,
                "passes_threshold": r.passes_threshold,
            }
            for dim, score in r.dimension_scores.items():
                row[f"dim_{dim}"] = score.score
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_rejected_indices(self) -> List[int]:
        """Get indices of rejected records."""
        return self.rejected_indices
    
    def get_review_indices(self) -> List[int]:
        """Get indices of records needing review."""
        return self.review_indices
