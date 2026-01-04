"""
Layer 4.3: Semantic Validation

Purpose: Validate business rules and domain-specific logic
Type: 100% Deterministic - CAN REJECT
Failure Mode: SEMANTIC_VIOLATION → REJECT or FLAG

This layer checks:
- Business rules (from schema definition)
- Domain rationality (amount vs category, geo-consistency)
- Multi-hop logical chains
- Cross-field semantic relationships

Output: Semantic score + violation catalog per record
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import re

from ..config import LayerStatus
from .layer1_input_contract import LayerResult


@dataclass
class RuleResult:
    """Result of a business rule evaluation."""
    rule_id: str
    rule_name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    message: str = ""


@dataclass 
class SemanticValidation:
    """Semantic validation result for a single record."""
    record_id: Any
    semantic_score: float  # 0-100
    rules_passed: int
    rules_failed: int
    critical_violations: List[RuleResult]
    warnings: List[RuleResult]
    passes_validation: bool


class SemanticValidationLayer:
    """
    Layer 4.3: Semantic Validation
    
    Validates business rules and domain-specific logic.
    Critical violations cause rejection; warnings are flagged.
    """
    
    LAYER_ID = 4.3
    LAYER_NAME = "semantic_validation"
    
    def __init__(self):
        self.validation_results: List[SemanticValidation] = []
        self.rejected_indices: List[int] = []
        self.flagged_indices: List[int] = []
        
        # Define business rules
        self.rules = self._define_rules()
    
    def _define_rules(self) -> List[Dict[str, Any]]:
        """Define all business rules to evaluate."""
        return [
            # === CRITICAL RULES (Violations cause rejection) ===
            {
                "rule_id": "BR001",
                "name": "Amount must be positive",
                "severity": "critical",
                "check": self._check_positive_amount,
            },
            {
                "rule_id": "BR002",
                "name": "Net amount must equal gross minus fees",
                "severity": "critical",
                "check": self._check_settlement_math,
            },
            {
                "rule_id": "BR003",
                "name": "Settlement date must be after clearing date",
                "severity": "critical",
                "check": self._check_settlement_sequence,
            },
            {
                "rule_id": "BR004",
                "name": "Approved transactions must have authorization code",
                "severity": "critical",
                "check": self._check_auth_code_present,
            },
            {
                "rule_id": "BR005",
                "name": "Card expiry must not be in past",
                "severity": "critical",
                "check": self._check_card_not_expired,
            },
            
            # === WARNING RULES (Violations flagged for review) ===
            {
                "rule_id": "BR006",
                "name": "Amount should be rational for category",
                "severity": "warning",
                "check": self._check_amount_category_rationality,
            },
            {
                "rule_id": "BR007",
                "name": "Risk score should match risk level",
                "severity": "warning",
                "check": self._check_risk_consistency,
            },
            {
                "rule_id": "BR008",
                "name": "Domestic transaction should use domestic IP",
                "severity": "warning",
                "check": self._check_geo_consistency,
            },
            {
                "rule_id": "BR009",
                "name": "3DS authentication should be present for high-value",
                "severity": "warning",
                "check": self._check_3ds_for_high_value,
            },
            {
                "rule_id": "BR010",
                "name": "Failed velocity check should increase risk",
                "severity": "warning",
                "check": self._check_velocity_risk_correlation,
            },
            {
                "rule_id": "BR011",
                "name": "Fee ratio should be reasonable (< 5%)",
                "severity": "warning",
                "check": self._check_fee_ratio,
            },
            {
                "rule_id": "BR012",
                "name": "Billing and shipping country should match for domestic",
                "severity": "warning",
                "check": self._check_address_country_match,
            },
        ]
    
    def validate(
        self,
        dataframe: pd.DataFrame,
        features_df: pd.DataFrame,
        valid_indices: List[int] = None,
    ) -> LayerResult:
        """
        Validate all records against business rules.
        
        Args:
            dataframe: Original validated DataFrame
            features_df: Features DataFrame from Layer 3
            valid_indices: Indices of records that passed previous layers
            
        Returns:
            LayerResult with semantic validation results
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
            
            self.validation_results = []
            self.rejected_indices = []
            self.flagged_indices = []
            semantic_scores = []
            
            # Get primary key column
            pk_col = self._get_column(df, ["txn_transaction_id", "transaction_id", "id"])
            
            # ================================================================
            # VALIDATE EACH RECORD
            # ================================================================
            checks_performed += 1
            
            for idx in process_indices:
                if idx >= len(df):
                    continue
                    
                row = df.iloc[idx] if isinstance(idx, int) else df.loc[idx]
                feat_row = feat_df.iloc[idx] if isinstance(idx, int) else feat_df.loc[idx]
                record_id = row.get(pk_col, idx) if pk_col else idx
                
                # Evaluate all rules
                critical_violations = []
                rule_warnings = []
                rules_passed = 0
                rules_failed = 0
                
                for rule in self.rules:
                    try:
                        result = rule["check"](row, feat_row)
                        result.rule_id = rule["rule_id"]
                        result.rule_name = rule["name"]
                        result.severity = rule["severity"]
                        
                        if result.passed:
                            rules_passed += 1
                        else:
                            rules_failed += 1
                            if result.severity == "critical":
                                critical_violations.append(result)
                            else:
                                rule_warnings.append(result)
                    except Exception as e:
                        # Rule evaluation error - treat as warning
                        rule_warnings.append(RuleResult(
                            rule_id=rule["rule_id"],
                            rule_name=rule["name"],
                            passed=False,
                            severity="warning",
                            message=f"Rule evaluation error: {str(e)}",
                        ))
                        rules_failed += 1
                
                # Calculate semantic score
                total_rules = rules_passed + rules_failed
                semantic_score = (rules_passed / total_rules * 100) if total_rules > 0 else 100
                
                # Determine if passes validation (no critical violations)
                passes_validation = len(critical_violations) == 0
                
                self.validation_results.append(SemanticValidation(
                    record_id=record_id,
                    semantic_score=semantic_score,
                    rules_passed=rules_passed,
                    rules_failed=rules_failed,
                    critical_violations=critical_violations,
                    warnings=rule_warnings,
                    passes_validation=passes_validation,
                ))
                
                semantic_scores.append(semantic_score)
                
                if not passes_validation:
                    self.rejected_indices.append(idx)
                elif len(rule_warnings) > 0:
                    self.flagged_indices.append(idx)
            
            checks_passed += 1
            
            # ================================================================
            # SUMMARY
            # ================================================================
            n_validated = len(semantic_scores)
            n_rejected = len(self.rejected_indices)
            n_flagged = len(self.flagged_indices)
            
            mean_score = np.mean(semantic_scores) if semantic_scores else 0
            
            # Rule pass rates
            rule_stats = {}
            for rule in self.rules:
                passed = sum(1 for r in self.validation_results 
                           if not any(v.rule_id == rule["rule_id"] 
                                     for v in r.critical_violations + r.warnings))
                rule_stats[rule["rule_id"]] = {
                    "name": rule["name"],
                    "pass_rate": round(passed / n_validated * 100, 1) if n_validated > 0 else 100,
                }
            
            # Determine status
            if n_rejected == n_validated and n_validated > 0:
                status = LayerStatus.FAILED
                can_continue = False
            elif n_rejected > 0 or n_flagged > 0:
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
                    "records_validated": n_validated,
                    "records_rejected": n_rejected,
                    "records_flagged": n_flagged,
                    "semantic_score_mean": round(mean_score, 2),
                    "rules_evaluated": len(self.rules),
                    "rule_statistics": rule_stats,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "SEMANTIC_FAILURE",
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
    # CRITICAL BUSINESS RULES
    # ========================================================================
    
    def _check_positive_amount(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR001: Amount must be positive."""
        amount = feat_row.get("txn_amount", 0)
        passed = amount > 0
        return RuleResult(
            rule_id="BR001",
            rule_name="Amount must be positive",
            passed=passed,
            severity="critical",
            message="" if passed else f"Non-positive amount: {amount}",
        )
    
    def _check_settlement_math(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR002: Net = Gross - Interchange - Gateway."""
        gross = row.get("settlement_gross_amount")
        int_fee = row.get("settlement_interchange_fee")
        gw_fee = row.get("settlement_gateway_fee")
        net = row.get("settlement_net_amount")
        
        # Skip if data not present
        if any(pd.isna(x) for x in [gross, int_fee, gw_fee, net]):
            return RuleResult(
                rule_id="BR002",
                rule_name="Settlement math",
                passed=True,
                severity="critical",
                message="Settlement data not present",
            )
        
        try:
            expected = float(gross) - float(int_fee) - float(gw_fee)
            actual = float(net)
            passed = abs(actual - expected) < 1  # Allow 1 unit tolerance
            return RuleResult(
                rule_id="BR002",
                rule_name="Settlement math",
                passed=passed,
                severity="critical",
                message="" if passed else f"Net {actual} != Expected {expected}",
            )
        except:
            return RuleResult(
                rule_id="BR002",
                rule_name="Settlement math",
                passed=True,
                severity="critical",
            )
    
    def _check_settlement_sequence(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR003: Settlement date must be after clearing date."""
        clearing = row.get("settlement_clearing_date")
        settlement = row.get("settlement_settlement_date")
        
        if pd.isna(clearing) or pd.isna(settlement):
            return RuleResult(
                rule_id="BR003",
                rule_name="Settlement sequence",
                passed=True,
                severity="critical",
            )
        
        try:
            clear_date = pd.to_datetime(clearing)
            settle_date = pd.to_datetime(settlement)
            passed = settle_date >= clear_date
            return RuleResult(
                rule_id="BR003",
                rule_name="Settlement sequence",
                passed=passed,
                severity="critical",
                message="" if passed else f"Settlement {settlement} before clearing {clearing}",
            )
        except:
            return RuleResult(
                rule_id="BR003",
                rule_name="Settlement sequence",
                passed=True,
                severity="critical",
            )
    
    def _check_auth_code_present(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR004: Approved transactions must have auth code."""
        status = str(row.get("txn_status", "")).lower()
        auth_code = row.get("txn_authorization_code", "")
        
        if status != "approved":
            return RuleResult(
                rule_id="BR004",
                rule_name="Auth code for approved",
                passed=True,
                severity="critical",
            )
        
        passed = pd.notna(auth_code) and str(auth_code).strip() != ""
        return RuleResult(
            rule_id="BR004",
            rule_name="Auth code for approved",
            passed=passed,
            severity="critical",
            message="" if passed else "Approved transaction without auth code",
        )
    
    def _check_card_not_expired(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR005: Card expiry must not be in past."""
        months_remaining = feat_row.get("card_expiry_months_remaining", 12)
        passed = months_remaining >= 0
        return RuleResult(
            rule_id="BR005",
            rule_name="Card not expired",
            passed=passed,
            severity="critical",
            message="" if passed else f"Card expired {-months_remaining} months ago",
        )
    
    # ========================================================================
    # WARNING BUSINESS RULES
    # ========================================================================
    
    def _check_amount_category_rationality(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR006: Amount should be rational for merchant category."""
        amount = feat_row.get("txn_amount", 0)
        mcc_first2 = feat_row.get("merchant_mcc_first2", 0)
        
        # MCC-based expected ranges
        mcc_ranges = {
            58: (100, 10000),    # Restaurant
            54: (50, 15000),     # Grocery
            55: (100, 10000),    # Gas
            53: (200, 50000),    # Department store
            41: (50, 5000),      # Transport
            70: (1000, 100000),  # Hotel
            56: (200, 30000),    # Clothing
        }
        
        expected_range = mcc_ranges.get(int(mcc_first2), (50, 100000))
        passed = expected_range[0] <= amount <= expected_range[1]
        
        return RuleResult(
            rule_id="BR006",
            rule_name="Amount rationality",
            passed=passed,
            severity="warning",
            message="" if passed else f"Amount {amount} unusual for MCC {mcc_first2}",
        )
    
    def _check_risk_consistency(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR007: Risk score should match risk level."""
        risk_score = feat_row.get("fraud_risk_score", 0)
        risk_level = feat_row.get("fraud_risk_level_encoded", 0)
        
        expected = 2 if risk_score > 70 else (1 if risk_score > 40 else 0)
        passed = risk_level == expected
        
        return RuleResult(
            rule_id="BR007",
            rule_name="Risk consistency",
            passed=passed,
            severity="warning",
            message="" if passed else f"Risk level {risk_level} inconsistent with score {risk_score}",
        )
    
    def _check_geo_consistency(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR008: Domestic transaction should use domestic IP."""
        is_domestic = feat_row.get("merchant_is_domestic", 1)
        ip_domestic = feat_row.get("customer_ip_is_domestic", 1)
        
        # For domestic merchants, IP should ideally be domestic
        if is_domestic == 1:
            passed = ip_domestic == 1
        else:
            passed = True  # International transactions can have any IP
        
        return RuleResult(
            rule_id="BR008",
            rule_name="Geo consistency",
            passed=passed,
            severity="warning",
            message="" if passed else "Domestic merchant with foreign IP",
        )
    
    def _check_3ds_for_high_value(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR009: 3DS should be present for high-value transactions."""
        amount = feat_row.get("txn_amount", 0)
        auth_result = feat_row.get("auth_result_encoded", 0)
        
        # High value = above 10000
        if amount > 10000:
            passed = auth_result == 0  # authenticated
        else:
            passed = True
        
        return RuleResult(
            rule_id="BR009",
            rule_name="3DS for high value",
            passed=passed,
            severity="warning",
            message="" if passed else f"High value ₹{amount} without 3DS authentication",
        )
    
    def _check_velocity_risk_correlation(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR010: Failed velocity check should increase risk."""
        velocity_passed = feat_row.get("fraud_velocity_passed", 1)
        risk_score = feat_row.get("fraud_risk_score", 0)
        
        if velocity_passed == 0:
            passed = risk_score >= 40  # Should have elevated risk
        else:
            passed = True
        
        return RuleResult(
            rule_id="BR010",
            rule_name="Velocity-risk correlation",
            passed=passed,
            severity="warning",
            message="" if passed else f"Failed velocity but low risk: {risk_score}",
        )
    
    def _check_fee_ratio(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR011: Fee ratio should be reasonable (< 5%)."""
        fee_ratio = feat_row.get("settlement_fee_ratio", 0.02)
        passed = fee_ratio < 0.05
        
        return RuleResult(
            rule_id="BR011",
            rule_name="Reasonable fee ratio",
            passed=passed,
            severity="warning",
            message="" if passed else f"High fee ratio: {fee_ratio*100:.1f}%",
        )
    
    def _check_address_country_match(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
        """BR012: Billing and shipping country should match for domestic."""
        billing_country = str(row.get("customer_billing_address_country", "")).upper()
        shipping_country = str(row.get("customer_shipping_address_country", "")).upper()
        merchant_country = str(row.get("merchant_country", "")).upper()
        
        # For domestic merchants, addresses should match
        if merchant_country == "IN":
            passed = (billing_country in ["IN", ""] and shipping_country in ["IN", ""])
        else:
            passed = True
        
        return RuleResult(
            rule_id="BR012",
            rule_name="Address country match",
            passed=passed,
            severity="warning",
            message="" if passed else f"Domestic merchant but foreign address",
        )
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
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
    
    def get_validation_results(self) -> List[SemanticValidation]:
        """Get all validation results."""
        return self.validation_results
    
    def get_rejected_indices(self) -> List[int]:
        """Get indices of rejected records."""
        return self.rejected_indices
    
    def get_flagged_indices(self) -> List[int]:
        """Get indices of flagged records."""
        return self.flagged_indices
    
    def get_semantic_scores(self) -> pd.DataFrame:
        """Get semantic scores as DataFrame."""
        if not self.validation_results:
            return pd.DataFrame()
        
        data = []
        for r in self.validation_results:
            data.append({
                "record_id": r.record_id,
                "semantic_score": r.semantic_score,
                "rules_passed": r.rules_passed,
                "rules_failed": r.rules_failed,
                "critical_violations": len(r.critical_violations),
                "warnings": len(r.warnings),
                "passes_validation": r.passes_validation,
            })
        
        return pd.DataFrame(data)
