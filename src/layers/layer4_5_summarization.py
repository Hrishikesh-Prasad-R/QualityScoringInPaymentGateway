"""
Layer 4.5: GenAI Summarization

Purpose: Generate human-readable explanations for quality issues
Type: AI-ASSISTED - CAN ONLY INFORM (not reject or flag)
Failure Mode: SUMMARY_FAILURE → Use template fallback

CRITICAL PRINCIPLE: "ML informs, Rules enforce, Humans decide"
- This layer provides explanatory context only
- It CANNOT make any decisions
- Uses Gemini API with deterministic template fallback

Output: Human-readable summary of quality issues per record
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os

from ..config import LayerStatus
from .layer1_input_contract import LayerResult

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class QualitySummary:
    """Quality summary for a single record."""
    record_id: Any
    summary: str
    priority: str  # "critical", "high", "medium", "low", "none"
    key_issues: List[str]
    recommendations: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    ai_enhanced: bool = False  # True if Gemini was used


class GenAISummarizationLayer:
    """
    Layer 4.5: GenAI Summarization
    
    Generates human-readable explanations for quality issues.
    Uses Gemini API when available, falls back to templates.
    
    IMPORTANT: This layer is INFORMATIONAL only - no decisions.
    """
    
    LAYER_ID = 4.5
    LAYER_NAME = "genai_summarization"
    
    # Priority thresholds
    DQS_CRITICAL = 40
    DQS_HIGH = 60
    DQS_MEDIUM = 75
    
    # Gemini configuration
    GEMINI_MODEL = "gemini-2.5-flash"
    
    def __init__(self, api_key: str = None, use_ai: bool = False):
        """
        Initialize the summarization layer.
        
        Args:
            api_key: Gemini API key (optional, can also use GEMINI_API_KEY env var)
            use_ai: Whether to use Gemini for enhanced summaries
        """
        self.summaries: List[QualitySummary] = []
        self.use_ai = use_ai and GEMINI_AVAILABLE
        self.gemini_model = None
        self.ai_calls_made = 0
        self.ai_calls_failed = 0
        
        # Configure Gemini if available and requested
        if self.use_ai:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                try:
                    genai.configure(api_key=key)
                    self.gemini_model = genai.GenerativeModel(self.GEMINI_MODEL)
                except Exception as e:
                    self.use_ai = False
    
    def summarize(
        self,
        dataframe: pd.DataFrame,
        features_df: pd.DataFrame,
        dqs_scores: pd.DataFrame = None,
        anomaly_results: pd.DataFrame = None,
        semantic_results: List[Any] = None,
    ) -> LayerResult:
        """
        Generate quality summaries for all records.
        
        Args:
            dataframe: Original validated DataFrame
            features_df: Features DataFrame from Layer 3
            dqs_scores: DQS scores from Layer 4.2
            anomaly_results: Anomaly results from Layer 4.4
            semantic_results: Semantic validation results from Layer 4.3
            
        Returns:
            LayerResult with summarization results
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
            self.summaries = []
            
            # Get primary key column
            pk_col = self._get_column(df, ["txn_transaction_id", "transaction_id", "id"])
            
            # ================================================================
            # GENERATE SUMMARIES FOR EACH RECORD
            # ================================================================
            checks_performed += 1
            
            for idx in range(n_records):
                row = df.iloc[idx]
                feat_row = feat_df.iloc[idx]
                record_id = row.get(pk_col, idx) if pk_col else idx
                
                # Gather quality data
                quality_data = self._gather_quality_data(
                    idx, row, feat_row, dqs_scores, anomaly_results, semantic_results
                )
                
                # Generate summary
                summary = self._generate_summary(record_id, quality_data)
                self.summaries.append(summary)
            
            checks_passed += 1
            
            # ================================================================
            # AGGREGATE STATISTICS
            # ================================================================
            priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "none": 0}
            for s in self.summaries:
                priority_counts[s.priority] += 1
            
            return self._create_result(
                status=LayerStatus.PASSED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=True,
                details={
                    "records_summarized": n_records,
                    "priority_counts": priority_counts,
                    "ai_enabled": self.use_ai,
                    "ai_calls_made": self.ai_calls_made,
                    "ai_calls_failed": self.ai_calls_failed,
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "SUMMARY_FAILURE",
                "code": "UNEXPECTED_ERROR",
                "message": f"Summarization error: {str(e)}",
                "severity": "warning",
            })
            return self._create_result(
                status=LayerStatus.DEGRADED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                can_continue=True,
            )
    
    def _gather_quality_data(
        self,
        idx: int,
        row: pd.Series,
        feat_row: pd.Series,
        dqs_scores: pd.DataFrame,
        anomaly_results: pd.DataFrame,
        semantic_results: List[Any],
    ) -> Dict[str, Any]:
        """Gather all quality data for a record."""
        data = {
            "dqs_base": 100.0,
            "dimensions": {},
            "anomaly_score": 0.0,
            "anomaly_flags": [],
            "semantic_violations": [],
            "semantic_warnings": [],
        }
        
        # DQS data
        if dqs_scores is not None and len(dqs_scores) > idx:
            dqs_row = dqs_scores.iloc[idx]
            data["dqs_base"] = dqs_row.get("dqs_base", 100.0)
            
            # Get dimension scores
            for col in dqs_row.index:
                if col.startswith("dim_"):
                    dim_name = col.replace("dim_", "")
                    data["dimensions"][dim_name] = dqs_row[col]
        
        # Anomaly data
        if anomaly_results is not None and len(anomaly_results) > idx:
            anom_row = anomaly_results.iloc[idx]
            data["anomaly_score"] = anom_row.get("anomaly_score", 0.0)
            flags_str = anom_row.get("flags", "")
            if flags_str:
                data["anomaly_flags"] = flags_str.split(",")
        
        # Semantic data
        if semantic_results and idx < len(semantic_results):
            sem_result = semantic_results[idx]
            if hasattr(sem_result, "critical_violations"):
                data["semantic_violations"] = [
                    v.message for v in sem_result.critical_violations
                ]
            if hasattr(sem_result, "warnings"):
                data["semantic_warnings"] = [
                    w.message for w in sem_result.warnings
                ]
        
        # Feature data
        data["amount"] = feat_row.get("txn_amount", 0)
        data["risk_score"] = feat_row.get("fraud_risk_score", 0)
        data["is_domestic"] = feat_row.get("merchant_is_domestic", 1)
        data["velocity_passed"] = feat_row.get("fraud_velocity_passed", 1)
        
        return data
    
    def _generate_summary(
        self,
        record_id: Any,
        quality_data: Dict[str, Any],
    ) -> QualitySummary:
        """Generate human-readable summary for a record."""
        dqs = quality_data.get("dqs_base", 100)
        anomaly_score = quality_data.get("anomaly_score", 0)
        amount = quality_data.get("amount", 0)
        risk_score = quality_data.get("risk_score", 0)
        
        # Determine priority
        if dqs < self.DQS_CRITICAL or len(quality_data.get("semantic_violations", [])) > 0:
            priority = "critical"
        elif dqs < self.DQS_HIGH or anomaly_score > 0.75:
            priority = "high"
        elif dqs < self.DQS_MEDIUM or anomaly_score > 0.5:
            priority = "medium"
        elif anomaly_score > 0.25:
            priority = "low"
        else:
            priority = "none"
        
        # Collect key issues
        key_issues = []
        recommendations = []
        
        # DQS issues
        if dqs < self.DQS_CRITICAL:
            key_issues.append(f"Critical quality score ({dqs:.1f}/100)")
            recommendations.append("Manual review required before processing")
        elif dqs < self.DQS_HIGH:
            key_issues.append(f"Low quality score ({dqs:.1f}/100)")
            recommendations.append("Consider additional validation")
        
        # Dimension-specific issues
        dimensions = quality_data.get("dimensions", {})
        for dim, score in dimensions.items():
            if score < 70:
                key_issues.append(f"{dim.capitalize()} issue ({score:.0f}%)")
        
        # Anomaly issues
        flags = quality_data.get("anomaly_flags", [])
        for flag in flags[:3]:  # Top 3 flags
            key_issues.append(f"Anomaly: {flag.replace('_', ' ').title()}")
        
        if anomaly_score > 0.75:
            recommendations.append("High anomaly score - investigate transaction pattern")
        
        # Semantic violations
        for violation in quality_data.get("semantic_violations", [])[:2]:
            key_issues.append(f"Rule violation: {violation}")
            recommendations.append("Business rule violation requires review")
        
        # Risk-based issues
        if risk_score > 70:
            key_issues.append(f"High fraud risk score ({risk_score})")
            recommendations.append("Apply enhanced due diligence")
        
        if quality_data.get("velocity_passed", 1) == 0:
            key_issues.append("Failed velocity check")
            recommendations.append("Review transaction frequency patterns")
        
        # Generate summary text (with optional AI enhancement)
        ai_enhanced = False
        if self.use_ai and self.gemini_model and priority in ["critical", "high"]:
            # Use AI for high-priority records
            ai_summary = self._generate_ai_summary(
                record_id, priority, dqs, amount, risk_score, 
                key_issues, recommendations, quality_data
            )
            if ai_summary:
                summary_text = ai_summary
                ai_enhanced = True
            else:
                summary_text = self._generate_summary_text(
                    record_id, priority, dqs, amount, key_issues
                )
        else:
            summary_text = self._generate_summary_text(
                record_id, priority, dqs, amount, key_issues
            )
        
        return QualitySummary(
            record_id=record_id,
            summary=summary_text,
            priority=priority,
            key_issues=key_issues[:5],  # Top 5 issues
            recommendations=list(set(recommendations))[:3],  # Top 3 unique recommendations
            context={
                "dqs_base": dqs,
                "anomaly_score": anomaly_score,
                "amount": amount,
                "risk_score": risk_score,
            },
            ai_enhanced=ai_enhanced,
        )
    
    def _generate_ai_summary(
        self,
        record_id: Any,
        priority: str,
        dqs: float,
        amount: float,
        risk_score: float,
        key_issues: List[str],
        recommendations: List[str],
        quality_data: Dict[str, Any],
    ) -> Optional[str]:
        """Generate AI-enhanced summary using Gemini."""
        if not self.gemini_model:
            return None
        
        try:
            self.ai_calls_made += 1
            
            # Build prompt
            prompt = f"""You are a data quality analyst. Generate a concise, professional summary for this transaction quality assessment.

Transaction ID: {record_id}
Amount: Rs {amount:,.0f}
Priority: {priority.upper()}
Quality Score: {dqs:.1f}/100
Risk Score: {risk_score}

Key Issues Found:
{chr(10).join(f'- {issue}' for issue in key_issues[:5])}

Anomaly Score: {quality_data.get('anomaly_score', 0):.2f}
Anomaly Flags: {', '.join(quality_data.get('anomaly_flags', [])) or 'None'}

Generate a 2-3 sentence summary that:
1. States the priority level and main concern
2. Highlights the most critical issue
3. Suggests immediate action

Be concise and professional. Use plain text only (no markdown, no emojis)."""

            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            
            return None
            
        except Exception as e:
            self.ai_calls_failed += 1
            return None
    
    def _generate_summary_text(
        self,
        record_id: Any,
        priority: str,
        dqs: float,
        amount: float,
        key_issues: List[str],
    ) -> str:
        """Generate human-readable summary text."""
        if priority == "none":
            return f"Transaction {record_id}: Quality PASSED (DQS: {dqs:.1f}/100). No issues detected."
        
        if priority == "critical":
            prefix = f"[CRITICAL] Transaction {record_id}"
        elif priority == "high":
            prefix = f"[HIGH] Transaction {record_id}"
        elif priority == "medium":
            prefix = f"[MEDIUM] Transaction {record_id}"
        else:
            prefix = f"[LOW] Transaction {record_id}"
        
        issue_text = "; ".join(key_issues[:3]) if key_issues else "Minor quality concerns"
        
        return f"{prefix} (₹{amount:,.0f}, DQS: {dqs:.1f}/100). Issues: {issue_text}"
    
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
    
    def get_summaries(self) -> List[QualitySummary]:
        """Get all quality summaries."""
        return self.summaries
    
    def get_summaries_dataframe(self) -> pd.DataFrame:
        """Get summaries as DataFrame."""
        if not self.summaries:
            return pd.DataFrame()
        
        data = []
        for s in self.summaries:
            data.append({
                "record_id": s.record_id,
                "priority": s.priority,
                "summary": s.summary,
                "issues_count": len(s.key_issues),
                "key_issue": s.key_issues[0] if s.key_issues else "",
                "recommendation": s.recommendations[0] if s.recommendations else "",
            })
        
        return pd.DataFrame(data)
    
    def get_critical_summaries(self) -> List[QualitySummary]:
        """Get summaries with critical priority."""
        return [s for s in self.summaries if s.priority == "critical"]
    
    def get_high_priority_summaries(self) -> List[QualitySummary]:
        """Get summaries with high or critical priority."""
        return [s for s in self.summaries if s.priority in ["critical", "high"]]
    
    def generate_batch_report(self) -> str:
        """Generate a batch summary report."""
        if not self.summaries:
            return "No records summarized."
        
        critical = len([s for s in self.summaries if s.priority == "critical"])
        high = len([s for s in self.summaries if s.priority == "high"])
        medium = len([s for s in self.summaries if s.priority == "medium"])
        low = len([s for s in self.summaries if s.priority == "low"])
        clean = len([s for s in self.summaries if s.priority == "none"])
        
        report = f"""
============================================================
       DATA QUALITY BATCH SUMMARY REPORT
============================================================

Total Records Analyzed: {len(self.summaries)}

Priority Breakdown:
  [!!] Critical:  {critical:5d}  {'#' * min(critical, 20)}
  [!!] High:      {high:5d}  {'#' * min(high, 20)}
  [??] Medium:    {medium:5d}  {'#' * min(medium, 20)}
  [..] Low:       {low:5d}  {'#' * min(low, 20)}
  [OK] Clean:     {clean:5d}  {'#' * min(clean, 20)}

Quality Rate: {clean / len(self.summaries) * 100:.1f}% clean
Review Rate:  {(critical + high) / len(self.summaries) * 100:.1f}% need review

============================================================
"""
        
        # Add top critical issues
        if critical > 0:
            report += "\n[!] TOP CRITICAL ISSUES:\n"
            for s in self.get_critical_summaries()[:5]:
                report += f"  - {s.summary}\n"
        
        return report
