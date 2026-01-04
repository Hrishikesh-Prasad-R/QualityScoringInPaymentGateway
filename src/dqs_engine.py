"""
DQS Engine: Complete Data Quality Scoring Pipeline

This is the main orchestrator that runs all 15 layers of the
Data Quality Scoring Engine with proper logging and timestamps.
"""
import time
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DQS_Engine")


@dataclass
class LayerTiming:
    """Timing information for a layer."""
    layer_id: float
    layer_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: str


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    success: bool
    batch_id: str
    execution_id: str
    
    # Counts
    total_records: int
    safe_count: int
    review_count: int
    escalate_count: int
    rejected_count: int
    
    # Quality metrics
    average_dqs: float
    quality_rate: float
    
    # Timing
    total_duration_ms: float
    layer_timings: List[LayerTiming]
    
    # Reports
    decision_report: str
    execution_report: str
    
    # Errors
    errors: List[str] = field(default_factory=list)


class DQSEngine:
    """
    Data Quality Scoring Engine
    
    Orchestrates all 15 layers of the DQS pipeline with
    proper logging, timestamps, and error handling.
    """
    
    def __init__(self, gemini_api_key: str = None, use_ai: bool = False):
        """
        Initialize the DQS Engine.
        
        Args:
            gemini_api_key: Optional Gemini API key for AI summaries
            use_ai: Whether to use AI-enhanced summaries
        """
        self.gemini_api_key = gemini_api_key
        self.use_ai = use_ai
        self.layer_timings: List[LayerTiming] = []
        self.layer_results: Dict[float, Any] = {}
        
        # Import layers
        from src.layers import (
            InputContractLayer,
            InputValidationLayer,
            FeatureExtractionLayer,
            StructuralIntegrityLayer,
            FieldComplianceLayer,
            SemanticValidationLayer,
            AnomalyDetectionLayer,
            GenAISummarizationLayer,
            OutputContractLayer,
            StabilityConsistencyLayer,
            ConflictDetectionLayer,
            ConfidenceBandLayer,
            DecisionGateLayer,
            ResponsibilityBoundaryLayer,
            LoggingTraceLayer,
        )
        
        self.layers = {
            1: InputContractLayer,
            2: InputValidationLayer,
            3: FeatureExtractionLayer,
            4.1: StructuralIntegrityLayer,
            4.2: FieldComplianceLayer,
            4.3: SemanticValidationLayer,
            4.4: AnomalyDetectionLayer,
            4.5: GenAISummarizationLayer,
            5: OutputContractLayer,
            6: StabilityConsistencyLayer,
            7: ConflictDetectionLayer,
            8: ConfidenceBandLayer,
            9: DecisionGateLayer,
            10: ResponsibilityBoundaryLayer,
            11: LoggingTraceLayer,
        }
    
    def _log_layer_start(self, layer_id: float, layer_name: str):
        """Log layer start."""
        logger.info(f"[L{layer_id}] Starting {layer_name}...")
        return datetime.now()
    
    def _log_layer_end(self, layer_id: float, layer_name: str, start: datetime, status: str):
        """Log layer end and record timing."""
        end = datetime.now()
        duration_ms = (end - start).total_seconds() * 1000
        
        timing = LayerTiming(
            layer_id=layer_id,
            layer_name=layer_name,
            start_time=start,
            end_time=end,
            duration_ms=duration_ms,
            status=status,
        )
        self.layer_timings.append(timing)
        
        status_symbol = "OK" if status == "passed" else "!!" if status == "failed" else "??"
        logger.info(f"[L{layer_id}] Completed {layer_name} [{status_symbol}] ({duration_ms:.1f}ms)")
        
        return timing
    
    def run(self, transactions: Any) -> PipelineResult:
        """
        Run the complete DQS pipeline.
        
        Args:
            transactions: VISA transactions (single dict or list)
            
        Returns:
            PipelineResult with all metrics and reports
        """
        import uuid
        
        pipeline_start = datetime.now()
        self.layer_timings = []
        self.layer_results = {}
        errors = []
        
        logger.info("=" * 60)
        logger.info("  DQS ENGINE - Starting Pipeline Execution")
        logger.info("=" * 60)
        
        try:
            # ================================================================
            # PHASE 1: FOUNDATION (Layers 1-2)
            # ================================================================
            logger.info("-" * 40)
            logger.info("PHASE 1: Foundation")
            logger.info("-" * 40)
            
            # Layer 1: Input Contract
            start = self._log_layer_start(1, "Input Contract")
            layer1 = self.layers[1]()
            result1 = layer1.validate_schema_manifest(use_default=True)
            self.layer_results[1] = result1
            self._log_layer_end(1, "Input Contract", start, result1.status.value)
            
            # Layer 2: Input Validation
            start = self._log_layer_start(2, "Input Validation")
            layer2 = self.layers[2](layer1.get_schema())
            result2 = layer2.validate(json_data=transactions)
            self.layer_results[2] = result2
            self._log_layer_end(2, "Input Validation", start, result2.status.value)
            
            dataframe = layer2.get_dataframe()
            n_records = len(dataframe)
            logger.info(f"  Records loaded: {n_records}")
            
            # ================================================================
            # PHASE 2: FEATURE EXTRACTION (Layer 3)
            # ================================================================
            logger.info("-" * 40)
            logger.info("PHASE 2: Feature Extraction")
            logger.info("-" * 40)
            
            # Layer 3: Feature Extraction
            start = self._log_layer_start(3, "Feature Extraction")
            layer3 = self.layers[3]()
            result3 = layer3.extract_features(dataframe)
            self.layer_results[3] = result3
            self._log_layer_end(3, "Feature Extraction", start, result3.status.value)
            
            features_df = layer3.get_features()
            logger.info(f"  Features extracted: {result3.details.get('features_extracted', 0)}")
            
            # ================================================================
            # PHASE 3: DETERMINISTIC INFERENCE (Layers 4.1-4.3)
            # ================================================================
            logger.info("-" * 40)
            logger.info("PHASE 3: Deterministic Inference")
            logger.info("-" * 40)
            
            # Layer 4.1: Structural Integrity
            start = self._log_layer_start(4.1, "Structural Integrity")
            layer41 = self.layers[4.1]()
            result41 = layer41.validate(dataframe, features_df)
            self.layer_results[4.1] = result41
            self._log_layer_end(4.1, "Structural Integrity", start, result41.status.value)
            logger.info(f"  Valid: {result41.details.get('valid_records', 0)}, Rejected: {result41.details.get('rejected_records', 0)}")
            
            valid_indices = layer41.get_valid_indices()
            
            # Layer 4.2: Field Compliance
            start = self._log_layer_start(4.2, "Field Compliance")
            layer42 = self.layers[4.2]()
            result42 = layer42.score(dataframe, features_df, valid_indices)
            self.layer_results[4.2] = result42
            self._log_layer_end(4.2, "Field Compliance", start, result42.status.value)
            logger.info(f"  Mean DQS: {result42.details.get('dqs_mean', 0):.1f}")
            
            # Layer 4.3: Semantic Validation
            start = self._log_layer_start(4.3, "Semantic Validation")
            layer43 = self.layers[4.3]()
            result43 = layer43.validate(dataframe, features_df, valid_indices)
            self.layer_results[4.3] = result43
            self._log_layer_end(4.3, "Semantic Validation", start, result43.status.value)
            
            # ================================================================
            # PHASE 4: AI INFERENCE (Layers 4.4-4.5)
            # ================================================================
            logger.info("-" * 40)
            logger.info("PHASE 4: AI Inference")
            logger.info("-" * 40)
            
            # Layer 4.4: Anomaly Detection
            start = self._log_layer_start(4.4, "Anomaly Detection")
            layer44 = self.layers[4.4]()
            result44 = layer44.detect(features_df, valid_indices)
            self.layer_results[4.4] = result44
            self._log_layer_end(4.4, "Anomaly Detection", start, result44.status.value)
            logger.info(f"  Flagged: {result44.details.get('records_flagged', 0)}")
            
            # Layer 4.5: GenAI Summarization
            start = self._log_layer_start(4.5, "GenAI Summarization")
            layer45 = self.layers[4.5](api_key=self.gemini_api_key, use_ai=self.use_ai)
            result45 = layer45.summarize(
                dataframe,
                features_df,
                layer42.get_dqs_dataframe(),
                layer44.get_anomaly_dataframe(),
                layer43.get_validation_results(),
            )
            self.layer_results[4.5] = result45
            self._log_layer_end(4.5, "GenAI Summarization", start, result45.status.value)
            
            # ================================================================
            # PHASE 5: OUTPUT & DECISION (Layers 5-9)
            # ================================================================
            logger.info("-" * 40)
            logger.info("PHASE 5: Output & Decision")
            logger.info("-" * 40)
            
            # Layer 5: Output Contract
            start = self._log_layer_start(5, "Output Contract")
            layer5 = self.layers[5]()
            result5 = layer5.validate_and_structure(
                layer_results=self.layer_results,
                dataframe=dataframe,
                features_df=features_df,
                dqs_df=layer42.get_dqs_dataframe(),
                anomaly_df=layer44.get_anomaly_dataframe(),
                summaries=layer45.get_summaries(),
                structural_results=layer41.get_validation_results(),
                semantic_results=layer43.get_validation_results(),
            )
            self.layer_results[5] = result5
            self._log_layer_end(5, "Output Contract", start, result5.status.value)
            
            batch_payload = layer5.get_batch_payload()
            batch_id = batch_payload.batch_id
            
            # Layer 6: Stability
            start = self._log_layer_start(6, "Stability & Consistency")
            layer6 = self.layers[6]()
            result6 = layer6.validate(layer5.get_record_payloads())
            self.layer_results[6] = result6
            self._log_layer_end(6, "Stability & Consistency", start, result6.status.value)
            
            # Layer 7: Conflict Detection
            start = self._log_layer_start(7, "Conflict Detection")
            layer7 = self.layers[7]()
            result7 = layer7.detect(layer5.get_record_payloads(), layer42.get_dqs_dataframe())
            self.layer_results[7] = result7
            self._log_layer_end(7, "Conflict Detection", start, result7.status.value)
            logger.info(f"  Conflicts: {result7.details.get('total_conflicts', 0)}")
            
            # Layer 8: Confidence Band
            start = self._log_layer_start(8, "Confidence Band")
            layer8 = self.layers[8]()
            stability_score = layer6.get_stability_metrics().consistency_score if layer6.get_stability_metrics() else 100
            result8 = layer8.assess(
                layer5.get_record_payloads(),
                layer6.get_consistency_flags(),
                layer7.get_conflicts(),
                stability_score,
            )
            self.layer_results[8] = result8
            self._log_layer_end(8, "Confidence Band", start, result8.status.value)
            
            # Layer 9: Decision Gate
            start = self._log_layer_start(9, "Decision Gate")
            layer9 = self.layers[9]()
            result9 = layer9.decide(
                layer5.get_record_payloads(),
                layer8.get_assessments(),
                batch_id=batch_id,
            )
            self.layer_results[9] = result9
            self._log_layer_end(9, "Decision Gate", start, result9.status.value)
            
            safe_count = result9.details.get('safe_count', 0)
            review_count = result9.details.get('review_count', 0)
            escalate_count = result9.details.get('escalate_count', 0)
            no_action_count = result9.details.get('no_action_count', 0)
            
            logger.info(f"  SAFE: {safe_count}, REVIEW: {review_count}, ESCALATE: {escalate_count}, NO_ACTION: {no_action_count}")
            
            # ================================================================
            # PHASE 6: RESPONSIBILITY & LOGGING (Layers 10-11)
            # ================================================================
            logger.info("-" * 40)
            logger.info("PHASE 6: Responsibility & Logging")
            logger.info("-" * 40)
            
            # Layer 10: Responsibility Boundary
            start = self._log_layer_start(10, "Responsibility Boundary")
            layer10 = self.layers[10]()
            result10 = layer10.assign(layer9.get_decisions(), batch_id=batch_id)
            self.layer_results[10] = result10
            self._log_layer_end(10, "Responsibility Boundary", start, result10.status.value)
            
            # Layer 11: Logging & Trace
            start = self._log_layer_start(11, "Logging & Trace")
            layer11 = self.layers[11]()
            layer11.start_pipeline()
            result11 = layer11.log(
                self.layer_results,
                layer9.get_decisions(),
                layer10.get_assignments(),
                batch_id=batch_id,
                total_records=n_records,
            )
            self.layer_results[11] = result11
            self._log_layer_end(11, "Logging & Trace", start, result11.status.value)
            
            # ================================================================
            # COMPLETE
            # ================================================================
            pipeline_end = datetime.now()
            total_duration = (pipeline_end - pipeline_start).total_seconds() * 1000
            
            logger.info("=" * 60)
            logger.info("  DQS ENGINE - Pipeline Complete")
            logger.info("=" * 60)
            logger.info(f"  Total Duration: {total_duration:.2f}ms")
            logger.info(f"  Records Processed: {n_records}")
            logger.info(f"  Quality Rate: {result9.details.get('quality_rate', 0):.1f}%")
            logger.info("=" * 60)
            
            # Build result
            return PipelineResult(
                success=True,
                batch_id=batch_id,
                execution_id=result11.details.get('execution_id', str(uuid.uuid4())),
                total_records=n_records,
                safe_count=safe_count,
                review_count=review_count,
                escalate_count=escalate_count,
                rejected_count=no_action_count,
                average_dqs=result42.details.get('dqs_mean', 0),
                quality_rate=result9.details.get('quality_rate', 0),
                total_duration_ms=total_duration,
                layer_timings=self.layer_timings,
                decision_report=layer9.generate_decision_report(),
                execution_report=layer11.generate_execution_report(),
                errors=errors,
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            errors.append(str(e))
            
            pipeline_end = datetime.now()
            total_duration = (pipeline_end - pipeline_start).total_seconds() * 1000
            
            return PipelineResult(
                success=False,
                batch_id="",
                execution_id="",
                total_records=0,
                safe_count=0,
                review_count=0,
                escalate_count=0,
                rejected_count=0,
                average_dqs=0,
                quality_rate=0,
                total_duration_ms=total_duration,
                layer_timings=self.layer_timings,
                decision_report="",
                execution_report="",
                errors=errors,
            )
    
    def get_layer_timings_report(self) -> str:
        """Generate a report of layer timings."""
        if not self.layer_timings:
            return "No layer timings available."
        
        report = """
============================================================
               LAYER TIMING REPORT
============================================================

"""
        total = 0
        for t in self.layer_timings:
            status = "OK" if t.status == "passed" else "!!"
            report += f"  L{t.layer_id:4} | {t.layer_name:25s} | {t.duration_ms:8.2f}ms | [{status}]\n"
            total += t.duration_ms
        
        report += f"""
------------------------------------------------------------
  TOTAL                                     | {total:8.2f}ms
============================================================
"""
        return report


def main():
    """Run the DQS Engine demo."""
    import sys
    sys.path.insert(0, '.')
    
    from src.data_generator import generate_visa_transactions
    
    print("\n" + "=" * 60)
    print("  DATA QUALITY SCORING ENGINE - DEMO")
    print("=" * 60 + "\n")
    
    # Generate test data
    print("Generating 50 test transactions (15% anomalies)...\n")
    transactions = generate_visa_transactions(
        n_transactions=50,
        anomaly_rate=0.15,
        random_seed=42,
    )
    
    # Run engine
    engine = DQSEngine(use_ai=False)
    result = engine.run(transactions)
    
    # Print reports
    print("\n" + result.decision_report)
    print(engine.get_layer_timings_report())
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Success: {result.success}")
    print(f"  Batch ID: {result.batch_id}")
    print(f"  Total Records: {result.total_records}")
    print(f"  Quality Rate: {result.quality_rate:.1f}%")
    print(f"  Total Duration: {result.total_duration_ms:.2f}ms")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
