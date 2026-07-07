# Graph Report - IITM VISA AI Hackathon  (2026-07-08)

## Corpus Check
- 64 files · ~68,506 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1098 nodes · 2412 edges · 66 communities (52 shown, 14 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 328 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `15cc90c5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- script.js
- FeatureExtractionLayer
- SemanticValidationLayer
- FieldComplianceLayer
- generate_visa_transactions
- app.py
- DQSEngine
- LayerStatus
- OutputContractLayer
- schema.py
- LayerResult
- StructuralIntegrityLayer
- ResponsibilityBoundaryLayer
- InputValidationLayer
- RecordPayload
- test_phase1.py
- InputContractLayer
- GenAISummarizationLayer
- StabilityConsistencyLayer
- AnomalyDetectionLayer
- csv_adapter.py
- LoggingTraceLayer
- LiveLogStorage
- DecisionGateLayer
- flatten_transactions
- run_full_pipeline
- ._create_result
- layer10_responsibility.py
- TestLayer44AnomalyDetection
- Decision
- SchemaManifest
- .run
- ConfidenceBandLayer
- sample_csv_generator.py
- Any
- TestLayer45Summarization
- QualitySummary
- TestPhase4Integration
- RecordValidation
- AnomalyResult
- .get_assessments_dataframe
- .get_decisions_dataframe
- TestDQSEngineLayerIntegrity
- streaming_worker
- SafeJSONEncoder
- .get_assignments_dataframe
- generate_data
- get_layers
- get_live_logs
- handle_connect
- handle_disconnect
- handle_ping
- health
- set_live_api_url
- .test_confidence_assessment
- DQSEngineLoadTest
- TestFeatureRobustness
- security.py
- run_pipeline
- .get_original_data
- benchmark_throughput.py
- generate_evaluation_dataset.py

## God Nodes (most connected - your core abstractions)
1. `LayerResult` - 92 edges
2. `FeatureExtractionLayer` - 73 edges
3. `LayerStatus` - 71 edges
4. `InputValidationLayer` - 58 edges
5. `generate_visa_transactions()` - 54 edges
6. `DQSEngine` - 54 edges
7. `SemanticValidationLayer` - 52 edges
8. `InputContractLayer` - 50 edges
9. `FieldComplianceLayer` - 43 edges
10. `GenAISummarizationLayer` - 38 edges

## Surprising Connections (you probably didn't know these)
- `SafeJSONEncoder` --uses--> `Action`  [INFERRED]
  app.py → src/config.py
- `SafeJSONEncoder` --uses--> `DQSEngine`  [INFERRED]
  app.py → src/dqs_engine.py
- `SafeJSONEncoder` --uses--> `LiveDataGenerator`  [INFERRED]
  app.py → src/live_data_generator.py
- `SafeJSONEncoder` --uses--> `LiveLogStorage`  [INFERRED]
  app.py → src/live_data_generator.py
- `main()` --calls--> `InputContractLayer`  [INFERRED]
  demo_gemini.py → src/layers/layer1_input_contract.py

## Import Cycles
- None detected.

## Communities (66 total, 14 thin omitted)

### Community 0 - "script.js"
Cohesion: 0.06
Nodes (66): addLiveLogEntry(), appendMasterLog(), checkHealth(), clearLiveLogs(), copyToClipboard(), displayActionRecords(), displayLayerLog(), displayOutputTable() (+58 more)

### Community 1 - "FeatureExtractionLayer"
Cohesion: 0.10
Nodes (15): FeatureExtractionLayer, Layer 3: Feature Extraction Layer          Extracts deterministic features fro, Get list of feature names., Tests for Layer 3: Feature Extraction., Test that feature extraction completes successfully., Test that all 35 expected features are extracted., Test transaction feature extraction., Test card feature extraction. (+7 more)

### Community 2 - "SemanticValidationLayer"
Cohesion: 0.05
Nodes (28): Any, DataFrame, Series, Validate all records against business rules.                  Args:, Result of a business rule evaluation., BR001: Amount must be positive., BR002: Net = Gross - Interchange - Gateway., Semantic validation result for a single record. (+20 more)

### Community 3 - "FieldComplianceLayer"
Cohesion: 0.09
Nodes (20): DimensionScore, Any, DataFrame, Series, Score completeness - % of non-null values in required fields., Score accuracy - % of values matching expected patterns., Score for a single quality dimension., Score validity - % of values within valid ranges/enums. (+12 more)

### Community 4 - "generate_visa_transactions"
Cohesion: 0.20
Nodes (12): main(), Gemini API Integration Demo - Test Phase 4 with AI Summaries, main(), Phase 4 Demo Script - Test the pipeline end-to-end, generate_sample_data(), generate_visa_transactions(), Any, VISA Transaction Data Generator  Generates sample VISA transaction data follow (+4 more)

### Community 5 - "app.py"
Cohesion: 0.06
Nodes (34): clear_live_logs(), generate_data(), get_layers(), get_live_logs(), get_live_stats(), get_schema(), handle_connect(), handle_disconnect() (+26 more)

### Community 6 - "DQSEngine"
Cohesion: 0.15
Nodes (8): debug_one_transaction(), LiveDataGenerator, Live Data Generator for Real-Time Streaming  Generates realistic VISA transact, Generates realistic transaction data for live streaming.          In productio, Initialize the generator.                  Args:             api_key: Optiona, Set the API key for external data source., Set the external API URL for fetching real transactions., Set the anomaly rate (0.0 - 1.0).

### Community 8 - "OutputContractLayer"
Cohesion: 0.10
Nodes (14): OutputContractLayer, Any, DataFrame, Validate outputs from all layers and create structured payload., Find the first matching column., Create a standardized layer result., Get the batch payload., Get all record payloads. (+6 more)

### Community 9 - "schema.py"
Cohesion: 0.13
Nodes (22): BaseModel, AddressDetails, AuthenticationDetails, BusinessMetadata, BusinessRule, CardDetails, ComplianceDetails, CustomerDetails (+14 more)

### Community 10 - "LayerResult"
Cohesion: 0.11
Nodes (33): Action, LayerStatus, Final pipeline actions - the 4 possible outcomes., Status codes for layer execution., PipelineResult, Complete pipeline execution result., Phase 5 Tests: Output & Decision (Layers 5-9)  Tests for output contract, stab, Tests for Layer 5: Output Contract. (+25 more)

### Community 11 - "StructuralIntegrityLayer"
Cohesion: 0.09
Nodes (18): FieldComplianceLayer, Layer 4.2: Field-Level Compliance Scoring          Scores each record across 7, Get indices of rejected records., Get indices of records needing review., get_processed_data(), Phase 3 Tests: Model Inference - Deterministic (Layers 4.1-4.3)  Tests for str, Run Layers 1-3 and return dataframe + features., Tests for Layer 4.2: Field-Level Compliance. (+10 more)

### Community 12 - "ResponsibilityBoundaryLayer"
Cohesion: 0.07
Nodes (28): BatchResponsibility, Any, Assign responsibility for each decision.                  Args:             d, Determine who owns this decision., Determine the source of the decision., Get contribution from each layer., Get escalation path for the decision., Create a standardized layer result. (+20 more)

### Community 13 - "InputValidationLayer"
Cohesion: 0.16
Nodes (9): Any, DataFrame, Validate from file path. Returns LayerResult on failure, None on success., Validate from direct JSON data. Returns LayerResult on failure, None on success., Calculate SHA-256 hash of file for traceability., Create a standardized layer result., Get the validated DataFrame (flattened if from JSON)., Get raw transaction records (if loaded from JSON). (+1 more)

### Community 14 - "RecordPayload"
Cohesion: 0.09
Nodes (43): datetime, Enum, ConfidenceBand, Configuration constants for the Data Quality Scoring Engine. All thresholds and, Confidence classification for decisions., DecisionSource, Layer 10: Responsibility Boundary  Purpose: Track responsibility for decisions, Source of the decision. (+35 more)

### Community 15 - "test_phase1.py"
Cohesion: 0.29
Nodes (5): generate_mixed_csv(), parse_visa_transaction(), Parse a dictionary into a VisaTransaction object., Test that sample transaction parses correctly., Test that transaction flattens correctly.

### Community 16 - "InputContractLayer"
Cohesion: 0.11
Nodes (20): Task 1: Benchmark Anomaly Detector =================================== Measures, run_benchmark(), Task 6: Train Anomaly Detection Model Offline ==================================, train_model(), InputContractLayer, Layer 1: Input Contract Layer          Validates that a valid schema manifest, InputValidationLayer, Layer 2: Input Validation Layer          Validates that actual data complies w (+12 more)

### Community 17 - "GenAISummarizationLayer"
Cohesion: 0.05
Nodes (32): GenAISchemaMapper, Any, Layer 1.2: GenAI Schema Mapper ================================ Uses Gemini and, Pydantic model representing the mapping results to enforce strict schema structu, Queries Gemini to map raw headers to standard fields using Instructor for valida, SchemaMappingResult, FieldCorrection, FieldCorrectionsBatch (+24 more)

### Community 18 - "StabilityConsistencyLayer"
Cohesion: 0.12
Nodes (10): Any, DataFrame, Create a standardized layer result., Get stability metrics., Get all consistency flags., Get consistency flags as DataFrame., Layer 6: Stability & Consistency          Validates that scoring is stable and, StabilityConsistencyLayer (+2 more)

### Community 19 - "AnomalyDetectionLayer"
Cohesion: 0.12
Nodes (12): AnomalyResult, Any, DataFrame, Series, Detect anomalies in the feature DataFrame.                  Args:, Calculate statistical anomaly scores based on z-scores., Calculate rule-based anomaly flags., Anomaly detection result for a single record. (+4 more)

### Community 20 - "csv_adapter.py"
Cohesion: 0.14
Nodes (18): test_csv_adapter(), test_evaluation(), Task 10: Dynamic Evaluation Dataset Runner =====================================, run_evaluation(), adapt_csv_to_visa(), adapt_flat_json_to_visa(), calculate_schema_compliance(), convert_csv_row_to_visa() (+10 more)

### Community 22 - "LiveLogStorage"
Cohesion: 0.17
Nodes (8): LiveLogStorage, Persistent storage for live stream logs.     Stores logs to a JSON file for per, Load existing logs from file., Save logs to file (must be called with lock held)., Add a processed transaction log (thread-safe)., Get logs filtered by time range (thread-safe)., Clear all logs (thread-safe)., Force save logs to disk (thread-safe).

### Community 23 - "DecisionGateLayer"
Cohesion: 0.15
Nodes (8): DecisionGateLayer, Any, Determine action for a record., Create a standardized layer result., Get batch decision summary., Generate a decision summary report., Layer 9: Decision Gate          Makes final action decisions based on all accu, Make final decisions for all records.                  Args:             reco

### Community 24 - "flatten_transactions"
Cohesion: 0.15
Nodes (12): Layer 2: Input Validation Layer  Purpose: Verify that actual data complies wit, flatten_transactions(), Any, Complete VISA Transaction Record.     This is the primary data structure for qu, Flatten nested structure to single-level dict for DataFrame processing., Convert to dictionary for serialization., Flatten a list of nested transactions for DataFrame creation., VisaTransaction (+4 more)

### Community 25 - "run_full_pipeline"
Cohesion: 0.09
Nodes (23): Who is responsible for the decision., ResponsibilityOwner, LoggingTraceLayer, Any, Layer 11: Logging & Trace          Provides comprehensive logging and traceabi, Mark the start of pipeline execution., Create a standardized layer result., Export execution log to JSON. (+15 more)

### Community 26 - "._create_result"
Cohesion: 0.06
Nodes (18): debug_one_transaction(), DQSEngine, Data Quality Scoring Engine          Orchestrates all 15 layers of the DQS pip, Test engine initializes correctly., Test processing a single transaction., Test that result has all required fields., Test processing 20 records., Test that action counts sum to total. (+10 more)

### Community 27 - "layer10_responsibility.py"
Cohesion: 0.20
Nodes (12): ColumnDefinition, DataType, QualityThresholds, Supported data types for columns., Definition for a single column in the schema., Quality thresholds per dimension., str, Phase 1 Tests: Input Contract (Layer 1) and Input Validation (Layer 2)  Update (+4 more)

### Community 28 - "TestLayer44AnomalyDetection"
Cohesion: 0.11
Nodes (17): AnomalyDetectionLayer, Get indices of flagged records., Get indices of high-risk anomalies., Layer 4.4: Anomaly Detection          Uses ensemble methods to detect anomalou, Phase 4 Tests: AI Model Inference (Layers 4.4-4.5)  Tests for anomaly detectio, Run Layers 1-3 and return dataframe + features., Tests for Layer 4.4: Anomaly Detection., Test that anomaly detection completes. (+9 more)

### Community 29 - "Decision"
Cohesion: 0.25
Nodes (8): handle_start_stream(), process_single_transaction(), Process a single transaction through DQS engine., Recursively sanitize an object for JSON serialization., Background worker for streaming transactions (thread-safe)., Start the live transaction stream (thread-safe)., sanitize_for_json(), streaming_worker()

### Community 30 - "SchemaManifest"
Cohesion: 0.22
Nodes (5): Get the validated schema manifest., The complete schema manifest that defines what data we accept.     This is the, Get list of required column names., Get column definition by name., SchemaManifest

### Community 31 - ".run"
Cohesion: 0.17
Nodes (8): LayerTiming, main(), Any, Log layer end and record timing., Run the complete DQS pipeline.                  Args:             transaction, Timing information for a layer., Generate a report of layer timings., Run the DQS Engine demo.

### Community 33 - "sample_csv_generator.py"
Cohesion: 0.23
Nodes (11): generate_high_quality_csv(), generate_low_quality_csv(), generate_medium_quality_csv(), generate_nonstandard_csv(), generate_sample_csvs(), Sample CSV Generator  Creates sample CSV files with varying quality levels for, Generate low-quality transactions with many anomalies., Generate sample CSV files with different quality levels. (+3 more)

### Community 35 - "TestLayer45Summarization"
Cohesion: 0.16
Nodes (10): GenAISummarizationLayer, Layer 4.5: GenAI Summarization          Generates human-readable explanations, Initialize the summarization layer.                  Args:             api_ke, Tests for Layer 4.5: GenAI Summarization., Test that summarization completes., Test that summary has all required fields., Test that clean record gets 'none' priority., Test that batch report is generated. (+2 more)

### Community 36 - "QualitySummary"
Cohesion: 0.13
Nodes (11): DataFrame, Extract all features from the validated VISA transaction DataFrame., Extract transaction-related features., Extract card-related features., Extract merchant-related features., Extract customer-related features., Extract fraud-related features., Extract authentication-related features. (+3 more)

### Community 37 - "TestPhase4Integration"
Cohesion: 0.15
Nodes (12): test_database(), DQSDatabase, mask_pii(), DQS Engine History Database Manager ==================================== Manages, Retrieves recent run history., Helper to redact sensitive data from strings (GDPR / PCI compliance)., Creates the run history table if it doesn't exist., Persists a pipeline execution run summary. (+4 more)

### Community 39 - "AnomalyResult"
Cohesion: 0.13
Nodes (12): Any, DataFrame, Series, Check for corrupted data patterns., Validation result for a single record., Add a rejection to the list., Summarize issues by type., Find the first matching column. (+4 more)

### Community 45 - "streaming_worker"
Cohesion: 0.13
Nodes (8): Comprehensive codebase test - checks everything that isn't covered by pytest, test_data_generator(), test_full_pipeline(), test_models(), test_rules_count(), test_security(), Decorator to validate HMAC-SHA256 signature on requests.     Validates if REQUIR, require_hmac_signature()

### Community 48 - "generate_data"
Cohesion: 0.16
Nodes (9): Layer 4.1: Structural Integrity Validation          Validates that each record, Get indices of valid records., Get indices of rejected records., StructuralIntegrityLayer, Tests for Layer 4.1: Structural Integrity., Test that valid record passes structural checks., Test that duplicate primary keys are rejected., Test that missing required field causes rejection. (+1 more)

### Community 49 - "get_layers"
Cohesion: 0.15
Nodes (9): ConflictDetectionLayer, Any, DataFrame, Create a standardized layer result., Get all detected conflicts., Get high severity conflicts., Get conflicts as DataFrame., Layer 7: Conflict Detection          Detects conflicts between different quali (+1 more)

### Community 50 - "get_live_logs"
Cohesion: 0.14
Nodes (8): ConfidenceBandLayer, Any, Create a standardized layer result., Get all confidence assessments., Get records with low confidence., Layer 8: Confidence Band          Calculates confidence levels for quality ass, Test decision gate produces action., Test that clean record gets SAFE_TO_USE.

### Community 51 - "handle_connect"
Cohesion: 0.14
Nodes (8): Tests for Layer 2: Input Validation with VISA schema., Test that single JSON transaction validates., Test that multiple JSON transactions validate., Test JSON file validation., Test that flattened DataFrame has expected columns., Test that non-existent file fails., Test that no data provided fails., TestLayer2InputValidation

### Community 52 - "handle_disconnect"
Cohesion: 0.17
Nodes (8): DQS Engine: Complete Data Quality Scoring Pipeline  This is the main orchestra, Initialize the DQS Engine.                  Args:             gemini_api_key:, Layer 4.3: Semantic Validation          Validates business rules and domain-sp, Get indices of rejected records., Get indices of flagged records., SemanticValidationLayer, Run Layers 1-4.5 and return all outputs., run_layers_1_to_4()

### Community 53 - "handle_ping"
Cohesion: 0.22
Nodes (6): Any, Wrap flat external data into expected nested format., Generate a simulated transaction (original logic)., Flatten nested transaction to match DQS engine input format.         Uses prefi, Get aggregate statistics (thread-safe)., Generate a single realistic transaction.                  If api_url is config

### Community 54 - "health"
Cohesion: 0.22
Nodes (6): Any, Create a standardized layer result., Validate the provided schema manifest.                  Args:             sch, create_default_transaction_schema(), Create the default VISA transaction schema., Get default schema for testing.

### Community 55 - "set_live_api_url"
Cohesion: 0.20
Nodes (6): Tests for Layer 1: Input Contract validation., Test that default schema validates successfully., Test that missing schema causes CONTRACT_VIOLATION., Test that a valid JSON schema file passes., Test that invalid JSON causes failure., TestLayer1InputContract

### Community 56 - ".test_confidence_assessment"
Cohesion: 0.33
Nodes (4): Any, Create a standardized layer result., Create a failed layer result., Get statistics about extracted features.

### Community 57 - "DQSEngineLoadTest"
Cohesion: 0.33
Nodes (4): HttpUser, DQSEngineLoadTest, Locust Performance Load Test Script for DQS Engine =============================, Simulates users sending batches of transactions for real-time quality scoring.

### Community 58 - "TestFeatureRobustness"
Cohesion: 0.33
Nodes (4): Test robustness and edge cases., Test extraction with missing optional fields., Test feature extraction for high-risk transaction., TestFeatureRobustness

### Community 60 - "run_pipeline"
Cohesion: 0.50
Nodes (4): get_engine(), Get or create engine instance (thread-safe singleton)., Run the complete DQS pipeline with graceful CSV/JSON handling., run_pipeline()

## Knowledge Gaps
- **3 isolated node(s):** `LAYERS`, `LAYER_ORDER`, `liveTransactionLogs`
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LayerResult` connect `RecordPayload` to `FeatureExtractionLayer`, `SemanticValidationLayer`, `FieldComplianceLayer`, `OutputContractLayer`, `StructuralIntegrityLayer`, `ResponsibilityBoundaryLayer`, `InputValidationLayer`, `InputContractLayer`, `GenAISummarizationLayer`, `StabilityConsistencyLayer`, `AnomalyDetectionLayer`, `DecisionGateLayer`, `flatten_transactions`, `run_full_pipeline`, `TestLayer44AnomalyDetection`, `TestLayer45Summarization`, `QualitySummary`, `AnomalyResult`, `generate_data`, `get_layers`, `get_live_logs`, `handle_disconnect`, `health`, `.test_confidence_assessment`?**
  _High betweenness centrality (0.194) - this node is a cross-community bridge._
- **Why does `FieldComplianceLayer` connect `StructuralIntegrityLayer` to `FieldComplianceLayer`, `generate_visa_transactions`, `TestLayer45Summarization`, `streaming_worker`, `RecordPayload`, `generate_data`, `InputContractLayer`, `handle_disconnect`, `run_full_pipeline`, `TestLayer44AnomalyDetection`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `LayerStatus` connect `LayerResult` to `FeatureExtractionLayer`, `SemanticValidationLayer`, `FieldComplianceLayer`, `OutputContractLayer`, `StructuralIntegrityLayer`, `ResponsibilityBoundaryLayer`, `InputValidationLayer`, `RecordPayload`, `InputContractLayer`, `GenAISummarizationLayer`, `StabilityConsistencyLayer`, `AnomalyDetectionLayer`, `DecisionGateLayer`, `flatten_transactions`, `run_full_pipeline`, `layer10_responsibility.py`, `TestLayer44AnomalyDetection`, `TestLayer45Summarization`, `AnomalyResult`, `streaming_worker`, `generate_data`, `get_layers`, `get_live_logs`, `handle_connect`, `health`, `set_live_api_url`, `.test_confidence_assessment`, `TestFeatureRobustness`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 41 inferred relationships involving `LayerResult` (e.g. with `BatchResponsibility` and `DecisionSource`) actually correct?**
  _`LayerResult` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `FeatureExtractionLayer` (e.g. with `main()` and `main()`) actually correct?**
  _`FeatureExtractionLayer` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `LayerStatus` (e.g. with `TestLayer1InputContract` and `TestLayer2InputValidation`) actually correct?**
  _`LayerStatus` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `InputValidationLayer` (e.g. with `main()` and `main()`) actually correct?**
  _`InputValidationLayer` has 20 INFERRED edges - model-reasoned connections that need verification._