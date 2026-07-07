# Layer Reference — All 15 Pipeline Layers

---

## L1 — Input Contract (`layer1_input_contract.py`)

**Class:** `InputContractLayer`  
**Phase:** Foundation  
**Type:** Deterministic

Validates the shape and structure of incoming data before any processing begins. Generates a `batch_id`, timestamps the execution, and attaches the schema manifest. Accepts both single transactions (dict) and batches (list).

**Outputs:**
- `LayerResult` with status, timing
- `get_schema()` → schema manifest dict

**Failure modes:**
- Returns `FAILED` if input is not a dict or list
- Returns `FAILED` if schema manifest cannot be loaded

---

## L2 — Input Validation (`layer2_input_validation.py`)

**Class:** `InputValidationLayer`  
**Phase:** Foundation  
**Type:** Deterministic

Flattens nested transaction JSON into a pandas DataFrame using the schema manifest. Performs column normalization, type coercion, and basic presence checks. Handles both the canonical nested format and the flat CSV-adapted format.

**Outputs:**
- `LayerResult`
- `get_dataframe()` → flat pd.DataFrame (e.g., `txn_amount`, `card_network`, `merchant_merchant_id`, ...)

**Key column naming convention:** `{section}_{field}` (e.g., `customer_email`, `fraud_risk_score`)

---

## L3 — Feature Extraction (`layer3_feature_extraction.py`)

**Class:** `FeatureExtractionLayer`  
**Phase:** Feature Extraction  
**Type:** Deterministic

Engineers 35 numerical features from the flat DataFrame. These features are used by both the deterministic rules (L4.3) and the ML anomaly detector (L4.4).

**Feature groups:**

| Group | Features |
|---|---|
| Amount | `txn_amount`, `txn_amount_zscore`, `txn_amount_percentile` |
| Temporal | `txn_hour`, `txn_is_weekend` |
| Card | `card_type_encoded`, `card_is_domestic_issuer`, `card_bin_category` |
| Merchant | `merchant_is_domestic`, `merchant_country_risk` |
| Risk | `fraud_risk_score`, `fraud_velocity_passed`, `fraud_geo_passed` |
| Settlement | `settlement_fee_ratio` |
| Auth | `auth_result_encoded` |
| Customer | `customer_address_match`, `customer_ip_is_domestic` |

**Special computation — bidirectional velocity:**
```
time_diff = min(time_diff_prev, time_diff_next)
fraud_velocity_passed = 0 if time_diff < 60s else 1
```
This ensures the first transaction in a velocity burst is also flagged.

**Outputs:**
- `LayerResult`
- `get_features()` → pd.DataFrame with 35 feature columns

---

## L4.1 — Structural Integrity (`layer4_1_structural.py`)

**Class:** `StructuralIntegrityLayer`  
**Phase:** Deterministic Inference  
**Type:** Deterministic

Checks structural properties across the dataset:
- **Primary key uniqueness:** Duplicate `txn_transaction_id` → duplicate records rejected
- **Required field presence:** Missing critical fields → record rejected
- **Dtype consistency:** Numeric fields must be numeric

**Outputs:**
- `get_valid_indices()` → list of row indices that passed all checks
- `get_rejected_indices()` → list of rejected indices
- `get_validation_results()` → per-record structural audit

---

## L4.2 — Field Compliance (`layer4_2_field_compliance.py`)

**Class:** `FieldComplianceLayer`  
**Phase:** Deterministic Inference  
**Type:** Deterministic

Scores each record across 7 data quality dimensions (0–100 each). Computes a weighted **Data Quality Score (DQS)**:

```
DQS = Σ(dimension_score × weight) / Σ(weights)
```

| Dimension | Weight | Key checks |
|---|---|---|
| Completeness | 0.25 | Required fields non-null |
| Accuracy | 0.20 | TXN ID, BIN, email, MCC format regex |
| Validity | 0.20 | Amount > 0, status in enum, card network in list |
| Uniqueness | 0.10 | No duplicate TXN IDs in batch |
| Consistency | 0.15 | Settlement math, date ordering |
| Timeliness | 0.05 | Timestamps not stale (>30 days) |
| Integrity | 0.05 | Merchant ID ↔ name, network TXN ID for approved |

**Decision thresholds:**
- `DQS < 60` → `REJECTION_THRESHOLD` (rejected)
- `DQS < 75` → `REVIEW_THRESHOLD` (flagged for review)
- `DQS ≥ 75` → passes threshold

**Outputs:**
- `get_dqs_dataframe()` → DataFrame with `dqs_base`, `dim_*` columns
- `get_record_scores()` → list of `RecordQualityScore` objects
- `get_rejected_indices()`, `get_review_indices()`

---

## L4.3 — Semantic Validation (`layer4_3_semantic.py`)

**Class:** `SemanticValidationLayer`  
**Phase:** Deterministic Inference  
**Type:** Deterministic

Evaluates 15 business rules (BR001–BR015) on each record. See [`semantic_rules.md`](semantic_rules.md) for full rule reference.

**Rule severity override:** The orchestrator always applies the severity declared in `_define_rules()`, not the one returned by the check method.

**Outputs:**
- `get_validation_results()` → list of `SemanticValidation` objects
- `get_rejected_indices()` → records with critical violations
- `get_flagged_indices()` → records with warning violations

---

## L4.4 — Anomaly Detection (`layer4_4_anomaly.py`)

**Class:** `AnomalyDetectionLayer`  
**Phase:** AI Inference  
**Type:** ML (IsolationForest)

Loads a pre-trained `IsolationForest` model from `src/resources/anomaly_model.pkl`. Applies `StandardScaler` normalization then predicts anomaly scores for 14 numerical features. Combines ML score with a rule-based flag score for a final `anomaly_score ∈ [0,1]`.

**Rule-based flags and score contributions:**

| Flag | Score added | Condition |
|---|---|---|
| `EXTREME_AMOUNT` | +0.4 | `txn_amount_percentile > 99` |
| `HIGH_RISK_SCORE` | +0.5 | `fraud_risk_score > 85` |
| `VELOCITY_FAIL` | +0.3 | `fraud_velocity_passed == 0` |
| `GEO_FAIL` | +0.3 | `fraud_geo_passed == 0` |
| `INTL_HIGH_RISK` | +0.4 | International + risk > 50 |
| `ADDRESS_MISMATCH_HIGH_VALUE` | +0.3 | Address mismatch + high amount |
| `AUTH_FAILED` | +0.4 | `auth_result_encoded == 2` |
| `UNUSUAL_HOUR` | +0.2 | Hour in 2–5 AM |

Final score = `ML_score × 0.6 + rule_score × 0.4` (capped at 1.0)

**Outputs:**
- `get_anomaly_dataframe()` → DataFrame with `anomaly_score`, `is_anomaly`, flags
- Threshold: `is_anomaly = True` if `anomaly_score > ANOMALY_MEDIUM_THRESHOLD` (0.5)

---

## L4.5 — GenAI Summarization (`layer4_5_summarization.py`)

**Class:** `GenAISummarizationLayer`  
**Phase:** AI Inference  
**Type:** GenAI (Gemini)

Generates human-readable quality summaries for each record. When `use_ai=True`, calls the Gemini API using the `instructor` library to enforce Pydantic structured output. Falls back to deterministic template-based summaries when `use_ai=False`.

**Instructor retry loop:** If Gemini returns invalid/incomplete JSON (max 3 retries), the validation error is fed back as a correction prompt.

**Outputs:**
- `get_summaries()` → list of `QualitySummary` objects
  - `overall_quality`: "excellent" / "good" / "fair" / "poor"
  - `key_issues`: list of issue strings
  - `recommended_action`: human-readable action
  - `context`: dict with quality data, suggested corrections

---

## L5 — Output Contract (`layer5_output_contract.py`)

**Class:** `OutputContractLayer`  
**Phase:** Output & Decision  
**Type:** Deterministic

Assembles all layer outputs into standardized `RecordPayload` dataclass objects for downstream consumption.

**`RecordPayload` fields:**
```python
record_id: str
dqs_base: float           # From L4.2
semantic_score: float     # From L4.3
anomaly_score: float      # From L4.4
is_anomaly: bool
anomaly_flags: List[str]
semantic_violations: List[str]   # Critical violations only
warning_violations: List[str]
structural_issues: List[str]
is_valid: bool
priority: str             # "none" / "low" / "medium" / "high" / "critical"
```

---

## L6 — Stability & Consistency (`layer6_stability.py`)

**Class:** `StabilityConsistencyLayer`  
**Phase:** Output & Decision  
**Type:** Deterministic

Computes batch-level statistical properties:
- DQS distribution (mean, std, skewness, kurtosis)
- Detects bimodal or highly skewed DQS distributions
- Returns `consistency_score` used by L8

---

## L7 — Conflict Detection (`layer7_conflict.py`)

**Class:** `ConflictDetectionLayer`  
**Phase:** Output & Decision  
**Type:** Deterministic

Detects four types of signal contradictions:

| Conflict Type | Description |
|---|---|
| `RULE_ML_CONFLICT` | Passes all rules but ML anomaly > 0.7 |
| `SCORE_PRIORITY_CONFLICT` | High DQS but high priority assigned |
| `DIMENSION_OVERALL_CONFLICT` | One dimension < 50 while overall DQS > 70 |
| `VALIDITY_ANOMALY_CONFLICT` | Structurally valid but highly anomalous |

---

## L8 — Confidence Band (`layer8_confidence.py`)

**Class:** `ConfidenceBandLayer`  
**Phase:** Output & Decision  
**Type:** Deterministic

Assigns a confidence band to each record based on:
- DQS score and its deviation from batch mean
- Number of conflicts detected
- Batch consistency score
- Conflict severity

**Bands:** `HIGH` → `MEDIUM` → `LOW`

Used by L9 to gate ML anomaly escalations: a record with anomaly_score=0.7 is only escalated if confidence is `HIGH` or `MEDIUM`.

---

## L9 — Decision Gate (`layer9_decision.py`)

**Class:** `DecisionGateLayer`  
**Phase:** Output & Decision  
**Type:** Deterministic

The final decision maker. Applies rules in strict priority order:

```
Rule 1: anomaly_score > 0.9            → ESCALATE (unconditional)
Rule 2: is_anomaly + score > 0.75      → ESCALATE (if HIGH/MEDIUM confidence)
Rule 3: semantic_violations present    → ESCALATE
Rule 4: dqs_base < 75                  → REVIEW
Rule 5: All clear                      → SAFE_TO_USE
```

---

## L10 — Responsibility Boundary (`layer10_responsibility.py`)

**Class:** `ResponsibilityBoundaryLayer`  
**Phase:** Responsibility & Logging  
**Type:** Deterministic

Tags each decision with accountability metadata:
- Which layer triggered the decision
- Which rule was the proximate cause
- Confidence band at time of decision
- Audit trail for regulatory review

---

## L11 — Logging & Trace (`layer11_logging.py`)

**Class:** `LoggingTraceLayer`  
**Phase:** Responsibility & Logging  
**Type:** Deterministic

Writes structured execution logs for the entire batch:
- Per-layer timing and status
- Per-record decision trace
- Batch-level statistics
- **PII-stripped:** no customer email, PAN, or IP in log entries
