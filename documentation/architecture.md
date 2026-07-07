# Architecture — DQS Engine

## Overview

The DQS Engine is a **15-layer sequential pipeline** built on a strict hierarchy:

> **Rules Enforce. ML Informs. Humans Decide.**

- **Deterministic layers** (L4.1–L4.3) have **hard power** — they can reject or escalate records based on rule violations.
- **AI layers** (L4.4–L4.5) have **soft power** — they can raise anomaly scores and generate summaries, but cannot override deterministic decisions.
- **Decision Gate** (L9) synthesizes all signals into a final action.

---

## Pipeline Phases

### Phase 1 — Foundation

```
Raw JSON / CSV input
        ↓
L1: Input Contract
    - Validates input structure (batch vs. single)
    - Attaches schema manifest
    - Generates batch_id and execution metadata
        ↓
L2: Input Validation
    - Flattens nested transaction JSON to flat DataFrame
    - Checks column presence and type coercion
    - Loads 1,000+ records in ~115ms
```

### Phase 2 — Feature Extraction

```
L3: Feature Extraction (35 features)
    - Numeric: txn_amount_zscore, txn_amount_percentile
    - Temporal: txn_hour, txn_is_weekend
    - Risk: fraud_risk_score, fraud_velocity_passed, fraud_geo_passed
    - Card: card_type_encoded, card_bin_category (VISA=0, MC=1, Other=2)
    - Merchant: merchant_is_domestic, merchant_country_risk
    - Settlement: settlement_fee_ratio
    - Auth: auth_result_encoded
    - Velocity: bidirectional time-diff to detect full attack sequences
    - Geo: country-change detection for impossible travel
```

### Phase 3 — Deterministic Inference

```
L4.1: Structural Integrity
    - Primary key uniqueness (txn_transaction_id)
    - Required field presence
    - Dtype consistency
    → produces: valid_indices (list of passing record indices)

L4.2: Field Compliance (7-Dimension DQS Scoring)
    - Scores each record on 7 dimensions (0–100 each)
    - Computes weighted DQS: sum(score × weight) / sum(weights)
    - Flags records below thresholds for rejection or review
    → produces: DQS DataFrame with dim_* columns

L4.3: Semantic Validation (15 Business Rules)
    - Evaluates BR001–BR015 on each record
    - Critical violations → semantic_violations list
    - Warning violations → warning_violations list
    → produces: SemanticValidation objects per record
```

### Phase 4 — AI Inference

```
L4.4: Anomaly Detection (IsolationForest)
    - Loads pre-trained model from src/resources/anomaly_model.pkl
    - Applies StandardScaler to 14 numerical features
    - Computes anomaly score in [0, 1]
    - Appends rule-based flags: EXTREME_AMOUNT, VELOCITY_FAIL, GEO_FAIL, etc.
    → produces: AnomalyResult objects per record

L4.5: GenAI Summarization (Gemini)
    - When use_ai=True: calls Gemini API for human-readable summaries
    - When use_ai=False: generates deterministic fallback summaries
    - Uses `instructor` library for validated Pydantic structured output
    - 3-pass retry loop on validation failures
    → produces: QualitySummary objects per record
```

### Phase 5 — Output & Decision

```
L5: Output Contract
    - Assembles RecordPayload dataclass for each record:
      { record_id, dqs_base, semantic_score, anomaly_score,
        is_anomaly, anomaly_flags, semantic_violations,
        structural_issues, priority, confidence_score }

L6: Stability & Consistency
    - Computes batch-level DQS distribution stats (mean, std, skewness)
    - Flags batches with unusual statistical profiles

L7: Conflict Detection
    - Detects contradictions between signals:
      - Rule OK but ML flags high anomaly
      - Low DQS but no anomaly flag
      - Dimension much lower than overall DQS

L8: Confidence Band
    - Assigns HIGH / MEDIUM / LOW confidence to each record
    - Based on: DQS stability, conflict count, consistency score

L9: Decision Gate (final action)
    - Rule 1: Critical anomaly score > 0.9 → ESCALATE unconditionally
    - Rule 2: is_anomaly + score > 0.75 + HIGH/MEDIUM confidence → ESCALATE
    - Rule 3: semantic_violations present → ESCALATE
    - Rule 4: DQS < borderline threshold → REVIEW
    - Rule 5: All clear → SAFE_TO_USE
```

### Phase 6 — Responsibility & Logging

```
L10: Responsibility Boundary
    - Tags each decision with: layer_responsible, rule_triggered, confidence
    - Produces accountability_report for regulatory audit trail

L11: Logging & Trace
    - Writes structured logs for every layer execution
    - Full execution trace per batch_id
    - PII-stripped (no raw customer data in logs)
```

---

## Data Flow Diagram

```
CSV / JSON
    │
    ▼
┌──────────┐    ┌──────────┐    ┌──────────────────┐
│ L1 Input │───▶│L2 Validate│──▶│ L3 Features (35) │
│ Contract │    │& Flatten  │    └────────┬─────────┘
└──────────┘    └──────────┘             │
                                         ▼
                              ┌──────────────────────┐
                              │  L4.1 Structural     │──── rejected_indices
                              └──────────┬───────────┘
                                         │ valid_indices
                                         ▼
                              ┌──────────────────────┐
                              │  L4.2 Field Compliance│──── DQS scores (7 dims)
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  L4.3 Semantic Rules  │──── violations (BR001-BR015)
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┤
                    │                    │
                    ▼                    ▼
          ┌─────────────────┐  ┌──────────────────────┐
          │ L4.4 Anomaly    │  │ L4.5 GenAI Summary   │
          │ (IsolationForest│  │ (Gemini / fallback)  │
          └────────┬────────┘  └──────────┬───────────┘
                   │                       │
                   └──────────┬────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   L5 RecordPayload   │
                   └──────────┬───────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                  L6 Stab   L7 Conf   (scores)
                    └────────┬─────────┘
                             ▼
                   ┌──────────────────────┐
                   │  L8 Confidence Band  │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │  L9 Decision Gate    │
                   │  SAFE / REVIEW / ESC │
                   └──────────┬───────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼                   ▼
               L10 Responsibility   L11 Logging
```

---

## Key Design Decisions

### 1. Separation of Training and Inference
The `IsolationForest` model is **pre-trained offline** via `scripts/train_anomaly_model.py` on 5,000 synthetic transactions and saved as `src/resources/anomaly_model.pkl`. At runtime, Layer 4.4 loads the model and calls `.predict()` — no fitting at inference time. This is a production MLOps pattern: O(1) startup, reproducible predictions.

### 2. Bidirectional Velocity Detection
The velocity check uses `min(time_diff_prev, time_diff_next)` rather than just backward `diff()`. This ensures the **first transaction** in a high-velocity burst is also flagged — critical for batch-mode processing where all members of an attack sequence are visible simultaneously.

### 3. Instructor + Retry Loop for LLM Output
Layer 4.5 uses the `instructor` library to enforce Pydantic model validation on Gemini responses. If validation fails, the error message is sent back to Gemini as a correction prompt (max 3 retries). This eliminates hallucinated or malformed JSON responses.

### 4. Confidence Band as a Safety Gate
The confidence band (L8) acts as a safety layer between ML anomaly scores and final decisions. A record with anomaly_score=0.7 and `LOW` confidence will be marked `SAFE_TO_USE`, while the same score with `HIGH` confidence triggers `ESCALATE`. This prevents ML noise from causing unnecessary escalations.
