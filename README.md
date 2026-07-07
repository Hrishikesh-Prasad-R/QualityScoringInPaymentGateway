# DQS Engine — Data Quality Scoring for Payment Gateways

> **"Rules Enforce. ML Informs. Humans Decide."**
>
> A production-grade, 15-layer data quality pipeline that scores VISA payment transactions across 7 compliance dimensions, detects anomalies, and produces deterministic, auditable decisions — all in under 5ms per record.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-IsolationForest-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Gemini-GenAI-purple?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Flask-WebSocket-green?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/Tests-95%2F95%20Passed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Recall-100%25-red?style=for-the-badge" />
</p>

---

## 🏆 Key Numbers

| Metric | Value |
|--------|-------|
| **Anomaly Recall** | **100%** (0 missed anomalies on 1,000-record test set) |
| **Anomaly Precision** | **100%** (0 false positives on evaluation dataset) |
| **Pipeline Throughput** | **~5ms / record** (batch=500) |
| **Pipeline Layers** | **15 layers** across 6 phases |
| **DQS Dimensions** | **7** (completeness, accuracy, validity, uniqueness, consistency, timeliness, integrity) |
| **Semantic Rules** | **15 business rules** (BR001–BR015) |
| **Test Coverage** | **95/95 tests passing** |
| **Anomaly Profiles Detected** | **5/5** (velocity, extreme amount, BIN mismatch, geo-mismatch, malformed fields) |

---

## 🎯 What It Does

The DQS Engine takes raw payment transaction data (from CSV upload or JSON API), runs it through 15 sequential validation and scoring layers, and outputs a quality-assessed, anomaly-flagged, auditable decision for every record:

- **`SAFE_TO_USE`** — passes all quality checks
- **`ESCALATE`** — critical violation or confirmed anomaly detected
- **`REVIEW`** — borderline quality, needs human review

### Anomaly Profiles Detected (100% Recall on all 5)

| Profile | Detection Method |
|---|---|
| Velocity Anomaly (card used 5× in 60s) | Bidirectional time-diff feature + BR010 semantic rule |
| Extreme Amount (₹4.5L on ₹8K median) | BR015 absolute threshold rule |
| BIN–Network Mismatch (VISA BIN on Mastercard) | BR013 semantic rule |
| Geo-Mismatch / Impossible Travel | BR008 + `fraud_geo_passed` feature |
| Malformed Fields (invalid MCC, email) | BR014 regex format rule |

---

## 🏗️ Architecture — 15-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Foundation                          │
│  L1 Input Contract → L2 Input Validation                        │
├─────────────────────────────────────────────────────────────────┤
│                 PHASE 2: Feature Extraction                     │
│  L3 Feature Extraction (35 features)                            │
├─────────────────────────────────────────────────────────────────┤
│              PHASE 3: Deterministic Inference                   │
│  L4.1 Structural Integrity                                      │
│  L4.2 Field Compliance (7-dimension DQS scoring)                │
│  L4.3 Semantic Validation (15 business rules)  ← ENFORCE        │
├─────────────────────────────────────────────────────────────────┤
│                  PHASE 4: AI Inference                          │
│  L4.4 Anomaly Detection (IsolationForest)       ← INFORM        │
│  L4.5 GenAI Summarization (Gemini)              ← EXPLAIN       │
├─────────────────────────────────────────────────────────────────┤
│               PHASE 5: Output & Decision                        │
│  L5 Output Contract → L6 Stability → L7 Conflict               │
│  L8 Confidence Band → L9 Decision Gate                          │
├─────────────────────────────────────────────────────────────────┤
│             PHASE 6: Responsibility & Logging                   │
│  L10 Responsibility Boundary → L11 Logging & Trace             │
└─────────────────────────────────────────────────────────────────┘
```

**Design principle:** Deterministic layers (L4.1–L4.3) can **reject or escalate** records directly. AI layers (L4.4–L4.5) can only **inform** — they never override deterministic decisions. This ensures full auditability and reproducibility.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Core Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Anomaly Detection | Scikit-learn IsolationForest |
| GenAI Summarization | Google Gemini API |
| Structured LLM Output | `instructor` + Pydantic |
| Web Framework | Flask + Flask-SocketIO |
| Real-time Streaming | WebSocket (eventlet) |
| Security | HMAC-SHA256 webhook signatures |
| Testing | pytest (95 tests) |
| Deployment | Render (Procfile + render.yaml) |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- `pip`
- (Optional) Google Gemini API key for AI features

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Hrishikesh-Prasad-R/QualityScoringInPaymentGateway.git
cd QualityScoringInPaymentGateway

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the anomaly model (one-time)
python scripts/train_anomaly_model.py

# 5. (Optional) Set Gemini API key
set GEMINI_API_KEY=your_api_key_here    # Windows
# export GEMINI_API_KEY=your_api_key_here  # macOS/Linux
```

---

## ⚡ Quick Start

### Option A — Python API

```python
from src.dqs_engine import DQSEngine
from src.data_generator import generate_visa_transactions

# Generate sample transactions
transactions = generate_visa_transactions(n_transactions=100)

# Run the pipeline (AI optional)
engine = DQSEngine(use_ai=False)   # use_ai=True requires GEMINI_API_KEY
result = engine.run(transactions)

# Inspect decisions
for decision in engine.decisions:
    print(f"{decision.record_id}: {decision.action.value} — {decision.primary_reason}")
```

### Option B — CSV Upload (Web UI)

```bash
# Start the Flask server
python app.py

# Open browser at http://localhost:5000
# Upload any CSV with payment transaction data
```

### Option C — REST API

```bash
# Health check
curl http://localhost:5000/api/health

# Run pipeline (JSON body)
curl -X POST http://localhost:5000/api/run \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

---

## 📊 DQS Scoring — 7 Dimensions

Each transaction receives a **Data Quality Score (DQS)** from 0–100, computed as a weighted average:

| Dimension | Weight | What It Checks |
|---|---|---|
| Completeness | 25% | Required fields present and non-null |
| Accuracy | 20% | Field formats correct (TXN ID, BIN, email, MCC) |
| Validity | 20% | Values within allowed ranges and enumerations |
| Uniqueness | 10% | No duplicate transaction IDs |
| Consistency | 15% | Cross-field logic (settlement math, date ordering) |
| Timeliness | 5% | Timestamps not stale, clearing before settlement |
| Integrity | 5% | Referential integrity (merchant ID ↔ name, etc.) |

**Decision thresholds:**
- DQS ≥ 85 + no violations → `SAFE_TO_USE`
- DQS 75–85 → `REVIEW`
- DQS < 75 or any critical violation → `ESCALATE`

---

## 🔍 Semantic Rules (BR001–BR015)

| Rule | Description | Severity |
|---|---|---|
| BR001 | Amount must be positive | Critical |
| BR002 | Net = Gross − Fees (settlement math) | Critical |
| BR003 | Settlement date after clearing date | Critical |
| BR004 | Approved transactions must have auth code | Critical |
| BR005 | Card must not be expired | Critical |
| BR006 | Amount rational for merchant category | Warning |
| BR007 | Risk score matches risk level label | Warning |
| BR008 | Geo consistency (IP ↔ merchant country) | Critical |
| BR009 | 3DS required for high-value (>₹10K) | Warning |
| BR010 | Failed velocity check → elevated risk score | Critical |
| BR011 | Fee ratio < 5% | Warning |
| BR012 | Billing/shipping country matches merchant | Warning |
| BR013 | BIN prefix matches card network brand | Critical |
| BR014 | Email, MCC, TXN ID format validation | Critical |
| BR015 | Transaction amount < ₹1,00,000 threshold | Critical |

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific phase
pytest tests/test_phase3.py -v    # Semantic validation

# Run evaluation (reproduces 100% recall result)
python scripts/run_evaluation.py

# Benchmarks
python scripts/benchmark_anomaly.py      # Anomaly detector metrics
python scripts/benchmark_throughput.py   # Throughput ms/record
```

---

## 📁 Project Structure

```
QualityScoringInPaymentGateway/
├── app.py                          # Flask + SocketIO web server
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
├── render.yaml                     # Render service config
│
├── src/
│   ├── dqs_engine.py               # Main pipeline orchestrator
│   ├── csv_adapter.py              # Universal CSV ingestion (AI-powered mapping)
│   ├── data_generator.py           # Synthetic VISA transaction generator
│   ├── database.py                 # SQLite metadata persistence (DQSDatabase)
│   ├── security.py                 # HMAC-SHA256 webhook security
│   ├── config.py                   # Thresholds, weights, enums
│   ├── sample_csv_generator.py     # High/medium/low quality CSV generator
│   ├── resources/
│   │   ├── anomaly_model.pkl       # Pre-trained IsolationForest
│   │   └── anomaly_scaler.pkl      # StandardScaler for feature normalization
│   ├── models/
│   │   └── schema.py               # VISA transaction dataclass schema
│   └── layers/
│       ├── layer1_input_contract.py
│       ├── layer1_2_schema_mapper.py    # GenAI CSV column mapper
│       ├── layer2_input_validation.py
│       ├── layer3_feature_extraction.py # 35-feature engineering
│       ├── layer4_1_structural.py
│       ├── layer4_2_field_compliance.py # 7-dimension DQS scoring
│       ├── layer4_3_semantic.py         # 15 business rules (BR001-BR015)
│       ├── layer4_4_anomaly.py          # IsolationForest anomaly detection
│       ├── layer4_5_summarization.py    # Gemini AI summaries + corrections
│       ├── layer5_output_contract.py
│       ├── layer6_stability.py
│       ├── layer7_conflict.py
│       ├── layer8_confidence.py
│       ├── layer9_decision.py           # Final SAFE/REVIEW/ESCALATE gate
│       ├── layer10_responsibility.py
│       └── layer11_logging.py
│
├── scripts/
│   ├── run_evaluation.py            # 100% recall evaluation runner
│   ├── train_anomaly_model.py       # Offline model training (MLOps)
│   ├── benchmark_anomaly.py         # Anomaly detector metrics
│   ├── benchmark_throughput.py      # Pipeline throughput benchmark
│   └── generate_evaluation_dataset.py # Labeled test dataset generator
│
├── tests/                           # 95-test pytest suite
│   ├── test_phase1.py through test_phase7.py
│   ├── test_database.py
│   ├── test_security.py
│   └── test_instructor.py
│
├── data/
│   └── evaluation_visa_dataset.csv  # 1,000-record labeled test set
│
├── frontend/
│   ├── index.html                   # Single-page UI
│   ├── script.js                    # WebSocket + API client
│   └── styles.css                   # UI styles
│
├── documentation/                   # Full project documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── layer_reference.md
│   ├── semantic_rules.md
│   └── deployment.md
│
└── BENCHMARKS.md                    # Reproducible benchmark results
```

---

## 🔐 Security

- **HMAC-SHA256** webhook signature verification on all API endpoints
- **PII masking** — emails and PAN tokens masked before Layer 3 processing
- **Metadata-only persistence** — SQLite stores only DQS scores and decisions, never raw transaction values
- **Stripped audit logs** — no customer data in log entries

---

## 🚀 Deployment

### Render (Free Tier)

1. Fork this repo and connect to [Render](https://render.com)
2. Set environment variables: `GEMINI_API_KEY`, `PORT=5000`
3. Build command: `pip install -r requirements.txt && python scripts/train_anomaly_model.py`
4. Start command: `gunicorn --worker-class eventlet -w 1 app:app`

See [`documentation/deployment.md`](documentation/deployment.md) for full instructions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🏫 Context

Built for the **IITM × VISA AI Hackathon** (Payment Gateway Quality Scoring track).  
Problem Statement: *Design an AI-powered data quality scoring engine for payment gateway transactions.*
