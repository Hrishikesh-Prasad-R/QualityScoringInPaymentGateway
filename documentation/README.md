# Documentation Index — DQS Engine

Welcome to the complete documentation for the **Data Quality Scoring Engine for Payment Gateways**.

---

## Contents

| Document | Description |
|---|---|
| [architecture.md](architecture.md) | Full 15-layer pipeline architecture, phase breakdown, data flow diagram, and key design decisions |
| [api_reference.md](api_reference.md) | REST API endpoints, request/response schemas, WebSocket events, and security |
| [layer_reference.md](layer_reference.md) | Per-layer documentation — class names, inputs, outputs, thresholds, and failure modes |
| [semantic_rules.md](semantic_rules.md) | All 15 business rules (BR001–BR015) with check logic, rationale, and extension guide |
| [deployment.md](deployment.md) | Local dev setup, Render deployment, Docker, and production checklist |

---

## Quick Links

- **README** → [../README.md](../README.md) — Project overview, badges, key numbers
- **Benchmarks** → [../BENCHMARKS.md](../BENCHMARKS.md) — Reproducible benchmark results
- **Tests** → [../tests/](../tests/) — 95-test pytest suite
- **Evaluation** → `python scripts/run_evaluation.py` — Reproduces 100% recall result

---

## Project at a Glance

```
Input (CSV/JSON)
    ↓
15-Layer Pipeline (Rules Enforce → ML Informs → Humans Decide)
    ↓
Per-record decisions: SAFE_TO_USE / REVIEW / ESCALATE
    ↓
7-dimension DQS score + anomaly flags + audit trail
```

**Evaluation results on 1,000-record labeled dataset:**
- Precision: **100%** (0 false positives)
- Recall: **100%** (0 missed anomalies)
- F1: **100%**
- Throughput: **~5ms/record**

---

## Key Files

| File | Purpose |
|---|---|
| `src/dqs_engine.py` | Main pipeline orchestrator |
| `src/layers/layer4_3_semantic.py` | 15 business rules |
| `src/layers/layer9_decision.py` | Final SAFE/REVIEW/ESCALATE decision |
| `src/csv_adapter.py` | Universal CSV ingestion |
| `app.py` | Flask + WebSocket server |
| `scripts/run_evaluation.py` | Evaluation runner |
| `scripts/train_anomaly_model.py` | Model training |
