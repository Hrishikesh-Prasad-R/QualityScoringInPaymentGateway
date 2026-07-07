# Resume Bullets — DQS Engine Project

> Copy these directly into your resume/CV.
> Numbers sourced from BENCHMARKS.md (Tasks 1 & 2).
> Update the [RENDER_URL] placeholder after Task 4 (deploy).

---

## Option A — Single Comprehensive Bullet (Recommended for 1-page resume)

**Engineered a GenAI-augmented data quality scoring engine** for VISA payment transactions; 15-layer pipeline scores data across 7 compliance dimensions (completeness, accuracy, validity, uniqueness, consistency, timeliness, integrity), processes 500 records in 1.6s (3.2ms/record), and achieves 100% anomaly recall at 12% FPR (F1=81%) using an ensemble of IsolationForest, z-score analysis, and rule-based flags.

---

## Option B — Expanded 4-Bullet Version (For detailed project section)

- **Architected a 15-layer, 6-phase DQS pipeline** following a "Rules enforce, AI informs" design principle — deterministic layers can reject transactions; the GenAI layer (Gemini) can only summarize and recommend, preventing hallucination-driven false rejections.

- **Anomaly detection:** IsolationForest ensemble (IsolationForest + z-score + rule flags) achieved **100% recall at 12% false positive rate** across 1,000 labeled VISA transactions (F1=81%, accuracy=90.7%); model trained offline with joblib serialization for O(1) startup vs O(n) online fitting.

- **Performance:** Full pipeline (15 layers, 35 features, 7 quality dimensions) processes **500 transactions in 1.6 seconds (3.2ms/record)**; per-record cost drops from 18ms at batch=10 to 3.2ms at batch=500 due to amortized sklearn/scaler overhead.

- **GenAI integration:** Gemini API generates regulatory-context explanations (PCI-DSS, RBI, AML) and infers corrected values for malformed fields from transaction context (e.g., maps merchant name to ISO MCC code); fallback template ensures 100% uptime without API key; deployed live at **[RENDER_URL]**.

---

## Option C — Single-Line Versions (For skills/experience tables)

**Short:**
> DQS Engine — 15-layer VISA payment data quality pipeline; 100% anomaly recall, 3.2ms/record, GenAI-powered field correction and regulatory explainability. [Live: RENDER_URL]

**Very short (for a list):**
> DQS Engine (Python, Flask, Gemini, IsolationForest) — 100% recall, 3.2ms/record, live demo

---

## Interview Talking Points (Not for resume, for verbal use)

**On architecture:**
> "I built a 15-layer pipeline split into 6 phases. The key design decision was keeping deterministic layers and AI layers separate — deterministic rules can hard-reject a transaction, the AI can only flag or explain. This means no hallucination can cause a false rejection."

**On performance numbers:**
> "The pipeline processes 500 transactions in 1.6 seconds — that's 3.2ms per record end-to-end including all 15 layers. The per-record cost drops from 18ms at small batches to 3.2ms at 500 because sklearn and scaler initialization is a fixed cost that gets amortized."

**On the anomaly detector:**
> "IsolationForest hit 100% recall — it caught every single labeled anomaly. The trade-off is 12% false positive rate, so it's tuned for high sensitivity which is correct for a fraud-adjacent use case. I've since moved the model to offline training with joblib so it loads in milliseconds instead of fitting on the batch."

**On GenAI:**
> "Gemini does two things: it generates a regulatory-context explanation for each failing dimension — for example, it links low completeness in address fields to KYC compliance under RBI guidelines. And it infers corrected values for malformed fields — if merchant_category_code is invalid but merchant_name is 'Swiggy', it returns the correct ISO MCC code with a confidence score."

**On what you'd do differently:**
> "At scale I'd replace Flask with FastAPI for async support, SQLite with PostgreSQL, and add a Kafka topic for the live transaction stream. I'd also add model drift detection — alert when the anomaly rate deviates more than 2 standard deviations from the baseline, which triggers a retrain."
