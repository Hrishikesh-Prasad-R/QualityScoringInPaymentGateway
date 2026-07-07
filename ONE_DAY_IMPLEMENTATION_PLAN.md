# Post-Hackathon Implementation Plan: Interview & Resume Edition

> **Context shift:** Hackathon is over. Goal is now to make this project genuinely impressive
> in a technical interview — not to satisfy a PS rubric. Tasks are ordered by resume ROI.
> Drop anything that doesn't give you a number, a story, or a live demo.

---

## 🔴 CRITICAL — Do These First (Gets You Numbers on Resume)

---

### 1. Benchmark the Anomaly Detector (2h)
* **Goal:** Get a real F1/recall number you can say in an interview.
* **Actionable Steps:**
  1. Open `src/data_generator.py`. Confirm `generate_visa_transactions()` returns transactions where each dict has `_metadata.is_anomaly` as a boolean label.
  2. Create `scripts/benchmark_anomaly.py`. Inside it:
     - Call `generate_visa_transactions(n_transactions=1000, anomaly_rate=0.20, random_seed=42)`.
     - Run the full DQS engine on these 1000 records: `engine.run(transactions)`.
     - Extract the ground truth labels: `y_true = [1 if t['_metadata']['is_anomaly'] else 0 for t in transactions]`.
     - Extract predicted labels from Layer 4.4 anomaly results: access `engine.layer_results[4.4].details` → get `records_flagged` indices → build `y_pred` array (1 if index in flagged set, else 0).
     - Compute and print: `sklearn.metrics.classification_report(y_true, y_pred)` and `f1_score(y_true, y_pred)`.
  3. Run: `.venv\Scripts\python scripts/benchmark_anomaly.py`. Record the F1, precision, recall numbers.
  4. Write this into a `BENCHMARKS.md` file at the project root.
  5. **Resume bullet:** *"IsolationForest anomaly detector achieved [X]% recall at [Y]% false positive rate across 1,000 labeled VISA transactions"*

---

### 2. Measure Pipeline Throughput (1h)
* **Goal:** Get a ms/record number you can quote in interviews.
* **Actionable Steps:**
  1. Create `scripts/benchmark_throughput.py`. Inside it:
     - Run the full engine on batch sizes: `[10, 50, 100, 200, 500]`.
     - For each batch size, time `engine.run(transactions)` using `time.perf_counter()` — run 3 times and take the median.
     - Print: batch size, total ms, ms/record.
  2. Run it: `.venv\Scripts\python scripts/benchmark_throughput.py`. Record the numbers.
  3. Append results to `BENCHMARKS.md`.
  4. **Resume bullet:** *"15-layer pipeline processes 500 transactions in under [X]ms (avg [Y]ms/record end-to-end)"*

---

### 3. Rewrite Resume Bullets (1h)
* **Goal:** Stop describing what the project IS — describe what it DOES and MEASURES.
* **Actionable Steps:**
  1. After completing Tasks 1 and 2, open your resume.
  2. Replace every descriptive sentence with an impact sentence using this formula:
     `[Action verb] + [what] + [scale/number] + [outcome/result]`
  3. Use these templates, filling in your actual benchmark numbers:
     - *"Engineered a 15-layer GenAI-augmented data quality pipeline scoring VISA transactions across 7 compliance dimensions (completeness, accuracy, validity, uniqueness, consistency, timeliness, integrity)"*
     - *"IsolationForest anomaly detector achieved [F1]% F1-score across 1,000 labeled transactions; pipeline processes 500 records in [X]ms ([Y]ms/record)"*
     - *"GenAI layer (Gemini) auto-corrects malformed fields using contextual inference and generates regulatory-context explanations mapped to PCI-DSS, RBI, and AML frameworks"*
     - *"Implemented 3-layer privacy stack: PII masking at ingestion, metadata-only SQLite persistence, and stripped transaction logs — zero raw customer data stored"*
     - *"Deployed as a live web application with real-time WebSocket streaming at [render URL]"*

---

## 🟠 HIGH VALUE — Do These Second (Gets You Stories + Live Demo)

---

### 4. Deploy to Render (3h)
* **Goal:** Get a live URL. A working URL in an interview beats any code explanation.
* **⚠️ Known Risk:** `app.py` uses `eventlet.monkey_patch()` at the top. This can behave differently on Render's Linux environment vs Windows dev. If the deploy crashes, check Render logs — if you see a `greenlet` or `eventlet` error, change `async_mode='threading'` in the `SocketIO()` init line in `app.py` to `async_mode='eventlet'` and redeploy.
* **Actionable Steps:**
  1. The repo already has `render.yaml` and `Procfile` configured. Confirm `Procfile` contains: `web: python app.py`.
  2. Go to [render.com](https://render.com), create a free account, click "New Web Service", connect your GitHub repo.
  3. Set environment variables in Render dashboard: `GEMINI_API_KEY=<your key>`, `PORT=5000`.
  4. Set build command: `pip install -r requirements.txt` (let Render auto-detect Python — do NOT use `.venv/` path on Render).
  5. Deploy. Wait for build. Confirm `https://[your-app].onrender.com/api/health` returns `{"status": "healthy"}`.
  6. If health check fails, check Render logs. Most common fix: add `gunicorn` to `requirements.txt` and change `Procfile` to `web: gunicorn --worker-class eventlet -w 1 app:app`.
  7. Add the live URL to your `README.md` at the top as a badge or link.
  8. **Resume line:** *"Live demo: [url]"*

---

### 5. Make Gemini Auto-Correct Failing Fields (5h)
* **Goal:** Make the GenAI component do something the deterministic layers cannot — this is your "agentic" story.
* **⚠️ Known Risk:** Gemini often wraps JSON in markdown code fences even when told not to. After calling `response.text`, always parse with: `raw = response.text.strip().strip('```json').strip('```').strip()` before `json.loads(raw)`. Wrap in a try/except and return `[]` on failure.
* **Actionable Steps:**
  1. Open `src/layers/layer4_5_summarization.py`. Add a new method `_generate_field_corrections(self, record_id, row, failing_issues)` below `_generate_ai_summary()`.
  2. Inside `_generate_field_corrections()`, build a Gemini prompt:
     ```
     You are a payments data repair agent. Given these field-level data quality failures,
     suggest the correct value for each failing field based on context from other fields.

     Record context:
     - Merchant Name: {merchant_name}
     - Transaction Amount: {amount}
     - Card Network: {card_network}
     - Country: {country}

     Failing fields:
     {failing_issues_list}

     For each failing field, return a JSON object: {"field_name": "corrected_value", "confidence": 0-100, "reasoning": "..."}.
     Return only a JSON array. No markdown. No code fences.
     ```
  3. Parse the response safely: `raw = response.text.strip().strip('```json').strip('```').strip()` then `corrections = json.loads(raw)`. Wrap in try/except returning `[]` on any parse failure.
  4. Call this method only when `use_ai=True` and `priority in ["critical", "high"]` and `len(key_issues) > 0`.
  5. Attach the returned corrections as `"suggested_corrections"` inside the `QualitySummary.context` dict.
  6. In `app.py`, expose `suggested_corrections` in the API response alongside the existing summary fields.
  7. **Interview story:** *"The GenAI component infers correct field values from transaction context — e.g., if merchant_category_code is malformed but merchant_name is 'Swiggy', Gemini returns '5812' (restaurant) with 91% confidence"*

---

### 6. Offline IsolationForest — MLOps Pattern (3h)
* **Goal:** Separate training from inference. This is a real MLOps pattern interviewers ask about.
* **Actionable Steps:**
  1. Create `scripts/train_anomaly_model.py`. Inside it:
     - Generate 5,000 transactions: `generate_visa_transactions(5000, anomaly_rate=0.15, random_seed=42)`.
     - Run through Layers 1–3 to get `features_df`.
     - Fit `StandardScaler` and `IsolationForest(contamination=0.05, random_state=42)` on `features_df`.
     - Create `src/resources/` directory.
     - Serialize both: `joblib.dump(scaler, 'src/resources/anomaly_scaler.pkl')` and `joblib.dump(model, 'src/resources/anomaly_model.pkl')`.
     - Print: training set size, contamination rate, number of features used.
  2. Run the script once: `.venv\Scripts\python scripts/train_anomaly_model.py`.
  3. Open `src/layers/layer4_4_anomaly.py`. In `__init__()`, add:
     ```python
     import joblib, os
     model_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'anomaly_model.pkl')
     scaler_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'anomaly_scaler.pkl')
     if os.path.exists(model_path) and os.path.exists(scaler_path):
         self.model = joblib.load(model_path)
         self.scaler = joblib.load(scaler_path)
         self.pretrained = True
     else:
         self.pretrained = False
     ```
  4. In the `detect()` method, add a check: if `self.pretrained`, skip the `model.fit()` call and go straight to `model.predict()`.
  5. **Interview story:** *"I separated offline training from online inference — startup time dropped from O(n) fitting to O(1) model load, making it production-deployable"*

---

### 7. GenAI Column Mapper — Universal Ingestion (3-4h)
* **Goal:** Handle any CSV regardless of headers. This solves the "universal" part of the PS and is a good GenAI story.
* **⚠️ Known Risk:** Same as Task 5 — Gemini wraps JSON in markdown. Use the same safe parse pattern: `raw = response.text.strip().strip('```json').strip('```').strip()` before `json.loads(raw)`. Also, Gemini sometimes returns field names with different casing — normalize both sides with `.lower().strip()` before doing the column rename.
* **Actionable Steps:**
  1. Create `src/layers/layer1_2_schema_mapper.py` with class `GenAISchemaMapper`.
  2. Implement one method: `map_columns(self, raw_headers: list, sample_rows: list) -> dict`.
  3. Inside it, build a Gemini prompt:
     ```
     You are a data schema mapping expert for payment systems.
     Map these raw CSV column headers to the standard VISA transaction schema fields.

     Raw headers: {raw_headers}
     Sample values (first 3 rows): {sample_rows}

     Standard fields to map to: transaction_id, amount, currency, timestamp, status,
     card_network, card_type, merchant_id, merchant_category_code, country,
     customer_id, customer_email, risk_score

     Return a JSON object: {"raw_header": "standard_field"} for each mapping.
     If a raw header has no match, map it to null.
     Return only valid JSON. No markdown. No code fences.
     ```
  4. Parse safely: `raw = response.text.strip().strip('```json').strip('```').strip()` then `mapping = json.loads(raw)`. Normalize keys: `{k.lower().strip(): v for k, v in mapping.items()}`.
  5. In `src/csv_adapter.py`, import `GenAISchemaMapper`. After loading the CSV into a DataFrame, if schema compliance < 80%, call `mapper.map_columns(df.columns.tolist(), df.head(3).to_dict('records'))` and rename the DataFrame columns using the returned mapping.
  6. **Interview story:** *"Built a GenAI-powered universal ingestion layer — upload any CSV with any headers and Gemini maps them to the canonical schema, so the engine works on Stripe, Razorpay, or any payment provider's export"*

---

## 🟡 MEDIUM VALUE — Do If You Have Time

---

### 8. Prove DQS Stratification Using Sample CSV Generator (30min)
* **Goal:** Eliminate "simulated data only" weakness without touching a broken external dataset. Prove the scoring is meaningful by showing it correctly separates high/low quality data.
* **⚠️ Why not Kaggle Credit Card Fraud:** That dataset has columns `V1, V2, ... V28` which are PCA-transformed anonymous features. They cannot be meaningfully mapped to `merchant_id`, `card_network`, etc. Running DQS on them produces garbage scores. Don't use it.
* **Actionable Steps:**
  1. Open `src/sample_csv_generator.py` — it already has `generate_high_quality_csv()`, `generate_medium_quality_csv()`, and `generate_low_quality_csv()`. These are already wired to produce intentionally degraded data.
  2. Create `scripts/benchmark_stratification.py`. Inside it:
     - Call `generate_sample_csvs()` to produce all three quality CSVs into a temp directory.
     - Load each CSV, run it through `adapt_csv_to_visa()` then `engine.run()`.
     - Print the average DQS for each: high quality, medium quality, low quality.
  3. Run it: `.venv\Scripts\python scripts/benchmark_stratification.py`. You should see: high ≥ 85, medium 55–75, low ≤ 45.
  4. Append the 3 numbers to `BENCHMARKS.md`.
  5. **Resume bullet:** *"DQS scoring correctly stratifies data quality: high-quality datasets score ≥85, degraded datasets score ≤45 — demonstrating >40-point separation across 7 dimensions"*

---

### 9. Privacy Stack (Compliance Story) (2h)
* **Goal:** Group Tasks 1+4+5 from the original plan into one coherent "compliance architecture" you can explain in 60 seconds.
* **Actionable Steps:**
  1. **Task 1 — Strip raw data from logs:** Open `src/live_data_generator.py`, `add_log()` method (~line 486). Remove `"full_transaction": transaction` and `"full_result": result` from the log entry dict. Keep only: `timestamp`, `transaction_id`, `dqs_score`, `action`, `processing_time_ms`, `flags`.
  2. **Task 4 — PII masking:** Create `src/layers/layer2_2_pii_masking.py`. Add two functions:
     - `mask_email(email: str) -> str`: keep first 2 chars + domain → `jo**@gmail.com`
     - `mask_pan(pan_token: str) -> str`: return `"tok_****" + pan_token[-4:]`
     - In `src/dqs_engine.py` `run()` method, after Layer 2 loads the DataFrame, call these on `customer_email` and `card_pan_token` columns before passing to Layer 3.
  3. **Task 5 — SQLite metadata DB:** Create `src/database.py`. Define two SQLAlchemy tables: `DQSRun` (run_id, timestamp, total_records, avg_dqs, quality_rate, duration_ms) and `DQSRecord` (record_id, run_id, dqs_score, action, dim_completeness, dim_accuracy, dim_validity, dim_uniqueness, dim_consistency, dim_timeliness, dim_integrity). No raw transaction values in any column. In `app.py` `run_pipeline()`, after building the response, write one `DQSRun` row and one `DQSRecord` row per record to SQLite.
  4. **Interview story:** *"Built a 3-layer privacy stack: PII masking at ingestion before any processing, metadata-only SQLite persistence (zero raw transaction values stored), and stripped audit logs — fully compliant with the PS governance requirement"*

---

### 10. Adaptive Dimension Detection (1h)
* **Goal:** Skip dimensions whose required columns are missing instead of returning misleading 100% scores.
* **Actionable Steps:**
  1. Open `src/layers/layer4_2_field_compliance.py`. Add method `_check_applicability(self, df_columns: list) -> dict` returning `{"completeness": True/False, ...}`.
  2. Define column requirements:
     ```python
     REQUIRED = {
         "completeness": ["txn_transaction_id", "txn_amount", "txn_status"],
         "accuracy":     ["txn_transaction_id", "card_bin", "merchant_merchant_category_code"],
         "validity":     ["txn_amount", "txn_status", "card_network"],
         "uniqueness":   ["txn_transaction_id"],
         "consistency":  ["settlement_gross_amount", "settlement_net_amount"],
         "timeliness":   ["txn_timestamp", "settlement_clearing_date"],
         "integrity":    ["customer_customer_id", "merchant_merchant_id"],
     }
     ```
  3. In `score()`, call `_check_applicability(df.columns.tolist())` before the record loop. Skip scoring calls for `False` dimensions. In `_calculate_dqs()`, skip `None` scores.
  4. Add `"applicable_dimensions"` to the result `details` dict.
  5. **Interview story:** *"The scoring engine self-configures — it inspects the uploaded dataset's schema and only scores the dimensions for which sufficient columns exist, preventing misleading 100% scores on unevaluated dimensions"*

---

## ❌ DROPPED TASKS (Original Plan Tasks 6, 7, 8, 9, 13, 14)

These were written for PS rubric compliance. They have near-zero interview value:
- Task 6 (dual-audience UI tabs) — UI polish, not a technical story
- Task 7 (auto-cleaner button) — trivial, not worth explaining
- Task 8 (dimension score UI table) — backend data already exists; the frontend work isn't the point
- Task 9 (hover tooltips) — frontend fluff
- Task 13 (improvement pathways UI panel) — more UI fluff
- Task 14 (API key middleware) — 10 lines of Flask code, unimpressive

---

## Interview Prep — Stories to Know Cold

Prepare a 90-second answer for each of these. You will be asked at least 2 of them:

1. **"Walk me through the architecture"**
   → 15 layers, 6 phases. Key design principle: *Rules > AI* — deterministic layers can reject; AI layer can only inform. Explain why: auditability, reproducibility, no hallucination-driven rejections.

2. **"What was the hardest engineering problem?"**
   → Thread-safe live streaming: multiple clients, one background worker, shared state. Used `threading.Lock()` on both the engine singleton and the streaming state. Explain the race condition you avoided.

3. **"Why IsolationForest? Why not [other model]?"**
   → Unsupervised — no labeled fraud data needed at runtime. Handles high-dimensional data (35+ features). Contamination param maps directly to expected anomaly rate. Offline training separates fit from predict.

4. **"What would you do differently at scale?"**
   → Replace Flask + SQLite with FastAPI + PostgreSQL. Replace in-process streaming with Kafka. Add model drift detection — retrain when anomaly rate deviates from baseline by >2σ. Add a proper feature store.

5. **"What's the accuracy of your anomaly detector?"**
   → Answer with your actual numbers from Task 1. If you don't have numbers, the project is hollow.
