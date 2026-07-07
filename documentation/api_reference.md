# API Reference — DQS Engine

Base URL: `http://localhost:5000`

---

## `GET /api/health`

```json
{ "status": "healthy", "timestamp": "2026-07-08T00:00:00Z" }
```

---

## `POST /api/run` — JSON

Run the full pipeline on a batch of transactions.

**Minimal request body:**
```json
{
  "transactions": [
    {
      "transaction": { "transaction_id": "TXN_001", "amount": 4500, "currency": "INR",
                       "timestamp": "2026-07-08T10:30:00Z", "status": "approved",
                       "type": "payment", "authorization_code": "AUTH123" },
      "card": { "network": "VISA", "pan_token": "tok_xxxx1234", "bin": "411111",
                "last4": "1111", "expiry_month": "08", "expiry_year": "2027", "card_type": "credit" },
      "merchant": { "merchant_id": "MID_001", "merchant_name": "Example Store",
                    "merchant_category_code": "5812", "country": "IN" },
      "customer": { "customer_id": "CUST_001", "email": "user@example.com", "ip_address": "103.25.30.40" },
      "fraud": { "risk_score": 32, "risk_level": "low", "velocity_check": "pass", "geo_check": "pass" },
      "settlement": { "clearing_date": "2026-07-09", "settlement_date": "2026-07-10",
                      "gross_amount": 4500, "interchange_fee": 36, "gateway_fee": 13.5, "net_amount": 4450.5 }
    }
  ],
  "use_ai": false
}
```

**Response:**
```json
{
  "success": true,
  "batch_id": "batch_20260708_001",
  "total_records": 1,
  "quality_rate": 100.0,
  "processing_time_ms": 450.2,
  "decisions": [
    {
      "record_id": "TXN_001",
      "action": "SAFE_TO_USE",
      "primary_reason": "Record passes all quality checks",
      "dqs_score": 94.0,
      "anomaly_score": 0.12,
      "semantic_violations": [],
      "anomaly_flags": [],
      "priority": "none",
      "confidence_band": "HIGH"
    }
  ],
  "summary": { "safe": 1, "review": 0, "escalate": 0, "mean_dqs": 94.0 }
}
```

**`action` values:**

| Value | Meaning |
|---|---|
| `SAFE_TO_USE` | All checks pass |
| `REVIEW` | Borderline — human review recommended |
| `ESCALATE` | Critical violation or confirmed anomaly |

---

## `POST /api/run` — CSV Upload

Upload a CSV file via `multipart/form-data`.

| Field | Type | Required |
|---|---|---|
| `file` | File (.csv) | Yes |
| `use_ai` | boolean | No (default: false) |

**Minimum CSV columns:** `transaction_id, amount, timestamp, status, network, merchant_id, customer_id`

Non-standard column names are auto-mapped by the GenAI schema mapper (Layer 1.2).

---

## `GET /api/schema`

Returns the canonical VISA transaction schema structure.

---

## `GET /api/layers`

Returns metadata about all 15 pipeline layers (id, name, phase, type).

---

## `POST /api/generate`

Generate synthetic test transactions.

```json
{ "n_transactions": 100, "anomaly_rate": 0.05, "random_seed": 42 }
```

---

## Live Streaming (WebSocket)

| Endpoint | Description |
|---|---|
| `GET /api/live/stats` | Live session statistics |
| `GET /api/live/logs` | Recent log entries |
| `POST /api/live/set-anomaly-rate` | Set injection rate |
| `POST /api/live/set-api-key` | Update Gemini key at runtime |
| `POST /api/live/clear` | Reset session |

**WebSocket events received by client:**
- `transaction_processed` — per-record result
- `stats_update` — batch statistics
- `live_log` — real-time logs

---

## Security — HMAC-SHA256

Set `REQUIRE_HMAC=true` to enforce signature verification.

```python
import hmac, hashlib, json
secret = b"your_webhook_secret"
payload = json.dumps(body).encode()
sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
# Send in header: X-Signature: <sig>
```

Environment variables: `DQS_WEBHOOK_SECRET`, `REQUIRE_HMAC`

---

## Error Responses

```json
{ "success": false, "error": "message", "code": "ERROR_CODE" }
```

| HTTP | Meaning |
|---|---|
| 400 | Invalid request |
| 401 | Bad HMAC signature |
| 413 | File too large (>10MB) |
| 500 | Internal error |
