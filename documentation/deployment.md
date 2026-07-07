# Deployment Guide

## Local Development

```bash
# 1. Clone and install
git clone https://github.com/Hrishikesh-Prasad-R/QualityScoringInPaymentGateway.git
cd QualityScoringInPaymentGateway
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Train anomaly model (required once)
python scripts/train_anomaly_model.py

# 3. Start server
python app.py
# → http://localhost:5000
```

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | No | Enables GenAI features in Layer 4.5 |
| `DQS_WEBHOOK_SECRET` | No | HMAC key (default: built-in dev key) |
| `REQUIRE_HMAC` | No | Set `true` to enforce signature checks |
| `PORT` | No | Server port (default: 5000) |

---

## Render (Recommended — Free Tier)

### Step 1 — Prepare the repo

Ensure these files exist at root:

**`Procfile`:**
```
web: gunicorn --worker-class eventlet -w 1 app:app
```

**`render.yaml`:**
```yaml
services:
  - type: web
    name: dqs-engine
    env: python
    buildCommand: pip install -r requirements.txt && python scripts/train_anomaly_model.py
    startCommand: gunicorn --worker-class eventlet -w 1 app:app
    envVars:
      - key: GEMINI_API_KEY
        sync: false
      - key: PORT
        value: 5000
```

### Step 2 — Deploy

1. Go to [render.com](https://render.com) → New Web Service
2. Connect your GitHub repo
3. Set environment variables in Render dashboard:
   - `GEMINI_API_KEY` = your key
   - `PORT` = 5000
4. Click Deploy

### Step 3 — Verify

```bash
curl https://your-app.onrender.com/api/health
# → { "status": "healthy" }
```

### Troubleshooting Render

| Problem | Fix |
|---|---|
| `greenlet` import error | Add `eventlet>=0.35.0` to requirements.txt |
| WebSocket connection fails | Check `async_mode='eventlet'` in `SocketIO()` init |
| Model file not found | Add `python scripts/train_anomaly_model.py` to build command |
| Gemini API errors | Verify `GEMINI_API_KEY` is set in Render env vars |
| Port binding error | Ensure `PORT` env var matches Procfile |

---

## Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python scripts/train_anomaly_model.py

EXPOSE 5000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t dqs-engine .
docker run -p 5000:5000 -e GEMINI_API_KEY=your_key dqs-engine
```

---

## Production Checklist

- [ ] `GEMINI_API_KEY` set in environment
- [ ] `REQUIRE_HMAC=true` set for webhook security
- [ ] `DQS_WEBHOOK_SECRET` set to a strong random value
- [ ] Anomaly model trained (`src/resources/anomaly_model.pkl` exists)
- [ ] `python -m pytest tests/` → all 95 tests pass
- [ ] `python scripts/run_evaluation.py` → 100% recall confirmed
- [ ] Health endpoint returns `"status": "healthy"`
