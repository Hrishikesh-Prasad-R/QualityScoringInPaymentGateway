# Deployment Guide

---

## Option A — All-in-one (Flask serves frontend)

The simplest setup. Flask serves both the API and the frontend from the same process.
No configuration needed — works out of the box locally and on Render.

```bash
python app.py          # local
# or on Render:
gunicorn --worker-class eventlet -w 1 app:app
```

---

## Option B — Split Deploy: Vercel (frontend) + Render (backend)

Use this when you want the frontend on Vercel's global CDN and the backend on Render.

### Overview

```
Browser → Vercel (static HTML/JS/CSS)
              ↓
         API calls to
              ↓
         Render (Flask backend)
```

### Step 1 — Deploy backend to Render

1. Go to [render.com](https://render.com) → **New Web Service** → connect your GitHub repo
2. Render will auto-detect `render.yaml` — confirm these settings:
   - **Build command:** `pip install -r requirements.txt && python scripts/train_anomaly_model.py`
   - **Start command:** `gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app`
3. In **Environment Variables**, set:
   | Key | Value |
   |---|---|
   | `GEMINI_API_KEY` | your Gemini key (optional) |
   | `ALLOWED_ORIGINS` | *(leave blank for now — fill in after step 3)* |
4. Click **Deploy** and wait for it to go live
5. Note your Render URL: `https://dqs-engine-xxxx.onrender.com`

### Step 2 — Configure frontend with your Render URL

Edit `frontend/index.html` — find this block (near line 16):

```html
<script>
  window.DQS_CONFIG = {
    apiBase: ''   // ← Replace with your Render URL when deploying to Vercel
  };
</script>
```

Change it to:
```html
<script>
  window.DQS_CONFIG = {
    apiBase: 'https://dqs-engine-xxxx.onrender.com'
  };
</script>
```

Commit and push this change.

### Step 3 — Deploy frontend to Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project** → import your GitHub repo
2. Vercel will detect `vercel.json` and configure automatically:
   - **Output directory:** `frontend`
   - **Framework:** Other (static)
3. Click **Deploy**
4. Note your Vercel URL: `https://dqs-engine.vercel.app`

### Step 4 — Set CORS on Render

Go back to your **Render dashboard** → Environment Variables → set:

```
ALLOWED_ORIGINS = https://dqs-engine.vercel.app,https://dqs-engine-git-main-yourname.vercel.app
```

> [!IMPORTANT]
> Include ALL your Vercel preview URLs (not just production) to avoid CORS errors during PR previews.
> Get the full list from Vercel dashboard → Deployments.

Click **Save** — Render will redeploy automatically.

### Step 5 — Verify

```bash
# Backend health
curl https://dqs-engine-xxxx.onrender.com/api/health
# → { "status": "healthy" }

# Frontend
open https://dqs-engine.vercel.app
# → Should show "Connected" in top-right status indicator
```

---

## Troubleshooting Split Deployment

| Problem | Cause | Fix |
|---|---|---|
| "Disconnected" in UI | CORS blocked | Check `ALLOWED_ORIGINS` includes your exact Vercel URL |
| Network error on `/api/run` | Wrong `apiBase` | Check `window.DQS_CONFIG.apiBase` in browser console |
| WebSocket connection fails | Eventlet / CORS | Ensure `async_mode='eventlet'` and Vercel URL is in `ALLOWED_ORIGINS` |
| 502 on Render | Gunicorn crash | Check Render logs — usually missing model pkl or missing env var |
| Model file not found | Build skipped training | Ensure build command includes `python scripts/train_anomaly_model.py` |

---

## Local Development

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Train model (one-time)
python scripts/train_anomaly_model.py

# 3. Run (serves frontend at http://localhost:5000)
python app.py

# 4. (Optional) Gemini AI
set GEMINI_API_KEY=your_key
```

---

## Docker

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

---

## Production Checklist

- [ ] Anomaly model trained (`src/resources/anomaly_model.pkl` committed)
- [ ] `pytest tests/` → 95/95 pass
- [ ] `python scripts/run_evaluation.py` → 100% recall confirmed
- [ ] Backend `/api/health` returns `{ "status": "healthy" }`
- [ ] `ALLOWED_ORIGINS` set in Render with Vercel URL
- [ ] `window.DQS_CONFIG.apiBase` set in `frontend/index.html` with Render URL
- [ ] Frontend status indicator shows **"Connected"**
- [ ] `GEMINI_API_KEY` set (optional — AI features gracefully degrade without it)
- [ ] `SECRET_KEY` set to a strong random value in Render
