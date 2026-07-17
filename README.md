# 👁 Safety Eye — AI-Powered Workplace Safety Monitor

> Automatically detects PPE violations on construction sites using computer vision. Upload a video or use your webcam — Safety Eye flags workers missing helmets, gloves, boots, or goggles in real time.

![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-FF6B35)

---

## 🚀 Quick Start — Zero Setup Required

> **Only requirement: [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your machine.**

### Step 1 — Clone the repository

```bash
git clone https://github.com/springboardmentor425/B-13-SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor.git
cd B-13-SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor
```

### Step 2 — Start everything

```bash
docker compose -f docker-compose.hub.yml up
```

Wait about 30–60 seconds for all services to start. You will see logs scrolling — that is normal.

### Step 3 — Open the app

| What | URL |
|---|---|
| 🖥 **Frontend** | http://localhost:8100 |
| 📊 **Grafana Dashboard** | http://localhost:3100 |
| 📖 **API Docs** | http://localhost:8100/docs |

> **Grafana login:** Username `Admin` · Password `MyStrongPass123`

### Step 4 — Stop everything

```bash
docker compose -f docker-compose.hub.yml down
```

---

## 📸 What It Does

| Feature | Description |
|---|---|
| 🎥 **Video Upload** | Upload MP4/AVI videos — AI scans every frame for PPE violations |
| 🌐 **YouTube Support** | Paste a YouTube URL → auto-downloads at 720p and processes it |
| 📷 **Live Webcam** | Stream from your camera and get real-time detection results |
| ⏱️ **Live Preview** | Watch detection frames update live during video processing |
| 📋 **Job Management** | View all running jobs, monitor progress, cancel anytime |
| ⚠️ **Violations Feed** | Browse, filter, and acknowledge detected safety violations |
| 📊 **Grafana Dashboard** | Rich analytics: compliance rate, violation trends, occupancy heatmaps |
| 🔌 **REST API** | Full API with Swagger docs at `/docs` |

---

## 🧠 What the AI Detects

The model was trained on the **construction-ppe** dataset and detects 11 classes:

| Class | Type |
|---|---|
| `helmet`, `gloves`, `vest`, `boots`, `goggles` | ✅ Compliant — worker is wearing gear |
| `Person` | 👤 Neutral — person detected |
| `no_helmet`, `no_goggle`, `no_gloves`, `no_boots` | ❌ Violation — specific gear missing |
| `none` | 🚨 High severity — no gear at all |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────┐
│                  Docker Compose                  │
│                                                  │
│  ┌─────────────────┐    ┌─────────────────────┐ │
│  │  safety_eye_api  │    │  safety_eye_grafana  │ │
│  │                 │    │                     │ │
│  │  FastAPI :8000  │    │   Grafana  :3000    │ │
│  │  React (static) │    │  (provisioned auto) │ │
│  │  ONNX inference │    └──────────┬──────────┘ │
│  └────────┬────────┘               │             │
│           └──────────┬─────────────┘             │
│                      │                           │
│          ┌───────────▼───────────┐               │
│          │  safety_eye_postgres  │               │
│          │    PostgreSQL :5432   │               │
│          └───────────────────────┘               │
└─────────────────────────────────────────────────┘

Host ports:  API + Frontend → 8100  |  Grafana → 3100  |  DB → 5433
```

---

## 🗂 Project Structure

```
safty_eye/
├── main.py                  # FastAPI app — all API endpoints + serves frontend
├── inference.py             # ONNX inference wrapper (no PyTorch needed)
├── video_processor.py       # Frame-by-frame video analysis pipeline
├── models.py                # SQLAlchemy ORM — 4 database tables
├── database.py              # DB engine and session factory
├── config.py                # Environment config (DB, model path, etc.)
│
├── model/
│   └── weights/
│       └── best.onnx        # Trained YOLOv8 model (ONNX format, 212MB)
│
├── frontend/                # React + Vite frontend
│   └── src/
│       ├── pages/           # Dashboard, Upload, Violations, Live
│       └── components/      # Sidebar, StatCard
│
├── grafana/
│   └── provisioning/        # Auto-configured datasource + dashboard
│
├── Dockerfile               # Multi-stage: builds React then embeds in Python image
├── docker-compose.yml       # Local build (for developers)
├── docker-compose.hub.yml   # Pre-built image from Docker Hub (for end users)
├── init.sql                 # PostgreSQL schema — runs automatically on first start
└── .env                     # Port and credential configuration
```

---

## ⚙️ Configuration

All settings live in the `.env` file. Edit before running `docker compose up`:

```env
# Database
DB_NAME=safety_eye
DB_USER=postgres
DB_PASSWORD=123          # change this in production
DB_PORT=5433

# API + Frontend
API_PORT=8100

# Grafana
GRAFANA_PORT=3100
GRAFANA_USER=Admin
GRAFANA_PASSWORD=MyStrongPass123   # change this in production
```

---

## 🛠 For Developers (Local Build)

If you want to modify the code and build your own image:

### Prerequisites
- Docker Desktop
- Python 3.11+
- Node.js 20+

### Run locally with hot reload

```bash
# 1. Start database and Grafana only
docker compose up postgres grafana

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the API
python main.py

# 4. Start the frontend dev server (in a new terminal)
cd frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| Frontend (hot reload) | http://localhost:5173 |
| API | http://localhost:8000 |
| Grafana | http://localhost:3100 |

### Build and push your own Docker image

```bash
docker build -t your-dockerhub-username/safety-eye:latest .
docker push your-dockerhub-username/safety-eye:latest
```

---

## 📡 API Reference

### Inference
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Run inference on an image, returns annotated result |
| `POST` | `/predict-raw` | Run inference, returns JSON only |

### Video Processing
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/process-video` | Upload video for background processing |
| `POST` | `/process-youtube` | Process YouTube URL (auto-downloads at 720p) |

### Job Management
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/jobs` | List all processing jobs |
| `GET` | `/jobs/{id}` | Check job status |
| `GET` | `/jobs/{id}/preview` | Get latest annotated frame from a job |
| `POST` | `/jobs/{id}/cancel` | Cancel a running job |

### Data & Analysis
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/sessions` | List all detection sessions |
| `GET` | `/sessions/{id}/frame` | Get annotated JPEG for a session |
| `GET` | `/sessions/{id}/detections` | Get all detections for a session |
| `GET` | `/violations` | List violations (filterable by type, severity, status) |
| `PATCH` | `/violations/{id}/acknowledge` | Mark a violation as acknowledged |
| `GET` | `/stats` | Summary statistics (compliance rate, violation count, etc.) |
| `GET` | `/sources` | List processed video sources |

**Full interactive documentation:** http://localhost:8100/docs

---

## 🗄 Database Schema

```
video_sources
  └── detection_sessions    (one row per sampled frame)
        ├── detections       (one row per bounding box detected)
        └── violations       (filtered subset — only violation detections)
```

---

## 🔄 CI/CD

Every push to `main` automatically:
1. Pulls the ONNX model from the existing Docker Hub image
2. Rebuilds the Docker image (updated code + React frontend + model)
3. Pushes `amalsalilan/safety-eye:latest` to Docker Hub

Powered by GitHub Actions — see [`.github/workflows/docker-build.yml`](.github/workflows/docker-build.yml).

---

## 🛟 Troubleshooting

**Containers won't start**
```bash
docker compose -f docker-compose.hub.yml down
docker compose -f docker-compose.hub.yml up
```

**Port already in use**
Edit the port numbers in `.env` and restart.

**Grafana dashboard shows no data**
The dashboard only shows data after you process a video. Go to the **Upload** page, upload a video, and wait for it to finish processing.

**Camera not working on Live page**
Your browser needs camera permission. Click the lock icon in the address bar and allow camera access. Also make sure no other app is using the camera.

**API shows 503 / connection refused**
Wait a few more seconds — the API waits for PostgreSQL to be healthy before starting.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| AI Model | YOLOv8 (exported to ONNX) |
| Inference | ONNX Runtime (no PyTorch needed at runtime) |
| Backend | FastAPI + Python 3.11 |
| Database | PostgreSQL 15 |
| Frontend | React 18 + Vite |
| Analytics | Grafana 10 |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions → Docker Hub |

---

## 📄 License

MIT License — free to use, modify, and distribute.
