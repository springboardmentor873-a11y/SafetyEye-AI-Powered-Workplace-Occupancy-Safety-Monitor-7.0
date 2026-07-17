# ── Stage 1: Build React frontend ─────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python API + embedded frontend ────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Embed the built React app — served by FastAPI at /
COPY --from=frontend /frontend/dist ./static

RUN mkdir -p /app/frames

EXPOSE 8000
CMD ["python", "main.py"]
