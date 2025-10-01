# 🚦 Traffic Metrics – Computer Vision for Smart Intersections

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)  
[![FastAPI](https://img.shields.io/badge/api-fastapi-green)](https://fastapi.tiangolo.com/)  
[![YOLOv8](https://img.shields.io/badge/detection-yolov8-orange)](https://github.com/ultralytics/ultralytics)  
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)  

**Smart traffic analytics with computer vision — YOLO + KCF + FastAPI, Docker-ready.**

---

## 📖 Overview

Traffic Metrics is a modular **computer vision system** for analyzing traffic at signalized intersections with crosswalks.  
It combines **deep learning** for detection (YOLOv8) with **classical tracking** (KCF) to provide real-time and offline metrics such as:

- Vehicle & pedestrian detection  
- Multi-object tracking with unique IDs  
- Lane occupancy and stop-line control  
- Waiting times and queue lengths (planned)  
- Red-light violations and near-miss analysis (future work)  

The project is designed for **research and prototyping**, fully reproducible with Docker.

---

## 🏗️ Architecture

```text
Video Source (File / RTSP / Webcam)
            │
        Ingest (OpenCV)
            │
    ┌───────┴────────┐
    │ Detection (YOLOv8) ── Deep Learning
    │ Tracking (KCF)  ──── Classical CV
    └───────┬────────┘
            │
        Zone Mapping (lanes, crosswalks, stop-lines)
            │
        Metrics Engine (queues, speeds, waits)
            │
   API (FastAPI) ──▶ Dashboards / Stream / DB
```

---

## 📂 Project Structure

```
traffic-metrics/
│── api/                # FastAPI endpoints (health, stream, etc.)
│   └── stream.py
│── ingest/             # Video capture
│── detect/             # Detection logic
│── track/              # Tracking (KCF)
│── metrics/            # Metrics calculation (future)
│── viz/                # Visualization / dashboards
│── scripts/            # CLI utilities
│   ├── mark_zones.py   # GUI tool to mark lanes/crosswalks
│   ├── show_zones.py   # Local validation with OpenCV
│   ├── detect_mvp.py   # Basic YOLO detection
│   ├── track_kcf.py    # YOLO + KCF tracking
│   └── detect_realtime.py / stream_fast.py
│── configs/            # Intersection configs (zones.yaml)
│── output/             # Processed video outputs
│── Dockerfile
│── docker-compose.yml
│── requirements.txt
│── requirements-dev.txt
│── .gitignore
│── .env.example
│── README.md
```

---

## 🚀 Getting Started

### 1. Clone repo
```bash
git clone https://github.com/YOURUSER/traffic-metrics.git
cd traffic-metrics
```

### 2. Configure environment
Copy `.env.example` → `.env` and set video source:
```env
VIDEO_SOURCE=/data/roswell.mov        # file (mounted from host)
# VIDEO_SOURCE=0                      # local webcam
# VIDEO_SOURCE=rtsp://user:pass@ip/   # RTSP camera
```

### 3. Build with Docker
```bash
docker compose build
docker compose up
```

API will be available at:
- `http://localhost:8000/` → root  
- `http://localhost:8000/docs` → Swagger docs  
- `http://localhost:8000/health` → healthcheck  
- `http://localhost:8000/stream` → live detections (MJPEG)  

---

## 🎥 Usage Examples

### Mark intersection zones (lanes, crosswalks, stop-lines)
```bash
python scripts/mark_zones.py
```
Result saved in `configs/config.yaml`.

### Validate zones visually
```bash
python scripts/show_zones.py
```

### Run detection MVP (offline)
```bash
docker exec -it traffic_api python scripts/detect_mvp.py
```
Output saved in `output/detect_mvp.mp4`.

### Run YOLO + KCF tracking (offline)
```bash
docker exec -it traffic_api python scripts/track_kcf.py
```
Output saved in `output/track_kcf_fast.mp4`.

### Real-time detection (API stream)
```bash
open http://localhost:8000/stream
```

---

## ✅ Roadmap (Levels)

- **Level 0** – Foundations: structure, Docker, API ✅  
- **Level 1** – Zone definition & validation ✅  
- **Level 2** – Detection MVP (YOLOv8) + real-time stream ✅  
- **Level 3** – Tracking with KCF (multi-object IDs) ✅  
- **Level 4** – Calibration & metric map (px → meters) 🚧  
- **Level 5** – Lane assignment & movement classification  
- **Level 6** – Signal phase detection (API or vision)  
- **Level 7** – Metrics v1: counts, queues, waits  
- **Level 8+** – Safety analysis: near-misses, violations  

---

## 🛠️ Requirements

- Python 3.10+ (for local tools)  
- Docker + Docker Compose  
- Optional: GPU with CUDA for faster YOLO inference  

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.  
Future improvements include multi-camera support, database integration, and advanced metrics.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
