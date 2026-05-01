# traffic-metrics

> Open-source computer vision pipeline for vehicle counting and headway estimation at urban intersections.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-EE4C2C?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## What it does

**traffic-metrics** processes video from a fixed urban camera and estimates fundamental traffic flow metrics — without cloud services, proprietary software, or dedicated hardware.

Given a video file, the pipeline:

1. Runs frame-by-frame vehicle detection using **YOLOv8** (classes: car, truck, bus, motorcycle)
2. Defines a **region of interest (ROI)** and a **virtual counting line** to filter irrelevant detections
3. Records a crossing event each time a vehicle's bounding box crosses the line
4. Computes **temporal headways** (Δt between consecutive crossings) and **vehicle count**
5. Exports results to **CSV** for downstream analysis

The system was validated on real footage from the Tobalaba intersection in Santiago, Chile, across multiple resolutions (480p, 720p, 1080p) and lighting conditions.

---

## Results

| Video | Resolution | Auto count | Manual count |
|-------|-----------|-----------|-------------|
| Tobalaba | 1920×1080 | 45 | 42 |
| Tobalaba | 640×480 | 47 | 35 |

- **Best accuracy:** 1080p, evening hours (18:00–20:30) when artificial lighting complements natural light without backlight interference
- **Estimated flow:** ~56.8 veh/min on the analyzed segment
- **Headway distribution:** mean 1.03 s, median 0.33 s — low-headway spikes correspond to traffic light release bursts and occasional double-detections

Compared qualitatively against [Data From Sky](https://datafromsky.com/), a commercial aerial traffic analytics platform. The open-source pipeline reproduces core metrics (count, headway, flow) at a fraction of the cost, with higher variability due to the absence of a robust multi-object tracker.

---

## Pipeline

```
Video → ROI crop → YOLOv8 detection → Periodic re-detection → Virtual line crossing → CSV / metrics
```

Early versions used OpenCV's KCF tracker, which proved unstable under occlusions and scale changes. The current approach replaces it with periodic YOLO re-detection, improving both accuracy and processing speed.

---

## Setup

```bash
git clone https://github.com/jimunoza/traffic-metrics.git
cd traffic-metrics
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, `ultralytics`, `opencv-python`, `pandas`

---

## Usage

```bash
python main.py --video path/to/video.mp4 --model yolov8m.pt
```

Output: a CSV file with per-crossing timestamps and computed headways, plus an annotated video with bounding boxes and the virtual counting line overlaid.

---

## Limitations & future work

- No robust multi-object tracker (BOT-SORT, ByteTrack) — double-counting can occur in dense traffic
- ROI and counting line are currently hardcoded per scene; a configuration interface would improve reusability
- No support for multi-lane or multi-direction counting yet

---

## Authors

José Ignacio Muñoz · Rodrigo Zárate — Universidad de los Andes, Santiago, Chile

---

## References

- Ultralytics YOLOv8 — https://docs.ultralytics.com/
- Data From Sky — https://datafromsky.com/