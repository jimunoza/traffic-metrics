import cv2
import yaml
import os
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def draw_zones(frame, config):
    overlay = frame.copy()

    # Lanes (verde)
    for poly in config["zones"].get("lanes", []):
        pts = np.array([[int(x), int(y)] for (x, y) in poly], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0,255,0), thickness=2)

    # Crosswalks (amarillo)
    for poly in config["zones"].get("crosswalks", []):
        pts = np.array([[int(x), int(y)] for (x, y) in poly], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(255,200,0), thickness=2)

    # Stop lines (rojo)
    for seg in config["zones"].get("stop_lines", []):
        p1 = (int(seg[0][0]), int(seg[0][1]))
        p2 = (int(seg[1][0]), int(seg[1][1]))
        cv2.line(overlay, p1, p2, (0,0,255), 2)

    return overlay

if __name__ == "__main__":
    load_dotenv()
    source = os.getenv("VIDEO_SOURCE", "")
    if not source:
        raise RuntimeError("❌ Define VIDEO_SOURCE en .env")

    config = load_config("configs/config.yaml")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir {source}")

    # Datos del video original
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Preparar salida
    output_path = Path("/output/zones_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = draw_zones(frame, config)
        writer.write(overlay)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Procesados {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"✅ Video guardado en {output_path}")
