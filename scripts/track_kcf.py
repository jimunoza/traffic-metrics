import os
from pathlib import Path
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO

# Clases COCO relevantes (solo personas, autos, buses, camiones)
CLASSES = {0: "person", 2: "car", 5: "bus", 7: "truck"}

def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("❌ Tracker KCF no está disponible. Instala opencv-contrib-python.")

def main():
    load_dotenv()
    source = os.getenv("VIDEO_SOURCE", "")
    if not source:
        raise RuntimeError("❌ Define VIDEO_SOURCE en .env")

    # Modelo YOLOv8n
    print("📥 Cargando modelo YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # Video input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output
    output_path = Path("/output/track_kcf_fast.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trackers = []   # lista de (tracker, id, label)
    next_id = 0
    frame_count = 0
    DETECT_INTERVAL = 60  # detección cada 60 frames
    SCALE = 0.5  # reducir resolución a la mitad

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- Re-detección periódica con YOLO ---
        if frame_count == 1 or frame_count % DETECT_INTERVAL == 0:
            trackers = []

            # Reducir frame
            small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

            results = model.predict(
                small,
                classes=list(CLASSES.keys()),
                conf=0.30,
                verbose=False
            )

            for box in results[0].boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = [int(v / SCALE) for v in box.xyxy[0]]
                w, h = x2 - x1, y2 - y1

                # Filtros de tamaño
                if cls == 0:  # persona
                    if w * h < 150:
                        continue
                else:  # vehículos
                    if w * h < 400:
                        continue

                tracker = create_tracker()
                tracker.init(frame, (x1, y1, w, h))
                trackers.append((tracker, next_id, CLASSES[cls]))
                next_id += 1

        # --- Actualizar trackers ---
        new_trackers = []
        for tracker, obj_id, label in trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}-{obj_id}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                new_trackers.append((tracker, obj_id, label))
        trackers = new_trackers

        writer.write(frame)

        if frame_count % 100 == 0:
            print(f"Procesados {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"✅ Video rápido guardado en {output_path}")

if __name__ == "__main__":
    main()
