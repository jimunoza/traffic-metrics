import os
from pathlib import Path
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO
import torch

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Flags
LIVE_VIEW = False # flag ya no usado, no integrar vista en vivo
SAVE_VIDEO = True  # default = no guardar (máxima velocidad)

# Parámetros de reducción
TARGET_FPS = 30
PROC_W, PROC_H = 1280, 960  # resolución reducida para procesar/guardar

def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("❌ Tracker KCF no disponible. Instala opencv-contrib-python.")

def main():
    load_dotenv()
    source = os.getenv("VIDEO_SOURCE", "")
    if not source:
        raise RuntimeError("❌ Define VIDEO_SOURCE en .env")

    # Selección de dispositivo (MPS en Mac M1 si está disponible)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Usando dispositivo: {device}")

    # Modelo YOLO
    model = YOLO("yolov8n.pt").to(device)

    # Video input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🎥 Video original: {fps_in:.1f} FPS, {orig_w}x{orig_h}")

    # Output (si está activado)
    writer = None
    output_path = Path("/output/track_kcf_fast.mp4")
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, (PROC_W, PROC_H))

    trackers = []
    next_id = 0
    frame_count = 0
    DETECT_INTERVAL = 90

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ⚡ Forzar 30 FPS: si el video original es >30fps, saltar frames
        if fps_in > TARGET_FPS and frame_count % int(fps_in / TARGET_FPS) != 0:
            continue

        # Redimensionar frame
        frame_resized = cv2.resize(frame, (PROC_W, PROC_H))

        # Re-detección con YOLO cada N frames
        if frame_count == 1 or frame_count % DETECT_INTERVAL == 0:
            trackers = []
            results = model.predict(
                frame_resized,
                classes=list(CLASSES.keys()),
                conf=0.3,
                imgsz=480,  # 🚀 más chico = más rápido
                verbose=False,
                device=device
            )

            for box in results[0].boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                if w * h < 300:  # filtrar detecciones muy pequeñas
                    continue

                tracker = create_tracker()
                tracker.init(frame_resized, (x1, y1, w, h))
                trackers.append((tracker, next_id, CLASSES[cls]))
                next_id += 1

        # Actualizar trackers
        new_trackers = []
        for tracker, obj_id, label in trackers:
            ok, bbox = tracker.update(frame_resized)
            if ok:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{label}-{obj_id}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                new_trackers.append((tracker, obj_id, label))
        trackers = new_trackers

        # Guardar si está activado
        if SAVE_VIDEO and writer is not None:
            writer.write(frame_resized)

        # Mostrar live
        if LIVE_VIEW:
            cv2.imshow("Tracking rápido 30fps", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_count % 50 == 0:
            print(f"Procesados {frame_count} frames...")

    cap.release()
    if SAVE_VIDEO and writer is not None:
        writer.release()
        print(f"✅ Video con tracking guardado en {output_path}")
    if LIVE_VIEW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
