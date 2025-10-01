import os
from pathlib import Path
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO

def main():
    load_dotenv()
    source = os.getenv("VIDEO_SOURCE", "")
    if not source:
        raise RuntimeError("❌ Define VIDEO_SOURCE en .env")

    # Cargar modelo YOLOv8n
    print("📥 Cargando modelo YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # Abrir video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output
    output_path = Path("/output/detect_mvp_fast.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    SCALE = 0.5   # bajar resolución
    FRAME_SKIP = 2  # procesar 1 de cada 2 frames
    frame_count = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Saltar frames
        if frame_count % FRAME_SKIP != 0:
            continue

        # Reducir resolución antes de detección
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        # Detección solo en clases relevantes
        results = model.predict(
            small,
            classes=[0,2,5,7],
            conf=0.30,
            verbose=False
        )

        # Dibujar en frame original
        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (width, height))

        writer.write(annotated)
        processed += 1

        if processed % 50 == 0:
            print(f"Procesados {processed} frames (saltados {frame_count - processed})...")

    cap.release()
    writer.release()
    print(f"✅ Video rápido guardado en {output_path}")

if __name__ == "__main__":
    main()
