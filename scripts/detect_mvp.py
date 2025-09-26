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

    # Cargar modelo preentrenado YOLOv8n
    print("📥 Cargando modelo YOLOv8n...")
    model = YOLO("yolov8n.pt")  # COCO dataset (80 clases)

    # Abrir video fuente
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path("/output/detect_mvp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ejecutar detección (solo clases relevantes)
        """ 0 → person
            1 → bicycle
            2 → car
            3 → motorcycle
            5 → bus
            7 → truck"""
        results = model.predict(frame, classes=[0,1,2,3,5,7], conf=0.25, verbose=False)
        annotated_frame = results[0].plot()  # dibuja bounding boxes

        writer.write(annotated_frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Procesados {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"✅ Video con detecciones guardado en {output_path}")

if __name__ == "__main__":
    main()
