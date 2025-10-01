import os
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

def main():
    load_dotenv()
    source = os.getenv("VIDEO_SOURCE", "0")  # webcam por defecto
    try:
        source = int(source)  # si es número, usar como cámara local
    except ValueError:
        pass

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir {source}")

    SCALE = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        # Detectar
        results = model.predict(small, classes=[0,2,5,7], conf=0.25, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))

        # Mostrar
        cv2.imshow("Real-time detection (press Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
