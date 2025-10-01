import os, cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
source = os.getenv("VIDEO_SOURCE", "0")
try:
    source = int(source)
except ValueError:
    pass

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"❌ No se pudo abrir {source}")

model = YOLO("yolov8n.pt")
SCALE = 0.33
frame_count = 0
last_result = None

def generate_frames():
    global frame_count, last_result
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Redimensionar
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        # Correr YOLO solo cada 5 frames
        if frame_count % 5 == 0:
            last_result = model.predict(
                small,
                classes=[0,2],  # solo personas y autos
                conf=0.25,
                verbose=False
            )

        if last_result:
            annotated = last_result[0].plot()
            annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))
        else:
            annotated = frame

        ret, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@router.get("/stream_fast")
def stream_video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
