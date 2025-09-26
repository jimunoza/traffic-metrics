import cv2
import os

def open_stream(source: str):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir el stream/archivo: {source}")
    print(f"✅ Fuente abierta: {source}")
    return cap

if __name__ == "__main__":
    # Ruta desde variable de entorno
    source = os.getenv(
        "VIDEO_SOURCE",
        "sample.mp4"  # valor por defecto si no hay env
    )

    cap = open_stream(source)

    frame_count = 0
    while frame_count < 20:  # solo probar con algunos frames
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame (fin de archivo o error).")
            break
        print(f"Frame {frame_count}: {frame.shape}")
        frame_count += 1

    cap.release()
    print("🎬 Lectura terminada.")
