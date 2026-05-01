import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# ⚙️ CONFIGURACIÓN GENERAL
# ===========================
VIDEO_PATH = "/Users/jose/Desktop/tobalaba.mp4"
MODEL_PATH = "yolov8m.pt"                  # modelo YOLO
OUTPUT_CSV = "headway_events.csv"

# Flags
USE_ROI = True       # usa ROI y línea desde roi_config.py
SHOW_ROI = True      # muestra ventana auxiliar con la ROI
SAVE_VIDEO = False   # opcional para grabar video de salida

if USE_ROI:
    import roi_config                     # archivo separado (definido abajo)

# ===========================
# 🔧 PARÁMETROS DE DETECCIÓN
# ===========================
CONF_THRESH = 0.25
TARGET_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
DETECT_INTERVAL = 10
MIN_AREA = 100  # tamaño mínimo del bbox

# ===========================
# 🚀 CARGA DEL MODELO
# ===========================
device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Dispositivo: {device}")

model = YOLO(MODEL_PATH).to(device)
print("✅ Modelo YOLO cargado correctamente.")

# ===========================
# 🎥 VIDEO DE ENTRADA
# ===========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"❌ No se pudo abrir el video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎞️ Video: {width}x{height} a {fps:.1f} FPS")

if SAVE_VIDEO:
    out = cv2.VideoWriter(
        "output_tracking.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

# ===========================
# VARIABLES GLOBALES
# ===========================
trackers = []
next_id = 0
frame_idx = 0
cross_events = []

# ===========================
# LOOP PRINCIPAL
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    timestamp = frame_idx / fps

    # ROI opcional
    if USE_ROI:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(roi_config.ROI_POINTS)], 255)
        frame_roi = cv2.bitwise_and(frame, frame, mask=mask)
        detect_frame = frame_roi
        if SHOW_ROI and frame_idx % 20 == 0:
            cv2.imshow("ROI usada", frame_roi)
    else:
        detect_frame = frame

    # Detección periódica
    if frame_idx == 1 or frame_idx % DETECT_INTERVAL == 0:
        trackers = []
        results = model.predict(detect_frame, conf=CONF_THRESH, verbose=False, device=device)
        boxes = results[0].boxes

        print(f"[Frame {frame_idx}] Detecciones: {len(boxes)}")

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_AREA:
                continue

            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            trackers.append((tracker, next_id, None))
            next_id += 1

    # Actualización de trackers
    # Actualización de trackers
    TOLERANCIA = 5  # margen vertical en píxeles

    for i, (tracker, obj_id, prev_center) in enumerate(trackers):
        ok, bbox = tracker.update(frame)
        if not ok:
            continue
        x, y, w, h = map(int, bbox)
        cx, cy = x + w // 2, y + h // 2
        bottom_y = y + h  # borde inferior del bbox

        # ===== Verificar cruce de línea =====
        if USE_ROI:
            p1, p2 = roi_config.LINE_P1, roi_config.LINE_P2
            y_line = p1[1] + (p2[1] - p1[1]) * ((cx - p1[0]) / (p2[0] - p1[0]))

            # Vehículo nuevo que ya aparece cruzando
            if prev_center is None and bottom_y >= y_line - TOLERANCIA:
                cross_events.append((obj_id, timestamp))
                print(f"🚗 [NEW] Vehículo {obj_id} cruzó (inicio) a {timestamp:.2f}s")

            # Vehículo que cruza desde arriba hacia abajo (con tolerancia)
            elif prev_center and prev_center[1] < (y_line - TOLERANCIA) and bottom_y >= (y_line + TOLERANCIA):
                cross_events.append((obj_id, timestamp))
                print(f"🚗 Vehículo {obj_id} cruzó a {timestamp:.2f}s")

            trackers[i] = (tracker, obj_id, (cx, cy))

        else:
            # Línea horizontal simple (sin ROI)
            LINE_Y = 550
            if prev_center is None and bottom_y >= LINE_Y - TOLERANCIA:
                cross_events.append((obj_id, timestamp))
            elif prev_center and prev_center[1] < (LINE_Y - TOLERANCIA) and bottom_y >= (LINE_Y + TOLERANCIA):
                cross_events.append((obj_id, timestamp))
                print(f"🚗 Vehículo {obj_id} cruzó a {timestamp:.2f}s")
            trackers[i] = (tracker, obj_id, (cx, cy))

        # ===== Dibujar bounding boxes y puntos de referencia =====
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Visualizar punto inferior y centro
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)      # centro
        cv2.circle(frame, (cx, bottom_y), 3, (255, 0, 255), -1)  # borde inferior


    # Dibujar línea y ROI
    if USE_ROI:
        cv2.line(frame, roi_config.LINE_P1, roi_config.LINE_P2, (0, 0, 255), 3)
        cv2.polylines(frame, [np.array(roi_config.ROI_POINTS)], True, (255, 0, 0), 2)
    else:
        cv2.line(frame, (0, 550), (frame.shape[1], 550), (0, 0, 255), 3)

    if SAVE_VIDEO:
        out.write(frame)

    cv2.imshow("Headway Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()

# ===========================
# CÁLCULO DE HEADWAYS
# ===========================
if cross_events:
    # Crear DataFrame con los cruces
    df = pd.DataFrame(cross_events, columns=["id", "timestamp"])
    df = df.sort_values("timestamp")
    df["headway_s"] = df["timestamp"].diff()

    # Guardar datos base
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Resultados guardados en {OUTPUT_CSV}")

    # ======= Estadísticas descriptivas =======
    summary = df.describe()
    summary_path = OUTPUT_CSV.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"📊 Resumen estadístico guardado en {summary_path}")
    print(summary)

    # ======= Calcular flujo vehicular =======
    total_cruces = len(df)
    duracion_seg = frame_idx / fps if fps > 0 else None
    flujo_veh_min = None
    if duracion_seg and duracion_seg > 0:
        flujo_veh_min = total_cruces / (duracion_seg / 60)
        print(f"🚗 Flujo vehicular aproximado: {flujo_veh_min:.1f} vehículos/minuto")

    # ======= Gráfico de distribución de headways =======
    valid_headways = df["headway_s"].dropna()
    if not valid_headways.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(valid_headways, bins=10, color="#4A90E2", edgecolor="black", alpha=0.8)

        # Líneas de referencia
        plt.axvline(valid_headways.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Promedio = {valid_headways.mean():.2f} s")
        plt.axvline(valid_headways.median(), color="green", linestyle=":", linewidth=1.5,
                    label=f"Mediana = {valid_headways.median():.2f} s")

        # Título dinámico con flujo
        titulo = "Distribución de Headways"
        if flujo_veh_min:
            titulo += f"\nFlujo: {flujo_veh_min:.1f} veh/min"

        plt.title(titulo, fontsize=14, weight="bold")
        plt.xlabel("Headway (s)")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Guardar gráfico
        plot_path = OUTPUT_CSV.replace(".csv", "_hist.png")
        plt.savefig(plot_path, dpi=300)
        print(f"📈 Gráfico guardado en {plot_path}")
        plt.show()
    else:
        print("⚠️ No hay headways válidos para graficar.")

else:
    print("\n⚠️ No se registraron cruces.")