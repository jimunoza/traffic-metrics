import cv2

# Ruta de tu imagen o un frame del video
IMAGE_PATH = "/Users/jose/Desktop/roi.png"  # usa tu imagen .png o .jpg

# Cargar imagen
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError(f"No se pudo abrir la imagen: {IMAGE_PATH}")

# Función callback para clicks del mouse
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordenadas clic: ({x}, {y})")
        # Dibuja un punto visible donde hiciste clic
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("Selecciona puntos", img)

# Crear ventana interactiva
cv2.imshow("Selecciona puntos", img)
cv2.setMouseCallback("Selecciona puntos", click_event)

print("🖱️ Haz clic en los puntos que quieras (ESC para salir)...")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break

cv2.destroyAllWindows()
