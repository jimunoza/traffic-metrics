import os
import cv2
import yaml
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# ---------- Helpers ----------

class ZoneEditor:
    def __init__(self, frame, display_max_w=1280):
        self.frame_orig = frame
        self.h, self.w = frame.shape[:2]
        self.scale = 1.0
        if self.w > display_max_w:
            self.scale = display_max_w / self.w
        self.frame_disp = cv2.resize(self.frame_orig, (int(self.w*self.scale), int(self.h*self.scale)))
        self.overlay = self.frame_disp.copy()
        self.cur_points = []  # puntos (display space)
        self.mode = "lane"    # lane | crosswalk | stop_line
        self.lanes = []       # list[list[(x,y)] en original]
        self.crosswalks = []  # list[list[(x,y)] en original]
        self.stop_lines = []  # list[ [(x1,y1),(x2,y2)] en original ]
        self.message = "Click para añadir puntos. ENTER=cerrar polígono/segmento · BACKSPACE=deshacer · TAB=cambiar tipo · S=guardar · R=reset · Q=salir"

    def disp2orig(self, p):
        return (int(p[0] / self.scale), int(p[1] / self.scale))

    def draw_hud(self):
        # Redibujar
        self.overlay = self.frame_disp.copy()
        # dibujar ya guardados
        # lanes
        for poly in self.lanes:
            pts = [(int(x*self.scale), int(y*self.scale)) for (x,y) in poly]
            for i in range(len(pts)):
                cv2.line(self.overlay, pts[i], pts[(i+1)%len(pts)], (0,255,0), 2)
            for p in pts:
                cv2.circle(self.overlay, p, 3, (0,180,0), -1)
        # crosswalks
        for poly in self.crosswalks:
            pts = [(int(x*self.scale), int(y*self.scale)) for (x,y) in poly]
            for i in range(len(pts)):
                cv2.line(self.overlay, pts[i], pts[(i+1)%len(pts)], (255,200,0), 2)
            for p in pts:
                cv2.circle(self.overlay, p, 3, (180,160,0), -1)
        # stop_lines
        for seg in self.stop_lines:
            p1 = (int(seg[0][0]*self.scale), int(seg[0][1]*self.scale))
            p2 = (int(seg[1][0]*self.scale), int(seg[1][1]*self.scale))
            cv2.line(self.overlay, p1, p2, (0,0,255), 2)
            cv2.circle(self.overlay, p1, 4, (0,0,160), -1)
            cv2.circle(self.overlay, p2, 4, (0,0,160), -1)

        # puntos actuales (en edición)
        color = {"lane": (0,255,0), "crosswalk": (255,200,0), "stop_line": (0,0,255)}[self.mode]
        for i in range(1, len(self.cur_points)):
            cv2.line(self.overlay, self.cur_points[i-1], self.cur_points[i], color, 2)
        for p in self.cur_points:
            cv2.circle(self.overlay, p, 4, color, -1)

        # HUD
        bar = self.overlay.copy()
        cv2.rectangle(bar, (0,0), (bar.shape[1], 50), (30,30,30), -1)
        cv2.putText(bar, f"Mode: {self.mode}  |  {self.message}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        self.overlay = bar

    def commit_shape(self):
        if self.mode in ("lane", "crosswalk"):
            if len(self.cur_points) >= 3:
                poly_orig = [self.disp2orig(p) for p in self.cur_points]
                if self.mode == "lane":
                    self.lanes.append(poly_orig)
                else:
                    self.crosswalks.append(poly_orig)
                self.cur_points = []
        elif self.mode == "stop_line":
            if len(self.cur_points) == 2:
                seg_orig = [self.disp2orig(p) for p in self.cur_points]
                self.stop_lines.append(seg_orig)
                self.cur_points = []

    def save_yaml(self, path):
        data = {
            "meta": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "image_size": [self.w, self.h]
            },
            "zones": {
                "lanes": [[list(pt) for pt in poly] for poly in self.lanes],
                "crosswalks": [[list(pt) for pt in poly] for poly in self.crosswalks],
                "stop_lines": [[list(pt) for pt in seg] for seg in self.stop_lines],
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"✅ Guardado en {path}")

# ---------- Main ----------

def main():
    # cargar fuente
    load_dotenv()
    source = os.getenv("VIDEO_SOURCE", "")
    if not source:
        print("❌ Define VIDEO_SOURCE en .env")
        return

    # abrir video y tomar primer frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {source}")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ No se pudo leer el primer frame.")
        return

    editor = ZoneEditor(frame)
    win = "Zone Editor — ENTER=cerrar · TAB=cambiar tipo · S=guardar · R=reset · Q=salir"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, editor.overlay.shape[1], editor.overlay.shape[0])

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.cur_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # deshacer último punto
            if editor.cur_points:
                editor.cur_points.pop()

    cv2.setMouseCallback(win, on_mouse)

    save_path = Path("configs/config.yaml")

    while True:
        editor.draw_hud()
        cv2.imshow(win, editor.overlay)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:  # no key
            continue

        if key in (ord('\r'), 13):  # ENTER
            editor.commit_shape()
        elif key == 8:  # BACKSPACE
            if editor.cur_points:
                editor.cur_points.pop()
        elif key == 9:  # TAB
            # rotar modo
            modes = ["lane", "crosswalk", "stop_line"]
            idx = modes.index(editor.mode)
            editor.mode = modes[(idx + 1) % len(modes)]
        elif key in (ord('s'), ord('S')):
            editor.save_yaml(save_path)
        elif key in (ord('r'), ord('R')):
            editor.cur_points.clear()
            editor.lanes.clear()
            editor.crosswalks.clear()
            editor.stop_lines.clear()
        elif key in (ord('q'), ord('Q'), 27):  # Q o ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
