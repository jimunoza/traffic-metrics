🚦 Traffic Metrics – Computer Vision for Intersections

Proyecto de detección y análisis de métricas de tránsito en una intersección con semáforos y cruces peatonales, usando visión por computador.

Actualmente incluye:

📂 Estructura modular (ingest, detect, track, metrics, viz, api, scripts)

🐳 Infraestructura con Docker + Docker Compose

🎥 Ingesta offline desde un clip de video (EarthCam Roswell)

📝 Herramienta para marcar zonas (carriles, cruces peatonales, líneas de detención)

✅ Script de validación de zonas, tanto local (GUI) como en Docker (video overlay)

📂 Estructura del proyecto
traffic-metrics/
│── api/                # FastAPI (API REST)
│── ingest/             # Ingesta de video
│── detect/             # Modelos de detección (próximo paso)
│── track/              # Tracking multi-objeto (más adelante)
│── metrics/            # Cálculo de métricas de tránsito
│── viz/                # Visualización y dashboards
│── scripts/            # Utilidades (marcado, validación, etc.)
│── configs/            # Configuración de zonas y parámetros
│── output/             # Resultados de validación (videos procesados)
│── Dockerfile
│── docker-compose.yml
│── requirements.txt
│── requirements-dev.txt
│── .gitignore
│── README.md
│── .env.example

🚀 Cómo correr el proyecto
1. Configurar variables de entorno

Copia .env.example → .env y edita:

VIDEO_SOURCE=/data/roswell.mov   # ruta al video (mapeado desde tu Desktop)

2. Levantar contenedor con API
docker compose up --build


Endpoints disponibles:

http://localhost:8000/ → raíz API

http://localhost:8000/health → healthcheck

http://localhost:8000/docs → Swagger UI

3. Marcar zonas (local, fuera de Docker)
python scripts/mark_zones.py


Click izquierdo = añadir punto

ENTER = cerrar polígono/segmento

TAB = cambiar tipo (lane → crosswalk → stop_line)

S = guardar en configs/config.yaml

Q = salir

4. Validar zonas

Modo local (ventana OpenCV):

python scripts/show_zones.py


Modo Docker (exportar video con overlay):

docker exec -it traffic_api bash -c "python scripts/show_zones_docker.py"


Resultado en ./output/zones_overlay.mp4.

✅ Estado de avance por niveles

Nivel 0 – Fundaciones ✅
Estructura, Docker, API mínima, ingesta offline.

Nivel 1 – Zonas de intersección ✅
Editor manual + validación.

Nivel 2 – Detección (próximo) 🚧
Integración de un detector (ej. YOLOv8) para autos y peatones.

🛠 Requisitos

Docker + Docker Compose

(opcional) Python 3.10+ para correr herramientas locales (requirements-dev.txt)