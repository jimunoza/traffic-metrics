from fastapi import FastAPI
from .stream import router as stream_router

app = FastAPI(title="Traffic Metrics API", version="0.1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Traffic Metrics API running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Registrar ruta de streaming
app.include_router(stream_router)
