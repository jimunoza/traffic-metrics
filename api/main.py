from fastapi import FastAPI

app = FastAPI(title="Traffic Metrics API", version="0.1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Traffic Metrics API running (offline mode)"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
