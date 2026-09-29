from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "models" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

app = FastAPI(title="Road Damage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(MODEL_PATH))


@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found."}


@app.get("/styles.css")
def serve_styles():
    styles_path = FRONTEND_DIR / "styles.css"
    if styles_path.exists():
        return FileResponse(styles_path)
    return {"message": "Styles not found."}


@app.get("/app.js")
def serve_app_js():
    app_js_path = FRONTEND_DIR / "app.js"
    if app_js_path.exists():
        return FileResponse(app_js_path)
    return {"message": "Frontend script not found."}


@app.get("/health")
def health_check():
    return {"status": "ok", "model": str(MODEL_PATH.name)}


@app.post("/predict")
async def predict_image(
    image: UploadFile = File(...),
    confidence: float = Form(0.05),
):
    if not image:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    file_bytes = await image.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    array = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    results = model(
        frame,
        conf=confidence,
        iou=0.45,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()

    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    detections: list[str] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = model.names[class_id]
            detections.append(f"{class_name} - {conf:.2f}")

    return {
        "message": "Detection complete." if detections else "No road damage detected.",
        "count": len(detections),
        "detections": detections,
        "image": f"data:image/png;base64,{image_b64}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
