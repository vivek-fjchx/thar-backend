from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import os
import gc

from app.predict import Predictor


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()


# -------------------------------
# CORS (required for Next.js)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Lazy load predictor
# -------------------------------
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        print("Loading model for the first time...")
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        models_dir = os.path.abspath(models_dir)

        onnx_path = os.path.join(models_dir, "thar_wrangler.onnx")
        pytorch_weights = os.path.join(models_dir, "thar_wrangler_mobilenetv2.pth")

        predictor = Predictor(onnx_path, pytorch_weights)
        gc.collect()
        print("Model loaded successfully!")
    return predictor


# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Backend Running!", "status": "healthy"}


# -------------------------------
# Health check
# -------------------------------
@app.get("/api/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}



# ================================================================
# INTERNAL SHARED PREDICTION HANDLER
# ================================================================
async def _predict_single(image: UploadFile):
    """Handles prediction for a single uploaded image."""
    try:
        pred = get_predictor()

        image_bytes = await image.read()

        response = pred.predict_from_bytes(image_bytes)

        del image_bytes
        gc.collect()

        return {
            "filename": image.filename,
            "class": response["class"],
            "confidence": response["confidence"],
            "gradcam": response["gradcam"],
        }

    except MemoryError:
        gc.collect()
        return {
            "filename": image.filename,
            "error": "Out of memory. Render free tier limit reached."
        }

    except Exception as e:
        gc.collect()
        return {
            "filename": image.filename,
            "error": str(e)
        }



# ================================================================
# MULTIPLE FILE PREDICTION ENDPOINT
# ================================================================
@app.post("/api/predict")
async def predict_api(files: List[UploadFile] = File(...)):
    """Accepts MULTIPLE images and returns predictions."""
    results = []

    for img in files:
        result = await _predict_single(img)
        results.append(result)

    return JSONResponse({"results": results})


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """Alias endpoint for /predict"""
    results = []

    for img in files:
        result = await _predict_single(img)
        results.append(result)

    return JSONResponse({"results": results})



# -------------------------------
# Startup
# -------------------------------
@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting up… model loads on first request.")
    gc.collect()


# -------------------------------
# Shutdown
# -------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global predictor
    if predictor is not None:
        del predictor
    gc.collect()
    print("👋 Server shutting down…")
