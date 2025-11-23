from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import gc

# Import predictor class (ONNX-only)
from app.predict import Predictor


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()


# -------------------------------
# CORS
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
    """Load ONNX model only when first prediction is requested"""
    global predictor
    if predictor is None:
        print("Loading ONNX model for the first time...")

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

        onnx_path = os.path.join(base_dir, "thar_wrangler.onnx")

        # Load ONNX-only Predictor (no PyTorch)
        predictor = Predictor(onnx_path=onnx_path)

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
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


# -------------------------------
# Prediction API
# -------------------------------
@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    try:
        pred = get_predictor()

        image_bytes = await image.read()

        response = pred.predict_from_bytes(image_bytes)

        del image_bytes
        gc.collect()

        return response

    except MemoryError as e:
        print(f"🔥 MEMORY ERROR: {e}")
        gc.collect()
        return JSONResponse(
            content={"error": "Server out of memory. Free tier has limited RAM."},
            status_code=503
        )

    except Exception as e:
        print(f"🔥 BACKEND ERROR: {e}")
        gc.collect()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# -------------------------------
# Startup event
# -------------------------------
@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting...")
    print("ONNX model will load on first request.")
    gc.collect()


# -------------------------------
# Shutdown event
# -------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global predictor
    if predictor is not None:
        del predictor
    gc.collect()
    print("👋 Server shutting down...")
