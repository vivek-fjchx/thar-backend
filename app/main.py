from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import gc

# Import predictor class but don't instantiate yet
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
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Lazy load predictor
# -------------------------------
predictor = None

def get_predictor():
    """Load model only when first prediction is requested"""
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
# Health check (doesn't load model)
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
async def _handle_predict_async(image: UploadFile):
    """Shared async prediction logic"""
    try:
        # Lazy load predictor
        pred = get_predictor()
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Make prediction
        response = pred.predict_from_bytes(image_bytes)
        
        # Clean up
        del image_bytes
        gc.collect()
        
        return response

    except MemoryError as e:
        print(f"🔥 MEMORY ERROR: {e}")
        gc.collect()
        return JSONResponse(
            content={
                "error": "Server out of memory. The free tier has limited RAM. Please try again or contact support."
            },
            status_code=503
        )
    
    except Exception as e:
        print(f"🔥 BACKEND ERROR: {e}")
        gc.collect()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    """Prediction endpoint - matches /api/predict"""
    return await _handle_predict_async(image)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Prediction endpoint - matches frontend /predict"""
    return await _handle_predict_async(image)


# -------------------------------
# Startup event
# -------------------------------
@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    print("🚀 Server starting up...")
    print("Model will be loaded on first prediction request")
    gc.collect()


# -------------------------------
# Shutdown event
# -------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global predictor
    if predictor is not None:
        del predictor
    gc.collect()
    print("👋 Server shutting down...")
