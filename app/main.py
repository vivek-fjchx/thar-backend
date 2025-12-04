from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.cam_predict import CamPredictor
import os
import gc

# Import predictor class but don't instantiate yet
from app.predict import Predictor

# >>> NEW — import the GradCAM predictor
#from app.gradcam_predict import GradCAMPredictor


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
# Lazy load predictors
# -------------------------------
predictor = None
#gradcam_predictor = None   # >>> NEW


def get_predictor():
    """Load ONNX model only when first prediction is requested"""
    global predictor
    if predictor is None:
        print("Loading ONNX model for the first time...")
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        models_dir = os.path.abspath(models_dir)
        
        onnx_path = os.path.join(models_dir, "thar_wrangler.onnx")
        pytorch_weights = None   # ONNX-only now

        predictor = Predictor(onnx_path, pytorch_weights)
        gc.collect()
        print("ONNX model loaded!")
    return predictor


cam_predictor = None

def get_cam_predictor():
    global cam_predictor
    if cam_predictor is None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        onnx_path = os.path.join(models_dir, "thar_wrangler.onnx")
        cam_predictor = CamPredictor(onnx_path)
    return cam_predictor



# >>> NEW - Lazy load PyTorch GradCAM model
'''def get_gradcam_predictor():
    """Load PyTorch model only when GradCAM is requested"""
    global gradcam_predictor
    if gradcam_predictor is None:
        print("Loading PyTorch GradCAM model...")
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        models_dir = os.path.abspath(models_dir)

        pytorch_weights = os.path.join(models_dir, "thar_wrangler_mobilenetv2.pth")
        gradcam_predictor = GradCAMPredictor(pytorch_weights)
        gc.collect()
        print("GradCAM model loaded!")
    return gradcam_predictor'''



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
        "onnx_loaded": predictor is not None,
        "cam_loaded": cam_predictor is not None
    }


# -------------------------------
# Prediction API
# -------------------------------
async def _handle_predict_async(image: UploadFile):
    """Shared async prediction logic"""
    try:
        pred = get_predictor()
        image_bytes = await image.read()

        response = pred.predict_from_bytes(image_bytes)

        del image_bytes
        gc.collect()

        return response

    except MemoryError:
        gc.collect()
        return JSONResponse(
            content={"error": "Server out of memory. Try again."},
            status_code=503
        )

    except Exception as e:
        gc.collect()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    return await _handle_predict_async(image)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    return await _handle_predict_async(image)



# -------------------------------
# >>> NEW — GradCAM Endpoint
# -------------------------------
'''@app.post("/api/gradcam")
async def gradcam_api(image: UploadFile = File(...)):
    try:
        model = get_gradcam_predictor()
        image_bytes = await image.read()
        result = model.generate_gradcam(image_bytes)

        del image_bytes
        gc.collect()

        return result

    except Exception as e:
        print("GradCAM ERROR:", e)
        gc.collect()
        return JSONResponse({"error": str(e)}, status_code=500)'''


@app.post("/api/cam")
async def cam_api(image: UploadFile = File(...)):
    try:
        model = get_cam_predictor()
        image_bytes = await image.read()
        return model.generate_cam(image_bytes)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)




# -------------------------------
# Startup event
# -------------------------------
@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting...")
    print("Models will load lazily.")
    gc.collect()


# -------------------------------
# Shutdown event
# -------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global predictor, cam_predictor
    if predictor is not None:
        del predictor
    if cam_predictor is not None:
        del cam_predictor
    gc.collect()
    print("👋 Server shutting down...")
