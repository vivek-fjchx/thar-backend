from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import gc
import torch
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

# 💡 Your existing ONNX predictor
from app.predict import Predictor

# 💡 GradCAM utilities (use your file or module)
from app.gradcam import GradCAM, overlay_heatmap, encode_image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Lazy-loaded predictors
# -------------------------------------------------
predictor = None
gradcam_model = None
target_layer = None
preprocess = None


# -------------------------------------------------
# Load ONNX model (normal prediction)
# -------------------------------------------------
def get_predictor():
    global predictor
    if predictor is None:
        print("🔧 Loading ONNX predictor...")
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        onnx_path = os.path.join(models_dir, "thar_wrangler.onnx")
        pytorch_weights = os.path.join(models_dir, "thar_wrangler_mobilenetv2.pth")

        predictor = Predictor(onnx_path, pytorch_weights)
        print("✅ ONNX predictor loaded")
        gc.collect()

    return predictor


# -------------------------------------------------
# Load PyTorch model for GradCAM
# -------------------------------------------------
def get_gradcam_model():
    global gradcam_model, target_layer, preprocess
    
    if gradcam_model is None:
        print("🔧 Loading PyTorch model for GradCAM...")

        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        pth_path = os.path.join(models_dir, "thar_wrangler_mobilenetv2.pth")

        # Recreate MobileNetV2 backbone
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

        # Load state_dict
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Target last convolution layer
        target_layer = model.features[-1]

        # Build GradCAM wrapper
        gradcam_model = GradCAM(model, target_layer)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("✅ GradCAM model loaded")
        gc.collect()

    return gradcam_model


# -------------------------------------------------
# Root endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Backend Running!", "status": "healthy"}


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "gradcam_loaded": gradcam_model is not None
    }


# -------------------------------------------------
# Prediction API (ONNX)
# -------------------------------------------------
async def _handle_predict_async(image: UploadFile):
    try:
        pred = get_predictor()
        image_bytes = await image.read()
        
        result = pred.predict_from_bytes(image_bytes)

        del image_bytes
        gc.collect()
        return result

    except Exception as e:
        print("🔥 Prediction error:", e)
        gc.collect()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    return await _handle_predict_async(image)


# -------------------------------------------------
# GRADCAM API (PyTorch)
# -------------------------------------------------
@app.post("/api/gradcam")
async def gradcam_api(image: UploadFile = File(...)):
    try:
        model = get_gradcam_model()

        # Read image bytes
        image_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess for MobileNetV2
        tensor = preprocess(pil_img).unsqueeze(0)

        # Generate CAM
        cam_map, pred_class = model(tensor)

        class_names = ["thar", "wrangler"]

        # Overlay heatmap
        heatmap = overlay_heatmap(pil_img, cam_map)
        encoded = encode_image(heatmap)

        return {
            "class": class_names[pred_class],
            "gradcam": encoded
        }

    except Exception as e:
        print("🔥 GradCAM error:", e)
        gc.collect()
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------
# Startup & Shutdown
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting (lazy load enabled)")
    gc.collect()

@app.on_event("shutdown")
async def shutdown_event():
    global predictor, gradcam_model
    predictor = None
    gradcam_model = None
    gc.collect()
    print("👋 Server shutting down...")
