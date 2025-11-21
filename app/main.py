from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS FIRST (before anything else)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For now allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
from app.predict import Predictor

model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "thar_wrangler_mobilenetv2.pth")
)

predictor = Predictor(model_path)

@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        return predictor.predict_from_bytes(image_bytes)
    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Backend Running!"}
