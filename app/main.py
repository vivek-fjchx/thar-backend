from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=[
        "https://thar-frontend-5vtq.vercel.app",  # Your production frontend
        "http://localhost:3000",                   # Local development
        "*"                                        # Allow all (use cautiously in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly include OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],  # Add this to expose headers to the frontend
    max_age=3600,          # Cache preflight requests for 1 hour
)


# -------------------------------
# Load the Thar vs Wrangler model
# (Loaded only once at startup)
# -------------------------------
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "thar_wrangler_mobilenetv2.pth"
)
model_path = os.path.abspath(model_path)

predictor = Predictor(model_path)


# -------------------------------
# OPTIONS endpoint for preflight
# (Add this to handle CORS preflight explicitly)
# -------------------------------
@app.options("/api/predict")
async def predict_options():
    return {"message": "OK"}


# -------------------------------
# Prediction API
# -------------------------------
@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        response = predictor.predict_from_bytes(image_bytes)
        return response

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return {"error": str(e)}


# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Backend Running!"}
