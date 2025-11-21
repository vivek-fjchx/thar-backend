from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.predict import Predictor


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()


# -------------------------------
# CORS - Method 1: Middleware (Most permissive for debugging)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,  # Set to False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# CORS - Method 2: Manual headers on every response
# -------------------------------
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# -------------------------------
# Handle OPTIONS preflight for all routes
# -------------------------------
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
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
# Prediction API
# -------------------------------
@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        response = predictor.predict_from_bytes(image_bytes)
        return JSONResponse(
            content=response,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
            }
        )


# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return JSONResponse(
        content={"message": "Backend Running!"},
        headers={"Access-Control-Allow-Origin": "*"}
    )
