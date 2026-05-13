# Accident Analysis Backend (MobileNetV2 + FastAPI)

## 📌 Overview

This repository contains the backend for the **Accident Analysis Project** (formerly Thar–Wrangler classification).
It provides a REST API that takes an image as input and returns the predicted vehicle class along with confidence.

---

## 🛠️ Tech Stack

* FastAPI (REST API framework)
* MobileNetV2 (CNN model)
* PyTorch (model inference)
* Uvicorn (ASGI server)
* Pillow (image processing)

---

## 📁 Project Structure

```
accident-analysis-backend/
│
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── routes/
│   │   └── predict.py         # API endpoints
│   ├── services/
│   │   └── inference.py       # Model loading & prediction
│   ├── models/
│   │   └── mobilenet.py       # Model architecture
│   ├── utils/
│   │   └── preprocess.py      # Image preprocessing
│   └── config.py              # Configurations (labels, paths)
│
├── weights/
│   └── model.pth              # Trained model weights
│
├── requirements.txt
├── README.md
└── run.py
```

---

## 🔄 API Workflow

```
Client → POST /predict
       → Image Upload
       → Preprocessing (Resize, Normalize)
       → Model Inference (MobileNetV2)
       → Softmax Output
       → JSON Response
```

---

## 🚀 Installation & Setup

### 1. Clone Repository

```
git clone <your-repo-url>
cd accident-analysis-backend
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Server

```
uvicorn app.main:app --reload
```

### 4. Open API Docs

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoint

### POST /predict

#### Request:

* Content-Type: multipart/form-data
* Body: image file

#### Response:

```
{
  "class": "Thar",
  "confidence": 0.94
}
```

---

## 🧠 Model Details

* Architecture: MobileNetV2
* Input Size: 224 × 224
* Output: Softmax probabilities
* Current Classes:

  * Thar
  * Wrangler

---

## 🧩 Future Improvements

* Multi-class vehicle classification (e.g., Swift, SUV, Sedan)
* Accident severity detection module
* Confidence threshold for "Unknown" class
* Docker deployment for offline/mobile usage
* Integration with frontend (Next.js)

---

## 📦 Dependencies

```
fastapi
uvicorn
torch
torchvision
pillow
python-multipart
```

---

## ▶️ Notes

* Ensure `weights/model.pth` is present before running.
* Update class labels in `mobilenet.py` when adding new vehicles.
* Extend preprocessing if needed for real-world robustness.

---

## 📬 Contact

For queries or collaboration, reach out via your project channel.
