import io
import traceback
import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import gc
import time
import cv2  # for image stats
import numpy as np
from PIL import Image, ImageStat

def compute_image_stats(pil_image):
    """Return heuristic image stats"""
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Brightness (average pixel intensity)
    brightness = np.mean(gray)
    if brightness < 60:
        brightness_label = "Dark"
    elif brightness < 120:
        brightness_label = "Normal"
    else:
        brightness_label = "Bright"

    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 70:
        sharpness_label = "Blurry"
    elif laplacian_var < 150:
        sharpness_label = "Normal"
    else:
        sharpness_label = "Sharp"

    # Contrast (RMS contrast)
    contrast_val = gray.std()
    if contrast_val < 50:
        contrast_label = "Low"
    elif contrast_val < 100:
        contrast_label = "Medium"
    else:
        contrast_label = "High"

    # Visibility: simple heuristic based on edge density in central area
    h, w = gray.shape
    central = gray[h//4: 3*h//4, w//4: 3*w//4]
    edges = cv2.Canny(central, 50, 150)
    edge_density = edges.sum() / (central.size)
    if edge_density > 0.05:
        visibility_label = "High"
    elif edge_density > 0.02:
        visibility_label = "Medium"
    else:
        visibility_label = "Low"

    return {
        "brightness": brightness_label,
        "sharpness": sharpness_label,
        "contrast": contrast_label,
        "visibility": visibility_label
    }



class Predictor:
    def __init__(self, onnx_path: str, pytorch_weights: str = None):
        self.device = torch.device("cpu")

        # ---------- ONNX session ----------
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = False

            self.ort_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            print("ERROR creating ONNX session:", e)
            traceback.print_exc()
            raise

        self.class_names = ["thar", "wrangler"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])


    # ----------------------------------------
    # ONNX inference
    # ----------------------------------------
    def onnx_predict(self, x: torch.Tensor):
        try:
            x_np = x.cpu().numpy().astype(np.float32, copy=False)
            input_name = self.ort_session.get_inputs()[0].name

            ort_inputs = {input_name: x_np}
            ort_outs = self.ort_session.run(None, ort_inputs)

            logits = torch.from_numpy(ort_outs[0])
            probs = torch.softmax(logits, dim=1)

            del logits, ort_outs
            gc.collect()

            return probs
        except Exception as e:
            print("ERROR in ONNX predict:", e)
            traceback.print_exc()
            raise


    # ----------------------------------------
    # Main prediction
    # ----------------------------------------
    def predict_from_bytes(self, image_bytes):        
        try:
            import time
            start_time = time.time()
    
            # ---- Load image ----
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid image.")
            
            del image_bytes
            gc.collect()
    
            # ---- Image stats ----
            image_stats = compute_image_stats(img)
    
            # ---- Preprocess ----
            x = self.transform(img).unsqueeze(0)
    
            # ---- ONNX inference ----
            probs_tensor = self.onnx_predict(x)
    
            confidence_tensor, idx_tensor = torch.max(probs_tensor, dim=1)
            confidence_val = float(confidence_tensor.item())
            class_name = self.class_names[int(idx_tensor.item())]
    
            # Per-class probabilities
            probabilities = {
                "thar": float(probs_tensor[0, 0].item()),
                "wrangler": float(probs_tensor[0, 1].item())
            }
    
            # ---- Certainty (margin) ----
            margin = abs(probabilities["thar"] - probabilities["wrangler"])
            if margin > 0.4:
                certainty = "High"
            elif margin > 0.2:
                certainty = "Medium"
            else:
                certainty = "Low"
    
            latency_ms = round((time.time() - start_time) * 1000, 2)
    
            # ---- Clean up tensors ----
            del probs_tensor, confidence_tensor, idx_tensor, x
            gc.collect()
    
            del img
            gc.collect()
    
            return {
                "class": class_name,
                "confidence": confidence_val,
                "probabilities": probabilities,
                "certainty": certainty,
                "imageStats": image_stats,
                "system": {
                    "latency_ms": latency_ms,
                    "modelName": "MobileNetV2"
                },
                "gradcam": None
            }
    
        except Exception as e:
            print("ERROR in predict_from_bytes:", e)
            traceback.print_exc()
            raise
