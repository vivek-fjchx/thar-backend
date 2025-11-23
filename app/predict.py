import io
import torch
import onnxruntime as ort
from torchvision import models, transforms
from PIL import Image
import numpy as np

from app.gradcam import GradCAM, overlay_heatmap, encode_image


class Predictor:
    def __init__(self, onnx_path: str, pytorch_weights: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------
        # ONNX Runtime Session  (Fast, Low RAM)
        # -----------------------------
        self.ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

        # -----------------------------
        # PyTorch model (Only for GradCAM)
        # -----------------------------
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)

        # Load the .pth only for GradCAM
        state_dict = torch.load(pytorch_weights, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Target conv layer for GradCAM
        self.target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Class names
        self.class_names = ["thar", "wrangler"]

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # -----------------------------------------
    # ONNX Prediction (Fast Inference)
    # -----------------------------------------
    def onnx_predict(self, x):
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: x.cpu().numpy()
        }

        ort_outs = self.ort_session.run(None, ort_inputs)
        logits = torch.tensor(ort_outs[0])
        probs = torch.softmax(logits, dim=1)
        return probs

    # -----------------------------------------
    # Main API call
    # -----------------------------------------
    def predict_from_bytes(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(img).unsqueeze(0)

        # ---------- ONNX inference ----------
        probs = self.onnx_predict(x)
        confidence, idx = torch.max(probs, dim=1)
        class_name = self.class_names[idx.item()]

        # ---------- GradCAM (PyTorch) ----------
        x_cam = x.to(self.device)
        cam_map, _ = self.gradcam(x_cam)
        overlay = overlay_heatmap(img, cam_map)
        gradcam_image_base64 = encode_image(overlay)

        return {
            "class": class_name,
            "confidence": float(confidence.item()),
            "gradcam": gradcam_image_base64
        }
