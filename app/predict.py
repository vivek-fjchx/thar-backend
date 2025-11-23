import io
import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np

# ---- Keep these imports; they won’t be used but stay for future GradCAM re-enable ----
# from app.gradcam import GradCAM, overlay_heatmap, encode_image


class Predictor:
    def __init__(self, onnx_path: str):
        # ONNX CPU session (very lightweight)
        self.ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

        # ---- REMOVE PYTORCH MODEL (RAM heavy) ----
        # self.model = models.mobilenet_v2(weights=None)
        # self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
        # state_dict = torch.load(pytorch_weights, map_location="cpu")
        # self.model.load_state_dict(state_dict)
        # self.model.eval()

        # ---- REMOVE GRADCAM ----
        # self.target_layer = self.model.features[-1]
        # self.gradcam = GradCAM(self.model, self.target_layer)

        # Class names
        self.class_names = ["thar", "wrangler"]

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # -----------------------------------------
    # Prediction using ONNX Runtime
    # -----------------------------------------
    def onnx_predict(self, x):
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: x.numpy()
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        logits = torch.tensor(ort_outs[0])
        probs = torch.softmax(logits, dim=1)
        return probs

    # ------------------------------------------
    # Main API call
    # ------------------------------------------
    def predict_from_bytes(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(img).unsqueeze(0)

        # ---- ONNX PREDICTION ----
        probs = self.onnx_predict(x)
        confidence, idx = torch.max(probs, dim=1)
        class_name = self.class_names[idx.item()]

        # ---- REMOVED GRADCAM ----
        # x_cam = x
        # cam_map, _ = self.gradcam(x_cam)
        # overlay = overlay_heatmap(img, cam_map)
        # gradcam_image_base64 = encode_image(overlay)

        # Return prediction ONLY
        return {
            "class": class_name,
            "confidence": float(confidence.item()),
            "gradcam": None   # <--- kept for API compatibility
        }
