import io
import torch
from torchvision import models, transforms
from PIL import Image

from app.gradcam import GradCAM, overlay_heatmap, encode_image


class Predictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- Load MobileNetV2 ----------
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)

        # Load pretrained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        # ---------- Target layer for GradCAM ----------
        # MobileNetV2 last feature layer is features[-1]
        self.target_layer = self.model.features[-1]

        # Create GradCAM once at startup
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Class names
        self.class_names = ["thar", "wrangler"]

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict_from_bytes(self, image_bytes):

        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        confidence, idx = torch.max(probs, dim=1)
        class_name = self.class_names[idx.item()]

        # GradCAM (runs forward+backward internally)
        cam_map, _ = self.gradcam(x)

        # Overlay heatmap on original image
        overlay = overlay_heatmap(img, cam_map)

        # Encode final GradCAM image
        gradcam_image_base64 = encode_image(overlay)

        return {
            "class": class_name,
            "confidence": float(confidence.item()),
            "gradcam": gradcam_image_base64
        }
