import io
import torch
from torchvision import models, transforms
from PIL import Image

from app.gradcam import GradCAM, overlay_heatmap, encode_image


class Predictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet50
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Target layer for GradCAM
        self.target_layer = self.model.layer4[-1]

        # Create GradCAM once at startup
        self.gradcam = GradCAM(self.model, self.target_layer)

        self.class_names = ["thar", "wrangler"]

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

        # GradCAM (this runs forward+backward inside)
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
