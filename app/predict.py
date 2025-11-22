import io
import torch
from torchvision import models, transforms
from PIL import Image
import gc

from app.gradcam import GradCAM, overlay_heatmap, encode_image


class Predictor:
    def __init__(self, model_path: str):
        # Force CPU for memory efficiency on free tier
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # ---------- Load MobileNetV2 ----------
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)

        # Load pretrained weights with map_location
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        del state_dict  # Free memory immediately
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients for all parameters except during GradCAM
        for param in self.model.parameters():
            param.requires_grad = False

        # ---------- Target layer for GradCAM ----------
        self.target_layer = self.model.features[-1]

        # GradCAM created on-demand, not at startup
        self.gradcam = None

        # Class names
        self.class_names = ["thar", "wrangler"]

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        # Force garbage collection after initialization
        gc.collect()
        print("Model loaded successfully")

    def predict_from_bytes(self, image_bytes):
        try:
            # Load image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            x = self.transform(img).unsqueeze(0).to(self.device)

            # Forward pass with no_grad for inference
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, idx = torch.max(probs, dim=1)
            
            class_name = self.class_names[idx.item()]
            conf_value = float(confidence.item())

            # Create GradCAM only when needed
            if self.gradcam is None:
                # Enable gradients temporarily for GradCAM
                for param in self.model.parameters():
                    param.requires_grad = True
                self.gradcam = GradCAM(self.model, self.target_layer)

            # GradCAM (runs forward+backward internally)
            cam_map, _ = self.gradcam(x)

            # Overlay heatmap on original image
            overlay = overlay_heatmap(img, cam_map)

            # Encode final GradCAM image
            gradcam_image_base64 = encode_image(overlay)

            # Clean up tensors and images
            del x, logits, probs, cam_map, overlay, img
            gc.collect()

            return {
                "class": class_name,
                "confidence": conf_value,
                "gradcam": gradcam_image_base64
            }
            
        except Exception as e:
            # Clean up on error
            gc.collect()
            raise Exception(f"Prediction failed: {str(e)}")
    
    def __del__(self):
        """Cleanup when predictor is destroyed"""
        if self.gradcam is not None:
            self.gradcam.remove_hooks()
        gc.collect()
