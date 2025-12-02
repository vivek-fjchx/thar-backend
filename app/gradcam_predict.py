import io
import traceback
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import gc

from app.gradcam import GradCAM, overlay_heatmap, encode_image


class GradCAMPredictor:
    def __init__(self, pytorch_weights: str):
        self.device = torch.device("cpu")

        # Load MobileNetV2 only for GradCAM
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)

        state = torch.load(pytorch_weights, map_location=self.device)
        self.model.load_state_dict(state)
        del state
        gc.collect()

        self.model.to(self.device)
        self.model.eval()

        self.target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

        self.class_names = ["thar", "wrangler"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def generate_gradcam(self, image_bytes):
        try:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Not a valid image.")

            x = self.transform(img).unsqueeze(0).to(self.device)
            x.requires_grad = True

            cam_map, pred_idx = self.gradcam(x)

            overlay = overlay_heatmap(img, cam_map)
            gradcam_b64 = encode_image(overlay)

            # Predicted class (optional if you want it)
            pred_class = self.class_names[int(pred_idx)]

            del img, x, cam_map
            gc.collect()

            return {
                "class": pred_class,
                "gradcam": gradcam_b64
            }

        except Exception as e:
            print("GradCAM ERROR:", e)
            traceback.print_exc()
            raise
