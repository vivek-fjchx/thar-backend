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
        # ONNX Runtime Session (Memory Optimized)
        # -----------------------------
        # Configure ONNX Runtime for minimal memory usage
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = False  # Disable memory arena to reduce peak memory
        
        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # -----------------------------
        # PyTorch model (GradCAM only)
        # -----------------------------
        # Load with minimal memory footprint
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)

        # Load weights with map_location to avoid duplicating in memory
        state_dict = torch.load(pytorch_weights, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Free state_dict immediately
        del state_dict
        import gc
        gc.collect()

        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Use torch.no_grad for inference to save memory
        for param in self.model.parameters():
            param.requires_grad = False

        # Target layer for GradCAM
        self.target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

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
    # Prediction with ONNX Runtime
    # -----------------------------------------
    def onnx_predict(self, x):
        # Convert to numpy and free torch tensor immediately
        x_np = x.cpu().numpy()
        del x
        import gc
        gc.collect()
        
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: x_np
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # Free input immediately
        del x_np, ort_inputs
        gc.collect()
        
        logits = torch.tensor(ort_outs[0])
        probs = torch.softmax(logits, dim=1)
        
        # Free intermediate tensors
        del ort_outs, logits
        gc.collect()
        
        return probs

    # -----------------------------------------
    # Main API call
    # -----------------------------------------
    def predict_from_bytes(self, image_bytes):
        import gc
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Free image_bytes immediately
        del image_bytes
        gc.collect()
        
        x = self.transform(img).unsqueeze(0)

        # ---------- ONNX inference ----------
        # x is passed by reference, so deleting it inside onnx_predict 
        # only deletes the local parameter, not the original
        probs = self.onnx_predict(x)
        confidence, idx = torch.max(probs, dim=1)
        class_name = self.class_names[idx.item()]
        
        # Free probs after extracting values
        confidence_val = float(confidence.item())
        del probs, confidence, idx
        gc.collect()

        # ---------- GradCAM (PyTorch) ----------
        # Use original x for GradCAM
        x_cam = x.to(self.device)
        del x
        gc.collect()
        
        # GradCAM needs gradients, so we can't use no_grad()
        # But we'll clean up immediately after
        cam_map, _ = self.gradcam(x_cam)
        
        del x_cam
        gc.collect()
        
        overlay = overlay_heatmap(img, cam_map)
        del img, cam_map
        gc.collect()
        
        gradcam_image_base64 = encode_image(overlay)
        del overlay
        gc.collect()

        return {
            "class": class_name,
            "confidence": confidence_val,
            "gradcam": gradcam_image_base64
        }
