import io
import traceback
import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import gc

# from app.gradcam import GradCAM, overlay_heatmap, encode_image    # ← DISABLED


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

        # ---------- REMOVE PYTORCH MODEL & GRADCAM ----------
        # self.model = ...
        # self.gradcam = ...
        # self.target_layer = ...
        # (Disabled to save ~300–400MB memory)

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
    # Main prediction entry
    # ----------------------------------------
    def predict_from_bytes(self, image_bytes):
        try:
            # Load image
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid image.")

            del image_bytes
            gc.collect()

            x = self.transform(img).unsqueeze(0)

            # ONNX inference
            probs = self.onnx_predict(x)

            confidence_tensor, idx_tensor = torch.max(probs, dim=1)
            confidence_val = float(confidence_tensor.item())
            class_name = self.class_names[int(idx_tensor.item())]

            del probs, confidence_tensor, idx_tensor
            gc.collect()

            # ---------- GRADCAM DISABLED ----------
            gradcam_b64 = None
            # try:
            #     x_cam = x.to(self.device)
            #     x_cam.requires_grad = True
            #     cam_map, _ = self.gradcam(x_cam)
            #     overlay = overlay_heatmap(img, cam_map)
            #     gradcam_b64 = encode_image(overlay)
            # except Exception as e:
            #     print("GradCAM disabled / failed:", e)
            # gradcam_b64 = None

            del img
            gc.collect()

            return {
                "class": class_name,
                "confidence": confidence_val,
                "gradcam": gradcam_b64,
            }

        except Exception as e:
            print("ERROR in predict_from_bytes:", e)
            traceback.print_exc()
            raise
