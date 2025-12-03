import io
import traceback
import torch
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
import numpy as np
import gc


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
        # (Disabled to save RAM)

        self.class_names = ["thar", "wrangler"]

        # ------------ REPLACEMENT FOR TORCHVISION TRANSFORMS ------------
        # (lightweight + same output)
        def preprocess(pil_img):
            pil_img = pil_img.resize((224, 224))
            img_np = np.array(pil_img).astype(np.float32)

            img_np = img_np / 255.0
            img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            # HWC → CHW
            img_np = np.transpose(img_np, (2, 0, 1))

            return torch.from_numpy(img_np).unsqueeze(0)

        self.preprocess = preprocess

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

            # --------- USE NEW LIGHTWEIGHT PREPROCESS ---------
            x = self.preprocess(img)

            # ONNX inference
            probs = self.onnx_predict(x)

            confidence_tensor, idx_tensor = torch.max(probs, dim=1)
            confidence_val = float(confidence_tensor.item())
            class_name = self.class_names[int(idx_tensor.item())]

            del probs, confidence_tensor, idx_tensor
            gc.collect()

            # ---------- GRADCAM DISABLED ----------
            gradcam_b64 = None

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
