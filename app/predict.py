import io
import traceback
import torch
import onnxruntime as ort
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import gc
from fastapi import HTTPException  # only used for clear local raising if needed

from app.gradcam import GradCAM, overlay_heatmap, encode_image


class Predictor:
    def __init__(self, onnx_path: str, pytorch_weights: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- ONNX session creation (defensive) ----------
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = False  # reduce peak memory

            # If onnx_path is invalid this will raise
            self.ort_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            # Keep a clear error message and re-raise so the caller can handle it
            print("ERROR creating ONNX session:", e)
            traceback.print_exc()
            raise

        # ---------- PyTorch model (GradCAM only) ----------
        try:
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
            state_dict = torch.load(pytorch_weights, map_location=self.device)
            self.model.load_state_dict(state_dict)
            # free state_dict
            del state_dict
            gc.collect()

            self.model = self.model.to(self.device)
            self.model.eval()

            # Freeze parameters (GradCAM still works as it computes grads w.r.t activations)
            for param in self.model.parameters():
                param.requires_grad = False

            # target layer for Grad-CAM; keep a try/except to avoid silent crash if wrong
            try:
                self.target_layer = self.model.features[-1]
            except Exception:
                # fallback - last feature block
                self.target_layer = list(self.model.features.children())[-1]
            self.gradcam = GradCAM(self.model, self.target_layer)
        except Exception as e:
            print("ERROR loading PyTorch model for GradCAM:", e)
            traceback.print_exc()
            raise

        # Class names
        self.class_names = ["thar", "wrangler"]

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # ---------------------------
    # ONNX inference wrapper
    # ---------------------------
    def onnx_predict(self, x: torch.Tensor):
        """
        x: torch.Tensor on CPU, shape (N,C,H,W)
        returns: torch.Tensor probabilities on CPU shape (N, num_classes)
        """
        # Ensure CPU, contiguous, float32 numpy
        try:
            x_cpu = x.detach().cpu()
            x_np = x_cpu.numpy()
            # ensure dtype
            if x_np.dtype != np.float32:
                x_np = x_np.astype(np.float32, copy=False)
            # Free torch objects asap
            del x_cpu
            del x
            gc.collect()

            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: x_np}
            ort_outs = self.ort_session.run(None, ort_inputs)

            # Convert output to torch tensor
            logits_np = ort_outs[0]
            # free ort outputs
            del ort_outs
            gc.collect()

            logits = torch.from_numpy(np.asarray(logits_np))
            probs = torch.softmax(logits, dim=1)

            # free logits_np if needed
            del logits_np, logits
            gc.collect()

            return probs
        except Exception as e:
            # log full traceback then re-raise so caller returns a clear error
            print("ERROR during ONNX inference:", e)
            traceback.print_exc()
            raise

    # ---------------------------
    # Main prediction entry
    # ---------------------------
    def predict_from_bytes(self, image_bytes):
        """
        Returns a dict: { "class": str, "confidence": float, "gradcam": base64 str or None }
        """
        try:
            # Load image
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid image.")
            except Exception as e:
                print("Error opening image:", e)
                raise

            # free raw bytes
            try:
                del image_bytes
            except Exception:
                pass
            gc.collect()

            # transform -> torch tensor on CPU
            x = self.transform(img).unsqueeze(0)  # shape (1,C,H,W)

            # ONNX inference (returns probs tensor on CPU)
            probs = self.onnx_predict(x)

            # get predicted class and confidence (float)
            confidence_tensor, idx_tensor = torch.max(probs, dim=1)
            confidence_val = float(confidence_tensor.item())
            class_name = self.class_names[int(idx_tensor.item())]

            # free intermediate tensors
            del probs, confidence_tensor, idx_tensor
            gc.collect()

            # Try to compute Grad-CAM — but keep it optional so failures don't break the endpoint
            gradcam_b64 = None
            try:
                # move input to device for Grad-CAM
                x_cam = x.to(self.device)
                # Ensure requires_grad for input if GradCAM implementation needs it
                x_cam.requires_grad = True

                cam_map, _ = self.gradcam(x_cam)
                del x_cam
                gc.collect()

                overlay = overlay_heatmap(img, cam_map)
                gradcam_b64 = encode_image(overlay)

                # free overlay and cam_map
                del overlay, cam_map
                gc.collect()
            except Exception as e:
                # Log GradCAM failure but continue with result
                print("WARNING: GradCAM failed:", e)
                traceback.print_exc()
                gradcam_b64 = None

            # free image if no longer needed
            try:
                del img
            except Exception:
                pass
            gc.collect()

            return {
                "class": class_name,
                "confidence": confidence_val,
                "gradcam": gradcam_b64
            }

        except Exception as e:
            # log full traceback and raise an exception that the outer FastAPI handler will catch
            print("ERROR in predict_from_bytes:", e)
            traceback.print_exc()
            # raise so FastAPI's handler returns proper status and message
            raise

