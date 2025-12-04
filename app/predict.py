import io
import traceback
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import gc


class Predictor:

    def __init__(self, onnx_path: str, pytorch_weights: str = None):

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

        self.class_names = ["thar", "wrangler"]

    # --------------------------------------------------------
    # Preprocess — OpenCV fast version (pure numpy)
    # --------------------------------------------------------
    def preprocess(self, pil_img):

        # PIL → NumPy
        img_np = np.array(pil_img)

        # Resize
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize
        img_np = img_np.astype(np.float32) / 255.0
        img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # HWC → CHW
        img_np = np.transpose(img_np, (2, 0, 1))

        # Add batch dimension → (1, 3, 224, 224)
        return img_np[np.newaxis, :, :, :]

    # --------------------------------------------------------
    # ONNX runtime forward pass — pure numpy
    # --------------------------------------------------------
    def onnx_predict(self, x_np):

        try:
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: x_np}

            ort_outs = self.ort_session.run(None, ort_inputs)

            logits = ort_outs[0]    # shape (1,2)

            # Softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp, axis=1, keepdims=True)

            return probs

        except Exception as e:
            print("ERROR in ONNX predict:", e)
            traceback.print_exc()
            raise

    # --------------------------------------------------------
    # Main public API — pure numpy
    # --------------------------------------------------------
    def predict_from_bytes(self, image_bytes):

        try:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid image.")

            del image_bytes
            gc.collect()

            # Preprocess
            x_np = self.preprocess(img)

            # ONNX inference
            probs = self.onnx_predict(x_np)

            # Argmax
            idx = int(np.argmax(probs, axis=1)[0])
            confidence_val = float(probs[0][idx])
            class_name = self.class_names[idx]

            del probs
            gc.collect()

            # GradCAM disabled
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
