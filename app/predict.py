import io
import traceback
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
import numpy as np
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

        # ---------- LIGHTWEIGHT PREPROCESS (pure numpy) ----------
        def preprocess(pil_img):
            pil_img = pil_img.resize((224, 224))
            img_np = np.array(pil_img).astype(np.float32)

            img_np = img_np / 255.0
            img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            # HWC → CHW
            img_np = np.transpose(img_np, (2, 0, 1))

            # Add batch dimension
            return img_np[np.newaxis, :, :, :]   # shape (1,3,224,224)

        self.preprocess = preprocess

    # ----------------------------------------
    # ONNX inference (pure numpy)
    # ----------------------------------------
    def onnx_predict(self, x_np):
        try:
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: x_np}
            ort_outs = self.ort_session.run(None, ort_inputs)

            logits = ort_outs[0]   # numpy array (1,2)

            # softmax manually
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp, axis=1, keepdims=True)

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
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid image.")

            del image_bytes
            gc.collect()

            # preprocess → numpy
            x_np = self.preprocess(img)

            # ONNX inference → numpy
            probs = self.onnx_predict(x_np)

            # numpy argmax
            idx = int(np.argmax(probs, axis=1)[0])
            confidence_val = float(probs[0][idx])
            class_name = self.class_names[idx]

            del probs
            gc.collect()

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
