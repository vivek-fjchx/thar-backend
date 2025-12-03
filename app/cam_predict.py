import io
import gc
import traceback
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import base64


class CamPredictor:
    def __init__(self, onnx_path: str):

        # Lightweight ONNX runtime
        self.sess_options = ort.SessionOptions()
        self.sess_options.enable_cpu_mem_arena = False
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=self.sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Class names
        self.class_names = ["thar", "wrangler"]

    # -------------------------------
    # Helper: encode numpy image to base64 PNG
    # -------------------------------
    def _encode_img(self, img_np):
        success, png = cv2.imencode(".png", img_np)
        if not success:
            raise RuntimeError("Failed to encode heatmap image")
        return base64.b64encode(png).decode("utf-8")

    # -------------------------------
    # Helper: resize + normalize manually (no torchvision)
    # -------------------------------
    def preprocess(self, pil_img):
        pil_img = pil_img.resize((224, 224))
        img_np = np.array(pil_img).astype(np.float32)

        # Normalize to torchvision ImageNet mean/std
        img_np = img_np / 255.0
        img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # HWC → CHW
        img_np = np.transpose(img_np, (0, 1, 2))
        img_np = np.transpose(img_np, (2, 0, 1))

        # Add batch dim
        return img_np[np.newaxis, :, :, :].astype(np.float32)

    # -------------------------------
    # Generate ONNX CAM
    # -------------------------------
    def generate_cam(self, image_bytes: bytes):
        try:
            # Load image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.preprocess(img)

            # Determine input + output names
            input_name = self.session.get_inputs()[0].name

            # IMPORTANT:
            # Your ONNX model must have 2 outputs:
            # 1. Last conv feature map
            # 2. Logits
            output_names = [o.name for o in self.session.get_outputs()]
            conv_name = output_names[0]
            logits_name = output_names[1]

            # Run ONNX inference to get both
            conv_out, logits = self.session.run(
                [conv_name, logits_name],
                {input_name: input_tensor}
            )

            # Convert logits → probabilities
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)
            pred_idx = int(np.argmax(probs))

            # ---------------------------
            # CAM computation (no gradients)
            # ---------------------------
            # conv_out: (1, C, H, W)
            feature_maps = conv_out[0]  # (C, H, W)

            # logits: (1, num_classes)
            weights = logits[0]  # (num_classes)

            # Weighted sum of conv channels
            cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            for c in range(feature_maps.shape[0]):
                cam += feature_maps[c] * weights[c]

            # Normalize
            cam = np.maximum(cam, 0)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-9)
            cam = (cam * 255).astype(np.uint8)

            # Resize to original image size
            cam = cv2.resize(cam, (img.width, img.height))

            # Apply color map
            cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

            # Overlay (RGB → BGR needed for OpenCV)
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_np, 0.55, cam_color, 0.45, 0)

            # Encode as base64
            cam_b64 = self._encode_img(overlay)

            gc.collect()

            return {
                "class": self.class_names[pred_idx],
                "gradcam": cam_b64
            }

        except Exception as e:
            print("CAM ERROR:", e)
            traceback.print_exc()
            raise
