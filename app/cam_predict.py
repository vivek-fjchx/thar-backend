import io
import gc
import traceback
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import torch
import cv2
import base64


class CampPredictor:
    def __init__(self, onnx_path: str):
        # Load ONNX model
        self.sess_options = ort.SessionOptions()
        self.sess_options.enable_cpu_mem_arena = False
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=self.sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.class_names = ["thar", "wrangler"]

    def _encode_img(self, img_np):
        _, png = cv2.imencode(".png", img_np)
        return base64.b64encode(png).decode("utf-8")

    def generate_cam(self, image_bytes: bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inp = self.transform(img).unsqueeze(0).numpy()

            # Run ONNX inference with intermediate outputs exposed
            input_name = self.session.get_inputs()[0].name
            output_names = [o.name for o in self.session.get_outputs()]
            conv_name = output_names[0]      # last conv feature maps
            fc_name = output_names[1]        # logits

            conv_out, logits = self.session.run([conv_name, fc_name],
                                                {input_name: inp.astype(np.float32)})

            # Prediction
            probs = torch.softmax(torch.tensor(logits), dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

            # FC weights (class × channels)
            fc_weights = self.session.get_outputs()[1].shape[1]

            # CAM = weighted sum(feature_maps × fc_weights)
            weights = logits[0]
            heatmap = np.maximum(np.dot(conv_out[0].transpose(1,2,0), weights), 0)

            # Normalize 0–255
            heatmap -= heatmap.min()
            heatmap /= heatmap.max()
            heatmap = (heatmap * 255).astype(np.uint8)

            # Resize to original image size
            heatmap = cv2.resize(heatmap, (img.width, img.height))

            # Apply color map
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Overlay with transparency
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)

            # Encode PNG
            heatmap_b64 = self._encode_img(overlay)

            gc.collect()

            return {
                "class": self.class_names[pred_idx],
                "gradcam": heatmap_b64
            }

        except Exception as e:
            print("CAM ERROR:", e)
            traceback.print_exc()
            raise
