import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Forward hook → captures activations
        target_layer.register_forward_hook(self._save_activation)

        # Backward hook → captures gradients
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()

        # Forward
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        output.backward(gradient=one_hot)

        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), pred_class


def overlay_heatmap(original_pil, cam_map):
    # Resize heatmap to original image size
    w, h = original_pil.size

    heatmap = cv2.applyColorMap(np.uint8(cam_map * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (w, h))

    original = np.array(original_pil)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay


def encode_image(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
