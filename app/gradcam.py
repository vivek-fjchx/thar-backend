import torch
import numpy as np
from PIL import Image
import io
import base64


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # register hooks and keep handles so we can remove them later if needed
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # store activations (keep as-is; we'll detach later)
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is gradients w.r.t. the activation output
        self.gradients = grad_output[0]

    def remove_hooks(self):
        try:
            if self.forward_handle is not None:
                self.forward_handle.remove()
                self.forward_handle = None
        except Exception:
            pass
        try:
            if self.backward_handle is not None:
                self.backward_handle.remove()
                self.backward_handle = None
        except Exception:
            pass

    def generate(self, input_tensor):
        """
        Returns:
          cam_map: numpy 2D array (values in 0..1)
          pred_class: int
        Notes:
          - This function performs a forward and a backward pass.
          - It clears saved activations/gradients afterwards to avoid memory leaks.
        """
        # Ensure model in eval and grads enabled just for this computation
        self.model.zero_grad()
        # forward
        output = self.model(input_tensor)
        pred_class = int(output.argmax(dim=1).item())

        # backward for the predicted class
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1.0

        # backward pass (this will populate self.gradients via the backward hook)
        output.backward(gradient=one_hot)

        # compute weights: global-average-pool of gradients
        if self.gradients is None or self.activations is None:
            # safety fallback
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # move to CPU and detach ASAP to free GPU memory
        grads = self.gradients.detach()
        acts = self.activations.detach()

        # compute channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # shape: [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        cam = torch.relu(cam)

        # normalize per-image
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.ndim == 3:  # batch dimension present
            cam_np = cam_np[0]

        # cleanup references to free memory
        self.gradients = None
        self.activations = None

        # if using CUDA, free cache (harmless on CPU)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return cam_np, pred_class

    # support calling instance directly (predict.py may use self.gradcam(x))
    def __call__(self, input_tensor):
        return self.generate(input_tensor)


def overlay_heatmap(original_pil, cam_map):
    """
    original_pil: PIL.Image (RGB)
    cam_map: 2D numpy array with values in [0..1]
    returns: numpy uint8 array (H,W,3) RGB
    """
    if not isinstance(original_pil, Image.Image):
        original_pil = Image.fromarray(np.uint8(original_pil)).convert("RGB")

    # Normalize cam_map to 0..255 uint8
    heat = np.uint8(np.clip(cam_map, 0.0, 1.0) * 255.0)

    # Convert heat to a simple color map (red -> yellow)
    # Create channels: R = heat, G = heat * 0.6, B = heat * 0.1 (tunable)
    r = heat
    g = np.uint8(heat * 0.6)
    b = np.uint8(heat * 0.1)
    heatmap = np.stack([r, g, b], axis=2)

    # Convert to PIL and resize to original image size
    heatmap_pil = Image.fromarray(heatmap).resize(original_pil.size, resample=Image.BILINEAR)

    # Blend images
    blended = Image.blend(original_pil.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=0.45).convert("RGB")

    return np.array(blended).astype(np.uint8)


def encode_image(img):
    """
    Accepts either a numpy array (H,W,3) or PIL.Image and returns base64 PNG string.
    """
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    else:
        pil_img = img

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
