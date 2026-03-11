"""
Depth Anything V2 monocular depth estimator.

Returns per-frame disparity maps (float32, [0,1], 1=closest) resized
to the input frame dimensions.  Runs every `stride` frames for speed;
the cached result is returned on skipped frames via last_depth().
"""
import numpy as np


class DepthEstimator:
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
        stride: int = 3,
    ):
        self.model_id = model_id
        self.device = device
        self.stride = stride

        self._processor = None
        self._model = None
        self._loaded = False
        self._last_depth: np.ndarray | None = None
        self._frame_count = 0

    def load(self):
        """Eagerly load model weights. Called automatically on first process() if not done."""
        if self._loaded:
            return
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        print(f"Loading depth model: {self.model_id} on {self.device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch
        self._loaded = True
        print("Depth model ready.")

    def process(self, frame_bgr: np.ndarray, frame_idx: int | None = None) -> np.ndarray | None:
        """
        Run depth estimation on frame_bgr.

        Returns float32 H×W disparity map (1=closest, 0=farthest), or None if
        this frame is skipped due to stride.  Always call last_depth() to get
        the most recent result regardless.

        Args:
            frame_bgr: BGR uint8 numpy array.
            frame_idx: optional external frame counter; if None, internal counter is used.
        """
        if not self._loaded:
            self.load()

        idx = frame_idx if frame_idx is not None else self._frame_count
        self._frame_count += 1

        if idx % self.stride != 0:
            return None

        import cv2
        from PIL import Image
        import torch

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H', W')

        # Resize to original frame dimensions
        depth_up = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth_np = depth_up.cpu().numpy().astype(np.float32)

        # Normalise to [0,1]; DA-V2 outputs disparity so high value = close = 1.0
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)

        self._last_depth = depth_norm
        return depth_norm

    def last_depth(self) -> np.ndarray | None:
        """Return the most recently computed depth map (may be from a prior frame)."""
        return self._last_depth

    def reset(self):
        self._last_depth = None
        self._frame_count = 0

    def colorise(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert float32 depth map to BGR uint8 via INFERNO colormap."""
        import cv2
        u8 = (depth_map * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
