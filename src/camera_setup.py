"""
Camera hardware auto-detection.

Priority order
--------------
1. Microsoft Kinect  — real-time depth + colour from a single sensor.
   Requires libfreenect (Linux) or pykinect2 (Windows).
2. Triple USB cameras — three webcams for multi-angle 2D detection.
3. Single USB camera  — one webcam, 2D detection only.

Usage
-----
    from src.camera_setup import detect_camera_setup, CameraMode

    cfg = detect_camera_setup()
    print(cfg.mode, cfg.camera_indices)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2


CameraMode = Literal["kinect", "triple", "single"]


@dataclass
class CameraConfig:
    """Describes the detected camera hardware."""
    mode: CameraMode
    # USB VideoCapture indices that were found open (may be empty for Kinect-only)
    camera_indices: list[int] = field(default_factory=list)
    has_kinect: bool = False
    # "freenect" | "pykinect2" | None
    kinect_backend: str | None = None

    def summary(self) -> str:
        if self.mode == "kinect":
            return (f"Kinect sensor detected (backend: {self.kinect_backend})"
                    + (f"  +  {len(self.camera_indices)} USB camera(s)"
                       if self.camera_indices else ""))
        if self.mode == "triple":
            return f"Triple-camera mode  (indices: {self.camera_indices})"
        n = len(self.camera_indices)
        return f"Single-camera mode  (index: {self.camera_indices[0] if n else 'none'})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_cameras(max_index: int = 8) -> list[int]:
    """Return indices of USB cameras that respond to VideoCapture.open()."""
    available: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def _detect_kinect() -> tuple[bool, str | None]:
    """
    Try to detect a Kinect sensor.

    Returns (found: bool, backend: str | None).
    backend is "freenect" (Linux/macOS) or "pykinect2" (Windows).
    """
    # --- freenect (Linux / macOS) ---
    try:
        import freenect  # type: ignore
        ctx = freenect.init()
        if ctx is not None and freenect.num_devices(ctx) > 0:
            freenect.shutdown(ctx)
            return True, "freenect"
        if ctx is not None:
            freenect.shutdown(ctx)
    except Exception:
        pass

    # --- pykinect2 (Windows / Kinect v2) ---
    try:
        from pykinect2 import PyKinectV2  # type: ignore
        _ = PyKinectV2.FrameSourceTypes_Color  # just importing is enough to probe
        return True, "pykinect2"
    except Exception:
        pass

    return False, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_camera_setup(force_mode: CameraMode | None = None) -> CameraConfig:
    """
    Auto-detect the best available camera configuration.

    Parameters
    ----------
    force_mode
        Override auto-detection ("kinect", "triple", or "single").

    Returns
    -------
    CameraConfig
        mode        : "kinect" | "triple" | "single"
        camera_indices : list of USB camera indices found
        has_kinect  : True if a Kinect sensor was confirmed
        kinect_backend : "freenect" | "pykinect2" | None
    """
    usb_cameras = _probe_cameras()

    if force_mode is not None:
        return CameraConfig(mode=force_mode, camera_indices=usb_cameras)

    # 1. Kinect?
    has_kinect, backend = _detect_kinect()
    if has_kinect:
        return CameraConfig(
            mode="kinect",
            camera_indices=usb_cameras,
            has_kinect=True,
            kinect_backend=backend,
        )

    # 2. Three or more USB cameras?
    if len(usb_cameras) >= 3:
        return CameraConfig(mode="triple", camera_indices=usb_cameras[:3])

    # 3. Fallback — single camera
    return CameraConfig(mode="single", camera_indices=usb_cameras[:1])
