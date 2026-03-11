#!/usr/bin/env python3
"""
Real-time fall and pre-fall detection.

Default behaviour (no arguments)
---------------------------------
Auto-detects available camera hardware:
  • Kinect sensor  → depth + RGB from the sensor itself
  • 3 USB cameras  → triple-angle 2D detection
  • 1 USB camera   → single-camera 2D detection

Video file mode
---------------
    python run.py data/videos/fall-01-cam0.mp4 --split-rgb
    python run.py data/videos/fall-01-cam0.mp4 --split-rgb --show-depth --save out.mp4

    With --split-rgb --show-depth the output is a 3-panel layout:
        [ depth (raw) | RGB + skeleton + HUD | DA-V2 depth ]

Webcam / multi-camera
---------------------
    python run.py               # auto-detect hardware
    python run.py 0             # force single webcam index 0
    python run.py 0 1 2         # force three-camera indices
    python run.py --no-depth    # disable Depth Anything V2
"""
import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

from src.detector import FallDetector, FallState
from src.video_source import VideoSource
from src.camera_setup import detect_camera_setup, CameraConfig, CameraMode

# State → overlay colour (BGR)
_STATE_COLOUR = {
    FallState.STANDING: (0,   200, 0),
    FallState.PREFALL:  (0,   165, 255),
    FallState.FALLEN:   (0,   0,   255),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time fall / pre-fall detector — auto-detects camera hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("source", nargs="*", default=[],
                   help="Camera index(es) or video file path "
                        "(default: auto-detect hardware)")
    p.add_argument("--camera-mode", choices=["kinect", "triple", "single"],
                   help="Force camera mode instead of auto-detecting")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless; print state changes to stdout")
    p.add_argument("--save", metavar="OUT.mp4",
                   help="Save annotated output to file")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier for video files (default: 1.0)")
    p.add_argument("--split-rgb", action="store_true",
                   help="Crop right half — UR dataset videos are depth|RGB side-by-side")
    p.add_argument("--show-depth", action="store_true",
                   help="Append Depth Anything V2 panel alongside the annotated frame")
    p.add_argument("--no-depth", action="store_true",
                   help="Disable Depth Anything V2 (2D detection only)")
    p.add_argument("--depth-model",
                   default="depth-anything/Depth-Anything-V2-Small-hf",
                   help="HuggingFace model ID (default: DA-V2-Small-hf)")
    p.add_argument("--depth-stride", type=int, default=3,
                   help="Run depth model every N frames (default: 3)")
    p.add_argument("--depth-device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Torch device for depth model (default: cpu)")
    p.add_argument("--fov", type=float, default=70.0,
                   help="Assumed horizontal FOV in degrees for 3D lifting (default: 70)")
    return p.parse_args()


def _resolve_source(raw: str):
    """Convert a CLI source string to int (camera index) or str (file path)."""
    try:
        return int(raw)
    except ValueError:
        return raw


# ---------------------------------------------------------------------------
# Frame annotation helpers
# ---------------------------------------------------------------------------

def _colorise_depth(depth_map: Optional[np.ndarray],
                    h: int, w: int, label: str) -> np.ndarray:
    """Return a labelled BGR depth panel of size (h, w)."""
    if depth_map is None:
        panel = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        u8 = (depth_map * 255).clip(0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        if panel.shape[:2] != (h, w):
            panel = cv2.resize(panel, (w, h))
    cv2.putText(panel, label, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _depth_panel_from_raw(raw_left: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert the raw depth half (e.g. from a split UR dataset frame) to a colourised panel."""
    grey = cv2.cvtColor(raw_left, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return _colorise_depth(grey, h, w, "depth (raw)")


def _stamp(frame: np.ndarray, frame_num: int, elapsed: float, state: FallState,
           label: str = ""):
    """Burn frame counter, wall clock, and state badge into the bottom-left corner."""
    h = frame.shape[0]
    colour = _STATE_COLOUR[state]
    text = f"  {state.name}  "
    if label:
        text = f"  {label}: {state.name}  "
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cv2.rectangle(frame, (0, h - th - bl - 6), (tw + 4, h), colour, -1)
    cv2.putText(frame, text, (2, h - bl - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    ts = f"f{frame_num:04d}  {elapsed:.2f}s"
    cv2.putText(frame, ts, (tw + 10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Single-camera loop (also used for video file playback)
# ---------------------------------------------------------------------------

def _run_single(src, args, depth_est=None, cfg: Optional[CameraConfig] = None):
    is_webcam = isinstance(src, int)
    try:
        video = VideoSource(src)
    except RuntimeError as e:
        if is_webcam:
            print(f"Camera {src} not available: {e}", file=sys.stderr)
            print("  Check that the device is connected and accessible.", file=sys.stderr)
        else:
            print(f"Error opening {src!r}: {e}", file=sys.stderr)
        sys.exit(1)

    fps = video.fps
    delay_ms = max(1, int((1000 / fps) / args.speed))
    vw, vh = video.frame_size
    rgb_w = vw // 2 if args.split_rgb else vw

    mode_str = "webcam" if is_webcam else "file"
    print(f"[{mode_str}] {src!r}  {fps:.0f} fps  {vw}x{vh}"
          + (f"  → RGB crop {rgb_w}x{vh}" if args.split_rgb else ""))

    # Panel layout
    if args.show_depth and depth_est is not None:
        n_panels = 3 if args.split_rgb else 2
    else:
        n_panels = 1
    out_w = rgb_w * n_panels

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (out_w, vh))
        print(f"Saving → {args.save}  ({out_w}x{vh}, {n_panels} panel(s))")

    prev_state = None
    frame_num = 0
    t0 = time.time()

    with video, FallDetector(depth_estimator=depth_est, fov_h_deg=args.fov) as det:
        for raw_frame in video:
            frame_num += 1
            elapsed = time.time() - t0

            raw_left = raw_frame[:, :raw_frame.shape[1] // 2, :] if args.split_rgb else None
            frame = raw_frame[:, raw_frame.shape[1] // 2:, :] if args.split_rgb else raw_frame

            annotated, state, metrics = det.process(frame)
            _stamp(annotated, frame_num, elapsed, state)

            if state != prev_state:
                tag = f"[{elapsed:6.2f}s | frame {frame_num:5d}] {state.name}"
                if "spine_angle_3d" in metrics:
                    tag += (f"  drop={metrics['com_drop_rate']:+.4f}"
                            f"  depthV={metrics['com_depth_velocity']:.4f}"
                            f"  spine={metrics['spine_angle_3d']:.1f}°")
                print(tag)
                prev_state = state

            if n_panels == 3:
                depth_raw_panel = _depth_panel_from_raw(raw_left, vh, rgb_w)
                da_panel = _colorise_depth(depth_est.last_depth(), vh, rgb_w, "DA-V2 depth")
                output = np.hstack([depth_raw_panel, annotated, da_panel])
            elif n_panels == 2:
                da_panel = _colorise_depth(depth_est.last_depth(), vh, rgb_w, "DA-V2 depth")
                output = np.hstack([annotated, da_panel])
            else:
                output = annotated

            if writer:
                writer.write(output)

            if not args.no_display:
                cv2.imshow("prefall", output)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    det.reset()
                    print("Detector reset.")

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {frame_num} frames  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Triple-camera loop (3 USB cameras side-by-side)
# ---------------------------------------------------------------------------

def _run_triple(indices: list[int], args, depth_est=None):
    caps = []
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Warning: camera {idx} not available — skipping.", file=sys.stderr)
        else:
            caps.append((idx, cap))

    if not caps:
        print("No cameras could be opened.", file=sys.stderr)
        sys.exit(1)

    print(f"[triple] {len(caps)} camera(s) active: {[i for i, _ in caps]}")
    print("  Press q/Esc to quit, r to reset all detectors.")

    # Shared FallDetector per camera
    detectors = [FallDetector(depth_estimator=depth_est, fov_h_deg=args.fov)
                 for _ in caps]
    for d in detectors:
        d.__enter__()

    writer = None
    frame_num = 0
    prev_states = [None] * len(caps)
    t0 = time.time()

    try:
        while True:
            frame_num += 1
            elapsed = time.time() - t0
            panels = []

            for cam_i, ((idx, cap), det, prev_st) in enumerate(
                    zip(caps, detectors, prev_states)):
                ret, frame = cap.read()
                if not ret:
                    h = 480
                    w = 640
                    blank = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAM {idx}: no signal",
                                (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (80, 80, 80), 2)
                    panels.append(blank)
                    continue

                annotated, state, _ = det.process(frame)
                _stamp(annotated, frame_num, elapsed, state, label=f"CAM{idx}")

                if state != prev_states[cam_i]:
                    print(f"[{elapsed:6.2f}s | f{frame_num:5d}] CAM{idx}: {state.name}")
                    prev_states[cam_i] = state

                panels.append(annotated)

            if panels:
                # Resize all panels to the same height before stacking
                target_h = min(p.shape[0] for p in panels)
                resized = []
                for p in panels:
                    if p.shape[0] != target_h:
                        scale = target_h / p.shape[0]
                        nw = int(p.shape[1] * scale)
                        p = cv2.resize(p, (nw, target_h))
                    resized.append(p)
                output = np.hstack(resized)

                if writer is None and args.save:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = caps[0][1].get(cv2.CAP_PROP_FPS) or 30
                    writer = cv2.VideoWriter(
                        args.save, fourcc, fps,
                        (output.shape[1], output.shape[0]))
                    print(f"Saving → {args.save}")

                if writer:
                    writer.write(output)

                if not args.no_display:
                    cv2.imshow("prefall — triple camera", output)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        for d in detectors:
                            d.reset()
                        print("All detectors reset.")
    finally:
        for d in detectors:
            d.__exit__(None, None, None)
        for _, cap in caps:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Done — {frame_num} frames  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Kinect loop
# ---------------------------------------------------------------------------

def _run_kinect(cfg: CameraConfig, args, depth_est=None):
    """
    Run fall detection using a Kinect sensor.

    freenect (Linux): reads colour + depth frames directly.
    pykinect2 (Windows): reads colour + depth frames via the SDK.

    Falls back to USB camera 0 if Kinect frame acquisition fails.
    """
    if cfg.kinect_backend == "freenect":
        _run_kinect_freenect(args, depth_est)
    elif cfg.kinect_backend == "pykinect2":
        _run_kinect_pykinect2(args, depth_est)
    else:
        print("Kinect detected but no supported backend — falling back to USB camera 0.",
              file=sys.stderr)
        _run_single(0, args, depth_est, cfg)


def _run_kinect_freenect(args, depth_est=None):
    """Kinect acquisition via libfreenect."""
    try:
        import freenect  # type: ignore
    except ImportError:
        print("freenect not installed. Run: pip install freenect", file=sys.stderr)
        sys.exit(1)

    print("[kinect] freenect backend — colour + depth at 30 fps")
    print("  Press q/Esc to quit, r to reset detector.")

    frame_num = 0
    prev_state = None
    t0 = time.time()
    writer = None
    out_w = out_h = None

    with FallDetector(depth_estimator=depth_est, fov_h_deg=args.fov) as det:
        def _body(dev, ctx):
            nonlocal frame_num, prev_state, writer, out_w, out_h

            frame_num += 1
            elapsed = time.time() - t0

            # Grab colour and depth frames synchronously
            colour = freenect.sync_get_video()[0]   # (480,640,3) RGB uint8
            depth_raw = freenect.sync_get_depth()[0]  # (480,640) uint16 mm

            bgr = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR)
            annotated, state, metrics = det.process(bgr)
            _stamp(annotated, frame_num, elapsed, state)

            if state != prev_state:
                print(f"[{elapsed:6.2f}s | f{frame_num:5d}] {state.name}")
                prev_state = state

            # Build depth panel from real Kinect depth
            d_norm = (depth_raw.astype(np.float32) / depth_raw.max()
                      if depth_raw.max() > 0 else np.zeros_like(depth_raw, dtype=np.float32))
            h, w = annotated.shape[:2]
            depth_panel = _colorise_depth(d_norm, h, w, "Kinect depth")
            output = np.hstack([depth_panel, annotated])

            if out_w is None:
                out_w, out_h = output.shape[1], output.shape[0]
                if args.save:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, 30, (out_w, out_h))
                    print(f"Saving → {args.save}")

            if writer:
                writer.write(output)

            if not args.no_display:
                cv2.imshow("prefall — Kinect", output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise freenect.Kill
                if key == ord("r"):
                    det.reset()
                    print("Detector reset.")

        try:
            freenect.runloop(body=_body)
        except Exception:
            pass

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"Done — {frame_num} frames  ({elapsed:.1f}s)")


def _run_kinect_pykinect2(args, depth_est=None):
    """Kinect v2 acquisition via PyKinectV2 (Windows)."""
    try:
        from pykinect2 import PyKinectRuntime, PyKinectV2  # type: ignore
        import ctypes
    except ImportError:
        print("pykinect2 not installed. Run: pip install pykinect2", file=sys.stderr)
        sys.exit(1)

    print("[kinect] pykinect2 backend — colour + depth at 30 fps")
    print("  Press q/Esc to quit, r to reset detector.")

    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )

    frame_num = 0
    prev_state = None
    writer = None
    t0 = time.time()

    with FallDetector(depth_estimator=depth_est, fov_h_deg=args.fov) as det:
        try:
            while True:
                if not kinect.has_new_color_frame():
                    time.sleep(0.005)
                    continue

                frame_num += 1
                elapsed = time.time() - t0

                colour_frame = kinect.get_last_color_frame()
                bgr = colour_frame.reshape((1080, 1920, 4))[:, :, :3].copy()
                bgr = cv2.resize(bgr, (960, 540))

                annotated, state, _ = det.process(bgr)
                _stamp(annotated, frame_num, elapsed, state)

                if state != prev_state:
                    print(f"[{elapsed:6.2f}s | f{frame_num:5d}] {state.name}")
                    prev_state = state

                if kinect.has_new_depth_frame():
                    depth_raw = kinect.get_last_depth_frame().reshape((424, 512))
                    d_norm = depth_raw.astype(np.float32) / 8000.0
                    h, w = annotated.shape[:2]
                    depth_panel = _colorise_depth(d_norm, h, w, "Kinect depth")
                    output = np.hstack([depth_panel, annotated])
                else:
                    output = annotated

                if writer is None and args.save:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        args.save, fourcc, 30,
                        (output.shape[1], output.shape[0]))
                    print(f"Saving → {args.save}")

                if writer:
                    writer.write(output)

                if not args.no_display:
                    cv2.imshow("prefall — Kinect", output)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        det.reset()
                        print("Detector reset.")
        finally:
            kinect.close()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"Done — {frame_num} frames  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve sources from CLI
    raw_sources = args.source

    # Determine if we're in file mode or camera mode
    is_file_mode = (len(raw_sources) == 1 and
                    not raw_sources[0].isdigit() and
                    raw_sources[0] not in ("-",))

    if is_file_mode:
        # Video file — run single-camera pipeline
        src = raw_sources[0]
        use_depth = not args.no_depth
        depth_est = _load_depth(args) if use_depth else None
        _run_single(src, args, depth_est)
        return

    # Camera mode — resolve explicit indices or auto-detect
    if raw_sources:
        explicit_indices = [int(s) for s in raw_sources if s.isdigit()]
    else:
        explicit_indices = []

    if explicit_indices:
        n = len(explicit_indices)
        if n >= 3:
            forced_mode: CameraMode = "triple"
        else:
            forced_mode = "single"
        cfg = CameraConfig(mode=forced_mode, camera_indices=explicit_indices)
    else:
        # Full auto-detect
        cfg = detect_camera_setup(
            force_mode=args.camera_mode  # type: ignore[arg-type]
        )

    print(f"Camera setup: {cfg.summary()}")

    # Depth estimator — not used in triple-camera mode (too slow for 3 streams)
    use_depth = (not args.no_depth) and (cfg.mode != "triple")
    depth_est = _load_depth(args) if use_depth else None

    if cfg.mode == "kinect":
        _run_kinect(cfg, args, depth_est)
    elif cfg.mode == "triple":
        _run_triple(cfg.camera_indices, args, depth_est)
    else:
        # Single camera
        src = cfg.camera_indices[0] if cfg.camera_indices else 0
        _run_single(src, args, depth_est, cfg)


def _load_depth(args):
    """Load and return the DepthEstimator, or None if loading fails."""
    try:
        from src.depth_estimator import DepthEstimator
        est = DepthEstimator(
            model_id=args.depth_model,
            device=args.depth_device,
            stride=args.depth_stride,
        )
        est.load()
        return est
    except Exception as e:
        print(f"Warning: could not load depth model ({e}). Running 2D only.",
              file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
