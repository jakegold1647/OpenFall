"""
3D biomechanical feature extraction.

Lifts 2D MediaPipe landmarks into camera-space 3D using a monocular depth map,
estimates the ground plane, and computes fall-relevant biomechanical features.

All 3D coordinates are in arbitrary scale (depth is relative/disparity-based).
Features are expressed as dimensionless fractions so they are invariant to
frame-to-frame depth scale shifts.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

# MediaPipe landmark indices used in 3D analysis
IDX_NOSE           = 0
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_HIP       = 23
IDX_RIGHT_HIP      = 24
IDX_LEFT_KNEE      = 25
IDX_RIGHT_KNEE     = 26
IDX_LEFT_ANKLE     = 27
IDX_RIGHT_ANKLE    = 28


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------

def build_camera_matrix(
    width: int,
    height: int,
    fov_h_deg: float = 70.0,
) -> tuple[float, float, float, float]:
    """
    Return pinhole intrinsics (fx, fy, cx, cy) for assumed horizontal FOV.
    Square pixels assumed (fx == fy).
    """
    fx = width / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# 3D lifting
# ---------------------------------------------------------------------------

def _sample_depth_bilinear(depth_map: np.ndarray, u: float, v: float) -> float:
    """Bilinear-interpolated depth sample at sub-pixel position (u, v)."""
    h, w = depth_map.shape
    u = float(np.clip(u, 0, w - 1))
    v = float(np.clip(v, 0, h - 1))
    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
    wu, wv = u - u0, v - v0
    return (
        depth_map[v0, u0] * (1 - wu) * (1 - wv)
        + depth_map[v0, u1] * wu * (1 - wv)
        + depth_map[v1, u0] * (1 - wu) * wv
        + depth_map[v1, u1] * wu * wv
    )


def lift_landmarks_3d(
    landmarks,                  # mediapipe NormalizedLandmark list (len=33)
    depth_map: np.ndarray,      # float32 H×W, [0,1], 1=closest
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Unproject 2D MediaPipe landmarks into 3D camera space.

    Returns float32 array of shape (33, 3) — columns are X, Y, Z.
    Z is disparity-based depth (1=closest to camera).
    X increases rightward, Y increases downward (OpenCV convention).
    """
    pts = np.zeros((len(landmarks), 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        u = lm.x * width
        v = lm.y * height
        d = _sample_depth_bilinear(depth_map, u, v)
        pts[i, 0] = (u - cx) * d / fx
        pts[i, 1] = (v - cy) * d / fy
        pts[i, 2] = d
    return pts


# ---------------------------------------------------------------------------
# Ground plane estimation
# ---------------------------------------------------------------------------

def estimate_ground_plane(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    bottom_fraction: float = 0.15,
    n_ransac_iters: int = 50,
    min_inliers: int = 30,
    inlier_thresh: float = 0.05,
) -> tuple[np.ndarray, float] | None:
    """
    Estimate ground plane from the bottom strip of the depth map using RANSAC.

    Returns (plane_normal, plane_d) where the plane is:
        dot(normal, P) = plane_d   (normal is unit length)

    Returns None if RANSAC cannot find a reliable plane (too few inliers).
    Caller should fall back to the previous estimate or a Y-down prior.

    Args:
        bottom_fraction: fraction of frame height to use as floor search region.
        n_ransac_iters: number of random 3-point trials.
        min_inliers: minimum inliers for a valid plane.
        inlier_thresh: point-plane distance threshold (in depth units, [0,1] scale).
    """
    h, w = depth_map.shape
    row_start = int(h * (1.0 - bottom_fraction))
    strip = depth_map[row_start:, :]           # shape (strip_h, W)
    sh, sw = strip.shape

    # Build 3D points for all strip pixels
    vs, us = np.mgrid[row_start : row_start + sh, 0:sw]  # pixel coords
    d_flat = strip.ravel().astype(np.float32)
    u_flat = us.ravel().astype(np.float32)
    v_flat = vs.ravel().astype(np.float32)

    X = (u_flat - cx) * d_flat / fx
    Y = (v_flat - cy) * d_flat / fy
    Z = d_flat
    pts = np.stack([X, Y, Z], axis=1)          # (N, 3)

    # Subsample for speed
    rng = np.random.default_rng(0)
    n = len(pts)
    if n > 400:
        idx = rng.choice(n, 400, replace=False)
        pts_sub = pts[idx]
    else:
        pts_sub = pts

    best_normal = None
    best_d      = 0.0
    best_count  = 0

    for _ in range(n_ransac_iters):
        tri = pts_sub[rng.choice(len(pts_sub), 3, replace=False)]
        v1, v2 = tri[1] - tri[0], tri[2] - tri[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len
        d = float(np.dot(normal, tri[0]))
        dists = np.abs(pts_sub @ normal - d)
        count = int(np.sum(dists < inlier_thresh))
        if count > best_count:
            best_count  = count
            best_normal = normal
            best_d      = d

    if best_count < min_inliers or best_normal is None:
        return None

    # Refine with all inliers from the full strip
    dists = np.abs(pts @ best_normal - best_d)
    inliers = pts[dists < inlier_thresh]
    if len(inliers) > 3:
        # Least-squares plane through inliers: solve A @ [a,b,c]' = 1
        # (for the form ax+by+cz=1 which avoids degenerate solutions when Z>>0)
        try:
            A = inliers
            b = np.ones(len(inliers))
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            norm_len = np.linalg.norm(coeffs)
            if norm_len > 1e-8:
                refined_normal = coeffs / norm_len
                refined_d = 1.0 / norm_len
                # Keep refinement only if it doesn't flip the normal
                if np.dot(refined_normal, best_normal) > 0.5:
                    best_normal = refined_normal
                    best_d = refined_d
        except np.linalg.LinAlgError:
            pass

    return best_normal, best_d


def fallback_ground_plane() -> tuple[np.ndarray, float]:
    """Y-down plane prior: floor is at Y=0.4 in camera space (rough side-view prior)."""
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = 0.4
    return normal, d


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def point_plane_distance(
    point: np.ndarray,          # shape (3,)
    plane_normal: np.ndarray,   # unit normal
    plane_d: float,
) -> float:
    """Signed distance of point from plane. Positive = above floor (away from ground)."""
    return float(np.dot(plane_normal, point) - plane_d)


def compute_3d_features(
    pts3d: np.ndarray,          # (33, 3) — X,Y,Z in camera space (Z = disparity value)
    plane_normal: np.ndarray,   # unit vector from RANSAC ground plane
    plane_d: float,
    com_history_3d: deque,      # deque of (3,) COM positions
) -> dict:
    """
    Compute 3D biomechanical fall-detection features.

    Two groups of features:
      - Plane-free (always reliable): spine angle against camera vertical, COM velocity.
      - Ground-plane-dependent (reliable only after RANSAC converges): hip/shoulder heights.

    The plane-free features are the primary signals used in state-machine scoring.
    Ground-plane heights are logged to metrics for monitoring but callers should
    guard them behind `_ground_plane_ready`.

    Camera convention: Y-down, X-right, Z = disparity (1=closest to camera).

    Returns dict with keys:
        spine_angle_3d    : angle (deg) of spine from camera vertical; 0=upright, 90=flat
        spine_horiz_3d    : complement; 90=upright, 0=flat  (alias for thresholding)
        com_3d            : np.ndarray (3,) centre of mass
        com_drop_rate     : float, downward (Y) COM velocity, normalised — prefall signal
        com_velocity_3d   : float, total 3D COM speed, normalised
        hip_height_3d     : float, hip height above ground plane (normalised; may be noisy)
        fall_height_3d    : float, shoulder height above ground plane (normalised; may be noisy)
    """
    depth_range = float(pts3d[:, 2].max() - pts3d[:, 2].min()) + 1e-6

    shoulder_mid = (pts3d[IDX_LEFT_SHOULDER] + pts3d[IDX_RIGHT_SHOULDER]) / 2.0
    hip_mid      = (pts3d[IDX_LEFT_HIP]      + pts3d[IDX_RIGHT_HIP])      / 2.0
    com_3d       = (shoulder_mid + hip_mid) / 2.0

    # Spine vector: shoulder_mid -> hip_mid (points down when upright)
    spine_vec  = hip_mid - shoulder_mid
    spine_unit = spine_vec / (np.linalg.norm(spine_vec) + 1e-8)

    # --- Plane-free: spine angle relative to camera vertical axis (Y-down = (0,1,0)) ---
    # This avoids ground-plane dependency and is robust regardless of depth representation.
    # 0 deg = spine is vertical = standing upright
    # 90 deg = spine is horizontal = fallen
    cam_vertical = np.array([0.0, 1.0, 0.0])
    cos_a = float(np.clip(abs(np.dot(spine_unit, cam_vertical)), 0.0, 1.0))
    spine_angle_3d = math.degrees(math.acos(cos_a))
    spine_horiz_3d = 90.0 - spine_angle_3d   # 90=upright, 0=flat

    # --- COM velocity (plane-free) ---
    com_history_3d.append(com_3d.copy())
    com_velocity_3d    = 0.0
    com_drop_rate      = 0.0
    com_depth_velocity = 0.0   # change in disparity Z — detects forward/backward falls
    if len(com_history_3d) >= 3:
        delta = com_history_3d[-1] - com_history_3d[-3]
        # Normalise by scene depth range for inter-frame scale invariance
        com_velocity_3d    = float(np.linalg.norm(delta)) / depth_range
        # Y-component: positive = moving down in image (camera-Y is down) = falling
        com_drop_rate      = float(delta[1]) / depth_range
        # Z-component: change in disparity — positive = moving TOWARD camera (forward fall)
        com_depth_velocity = float(abs(delta[2])) / depth_range

    # --- Ground-plane-dependent heights (informational; guard with _ground_plane_ready) ---
    def _h(pt):
        return point_plane_distance(pt, plane_normal, plane_d) / depth_range

    hip_height_3d  = _h(hip_mid)
    fall_height_3d = _h(shoulder_mid)

    return {
        # Plane-free (always reliable)
        "spine_angle_3d":    spine_angle_3d,    # informational only; use with caution in scoring
        "spine_horiz_3d":    spine_horiz_3d,
        "com_3d":            com_3d,
        "com_drop_rate":     com_drop_rate,     # downward Y velocity → prefall
        "com_depth_velocity": com_depth_velocity, # abs Z velocity → forward/backward fall
        "com_velocity_3d":   com_velocity_3d,
        # Ground-plane-dependent (guard with _ground_plane_ready)
        "hip_height_3d":     hip_height_3d,
        "fall_height_3d":    fall_height_3d,
    }
