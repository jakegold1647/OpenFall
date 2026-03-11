"""
Fall detector using MediaPipe Pose Landmarker (Tasks API, mediapipe >= 0.10).

Optional Depth Anything V2 integration: pass a DepthEstimator instance to enable
3D biomechanical feature extraction.  The 2D code path is always active; 3D signals
add bonus votes to the state-machine scorer without replacing 2D signals.

States:  STANDING  ->  PREFALL  ->  FALLEN
"""
import math
import os
from collections import deque
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

from src.biomechanics import (
    build_camera_matrix,
    compute_3d_features,
    estimate_ground_plane,
    fallback_ground_plane,
    lift_landmarks_3d,
)

# Landmark indices
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_HIP       = 23
IDX_RIGHT_HIP      = 24

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "data", "pose_landmarker_lite.task"
)

_DRAW_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 11),  (0, 12),
]


class FallState(Enum):
    STANDING = auto()
    PREFALL  = auto()
    FALLEN   = auto()


class FallDetector:
    def __init__(
        self,
        model_path=None,
        # --- 2D thresholds ---
        body_angle_fallen_thresh: float   = 50.0,
        body_angle_prefall_thresh: float  = 30.0,
        velocity_prefall_thresh: float    = 0.025,
        aspect_ratio_thresh: float        = 1.2,
        hip_height_thresh: float          = 0.72,
        # --- optional depth / 3D ---
        depth_estimator=None,
        fov_h_deg: float                  = 70.0,
        body_angle_3d_fallen_thresh: float  = 40.0,
        body_angle_3d_prefall_thresh: float = 20.0,
        hip_height_3d_fallen_thresh: float  = 0.15,
        fall_height_3d_fallen_thresh: float = 0.20,
        com_drop_rate_prefall_thresh: float = 0.03,
        ground_plane_update_interval: int   = 30,
        # ---
        history_len: int = 10,
    ):
        self.body_angle_fallen_thresh  = body_angle_fallen_thresh
        self.body_angle_prefall_thresh = body_angle_prefall_thresh
        self.velocity_prefall_thresh   = velocity_prefall_thresh
        self.aspect_ratio_thresh       = aspect_ratio_thresh
        self.hip_height_thresh         = hip_height_thresh

        self._depth_estimator            = depth_estimator
        self._fov_h_deg                  = fov_h_deg
        self.body_angle_3d_fallen_thresh = body_angle_3d_fallen_thresh
        self.body_angle_3d_prefall_thresh= body_angle_3d_prefall_thresh
        self.hip_height_3d_fallen_thresh = hip_height_3d_fallen_thresh
        self.fall_height_3d_fallen_thresh= fall_height_3d_fallen_thresh
        self.com_drop_rate_prefall_thresh= com_drop_rate_prefall_thresh
        self._ground_plane_interval      = ground_plane_update_interval

        model_path = os.path.abspath(model_path or _DEFAULT_MODEL)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found: {model_path}\n"
                "Download with:\n"
                "  curl -L https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/"
                "pose_landmarker_lite.task -o data/pose_landmarker_lite.task"
            )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker   = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._frame_num    = 0

        self._com_history    = deque(maxlen=history_len)
        self._com_history_3d = deque(maxlen=history_len)
        self._state              = FallState.STANDING
        self._state_frame_count  = 0

        # 3D state
        self._fx = self._fy = self._cx = self._cy = None
        self._ground_plane       = fallback_ground_plane()
        self._ground_plane_ready = False   # True once first RANSAC estimate succeeds
        self._pts3d_last         = None

    # ------------------------------------------------------------------
    @property
    def state(self):
        return self._state

    # ------------------------------------------------------------------
    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, FallState, dict]:
        """
        Process a single BGR frame.

        Returns:
            annotated_frame (BGR),
            state (FallState),
            metrics (dict)  — 2D keys always present, 3D keys added when depth active.
        """
        h, w = frame_bgr.shape[:2]

        if timestamp_ms is None:
            self._timestamp_ms += 33
            ts = self._timestamp_ms
        else:
            ts = int(timestamp_ms)

        self._frame_num += 1
        frame_idx = self._frame_num

        # ---- Lazy-init camera matrix ----
        if self._fx is None and self._depth_estimator is not None:
            self._fx, self._fy, self._cx, self._cy = build_camera_matrix(
                w, h, self._fov_h_deg
            )

        # ---- MediaPipe pose ----
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_img, ts)

        annotated = frame_bgr.copy()
        metrics   = {}

        if not result.pose_landmarks:
            metrics["detected"] = False
            self._draw_state(annotated)
            return annotated, self._state, metrics

        lm = result.pose_landmarks[0]
        metrics["detected"] = True

        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y])

        shoulder_mid = (pt(IDX_LEFT_SHOULDER) + pt(IDX_RIGHT_SHOULDER)) / 2.0
        hip_mid      = (pt(IDX_LEFT_HIP)      + pt(IDX_RIGHT_HIP))      / 2.0
        com_2d       = (shoulder_mid + hip_mid) / 2.0

        # ---- 2D features ----
        spine_vec = hip_mid - shoulder_mid
        angle_deg = math.degrees(
            math.atan2(abs(spine_vec[0]), abs(spine_vec[1]) + 1e-6)
        )
        xs      = [l.x for l in lm]
        ys      = [l.y for l in lm]
        bbox_w  = (max(xs) - min(xs)) * w
        bbox_h  = (max(ys) - min(ys)) * h
        aspect  = bbox_w / (bbox_h + 1e-6)
        hip_y   = hip_mid[1]

        self._com_history.append(com_2d[1])
        velocity_2d = 0.0
        if len(self._com_history) >= 3:
            velocity_2d = self._com_history[-1] - self._com_history[-3]

        metrics.update({
            "body_angle":        angle_deg,
            "aspect_ratio":      aspect,
            "hip_y":             hip_y,
            "com_velocity_down": velocity_2d,
        })

        # ---- 2D scores ----
        fallen_score = (
            int(angle_deg > self.body_angle_fallen_thresh)
            + int(aspect    > self.aspect_ratio_thresh)
            + int(hip_y     > self.hip_height_thresh)
        )
        prefall_score = (
            int(angle_deg    > self.body_angle_prefall_thresh)
            + int(velocity_2d > self.velocity_prefall_thresh)
        )

        # ---- Depth + 3D block ----
        if self._depth_estimator is not None:
            depth_map = self._depth_estimator.process(frame_bgr, frame_idx)
            if depth_map is None:
                depth_map = self._depth_estimator.last_depth()

            if depth_map is not None:
                # Refresh ground plane on first frame and every N frames after
                if not self._ground_plane_ready or frame_idx % self._ground_plane_interval == 0:
                    result_gp = estimate_ground_plane(
                        depth_map, self._fx, self._fy, self._cx, self._cy
                    )
                    if result_gp is not None:
                        self._ground_plane       = result_gp
                        self._ground_plane_ready = True

                pts3d = lift_landmarks_3d(
                    lm, depth_map, w, h,
                    self._fx, self._fy, self._cx, self._cy
                )
                self._pts3d_last = pts3d

                plane_normal, plane_d = self._ground_plane
                m3d = compute_3d_features(
                    pts3d, plane_normal, plane_d, self._com_history_3d
                )
                metrics.update(m3d)

                # 3D PREFALL bonus votes — downward Y velocity only.
                # com_depth_velocity (Z) is tracked in HUD but excluded from scoring: normal
                # walking generates large depth changes, making its threshold dataset-dependent.
                # It is most useful for forward/backward falls where the camera faces the subject.
                # FALLEN scoring remains 2D-only to avoid ground-plane instability.
                prefall_score += int(m3d["com_drop_rate"] > self.com_drop_rate_prefall_thresh)


        # ---- State machine ----
        if fallen_score >= 2:
            new_state = FallState.FALLEN
        elif prefall_score >= 1 and self._state != FallState.FALLEN:
            new_state = FallState.PREFALL
        else:
            new_state = FallState.STANDING

        if new_state != self._state:
            self._state = new_state
            self._state_frame_count = 0
        else:
            self._state_frame_count += 1

        # ---- Draw skeleton ----
        for a_idx, b_idx in _DRAW_CONNECTIONS:
            ax, ay = int(lm[a_idx].x * w), int(lm[a_idx].y * h)
            bx, by = int(lm[b_idx].x * w), int(lm[b_idx].y * h)
            cv2.line(annotated, (ax, ay), (bx, by), (0, 200, 80), 2, cv2.LINE_AA)
        for l in lm:
            cx_px = int(l.x * w)
            cy_px = int(l.y * h)
            cv2.circle(annotated, (cx_px, cy_px), 3, (0, 255, 0), -1, cv2.LINE_AA)

        self._draw_state(annotated)
        self._draw_hud(annotated, metrics)

        return annotated, self._state, metrics

    # ------------------------------------------------------------------
    def _draw_state(self, frame: np.ndarray):
        color = {
            FallState.STANDING: (0, 200, 0),
            FallState.PREFALL:  (0, 165, 255),
            FallState.FALLEN:   (0, 0, 255),
        }[self._state]
        cv2.putText(frame, self._state.name, (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    def _draw_hud(self, frame: np.ndarray, metrics: dict):
        lines = [
            f"Angle2D: {metrics.get('body_angle',       0.0):.1f}deg",
            f"Aspect:  {metrics.get('aspect_ratio',     0.0):.2f}",
            f"HipY:    {metrics.get('hip_y',            0.0):.2f}",
            f"Vel2D:   {metrics.get('com_velocity_down',0.0):.4f}",
        ]
        if "spine_angle_3d" in metrics:
            gp_label = "GP:ready" if self._ground_plane_ready else "GP:wait"
            lines += [
                f"--- 3D [{gp_label}] ---",
                f"SpineA:  {metrics['spine_angle_3d']:.1f}deg",
                f"DropV:   {metrics['com_drop_rate']:.4f}",
                f"DepthV:  {metrics['com_depth_velocity']:.4f}",
                f"HipH3D:  {metrics['hip_height_3d']:.3f}",
                f"FallH:   {metrics['fall_height_3d']:.3f}",
            ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 75 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # ------------------------------------------------------------------
    def reset(self):
        self._com_history.clear()
        self._com_history_3d.clear()
        self._state = FallState.STANDING
        self._state_frame_count = 0
        self._frame_num = 0
        self._pts3d_last = None
        self._ground_plane = fallback_ground_plane()
        self._ground_plane_ready = False
        if self._depth_estimator is not None:
            self._depth_estimator.reset()

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
