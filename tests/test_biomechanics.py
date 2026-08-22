import math
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from src.biomechanics import (
    IDX_LEFT_HIP,
    IDX_LEFT_SHOULDER,
    IDX_RIGHT_HIP,
    IDX_RIGHT_SHOULDER,
    build_camera_matrix,
    compute_3d_features,
    estimate_ground_plane,
    lift_landmarks_3d,
    point_plane_distance,
)


class CameraGeometryTests(unittest.TestCase):
    def test_camera_matrix_uses_the_horizontal_field_of_view(self):
        fx, fy, cx, cy = build_camera_matrix(640, 480, fov_h_deg=90.0)

        self.assertAlmostEqual(fx, 320.0)
        self.assertEqual(fy, fx)
        self.assertEqual(cx, 320.0)
        self.assertEqual(cy, 240.0)

    def test_lift_landmarks_unprojects_pixels_at_the_sampled_depth(self):
        landmarks = [
            SimpleNamespace(x=0.5, y=0.5),
            SimpleNamespace(x=0.0, y=0.0),
        ]
        depth_map = np.full((2, 2), 0.5, dtype=np.float32)

        points = lift_landmarks_3d(
            landmarks,
            depth_map,
            width=2,
            height=2,
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
        )

        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.5])
        np.testing.assert_allclose(points[1], [-0.5, -0.5, 0.5])

    def test_ground_plane_estimation_returns_a_unit_plane_for_flat_depth(self):
        depth_map = np.full((24, 32), 0.5, dtype=np.float32)
        fx, fy, cx, cy = build_camera_matrix(32, 24)

        result = estimate_ground_plane(
            depth_map,
            fx,
            fy,
            cx,
            cy,
            bottom_fraction=0.5,
            min_inliers=20,
        )

        self.assertIsNotNone(result)
        normal, plane_d = result
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=5)
        sample = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        self.assertLess(abs(point_plane_distance(sample, normal, plane_d)), 1e-4)


class FeatureTests(unittest.TestCase):
    @staticmethod
    def pose(shoulder_mid, hip_mid):
        points = np.zeros((33, 3), dtype=np.float32)
        shoulder_mid = np.asarray(shoulder_mid, dtype=np.float32)
        hip_mid = np.asarray(hip_mid, dtype=np.float32)
        points[IDX_LEFT_SHOULDER] = shoulder_mid
        points[IDX_RIGHT_SHOULDER] = shoulder_mid
        points[IDX_LEFT_HIP] = hip_mid
        points[IDX_RIGHT_HIP] = hip_mid
        return points

    def test_spine_angle_distinguishes_upright_and_horizontal_poses(self):
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        upright = compute_3d_features(
            self.pose([0.0, 0.0, 0.2], [0.0, 1.0, 0.2]),
            normal,
            0.0,
            deque(maxlen=10),
        )
        horizontal = compute_3d_features(
            self.pose([0.0, 0.0, 0.2], [1.0, 0.0, 0.2]),
            normal,
            0.0,
            deque(maxlen=10),
        )

        self.assertTrue(math.isclose(upright["spine_angle_3d"], 0.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(upright["spine_horiz_3d"], 90.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(horizontal["spine_angle_3d"], 90.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(horizontal["spine_horiz_3d"], 0.0, abs_tol=1e-5))

    def test_com_drop_rate_turns_positive_for_downward_motion(self):
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        history = deque(maxlen=10)

        for offset in (0.0, 0.1, 0.2):
            metrics = compute_3d_features(
                self.pose([0.0, offset, 0.2], [0.0, 1.0 + offset, 0.2]),
                normal,
                0.0,
                history,
            )

        self.assertGreater(metrics["com_drop_rate"], 0.0)
        self.assertGreater(metrics["com_velocity_3d"], 0.0)


if __name__ == "__main__":
    unittest.main()
