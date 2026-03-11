"""
Generate a synthetic test video showing a stick figure standing then falling.

Output: data/videos/synthetic_fall.mp4

The figure transitions through three phases:
  1. Standing upright (~ 3 s)
  2. Pre-fall: leaning / accelerating downward (~ 0.5 s)
  3. Fallen: horizontal on the ground (~ 2 s)
"""
import math
import os
import sys

import cv2
import numpy as np


W, H = 640, 480
FPS = 30
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "videos", "synthetic_fall.mp4")


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    return t * t * (3 - 2 * t)


def draw_stick_figure(img, cx, cy, angle_deg, scale=1.0, color=(200, 220, 255)):
    """
    Draw a simple stick figure.
    angle_deg: 0 = upright, 90 = horizontal (fallen to the right).
    cx, cy: centre of hips (pivot point).
    """
    rad = math.radians(angle_deg)
    # unit vectors along body (spine) and perpendicular
    spine_up  = np.array([-math.sin(rad), -math.cos(rad)])  # hip -> shoulder
    perp      = np.array([spine_up[1], -spine_up[0]])        # left direction

    seg = int(60 * scale)   # spine segment length

    hip    = np.array([cx, cy])
    shoulder = hip + spine_up * seg * 1.4
    head_c   = shoulder + spine_up * int(seg * 0.55)

    knee_l = hip + spine_up * (-seg * 0.7) + perp * int(seg * 0.3)
    knee_r = hip + spine_up * (-seg * 0.7) - perp * int(seg * 0.3)
    ankle_l = knee_l + spine_up * (-seg * 0.7)
    ankle_r = knee_r + spine_up * (-seg * 0.7)
    elbow_l = shoulder + perp * int(seg * 0.5) + spine_up * (-seg * 0.3)
    elbow_r = shoulder - perp * int(seg * 0.5) + spine_up * (-seg * 0.3)
    hand_l  = elbow_l + perp * int(seg * 0.4) + spine_up * (-seg * 0.3)
    hand_r  = elbow_r - perp * int(seg * 0.4) + spine_up * (-seg * 0.3)

    def ipt(p):
        return (int(p[0]), int(p[1]))

    thick = max(2, int(3 * scale))
    lines = [
        (hip, shoulder),
        (shoulder, knee_l), (shoulder, knee_r),   # misleading name reuse: body
        (hip, knee_l), (knee_l, ankle_l),
        (hip, knee_r), (knee_r, ankle_r),
        (shoulder, elbow_l), (elbow_l, hand_l),
        (shoulder, elbow_r), (elbow_r, hand_r),
    ]
    for a, b in lines:
        cv2.line(img, ipt(a), ipt(b), color, thick, cv2.LINE_AA)
    head_r_px = int(seg * 0.28)
    cv2.circle(img, ipt(head_c), head_r_px, color, thick, cv2.LINE_AA)


def make_background():
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    # floor
    cv2.rectangle(bg, (0, H - 60), (W, H), (60, 45, 30), -1)
    # wall
    bg[:H - 60] = (30, 30, 40)
    # floor line
    cv2.line(bg, (0, H - 60), (W, H - 60), (100, 80, 60), 2)
    return bg


def add_label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)


def generate(out_path=OUT_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    floor_y = H - 60
    stand_cx, stand_cy = W // 2, floor_y - 20   # hips just above floor

    # Phase durations in frames
    stand_frames  = int(3.0 * FPS)
    fall_frames   = int(0.8 * FPS)
    ground_frames = int(2.0 * FPS)
    total = stand_frames + fall_frames + ground_frames

    for f in range(total):
        bg = make_background()

        if f < stand_frames:
            angle = 0.0
            cy = stand_cy
            label = "STANDING"
            label_color = (0, 200, 0)

        elif f < stand_frames + fall_frames:
            t = (f - stand_frames) / fall_frames
            t_ease = ease_in_out(t)
            angle = lerp(0.0, 85.0, t_ease)
            # hips drop toward floor as person falls
            cy = lerp(stand_cy, floor_y - 10, t_ease)
            label = "FALLING"
            label_color = (0, 165, 255)

        else:
            angle = 88.0
            cy = floor_y - 8
            label = "FALLEN"
            label_color = (0, 0, 255)

        draw_stick_figure(bg, stand_cx, cy, angle)
        add_label(bg, label, label_color)

        # frame counter
        cv2.putText(bg, f"frame {f+1}/{total}", (W - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        writer.write(bg)

    writer.release()
    print(f"Saved {total} frames ({total/FPS:.1f}s) -> {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_PATH
    generate(out)
