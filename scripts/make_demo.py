"""
Build a demo reel from the annotated UR Fall Detection Dataset clips.

Structure
---------
  Title card
  Section 1 — "Fall Detection"       : 6 diverse fall clips
  Section 2 — "Pre-Fall Lead Time"   : 3 clips with longest lead times, slowed
  Section 3 — "Daily Activities"     : 3 clean ADL clips
  Stats card

Usage
-----
    python scripts/make_demo.py
    python scripts/make_demo.py --out demo.mp4 --fps 30
"""
from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANNOTATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "annotated")
OUT_W, OUT_H = 640, 480
FPS = 30  # overridden by --fps arg below

# Colours (BGR)
C_BG       = (18,  18,  18)
C_WHITE    = (240, 240, 240)
C_ORANGE   = (0,   165, 255)
C_GREEN    = (0,   200, 0)
C_RED      = (0,   0,   220)
C_GREY     = (120, 120, 120)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO  = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def blank() -> np.ndarray:
    return np.full((OUT_H, OUT_W, 3), C_BG, dtype=np.uint8)


def text_centred(frame, text, y, scale=1.0, colour=C_WHITE, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (OUT_W - tw) // 2
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def text_left(frame, text, x, y, scale=0.55, colour=C_WHITE, thickness=1):
    cv2.putText(frame, text, (x, y), FONT_MONO, scale, colour, thickness, cv2.LINE_AA)


def title_card(lines: list[tuple[str, float, tuple]],
               duration_s: float = 2.5) -> list[np.ndarray]:
    """Return a list of identical title frames for `duration_s` seconds."""
    f = blank()
    y = OUT_H // 2 - (len(lines) - 1) * 28
    for text, scale, colour in lines:
        text_centred(f, text, y, scale, colour)
        y += int(scale * 52 + 8)
    n = int(duration_s * FPS)
    return [f.copy() for _ in range(n)]


def fade_in(frames: list[np.ndarray], n: int = 15) -> list[np.ndarray]:
    out = []
    for i, f in enumerate(frames):
        if i < n:
            alpha = i / n
            f = (f.astype(np.float32) * alpha).astype(np.uint8)
        out.append(f)
    return out


def fade_out(frames: list[np.ndarray], n: int = 15) -> list[np.ndarray]:
    out = []
    total = len(frames)
    for i, f in enumerate(frames):
        remaining = total - i
        if remaining <= n:
            alpha = remaining / n
            f = (f.astype(np.float32) * alpha).astype(np.uint8)
        out.append(f)
    return out


def section_card(title: str, subtitle: str = "") -> list[np.ndarray]:
    lines = [(title, 0.9, C_ORANGE)]
    if subtitle:
        lines.append((subtitle, 0.48, C_GREY))
    return fade_in(fade_out(title_card(lines, 1.8)))


def stat_bar(frame, label, value, x, y, bar_w=320, bar_h=18,
             filled_colour=C_GREEN, fraction=1.0):
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * fraction), y + bar_h),
                  filled_colour, -1)
    text_left(frame, label, x, y - 4, 0.42, C_WHITE)
    text_left(frame, value, x + bar_w + 8, y + bar_h - 2, 0.42, C_WHITE)


# ---------------------------------------------------------------------------
# Clip loading
# ---------------------------------------------------------------------------

def load_clip(name: str, start_frame: int = 0,
              end_frame: int | None = None,
              slow: float = 1.0) -> list[np.ndarray]:
    """
    Load an annotated clip, optionally trimming and slowing it.
    Returns a list of BGR frames resized to (OUT_W, OUT_H).
    """
    path = os.path.join(ANNOTATED_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        return []

    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start_frame and (end_frame is None or idx < end_frame):
            frame = cv2.resize(frame, (OUT_W, OUT_H))
            frames.append(frame)
        idx += 1
    cap.release()

    if slow != 1.0 and slow > 1.0:
        # Duplicate frames to slow down
        factor = int(round(slow))
        slowed = []
        for f in frames:
            for _ in range(factor):
                slowed.append(f)
        frames = slowed

    return frames


def add_clip_label(frames: list[np.ndarray], label: str,
                   sublabel: str = "") -> list[np.ndarray]:
    """Burn a clip label into the top-left of every frame."""
    out = []
    for f in frames:
        f = f.copy()
        cv2.rectangle(f, (0, 0), (OUT_W, 36), (0, 0, 0), -1)
        text_left(f, label, 8, 22, 0.58, C_WHITE, 1)
        if sublabel:
            text_left(f, sublabel, OUT_W - 230, 22, 0.44, C_ORANGE, 1)
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demo.mp4")
    args = parser.parse_args()

    print("Building demo reel...")
    all_frames: list[np.ndarray] = []

    # ---- Title card ----
    all_frames += fade_in(title_card([
        ("OpenFall", 1.6, C_WHITE),
        ("Real-Time Pre-Fall Detection", 0.55, C_GREY),
        ("UR Fall Detection Dataset  —  30 falls  40 ADL", 0.38, C_GREY),
    ], 3.0))

    # ---- Section 1: Fall detection ----
    all_frames += section_card("Fall Detection", "STANDING  →  PREFALL  →  FALLEN")

    clips_s1 = [
        ("fall-01.mp4", 80,  None, 1.0, "fall-01",  "400 ms lead"),
        ("fall-07.mp4", 80,  None, 1.0, "fall-07",  "500 ms lead"),
        ("fall-09.mp4", 110, None, 1.0, "fall-09",  "467 ms lead"),
        ("fall-17.mp4", 55,  None, 1.0, "fall-17",  "267 ms lead"),
        ("fall-22.mp4", 0,   None, 1.0, "fall-22",  "233 ms lead — fast fall"),
        ("fall-03.mp4", 100, None, 1.0, "fall-03",  "833 ms lead — stumble then fall"),
    ]
    for fname, s, e, slow, label, sub in clips_s1:
        frames = load_clip(fname, s, e, slow)
        frames = add_clip_label(frames, label, sub)
        frames = fade_in(fade_out(frames, 8), 8)
        all_frames += frames
        print(f"  {label}: {len(frames)} frames")

    # ---- Section 2: Pre-fall lead time ----
    all_frames += section_card("Pre-Fall Lead Time",
                               "Slowed 2x  —  PREFALL fires before body reaches floor")

    clips_s2 = [
        ("fall-14.mp4", 0,  None, 2.0, "fall-14", "733 ms lead  (longest)"),
        ("fall-30.mp4", 0,  None, 2.0, "fall-30", "733 ms lead"),
        ("fall-26.mp4", 0,  None, 2.0, "fall-26", "667 ms lead"),
    ]
    for fname, s, e, slow, label, sub in clips_s2:
        frames = load_clip(fname, s, e, slow)
        frames = add_clip_label(frames, label, sub)
        frames = fade_in(fade_out(frames, 8), 8)
        all_frames += frames
        print(f"  {label}: {len(frames)} frames")

    # ---- Section 3: Daily activities ----
    all_frames += section_card("Daily Activities",
                               "No false FALLEN events on upright ADL sequences")

    clips_s3 = [
        ("adl-03.mp4", 0, None, 1.0, "adl-03", "clean — STANDING throughout"),
        ("adl-07.mp4", 0, None, 1.0, "adl-07", "clean — STANDING throughout"),
        ("adl-01.mp4", 0, None, 1.0, "adl-01", "brief PREFALL — no FALLEN"),
    ]
    for fname, s, e, slow, label, sub in clips_s3:
        frames = load_clip(fname, s, e, slow)
        frames = add_clip_label(frames, label, sub)
        frames = fade_in(fade_out(frames, 8), 8)
        all_frames += frames
        print(f"  {label}: {len(frames)} frames")

    # ---- Stats card ----
    all_frames += section_card("")  # short gap
    stats_frame = blank()
    text_centred(stats_frame, "Results  —  Full Dataset (70 sequences)", 60, 0.75, C_WHITE)

    metrics = [
        ("Fall detection rate",      "28 / 30  (93.3%)",   C_GREEN,  0.933),
        ("Mean PREFALL lead time",   "415 ms  (167–833)",  C_ORANGE, 0.415 / 0.833),
        ("False FALLEN — upright ADL","0 / 9   (0.0%)",   C_GREEN,  0.0),
        ("False FALLEN — all ADL",   "17 / 40  (42.5%)",  C_RED,    0.425),
    ]
    y = 130
    for label, value, colour, frac in metrics:
        stat_bar(stats_frame, label, value, 60, y, bar_w=360,
                 filled_colour=colour, fraction=frac)
        y += 72

    text_centred(stats_frame, "No training data required  |  Single RGB camera  |  CPU real-time",
                 OUT_H - 30, 0.38, C_GREY)

    all_frames += fade_in([stats_frame.copy() for _ in range(int(4.0 * FPS))], 20)
    all_frames = fade_out(all_frames, 20)

    # ---- Write output ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (OUT_W, OUT_H))
    for f in all_frames:
        writer.write(f)
    writer.release()

    duration = len(all_frames) / FPS
    print(f"\nSaved {args.out}  ({len(all_frames)} frames, {duration:.1f}s)")


if __name__ == "__main__":
    main()
