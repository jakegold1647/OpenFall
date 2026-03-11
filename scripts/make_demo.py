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
C_BG       = (15,  15,  20)    # Darker, slightly blue-tinted
C_WHITE    = (250, 250, 250)
C_ORANGE   = (0,   180, 255)   # Brighter orange
C_GREEN    = (80,  220, 120)   # Softer green
C_RED      = (60,  60,  240)   # Softer red
C_GREY     = (140, 140, 150)
C_ACCENT   = (255, 200, 0)     # Cyan accent

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO  = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def blank() -> np.ndarray:
    return np.full((OUT_H, OUT_W, 3), C_BG, dtype=np.uint8)


def text_centred(frame, text, y, scale=1.0, colour=C_WHITE, thickness=1, shadow=True):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (OUT_W - tw) // 2
    if shadow:
        # Draw shadow for better legibility
        cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def text_left(frame, text, x, y, scale=0.55, colour=C_WHITE, thickness=1, shadow=True):
    if shadow:
        cv2.putText(frame, text, (x + 1, y + 1), FONT_MONO, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT_MONO, scale, colour, thickness, cv2.LINE_AA)


def ease_in_out(t: float) -> float:
    """Smooth easing function (cubic ease-in-out)."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_back(t: float) -> float:
    """Ease out with slight overshoot (bouncy)."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def vignette(frame: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Apply subtle vignette effect (darken corners)."""
    rows, cols = frame.shape[:2]

    # Create radial gradient
    X = np.linspace(-1, 1, cols)
    Y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(X, Y)
    radius = np.sqrt(X**2 + Y**2)

    # Smooth falloff
    vignette_mask = 1 - np.clip(radius - 0.5, 0, 1) * strength
    vignette_mask = vignette_mask[:, :, np.newaxis]

    return (frame.astype(np.float32) * vignette_mask).astype(np.uint8)


def color_grade(frame: np.ndarray, contrast: float = 1.1, saturation: float = 1.15) -> np.ndarray:
    """Apply cinematic color grading."""
    # Increase contrast
    frame_float = frame.astype(np.float32)
    frame_float = ((frame_float / 255.0 - 0.5) * contrast + 0.5) * 255.0
    frame_float = np.clip(frame_float, 0, 255)

    # Increase saturation
    hsv = cv2.cvtColor(frame_float.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def title_card(lines: list[tuple[str, float, tuple]],
               duration_s: float = 2.5, animate: bool = True) -> list[np.ndarray]:
    """Return animated title card frames with scale-in effect."""
    n = int(duration_s * FPS)
    frames = []
    anim_frames = int(FPS * 0.6) if animate else 0  # 0.6s animation

    for frame_idx in range(n):
        f = blank()

        # Ken Burns: subtle zoom
        if frame_idx < n:
            zoom = 1.0 + (frame_idx / n) * 0.05  # Zoom from 1.0 to 1.05
            center = (OUT_W // 2, OUT_H // 2)
            M = cv2.getRotationMatrix2D(center, 0, zoom)
            f = cv2.warpAffine(f, M, (OUT_W, OUT_H), borderValue=C_BG)

        y_base = OUT_H // 2 - (len(lines) - 1) * 28

        for line_idx, (text, scale, colour) in enumerate(lines):
            y = y_base + line_idx * int(scale * 52 + 8)

            # Animate each line independently with stagger
            if animate and frame_idx < anim_frames:
                delay = line_idx * 5  # Stagger by 5 frames per line
                if frame_idx < delay:
                    continue
                progress = (frame_idx - delay) / (anim_frames - delay) if frame_idx >= delay else 0
                progress = min(progress, 1.0)

                # Scale + fade in
                anim_scale = ease_out_back(progress)
                alpha = progress

                # Draw with animation
                (tw, th), _ = cv2.getTextSize(text, FONT, scale * anim_scale, 1)
                x = (OUT_W - tw) // 2

                # Shadow
                shadow_color = tuple(int(c * alpha) for c in [0, 0, 0])
                cv2.putText(f, text, (x + 2, y + 2), FONT, scale * anim_scale,
                           shadow_color, 2, cv2.LINE_AA)

                # Text
                text_color = tuple(int(c * alpha) for c in colour)
                cv2.putText(f, text, (x, y), FONT, scale * anim_scale,
                           text_color, 1, cv2.LINE_AA)
            else:
                # Static after animation
                text_centred(f, text, y, scale, colour, shadow=True)

        # Apply vignette
        f = vignette(f, strength=0.3)
        frames.append(f)

    return frames


def fade_in(frames: list[np.ndarray], n: int = 15) -> list[np.ndarray]:
    """Fade in with smooth easing."""
    out = []
    for i, f in enumerate(frames):
        if i < n:
            alpha = ease_in_out(i / n)
            f = (f.astype(np.float32) * alpha).astype(np.uint8)
        out.append(f)
    return out


def fade_out(frames: list[np.ndarray], n: int = 15) -> list[np.ndarray]:
    """Fade out with smooth easing."""
    out = []
    total = len(frames)
    for i, f in enumerate(frames):
        remaining = total - i
        if remaining <= n:
            alpha = ease_in_out(remaining / n)
            f = (f.astype(np.float32) * alpha).astype(np.uint8)
        out.append(f)
    return out


def crossfade(frames_a: list[np.ndarray], frames_b: list[np.ndarray], n: int = 12) -> list[np.ndarray]:
    """Crossfade transition with slight zoom for cinematic effect."""
    if not frames_a or not frames_b:
        return frames_a + frames_b

    # Take last n frames from A and first n from B
    overlap_a = frames_a[-min(n, len(frames_a)):]
    overlap_b = frames_b[:min(n, len(overlap_a))]

    blended = []
    for i, (fa, fb) in enumerate(zip(overlap_a, overlap_b)):
        alpha = ease_in_out(i / len(overlap_a))

        # Subtle zoom on outgoing frame
        zoom_out = 1.0 + (alpha * 0.03)  # Zoom out slightly
        center = (OUT_W // 2, OUT_H // 2)
        M = cv2.getRotationMatrix2D(center, 0, zoom_out)
        fa_zoomed = cv2.warpAffine(fa, M, (OUT_W, OUT_H))

        # Zoom in on incoming frame
        zoom_in = 1.03 - (alpha * 0.03)  # Zoom in from slightly zoomed
        M = cv2.getRotationMatrix2D(center, 0, zoom_in)
        fb_zoomed = cv2.warpAffine(fb, M, (OUT_W, OUT_H))

        # Blend
        frame = cv2.addWeighted(fa_zoomed, 1 - alpha, fb_zoomed, alpha, 0)
        blended.append(frame)

    return frames_a[:-len(overlap_a)] + blended + frames_b[len(overlap_b):]


def section_card(title: str, subtitle: str = "") -> list[np.ndarray]:
    lines = [(title, 0.9, C_ORANGE)]
    if subtitle:
        lines.append((subtitle, 0.48, C_GREY))
    return fade_in(fade_out(title_card(lines, 2.0, animate=True), 12), 12)


def stat_bar(frame, label, value, x, y, bar_w=320, bar_h=20,
             filled_colour=C_GREEN, fraction=1.0, anim_progress=1.0):
    """Draw a professional stat bar with label and value."""
    # Background bar with subtle border
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (40, 40, 45), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (80, 80, 85), 1)

    # Filled portion with animation
    if fraction > 0 and anim_progress > 0:
        fill_w = int(bar_w * fraction * anim_progress)
        if fill_w > 1:
            # Gradient effect
            for i in range(fill_w):
                gradient_alpha = 0.8 + 0.2 * (i / fill_w)  # Brighter on right edge
                color = tuple(int(c * gradient_alpha) for c in filled_colour)
                cv2.line(frame, (x + 1 + i, y + 1), (x + 1 + i, y + bar_h - 1), color, 1)

            # Glow effect on leading edge
            if fill_w < bar_w - 2:
                glow_color = tuple(min(int(c * 1.3), 255) for c in filled_colour)
                cv2.line(frame, (x + fill_w, y), (x + fill_w, y + bar_h), glow_color, 2)

    # Label above bar
    text_left(frame, label, x, y - 6, 0.44, C_WHITE, 1, shadow=True)

    # Value to the right (fade in with animation)
    value_alpha = min(anim_progress, 1.0)
    if value_alpha > 0:
        # Create temporary frame for text with alpha
        text_color = tuple(int(c * value_alpha) for c in C_WHITE)
        text_left(frame, value, x + bar_w + 12, y + bar_h - 4, 0.46, text_color, 1, shadow=True)


def animated_stat_card(metrics: list[tuple[str, str, tuple, float]],
                       duration_s: float = 5.0) -> list[np.ndarray]:
    """Create animated stats card with bars filling up over time."""
    n_frames = int(duration_s * FPS)
    frames = []
    anim_duration = int(FPS * 1.5)  # 1.5s to fill bars

    for frame_idx in range(n_frames):
        f = blank()

        # Title with accent line
        text_centred(f, "Results - Full Dataset (70 sequences)", 50, 0.75, C_WHITE, 2, shadow=True)
        cv2.line(f, (OUT_W // 2 - 150, 70), (OUT_W // 2 + 150, 70), C_ACCENT, 2)

        # Calculate animation progress
        if frame_idx < anim_duration:
            anim_progress = ease_out_back(frame_idx / anim_duration)
        else:
            anim_progress = 1.0

        # Draw each stat bar with staggered animation
        y = 120
        for bar_idx, (label, value, colour, frac) in enumerate(metrics):
            # Stagger each bar by 8 frames
            bar_delay = bar_idx * 8
            bar_progress = max(0, min(1, (frame_idx - bar_delay) / anim_duration))
            bar_progress = ease_out_back(bar_progress) if bar_progress > 0 else 0

            stat_bar(f, label, value, 50, y, bar_w=400,
                    filled_colour=colour, fraction=frac, anim_progress=bar_progress)
            y += 70

        # Footer (fade in after bars complete)
        footer_alpha = max(0, min(1, (frame_idx - anim_duration) / 15))
        if footer_alpha > 0:
            cv2.line(f, (OUT_W // 2 - 180, OUT_H - 60),
                    (OUT_W // 2 + 180, OUT_H - 60),
                    tuple(int(c * footer_alpha) for c in C_GREY), 1)

            footer_color = tuple(int(c * footer_alpha) for c in C_GREY)
            text_centred(f, "No training data  |  Single RGB camera  |  CPU real-time",
                        OUT_H - 35, 0.42, footer_color, 1, shadow=False)

            accent_color = tuple(int(c * footer_alpha) for c in C_ACCENT)
            text_centred(f, "github.com/jakegold1647/OpenFall",
                        OUT_H - 12, 0.35, accent_color, 1, shadow=False)

        # Apply vignette
        f = vignette(f, strength=0.25)
        frames.append(f)

    return frames


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
                   sublabel: str = "", progress: tuple[int, int] = None,
                   apply_effects: bool = True) -> list[np.ndarray]:
    """Burn a clip label into the top with gradient background and optional progress indicator."""
    out = []
    n_frames = len(frames)

    for idx, f in enumerate(frames):
        f = f.copy()

        # Apply cinematic effects
        if apply_effects:
            f = color_grade(f, contrast=1.08, saturation=1.12)
            f = vignette(f, strength=0.35)

        # Gradient overlay for better legibility
        overlay = f.copy()
        for y in range(48):
            alpha = 0.88 - (y / 48) * 0.88  # Fade from opaque to transparent
            cv2.line(overlay, (0, y), (OUT_W, y), (0, 0, 0), 1)
        f = cv2.addWeighted(f, 0.35, overlay, 0.65, 0)

        # Main label with animated entry
        label_progress = min(1.0, idx / 8) if idx < 8 else 1.0
        label_x = int(10 - (1 - label_progress) * 30)  # Slide in from left
        label_alpha = label_progress

        label_color = tuple(int(c * label_alpha) for c in C_WHITE)
        if label_alpha > 0:
            text_left(f, label, label_x, 26, 0.64, label_color, 2, shadow=True)

        # Sublabel on the right
        if sublabel and label_alpha > 0.5:
            sub_alpha = min(1.0, (label_alpha - 0.5) * 2)
            sub_color = tuple(int(c * sub_alpha) for c in C_ACCENT)
            text_left(f, sublabel, OUT_W - 240, 26, 0.48, sub_color, 1, shadow=True)

        # Progress indicator (small dots) - animated fill
        if progress and label_alpha > 0.7:
            current, total = progress
            dot_x = 10
            dot_y = 42
            for i in range(total):
                if i < current:
                    color = C_ORANGE
                elif i == current:
                    # Pulsing current dot
                    pulse = 0.7 + 0.3 * abs(np.sin(idx * 0.15))
                    color = tuple(int(c * pulse) for c in C_ORANGE)
                else:
                    color = (50, 50, 50)

                cv2.circle(f, (dot_x + i * 14, dot_y), 3, color, -1)
                # Subtle glow on current
                if i == current:
                    cv2.circle(f, (dot_x + i * 14, dot_y), 5, color, 1)

        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global FPS
    parser = argparse.ArgumentParser(description="Build demo reel from annotated clips")
    parser.add_argument("--out", default="demo.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    args = parser.parse_args()

    FPS = args.fps
    print(f"Building demo reel at {FPS} fps...")
    all_frames: list[np.ndarray] = []

    # ---- Title card ----
    title_frames = title_card([
        ("OpenFall", 1.9, C_WHITE),
        ("Real-Time Pre-Fall Detection", 0.62, C_ACCENT),
        ("UR Fall Detection Dataset  -  30 falls, 40 ADL", 0.42, C_GREY),
    ], 4.0, animate=True)

    # Add accent line to title (animate in)
    for idx, frame in enumerate(title_frames):
        line_progress = min(1.0, idx / 25)
        if line_progress > 0:
            line_w = int(240 * line_progress)
            cv2.line(frame, (OUT_W // 2 - line_w // 2, OUT_H // 2 - 10),
                    (OUT_W // 2 + line_w // 2, OUT_H // 2 - 10), C_ACCENT, 3)

    all_frames += fade_in(title_frames, 30)

    # ---- Section 1: Fall detection ----
    all_frames += section_card("Fall Detection", "STANDING -> PREFALL -> FALLEN")

    clips_s1 = [
        ("fall-01.mp4", 80,  None, 1.0, "fall-01",  "400 ms lead"),
        ("fall-07.mp4", 80,  None, 1.0, "fall-07",  "500 ms lead"),
        ("fall-09.mp4", 110, None, 1.0, "fall-09",  "467 ms lead"),
        ("fall-17.mp4", 55,  None, 1.0, "fall-17",  "267 ms lead"),
        ("fall-22.mp4", 0,   None, 1.0, "fall-22",  "233 ms lead - fast fall"),
        ("fall-03.mp4", 100, None, 1.0, "fall-03",  "833 ms lead - stumble then fall"),
    ]
    prev_frames = None
    for idx, (fname, s, e, slow, label, sub) in enumerate(clips_s1):
        frames = load_clip(fname, s, e, slow)
        frames = add_clip_label(frames, label, sub, progress=(idx, len(clips_s1)))
        frames = fade_in(frames, 10)

        if prev_frames is not None:
            # Crossfade with previous clip
            all_frames = crossfade(all_frames, frames, n=12)
        else:
            all_frames += frames

        prev_frames = frames
        print(f"  {label}: {len(frames)} frames")

    # Fade out last clip of section
    if prev_frames:
        all_frames += fade_out([all_frames[-1].copy() for _ in range(10)], 10)[1:]

    # ---- Section 2: Pre-fall lead time ----
    all_frames += section_card("Pre-Fall Lead Time",
                               "Slowed 2x  -  PREFALL fires before body reaches floor")

    clips_s2 = [
        ("fall-14.mp4", 0,  None, 2.0, "fall-14", "733 ms lead (longest)"),
        ("fall-30.mp4", 0,  None, 2.0, "fall-30", "733 ms lead"),
        ("fall-26.mp4", 0,  None, 2.0, "fall-26", "667 ms lead"),
    ]
    prev_frames = None
    for idx, (fname, s, e, slow, label, sub) in enumerate(clips_s2):
        frames = load_clip(fname, s, e, slow)
        frames = add_clip_label(frames, label, sub, progress=(idx, len(clips_s2)))
        frames = fade_in(frames, 10)

        if prev_frames is not None:
            all_frames = crossfade(all_frames, frames, n=12)
        else:
            all_frames += frames

        prev_frames = frames
        print(f"  {label}: {len(frames)} frames")

    if prev_frames:
        all_frames += fade_out([all_frames[-1].copy() for _ in range(10)], 10)[1:]

    # ---- Section 3: Daily activities ----
    all_frames += section_card("Daily Activities",
                               "No false FALLEN events on upright ADL sequences")

    clips_s3 = [
        ("adl-03.mp4", 0, None, 1.0, "adl-03", "clean - STANDING throughout"),
        ("adl-07.mp4", 0, None, 1.0, "adl-07", "clean - STANDING throughout"),
        ("adl-01.mp4", 0, None, 1.0, "adl-01", "brief PREFALL - no FALLEN"),
    ]
    prev_frames = None
    for idx, (fname, s, e, slow, label, sub) in enumerate(clips_s3):
        frames = load_clip(fname, s, e, slow)
        frames = add_clip_label(frames, label, sub, progress=(idx, len(clips_s3)))
        frames = fade_in(frames, 10)

        if prev_frames is not None:
            all_frames = crossfade(all_frames, frames, n=12)
        else:
            all_frames += frames

        prev_frames = frames
        print(f"  {label}: {len(frames)} frames")

    if prev_frames:
        all_frames += fade_out([all_frames[-1].copy() for _ in range(10)], 10)[1:]

    # ---- Stats card ----
    all_frames += section_card("")  # short gap

    metrics = [
        ("Fall detection rate",      "28 / 30  (93.3%)",   C_GREEN,  0.933),
        ("Mean PREFALL lead time",   "415 ms  (167-833)",  C_ORANGE, 0.415 / 0.833),
        ("False FALLEN - upright ADL","0 / 9   (0.0%)",   C_GREEN,  0.0),
        ("False FALLEN - all ADL",   "17 / 40  (42.5%)",  C_RED,    0.425),
    ]

    stats_frames = animated_stat_card(metrics, duration_s=6.0)
    all_frames += fade_in(stats_frames, 20)
    all_frames = fade_out(all_frames, 30)

    # ---- Write output ----
    print(f"\nWriting {len(all_frames)} frames to {args.out}...")

    # Try H.264 codec first (better quality/compatibility), fallback to mp4v
    fourcc_options = [
        ("avc1", "H.264"),  # H.264 (most compatible)
        ("mp4v", "MPEG-4"),  # Fallback
    ]

    writer = None
    for fourcc_code, codec_name in fourcc_options:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(args.out, fourcc, FPS, (OUT_W, OUT_H))
        if writer.isOpened():
            print(f"Using {codec_name} codec ({fourcc_code})")
            break
        writer.release()
        writer = None

    if not writer or not writer.isOpened():
        print("ERROR: Could not initialize video writer")
        return

    for i, f in enumerate(all_frames):
        writer.write(f)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_frames)} frames written...")

    writer.release()

    duration = len(all_frames) / FPS
    print(f"\nSaved {args.out}  ({len(all_frames)} frames, {duration:.1f}s @ {FPS} fps)")


if __name__ == "__main__":
    main()
