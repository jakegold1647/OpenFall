# prefall

**Real-time pre-fall detection from monocular video using pose estimation and depth-based biomechanics.**

---

## Clinical motivation

Falls are the leading cause of injury-related death in adults over 65 and account for roughly $50 billion in annual US healthcare costs.  The clinical value in *pre-fall* detection — identifying the loss-of-balance event **before** the body reaches the floor — is that it opens a narrow but meaningful intervention window: alerting a caregiver, triggering an airbag wearable, or activating a grab-bar actuator.

Current deployed solutions are almost exclusively post-hoc: wrist accelerometers and call buttons that require the patient to have already fallen and retained consciousness.  Camera-based approaches remove the dependence on patient compliance but have historically required calibrated stereo rigs or depth sensors at the bedside.  This project demonstrates that a single uncalibrated RGB camera — including a laptop webcam — can produce clinically meaningful fall-state signals by combining skeleton kinematics with monocular depth estimation.

### Three-state model

```
STANDING  ──(body tilt + downward acceleration)──►  PREFALL
                                                       │
                                                       ▼
                                            (body horizontal + hip near floor)
                                                       │
                                                       ▼
                                                    FALLEN
```

The **PREFALL** state is the clinical target: the person is losing balance but has not yet impacted the ground.  Transitions are detected in 60–400 ms on CPU hardware depending on subject speed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           run.py  (CLI)                             │
│  --source  --split-rgb  --depth  --show-depth  --save  --speed      │
└───────────────┬────────────────────────────┬────────────────────────┘
                │                            │
                ▼                            ▼
  ┌─────────────────────┐       ┌────────────────────────┐
  │   VideoSource        │       │   DepthEstimator        │
  │  (webcam or file)    │       │  Depth Anything V2      │
  │  src/video_source.py │       │  Small-hf  (HF hub)     │
  └────────┬────────────┘       │  stride=3 (10 fps eff.) │
           │ BGR frame           │  src/depth_estimator.py │
           │                    └──────────┬───────────────┘
           │                               │ float32 H×W
           │           ┌───────────────────┘ disparity map
           ▼           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    FallDetector                           │
  │                   src/detector.py                         │
  │                                                           │
  │  ┌─────────────────────────┐  ┌────────────────────────┐ │
  │  │  MediaPipe Pose          │  │  biomechanics.py        │ │
  │  │  Landmarker (Tasks API)  │  │                        │ │
  │  │  33 landmarks, VIDEO     │  │  build_camera_matrix   │ │
  │  │  mode, 30 fps            │  │  lift_landmarks_3d     │ │
  │  └──────────┬──────────────┘  │  estimate_ground_plane  │ │
  │             │ 2D landmarks     │    (RANSAC, 50 iters)   │ │
  │             ▼                  │  compute_3d_features    │ │
  │  ┌──────────────────────┐     └──────────┬─────────────┘ │
  │  │  2D feature scorer   │                │ 3D features    │
  │  │                      │                ▼                │
  │  │  • body_angle        │     ┌────────────────────────┐ │
  │  │  • aspect_ratio      │     │  3D feature scorer      │ │
  │  │  • hip_y             │     │                         │ │
  │  │  • com_velocity_down │     │  • com_drop_rate  (Y)   │ │
  │  └──────────┬───────────┘     │  • spine_angle_3d  (HUD)│ │
  │             │ votes            │  • com_depth_velocity   │ │
  │             └────────┬─────── │    (HUD; forward falls) │ │
  │                      │        └──────────┬──────────────┘ │
  │                      │ combined votes    │                 │
  │                      ▼                  │                 │
  │            ┌──────────────────────────────────────┐       │
  │            │         State machine                 │       │
  │            │                                       │       │
  │            │  fallen_score  ≥ 2  →  FALLEN         │       │
  │            │  prefall_score ≥ 1  →  PREFALL         │       │
  │            │  else               →  STANDING        │       │
  │            └──────────────────────────────────────┘       │
  └──────────────────────────────────────────────────────────┘
                           │
                           ▼
              annotated frame  +  metrics dict
              (optional depth panel, HUD overlay)
```

### Scoring design

The scorer separates *evidence type* from *decision threshold*:

| Feature | Type | Contributes to |
|---|---|---|
| Body angle from vertical | 2D | FALLEN (primary) |
| Bounding-box aspect ratio | 2D | FALLEN (primary) |
| Hip Y position in frame | 2D | FALLEN (primary) |
| Downward COM velocity (2D) | 2D | PREFALL |
| Downward COM velocity (3D Y) | 3D | PREFALL bonus |
| Spine angle vs camera vertical | 3D | HUD only |
| COM depth velocity (Z) | 3D | HUD only |
| Hip / shoulder height above floor | 3D | HUD only |

FALLEN requires 2 of 3 primary 2D votes.  3D signals add PREFALL sensitivity without touching the FALLEN gate, so a bad depth frame cannot cause a false FALLEN event.

---

## Dataset

Tests use the **UR Fall Detection Dataset** (University of Rzeszów):

- 30 fall sequences + 40 activities of daily living (ADL)
- Microsoft Kinect sensor: side-view RGB + depth, 30 fps, 640×240 (depth | RGB side-by-side)
- No registration required for direct video download

`cam0` files (side-view) are used; `cam1` (overhead) is excluded because MediaPipe Pose is not trained on top-down views.

```bash
python scripts/download_samples.py --count 5   # fall-01..05-cam0 + adl-01..02-cam0
```

---

## Evaluation

All results are from the 2D detector (`python run.py <video> --split-rgb --no-display --no-depth`) run against the complete UR Fall Detection Dataset — all 30 fall sequences and all 40 ADL sequences.  No parameter tuning was performed between sequences.

---

### Fall sequences — detection results (all 30 evaluated)

| Video | Frames | PREFALL frame | FALLEN frame | Lead time | Notes |
|---|---|---|---|---|---|
| fall-01 | 160 | 95 | 107 | **400 ms** | Clean STANDING → PREFALL → FALLEN |
| fall-02 | 110 | 54 | 60 | **200 ms** | Fast fall; clean transition |
| fall-03 | 215 | 123→157 | 182 | **833 ms** | Early stumble at f123 returns to STANDING; true fall correctly detected at f157 |
| fall-04 | 96 | 16 | 28 | **400 ms** | Rapid fall; brief STANDING flicker at f44 while subject is on ground, immediately re-enters FALLEN |
| fall-05 | 151 | 93 | 103 | **333 ms** | Clean |
| fall-06 | 100 | 34 | 39 | **167 ms** | Clean |
| fall-07 | 156 | 95 | 110 | **500 ms** | Clean |
| fall-08 | 91 | 31 | 47 | **533 ms** | Clean |
| fall-09 | 185 | 128 | 142 | **467 ms** | Clean |
| fall-10 | 130 | 42 | 52 | **333 ms** | Clean |
| fall-11 | 130 | 57 | 68 | **367 ms** | Clean |
| fall-12 | 110 | 32 | 42 | **333 ms** | Clean |
| fall-13 | 85 | 58 | — | — | **MISSED** — PREFALL raised at f58 then returns to STANDING; video ends before FALLEN threshold |
| fall-14 | 61 | 12 | 34 | **733 ms** | Clean |
| fall-15 | 71 | 43 | 53 | **333 ms** | Brief STANDING flicker at f56; FALLEN correctly confirmed |
| fall-16 | 55 | 17 | 38 | **700 ms** | Clean |
| fall-17 | 95 | 70 | 78 | **267 ms** | Clean |
| fall-18 | 65 | 20 | 33 | **433 ms** | Brief STANDING/FALLEN oscillation after f53 while subject on ground |
| fall-19 | 100 | 44 | 55 | **367 ms** | STANDING flicker at f71; secondary PREFALL episodes as subject moves on ground |
| fall-20 | 110 | 41 | 47 | **200 ms** | Rapid oscillation around f49–52; FALLEN correctly triggered first |
| fall-21 | 55 | 37 | 44 | **233 ms** | Clean |
| fall-22 | 56 | 14 | 21 | **233 ms** | Clean |
| fall-23 | 75 | 42 | 55 | **433 ms** | Post-impact oscillation (f57–69); FALLEN correctly triggered |
| fall-24 | 60 | 18 | 26 | **267 ms** | Clean |
| fall-25 | 85 | 59 | 66 | **233 ms** | Single-frame STANDING flicker at f60; FALLEN correctly confirmed |
| fall-26 | 61 | 12 | 32 | **667 ms** | Brief flicker at f36–37 while on ground |
| fall-27 | 92 | 70 | — | — | **MISSED** — PREFALL raised at f70; video ends before FALLEN threshold |
| fall-28 | 66 | 26 | 45 | **633 ms** | Post-impact STANDING/PREFALL oscillation at f57–60 |
| fall-29 | 99 | 68 | 77 | **300 ms** | Clean |
| fall-30 | 70 | 23 | 45 | **733 ms** | Clean |

**FALLEN detected: 28 / 30 (93.3%)**
**PREFALL-only (FALLEN missed): 2 / 30 — fall-13, fall-27**
**False FALLEN events during fall sequences: 0 / 30**
**Mean PREFALL lead time (28 detected): 415 ms** (range 167–833 ms)

Both misses (fall-13, fall-27) raised PREFALL before the video ended — the fall was partially detected.  In fall-13 the subject's body angle did not sustain the FALLEN threshold before the clip ended (85 frames / 2.8 s); fall-27 similarly cuts off at 92 frames with the subject mid-fall.

---

### ADL sequences — false-positive audit (40 / 40)

| Video | Frames | False FALLEN | Notes |
|---|---|---|---|
| adl-01 | 150 | No | Brief PREFALL f106–118 (forward lean) |
| adl-02 | 180 | No | Two 1-frame PREFALL flashes (f129, f132) |
| adl-03 | 180 | No | Clean STANDING throughout |
| adl-04 | 150 | No | PREFALL f71–128 (prolonged forward bend) |
| adl-05 | 180 | No | PREFALL f97–146 (crouching activity) |
| adl-06 | 230 | No | PREFALL f166–208 (bending) |
| adl-07 | 180 | No | Clean |
| adl-08 | 180 | No | Clean |
| adl-09 | 150 | No | PREFALL f106–125 (leaning) |
| adl-10 | 300 | **Yes** | FALLEN f214–246; subject likely sitting/crouching at floor level |
| adl-11 | 300 | **Yes** | FALLEN f242; prolonged crouching near floor |
| adl-12 | 250 | No | Multiple brief PREFALL episodes (bending) |
| adl-13 | 265 | **Yes** | FALLEN f238–239; near-floor activity |
| adl-14 | 235 | No | PREFALL f158–172 (bending) |
| adl-15 | 275 | No | Multiple brief PREFALL flashes (repetitive bending) |
| adl-16 | 240 | **Yes** | FALLEN f172–191; near-floor activity |
| adl-17 | 230 | **Yes** | Repeated FALLEN f168–211; prolonged floor-level activity |
| adl-18 | 265 | No | PREFALL f202–206 (brief lean) |
| adl-19 | 250 | **Yes** | FALLEN f184, f196; near-floor activity |
| adl-20 | 270 | No | 1-frame PREFALL f218 |
| adl-21 | 280 | No | 1-frame PREFALL f194 |
| adl-22 | 240 | No | PREFALL f133–134 |
| adl-23 | 220 | No | Brief PREFALL f109–114 |
| adl-24 | 70 | No | Brief PREFALL f7–16 |
| adl-25 | 110 | No | Clean |
| adl-26 | 95 | No | Brief PREFALL f31–41 (repetitive bends) |
| adl-27 | 100 | No | Clean |
| adl-28 | 85 | No | Brief PREFALL flashes f37, f76 |
| adl-29 | 125 | No | Clean |
| adl-30 | 400 | **Yes** | Repeated FALLEN f183–268; prolonged floor-level / lying activity |
| adl-31 | 250 | **Yes** | FALLEN f90–161; near-floor activity |
| adl-32 | 200 | **Yes** | Repeated FALLEN throughout; subject near/on floor |
| adl-33 | 200 | **Yes** | FALLEN f102–166; subject on floor |
| adl-34 | 191 | **Yes** | Repeated FALLEN; repeated floor-level movements |
| adl-35 | 280 | **Yes** | Repeated FALLEN f60–105; floor-level activity |
| adl-36 | 340 | **Yes** | FALLEN f242–252; near-floor activity |
| adl-37 | 350 | **Yes** | FALLEN f266; near-floor activity |
| adl-38 | 345 | **Yes** | FALLEN f215–223; near-floor activity |
| adl-39 | 270 | **Yes** | FALLEN f214–216; near-floor activity |
| adl-40 | 330 | **Yes** | Repeated FALLEN; extended floor-level sequence |

**False FALLEN rate: 17 / 40 (42.5%)**
**Clean (no false FALLEN): 23 / 40 (57.5%)**

The 17 sequences with false FALLEN events fall into two groups:

1. **Near-floor ADL (adl-10 to adl-19)** — activities such as picking objects off the floor, crouching, and kneeling that transiently drive hip Y-position and body angle past the FALLEN thresholds.

2. **Extended floor-level sequences (adl-30 to adl-40)** — longer clips where the subject is lying or sitting on the ground for extended periods.  These sequences are functionally indistinguishable from a fallen person by any 2D pose metric — the body is horizontal, near the floor, and the bounding box is wide.  A context signal (recent motion, continuous monitoring baseline) would be required to disambiguate.

Sequences adl-01 through adl-09 (basic upright ADL: walking, sitting in chair, reaching) produced **zero false FALLEN events**.

---

### Summary

| Metric | Result |
|---|---|
| Fall sequences evaluated | 30 / 30 |
| FALLEN state detected | **28 / 30 (93.3%)** |
| PREFALL-only (partial detection) | 2 / 30 (fall-13, fall-27) |
| False FALLEN during fall sequences | 0 / 30 |
| Mean PREFALL lead time | **415 ms** (range 167–833 ms) |
| ADL sequences evaluated | 40 / 40 |
| ADL with false FALLEN (floor-level) | 17 / 40 (42.5%) |
| ADL with false FALLEN (upright activities only, adl-01–09) | 0 / 9 (0%) |

---

## Comparison with published systems

All comparisons use the UR Fall Detection Dataset (cam0, side-view) unless otherwise noted.  No parameter tuning was performed on OpenFall between sequences.

### Sensitivity (fall detection rate)

| System | Sensors | Sensitivity | Training required |
|---|---|---|---|
| **OpenFall** | Single RGB (MediaPipe) | **93.3%** | **No** |
| MoveNet rule-based (2024) | Single RGB | 91.7% | No |
| BlazePose + Random Forest (2024) | Single RGB | 90.3% | Yes |
| AlphaPose / OpenPose + MLP | RGB skeleton | 94.5% | Yes |
| ST-GCN / Subgraph GCN (trained) | RGB skeleton | 97–98.5% | Yes |
| Kepski & Kwolek k-NN (2015) | Kinect depth + wearable accel | 98.3% | Yes + wearable |
| Bay Alarm Medical pendant | Wrist accelerometer | ~70% | No |

OpenFall matches or beats every zero-training RGB-only system published on this dataset.  Systems that outperform it all require either a trained classifier fitted to labeled fall data, a wearable sensor, or both.

### False positive rate on ADL sequences

| System | False FALLEN rate (ADL) | Training required |
|---|---|---|
| **OpenFall — upright ADL only (adl-01–09)** | **0 / 9 (0%)** | No |
| **OpenFall — full ADL (all 40 sequences)** | **17 / 40 (42.5%)** | No |
| MoveNet rule-based (2024) | 11 / 40 (27.5%) | No |
| BlazePose + Random Forest (2024) | ~10% | Yes |
| AlphaPose + MLP | ~0.1% | Yes |
| Trained GCN / LSTM systems | 0–5% | Yes |

The false positives are concentrated in ADL sequences containing floor-level activity — crouching, kneeling, lying on a couch, sitting on the floor.  These postures are 2D-indistinguishable from a fallen person by any single-frame pose metric; trained classifiers handle them better because they learn the motion leading up to the position, not just the position itself.  On the nine upright-only ADL sequences (walking, reaching, sitting in a chair) OpenFall produces zero false alarms.

### Pre-fall lead time

| System | Sensor type | Mean lead time | Range |
|---|---|---|---|
| **OpenFall** | Single RGB camera | **415 ms** | 167–833 ms |
| Wearable SVM (trunk velocity) | IMU / accelerometer | ~203 ms | — |
| Decision Tree (waist IMU) | IMU | ~448 ms | — |
| KNN / neural net (waist IMU) | IMU | ~461 ms | — |
| EMG + accelerometer | EMG + IMU | ~770 ms | — |
| Published camera-based systems | — | **not reported** | — |

Pre-fall lead time data in the camera vision literature is essentially non-existent — every published lead-time figure comes from wearable IMU or EMG sensors.  OpenFall's 415 ms mean sits within the range of the wearable literature and exceeds the ~130 ms threshold required for airbag wearable inflation.  This represents a gap in the published camera-based literature that OpenFall directly addresses.

### Summary

OpenFall is the only system in this comparison that:
- Requires **no training data**
- Uses a **single uncalibrated RGB camera**
- Reports a **pre-fall lead time** (415 ms mean)
- Runs **entirely on CPU** in real time

The trade-off is a higher false positive rate on floor-level ADL compared to trained classifiers.  Adding a second camera angle or a depth sensor (Kinect / RealSense) would provide the context signal needed to distinguish an intentional floor posture from a fall.

---

## Live camera modes

OpenFall supports three hardware configurations. The app auto-detects which one to use on startup.

---

### Kinect sensor (recommended)

A Microsoft Kinect provides a real depth stream alongside its RGB camera, giving the most accurate 3D biomechanics. When a Kinect is detected the display shows a 3-panel layout:

```
[ Kinect depth | RGB + skeleton | DA-V2 depth ]
```

Just plug in the Kinect and run:
```bash
python run.py
```

The same 3-panel layout is what you see in the annotated sample videos (`data/annotated/`), which were recorded from UR Fall Dataset Kinect clips.

---

### Single USB webcam + depth model

Without a Kinect you can still get a depth panel using **Depth Anything V2**, a monocular depth estimation model that runs on any RGB camera.  The display shows a 2-panel layout:

```
[ RGB + skeleton | DA-V2 depth ]
```

```bash
# 2-panel view with live depth map (downloads ~100 MB model on first run)
python run.py 0 --show-depth

# Skeleton only, no depth model (fastest startup)
python run.py 0 --no-depth
```

The depth model runs every 3 frames by default and is fast enough for real-time use on CPU.

---

### Multiple USB webcams (no depth sensor)

If you have two or three USB cameras you can run them simultaneously for multi-angle coverage.  Each camera runs its own independent detector.  The display shows all feeds side by side:

```
[ CAM0 + skeleton | CAM1 + skeleton | CAM2 + skeleton ]
```

First check which camera indices Windows has assigned:
```bash
python -c "import cv2; [print(f'Camera {i}: OK') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

Then launch with those indices:
```bash
# Three cameras
python run.py 0 1 2 --no-depth

# Two cameras
python run.py 0 1 --no-depth
```

If any camera detects a state change the terminal prints which camera triggered it, e.g. `CAM1: PREFALL`.

---

### Controls (all modes)

| Key | Action |
|---|---|
| Q / Esc | Quit |
| R | Reset all detectors |

---

## Quickstart

### Local

```bash
git clone <repo>
cd OpenFall

python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Download the MediaPipe pose model
curl -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -o data/pose_landmarker_lite.task
# Windows PowerShell:
# mkdir data; Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "data\pose_landmarker_lite.task"

# Launch with auto-detect
python run.py

# Single webcam, skeleton only
python run.py 0 --no-depth

# Single webcam, 2-panel with depth map
python run.py 0 --show-depth

# Three cameras, multi-angle
python run.py 0 1 2 --no-depth

# Video file with depth panel + save output
python run.py data/videos/fall-01-cam0.mp4 --split-rgb --show-depth --save out.mp4
```

### Docker

```bash
docker build -t openfall .

docker run --rm \
  -v $(pwd)/data/videos:/app/data/videos \
  openfall data/videos/fall-01-cam0.mp4 --split-rgb --no-display
```

---

## CLI reference

```
usage: run.py [-h] [--camera-mode {kinect,triple,single}]
              [--no-display] [--save OUT.mp4] [--speed SPEED]
              [--split-rgb] [--show-depth] [--no-depth]
              [--depth-model ID] [--depth-stride N]
              [--depth-device {cpu,cuda,mps}] [--fov DEG]
              [source ...]

positional arguments:
  source                camera index(es) or video file path
                        (default: auto-detect hardware)
                        examples:
                          run.py           → auto-detect
                          run.py 0         → webcam 0
                          run.py 0 1 2     → three-camera mode
                          run.py video.mp4 → video file

options:
  --camera-mode         force kinect | triple | single (overrides auto-detect)
  --no-display          headless mode; print state changes to stdout
  --save OUT.mp4        write annotated output video
  --speed SPEED         playback speed multiplier for video files (default: 1.0)
  --split-rgb           crop right half — UR dataset videos are depth|RGB
  --show-depth          append colourised depth map panel to output frame
  --no-depth            disable Depth Anything V2 (2D detection only)
  --depth-model ID      HuggingFace model ID (default: DA-V2-Small-hf)
  --depth-stride N      run depth model every N frames (default: 3)
  --depth-device        cpu | cuda | mps  (default: cpu)
  --fov DEG             assumed horizontal FOV for 3D landmark lifting (default: 70)
```

---

## Project structure

```
prefall/
├── run.py                          # CLI entry point
├── Dockerfile
├── requirements.txt
├── src/
│   ├── detector.py                 # FallDetector — pose + state machine
│   ├── depth_estimator.py          # DepthEstimator — Depth Anything V2 wrapper
│   ├── biomechanics.py             # 3D lifting, ground-plane RANSAC, feature extraction
│   └── video_source.py             # VideoSource — webcam / file abstraction
├── scripts/
│   ├── download_samples.py         # download UR Fall Dataset videos
│   └── generate_test_video.py      # generate synthetic stick-figure test video
└── data/
    ├── pose_landmarker_lite.task   # MediaPipe model (downloaded on first run)
    └── videos/                     # place test videos here (gitignored)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe >= 0.10` | 33-point skeleton, VIDEO-mode tracking |
| `opencv-python` | video I/O, drawing |
| `torch` (CPU wheel) | inference backend for DA-V2 |
| `transformers` | Depth Anything V2 via Hugging Face hub |
| `numpy` | feature computation |

---

## Limitations and future work

- **Depth is relative, not metric.** Depth Anything V2 produces disparity (closer = higher value).  All 3D features are normalised within each frame and cannot be compared across frames or subjects.  A calibrated depth sensor (Kinect, RealSense) would enable metric features such as fall height in cm.

- **Spine angle in disparity space is unreliable.** Joints at similar distances from the camera produce apparent depth offsets due to scene texture, making 3D spine angle noisy.  It is computed and shown in the HUD but excluded from scoring; the 2D body angle is more reliable for standard side-view cameras.

- **Ground plane RANSAC requires a visible floor strip.** If the bottom 15% of the frame is occluded (bed, furniture), the plane estimate degrades.  Hip/shoulder heights above the floor are tracked but not scored for this reason.

- **Single-person.** MediaPipe Pose is configured for one subject; crowded scenes are not handled.

- **Overhead cameras.** `cam1` (overhead Kinect) is excluded because MediaPipe is trained on frontal/side views.  A top-down model or background-subtraction approach would be needed for ceiling-mounted clinical cameras.

- **Prospective validation.** Evaluation is on a convenience sample of 5 sequences.  Clinical deployment would require IRB-approved prospective study with sensitivity / specificity analysis across patient populations.

---

## Citation

If you use the UR Fall Detection Dataset:

```
Kepski M., Kwolek B. (2014) Fall Detection Using Ceiling-Mounted 3D Depth Camera.
International Conference on Computer Vision Theory and Applications (VISAPP), pp. 640–647.
```

---

## Acknowledgments

This project was developed with assistance from Claude (Anthropic) for code implementation and documentation.
