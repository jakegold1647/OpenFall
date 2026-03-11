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

All results below are from running the 2D detector (`python run.py <video> --split-rgb --no-display`) on UR Fall Detection Dataset `cam0` sequences.  No parameter tuning was performed between sequences.

### Fall sequences — detection results

| Video | Frames | PREFALL frame | FALLEN frame | Lead time | Notes |
|---|---|---|---|---|---|
| fall-01-cam0 | 160 | 95 (3.17 s) | 107 (3.57 s) | **400 ms** | clean STANDING → PREFALL → FALLEN |
| fall-02-cam0 | 110 | 54 (1.80 s) | 60 (2.00 s) | **200 ms** | fast fall; clean transition |
| fall-03-cam0 | 215 | 157 (5.23 s) | 182 (6.07 s) | **833 ms** | early stumble triggers brief PREFALL at frame 123 (returns to STANDING); final fall correctly detected |
| fall-04-cam0 | 96 | 16 (0.53 s) | 28 (0.93 s) | **400 ms** | very rapid fall; brief STANDING flicker at frame 44 while subject is already on ground, immediately re-enters FALLEN at frame 45 |
| fall-05-cam0 | 151 | 93 (3.10 s) | 103 (3.43 s) | **333 ms** | clean STANDING → PREFALL → FALLEN |

**Detection rate: 5 / 5 (100%)**
**False FALLEN events: 0 / 5**
**Mean PREFALL lead time: 433 ms** (range 200–833 ms)

### Activities of daily living — false-positive audit

| Video | Frames | PREFALL episodes | Max PREFALL duration | False FALLEN |
|---|---|---|---|---|
| adl-01-cam0 | 150 | 1 (frames 106–118) | 433 ms | 0 |
| adl-02-cam0 | 180 | 2 (frames 129–130, 132–133) | 67 ms | 0 |

Brief PREFALL episodes during ADL are consistent with the subject leaning forward or bending — behaviours that transiently satisfy the body-angle criterion.  Neither sequence produced a FALLEN event.

**False FALLEN rate: 0 / 2 (0%)**

> **Scope note.** This evaluation covers 7 of the 70 sequences in the UR dataset.  Systematic evaluation across all 30 fall + 40 ADL sequences, across different fall directions and subject demographics, is needed before clinical deployment.

---

## Quickstart

### Local

```bash
git clone <repo>
cd prefall

python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# MediaPipe model is fetched automatically on first run.
# Download UR Fall Dataset test videos (optional):
python scripts/download_samples.py

# Launch — auto-detects Kinect, 3-camera, or single webcam
python run.py

# Single webcam (index 0), 2D only
python run.py 0 --no-depth

# Three-camera setup (indices 0, 1, 2)
python run.py 0 1 2

# Video file with Depth Anything V2 and depth panel
python run.py data/videos/fall-01-cam0.mp4 --split-rgb --show-depth --save out.mp4
```

### Docker

```bash
docker build -t prefall .

docker run --rm \
  -v $(pwd)/data/videos:/app/data/videos \
  prefall data/videos/fall-01-cam0.mp4 --split-rgb --no-display
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
