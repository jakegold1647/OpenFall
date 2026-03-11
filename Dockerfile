FROM python:3.12-slim

WORKDIR /app

# System deps: libGL for OpenCV headless, libglib for mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (avoids pulling the large CUDA wheel)
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY run.py .

# Pre-download the MediaPipe pose model and Depth Anything V2 weights at build time
# so the container works offline at runtime.
RUN mkdir -p data && \
    python - <<'EOF'
import urllib.request, os
url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
dest = "data/pose_landmarker_lite.task"
if not os.path.exists(dest):
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {os.path.getsize(dest)//1024} KB -> {dest}")
EOF

RUN python - <<'EOF'
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
model_id = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"Caching {model_id} ...")
AutoImageProcessor.from_pretrained(model_id)
AutoModelForDepthEstimation.from_pretrained(model_id)
print("Done.")
EOF

# Video files are mounted at runtime — keep data/videos outside the image
VOLUME ["/app/data/videos"]

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
