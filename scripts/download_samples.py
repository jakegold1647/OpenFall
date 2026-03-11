"""
Download sample fall detection videos from the UR Fall Detection Dataset.
  https://fenix.ur.edu.pl/~mkepski/ds/uf.html  (University of Rzeszow)

Videos are 640x240 MP4 with depth on the left half and RGB on the right half.
Use --split-rgb when running the detector:
  python run.py data/videos/fall-01-cam0.mp4 --split-rgb

Sequences:
  fall-XX-cam0.mp4  : 30 fall events, side-view colour+depth  <- best for pose detection
  fall-XX-cam1.mp4  : 30 fall events, overhead colour+depth
  adl-XX-cam0.mp4   : 40 daily activity (non-fall) sequences

Usage:
    python scripts/download_samples.py
    python scripts/download_samples.py --count 5   # download 5 fall + 2 ADL videos
"""
import argparse
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
UR_BASE  = "https://fenix.ur.edu.pl/~mkepski/ds/data"


def download_file(url, dest_path, timeout=30):
    print(f"  {os.path.basename(dest_path)} ... ", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
             open(dest_path, "wb") as f:
            size = 0
            while chunk := resp.read(65536):
                f.write(chunk)
                size += len(chunk)
        print(f"{size // 1024} KB")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3,
                        help="Number of fall sequences to download (default: 3)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Build download list: cam0 fall sequences (side-view, best for pose detection)
    # + a couple of ADL (non-fall) baseline videos
    targets = []
    for i in range(1, args.count + 1):
        targets.append(f"fall-{i:02d}-cam0.mp4")
    targets += ["adl-01-cam0.mp4", "adl-02-cam0.mp4"]

    print(f"Downloading {len(targets)} videos from UR Fall Detection Dataset...")
    ok, skipped = 0, 0
    for fname in targets:
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  {fname} already exists, skipping")
            skipped += 1
            continue
        if download_file(f"{UR_BASE}/{fname}", dest):
            ok += 1

    print(f"\n{ok} downloaded, {skipped} already present.")
    print("\nAvailable test videos:")
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith((".mp4", ".avi")):
            continue
        path = os.path.join(DATA_DIR, f)
        size_kb = os.path.getsize(path) // 1024
        print(f"  {f}  ({size_kb} KB)")

    print("\nTo run the detector:")
    print("  python run.py data/videos/fall-01-cam0.mp4 --split-rgb")
    print("  python run.py data/videos/fall-01-cam0.mp4 --split-rgb --save out.mp4")


if __name__ == "__main__":
    main()
