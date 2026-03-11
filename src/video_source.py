"""Video source abstraction: supports webcam index or video file path."""
import cv2


class VideoSource:
    def __init__(self, source):
        """
        Args:
            source: int for webcam index (e.g. 0), or str path to video file.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def total_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        """Returns (ok, frame). Frame is BGR numpy array."""
        return self.cap.read()

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __iter__(self):
        consecutive_failures = 0
        while True:
            ok, frame = self.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    break  # give up after 30 consecutive failures
                continue
            consecutive_failures = 0
            yield frame
