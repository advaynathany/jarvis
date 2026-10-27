"""
Camera + MediaPipe hand landmarker, and the geometry helpers built on it.
"""

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(ROOT, "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Wrist + the four knuckles. None of these move when you pinch, so the
# cursor stays on target while clicking.
PALM_POINTS = (0, 5, 9, 13, 17)


@dataclass
class Hand:
    lm: list            # 21 normalized landmarks
    label: str          # "Left" / "Right" (as seen in the mirrored frame)
    palm: tuple         # stable palm point, camera pixels
    pinch_index: float  # thumb-index distance / hand scale
    pinch_middle: float


# ============================================================
# Geometry helpers
# ============================================================
def to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def normalized_distance(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def hand_scale(lm):
    return max(normalized_distance(lm[0], lm[9]), 1e-6)


def palm_point(lm, w, h):
    xs = sum(lm[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    ys = sum(lm[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    return xs * w, ys * h


def fingers_extended(lm, label="Right"):
    """[thumb, index, middle, ring, pinky] booleans."""
    # In the mirrored frame a right hand's thumb points to screen-left.
    thumb = lm[4].x < lm[3].x if label == "Right" else lm[4].x > lm[3].x
    fingers = [thumb]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers


def draw_hand(frame, lm, w, h, color=(0, 200, 255)):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, to_pixel(lm[a], w, h), to_pixel(lm[b], w, h),
                 (200, 200, 200), 2)
    for p in lm:
        cv2.circle(frame, to_pixel(p, w, h), 4, color, cv2.FILLED)


# ============================================================
# Camera selection
# ============================================================
def resolve_camera(camera):
    """Turn a camera setting into an OpenCV index.

    "builtin" picks the Mac's own camera; any other string matches part of a
    camera's name; an int is used as-is. Indices shift when an iPhone joins
    as a Continuity Camera, so a name is the reliable way to choose."""
    if isinstance(camera, int):
        return camera
    import AVFoundation as AV
    # Same enumeration, in the same order, as OpenCV's AVFoundation backend
    devices = (list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
               + list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)))
    names = [str(d.localizedName()) for d in devices]
    for i, d in enumerate(devices):
        if camera == "builtin":
            if d.deviceType() == AV.AVCaptureDeviceTypeBuiltInWideAngleCamera:
                return i
        elif camera.lower() in names[i].lower():
            return i
    raise RuntimeError(f"No camera matching {camera!r}. Cameras: {', '.join(names) or 'none'}")


# ============================================================
# Tracker
# ============================================================
class HandTracker:
    def __init__(self, camera="builtin", num_hands=2):
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading hand landmark model to {MODEL_PATH} ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        camera_index = resolve_camera(camera)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.cap.isOpened() or self.w == 0:
            raise RuntimeError(
                f"Camera {camera_index} didn't open. Check Camera permission for "
                "your terminal, or set tracking.camera in jarvis.toml.")
        print(f"Camera {camera_index}: {camera}")

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.start = time.time()
        self.last_ts = -1

    def read(self):
        """Returns (frame, hands, t) or (None, [], t) if the camera stalled."""
        ret, frame = self.cap.read()
        t = time.time()
        if not ret:
            return None, [], t
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((t - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        hands = []
        for i, lm in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name if result.handedness else "Right"
            scale = hand_scale(lm)
            hands.append(Hand(
                lm=lm, label=label,
                palm=palm_point(lm, self.w, self.h),
                pinch_index=normalized_distance(lm[4], lm[8]) / scale,
                pinch_middle=normalized_distance(lm[4], lm[12]) / scale,
            ))
        return frame, hands, t

    def close(self):
        self.cap.release()
        self.landmarker.close()
