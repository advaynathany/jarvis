"""
Layer 1: what is the hand doing?

  - Pinches are geometric (thumb-tip distance with hysteresis) because click
    feel depends on that on/off gap. The thresholds come from calibration.
  - Static poses (palm, fist, grab, peace, plus anything you record) come from
    a small classifier trained on your own hand, with the original hand-written
    rules as a fallback until you've trained one.
  - Swipes are motion, not pose: a fast flick of the cursor hand.
  - Two-hand gestures: both hands pinching -> zoom / rotate.
"""

import json
import math
import os
from collections import deque

import numpy as np

from .tracking import ROOT, fingers_extended

PROFILE_DIR = os.path.join(ROOT, "profile")
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")
SAMPLES_PATH = os.path.join(PROFILE_DIR, "samples.npz")
MODEL_PATH = os.path.join(PROFILE_DIR, "gesture_model.npz")

POSE_NONE         = "none"
POSE_OPEN         = "open_palm"     # cursor movement
POSE_FIST         = "fist"          # escape / abort (thumb extended out)
POSE_GRAB         = "grab"          # all 5 fingers closed — grab & move window
POSE_PEACE        = "peace"         # scroll
POSE_INDEX_PINCH  = "index_pinch"   # left click/drag
POSE_MIDDLE_PINCH = "middle_pinch"  # right click
POSE_NEUTRAL      = "neutral"       # something else

BUILTIN_POSES = [POSE_OPEN, POSE_FIST, POSE_GRAB, POSE_PEACE, POSE_NEUTRAL]
CLASSIFIER_MIN_CONFIDENCE = 0.6

POSE_CONFIRM_FRAMES = 2
CUSTOM_CONFIRM_FRAMES = 4   # trained one-shot gestures fire actions, so be surer


# ============================================================
# Pinch thresholds (calibrated)
# ============================================================
DEFAULT_PINCH = {"index_on": 0.065, "index_off": 0.11,
                 "middle_on": 0.065, "middle_off": 0.11}


def load_calibration():
    try:
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
        return {**DEFAULT_PINCH, **data.get("pinch", {})}, data
    except (OSError, ValueError):
        return dict(DEFAULT_PINCH), None


def thresholds_from_samples(pinched, relaxed):
    """Place on/off between the user's pinched and open distances.

    on  = 25% of the way from pinched to open (deliberate pinch needed)
    off = 55% of the way (hysteresis so a held pinch doesn't flicker)."""
    if len(pinched) < 10 or len(relaxed) < 10:
        return None
    lo = float(np.percentile(pinched, 90))
    hi = float(np.percentile(relaxed, 10))
    gap = hi - lo
    if gap < 0.05:
        return None
    return lo + 0.25 * gap, lo + 0.55 * gap


class PinchDetector:
    def __init__(self, thresholds):
        self.t = thresholds
        self.index = False
        self.middle = False

    def update(self, hand):
        t = self.t
        self.index = hand.pinch_index < (t["index_off"] if self.index else t["index_on"])
        self.middle = hand.pinch_middle < (t["middle_off"] if self.middle else t["middle_on"])
        if self.index and self.middle:
            return POSE_INDEX_PINCH if hand.pinch_index <= hand.pinch_middle else POSE_MIDDLE_PINCH
        if self.index:
            return POSE_INDEX_PINCH
        if self.middle:
            return POSE_MIDDLE_PINCH
        return None


# ============================================================
# Rule-based poses (fallback before training)
# ============================================================
def rule_pose(hand):
    f = fingers_extended(hand.lm, hand.label)
    # f = [thumb, index, middle, ring, pinky]

    # Peace: index + middle up, ring + pinky down (regardless of thumb)
    if f[1] and f[2] and not f[3] and not f[4]:
        return POSE_PEACE

    # Fist/Grab: zero non-thumb fingers up
    # Differentiate by thumb: thumb extended = fist (escape), thumb tucked = grab (window drag)
    if sum(f[1:]) == 0:
        return POSE_FIST if f[0] else POSE_GRAB

    # Open palm: 3+ of the 4 non-thumb fingers up
    if sum(f[1:]) >= 3:
        return POSE_OPEN

    return POSE_NEUTRAL


# ============================================================
# Learned poses
# ============================================================
def hand_features(lm, label="Right"):
    """63-d vector: landmarks relative to the wrist, scaled by hand size, with
    left hands mirrored so one model serves both."""
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    pts -= pts[0]
    pts /= max(float(np.linalg.norm(pts[9, :2])), 1e-6)
    if label == "Left":
        pts[:, 0] *= -1
    return pts.reshape(-1)


def augment(X, rng, copies=4):
    """Small in-plane rotations, scale changes and jitter, so a few seconds
    of recording cover how the hand actually varies."""
    out = [X]
    pts = X.reshape(len(X), 21, 3)
    for _ in range(copies):
        a = rng.uniform(-0.3, 0.3, len(X))
        c, s = np.cos(a), np.sin(a)
        rot = pts.copy()
        rot[:, :, 0] = c[:, None] * pts[:, :, 0] - s[:, None] * pts[:, :, 1]
        rot[:, :, 1] = s[:, None] * pts[:, :, 0] + c[:, None] * pts[:, :, 1]
        rot *= rng.uniform(0.9, 1.1, (len(X), 1, 1))
        rot += rng.normal(0, 0.02, rot.shape)
        out.append(rot.reshape(len(X), -1).astype(np.float32))
    return np.concatenate(out)


class GestureClassifier:
    """63 -> 64 (ReLU) -> K softmax, trained with Adam in numpy."""

    def __init__(self, labels, params):
        self.labels = list(labels)
        self.W1, self.b1, self.W2, self.b2, self.mu, self.sd = params

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=False)
        return cls(d["labels"].tolist(),
                   [d[k] for k in ("W1", "b1", "W2", "b2", "mu", "sd")])

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, labels=np.array(self.labels), W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2, mu=self.mu, sd=self.sd)

    def probs(self, X):
        h = np.maximum(0, ((X - self.mu) / self.sd) @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, x):
        p = self.probs(x[None])[0]
        i = int(p.argmax())
        return self.labels[i], float(p[i])

    @classmethod
    def train(cls, X, y, labels, epochs=400, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        n_aug = 4
        X = augment(X, rng, copies=n_aug)
        y = np.tile(y, n_aug + 1)
        mu, sd = X.mean(0), X.std(0) + 1e-4
        Xn = (X - mu) / sd
        k, d = len(labels), X.shape[1]
        Y = np.eye(k)[y]

        params = [rng.normal(0, math.sqrt(2 / d), (d, hidden)), np.zeros(hidden),
                  rng.normal(0, math.sqrt(1 / hidden), (hidden, k)), np.zeros(k)]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        lr, b1, b2, l2 = 0.01, 0.9, 0.999, 1e-4
        for step in range(1, epochs + 1):
            W1, c1, W2, c2 = params
            pre = Xn @ W1 + c1
            h = np.maximum(0, pre)
            z = h @ W2 + c2
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            g = (p - Y) / len(Xn)
            gW2 = h.T @ g + l2 * W2
            gh = (g @ W2.T) * (pre > 0)
            grads = [Xn.T @ gh + l2 * W1, gh.sum(0), gW2, g.sum(0)]
            for i, gr in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * gr
                v[i] = b2 * v[i] + (1 - b2) * gr * gr
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                params[i] = params[i] - lr * mh / (np.sqrt(vh) + 1e-8)
        acc = float((p.argmax(1) == y).mean())
        return cls(labels, params + [mu, sd]), acc


# ============================================================
# Pose classification
# ============================================================
class PoseClassifier:
    def __init__(self, pinch_thresholds, model=None):
        self.model = model
        self.thresholds = pinch_thresholds

    def detector(self):
        return PinchDetector(self.thresholds)

    def classify(self, hand, pinch):
        """pinch: this hand's PinchDetector (keeps hysteresis state)."""
        p = pinch.update(hand)
        if p:
            return p
        if self.model is None:
            return rule_pose(hand)
        label, conf = self.model.predict(hand_features(hand.lm, hand.label))
        return label if conf >= CLASSIFIER_MIN_CONFIDENCE else POSE_NEUTRAL


def is_palm_like(pose):
    """Movement is allowed during open palm, pinch, or grab."""
    return pose in (POSE_OPEN, POSE_INDEX_PINCH, POSE_MIDDLE_PINCH, POSE_GRAB)


class PoseStabilizer:
    def __init__(self, n_required=POSE_CONFIRM_FRAMES):
        self.n = n_required
        self.committed = POSE_NONE
        self.candidate = POSE_NONE
        self.candidate_count = 0

    def update(self, raw_pose):
        # Pinches and grab commit instantly — click/drag latency matters
        if raw_pose in (POSE_INDEX_PINCH, POSE_MIDDLE_PINCH, POSE_GRAB):
            self.committed = raw_pose
            self.candidate = raw_pose
            self.candidate_count = self.n
            return self.committed

        if raw_pose == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = raw_pose
            self.candidate_count = 1

        needed = self.n if raw_pose in BUILTIN_POSES or raw_pose == POSE_NONE else CUSTOM_CONFIRM_FRAMES
        if self.candidate_count >= needed:
            self.committed = self.candidate
        return self.committed


# ============================================================
# Swipes
# ============================================================
# The active region is ~28% of the frame, so ordinary cursor movement never
# travels this far this fast. That gap is what separates a flick from a move.
SWIPE_MIN_TRAVEL = 0.32     # fraction of frame width/height
SWIPE_MAX_MS = 300
SWIPE_DOMINANCE = 2.0       # main axis must beat the other by this factor
SWIPE_COOLDOWN_S = 0.8      # also swallows the hand's return stroke


class SwipeDetector:
    def __init__(self):
        self.history = deque()   # (t, x_frac, y_frac)
        self.cooldown_until = 0.0

    def reset(self):
        self.history.clear()

    def update(self, t, x_frac, y_frac):
        """Returns ('swipe_left'|..., t_start) or None."""
        self.history.append((t, x_frac, y_frac))
        while self.history and t - self.history[0][0] > SWIPE_MAX_MS / 1000:
            self.history.popleft()
        if t < self.cooldown_until or len(self.history) < 3:
            return None
        t0, x0, y0 = self.history[0]
        dx, dy = x_frac - x0, y_frac - y0
        name = None
        if abs(dx) > SWIPE_MIN_TRAVEL and abs(dx) > SWIPE_DOMINANCE * abs(dy):
            name = "swipe_left" if dx < 0 else "swipe_right"
        elif abs(dy) > SWIPE_MIN_TRAVEL and abs(dy) > SWIPE_DOMINANCE * abs(dx):
            name = "swipe_up" if dy < 0 else "swipe_down"
        if name:
            self.cooldown_until = t + SWIPE_COOLDOWN_S
            self.history.clear()
            return name, t0
        return None


# ============================================================
# Two hands: zoom & rotate
# ============================================================
ZOOM_STEP_RATIO = 1.18      # spread/close this much per zoom step
ROTATE_STEP_DEG = 30


class TwoHandGesture:
    """Both hands index-pinching. Fires one step per ZOOM_STEP_RATIO change in
    distance or ROTATE_STEP_DEG change in angle, re-baselining each time."""

    def __init__(self):
        self.active = False
        self.d0 = self.a0 = 0.0

    @staticmethod
    def _measure(a, b):
        pa = ((a.lm[4].x + a.lm[8].x) / 2, (a.lm[4].y + a.lm[8].y) / 2)
        pb = ((b.lm[4].x + b.lm[8].x) / 2, (b.lm[4].y + b.lm[8].y) / 2)
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))

    def start(self, left, right):
        self.active = True
        self.d0, self.a0 = self._measure(left, right)

    def stop(self):
        self.active = False

    def update(self, left, right):
        d, a = self._measure(left, right)
        if self.d0 > 1e-3:
            ratio = d / self.d0
            if ratio > ZOOM_STEP_RATIO:
                self.d0 = d
                return "zoom_in"
            if ratio < 1 / ZOOM_STEP_RATIO:
                self.d0 = d
                return "zoom_out"
        da = (a - self.a0 + 180) % 360 - 180
        if abs(da) > ROTATE_STEP_DEG:
            self.a0 = a
            # image y points down, so a positive angle change is clockwise on screen
            return "rotate_cw" if da > 0 else "rotate_ccw"
        return None
