"""
Personalisation:
  - calibrate(): ~10 s on first launch (or --calibrate). Measures how far
    apart your thumb and fingertips are when relaxed vs pinched and sets the
    pinch thresholds between them.
  - record(): hold each gesture for a few seconds while landmarks are
    captured, then train the pose classifier on everything recorded so far.
"""

import json
import os
import time

import cv2
import numpy as np

from .gestures import (
    BUILTIN_POSES, CALIBRATION_PATH, DEFAULT_PINCH, MODEL_PATH, PROFILE_DIR,
    SAMPLES_PATH, GestureClassifier, hand_features, thresholds_from_samples,
)
from .tracking import draw_hand

PROMPTS = {
    "open_palm": "open palm, fingers spread (moves the cursor)",
    "fist": "fist with the thumb sticking out (Escape)",
    "grab": "full fist, thumb tucked in (moves windows)",
    "peace": "peace sign, index + middle up (scroll)",
    "neutral": "anything else: relaxed, half-curled, pointing, mid-gesture",
}
READY_S = 2.5
RECORD_S = 4.0


def _phase(tracker, overlay, text, seconds, collect, show_camera, ready_s=1.0):
    """Count down, then call collect(hand) on each frame for `seconds`."""
    t_end_ready = time.time() + ready_s
    t_end = t_end_ready + seconds
    shown = None
    while True:
        frame, hands, t = tracker.read()
        if frame is None:
            continue
        if t >= t_end:
            return
        hand = max(hands, key=lambda h: h.palm[0]) if hands else None
        if t < t_end_ready:
            msg = f"{text} — get ready {int(t_end_ready - t) + 1}"
        else:
            msg = f"{text} — hold… {t_end - t:.0f}s" if hand else f"{text} — no hand visible"
            if hand:
                collect(hand)
        if msg != shown:
            overlay.send("setup", text=msg)
            shown = msg
        if show_camera:
            if hand:
                draw_hand(frame, hand.lm, tracker.w, tracker.h)
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("JARVIS setup", frame)
            cv2.waitKey(1)


def calibrate(tracker, overlay, show_camera=False):
    relaxed_i, relaxed_m, pinch_i, pinch_m = [], [], [], []

    def relaxed(h):
        relaxed_i.append(h.pinch_index)
        relaxed_m.append(h.pinch_middle)

    print("Calibrating (about 10 s). Use your cursor hand.")
    _phase(tracker, overlay, "Calibrating: relaxed open palm", 3, relaxed, show_camera, ready_s=1.5)
    _phase(tracker, overlay, "Pinch thumb + index and hold", 3, lambda h: pinch_i.append(h.pinch_index), show_camera)
    _phase(tracker, overlay, "Pinch thumb + middle and hold", 3, lambda h: pinch_m.append(h.pinch_middle), show_camera)
    if show_camera:
        cv2.destroyWindow("JARVIS setup")
        cv2.waitKey(1)

    pinch = dict(DEFAULT_PINCH)
    notes = []
    # Skip the first frames of each hold: the fingers are still closing
    for name, closed, open_ in (("index", pinch_i[8:], relaxed_i), ("middle", pinch_m[8:], relaxed_m)):
        th = thresholds_from_samples(closed, open_)
        if th:
            pinch[f"{name}_on"], pinch[f"{name}_off"] = round(th[0], 4), round(th[1], 4)
            notes.append(f"{name} {th[0]:.3f}/{th[1]:.3f}")
        else:
            notes.append(f"{name}: unclear, kept defaults")

    if all("unclear" in n for n in notes):
        print("Calibration couldn't see clear pinches; using defaults this time and "
              "trying again next launch.")
        overlay.send("setup", text="Calibration unclear — using defaults")
        return pinch, None

    data = {"pinch": pinch, "created": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved ({', '.join(notes)}) -> {CALIBRATION_PATH}")
    overlay.send("setup", text="Calibrated")
    return pinch, data


# ============================================================
# Recording + training
# ============================================================
def load_samples():
    if not os.path.exists(SAMPLES_PATH):
        return np.zeros((0, 63), np.float32), np.array([], dtype=str)
    d = np.load(SAMPLES_PATH, allow_pickle=False)
    return d["X"], d["y"]


def save_samples(X, y):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    np.savez(SAMPLES_PATH, X=X, y=y)


def record(tracker, overlay, names, show_camera=True):
    X, y = load_samples()
    have = set(y.tolist())
    missing = [p for p in BUILTIN_POSES if p not in have and p not in names]
    if missing:
        print(f"No samples yet for {', '.join(missing)}; recording those too so the "
              "built-in gestures keep working.")
    queue = missing + list(names)
    for name in queue:
        got = []
        prompt = PROMPTS.get(name, name.replace("_", " "))
        print(f"\n== {name}: {prompt}. Move it around a little while holding.")
        _phase(tracker, overlay, f"Record {name}: {prompt}", RECORD_S,
               lambda h: got.append(hand_features(h.lm, h.label)), show_camera, ready_s=READY_S)
        if len(got) < 20:
            print(f"   only {len(got)} frames with a hand — skipped {name}")
            continue
        X = np.concatenate([X[y != name], np.array(got, np.float32)])
        y = np.concatenate([y[y != name], np.array([name] * len(got))])
        print(f"   {len(got)} samples")
    if show_camera:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    save_samples(X, y)
    overlay.send("setup", text="Training…")
    return train()


def train():
    X, y = load_samples()
    labels = sorted(set(y.tolist()))
    missing = [p for p in BUILTIN_POSES if p not in labels]
    if missing:
        print(f"Can't train: no samples for {', '.join(missing)}. Run with --record.")
        return None
    idx = np.array([labels.index(v) for v in y])
    model, acc = GestureClassifier.train(X, idx, labels)
    model.save()
    counts = ", ".join(f"{l} {int((y == l).sum())}" for l in labels)
    print(f"Trained on {len(y)} samples ({counts}); training accuracy {acc:.1%} -> {MODEL_PATH}")
    custom = [l for l in labels if l not in BUILTIN_POSES]
    if custom:
        print(f"Bind custom gestures in jarvis.toml [actions], e.g.  {custom[0]} = \"cmd+tab\"")
    return model


def forget(names):
    X, y = load_samples()
    keep = ~np.isin(y, names)
    save_samples(X[keep], y[keep])
    print(f"Removed {int((~keep).sum())} samples for {', '.join(names)}")
    return train() if keep.any() else None
