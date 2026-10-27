"""
Logic tests with synthetic hands; no camera, and no events reach macOS.

  ./jarvis-env/bin/python tests/test_logic.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis import control, macos  # noqa: E402
from jarvis.config import Config, normalize_phrase  # noqa: E402
from jarvis.filters import OneEuroFilter  # noqa: E402
from jarvis.gestures import (  # noqa: E402
    BUILTIN_POSES, DEFAULT_PINCH, GestureClassifier, PoseClassifier, hand_features,
    thresholds_from_samples,
)
from jarvis.tracking import Hand, hand_scale, normalized_distance, palm_point  # noqa: E402

W, H = 1280, 720


# ============================================================
# Synthetic hands
# ============================================================
def make_lm(pose, cx=0.6, cy=0.55, s=0.12):
    """21 landmarks for a right hand, palm facing the (mirrored) camera."""
    P = lambda x, y: SimpleNamespace(x=cx + x * s, y=cy + y * s, z=0.0)  # noqa: E731
    lm = [None] * 21
    lm[0] = P(0, 0.9)
    knuckles = {5: -0.45, 9: -0.15, 13: 0.15, 17: 0.42}
    up = {"open_palm": [1, 1, 1, 1], "peace": [1, 1, 0, 0], "fist": [0, 0, 0, 0],
          "grab": [0, 0, 0, 0], "pinch": [1, 1, 1, 1], "mpinch": [1, 1, 1, 1],
          "point": [1, 0, 0, 0]}[pose]
    for (mcp, x), is_up in zip(knuckles.items(), up):
        lm[mcp] = P(x, 0)
        if is_up:
            for k, dy in enumerate((-0.35, -0.6, -0.8), 1):
                lm[mcp + k] = P(x, dy)
        else:
            for k, dy in enumerate((-0.2, 0.05, 0.15), 1):
                lm[mcp + k] = P(x, dy)
    # thumb: out to screen-left (extended) or tucked across the palm
    thumb_out = pose not in ("grab",)
    for k, (x, y) in enumerate([(-0.4, 0.7), (-0.65, 0.5), (-0.85, 0.35), (-1.0, 0.2)], 1):
        lm[k] = P(x if thumb_out else -x * 0.3, y)
    if pose == "pinch":
        lm[4] = SimpleNamespace(x=lm[8].x, y=lm[8].y + 0.002, z=0)
    if pose == "mpinch":
        lm[4] = SimpleNamespace(x=lm[12].x, y=lm[12].y + 0.002, z=0)
    return lm


def make_hand(pose, cx=0.6, cy=0.55, label="Right"):
    lm = make_lm(pose, cx, cy)
    if label == "Left":
        lm = [SimpleNamespace(x=2 * cx - p.x, y=p.y, z=p.z) for p in lm]
    sc = hand_scale(lm)
    return Hand(lm=lm, label=label, palm=palm_point(lm, W, H),
                pinch_index=normalized_distance(lm[4], lm[8]) / sc,
                pinch_middle=normalized_distance(lm[4], lm[12]) / sc)


class FakeOverlay:
    def __init__(self):
        self.sent = []

    def send(self, state, text=None, flash=None):
        self.sent.append((state, flash))

    def poll(self):
        return []

    @property
    def flashes(self):
        return [f for _, f in self.sent if f]


class FakeVoice:
    def __init__(self):
        self.recording = False
        self.busy = False
        self.ready = True
        self.log = []

    def start(self):
        self.recording = True
        self.log.append("start")

    def stop(self):
        self.recording = False
        self.log.append("stop")
        return True

    def cancel(self):
        self.recording = False


events = []


def stub_macos():
    for name in ("move", "click", "mouse_down", "mouse_up", "scroll", "press", "type_text"):
        setattr(macos, name, lambda *a, _n=name, **k: events.append((_n, a, k)))
    macos.cursor_position = lambda: (500.0, 400.0)
    macos.frontmost_app = lambda: "Finder"


def ev(name):
    return [e for e in events if e[0] == name]


def make_controller(voice=False):
    events.clear()
    cfg = Config()
    tracker = SimpleNamespace(w=W, h=H)
    poses = PoseClassifier(dict(DEFAULT_PINCH))
    c = control.Controller(tracker, FakeOverlay(), cfg, poses, voice=FakeVoice() if voice else None)
    return c


class Clock:
    def __init__(self, c, t=100.0):
        self.c, self.t = c, t

    def run(self, hands_fn, seconds, fps=30):
        for _ in range(int(seconds * fps)):
            self.t += 1 / fps
            hs = hands_fn(self.t) if callable(hands_fn) else hands_fn
            self.c.step(None, hs, self.t)


def wake(c, clk):
    clk.run([make_hand("open_palm")], 0.8)
    assert c.awake, "should wake after holding an open palm still"


# ============================================================
# Tests
# ============================================================
def test_one_euro_smooths_still_and_tracks_fast():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=control.FILTER_MIN_CUTOFF, beta=control.FILTER_BETA)
    out = [f(500 + rng.normal(0, 3), i / 30) for i in range(300)]
    assert np.std(out[30:]) < 1.35, "jitter at rest should drop well over 2x"
    f = OneEuroFilter(min_cutoff=control.FILTER_MIN_CUTOFF, beta=control.FILTER_BETA)
    xs = [f(i * 40.0, i / 30) for i in range(30)]   # 1200 px/s
    assert 29 * 40 - xs[-1] < 60, f"fast motion lags too much: {29 * 40 - xs[-1]:.0f}px"


def test_rules_classify_synthetic_poses():
    pc = PoseClassifier(dict(DEFAULT_PINCH))
    for pose, want in [("open_palm", "open_palm"), ("peace", "peace"), ("fist", "fist"),
                       ("grab", "grab"), ("pinch", "index_pinch"), ("mpinch", "middle_pinch")]:
        got = pc.classify(make_hand(pose), pc.detector())
        assert got == want, f"{pose}: got {got}"
    left = make_hand("fist", label="Left")
    assert pc.classify(left, pc.detector()) == "fist", "left-hand thumb test should mirror"


def test_palm_point_stable_during_pinch():
    a, b = make_hand("open_palm"), make_hand("pinch")
    assert abs(a.palm[0] - b.palm[0]) < 1e-6 and abs(a.palm[1] - b.palm[1]) < 1e-6


def test_calibration_thresholds():
    th = thresholds_from_samples(np.full(50, 0.04), np.full(50, 0.9))
    assert 0.04 < th[0] < th[1] < 0.9
    assert thresholds_from_samples(np.full(50, 0.3), np.full(50, 0.32)) is None


def test_classifier_learns_poses():
    rng = np.random.default_rng(1)
    X, y = [], []
    names = BUILTIN_POSES + ["point"]
    src = {"open_palm": "open_palm", "fist": "fist", "grab": "grab", "peace": "peace",
           "neutral": "mpinch", "point": "point"}
    for i, n in enumerate(names):
        for _ in range(60):
            h = make_hand(src[n], cx=rng.uniform(0.4, 0.7), cy=rng.uniform(0.4, 0.6))
            X.append(hand_features(h.lm) + rng.normal(0, 0.03, 63))
            y.append(i)
    model, acc = GestureClassifier.train(np.array(X, np.float32), np.array(y), names, epochs=200)
    assert acc > 0.95, acc
    label, conf = model.predict(hand_features(make_hand("point").lm))
    assert label == "point", (label, conf)


def test_config_resolution_and_voice_commands():
    cfg = Config()
    assert cfg.action_for("swipe_right", app="Safari") == "cmd+["
    assert cfg.action_for("swipe_right", app="Finder") == "ctrl+left"
    assert cfg.action_for("rotate_cw", app="Finder") == ""
    assert cfg.action_for("rotate_cw", app="Preview") == "cmd+r"
    assert cfg.voice_command(" Open Safari.") == "open:safari"
    assert cfg.voice_command("Next tab!") == "ctrl+tab"
    assert cfg.voice_command("hello there, how are you") is None
    assert normalize_phrase("  Close  THIS. ") == "close this"
    for combo in ["cmd+shift+t", "ctrl+left", "cmd+=", "cmd+-", "alt+cmd+r", "return", "cmd+space"]:
        macos.parse_combo(combo)


def test_sleeps_until_wake_gesture():
    c = make_controller()
    clk = Clock(c)
    # a moving hand doesn't wake it, and doesn't move the cursor
    clk.run(lambda t: [make_hand("open_palm", cx=0.4 + 0.3 * ((t * 2) % 1))], 1.5)
    assert not c.awake and not ev("move")
    # pinching while asleep doesn't click
    clk.run([make_hand("pinch")], 0.3)
    clk.run([make_hand("open_palm", cx=0.3)], 0.1)
    assert not ev("click")
    wake(c, clk)
    # falls back asleep with no input
    clk.run([], 6)
    assert not c.awake


def test_click_drag_double_click():
    c = make_controller()
    clk = Clock(c)
    wake(c, clk)
    events.clear()
    # move the cursor
    clk.run(lambda t: [make_hand("open_palm", cx=0.6 + 0.05 * min(1, (t - 100.8)))], 1.0)
    assert ev("move")
    # quick pinch -> one click, no mouse_down
    clk.run([make_hand("pinch", cx=0.65)], 0.1)
    clk.run([make_hand("open_palm", cx=0.65)], 0.15)
    assert len(ev("click")) == 1 and not ev("mouse_down")
    # second quick pinch -> double click (clickState 2)
    clk.run([make_hand("pinch", cx=0.65)], 0.1)
    clk.run([make_hand("open_palm", cx=0.65)], 0.15)
    assert ev("click")[-1][2].get("count") == 2
    # long pinch -> drag
    events.clear()
    clk.run([make_hand("pinch", cx=0.65)], 0.5)
    clk.run(lambda t: [make_hand("pinch", cx=0.65 + 0.02)], 0.3)
    clk.run([make_hand("open_palm", cx=0.67)], 0.1)
    assert ev("mouse_down") and ev("mouse_up") and not ev("click")


def test_swipe_fires_and_restores_cursor():
    c = make_controller()
    clk = Clock(c)
    wake(c, clk)
    before = (c.mapper.sx, c.mapper.sy)
    events.clear()
    start = clk.t
    clk.run(lambda t: [make_hand("open_palm", cx=0.75 - 2.5 * (t - start))], 0.2)  # fast flick left
    presses = [e[1][0] for e in ev("press")]
    assert presses == ["ctrl+right"], presses
    assert abs(c.mapper.sx - before[0]) < 40, "cursor should return to where the flick began"
    # normal movement doesn't swipe
    events.clear()
    clk.run([make_hand("open_palm", cx=0.45)], 1.0)
    start = clk.t
    clk.run(lambda t: [make_hand("open_palm", cx=0.45 + 0.4 * (t - start))], 0.5)
    assert not ev("press")


def test_two_hand_zoom_without_click():
    c = make_controller()
    clk = Clock(c)
    wake(c, clk)
    events.clear()
    start = clk.t
    hands = lambda t: [make_hand("pinch", cx=0.65 + 0.15 * (t - start)),  # noqa: E731
                       make_hand("pinch", cx=0.35 - 0.15 * (t - start), label="Left")]
    clk.run(hands, 1.0)
    clk.run([make_hand("open_palm", cx=0.8), make_hand("open_palm", cx=0.2, label="Left")], 0.3)
    presses = [e[1][0] for e in ev("press")]
    assert presses and set(presses) == {"cmd+="}, presses
    assert not ev("click") and not ev("mouse_down"), "zoom must not click"


def test_push_to_talk():
    c = make_controller(voice=True)
    clk = Clock(c)
    wake(c, clk)
    both = [make_hand("open_palm", cx=0.65), make_hand("pinch", cx=0.3, label="Left")]
    clk.run(both, 0.6)
    assert c.voice.recording
    clk.run([make_hand("open_palm", cx=0.65), make_hand("open_palm", cx=0.3, label="Left")], 0.2)
    assert c.voice.log == ["start", "stop"]
    events.clear()
    c.voice_results.put(" Next tab.")
    c.voice_results.put(" hello world")
    clk.run([make_hand("open_palm", cx=0.65)], 0.1)
    assert [e[1][0] for e in ev("press")] == ["ctrl+tab"]
    assert [e[1][0] for e in ev("type_text")] == ["hello world"]


def test_second_hand_doesnt_steal_cursor():
    c = make_controller()
    clk = Clock(c)
    wake(c, clk)
    clk.run([make_hand("open_palm", cx=0.6)], 0.3)
    # a left hand appears listed first; the primary should stay the right one
    clk.run([make_hand("open_palm", cx=0.3, label="Left"), make_hand("open_palm", cx=0.6)], 0.3)
    assert c.primary_pos[0] > 0.5 * W


if __name__ == "__main__":
    stub_macos()
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
