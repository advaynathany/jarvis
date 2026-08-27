"""
JARVIS-style Hand Tracking Screen Control (v4)
================================================
Gesture map:
  - Open palm (>=3 of 4 fingers up)  -> cursor moves
  - Thumb+index pinch                -> left click (held = drag)
  - Thumb+middle pinch               -> right click + 1.5s menu mode
  - Two quick pinches at same spot   -> double-click
  - Peace sign (index+middle only)   -> scroll mode (vertical hand motion)
  - Full fist (thumb out/extended)   -> Escape key + abort drag
  - Full grab (all 5 fingers closed) -> grab & move window under cursor

Architecture: pose classifier -> intent state machine.
Movement is gated on "palm-ish" (palm OR pinch in progress) so that
clicking/dragging doesn't lose cursor tracking.

Keys: q=quit, m=toggle mouse, r=recenter active region.

Requires: pip install opencv-python mediapipe pyautogui numpy
"""

import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================
# pyautogui setup
# ============================================================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ============================================================
# Model download
# ============================================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print(f"Downloading hand landmark model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ============================================================
# Camera — change index if your laptop camera is at 1 instead of 0
# ============================================================
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ============================================================
# MediaPipe Tasks API
# ============================================================
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ============================================================
# TUNABLES
# ============================================================
ACTIVE_REGION_W_FRAC = 0.28   # smaller region = more screen travel per cm of hand movement
ACTIVE_REGION_H_FRAC = 0.28

POSE_CONFIRM_FRAMES = 2

PINCH_ON  = 0.065   # require deliberate pinch; tighter = fewer accidental clicks
PINCH_OFF = 0.11

CLICK_MAX_MS    = 250   # shorter hold to enter drag
DRAG_MIN_PX     = 40    # less movement needed to start drag
CLICK_SETTLE_MS = 120   # cursor freeze at pinch start to absorb tremor
DOUBLE_CLICK_MS = 400
DOUBLE_CLICK_RADIUS_PX = 30

MENU_MODE_MS = 1500
MENU_SMOOTHING_BOOST = 1.8
MENU_DEADZONE_PX = 3

BASE_SMOOTHING = 1.5    # less lag
ACCEL_SLOW_GAIN = 1.3   # more gain even on slow movements
ACCEL_FAST_GAIN = 5.0   # stronger fast-movement amplification
ACCEL_SLOW_THRESHOLD = 4
ACCEL_FAST_THRESHOLD = 25

CLICK_FREEZE_MS = 60
CLICK_COOLDOWN  = 0.20

SCROLL_COOLDOWN = 0.05
SCROLL_SENSITIVITY = 200

HAND_LOSS_GRACE_MS = 300

ESCAPE_COOLDOWN = 0.6

SHOW_DEBUG = True


# ============================================================
# Geometry helpers
# ============================================================
def to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def normalized_distance(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def hand_scale(lm):
    return max(normalized_distance(lm[0], lm[9]), 1e-6)


def fingers_extended(lm):
    """[thumb, index, middle, ring, pinky] booleans."""
    thumb = lm[4].x < lm[3].x
    fingers = [thumb]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers


def mean_hand_position(lm, w, h):
    xs = sum(p.x for p in lm) / len(lm)
    ys = sum(p.y for p in lm) / len(lm)
    return xs * w, ys * h


def draw_hand(frame, lm, w, h):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, to_pixel(lm[a], w, h), to_pixel(lm[b], w, h),
                 (200, 200, 200), 2)
    for p in lm:
        cv2.circle(frame, to_pixel(p, w, h), 4, (0, 200, 255), cv2.FILLED)


# ============================================================
# Layer 1: Pose classifier
# ============================================================
POSE_NONE         = "none"
POSE_OPEN         = "open_palm"     # cursor movement
POSE_FIST         = "fist"          # escape / abort (thumb extended out)
POSE_GRAB         = "grab"          # all 5 fingers closed — grab & move window
POSE_PEACE        = "peace"         # scroll
POSE_INDEX_PINCH  = "index_pinch"   # left click/drag
POSE_MIDDLE_PINCH = "middle_pinch"  # right click
POSE_NEUTRAL      = "neutral"       # something else


def classify_pose(lm, prev_pinch_state):
    """Returns (pose, new_pinch_state, d_thumb_index, d_thumb_middle)."""
    scale = hand_scale(lm)
    d_thumb_index  = normalized_distance(lm[4], lm[8])  / scale
    d_thumb_middle = normalized_distance(lm[4], lm[12]) / scale

    on_index  = (d_thumb_index  < PINCH_ON) or (
        prev_pinch_state.get('index', False) and d_thumb_index < PINCH_OFF)
    on_middle = (d_thumb_middle < PINCH_ON) or (
        prev_pinch_state.get('middle', False) and d_thumb_middle < PINCH_OFF)
    new_pinch_state = {'index': on_index, 'middle': on_middle}

    # Pinches take priority for action triggering
    if on_index and on_middle:
        if d_thumb_index <= d_thumb_middle:
            return POSE_INDEX_PINCH, new_pinch_state, d_thumb_index, d_thumb_middle
        return POSE_MIDDLE_PINCH, new_pinch_state, d_thumb_index, d_thumb_middle
    if on_index:
        return POSE_INDEX_PINCH, new_pinch_state, d_thumb_index, d_thumb_middle
    if on_middle:
        return POSE_MIDDLE_PINCH, new_pinch_state, d_thumb_index, d_thumb_middle

    f = fingers_extended(lm)
    # f = [thumb, index, middle, ring, pinky]

    # Peace: index + middle up, ring + pinky down (regardless of thumb)
    if f[1] and f[2] and not f[3] and not f[4]:
        return POSE_PEACE, new_pinch_state, d_thumb_index, d_thumb_middle

    # Fist/Grab: zero non-thumb fingers up
    # Differentiate by thumb: thumb extended = fist (escape), thumb tucked = grab (window drag)
    if sum(f[1:]) == 0:
        if not f[0]:  # thumb also closed → full grab
            return POSE_GRAB, new_pinch_state, d_thumb_index, d_thumb_middle
        return POSE_FIST, new_pinch_state, d_thumb_index, d_thumb_middle

    # Open palm: 3+ of the 4 non-thumb fingers up
    if sum(f[1:]) >= 3:
        return POSE_OPEN, new_pinch_state, d_thumb_index, d_thumb_middle

    return POSE_NEUTRAL, new_pinch_state, d_thumb_index, d_thumb_middle


def is_palm_like(pose):
    """Movement is allowed during open palm, pinch, or grab."""
    return pose in (POSE_OPEN, POSE_INDEX_PINCH, POSE_MIDDLE_PINCH, POSE_GRAB)


# ============================================================
# Layer 2: Intent state machine
# ============================================================
@dataclass
class IntentState:
    mode: str = "idle"  # idle | clicking | dragging | scrolling

    pinch_start_time: float = 0.0
    pinch_start_pos: tuple = (0, 0)
    pinch_kind: str = ""
    mouse_down: bool = False
    mouse_down_kind: str = ""

    last_click_time: float = 0.0
    last_click_pos: tuple = (0, 0)
    last_click_kind: str = ""

    menu_until: float = 0.0
    freeze_until: float = 0.0

    scroll_history: deque = field(default_factory=lambda: deque(maxlen=5))
    last_scroll_time: float = 0.0

    last_hand_seen: float = 0.0
    last_escape_time: float = 0.0


# ============================================================
# Cursor mapper with adaptive active region + acceleration
# ============================================================
@dataclass
class CursorMapper:
    cx: float = CAM_W / 2
    cy: float = CAM_H / 2
    half_w: float = CAM_W * ACTIVE_REGION_W_FRAC / 2
    half_h: float = CAM_H * ACTIVE_REGION_H_FRAC / 2

    sx: float = SCREEN_W / 2
    sy: float = SCREEN_H / 2

    prev_cam_x: float = -1.0
    prev_cam_y: float = -1.0

    def recenter(self, cam_x, cam_y):
        self.cx, self.cy = cam_x, cam_y
        self.prev_cam_x, self.prev_cam_y = cam_x, cam_y

    def reset_velocity(self, cam_x, cam_y):
        """Call when re-engaging movement after a pause, to avoid a jump."""
        self.prev_cam_x, self.prev_cam_y = cam_x, cam_y

    def update(self, cam_x, cam_y, smoothing_mult=1.0, deadzone_px=0):
        if self.prev_cam_x < 0:
            self.prev_cam_x, self.prev_cam_y = cam_x, cam_y
            return self.sx, self.sy

        dx_cam = cam_x - self.prev_cam_x
        dy_cam = cam_y - self.prev_cam_y
        speed = np.hypot(dx_cam, dy_cam)

        if speed < deadzone_px:
            self.prev_cam_x, self.prev_cam_y = cam_x, cam_y
            return self.sx, self.sy

        if speed <= ACCEL_SLOW_THRESHOLD:
            gain = ACCEL_SLOW_GAIN
        elif speed >= ACCEL_FAST_THRESHOLD:
            gain = ACCEL_FAST_GAIN
        else:
            t = (speed - ACCEL_SLOW_THRESHOLD) / (ACCEL_FAST_THRESHOLD - ACCEL_SLOW_THRESHOLD)
            gain = ACCEL_SLOW_GAIN + t * (ACCEL_FAST_GAIN - ACCEL_SLOW_GAIN)

        screen_per_cam_x = SCREEN_W / (2 * self.half_w)
        screen_per_cam_y = SCREEN_H / (2 * self.half_h)
        target_dx = dx_cam * screen_per_cam_x * gain
        target_dy = dy_cam * screen_per_cam_y * gain

        smooth = max(BASE_SMOOTHING * smoothing_mult, 1.0)
        self.sx += target_dx / smooth
        self.sy += target_dy / smooth

        self.sx = float(np.clip(self.sx, 0, SCREEN_W - 1))
        self.sy = float(np.clip(self.sy, 0, SCREEN_H - 1))

        self.prev_cam_x, self.prev_cam_y = cam_x, cam_y
        return self.sx, self.sy


# ============================================================
# Pose stability filter
# ============================================================
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
            self.candidate_count = min(self.candidate_count + 1, self.n)
        else:
            self.candidate = raw_pose
            self.candidate_count = 1

        if self.candidate_count >= self.n:
            self.committed = self.candidate
        return self.committed


# ============================================================
# Main loop
# ============================================================
print(f"Screen: {SCREEN_W}x{SCREEN_H} | Camera: {CAM_W}x{CAM_H}")
print("Open palm = move | pinch = click/drag | thumb+middle pinch = right-click")
print("Full grab (all 5 fingers closed) = grab & move window | Peace = scroll | Fist (thumb out) = Escape")
print("Keys: q=quit, m=toggle mouse, r=recenter region")

mapper = CursorMapper()
stabilizer = PoseStabilizer()
state = IntentState()
prev_pinch = {'index': False, 'middle': False}

mouse_enabled = True
fps_history = deque(maxlen=30)
prev_frame_time = time.time()
start_time = time.time()

# Track previous-frame palm state to know when to reset velocity
was_palm_like = False


def safe_release_mouse():
    if state.mouse_down:
        try:
            pyautogui.mouseUp(button=state.mouse_down_kind or 'left')
        except Exception:
            pass
        state.mouse_down = False
        state.mouse_down_kind = ""


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    t_now = time.time()

    # Active region overlay
    if SHOW_DEBUG:
        x1 = int(mapper.cx - mapper.half_w)
        y1 = int(mapper.cy - mapper.half_h)
        x2 = int(mapper.cx + mapper.half_w)
        y2 = int(mapper.cy + mapper.half_h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.drawMarker(frame, (int(mapper.cx), int(mapper.cy)),
                       (255, 100, 0), cv2.MARKER_CROSS, 12, 2)

    pose_raw = POSE_NONE
    pose = POSE_NONE
    cam_x = cam_y = None

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        if SHOW_DEBUG:
            draw_hand(frame, lm, CAM_W, CAM_H)
        state.last_hand_seen = t_now

        cam_x, cam_y = mean_hand_position(lm, CAM_W, CAM_H)
        pose_raw, prev_pinch, _, _ = classify_pose(lm, prev_pinch)
        pose = stabilizer.update(pose_raw)

        if mapper.prev_cam_x < 0:
            mapper.prev_cam_x, mapper.prev_cam_y = cam_x, cam_y

        # Slow auto-recenter only when idle and palm
        if state.mode == "idle" and pose == POSE_OPEN:
            if abs(cam_x - mapper.cx) > mapper.half_w * 0.85:
                mapper.cx += (cam_x - mapper.cx) * 0.005
            if abs(cam_y - mapper.cy) > mapper.half_h * 0.85:
                mapper.cy += (cam_y - mapper.cy) * 0.005
    else:
        # Hand lost
        if state.mode == "dragging":
            if (t_now - state.last_hand_seen) * 1000 > HAND_LOSS_GRACE_MS:
                safe_release_mouse()
                state.mode = "idle"
        else:
            safe_release_mouse()
            state.scroll_history.clear()

    # ============================================================
    # Determine smoothing/deadzone
    # ============================================================
    in_menu = t_now < state.menu_until
    smoothing_mult = MENU_SMOOTHING_BOOST if in_menu else 1.0
    deadzone = MENU_DEADZONE_PX if in_menu else 0

    # ============================================================
    # Cursor movement: only when palm-like (palm OR pinch)
    # ============================================================
    palm_like = is_palm_like(pose)

    # Reset velocity when transitioning back into palm so we don't
    # snap-jump from old position
    if cam_x is not None and palm_like and not was_palm_like:
        mapper.reset_velocity(cam_x, cam_y)

    click_settling = (
        state.mode == "clicking" and
        (t_now - state.pinch_start_time) * 1000 < CLICK_SETTLE_MS
    )
    cursor_frozen = (t_now < state.freeze_until) or (state.mode == "scrolling") or click_settling

    if cam_x is not None and palm_like and not cursor_frozen:
        new_sx, new_sy = mapper.update(cam_x, cam_y,
                                       smoothing_mult=smoothing_mult,
                                       deadzone_px=deadzone)
        if mouse_enabled:
            pyautogui.moveTo(new_sx, new_sy)
    elif cam_x is not None:
        # Keep prev_cam in sync so we don't jump when movement re-engages
        mapper.prev_cam_x, mapper.prev_cam_y = cam_x, cam_y

    was_palm_like = palm_like

    # ============================================================
    # Pose-driven intents
    # ============================================================

    # GRAB = hold mouseDown and drag window (all 5 fingers closed)
    if pose == POSE_GRAB:
        if state.mode != "grabbing":
            safe_release_mouse()
            state.mode = "grabbing"
            if mouse_enabled:
                pyautogui.mouseDown(button='left')
                state.mouse_down = True
                state.mouse_down_kind = "left"
        # cursor movement happens via is_palm_like above

    # FIST = Escape (and abort drag / grab if active)
    elif pose == POSE_FIST:
        if t_now - state.last_escape_time > ESCAPE_COOLDOWN:
            if mouse_enabled:
                pyautogui.press('escape')
            state.last_escape_time = t_now
            if state.mode in ("dragging", "grabbing"):
                safe_release_mouse()
            state.mode = "idle"
            state.menu_until = 0
            state.scroll_history.clear()

    # PEACE = scroll
    elif pose == POSE_PEACE:
        # Cancel any pending click/drag/grab — peace is a different intent
        if state.mode in ("clicking", "dragging", "grabbing"):
            safe_release_mouse()
            state.mode = "idle"
        state.mode = "scrolling"
        if cam_x is not None:
            lm = result.hand_landmarks[0]
            ref_y = lm[9].y  # middle MCP — stable vertical reference
            state.scroll_history.append(ref_y)
            if len(state.scroll_history) >= 3 and mouse_enabled:
                delta = state.scroll_history[0] - state.scroll_history[-1]
                if abs(delta) > 0.02 and t_now - state.last_scroll_time > SCROLL_COOLDOWN:
                    pyautogui.scroll(int(delta * SCROLL_SENSITIVITY))
                    state.last_scroll_time = t_now

    else:
        # Leaving scroll or grab mode
        if state.mode == "scrolling":
            state.mode = "idle"
            state.scroll_history.clear()
        elif state.mode == "grabbing":
            safe_release_mouse()
            state.mode = "idle"

        # PINCH-DRIVEN INTENTS
        is_left_pinch = (pose == POSE_INDEX_PINCH)
        is_right_pinch = (pose == POSE_MIDDLE_PINCH)
        any_pinch = is_left_pinch or is_right_pinch

        # Pinch ENTRY
        if any_pinch and state.mode == "idle":
            state.mode = "clicking"
            state.pinch_start_time = t_now
            state.pinch_start_pos = (mapper.sx, mapper.sy)
            state.pinch_kind = "left" if is_left_pinch else "right"
            if state.pinch_kind == "left" and mouse_enabled:
                pyautogui.mouseDown(button='left')
                state.mouse_down = True
                state.mouse_down_kind = "left"

        # Pinch HELD
        elif any_pinch and state.mode in ("clicking", "dragging"):
            held_ms = (t_now - state.pinch_start_time) * 1000
            cur_pos = (mapper.sx, mapper.sy)
            moved_px = np.hypot(cur_pos[0] - state.pinch_start_pos[0],
                                cur_pos[1] - state.pinch_start_pos[1])
            if state.mode == "clicking" and state.pinch_kind == "left":
                if held_ms > CLICK_MAX_MS or moved_px > DRAG_MIN_PX:
                    state.mode = "dragging"

        # Pinch RELEASE
        elif not any_pinch and state.mode in ("clicking", "dragging"):
            cur_pos = (mapper.sx, mapper.sy)
            if state.mode == "dragging":
                safe_release_mouse()
            else:
                if state.pinch_kind == "left":
                    safe_release_mouse()
                    since_last = (t_now - state.last_click_time) * 1000
                    dist_last = np.hypot(cur_pos[0] - state.last_click_pos[0],
                                         cur_pos[1] - state.last_click_pos[1])
                    is_double = (
                        since_last < DOUBLE_CLICK_MS and
                        dist_last < DOUBLE_CLICK_RADIUS_PX and
                        state.last_click_kind == "left"
                    )
                    if is_double and mouse_enabled:
                        pyautogui.click(button='left')
                    state.last_click_time = t_now
                    state.last_click_pos = cur_pos
                    state.last_click_kind = "left"
                    state.freeze_until = t_now + CLICK_FREEZE_MS / 1000.0
                elif state.pinch_kind == "right":
                    if mouse_enabled:
                        pyautogui.click(button='right')
                    state.last_click_time = t_now
                    state.last_click_pos = cur_pos
                    state.last_click_kind = "right"
                    state.menu_until = t_now + MENU_MODE_MS / 1000.0
                    state.freeze_until = t_now + CLICK_FREEZE_MS / 1000.0
            state.mode = "idle"
            state.pinch_kind = ""

    # ============================================================
    # Visualization / HUD
    # ============================================================
    if SHOW_DEBUG and result.hand_landmarks and cam_x is not None:
        cv2.circle(frame, (int(cam_x), int(cam_y)), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (int(cam_x), int(cam_y)), 14, (0, 255, 0), 2)

        lm = result.hand_landmarks[0]
        if pose == POSE_INDEX_PINCH:
            mid = ((to_pixel(lm[4], CAM_W, CAM_H)[0] +
                    to_pixel(lm[8], CAM_W, CAM_H)[0]) // 2,
                   (to_pixel(lm[4], CAM_W, CAM_H)[1] +
                    to_pixel(lm[8], CAM_W, CAM_H)[1]) // 2)
            cv2.circle(frame, mid, 22, (0, 255, 255), 3)
        if pose == POSE_MIDDLE_PINCH:
            mid = ((to_pixel(lm[4], CAM_W, CAM_H)[0] +
                    to_pixel(lm[12], CAM_W, CAM_H)[0]) // 2,
                   (to_pixel(lm[4], CAM_W, CAM_H)[1] +
                    to_pixel(lm[12], CAM_W, CAM_H)[1]) // 2)
            cv2.circle(frame, mid, 22, (0, 100, 255), 3)

    new_frame_time = time.time()
    fps = 1 / max(new_frame_time - prev_frame_time, 1e-6)
    prev_frame_time = new_frame_time
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)

    cv2.rectangle(frame, (0, 0), (CAM_W, 70), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "JARVIS HAND CONTROL v4", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {avg_fps:.0f}  Pose: {pose}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    mode_colors = {
        "idle": (200, 200, 200),
        "clicking": (0, 255, 255),
        "dragging": (0, 255, 0),
        "scrolling": (255, 200, 0),
        "grabbing": (255, 80, 200),
    }
    display_mode = "menu" if in_menu else state.mode
    cv2.putText(frame, f"MODE: {display_mode.upper()}", (380, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                mode_colors.get(display_mode, (255, 100, 255)), 2)

    mouse_status = "MOUSE: ON" if mouse_enabled else "MOUSE: OFF"
    mouse_color = (0, 255, 0) if mouse_enabled else (0, 0, 255)
    cv2.putText(frame, mouse_status, (CAM_W - 180, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouse_color, 2)

    cv2.putText(frame, "[q]uit  [m]ouse  [r]ecenter",
                (10, CAM_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)

    cv2.imshow("JARVIS Hand Control", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mouse_enabled = not mouse_enabled
        if not mouse_enabled:
            safe_release_mouse()
        print(f"Mouse: {'on' if mouse_enabled else 'off'}")
    elif key == ord('r'):
        if cam_x is not None:
            mapper.recenter(cam_x, cam_y)
            print(f"Recentered to ({cam_x:.0f}, {cam_y:.0f})")

safe_release_mouse()
cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("Goodbye.")