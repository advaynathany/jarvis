"""
Layer 2: what does the user mean? Turns hand poses and motion into cursor
movement, clicks, drags, scrolls, swipes, zoom, and push-to-talk.

Jarvis starts asleep. Hold an open palm still for ~0.5 s to wake it; it goes
back to sleep after a few seconds without input, so hands that are typing or
talking don't move the cursor.
"""

import queue
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import macos
from .config import normalize_phrase, run_action
from .filters import OneEuro2D
from .gestures import (
    BUILTIN_POSES, POSE_FIST, POSE_GRAB, POSE_INDEX_PINCH, POSE_MIDDLE_PINCH,
    POSE_NONE, POSE_OPEN, POSE_PEACE, PoseStabilizer, SwipeDetector,
    TwoHandGesture, is_palm_like,
)
from .tracking import draw_hand, to_pixel

SCREEN_W, SCREEN_H = macos.SCREEN_W, macos.SCREEN_H

# ============================================================
# TUNABLES
# ============================================================
ACTIVE_REGION_W_FRAC = 0.28   # smaller region = more screen travel per cm of hand movement
ACTIVE_REGION_H_FRAC = 0.28
CURSOR_GAIN = 1.3

# One Euro filter on the palm point, in camera pixels
FILTER_MIN_CUTOFF = 0.7       # Hz; lower = steadier at rest
FILTER_BETA = 0.02            # how fast smoothing lets go as the hand speeds up
MENU_MIN_CUTOFF = 0.3         # extra-steady while a context menu is open

CLICK_MAX_MS    = 250   # shorter hold to enter drag
DRAG_MIN_PX     = 40    # less movement needed to start drag
CLICK_SETTLE_MS = 120   # cursor freeze at pinch start to absorb tremor
DOUBLE_CLICK_MS = 400
DOUBLE_CLICK_RADIUS_PX = 30
CLICK_FREEZE_MS = 60

MENU_MODE_MS = 1500

SCROLL_COOLDOWN = 0.05
SCROLL_SENSITIVITY = 200

HAND_LOSS_GRACE_MS = 300
ESCAPE_COOLDOWN = 0.6
CUSTOM_ACTION_COOLDOWN = 1.0

WAKE_STILL_FRAC = 0.03        # palm must stay within this fraction of frame width
INPUT_MOVE_PX = 25            # cursor travel that counts as "input" for sleep
PTT_ARM_MS = 250              # other-hand pinch must hold this long before recording
HAND_SWAP_FRAC = 0.2          # a primary-hand jump this big = different hand


# ============================================================
# Cursor mapper: active region + One Euro filter
# ============================================================
class CursorMapper:
    def __init__(self, cam_w, cam_h):
        self.cam_w, self.cam_h = cam_w, cam_h
        self.cx, self.cy = cam_w / 2, cam_h / 2
        self.half_w = cam_w * ACTIVE_REGION_W_FRAC / 2
        self.half_h = cam_h * ACTIVE_REGION_H_FRAC / 2
        self.sx, self.sy = SCREEN_W / 2, SCREEN_H / 2
        self.filter = OneEuro2D(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
        self.prev = None

    def recenter(self, cam_x, cam_y, t):
        self.cx, self.cy = cam_x, cam_y
        self.reset_velocity(cam_x, cam_y, t)

    def reset_velocity(self, cam_x, cam_y, t):
        """Call when re-engaging movement after a pause, to avoid a jump."""
        self.filter.reset()
        self.prev = self.filter(cam_x, cam_y, t)

    def update(self, cam_x, cam_y, t, min_cutoff=None):
        fx, fy = self.filter(cam_x, cam_y, t, min_cutoff)
        if self.prev is None:
            self.prev = (fx, fy)
            return self.sx, self.sy
        dx, dy = fx - self.prev[0], fy - self.prev[1]
        self.prev = (fx, fy)
        self.sx += dx * SCREEN_W / (2 * self.half_w) * CURSOR_GAIN
        self.sy += dy * SCREEN_H / (2 * self.half_h) * CURSOR_GAIN
        self.sx = float(np.clip(self.sx, 0, SCREEN_W - 1))
        self.sy = float(np.clip(self.sy, 0, SCREEN_H - 1))
        return self.sx, self.sy

    def drift_toward(self, cam_x, cam_y):
        """Slowly re-centre the region when the hand works near its edge."""
        if abs(cam_x - self.cx) > self.half_w * 0.85:
            self.cx += (cam_x - self.cx) * 0.005
        if abs(cam_y - self.cy) > self.half_h * 0.85:
            self.cy += (cam_y - self.cy) * 0.005


# ============================================================
# Intent state
# ============================================================
@dataclass
class IntentState:
    mode: str = "idle"  # idle | clicking | dragging | scrolling | grabbing | zooming

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

    pinch_suppressed: bool = False  # pinch left over from a zoom; ignore until released

    custom_fired: str = ""
    last_custom_time: float = 0.0


class Controller:
    def __init__(self, tracker, overlay, cfg, poses, voice=None, show_camera=False):
        self.tracker = tracker
        self.overlay = overlay
        self.cfg = cfg
        self.poses = poses
        self.voice = voice
        self.show_camera = show_camera

        self.mapper = CursorMapper(tracker.w, tracker.h)
        self.state = IntentState()
        self.stabilizer = PoseStabilizer()
        self.pinch = poses.detector()
        self.other_stabilizer = PoseStabilizer()
        self.other_pinch = poses.detector()
        self.swipe = SwipeDetector()
        self.two_hand = TwoHandGesture()
        self.cursor_history = deque(maxlen=60)

        self.awake = False
        self.mouse_enabled = True
        self.running = True
        self.wake_track = deque()
        self.last_input_time = 0.0
        self.last_input_pos = (0.0, 0.0)
        self.primary_pos = None
        self.primary_seen = 0.0
        self.was_moving = False
        self.ptt_start = 0.0
        self.voice_results = queue.Queue()
        self.last_config_check = 0.0
        self.fps_history = deque(maxlen=30)
        self.prev_frame_time = time.time()

        self.hooks = {"sleep": self.sleep, "mouse": self.toggle_mouse,
                      "recenter": self.recenter_on_hand}

    # ============================================================
    # Lifecycle
    # ============================================================
    def wake(self, hand, t):
        self.awake = True
        self.mapper.recenter(*hand.palm, t)
        self.mapper.sx, self.mapper.sy = macos.cursor_position()
        self.mark_input(t)
        self.swipe.reset()
        self.overlay.send("moving", flash="awake")

    def sleep(self):
        self.awake = False
        self.release_all()
        self.wake_track.clear()
        self.overlay.send("asleep")

    def toggle_mouse(self):
        self.mouse_enabled = not self.mouse_enabled
        if not self.mouse_enabled:
            self.release_all()
        print(f"Mouse: {'on' if self.mouse_enabled else 'off'}")

    def recenter_on_hand(self):
        if self.primary_pos:
            self.mapper.recenter(*self.primary_pos, time.time())
            print(f"Recentered to ({self.primary_pos[0]:.0f}, {self.primary_pos[1]:.0f})")

    def mark_input(self, t):
        self.last_input_time = t
        self.last_input_pos = (self.mapper.sx, self.mapper.sy)

    def release_all(self):
        self.safe_release_mouse()
        if self.voice:
            self.voice.cancel()
        self.ptt_start = 0.0
        self.two_hand.stop()
        self.state.mode = "idle"
        self.state.scroll_history.clear()

    def safe_release_mouse(self):
        if self.state.mouse_down:
            macos.mouse_up(self.mapper.sx, self.mapper.sy, self.state.mouse_down_kind or "left")
            self.state.mouse_down = False
            self.state.mouse_down_kind = ""

    def fire(self, gesture, t):
        """Run the configured action for a discrete gesture and flash it."""
        action = self.cfg.action_for(gesture)
        done = run_action(action, self.hooks)
        if done:
            self.overlay.send(self.display_state(t), flash=f"{gesture.replace('_', ' ')} → {done}")
            self.mark_input(t)
        return done

    # ============================================================
    # Hand assignment
    # ============================================================
    def pick_hands(self, hands, t):
        """(primary, other). Follows the primary hand by position once it's
        established, so a second hand entering doesn't steal the cursor."""
        if not hands:
            return None, None
        if len(hands) == 1:
            return hands[0], None
        a, b = hands[0], hands[1]
        if self.primary_pos and t - self.primary_seen < 0.5:
            da = np.hypot(a.palm[0] - self.primary_pos[0], a.palm[1] - self.primary_pos[1])
            db = np.hypot(b.palm[0] - self.primary_pos[0], b.palm[1] - self.primary_pos[1])
            return (a, b) if da <= db else (b, a)
        want = self.cfg.get("tracking", "primary_hand", "Right")
        if a.label != b.label:
            return (a, b) if a.label == want else (b, a)
        # Same label on both (misclassified): right hand is on the right in the mirrored frame
        right, left = (a, b) if a.palm[0] > b.palm[0] else (b, a)
        return (right, left) if want == "Right" else (left, right)

    # ============================================================
    # Main loop
    # ============================================================
    def run(self):
        self.overlay.send("asleep")
        print("Asleep. Hold an open palm still for half a second to wake.")
        try:
            self._loop()
        finally:
            self.release_all()

    def _loop(self):
        while self.running:
            frame, hands, t = self.tracker.read()
            if frame is None:
                continue
            self.handle_commands()
            if t - self.last_config_check > 1.0:
                self.cfg.reload_if_changed()
                self.last_config_check = t
            self.step(frame, hands, t)
            if self.show_camera:
                self.draw_hud(frame, hands)
                cv2.imshow("JARVIS Hand Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.toggle_mouse()
                elif key == ord('r'):
                    self.recenter_on_hand()

    def handle_commands(self):
        for cmd in self.overlay.poll():
            if cmd == "quit":
                self.running = False
            elif cmd == "toggle_mouse":
                self.toggle_mouse()
            elif cmd == "recenter":
                self.recenter_on_hand()
            elif cmd == "toggle_sleep":
                if self.awake:
                    self.sleep()
                else:
                    self.awake = True
                    self.mapper.sx, self.mapper.sy = macos.cursor_position()
                    self.mark_input(time.time())
                    self.overlay.send("moving", flash="awake")
            elif cmd == "toggle_camera":
                self.show_camera = not self.show_camera
                if not self.show_camera:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)

    def step(self, frame, hands, t):
        primary, other = self.pick_hands(hands, t)
        self.pose = POSE_NONE
        self.other_pose = POSE_NONE

        if primary:
            if self.primary_pos and np.hypot(primary.palm[0] - self.primary_pos[0],
                                             primary.palm[1] - self.primary_pos[1]) > HAND_SWAP_FRAC * self.tracker.w:
                self.mapper.reset_velocity(*primary.palm, t)
                self.swipe.reset()
            self.primary_pos = primary.palm
            self.primary_seen = t
            self.state.last_hand_seen = t
            self.pose = self.stabilizer.update(self.poses.classify(primary, self.pinch))
        else:
            self.stabilizer.update(POSE_NONE)
            self.pinch.index = self.pinch.middle = False
        if other:
            self.other_pose = self.other_stabilizer.update(self.poses.classify(other, self.other_pinch))
        else:
            self.other_stabilizer.update(POSE_NONE)
            self.other_pinch.index = self.other_pinch.middle = False

        self.drain_voice(t)

        if not self.awake:
            self.check_wake(primary, t)
            return
        if not self.mouse_enabled:
            self.overlay.send("paused")
            return

        busy = (self.state.mouse_down or self.state.mode != "idle"
                or (self.voice and (self.voice.recording or self.voice.busy)))
        sleep_after = self.cfg.get("tracking", "sleep_after_s", 5.0)
        if not busy and t - self.last_input_time > sleep_after:
            self.sleep()
            return

        if self.handle_two_hands(primary, other, t):
            self.overlay.send(self.display_state(t))
            return
        self.handle_primary(primary, t)

        if np.hypot(self.mapper.sx - self.last_input_pos[0],
                    self.mapper.sy - self.last_input_pos[1]) > INPUT_MOVE_PX:
            self.mark_input(t)
        self.overlay.send(self.display_state(t))

    def check_wake(self, hand, t):
        hold_s = self.cfg.get("tracking", "wake_hold_ms", 500) / 1000
        if hand is None or self.pose != POSE_OPEN:
            if self.wake_track:
                self.wake_track.clear()
                self.overlay.send("asleep")
            return
        self.wake_track.append((t, *hand.palm))
        xs = [p[1] for p in self.wake_track]
        ys = [p[2] for p in self.wake_track]
        if max(max(xs) - min(xs), max(ys) - min(ys)) > WAKE_STILL_FRAC * self.tracker.w:
            # moved: keep only the recent part of the window
            while len(self.wake_track) > 1 and t - self.wake_track[0][0] > 0.1:
                self.wake_track.popleft()
            return
        held = t - self.wake_track[0][0]
        if held >= hold_s:
            self.wake_track.clear()
            self.wake(hand, t)
        elif held > 0.15:
            self.overlay.send("waking")

    # ============================================================
    # Two hands: zoom/rotate, push-to-talk
    # ============================================================
    def handle_two_hands(self, primary, other, t):
        """Returns True if a two-hand gesture owns this frame."""
        st = self.state
        both_pinch = (primary and other and self.pose == POSE_INDEX_PINCH
                      and self.other_pose == POSE_INDEX_PINCH)
        recording = self.voice and self.voice.recording

        if self.two_hand.active:
            if not both_pinch:
                self.two_hand.stop()
                st.mode = "idle"
                st.pinch_suppressed = True
                if primary:
                    self.mapper.reset_velocity(*primary.palm, t)
                return False
            left, right = sorted((primary, other), key=lambda h: h.palm[0])
            step = self.two_hand.update(left, right)
            if step:
                self.fire(step, t)
            self.mark_input(t)
            return True

        if both_pinch and not recording and st.mode in ("idle", "clicking"):
            # A click that hadn't fired yet becomes the start of a zoom; nothing
            # was sent for it, because mouseDown is deferred until drag starts.
            st.mode = "zooming"
            st.pinch_kind = ""
            self.ptt_start = 0.0
            left, right = sorted((primary, other), key=lambda h: h.palm[0])
            self.two_hand.start(left, right)
            return True

        # Push-to-talk: other hand pinches, cursor hand doesn't
        if self.voice is None:
            return False
        other_pinching = other is not None and self.other_pose == POSE_INDEX_PINCH
        if other_pinching and not recording:
            if not self.ptt_start:
                self.ptt_start = t
            elif (t - self.ptt_start) * 1000 > PTT_ARM_MS and self.pose != POSE_INDEX_PINCH:
                if not self.voice.ready:
                    self.overlay.send(self.display_state(t), flash="voice still loading")
                    self.ptt_start = t + 2   # don't nag every frame
                else:
                    self.voice.start()
                    self.mark_input(t)
        elif not other_pinching:
            self.ptt_start = 0.0
            if recording:
                self.voice.stop()
                self.mark_input(t)
        return False

    def drain_voice(self, t):
        while not self.voice_results.empty():
            text = self.voice_results.get()
            norm = normalize_phrase(text)
            print(f"[voice] {text!r}")
            if norm.startswith("type "):
                macos.type_text(text.strip().split(" ", 1)[1])
                self.overlay.send(self.display_state(t), flash="typed")
            else:
                action = self.cfg.voice_command(text)
                if action:
                    done = run_action(action, self.hooks)
                    self.overlay.send(self.display_state(t), flash=f"“{norm}” → {done or 'nothing'}")
                else:
                    macos.type_text(text.strip())
                    self.overlay.send(self.display_state(t), flash="typed")
            self.mark_input(t)

    # ============================================================
    # Cursor hand
    # ============================================================
    def handle_primary(self, hand, t):
        st = self.state
        mapper = self.mapper
        pose = self.pose

        if hand is None:
            # Hand lost
            if st.mode in ("dragging", "grabbing"):
                if (t - st.last_hand_seen) * 1000 > HAND_LOSS_GRACE_MS:
                    self.safe_release_mouse()
                    st.mode = "idle"
            else:
                self.safe_release_mouse()
                st.mode = "idle"
                st.scroll_history.clear()
            self.swipe.reset()
            self.was_moving = False
            return

        cam_x, cam_y = hand.palm
        if st.mode == "idle" and pose == POSE_OPEN:
            mapper.drift_toward(cam_x, cam_y)

        # ---------------- swipes ----------------
        if st.mode == "idle" and pose not in (POSE_INDEX_PINCH, POSE_MIDDLE_PINCH, POSE_GRAB):
            hit = self.swipe.update(t, cam_x / self.tracker.w, cam_y / self.tracker.h)
            if hit:
                name, t0 = hit
                # Undo the cursor travel the flick caused before it was recognised
                for ts, sx, sy in reversed(self.cursor_history):
                    if ts <= t0:
                        mapper.sx, mapper.sy = sx, sy
                        macos.move(sx, sy)
                        break
                self.fire(name, t)
                st.freeze_until = self.swipe.cooldown_until
        else:
            self.swipe.reset()

        # ---------------- cursor movement ----------------
        in_menu = t < st.menu_until
        click_settling = (st.mode == "clicking"
                          and (t - st.pinch_start_time) * 1000 < CLICK_SETTLE_MS)
        moving = (is_palm_like(pose) and t >= st.freeze_until
                  and st.mode != "scrolling" and not click_settling)
        # Reset velocity when movement re-engages so we don't snap-jump
        if moving and not self.was_moving:
            mapper.reset_velocity(cam_x, cam_y, t)
        if moving:
            sx, sy = mapper.update(cam_x, cam_y, t, MENU_MIN_CUTOFF if in_menu else None)
            macos.move(sx, sy)
        self.was_moving = moving
        self.cursor_history.append((t, mapper.sx, mapper.sy))

        # ---------------- pose-driven intents ----------------
        # GRAB = hold mouseDown and drag window (all 5 fingers closed)
        if pose == POSE_GRAB:
            if st.mode != "grabbing":
                self.safe_release_mouse()
                st.mode = "grabbing"
                macos.mouse_down(mapper.sx, mapper.sy, "left")
                st.mouse_down = True
                st.mouse_down_kind = "left"

        # FIST = Escape (and abort drag / grab if active)
        elif pose == POSE_FIST:
            if t - st.last_escape_time > ESCAPE_COOLDOWN:
                self.fire("fist", t)
                st.last_escape_time = t
                if st.mode in ("dragging", "grabbing"):
                    self.safe_release_mouse()
                st.mode = "idle"
                st.menu_until = 0
                st.scroll_history.clear()

        # PEACE = scroll
        elif pose == POSE_PEACE:
            # Cancel any pending click/drag/grab — peace is a different intent
            if st.mode in ("clicking", "dragging", "grabbing"):
                self.safe_release_mouse()
            st.mode = "scrolling"
            st.scroll_history.append(hand.lm[9].y)  # middle MCP — stable vertical reference
            if len(st.scroll_history) >= 3:
                delta = st.scroll_history[0] - st.scroll_history[-1]
                if abs(delta) > 0.02 and t - st.last_scroll_time > SCROLL_COOLDOWN:
                    macos.scroll(int(delta * SCROLL_SENSITIVITY))
                    st.last_scroll_time = t
                    self.mark_input(t)

        else:
            # Leaving scroll or grab mode
            if st.mode == "scrolling":
                st.mode = "idle"
                st.scroll_history.clear()
            elif st.mode == "grabbing":
                self.safe_release_mouse()
                st.mode = "idle"
            elif st.mode == "zooming":
                st.mode = "idle"
            self.handle_pinch(pose, t)
            self.handle_custom(pose, t)

    def handle_pinch(self, pose, t):
        st = self.state
        mapper = self.mapper
        is_left_pinch = (pose == POSE_INDEX_PINCH)
        is_right_pinch = (pose == POSE_MIDDLE_PINCH)
        any_pinch = is_left_pinch or is_right_pinch
        if st.pinch_suppressed:
            st.pinch_suppressed = any_pinch
            return

        # Pinch ENTRY. Nothing is sent yet: a quick pinch becomes a click on
        # release, a held/moved one a drag, and a pinch joined by the other
        # hand a zoom.
        if any_pinch and st.mode == "idle":
            st.mode = "clicking"
            st.pinch_start_time = t
            st.pinch_start_pos = (mapper.sx, mapper.sy)
            st.pinch_kind = "left" if is_left_pinch else "right"

        # Pinch HELD
        elif any_pinch and st.mode in ("clicking", "dragging"):
            held_ms = (t - st.pinch_start_time) * 1000
            moved_px = np.hypot(mapper.sx - st.pinch_start_pos[0],
                                mapper.sy - st.pinch_start_pos[1])
            if st.mode == "clicking" and st.pinch_kind == "left":
                if held_ms > CLICK_MAX_MS or moved_px > DRAG_MIN_PX:
                    st.mode = "dragging"
                    macos.mouse_down(*st.pinch_start_pos, "left")
                    macos.move(mapper.sx, mapper.sy)
                    st.mouse_down = True
                    st.mouse_down_kind = "left"
            self.mark_input(t)

        # Pinch RELEASE
        elif not any_pinch and st.mode in ("clicking", "dragging"):
            if st.mode == "dragging":
                self.safe_release_mouse()
            else:
                # Click where the pinch started: any drift while pinching is tremor
                pos = st.pinch_start_pos
                mapper.sx, mapper.sy = pos
                since_last = (t - st.last_click_time) * 1000
                dist_last = np.hypot(pos[0] - st.last_click_pos[0], pos[1] - st.last_click_pos[1])
                if st.pinch_kind == "left":
                    is_double = (
                        since_last < DOUBLE_CLICK_MS and
                        dist_last < DOUBLE_CLICK_RADIUS_PX and
                        st.last_click_kind == "left"
                    )
                    macos.click(*pos, "left", count=2 if is_double else 1)
                    st.last_click_kind = "left"
                elif st.pinch_kind == "right":
                    macos.click(*pos, "right")
                    st.last_click_kind = "right"
                    st.menu_until = t + MENU_MODE_MS / 1000.0
                st.last_click_time = t
                st.last_click_pos = pos
                st.freeze_until = t + CLICK_FREEZE_MS / 1000.0
            st.mode = "idle"
            st.pinch_kind = ""
            self.mark_input(t)

    def handle_custom(self, pose, t):
        """Trained gestures beyond the built-ins fire their configured action
        once per hold."""
        st = self.state
        custom = pose not in BUILTIN_POSES and pose not in (
            POSE_NONE, POSE_INDEX_PINCH, POSE_MIDDLE_PINCH)
        if not custom:
            st.custom_fired = ""
            return
        if pose != st.custom_fired and t - st.last_custom_time > CUSTOM_ACTION_COOLDOWN:
            st.custom_fired = pose
            st.last_custom_time = t
            if not self.fire(pose, t):
                print(f"[gesture] {pose}: no action bound in jarvis.toml [actions]")

    # ============================================================
    # Display
    # ============================================================
    def display_state(self, t):
        if not self.awake:
            return "asleep"
        if self.voice and self.voice.recording:
            return "listening"
        if self.voice and self.voice.busy:
            return "transcribing"
        if self.state.mode == "zooming":
            return "zooming"
        if t < self.state.menu_until and self.state.mode == "idle":
            return "menu"
        mode = self.state.mode
        if mode == "idle":
            return "moving"
        return mode

    def draw_hud(self, frame, hands):
        w, h = self.tracker.w, self.tracker.h
        m = self.mapper
        cv2.rectangle(frame, (int(m.cx - m.half_w), int(m.cy - m.half_h)),
                      (int(m.cx + m.half_w), int(m.cy + m.half_h)), (255, 100, 0), 2)
        cv2.drawMarker(frame, (int(m.cx), int(m.cy)), (255, 100, 0), cv2.MARKER_CROSS, 12, 2)
        for hand in hands:
            is_primary = self.primary_pos == hand.palm
            draw_hand(frame, hand.lm, w, h, (0, 200, 255) if is_primary else (255, 120, 120))
            if is_primary:
                cv2.circle(frame, (int(hand.palm[0]), int(hand.palm[1])), 10, (0, 255, 0), cv2.FILLED)
                if self.pose in (POSE_INDEX_PINCH, POSE_MIDDLE_PINCH):
                    tip = 8 if self.pose == POSE_INDEX_PINCH else 12
                    a, b = to_pixel(hand.lm[4], w, h), to_pixel(hand.lm[tip], w, h)
                    cv2.circle(frame, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2), 22,
                               (0, 255, 255) if tip == 8 else (0, 100, 255), 3)

        now = time.time()
        fps = 1 / max(now - self.prev_frame_time, 1e-6)
        self.prev_frame_time = now
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, "JARVIS HAND CONTROL v5", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.0f}  Pose: {self.pose}  Other: {self.other_pose}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"MODE: {self.display_state(now).upper()}", (w - 330, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.awake else (150, 150, 150), 2)
        cv2.putText(frame, "[q]uit  [m]ouse  [r]ecenter", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
