"""
macOS input synthesis through Quartz event services.

Replaces pyautogui so we control the details that matter:
  - drags post *Dragged* events (some apps ignore plain moves with a button down)
  - double-clicks carry clickState=2, which is what apps actually check
  - arrow-key combos carry the fn/numpad flags Mission Control needs, so
    ctrl+left/right really switches desktops
  - text is typed as unicode, not key-by-key, so dictation handles any character
"""

import subprocess
import time

import Quartz as Q

# ============================================================
# Screen
# ============================================================
_bounds = Q.CGDisplayBounds(Q.CGMainDisplayID())
SCREEN_W = int(_bounds.size.width)
SCREEN_H = int(_bounds.size.height)


def _post(ev):
    Q.CGEventPost(Q.kCGHIDEventTap, ev)


# ============================================================
# Mouse
# ============================================================
_BUTTONS = {
    "left": (Q.kCGMouseButtonLeft, Q.kCGEventLeftMouseDown,
             Q.kCGEventLeftMouseUp, Q.kCGEventLeftMouseDragged),
    "right": (Q.kCGMouseButtonRight, Q.kCGEventRightMouseDown,
              Q.kCGEventRightMouseUp, Q.kCGEventRightMouseDragged),
}
_held = None  # button currently down, so moves become drags


def _mouse(kind, x, y, button, click_state=1):
    ev = Q.CGEventCreateMouseEvent(None, kind, (x, y), button)
    if click_state != 1:
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, click_state)
    _post(ev)


def cursor_position():
    p = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return float(p.x), float(p.y)


def move(x, y):
    if _held:
        btn, _, _, drag = _BUTTONS[_held]
        _mouse(drag, x, y, btn)
    else:
        _mouse(Q.kCGEventMouseMoved, x, y, Q.kCGMouseButtonLeft)


def mouse_down(x, y, button="left"):
    global _held
    btn, down, _, _ = _BUTTONS[button]
    _mouse(down, x, y, btn)
    _held = button


def mouse_up(x, y, button="left"):
    global _held
    btn, _, up, _ = _BUTTONS[button]
    _mouse(up, x, y, btn)
    _held = None


def click(x, y, button="left", count=1):
    """count=2 sends the second click of a double-click."""
    btn, down, up, _ = _BUTTONS[button]
    _mouse(down, x, y, btn, count)
    _mouse(up, x, y, btn, count)


def scroll(lines):
    _post(Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitLine, 1, int(lines)))


# ============================================================
# Keyboard
# ============================================================
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
    "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47,
    "`": 50, "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51,
    "backspace": 51, "escape": 53, "esc": 53, "forwarddelete": 117,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}
ALIASES = {"plus": "=", "minus": "-", "equal": "=", "comma": ",", "period": "."}
MODIFIERS = {
    "cmd": Q.kCGEventFlagMaskCommand, "command": Q.kCGEventFlagMaskCommand,
    "ctrl": Q.kCGEventFlagMaskControl, "control": Q.kCGEventFlagMaskControl,
    "alt": Q.kCGEventFlagMaskAlternate, "option": Q.kCGEventFlagMaskAlternate,
    "opt": Q.kCGEventFlagMaskAlternate,
    "shift": Q.kCGEventFlagMaskShift,
    "fn": Q.kCGEventFlagMaskSecondaryFn,
}
_ARROWS = {"left", "right", "up", "down"}
_FN_KEYS = {"home", "end", "pageup", "pagedown", "forwarddelete"} | {f"f{i}" for i in range(1, 13)}


def parse_combo(combo):
    """'cmd+shift+t' -> (keycode, flags). Raises ValueError if unknown."""
    parts = [p.strip().lower() for p in combo.split("+")]
    # "cmd++" means cmd and the plus key
    if combo.endswith("++"):
        parts = parts[:-2] + ["="]
    *mods, key = parts
    key = ALIASES.get(key, key)
    if key not in KEYCODES:
        raise ValueError(f"unknown key {key!r} in {combo!r}")
    flags = 0
    for m in mods:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r} in {combo!r}")
        flags |= MODIFIERS[m]
    if key in _ARROWS:
        flags |= Q.kCGEventFlagMaskNumericPad | Q.kCGEventFlagMaskSecondaryFn
    elif key in _FN_KEYS:
        flags |= Q.kCGEventFlagMaskSecondaryFn
    return KEYCODES[key], flags


def press(combo):
    code, flags = parse_combo(combo)
    for is_down in (True, False):
        ev = Q.CGEventCreateKeyboardEvent(None, code, is_down)
        Q.CGEventSetFlags(ev, flags)
        _post(ev)


def type_text(text):
    """Type arbitrary unicode into the focused field, in small chunks."""
    for i in range(0, len(text), 16):
        chunk = text[i:i + 16]
        for is_down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(None, 0, is_down)
            Q.CGEventSetFlags(ev, 0)
            Q.CGEventKeyboardSetUnicodeString(ev, len(chunk), chunk)
            _post(ev)
        time.sleep(0.005)


# ============================================================
# Apps
# ============================================================
_front_cache = (0.0, "")


def frontmost_app():
    """Name of the app owning the frontmost normal window, cached for 0.5 s.

    Uses the window list rather than NSWorkspace, which goes stale in a
    process without a Cocoa run loop."""
    global _front_cache
    now = time.time()
    if now - _front_cache[0] < 0.5:
        return _front_cache[1]
    name = ""
    windows = Q.CGWindowListCopyWindowInfo(
        Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements,
        Q.kCGNullWindowID) or []
    for w in windows:
        if w.get("kCGWindowLayer") == 0 and w.get("kCGWindowOwnerName"):
            name = str(w["kCGWindowOwnerName"])
            break
    _front_cache = (now, name)
    return name


def open_app(name):
    # Whisper gives lowercase words; try them as spoken, then Title Cased
    for candidate in dict.fromkeys([name, name.title()]):
        if subprocess.run(["open", "-a", candidate], capture_output=True).returncode == 0:
            return True
    return False
