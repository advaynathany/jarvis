"""
Gesture -> action config (jarvis.toml), with per-app overrides.

The file is re-read when its mtime changes, so bindings can be edited while
Jarvis runs.
"""

import os
import re
import tomllib

from . import macos

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jarvis.toml")


class Config:
    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.mtime = 0.0
        self.data = {}
        self.reload_if_changed()

    def reload_if_changed(self):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if mtime == self.mtime:
            return
        try:
            with open(self.path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            print(f"[config] {self.path}: {e} (keeping previous config)")
            self.mtime = mtime
            return
        self.data, self.mtime = data, mtime
        self._check()
        print(f"[config] loaded {self.path}")

    def _check(self):
        """Report bad key combos at load time, not when the gesture fires."""
        tables = [self.data.get("actions", {}), self.data.get("voice", {}).get("commands", {})]
        tables += list(self.data.get("apps", {}).values())
        for table in tables:
            for name, action in table.items():
                if isinstance(action, str) and action and ":" not in action:
                    try:
                        macos.parse_combo(action)
                    except ValueError as e:
                        print(f"[config] {name}: {e}")

    def section(self, name):
        return self.data.get(name, {})

    def get(self, section, key, default=None):
        return self.data.get(section, {}).get(key, default)

    def action_for(self, gesture, app=None):
        """Per-app binding if the frontmost app has one, else the default."""
        app = macos.frontmost_app() if app is None else app
        per_app = self.data.get("apps", {}).get(app, {})
        if gesture in per_app:
            return per_app[gesture]
        return self.data.get("actions", {}).get(gesture, "")

    def voice_command(self, phrase):
        """Map a transcript to an action, or None if it should be typed."""
        norm = normalize_phrase(phrase)
        if not norm:
            return None
        commands = {normalize_phrase(k): v for k, v in
                    self.data.get("voice", {}).get("commands", {}).items()}
        if norm in commands:
            return commands[norm]
        m = re.fullmatch(r"(?:open|launch|start) (.+)", norm)
        if m:
            return "open:" + m.group(1)
        return None


def normalize_phrase(text):
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(text.split())


def run_action(action, hooks=None):
    """Execute an action string. hooks maps 'jarvis:*' names to callables.
    Returns a short description for the overlay, or '' if nothing ran."""
    if not action:
        return ""
    kind, _, arg = action.partition(":")
    if arg and kind == "open":
        ok = macos.open_app(arg)
        return f"open {arg}" if ok else f"no app {arg!r}"
    if arg and kind == "type":
        macos.type_text(arg)
        return "typed"
    if arg and kind == "jarvis":
        fn = (hooks or {}).get(arg)
        if fn:
            fn()
            return arg
        return ""
    try:
        macos.press(action)
    except ValueError as e:
        print(f"[action] {e}")
        return ""
    return action
