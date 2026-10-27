"""
On-screen state indicator + menu bar item, replacing the camera window.

AppKit wants the main thread and a run loop, which the camera loop can't
give it, so the overlay runs as its own process (`python -m jarvis.overlay`):
  parent -> child  JSON lines on stdin:  {"state": "moving", "flash": "ctrl+right"}
  child -> parent  plain lines on stdout: "quit", "toggle_mouse", ...
If the parent dies, stdin closes and the overlay exits with it.
"""

import json
import os
import queue
import subprocess
import sys
import threading

# state -> (r, g, b) for the dot, and the label shown next to it
STATES = {
    "asleep":       ((0.55, 0.55, 0.58), ""),
    "waking":       ((0.40, 0.80, 1.00), "waking"),
    "moving":       ((0.30, 0.85, 0.55), "moving"),
    "clicking":     ((1.00, 0.85, 0.20), "click"),
    "dragging":     ((0.25, 0.95, 0.35), "dragging"),
    "scrolling":    ((1.00, 0.65, 0.20), "scrolling"),
    "grabbing":     ((1.00, 0.35, 0.80), "moving window"),
    "menu":         ((0.85, 0.45, 1.00), "menu"),
    "zooming":      ((0.65, 0.55, 1.00), "zoom / rotate"),
    "listening":    ((1.00, 0.30, 0.30), "listening"),
    "transcribing": ((0.35, 0.60, 1.00), "transcribing"),
    "paused":       ((0.90, 0.30, 0.30), "mouse off"),
    "setup":        ((1.00, 1.00, 1.00), ""),
}


# ============================================================
# Client (used by the camera process)
# ============================================================
class Overlay:
    def __init__(self, enabled=True):
        self.proc = None
        self.last = None
        self.commands = queue.Queue()
        if not enabled:
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "jarvis.overlay"], cwd=root,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            print(f"[overlay] failed to start: {e}")
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.commands.put(line.strip())

    def send(self, state, text=None, flash=None):
        msg = {"state": state}
        if text is not None:
            msg["text"] = text
        if flash:
            msg["flash"] = flash
        if msg == self.last and not flash:
            return
        self.last = msg
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps(msg) + "\n")
            except (BrokenPipeError, OSError):
                self.proc = None
        if text and state == "setup":
            print(text)

    def poll(self):
        out = []
        while not self.commands.empty():
            out.append(self.commands.get())
        return out

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ============================================================
# Overlay process
# ============================================================
def main():
    import objc
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
        NSColor, NSEvent, NSFont, NSMakeRect, NSMenu, NSMenuItem, NSPanel, NSScreen,
        NSStatusBar, NSTextField, NSTimer, NSVariableStatusItemLength, NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSObject
    from PyObjCTools import AppHelper
    from Quartz import CGColorCreateGenericGray, CGColorCreateGenericRGB

    H = 26
    DOT = 10
    PAD = 9
    FLASH_S = 1.2

    def emit(cmd):
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()

    class Controller(NSObject):
        def init(self):
            self = objc.super(Controller, self).init()
            self.state = "asleep"
            self.text = ""
            self.flash = ""
            self.flash_ticks = 0
            self.build_panel()
            self.build_menu()
            self.render()
            return self

        def build_panel(self):
            self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 120, H),
                NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
                NSBackingStoreBuffered, False)
            self.panel.setOpaque_(False)
            self.panel.setBackgroundColor_(NSColor.clearColor())
            self.panel.setHasShadow_(False)
            self.panel.setIgnoresMouseEvents_(True)
            self.panel.setLevel_(1000)  # above menus and full-screen apps
            self.panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary)

            content = self.panel.contentView()
            content.setWantsLayer_(True)
            content.layer().setCornerRadius_(H / 2)
            content.layer().setBackgroundColor_(CGColorCreateGenericGray(0.08, 0.72))

            self.dot = NSView.alloc().initWithFrame_(NSMakeRect(PAD, (H - DOT) / 2, DOT, DOT))
            self.dot.setWantsLayer_(True)
            self.dot.layer().setCornerRadius_(DOT / 2)
            content.addSubview_(self.dot)

            self.label = NSTextField.labelWithString_("")
            self.label.setFont_(NSFont.systemFontOfSize_weight_(12, 0.3))
            self.label.setTextColor_(NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.95))
            content.addSubview_(self.label)
            self.panel.orderFrontRegardless()

            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1 / 60, self, "tick:", None, True)

        def build_menu(self):
            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            self.item.button().setTitle_("✋")
            menu = NSMenu.alloc().init()
            for title, cmd in [("Wake / Sleep", "toggle_sleep"),
                               ("Mouse On / Off", "toggle_mouse"),
                               ("Recenter on hand", "recenter"),
                               ("Show / Hide camera", "toggle_camera"),
                               (None, None),
                               ("Quit Jarvis", "quit")]:
                if title is None:
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "menu:", "")
                mi.setTarget_(self)
                mi.setRepresentedObject_(cmd)
                menu.addItem_(mi)
            self.item.setMenu_(menu)

        def menu_(self, sender):
            emit(sender.representedObject())

        def apply_(self, msg):
            self.state = msg.get("state", self.state)
            self.text = msg.get("text", "")
            if msg.get("flash"):
                self.flash = msg["flash"]
                self.flash_ticks = int(FLASH_S * 60)
            self.render()

        def render(self):
            rgb, default = STATES.get(self.state, STATES["moving"])
            text = self.flash or self.text or default
            self.dot.layer().setBackgroundColor_(CGColorCreateGenericRGB(*rgb, 1.0))
            self.label.setStringValue_(text)
            self.label.sizeToFit()
            lw = self.label.frame().size.width if text else 0
            self.label.setFrameOrigin_((PAD + DOT + 6, (H - self.label.frame().size.height) / 2))
            w = PAD + DOT + (6 + lw + PAD if text else PAD)
            f = self.panel.frame()
            self.panel.setFrame_display_(NSMakeRect(f.origin.x, f.origin.y, w, H), True)
            # asleep: just a faint dot, so it's there but out of the way
            self.panel.setAlphaValue_(0.45 if self.state == "asleep" and not text else 1.0)
            title = {"asleep": "✋", "paused": "✋̸"}.get(self.state, "🖐")
            self.item.button().setTitle_(title)

        def tick_(self, timer):
            if self.flash_ticks:
                self.flash_ticks -= 1
                if not self.flash_ticks:
                    self.flash = ""
                    self.render()
            m = NSEvent.mouseLocation()
            f = self.panel.frame()
            x, y = m.x + 16, m.y - 16 - H
            for s in NSScreen.screens():
                sf = s.frame()
                if sf.origin.x <= m.x < sf.origin.x + sf.size.width and \
                        sf.origin.y <= m.y < sf.origin.y + sf.size.height:
                    if x + f.size.width > sf.origin.x + sf.size.width:
                        x = m.x - 16 - f.size.width
                    if y < sf.origin.y:
                        y = m.y + 16
                    break
            self.panel.setFrameOrigin_((x, y))

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    ctrl = Controller.alloc().init()

    def read_stdin():
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            AppHelper.callAfter(ctrl.apply_, msg)
        AppHelper.callAfter(AppHelper.stopEventLoop)

    threading.Thread(target=read_stdin, daemon=True).start()
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
