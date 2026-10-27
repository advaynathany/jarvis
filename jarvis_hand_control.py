"""
JARVIS-style Hand Tracking Screen Control (v5)
================================================
Starts asleep. Hold an open palm still for half a second to wake; it sleeps
again after a few seconds without input.

Cursor hand:
  - Open palm (>=3 of 4 fingers up)  -> cursor moves
  - Thumb+index pinch                -> left click (held = drag)
  - Thumb+middle pinch               -> right click + 1.5s menu mode
  - Two quick pinches at same spot   -> double-click
  - Peace sign (index+middle only)   -> scroll mode (vertical hand motion)
  - Full fist (thumb out/extended)   -> Escape key + abort drag
  - Full grab (all 5 fingers closed) -> grab & move window under cursor
  - Fast flick left/right/up/down    -> swipe (desktops, Mission Control; per app)
Both hands:
  - Both thumb+index pinching        -> spread/close = zoom, twist = rotate
  - Other hand pinch (hold)          -> push-to-talk; speak a command or dictate

Gesture -> action bindings, per-app overrides and voice commands live in
jarvis.toml. A menu bar item (✋) has Sleep, Mouse, Recenter, Camera, Quit.

  python jarvis_hand_control.py                 run
  python jarvis_hand_control.py --calibrate     redo the 10 s pinch calibration
  python jarvis_hand_control.py --record        record the built-in poses on your hand, then train
  python jarvis_hand_control.py --record thumbs_up   add a new gesture (bind it in jarvis.toml)
  python jarvis_hand_control.py --show-camera   camera window with landmarks (keys q, m, r)
"""

import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from jarvis.config import Config
from jarvis.gestures import CALIBRATION_PATH, GestureClassifier, PoseClassifier, load_calibration
from jarvis.overlay import Overlay


def main():
    ap = argparse.ArgumentParser(description="Hand-gesture control for macOS.")
    ap.add_argument("--calibrate", action="store_true", help="redo pinch calibration")
    ap.add_argument("--record", nargs="*", metavar="GESTURE",
                    help="record gestures on your hand and retrain (no names = the built-ins)")
    ap.add_argument("--train", action="store_true", help="retrain from recorded samples")
    ap.add_argument("--forget", nargs="+", metavar="GESTURE", help="delete a recorded gesture")
    ap.add_argument("--rules", action="store_true", help="ignore the trained model; use hand-written rules")
    ap.add_argument("--show-camera", action="store_true", help="show the camera window")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--camera", help="camera name, 'builtin', or index (overrides jarvis.toml)")
    args = ap.parse_args()

    from jarvis import calibrate
    if args.train:
        calibrate.train()
        return
    if args.forget:
        calibrate.forget(args.forget)
        return

    cfg = Config()
    from jarvis.tracking import HandTracker
    camera = args.camera if args.camera is not None else cfg.get("tracking", "camera", "builtin")
    if isinstance(camera, str) and camera.isdigit():
        camera = int(camera)
    try:
        tracker = HandTracker(camera)
    except RuntimeError as e:
        sys.exit(str(e))
    overlay = Overlay()

    try:
        if args.record is not None:
            calibrate.record(tracker, overlay, args.record)
            return

        pinch, _ = load_calibration()
        if args.calibrate or not os.path.exists(CALIBRATION_PATH):
            pinch, _ = calibrate.calibrate(tracker, overlay, args.show_camera)

        model = None if args.rules else GestureClassifier.load()
        print(f"Poses: {'trained model (' + ', '.join(model.labels) + ')' if model else 'hand-written rules'}")

        from jarvis.control import Controller
        controller = Controller(tracker, overlay, cfg, PoseClassifier(pinch, model),
                                show_camera=args.show_camera)
        if cfg.get("voice", "enabled", True) and not args.no_voice:
            from jarvis.voice import Voice
            controller.voice = Voice(cfg.get("voice", "model", "mlx-community/whisper-base.en-mlx"),
                                     on_text=controller.voice_results.put)
            print("Voice: loading Whisper in the background. Pinch with your other hand to talk.")

        from jarvis.macos import SCREEN_W, SCREEN_H
        print(f"Screen: {SCREEN_W}x{SCREEN_H} | Camera: {tracker.w}x{tracker.h}")
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        overlay.close()
        tracker.close()
        print("Goodbye.")


if __name__ == "__main__":
    main()
