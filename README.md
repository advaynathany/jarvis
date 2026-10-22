# Jarvis

Control a Mac with hand gestures through the laptop camera. MediaPipe tracks 21
hand landmarks per frame; a pose classifier turns them into gestures, and an
intent state machine turns gestures into cursor movement, clicks, drags, scrolls
and window moves.

| Gesture | Action |
|---|---|
| Open palm (3+ fingers up) | Move the cursor |
| Thumb + index pinch | Left click; hold to drag |
| Two quick pinches in place | Double-click |
| Thumb + middle pinch | Right click, then 1.5 s of steadier tracking for the menu |
| Peace sign | Scroll with vertical hand motion |
| Fist, thumb out | Escape, and abort any drag |
| Grab (all five closed) | Move the window under the cursor |

## How it works

- **Pose, then intent.** Classifying a pose and deciding what it means are separate
  stages. A `PoseStabilizer` makes palm, peace and fist hold for a few frames
  before they count, so a hand passing between them doesn't flicker modes.
  Pinches and grabs skip the wait, because click latency is what you feel most.
- **Movement stays on while clicking.** Cursor motion is gated on "palm-ish"
  poses, which include a pinch in progress, so the cursor doesn't freeze the
  moment you start a drag.
- **Active region, not the whole frame.** Only a box covering about 28% of the
  camera width maps to the full screen, so small hand movements cover the display.
  Press `r` to re-centre it on your hand.
- **Smoothing with acceleration** removes tremor without slowing deliberate
  motion. While a context menu is open, smoothing increases and a small deadzone
  turns on, so you can land on a menu item.

## Run

```bash
python3 -m venv jarvis-env && source jarvis-env/bin/activate
pip install -r requirements.txt
python jarvis_hand_control.py      # downloads the 7.8 MB hand model on first run
```

macOS needs Camera permission for the terminal, plus Accessibility permission
for cursor and keyboard control. Keys: `q` quit, `m` toggle mouse control,
`r` recenter the active region. If the wrong camera opens, change `CAMERA_INDEX`.
