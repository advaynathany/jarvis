# Jarvis

Control a Mac with your hands and voice through the laptop camera. MediaPipe
tracks 21 landmarks on each of up to two hands; a pose classifier trained on
your own hand turns them into gestures, and an intent state machine turns
gestures into cursor movement, clicks, drags, scrolls, swipes, zoom and
push-to-talk.

Jarvis starts **asleep**. Hold an open palm still for half a second to wake it.
It goes back to sleep after 5 seconds without input, so typing or talking with
your hands doesn't move the cursor. A small pill next to the cursor shows the
state (asleep, moving, dragging, scrolling, listening…), and a ✋ menu bar item
has Sleep, Mouse on/off, Recenter, Show camera and Quit.

| Gesture | Action |
|---|---|
| Open palm (3+ fingers up) | Move the cursor |
| Thumb + index pinch | Left click; hold to drag |
| Two quick pinches in place | Double-click |
| Thumb + middle pinch | Right click, then 1.5 s of steadier tracking for the menu |
| Peace sign | Scroll with vertical hand motion |
| Fist, thumb out | Escape, and abort any drag |
| Grab (all five closed) | Move the window under the cursor |
| Fast flick left / right | Next / previous desktop (browser: forward / back; Keynote: slides) |
| Fast flick up / down | Mission Control / App Exposé |
| Both hands pinching, spread or close | Zoom in / out (⌘+ / ⌘−) |
| Both hands pinching, twist | Rotate (Preview and Photos) |
| Pinch with the *other* hand, hold | Push-to-talk |

## Voice

Hold a pinch with your non-cursor hand and speak, then let go. Whisper runs
locally (mlx-whisper on the Apple GPU, about 0.1 s per phrase on an M2). If the
whole phrase is a command it runs: "open Safari", "close this", "next tab",
"new tab", "go back", "copy", "paste", "undo", "mission control", "go to sleep"
and the rest of `[voice.commands]` in `jarvis.toml`. Anything else is typed
into the focused field. Start with "type …" to force dictation.

## Configuration

`jarvis.toml` maps gestures to actions, with per-app overrides matched on the
frontmost app's name, and holds the voice commands. It reloads on save. An
action is a key combo (`"cmd+shift+t"`), `"open:AppName"`, `"type:text"`,
`"jarvis:sleep"`, or `""` for nothing.

```toml
[actions]
swipe_left = "ctrl+right"

[apps.Keynote]
swipe_left = "right"     # next slide
```

## Training it on your hand

- **Calibration** (about 10 s) runs on first launch: relaxed palm, then a held
  thumb–index pinch and a held thumb–middle pinch. It sets your pinch
  thresholds between your own pinched and relaxed distances. Redo it with
  `--calibrate`.
- **Recorded gestures.** `--record` has you hold each built-in pose for 4 s and
  trains a small classifier (a numpy MLP) on the landmarks. Until you do this,
  hand-written rules are used. `--record thumbs_up` adds a new gesture; bind it
  in `[actions]` (`thumbs_up = "cmd+tab"`). `--forget thumbs_up` removes it.

Your calibration and samples live in `profile/`, which is git-ignored.

## How it works

- **Pose, then intent.** Classifying a pose and deciding what it means are separate
  stages. A `PoseStabilizer` makes palm, peace and fist hold for a few frames
  before they count. Pinches and grabs skip the wait, because click latency is
  what you feel most.
- **The cursor follows a stable point:** the centre of the wrist and four
  knuckles, which doesn't move when you pinch, so clicks land where you aimed.
- **One Euro filter.** Smooths heavily when the hand is nearly still and barely
  at all when it moves fast, replacing the old smoothing and acceleration
  constants. A context menu lowers its cutoff so you can land on an item.
- **Active region, not the whole frame.** Only a box covering about 28% of the
  camera width maps to the full screen, so small hand movements cover the
  display. It re-centres on your hand when you wake Jarvis.
- **Clicks are decided on release.** A pinch sends nothing until it's released
  (click), held or moved (drag), or joined by the other hand (zoom). So a zoom
  never clicks by accident.
- **Swipes vs moves.** Moving the cursor across the screen only takes about 28%
  of the frame. A flick that covers more than 32% within 300 ms counts as a
  swipe, and the cursor snaps back to where the flick began.
- Input goes through Quartz events directly. Drags send real drag events,
  double-clicks carry the click count apps check for, and arrow combos carry
  the flags Mission Control needs.

## Run

```bash
python3 -m venv jarvis-env && source jarvis-env/bin/activate
pip install -r requirements.txt
python jarvis_hand_control.py      # downloads the 7.8 MB hand model on first run
```

Flags: `--show-camera` (landmark window; keys `q` quit, `m` mouse, `r`
recenter), `--no-voice`, `--rules`, `--camera NAME|INDEX`. The first voice use
downloads the Whisper model (about 140 MB).

The camera defaults to the Mac's built-in one, chosen by device type rather
than index, so an iPhone joining as a Continuity Camera doesn't take over.

macOS needs Camera, Microphone and Accessibility permission for your terminal.

Tests (synthetic hands, nothing sent to macOS): `python tests/test_logic.py`
