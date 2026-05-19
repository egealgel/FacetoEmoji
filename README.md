# FacetoEmoji

Real-time webcam-based facial expression to emoji renderer.
The left half of the window shows the camera feed; the right half displays
a large, crisp emoji that matches your current expression or gesture.

![demo placeholder — record one with QuickTime and drop it here]()

## Features

- **7 base emotions** via [DeepFace](https://github.com/serengil/deepface):
  happy 😀, sad 😢, angry 😠, surprise 😲, fear 😨, disgust 🤢, neutral 😐.
- **Gestures** via [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe)
  blendshapes:
  - wink 😉 (`eyeBlinkLeft` vs `eyeBlinkRight` asymmetry)
  - kiss 😘 (`mouthPucker`)
  - tongue out 😛 (`tongueOut`)
- **Anger override** from `browDownLeft/Right` and `noseSneerLeft/Right`,
  because DeepFace's FER model under-predicts anger.
- **Crisp emojis**: Google Noto Emoji 512 px PNGs, cached to disk on first use.
- **Smooth FPS**: DeepFace runs in a background thread with a 1-slot ring
  buffer, MediaPipe runs every Nth frame, rendered emoji panels are cached
  per `(emoji, label)`.

## Requirements

- macOS (tested on Apple Silicon) or Linux. Windows untested.
- **Python 3.12** (TensorFlow has no wheels for 3.13+ yet).
- A webcam.

## Setup

```bash
git clone <your repo url>
cd FacetoEmoji
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS install Python 3.12 via Homebrew if needed:
`brew install python@3.12`.

## Run

```bash
python face_to_emoji.py
```

Press `q` to quit. On first launch:
- DeepFace downloads its emotion model (~5 MB) into `~/.deepface/weights/`.
- MediaPipe downloads `face_landmarker.task` (~3 MB) into the project folder.
- Each emoji is fetched from Noto on first display and cached in
  `emoji_cache/`. After that the app works fully offline.

macOS will prompt for camera permission the first time; allow your
terminal / IDE in System Settings → Privacy & Security → Camera and run again.

## Tuning

All thresholds live as module-level constants in
[`face_to_emoji.py`](face_to_emoji.py):

| Knob | Default | Effect |
| --- | --- | --- |
| `LANDMARK_EVERY_N` | `3` | Lower = snappier gestures, higher = better FPS. |
| `EmotionWorker.EMA_ALPHA` | `0.55` | Higher = faster reaction to emotion change. |
| `EmotionWorker.NEUTRAL_PENALTY` | `0.55` | Lower = neutral suppressed more, other emotions easier. |
| `EmotionWorker.SWITCH_MARGIN` | `8.0` | Hysteresis between dominant emotions to avoid flicker. |
| `detect_gesture` thresholds | inline | Wink / kiss / tongue sensitivity. |
| `detect_emotion_override` thresholds | inline | Anger sensitivity from brow furrow + sneer. |

The bottom of the camera feed prints live blendshape scores
(`blinkL blinkR pucker tongue brow`) plus the top-3 emotion scores — handy
for tuning the thresholds to your own face.

## How it works

```
                ┌─────────────────┐
   webcam ───▶ │  cv2.VideoCapture │
                └────────┬────────┘
                         │ frame
                         ├──────────────▶ EmotionWorker (background thread)
                         │                │
                         │                ▼  DeepFace.analyze
                         │             EMA scores ─▶ argmax with neutral penalty
                         │                │
                         ▼                ▼
            MediaPipe FaceLandmarker   worker.result
            ├─ blendshapes ─▶ detect_gesture()  → wink / kiss / tongue
            └─ blendshapes ─▶ detect_emotion_override() → angry
                         │
                         ▼
            emoji  +  label  ──▶  panel cache  ──▶  _render_emoji_panel
                                                       │
                                                       ▼
                                              hstack(camera, panel) → imshow
```

Gestures override emotions; anger override beats DeepFace; otherwise the
DeepFace EMA pick wins.

## License

MIT.
