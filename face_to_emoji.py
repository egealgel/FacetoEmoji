#!/usr/bin/env python3
"""Kameradan yüzü tarayıp ifadeye göre ekrana emoji basar.

Çalıştır:
    python face_to_emoji.py
"""
import os
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path
from threading import Lock, Thread

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

EMOTION_EMOJI = {
    "happy":    "😀",
    "sad":      "😢",
    "angry":    "😠",
    "surprise": "😲",
    "fear":     "😨",
    "disgust":  "🤢",
    "neutral":  "😐",
}

GESTURE_EMOJI = {
    "wink":   "😉",
    "kiss":   "😘",
    "tongue": "😛",
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

EMOJI_CACHE_DIR = Path(__file__).resolve().parent / "emoji_cache"
EMOJI_CACHE_DIR.mkdir(exist_ok=True)
NOTO_URL = ("https://raw.githubusercontent.com/googlefonts/noto-emoji/main/"
            "png/512/emoji_u{cp}.png")
TWEMOJI_URL = ("https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/"
               "assets/72x72/{cp}.png")


def _codepoint(emoji_char: str, sep: str) -> str:
    return sep.join(f"{ord(c):x}" for c in emoji_char if c != "️")


def _load_emoji_png(emoji_char: str):
    cp_u = _codepoint(emoji_char, "_")
    cache = EMOJI_CACHE_DIR / f"{cp_u}.png"
    if not cache.exists():
        for url in (NOTO_URL.format(cp=cp_u),
                    TWEMOJI_URL.format(cp=_codepoint(emoji_char, "-"))):
            try:
                urllib.request.urlretrieve(url, cache)
                if cache.stat().st_size > 0:
                    break
            except Exception:
                if cache.exists():
                    cache.unlink()
        else:
            return None
    return cv2.imread(str(cache), cv2.IMREAD_UNCHANGED)


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"MediaPipe yüz modeli indiriliyor → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker() -> mp_vision.FaceLandmarker:
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
        output_face_blendshapes=True,
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_gesture(blendshapes):
    """MediaPipe blendshape skorlarından (0..1) hareket çıkarımı."""
    scores = {bs.category_name: bs.score for bs in blendshapes}
    blink_l = scores.get("eyeBlinkLeft", 0.0)
    blink_r = scores.get("eyeBlinkRight", 0.0)
    pucker  = scores.get("mouthPucker", 0.0)
    tongue  = scores.get("tongueOut", 0.0)

    if tongue > 0.25:
        return "tongue", scores
    if abs(blink_l - blink_r) > 0.30 and max(blink_l, blink_r) > 0.45:
        return "wink", scores
    if pucker > 0.40:
        return "kiss", scores
    return None, scores


def detect_emotion_override(scores: dict[str, float]) -> str | None:
    """DeepFace'in zayıf olduğu duyguları blendshape'lerden yakala."""
    brow_down = (scores.get("browDownLeft", 0.0)
                 + scores.get("browDownRight", 0.0)) / 2
    sneer = (scores.get("noseSneerLeft", 0.0)
             + scores.get("noseSneerRight", 0.0)) / 2
    if brow_down > 0.40 or sneer > 0.35:
        return "angry"
    return None


class EmotionWorker:
    """DeepFace.analyze yavaş; kamera takılmasın diye arka planda çalıştırır.

    Çoğunluk oylaması yerine ham skorlar üzerinde EMA tutuyoruz:
    daha hızlı tepki + neutral'a karşı hafif ceza ile diğer duygulara şans.
    """

    EMA_ALPHA = 0.55          # büyük = daha duyarlı
    NEUTRAL_PENALTY = 0.55    # neutral skorunu küçültür
    SWITCH_MARGIN = 8.0       # yeni duygunun mevcuda göre öne geçmesi gereken fark
    SCORES = (
        "angry", "disgust", "fear", "happy",
        "sad", "surprise", "neutral",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._frame = None
        self._running = True
        self.result = "neutral"
        self.scores: dict[str, float] = {k: 0.0 for k in self.SCORES}
        Thread(target=self._loop, daemon=True).start()

    def submit(self, frame) -> None:
        # Ring buffer of 1 — worker meşgulse eski kareler düşer.
        with self._lock:
            self._frame = frame.copy()

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                r = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(r, list):
                    r = r[0]
                raw = r.get("emotion", {})
                if not raw:
                    continue
                a = self.EMA_ALPHA
                for k in self.SCORES:
                    v = float(raw.get(k, 0.0))
                    self.scores[k] = a * v + (1 - a) * self.scores[k]
                biased = dict(self.scores)
                biased["neutral"] *= self.NEUTRAL_PENALTY
                top = max(biased, key=biased.get)
                # Histerezis: yeni galip yeterince öne geçmeden değiştirme.
                if top != self.result and (
                    biased[top] - biased.get(self.result, 0.0) >= self.SWITCH_MARGIN
                ):
                    self.result = top
                elif self.result not in biased or biased[self.result] < 5.0:
                    # Mevcut etiket çok düşükse yine de güncelle.
                    self.result = top
            except Exception as e:
                print("DeepFace hata:", e)

    def stop(self) -> None:
        self._running = False


PANEL_W, PANEL_H = 640, 480
LANDMARK_EVERY_N = 3              # MediaPipe her N karede bir
BG_COLOR_BGR     = np.array([28, 24, 24], dtype=np.uint8)


def _render_emoji_panel(emoji_char: str, label: str) -> np.ndarray:
    """Noto Emoji'nin 512px PNG'sini indirip alpha-blend ile ortalıyoruz."""
    panel = np.full((PANEL_H, PANEL_W, 3), BG_COLOR_BGR, dtype=np.uint8)
    img = _load_emoji_png(emoji_char)
    if img is not None:
        target = int(min(PANEL_W, PANEL_H) * 0.85)
        h0, w0 = img.shape[:2]
        scale = target / max(h0, w0)
        new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        img = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)

        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[..., :3].astype(np.float32)
            alpha = img[..., 3:4].astype(np.float32) / 255.0
        else:
            bgr = img.astype(np.float32)
            alpha = np.ones((new_h, new_w, 1), dtype=np.float32)

        px = (PANEL_W - new_w) // 2
        py = (PANEL_H - new_h) // 2 - 20
        roi = panel[py:py + new_h, px:px + new_w].astype(np.float32)
        panel[py:py + new_h, px:px + new_w] = (
            bgr * alpha + roi * (1 - alpha)
        ).astype(np.uint8)

    cv2.putText(panel, label, (20, PANEL_H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Kamera açılamadı.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PANEL_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    landmarker = make_landmarker()
    worker = EmotionWorker()

    frame_idx = 0
    current_gesture = None
    last_scores: dict[str, float] = {}
    panel_cache: dict[tuple[str, str], np.ndarray] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != PANEL_W or frame.shape[0] != PANEL_H:
                frame = cv2.resize(frame, (PANEL_W, PANEL_H))

            worker.submit(frame)

            # MediaPipe pahalı; her N. karede çalıştır, ara karelerde son sonucu kullan.
            if frame_idx % LANDMARK_EVERY_N == 0:
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect(mp_img)
                if result.face_blendshapes:
                    current_gesture, last_scores = detect_gesture(
                        result.face_blendshapes[0]
                    )
                else:
                    current_gesture, last_scores = None, {}
            frame_idx += 1

            if current_gesture:
                label = current_gesture.upper()
                emoji = GESTURE_EMOJI[current_gesture]
            else:
                override = detect_emotion_override(last_scores)
                emo = override or worker.result
                label = emo.upper()
                emoji = EMOTION_EMOJI.get(emo, "😐")

            key = (emoji, label)
            panel = panel_cache.get(key)
            if panel is None:
                panel = _render_emoji_panel(emoji, label)
                panel_cache[key] = panel

            # Sol panele canlı tanı satırları — neyin tetiklendiğini görmek için.
            top3 = sorted(worker.scores.items(), key=lambda kv: -kv[1])[:3]
            top_str = "  ".join(f"{k[:3]}:{v:.0f}" for k, v in top3)
            cv2.putText(frame, f"emo: {worker.result}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, top_str, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
            if last_scores:
                brow = (last_scores.get('browDownLeft', 0)
                        + last_scores.get('browDownRight', 0)) / 2
                dbg = (
                    f"blinkL {last_scores.get('eyeBlinkLeft',0):.2f}  "
                    f"blinkR {last_scores.get('eyeBlinkRight',0):.2f}  "
                    f"pucker {last_scores.get('mouthPucker',0):.2f}  "
                    f"tongue {last_scores.get('tongueOut',0):.2f}  "
                    f"brow {brow:.2f}"
                )
                cv2.putText(frame, dbg, (10, PANEL_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 255, 200), 1, cv2.LINE_AA)

            combined = np.hstack([frame, panel])
            cv2.imshow("FacetoEmoji  (q ile çık)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
