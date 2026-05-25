"""
Deterministic profile banner alignment (765×480).

Pipeline: background removal → MediaPipe face metrics → scale/translate cutout
→ composite on fixed gradient template using designer guide Y positions.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PIL import Image

PROFILE_WIDTH = 765
PROFILE_HEIGHT = 480
GUIDE_TOP_Y = 74
GUIDE_BOTTOM_Y = 303
GUIDE_CENTER_X = PROFILE_WIDTH / 2.0
FACE_BAND_HEIGHT = GUIDE_BOTTOM_Y - GUIDE_TOP_Y

_DIR = os.path.dirname(os.path.abspath(__file__))
BLACK_GRADIENT_REFERENCE_PATH = os.path.join(_DIR, "black_gradient_reference.png")
FACE_LANDMARKER_MODEL_PATH = os.path.join(_DIR, "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Extend above forehead landmark to approximate hairline (fraction of forehead→chin).
HAIRLINE_EXTENSION = 0.22

_face_landmarker: FaceLandmarker | None = None


@lru_cache(maxsize=1)
def get_profile_template() -> Image.Image:
    """Build 765×480 background: radial blue vignette + black bottom gradation overlay."""
    w, h = PROFILE_WIDTH, PROFILE_HEIGHT
    cx, cy = w * 0.5, h * 0.32
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = float(np.sqrt(cx**2 + cy**2) * 1.05)
    t = np.clip(dist / r_max, 0.0, 1.0)

    # RGB — bright blue behind head, darker corners (engage4more-style).
    center = np.array([55.0, 95.0, 165.0], dtype=np.float32)
    edge = np.array([8.0, 18.0, 42.0], dtype=np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = center[c] * (1.0 - t) + edge[c] * t

    # Vertical darken toward bottom (before black gradation asset).
    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)
    bottom_fade = np.clip((y_norm - 0.42) / 0.58, 0.0, 1.0).reshape(h, 1, 1)
    img *= 1.0 - bottom_fade * 0.55

    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

    if os.path.isfile(BLACK_GRADIENT_REFERENCE_PATH):
        overlay = Image.open(BLACK_GRADIENT_REFERENCE_PATH).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB")


def _ensure_face_landmarker_model() -> None:
    if os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
        return
    import urllib.request

    print(f"[profile_align] Downloading face landmarker model to {FACE_LANDMARKER_MODEL_PATH} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)


def _get_face_landmarker() -> FaceLandmarker:
    global _face_landmarker
    if _face_landmarker is None:
        _ensure_face_landmarker_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


def _face_metrics_mediapipe(rgb: np.ndarray) -> tuple[float, float, float] | None:
    """Return (hairline_y, chin_y, center_x) in pixel coordinates."""
    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_face_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    chin_y = lm[152].y * h
    forehead_y = lm[10].y * h
    face_len = max(chin_y - forehead_y, 1.0)
    hairline_y = forehead_y - HAIRLINE_EXTENSION * face_len

    left_x = lm[234].x * w
    right_x = lm[454].x * w
    center_x = (left_x + right_x) / 2.0

    return (hairline_y, chin_y, center_x)


def _face_metrics_haar_fallback(rgb: np.ndarray) -> tuple[float, float, float] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if faces is None or len(faces) == 0:
        return None
    areas = [int(fw) * int(fh) for (_x, _y, fw, fh) in faces]
    x, y, fw, fh = [int(v) for v in faces[int(np.argmax(areas))]]
    hairline_y = y - 0.35 * fh
    chin_y = y + fh
    center_x = x + fw / 2.0
    return (hairline_y, chin_y, center_x)


def detect_face_metrics(image: Image.Image) -> tuple[float, float, float]:
    rgb = np.asarray(image.convert("RGB"))
    metrics = _face_metrics_mediapipe(rgb)
    if metrics is None:
        metrics = _face_metrics_haar_fallback(rgb)
    if metrics is None:
        w, h = image.size
        return (h * 0.15, h * 0.55, w / 2.0)
    return metrics


def remove_background_rgba(image: Image.Image) -> Image.Image:
    from rembg import remove

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    out_bytes = remove(buf.getvalue())
    return Image.open(io.BytesIO(out_bytes)).convert("RGBA")


def align_cutout_on_template(
    cutout_rgba: Image.Image,
    metrics: tuple[float, float, float],
    *,
    template: Image.Image | None = None,
) -> Image.Image:
    hairline_y, chin_y, center_x = metrics
    face_h = max(chin_y - hairline_y, 1.0)
    scale = FACE_BAND_HEIGHT / face_h

    new_w = max(1, int(round(cutout_rgba.width * scale)))
    new_h = max(1, int(round(cutout_rgba.height * scale)))
    scaled = cutout_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    paste_x = int(round(GUIDE_CENTER_X - center_x * scale))
    paste_y = int(round(GUIDE_TOP_Y - hairline_y * scale))

    if template is None:
        template = get_profile_template()
    canvas = template.convert("RGBA")
    canvas.paste(scaled, (paste_x, paste_y), scaled)
    return canvas.convert("RGB")


def align_profile_to_grid(image_bytes: bytes) -> Image.Image:
    """
    Full pipeline: load image → detect face → cut out subject → align on template.
    Returns 765×480 RGB PIL Image.
    """
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    metrics = detect_face_metrics(source)
    cutout = remove_background_rgba(source)
    return align_cutout_on_template(cutout, metrics)
