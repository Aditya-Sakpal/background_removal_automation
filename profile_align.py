"""
Deterministic profile banner alignment (765×480).

Default: scale + translate the full Gemini image so the face sits on designer
guides (Y 74–303, centered at X 382.5) while keeping Gemini's gradient/background.

Optional cutout+template path exists only as fallback (not used by default).
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

# Max long edge before alignment (avoids OOM on Render for huge Gemini PNGs).
_ALIGN_MAX_EDGE = int(os.getenv("PROFILE_ALIGN_MAX_EDGE", "1920"))

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

# MediaPipe indices for top-of-head band (used for hairline Y = guide top).
_CRANIAL_TOP_INDICES = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288)
_CHIN_INDEX = 152
_LEFT_TEMPLE_INDEX = 234
_RIGHT_TEMPLE_INDEX = 454
# Nudge above mesh cranial top to include visible hair (fraction of face height).
_HAIR_ABOVE_MESH = 0.04

_face_landmarker: FaceLandmarker | None = None


@lru_cache(maxsize=1)
def get_profile_template() -> Image.Image:
    """Synthetic 765×480 background (fallback only — prefer preserving Gemini BG)."""
    w, h = PROFILE_WIDTH, PROFILE_HEIGHT
    cx, cy = w * 0.5, h * 0.32
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = float(np.sqrt(cx**2 + cy**2) * 1.05)
    t = np.clip(dist / r_max, 0.0, 1.0)

    center = np.array([55.0, 95.0, 165.0], dtype=np.float32)
    edge = np.array([8.0, 18.0, 42.0], dtype=np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = center[c] * (1.0 - t) + edge[c] * t

    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)
    bottom_fade = np.clip((y_norm - 0.42) / 0.58, 0.0, 1.0).reshape(h, 1, 1)
    img *= 1.0 - bottom_fade * 0.55

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")


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


def _rgb_for_mediapipe(image: Image.Image) -> np.ndarray:
    """Contiguous uint8 RGB array — avoids MediaPipe Image __del__ errors on some hosts."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(arr)


def _face_metrics_mediapipe(rgb: np.ndarray) -> tuple[float, float, float] | None:
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)

    mp_image = None
    result = None
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _get_face_landmarker().detect(mp_image)
    finally:
        if mp_image is not None:
            close_fn = getattr(mp_image, "close", None)
            if callable(close_fn):
                close_fn()

    if result is None or not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    chin_y = lm[_CHIN_INDEX].y * h
    cranial_top_y = min(lm[i].y for i in _CRANIAL_TOP_INDICES) * h
    face_len = max(chin_y - cranial_top_y, 1.0)
    hairline_y = cranial_top_y - _HAIR_ABOVE_MESH * face_len

    left_x = lm[_LEFT_TEMPLE_INDEX].x * w
    right_x = lm[_RIGHT_TEMPLE_INDEX].x * w
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


# Max allowed landmark drift after alignment (pixels on final 765×480).
GUIDE_TOLERANCE_PX = 3.0
MAX_ALIGN_PASSES = 5


def face_guide_errors(image: Image.Image) -> tuple[float, float, float]:
    """Pixel error vs guides: (hairline_y, chin_y, center_x)."""
    hairline_y, chin_y, center_x = detect_face_metrics(image)
    return (
        hairline_y - GUIDE_TOP_Y,
        chin_y - GUIDE_BOTTOM_Y,
        center_x - GUIDE_CENTER_X,
    )


def is_face_on_guides(image: Image.Image, tolerance: float = GUIDE_TOLERANCE_PX) -> bool:
    """True if detected face band and center are within tolerance of designer guides."""
    top_err, chin_err, cx_err = face_guide_errors(image)
    return (
        abs(top_err) <= tolerance
        and abs(chin_err) <= tolerance
        and abs(cx_err) <= tolerance
    )


def ensure_face_landmarker_model() -> None:
    """Download face_landmarker.task if missing (call once at app startup)."""
    _ensure_face_landmarker_model()
    _get_face_landmarker()


def _use_mediapipe_detection() -> bool:
    """
    MediaPipe is accurate but can crash on cleanup (Render logs).
    PROFILE_ALIGN_USE_MEDIAPIPE: 1|0|auto (default auto → off when RENDER=true).
    """
    mode = os.getenv("PROFILE_ALIGN_USE_MEDIAPIPE", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    return not bool(os.getenv("RENDER"))


def _downscale_if_needed(image: Image.Image, max_edge: int = _ALIGN_MAX_EDGE) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    scale = max_edge / max(w, h)
    return image.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.Resampling.LANCZOS,
    )


def detect_face_metrics(image: Image.Image) -> tuple[float, float, float]:
    rgb = _rgb_for_mediapipe(image)
    metrics = None
    if _use_mediapipe_detection():
        try:
            metrics = _face_metrics_mediapipe(rgb)
        except Exception as exc:
            print(f"[profile_align] MediaPipe detect failed, using OpenCV: {exc}")
    if metrics is None:
        metrics = _face_metrics_haar_fallback(rgb)
    if metrics is None:
        w, h = image.size
        return (h * 0.15, h * 0.55, w / 2.0)
    return metrics


def _sample_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Median color from upper band (headroom area) for letterbox padding."""
    w, h = image.size
    band_h = max(8, min(int(h * 0.22), h))
    patch = np.array(image.crop((0, 0, w, band_h)).convert("RGB"))
    median = np.median(patch.reshape(-1, 3), axis=0)
    return tuple(int(v) for v in median)


def align_geometric_preserve_background(
    source: Image.Image,
    metrics: tuple[float, float, float],
) -> Image.Image:
    """
    Exact uniform scale + translate (affine) so that:
      hairline_y → GUIDE_TOP_Y (74)
      chin_y     → GUIDE_BOTTOM_Y (303)
      center_x   → GUIDE_CENTER_X (382.5)

    Scale s = 229 / (chin_y - hairline_y). Image is scaled up or down as needed.
    """
    hairline_y, chin_y, center_x = metrics
    face_h = max(chin_y - hairline_y, 1.0)
    scale = FACE_BAND_HEIGHT / face_h

    tx = GUIDE_CENTER_X - scale * center_x
    ty = GUIDE_TOP_Y - scale * hairline_y

    rgb = np.asarray(source.convert("RGB"))
    bg = _sample_background_color(source)
    border_bgr = (int(bg[2]), int(bg[1]), int(bg[0]))

    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    warped = cv2.warpAffine(
        bgr,
        matrix,
        (PROFILE_WIDTH, PROFILE_HEIGHT),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_bgr,
    )
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


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
    """Fallback: rembg cutout on synthetic template (discouraged for final output)."""
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
    Align face to guides on a 765×480 canvas.
    Preserves the full Gemini render (background + bottom fade); no rembg by default.
    Repeats until landmarks sit on guides (≤3 px) or MAX_ALIGN_PASSES.
    """
    current = _downscale_if_needed(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    result = current
    for pass_idx in range(MAX_ALIGN_PASSES):
        metrics = detect_face_metrics(current)
        result = align_geometric_preserve_background(current, metrics)
        if is_face_on_guides(result):
            break
        current = result
        if pass_idx == MAX_ALIGN_PASSES - 1:
            top_e, chin_e, cx_e = face_guide_errors(result)
            print(
                f"[profile_align] warning after {MAX_ALIGN_PASSES} passes: "
                f"hairline {top_e:+.1f}px, chin {chin_e:+.1f}px, center {cx_e:+.1f}px"
            )
    return result
