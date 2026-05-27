import base64
import gc
import io
import json
import os
import re
import zipfile

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image, ImageDraw

from google_images import (
    DEFAULT_TARGET_ASPECT,
    REFERENCE_IMAGE_POOL_SIZE,
    download_image_bytes,
    fetch_google_image_candidates,
)
from serpapi_images import fetch_serpapi_image_candidates

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash-image"  # Nano Banana — image generation
GEMINI_TEXT_MODEL = "gemini-2.5-flash"  # vision-capable text model for JSON checks
MIN_IMAGES = 2
MAX_IMAGES = 3
MAX_RETRY = 3
STYLE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "1768222411704.jpg")
HEAD_SPACE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "head_space_reference.png")
GRADIENT_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "gradient_reference.jpg")
COMPOSITION_GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "netanyahu's image.jpg")
VIGNETTE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "zakir-khan_Comedian.png")
BLACK_GRADIENT_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "black_gradient_reference.png")
PROFILE_WIDTH = 765
PROFILE_HEIGHT = 480
GUIDE_TOP_Y = 74  # hairline / top of head (designer grid)
GUIDE_BOTTOM_Y = 303  # chin (designer grid)
GUIDE_CENTER_X = PROFILE_WIDTH / 2.0
FACE_BAND_HEIGHT = GUIDE_BOTTOM_Y - GUIDE_TOP_Y
# Testing overlay: cyan Y=74, Y=303, center X=382.5 (PROFILE_SHOW_GUIDE_LINES=0 to hide)
PROFILE_SHOW_GUIDE_LINES = os.getenv("PROFILE_SHOW_GUIDE_LINES", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# After face alignment, Gemini pass to remove warp padding / double-background seams.
PROFILE_UNIFORM_BACKGROUND = os.getenv("PROFILE_UNIFORM_BACKGROUND", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
PROFILE_MAX_SIZE_KB = 50
GALLERY_WIDTH = 1176
GALLERY_HEIGHT = 516
GALLERY_MAX_SIZE_KB = 50
# Google image pool → OpenAI picks best gallery-style references
REFERENCE_TOP_K_OPENAI = 10
EXTRA_GALLERY_BATCH_SIZE = 10  # batch added when user clicks "Fetch 10 more"
NANO_BANANA_REGEN_MODEL = "gemini-2.5-flash-image"


# ---------------------------------------------------------------------------
# Clients (cached so they aren't recreated every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_image_for_openai(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def parse_json_response(raw: str | None) -> dict:
    """
    Strip markdown fences if present, extract the outer JSON object, parse it.
    Raises ValueError on failure with the raw content included for debugging.
    """
    if raw is None:
        raise ValueError("Empty response from the model (content was None).")

    raw = raw.strip()
    if not raw:
        raise ValueError("Empty response from the model (content was blank).")

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    # First try a direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fall back: extract the first {...} block and try again
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response. Raw content: {raw[:500]}")


def mime_from_name(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")


def slugify_artist_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return cleaned or "artist_name"


def is_valid_image_bytes(image_bytes: bytes) -> tuple[bool, str | None]:
    """Return whether bytes decode as an actual image."""
    if not image_bytes:
        return False, "Empty payload"
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
        return True, None
    except Exception as e:
        return False, f"Invalid image bytes ({e})"


class GeminiTransientError(RuntimeError):
    """Transient Gemini generation failure that's safe to retry (e.g. IMAGE_OTHER)."""


# Finish reasons that are flaky / non-deterministic and worth retrying.
# Compared as strings so we don't rely on the SDK enum import.
TRANSIENT_FINISH_REASONS = {"IMAGE_OTHER", "OTHER", "MAX_TOKENS"}


def extract_image_from_gemini_response(response, context: str = "image") -> bytes:
    """
    Pull the inline image bytes from a Gemini generate_content response.
    Raises GeminiTransientError for retryable failures, RuntimeError for permanent ones.
    """
    if response is None:
        raise RuntimeError(f"Gemini returned no response while generating {context}.")

    # Top-level prompt feedback (when the prompt itself is rejected).
    pf = getattr(response, "prompt_feedback", None)
    if pf is not None and getattr(pf, "block_reason", None):
        raise RuntimeError(
            f"Gemini blocked the prompt while generating {context}. "
            f"Reason: {pf.block_reason}. Detail: {getattr(pf, 'block_reason_message', '')}"
        )

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError(
            f"Gemini returned no candidates for {context}. "
            f"This usually means the prompt was blocked by safety filters."
        )

    cand = candidates[0]
    finish_reason = getattr(cand, "finish_reason", None)
    finish_str = str(finish_reason).split(".")[-1] if finish_reason is not None else ""
    content = getattr(cand, "content", None)

    if content is None or getattr(content, "parts", None) is None:
        safety_ratings = getattr(cand, "safety_ratings", None)
        msg = (
            f"Gemini returned an empty response for {context} "
            f"(finish_reason={finish_reason}). Safety ratings: {safety_ratings}"
        )
        if finish_str in TRANSIENT_FINISH_REASONS:
            raise GeminiTransientError(msg)
        raise RuntimeError(msg + " (likely safety filter on real-person inputs)")

    text_seen = []
    for part in content.parts:
        if getattr(part, "inline_data", None) is not None:
            return part.inline_data.data
        text = getattr(part, "text", None)
        if text:
            text_seen.append(text)

    text_blob = " | ".join(text_seen)[:300] if text_seen else "(no text)"
    msg = (
        f"Gemini did not return an image for {context} "
        f"(finish_reason={finish_reason}). Model said: {text_blob}"
    )
    if finish_str in TRANSIENT_FINISH_REASONS:
        raise GeminiTransientError(msg)
    raise RuntimeError(msg)


def call_gemini_with_retry(
    client,
    *,
    model: str,
    contents: list,
    config,
    context: str,
    max_attempts: int = 3,
) -> bytes:
    
    """
    Call generate_content and parse the image, retrying on transient errors
    (IMAGE_OTHER, OTHER, MAX_TOKENS). Hard failures (safety, prompt block)
    are raised immediately.
    """
    last_err: Exception | None = None
    last_response = None  # noqa: F841 — kept in scope so it's visible at the breakpoint below
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            last_response = response  # noqa: F841 — inspect this in the debugger
            return extract_image_from_gemini_response(response, context=context)
        except GeminiTransientError as e:
            last_err = e
            last_response = response  # noqa: F841 — inspect this in the debugger
            # ────────────────────────────────────────────────────────────
            # 🔴 BREAKPOINT HERE — Gemini returned a transient failure.
            #
            # Inspect in the Variables panel:
            #   response             → full SDK response object
            #   response.candidates  → list; check candidates[0].finish_reason
            #   response.prompt_feedback → block_reason / block_reason_message
            #   response.candidates[0].safety_ratings  → per-category scores
            #   response.candidates[0].content         → None means refused
            #   contents             → exact inputs (images + prompt) sent
            #   context              → which step failed ("profile image", etc.)
            #   e                    → parsed error message
            # ────────────────────────────────────────────────────────────
            print(f"[DEBUG] Gemini transient on attempt {attempt}: {e}")
            if response is not None:
                print(f"[DEBUG] finish_reason: {getattr(response.candidates[0], 'finish_reason', None)}")
                print(f"[DEBUG] prompt_feedback: {getattr(response, 'prompt_feedback', None)}")
                print(f"[DEBUG] safety_ratings: {getattr(response.candidates[0], 'safety_ratings', None)}")
            if attempt < max_attempts:
                import time

                time.sleep(0.5 * attempt)
                continue
            break
        except Exception as e:
            # Catch-all for non-transient failures so we can still inspect the response.
            last_err = e
            last_response = response  # noqa: F841 — inspect this in the debugger
            # ────────────────────────────────────────────────────────────
            # 🔴 BREAKPOINT HERE — Gemini returned a HARD failure (safety
            # block, prompt rejection, SDK error, etc.). Same inspection
            # variables as above.
            # ────────────────────────────────────────────────────────────
            print(f"[DEBUG] Gemini hard failure on attempt {attempt}: {type(e).__name__}: {e}")
            if response is not None:
                print(f"[DEBUG] finish_reason: {getattr(response.candidates[0], 'finish_reason', None)}")
                print(f"[DEBUG] prompt_feedback: {getattr(response, 'prompt_feedback', None)}")
            raise
    raise RuntimeError(
        f"Gemini failed {max_attempts} times for {context} (transient). Last error: {last_err}"
    )


def _face_center_from_image_rgb(img: Image.Image) -> tuple[float, float] | None:
    """Return (cx, cy) in pixel coords of largest frontal face, or None."""
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if faces is None or len(faces) == 0:
        return None
    areas = [int(fw) * int(fh) for (_x, _y, fw, fh) in faces]
    i = int(np.argmax(areas))
    x, y, fw, fh = [int(v) for v in faces[i]]
    return (x + fw / 2.0, y + fh / 2.0)


def crop_gallery_banner_face_centered_code_only(image_bytes: bytes) -> Image.Image:
    """
    Deterministic crop + resize only: no generative model. Fits aspect GALLERY_WIDTH:GALLERY_HEIGHT,
    uses largest detected frontal face to center the crop (fallback: image center).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w < 2 or h < 2:
        raise ValueError("Image too small to crop.")

    r_out = GALLERY_WIDTH / GALLERY_HEIGHT
    fc = _face_center_from_image_rgb(img)
    if fc is not None:
        fx, fy = fc
    else:
        fx, fy = w / 2.0, h / 2.0

    # Maximal axis-aligned crop inside (w,h) with aspect w_crop/h_crop = r_out
    if (w / h) >= r_out:
        crop_h = h
        crop_w = int(round(crop_h * r_out))
        if crop_w > w:
            crop_w = w
            crop_h = int(round(crop_w / r_out))
    else:
        crop_w = w
        crop_h = int(round(crop_w / r_out))
        if crop_h > h:
            crop_h = h
            crop_w = int(round(crop_h * r_out))

    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))

    left = int(round(fx - crop_w / 2.0))
    top = int(round(fy - crop_h / 2.0))
    left = max(0, min(left, w - crop_w))
    top = max(0, min(top, h - crop_h))

    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((GALLERY_WIDTH, GALLERY_HEIGHT), Image.Resampling.LANCZOS)


def prepare_gallery_webp_exact_rgb(img: Image.Image) -> tuple[bytes, bool]:
    """Encode already GALLERY-sized RGB image to WebP under size cap (no geometric crop)."""
    if img.size != (GALLERY_WIDTH, GALLERY_HEIGHT):
        img = img.resize((GALLERY_WIDTH, GALLERY_HEIGHT), Image.Resampling.LANCZOS)
    size_limit = GALLERY_MAX_SIZE_KB * 1024
    last_bytes = None
    for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality, method=6, optimize=True)
        candidate = buf.getvalue()
        last_bytes = candidate
        if len(candidate) <= size_limit:
            return candidate, True
    return last_bytes or b"", False


def _prepare_thumb_for_vision(image_bytes: bytes, max_edge: int = 384) -> tuple[bytes, str]:
    """Downscale for OpenAI vision to reduce payload size."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(max_edge / max(w, 1), max_edge / max(h, 1), 1.0)
    if scale < 1.0:
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _prepare_thumb_for_ranking(image_bytes: bytes, max_edge: int = 768) -> tuple[bytes, str]:
    """Larger thumbs for gallery-type scoring (event vs studio, face clarity)."""
    return _prepare_thumb_for_vision(image_bytes, max_edge=max_edge)


def _downscale_for_pool(image_bytes: bytes, max_edge: int = 1500) -> bytes:
    """
    Shrink a freshly-downloaded source image so the pool doesn't hold huge originals
    in memory. The gallery banner output is only 1176x516, so a 1500px long edge is
    plenty of detail with a fraction of the memory cost.
    Re-encoded as JPEG q=88 to keep the in-memory representation compact.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_edge:
                scale = max_edge / max(w, h)
                im = im.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88, optimize=True)
            return buf.getvalue()
    except Exception:
        # If decoding fails, leave the bytes untouched — caller already validates.
        return image_bytes


def _fallback_top_indices_by_aspect(rows: list[dict], target_aspect: float, k: int) -> list[int]:
    scored: list[tuple[float, int]] = []
    for i, r in enumerate(rows):
        if not r.get("bytes"):
            continue
        ar = r.get("aspect_ratio")
        if ar is None:
            scored.append((999.0, i))
        else:
            scored.append((abs(ar - target_aspect), i))
    scored.sort(key=lambda x: x[0])
    return [i for _, i in scored[:k]]


def rank_reference_images_openai(
    rows: list[dict],
    celebrity_name: str,
    target_aspect: float,
    top_k: int = REFERENCE_TOP_K_OPENAI,
) -> tuple[list[int], str | None]:
    """
    Ask GPT-4o (vision) to pick best `top_k` indices into `rows` (0-based).
    Returns (selected_indices, error_or_none).
    """
    with_bytes = [(i, r) for i, r in enumerate(rows) if r.get("bytes")]
    if not with_bytes:
        return [], "No images downloaded to rank."
    top_k = min(top_k, len(with_bytes))

    client = get_openai_client()
    target_ratio_str = f"{target_aspect:.3f} (width ÷ height, e.g. banner ~{GALLERY_WIDTH}×{GALLERY_HEIGHT})"

    image_content: list[dict] = []
    for i, r in with_bytes:
        image_content.append({"type": "text", "text": f"--- Image index {i} (shown below) ---"})
        tb, mime = _prepare_thumb_for_ranking(r["bytes"])
        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_for_openai(tb, mime), "detail": "high"},
            }
        )

    valid_idx_str = ", ".join(str(i) for i, _ in with_bytes)

    prompt = f"""You are selecting **live event / action gallery** reference photos for **{celebrity_name}** (artist on a booking-style site).

You see {len(with_bytes)} images in order. Each block is labeled **Image index N**. Valid indices only: {valid_idx_str}.

## Four target types (a strong pick should clearly match **at least one**)
**TYPE 1 — Speaking / performing on stage**  
Mic, podium, stage or event lighting, clearly presenting or performing (not a plain studio backdrop).

**TYPE 2 — Audience interaction**  
Visible crowd / audience, handshake, pointing to people, selfie-with-crowd energy, walkabout — clear **event + people** context.

**TYPE 3 — Candid natural moment**  
Feels like a real unposed slice of an event — natural smile, talking, walking, backstage vibe. **Deprioritize** static glamour **studio** portraits (flat backdrop, catalog beauty pose, no event context).

**TYPE 4 — Expressive action moment**  
Strong emotion or gesture: mid-speech, laugh, raised hand, performing energy — **not** stiff posed headshot.

## Hard checks (must pass for a top pick unless no better option exists)
- **Identity**: Main subject should plausibly be **{celebrity_name}** (reject obvious wrong person).
- **Exactly one person (strict)**: There must be **only one** human subject treated as the star of the photo. Reject: duets or two+ people sharing the frame with **similar prominence** (side-by-side presenters, couple shots, two faces large in frame), collages / split panels / before-after layouts, or any image where a **second person's face** is clearly visible at roughly **≥ ~40%** the size of the main subject's face. **Acceptable**: one clear subject with a **distant, out-of-focus crowd** or tiny background figures where no second individual reads as a co-subject.
- **One clear focal performer**: Not group collages or multi-panel images; not "cast photo" style with multiple faces along a line at similar scale.
- **Face**: Face **clearly visible** and sharp enough (not tiny silhouette, not heavy occlusion).
- **Clarity**: Not extremely blurry, dark, or low-res.
- **Text**: Reject heavy overlaid text, meme text, news banners, big watermarks; tiny corner logos OK.

## Banner shape (secondary)
Prefer width ÷ height near **{target_ratio_str}** for a wide gallery strip; avoid extreme vertical crops if you have alternatives.

## Your task
Pick exactly **{top_k}** distinct indices that **best** match the **event/action** brief (Types 1–4) while passing **all** hard checks — especially **exactly-one-person**. If fewer than {top_k} images qualify, still return **{top_k}** indices by filling with the **least-bad** remaining valid indices (never invent indices).

Prioritise **single-subject** compliance over minor aspect-ratio mismatch.

Respond ONLY with valid JSON (no markdown fences):
{{
  "selected_indices": [<exactly {top_k} integers from valid indices>],
  "note": "<optional one short sentence>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a strict event-photography editor. Enforce exactly-one-clear-subject per pick. Output only valid JSON as requested.",
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=900,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = parse_json_response(raw)
    except json.JSONDecodeError as e:
        fb = _fallback_top_indices_by_aspect(rows, target_aspect, top_k)
        return fb, f"OpenAI JSON parse failed ({e}); used aspect fallback."

    chosen = data.get("selected_indices") or data.get("chosen") or []
    if not isinstance(chosen, list):
        fb = _fallback_top_indices_by_aspect(rows, target_aspect, top_k)
        return fb, "OpenAI returned invalid format; used aspect fallback."

    out: list[int] = []
    seen: set[int] = set()
    valid_set = {i for i, _ in with_bytes}
    for x in chosen:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi not in valid_set or xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
        if len(out) >= top_k:
            break

    if len(out) < top_k:
        for i in _fallback_top_indices_by_aspect(rows, target_aspect, top_k * 2):
            if i not in seen and i in valid_set:
                seen.add(i)
                out.append(i)
            if len(out) >= top_k:
                break

    return out[:top_k], None


def resize_and_crop_to_fill(
    img: Image.Image,
    target_width: int,
    target_height: int,
    horizontal_focus: float = 0.5,
    vertical_focus: float = 0.5,
) -> Image.Image:
    """
    Resize while preserving aspect ratio, then crop to exact target size.
    `horizontal_focus` / `vertical_focus` are in [0, 1] and control where the crop
    window is anchored (0=start/top, 0.5=center, 1=end/bottom).
    """
    src_w, src_h = img.size
    target_ratio = target_width / target_height
    src_ratio = src_w / src_h if src_h else target_ratio

    if src_ratio > target_ratio:
        # Source is wider than target; fit height first, then crop width.
        new_h = target_height
        new_w = int(round(new_h * src_ratio))
    else:
        # Source is taller/narrower than target; fit width first, then crop height.
        new_w = target_width
        new_h = int(round(new_w / src_ratio)) if src_ratio else target_height

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    horizontal_focus = min(max(horizontal_focus, 0.0), 1.0)
    vertical_focus = min(max(vertical_focus, 0.0), 1.0)

    max_left = max(0, new_w - target_width)
    max_top = max(0, new_h - target_height)
    left = int(round(max_left * horizontal_focus))
    top = int(round(max_top * vertical_focus))
    right = left + target_width
    bottom = top + target_height

    return resized.crop((left, top, right, bottom))


def prepare_webp_with_constraints(
    image_bytes: bytes,
    width: int,
    height: int,
    max_size_kb: int,
    horizontal_focus: float = 0.5,
    vertical_focus: float = 0.5,
) -> tuple[bytes, bool]:
    """
    Convert generated image to exact output spec (size + format constraints).

    Returns:
      (webp_bytes, is_within_size_limit)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = resize_and_crop_to_fill(
        img,
        width,
        height,
        horizontal_focus=horizontal_focus,
        vertical_focus=vertical_focus,
    )
    size_limit = max_size_kb * 1024

    # Try descending quality levels; keep best effort if strict limit not reachable.
    last_bytes = None
    for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
        buf = io.BytesIO()
        processed.save(
            buf,
            format="WEBP",
            quality=quality,
            method=6,
            optimize=True,
        )
        candidate = buf.getvalue()
        last_bytes = candidate
        if len(candidate) <= size_limit:
            return candidate, True

    return last_bytes or b"", False


def _encode_rgb_to_webp(img: Image.Image, max_size_kb: int) -> tuple[bytes, bool]:
    """Encode an already-sized RGB PIL image to WebP under max_size_kb."""
    size_limit = max_size_kb * 1024
    last_bytes = None
    for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality, method=6, optimize=True)
        candidate = buf.getvalue()
        last_bytes = candidate
        if len(candidate) <= size_limit:
            return candidate, True
    return last_bytes or b"", False


def check_profile_align_ready() -> tuple[bool, str]:
    """Verify face alignment backend (MediaPipe or OpenCV-only on Render)."""
    try:
        from profile_align import (
            _use_mediapipe_detection,
            ensure_face_landmarker_model,
            verify_opencv_face_detection,
        )
    except Exception as e:
        return False, f"profile_align unavailable: {type(e).__name__}: {e}"

    if _use_mediapipe_detection():
        try:
            ensure_face_landmarker_model()
        except ImportError:
            return False, "mediapipe is not installed. Run: pip install -r requirements.txt (Python 3.12)."
        except Exception as e:
            return False, f"MediaPipe: {type(e).__name__}: {e}"
    else:
        ok, err = verify_opencv_face_detection()
        if not ok:
            return False, err
    return True, ""


def apply_profile_alignment(image_bytes: bytes) -> tuple[bytes, bool, str | None]:
    """
    Run face alignment on 765×480 PNG (preserves Gemini background).
    Returns (image_bytes, success, error_message).
    """
    from profile_align import align_profile_to_grid

    last_err: str | None = None
    for attempt in range(1, 4):
        try:
            aligned = align_profile_to_grid(image_bytes)
            if aligned.size != (PROFILE_WIDTH, PROFILE_HEIGHT):
                aligned = aligned.resize(
                    (PROFILE_WIDTH, PROFILE_HEIGHT), Image.Resampling.LANCZOS
                )
            buf = io.BytesIO()
            aligned.save(buf, format="PNG")
            return buf.getvalue(), True, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} (attempt {attempt}/3)"
            print(f"[profile_align] {last_err}")

    # Last resort: center-crop to banner size (better than raw oversized Gemini).
    try:
        cropped_bytes, _ = prepare_webp_with_constraints(
            image_bytes=image_bytes,
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            max_size_kb=500,
            horizontal_focus=0.5,
            vertical_focus=0.35,
        )
        img = Image.open(io.BytesIO(cropped_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return (
            buf.getvalue(),
            True,
            f"guide alignment failed ({last_err}); used center-crop fallback",
        )
    except Exception as crop_err:
        print(f"[profile_align] center-crop fallback failed: {crop_err}")

    return image_bytes, False, last_err


def draw_profile_guide_lines(img: Image.Image) -> Image.Image:
    """Cyan guides: Y=74, Y=303, vertical center X=382.5 (scaled to image size)."""
    out = img.convert("RGB").copy()
    w, h = out.size
    top_y = round(GUIDE_TOP_Y * h / PROFILE_HEIGHT)
    bottom_y = round(GUIDE_BOTTOM_Y * h / PROFILE_HEIGHT)
    center_x = round(GUIDE_CENTER_X * w / PROFILE_WIDTH)
    draw = ImageDraw.Draw(out)
    draw.line([(0, top_y), (w, top_y)], fill=(0, 255, 255), width=2)
    draw.line([(0, bottom_y), (w, bottom_y)], fill=(0, 255, 255), width=2)
    draw.line([(center_x, 0), (center_x, h)], fill=(0, 255, 255), width=2)
    return out


def profile_image_for_display(image_bytes: bytes) -> Image.Image:
    """PIL image for Streamlit preview; adds guide lines when PROFILE_SHOW_GUIDE_LINES is on."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if PROFILE_SHOW_GUIDE_LINES:
        return draw_profile_guide_lines(img)
    return img


def prepare_profile_webp(image_bytes: bytes) -> tuple[bytes, bool]:
    """
    Profile banner → WebP. Never re-align after Gemini background uniform (warp re-adds borders).
    Picker candidates are already align → uniform at 765×480 — encode only.
    """
    try:
        from profile_align import align_profile_to_grid, is_face_on_guides, format_guide_errors

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if img.size == (PROFILE_WIDTH, PROFILE_HEIGHT):
            if not is_face_on_guides(img):
                print(f"[profile_align] note (post-uniform, no re-align): {format_guide_errors(img)}")
            return _encode_rgb_to_webp(img, PROFILE_MAX_SIZE_KB)

        aligned = align_profile_to_grid(image_bytes)
        if aligned.size != (PROFILE_WIDTH, PROFILE_HEIGHT):
            aligned = aligned.resize((PROFILE_WIDTH, PROFILE_HEIGHT), Image.Resampling.LANCZOS)
        if PROFILE_UNIFORM_BACKGROUND:
            try:
                buf = io.BytesIO()
                aligned.save(buf, format="PNG")
                fixed = uniform_profile_background_gemini(buf.getvalue())
                aligned = Image.open(io.BytesIO(fixed)).convert("RGB")
            except Exception as e:
                print(f"[profile] background uniform failed: {e}")
        return _encode_rgb_to_webp(aligned, PROFILE_MAX_SIZE_KB)
    except Exception as e:
        print(f"[profile_align] fallback to center crop: {type(e).__name__}: {e}")
        return prepare_webp_with_constraints(
            image_bytes=image_bytes,
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            max_size_kb=PROFILE_MAX_SIZE_KB,
            horizontal_focus=0.5,
            vertical_focus=0.5,
        )


def prepare_gallery_webp(image_bytes: bytes) -> tuple[bytes, bool]:
    # Keep crop slightly higher than center to avoid chopping foreheads/faces
    # when portrait-oriented images are transformed into a wide banner.
    return prepare_webp_with_constraints(
        image_bytes=image_bytes,
        width=GALLERY_WIDTH,
        height=GALLERY_HEIGHT,
        max_size_kb=GALLERY_MAX_SIZE_KB,
        horizontal_focus=0.5,
        vertical_focus=0.25,
    )


def regenerate_gallery_image_nano_banana(
    source_image_bytes: bytes,
    *,
    model: str = NANO_BANANA_REGEN_MODEL,  # kept for signature compat, unused
) -> bytes:
    """
    Produce a gallery banner directly from the source SerpApi image — no Gemini edit.

    The Gemini "edit a real-person photo" path silently refuses (IMAGE_OTHER) because
    of deepfake / public-figure protections. Instead we use the source image as-is and
    apply a deterministic, face-centered crop to 1176x516.

    Returns PNG bytes at exactly GALLERY_WIDTH x GALLERY_HEIGHT.
    """
    final_img = crop_gallery_banner_face_centered_code_only(source_image_bytes)
    out = io.BytesIO()
    final_img.save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------------
# GUARDRAIL 1 – Input quality & visibility check
# ---------------------------------------------------------------------------
def input_guardrail(images: list[tuple[bytes, str]]) -> dict:
    client = get_openai_client()

    image_content = []
    for idx, (img_bytes, mime) in enumerate(images):
        image_content.append({"type": "text", "text": f"--- Image {idx + 1} ---"})
        image_content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(img_bytes, mime), "detail": "high"},
        })

    prompt = """You are an image quality inspector. You must ONLY check the criteria listed below. Do NOT check anything else — ignore background, cropping, framing, composition, text in image, or any other aspect not explicitly listed.

ONLY check this:
1. **Image quality**: Is the image extremely blurry, extremely dark/overexposed, or extremely pixelated (unrecognizably low resolution)? Minor imperfections are fine — only fail if the image is so poor that you cannot make out the person at all.

Be lenient. If a person is clearly visible in the photo, it passes.

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{
    "is_approved": true or false,
    "issues": [
        {
            "image_index": <0-based index>,
            "problems": ["description of problem", ...]
        }
    ]
}

If ALL images pass, set "is_approved" to true and "issues" to an empty array.
Only fail an image if it violates the criteria listed above. Do NOT fail for any other reason."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a lenient image quality inspector. ONLY check what is explicitly asked. Do not invent extra criteria. Respond only with JSON."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=1000,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# IMAGE GENERATION – Gemini Nano Banana
# ---------------------------------------------------------------------------
def generate_linkedin_image(
    images: list[tuple[bytes, str]],
    improvement_feedback: str | None = None,
    custom_prompt: str | None = None,
) -> bytes:
    client = get_gemini_client()

    contents: list = []

    # 1. SUBJECT PHOTOS FIRST — these are the identity anchor. Putting them first
    #    in the contents list gives them more weight in Gemini's multi-image
    #    attention, which reduces the chance of the composition reference's
    #    person bleeding into the output.
    subject_count = len(images)
    for img_bytes, _ in images:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contents.append(pil_img)
    contents.append(
        f"The first {subject_count} image(s) above are the SUBJECT photos. They define "
        "the IDENTITY of the person who must appear in the output:\n"
        "  • Same face — same facial structure, same skin tone, same nose, same lips, "
        "    same eyes, same eyebrows, same jawline.\n"
        "  • Same hair — same hair style, same hair colour, same hairline.\n"
        "  • Same facial hair (beard, moustache, stubble) — same shape, length, density.\n"
        "  • Same age, gender, and ethnicity.\n"
        "  • Same clothing — same colour, same style, same neckline, same fabric.\n"
        "The person rendered in the output MUST be this person. Not a different "
        "person who looks similar. Not a blend of this person with anyone else. "
        "The exact same individual shown in these subject photos."
    )

    # 2. CANONICAL COMPOSITION reference (Zakir Khan) — appears AFTER the subject
    #    photos so it doesn't dominate the identity signal. Used ONLY for the
    #    four composition rules; the person, clothing, and colour palette in
    #    this reference must be ignored.
    composition_ref = Image.open(VIGNETTE_REFERENCE_PATH).convert("RGB")
    contents.append(composition_ref)
    contents.append(
        "The LAST image above is the CANONICAL COMPOSITION REFERENCE. It is a photo "
        "of the Indian stand-up comedian ZAKIR KHAN. Zakir Khan is NOT the subject "
        "of this generation — he is shown ONLY as inspiration for the COMPOSITION "
        "PATTERN (framing, headroom, gradient, vignette). Read this carefully:\n"
        "\n"
        "  ⚠ ZAKIR KHAN IS NOT THE SUBJECT. ⚠\n"
        "  ⚠ DO NOT GENERATE ZAKIR KHAN IN THE OUTPUT. ⚠\n"
        "  ⚠ THE SUBJECT IS THE PERSON IN THE SUBJECT PHOTOS ABOVE — NOT ZAKIR KHAN. ⚠\n"
        "\n"
        "Treat Zakir Khan's body, face, hair, beard, ethnicity, expression, pose, "
        "blazer, and clothing as if they were invisible. You are looking at this "
        "image purely for the LAYOUT (where the head sits in the frame, how the "
        "background fades into the clothing, how the corners are darker, where the "
        "subject is positioned horizontally). The IDENTITY of the person in the "
        "output comes ONLY from the SUBJECT PHOTOS above — never from Zakir Khan.\n"
        "\n"
        "Specifically, the FOUR composition features you must replicate from this "
        "reference are:\n"
        "  (a) SUBJECT SIZE / HEADROOM — notice how small Zakir Khan is relative to "
        "the frame: a generous band of empty background sits above the top of his "
        "hair, occupying roughly the upper 20-25% of the frame. The subject in your "
        "output should be framed from a similar wider distance — head and shoulders "
        "take only the lower portion of the image, never filling it from edge to edge.\n"
        "  (b) BLACK BOTTOM GRADIENT — the background fades smoothly into a deep "
        "dark band along the bottom edge, and that dark band extends UPWARD into "
        "the lower portion of Zakir Khan's clothing so the bottom of the blazer "
        "dissolves into darkness with no sharp visible edge. Replicate this same "
        "blend on your subject's clothing.\n"
        "  (c) HORIZONTAL CENTRING — Zakir Khan sits roughly centred in the frame "
        "with similar amounts of background on the left and the right. Centre your "
        "subject the same way.\n"
        "  (d) SOFT RADIAL VIGNETTE — the area immediately behind/around Zakir "
        "Khan's head is the brightest part of the background; the corners and "
        "edges are noticeably darker in a smooth radial fade (no hard mask, no "
        "heavy border). Replicate the same vignette behind your subject's head.\n"
        "\n"
        "FORBIDDEN — none of these from Zakir Khan should appear in your output:\n"
        "  • Zakir Khan's face, beard, eyes, nose, lips, or jawline.\n"
        "  • Zakir Khan's hair style or hair colour.\n"
        "  • Zakir Khan's ethnicity, age, gender expression, or body type.\n"
        "  • Zakir Khan's pose, hand gestures, or microphone.\n"
        "  • Zakir Khan's blazer, shirt, or any of his clothing.\n"
        "  • The maroon/red/dark background hue (use a colour that complements YOUR "
        "subject's clothing instead).\n"
        "If the output looks like Zakir Khan in any way, you have made a mistake."
    )

    # 3. BLACK BOTTOM GRADIENT reference — additional anchor for rule (b) only.
    #    May not always exist on every deployment; skip silently if missing.
    try:
        black_gradient_ref = Image.open(BLACK_GRADIENT_REFERENCE_PATH).convert("RGB")
        contents.append(black_gradient_ref)
        contents.append(
            "The image above is the BLACK BOTTOM GRADIENT REFERENCE. It exists ONLY "
            "to show the exact vertical fade pattern at the bottom of the frame — a "
            "smooth transition from the upper background colour down into a deep "
            "black band along the bottom edge, where the lower portion of the "
            "subject's clothing dissolves into the black with no hard visible edge. "
            "Use this image ONLY for the bottom-gradient pattern (rule (b)). "
            "DO NOT copy any person, clothing, face, or unrelated visual element "
            "from this image — and DO NOT copy its overall colour palette beyond "
            "the bottom-fade-to-black effect."
        )
    except FileNotFoundError:
        pass

    # 4. Generation prompt — composition rules + explicit identity anchor.
    prompt_text = f"""🚨 PRIMARY INSTRUCTION (READ FIRST, OVERRIDES EVERYTHING ELSE) 🚨

The last image in this input is a photo of ZAKIR KHAN (an Indian stand-up comedian). His image is provided STRICTLY AS A REFERENCE for the COMPOSITION ONLY. He is there for inspiration on how the layout / framing / background should look — that is the ONLY thing you should take from his image.

➤ DO NOT ADD ZAKIR KHAN IN THE IMAGE YOU'LL BE GENERATING.
➤ DO NOT make the subject look like Zakir Khan in any way.
➤ DO NOT borrow Zakir Khan's face, beard, hair, ethnicity, expression, pose, hand gestures, microphone, or clothing.
➤ The person in your output must be the person from the SUBJECT PHOTOS (the first images in the input) — a DIFFERENT individual from Zakir Khan.

If your output contains anyone who resembles Zakir Khan, the generation is wrong and must be redone.

────────────────────────────────────────────────────────────

Create a professional LinkedIn-style headshot of the SAME PERSON shown in the subject photos. The composition must satisfy ALL FOUR rules: (1) generous headroom, (2) black bottom gradient that blends into the subject's lower clothing, (3) horizontal centring, AND (4) a soft radial vignette darkening the corners — matching the composition layout from the Zakir Khan reference (last image), but with YOUR subject (not Zakir Khan) as the person in the frame.

IDENTITY (read this first — most common failure mode):
- The face, skin tone, hair, facial hair, ethnicity, age, gender, and overall identity in the output MUST come from the SUBJECT PHOTOS — the first images in the input. NOT from the composition reference.
- The composition reference (last image) is a photo of ZAKIR KHAN, an Indian stand-up comedian. Zakir Khan is provided ONLY as a layout/composition example. ZAKIR KHAN IS NOT THE SUBJECT. DO NOT generate Zakir Khan in your output. DO NOT copy his face, beard, hair, ethnicity, expression, blazer, hand pose, microphone, or any visual element of his.
- The subject of this generation is the person shown in the SUBJECT PHOTOS at the start of the input — a completely different individual from Zakir Khan.
- If your output ends up looking like Zakir Khan rather than the person in the subject photos, you have made a critical mistake. Start over with the subject photos as the identity source.
- Sanity check before finishing: cover the body in your output and look at only the face. Does that face match the person in the subject photos? If it instead matches Zakir Khan from the composition reference, regenerate.

TOP PRIORITY — HEADROOM via ZOOMED-OUT FRAMING (most important rule):
- Use a WIDER camera framing so the SUBJECT IS SMALLER in the frame. The subject's head + shoulders should occupy only the LOWER PORTION of the image, NOT fill the entire frame from top to bottom.
- Think medium-wide shot, not close-up. The camera is pulled back further than a typical headshot. The hair, face, neck, and shoulders together take roughly the lower 65-75% of the frame — never more.
- The upper portion of the frame (above Y={GUIDE_TOP_Y}) must be pure background with NO part of the subject in it.
- Face placement is defined by the designer grid below (hairline Y={GUIDE_TOP_Y}, chin Y={GUIDE_BOTTOM_Y}) — not by eye-line heuristics.
- IMPORTANT — do NOT try to add headroom by simply translating the subject downward in a tight crop. That just cuts off the shoulders/torso at the bottom. Instead, SHRINK the subject by zooming out — the subject must appear visibly smaller, with both empty background above the head AND the full shoulders/upper torso visible below.
- FORBIDDEN: hair touching or close to the top edge of the frame; head filling the upper portion of the frame; any cropping of the top of the head or the shoulders.

Style notes:
- A polished, photorealistic portrait with a warm, confident expression and a slight smile.
- The person's clothing should match what they wear in the subject photos (same outfit, same colours).
- Soft, diffused studio lighting with natural shadows under the chin and a subtle catch-light in the eyes.
- Render natural skin texture (subtle pores, fine details) — avoid an airbrushed or plastic look.
- Hair rendered with natural strands and volume, not a flat mass.

Background — vertical gradient + radial vignette (match the canonical reference for the bottom gradient AND the radial darkening):
- The TOP portion of the background is a clean, vivid colour that complements the clothing — for example a deep royal blue, teal, or polished grey-blue.
- The background transitions smoothly DOWNWARD into a rich, deep BLACK band across the bottom of the frame. The lower ~30-40% of the background fades into black, and this black band extends UPWARD into the lower portion of the subject's clothing so the bottom of the jacket dissolves into black with no hard visible edge.
- ON TOP OF the vertical gradient, apply a SOFT RADIAL VIGNETTE: the area immediately behind and around the subject's head is the brightest part of the background, and the corners and edges of the frame are noticeably darker. The vignette is smooth and subtle — no hard mask, no heavy black border, the subject is never silhouetted.
- The transitions (both vertical gradient and radial vignette) must be smooth — no banding, no hard lines.
- Net effect: the corners are dark, the bottom is darker still (fading into the subject's lower clothing), and the area behind the head has a softly-lit "halo" that draws the eye to the face.

Composition (landscape 3:2, exactly {PROFILE_WIDTH}×{PROFILE_HEIGHT} pixels):

MANDATORY FACE POSITION — designer grid (non-negotiable; post-processing depends on this):
- Output image size: exactly {PROFILE_WIDTH} pixels wide × {PROFILE_HEIGHT} pixels tall.
- Measure from the TOP-LEFT corner (0,0). Y increases downward.
- Top of hair / hairline must sit at Y = {GUIDE_TOP_Y} (not higher, not lower — not Y=70 or Y=80).
- Bottom of chin must sit at Y = {GUIDE_BOTTOM_Y} (not Y=295 or Y=310).
- The face vertical span from hairline to chin must be {FACE_BAND_HEIGHT} pixels ({GUIDE_BOTTOM_Y} minus {GUIDE_TOP_Y}).
- Horizontal centre of the face (midpoint between temples / centre of nose) must be at X = {GUIDE_CENTER_X} (horizontal centre of frame).
- Leave empty background above the hairline down to the top edge (pixels Y=0 to Y={GUIDE_TOP_Y}).
- Shoulders and upper torso extend below the chin; black gradient may cover lower clothing.
- If the face is too large, ZOOM OUT until hairline and chin hit these Y coordinates. If too small, ZOOM IN until they match.
- Do NOT place the face by "eye line at mid-frame" rules — use the Y={GUIDE_TOP_Y} and Y={GUIDE_BOTTOM_Y} coordinates above.

Additional framing:
- The full subject from top-of-hair to upper-torso fits inside the frame without cropping the top of the head or shoulders.
- Equal background on left and right of the centred face.
- A slightly angled pose works well; avoid a straight-on stare.

Self-check before finishing:
1. Is the person the SUBJECT from the upload photos (not Zakir Khan)?
2. Hairline at Y≈{GUIDE_TOP_Y}, chin at Y≈{GUIDE_BOTTOM_Y}, face centred at X≈{GUIDE_CENTER_X} on a {PROFILE_WIDTH}×{PROFILE_HEIGHT} canvas?
3. Clear background band above the hair (above Y={GUIDE_TOP_Y})?

Output one bright, polished landscape headshot at 3:2 ({PROFILE_WIDTH}×{PROFILE_HEIGHT})."""

    if custom_prompt:
        prompt_text += f"""

User preferences to incorporate where they fit naturally:
{custom_prompt}"""

    if improvement_feedback:
        prompt_text += f"""

Notes from a previous attempt — please address these in this version:
{improvement_feedback}"""

    contents.append(prompt_text)

    return call_gemini_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="3:2"),
        ),
        context="profile image",
    )


def _profile_png_bytes(image_bytes: bytes) -> bytes:
    """Normalize to 765×480 RGB PNG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.size != (PROFILE_WIDTH, PROFILE_HEIGHT):
        img = img.resize((PROFILE_WIDTH, PROFILE_HEIGHT), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def uniform_profile_background_gemini(aligned_image_bytes: bytes) -> bytes:
    """
    Gemini edit pass: remove inner frame / side bars / flat padding from affine alignment.
    Uses a full-bleed reference banner (engage4more style — no borders) as the target look.
    """
    client = get_gemini_client()
    src = Image.open(io.BytesIO(aligned_image_bytes)).convert("RGB")
    if src.size != (PROFILE_WIDTH, PROFILE_HEIGHT):
        src = src.resize((PROFILE_WIDTH, PROFILE_HEIGHT), Image.Resampling.LANCZOS)

    contents: list = [
        src,
        (
            "IMAGE 1 (above): Profile banner TO FIX. It has WRONG borders — an inner "
            "rectangle (smaller photo) inside the frame, with dark gray/black vertical "
            "bars on the left and right and/or flat padding around the subject. "
            "Picture-in-picture effect. This must be converted to full-bleed."
        ),
    ]

    # Reference: correct engage4more banner — gradient fills entire canvas, zero borders.
    ref_path = VIGNETTE_REFERENCE_PATH
    if os.path.isfile(ref_path):
        ref_img = Image.open(ref_path).convert("RGB")
        contents.append(ref_img)
        contents.append(
            "IMAGE 2 (above): TARGET STYLE — a correct engage4more profile banner. "
            "Notice: the background gradient and vignette extend to ALL FOUR EDGES of "
            "the frame. There is NO inner box, NO black side bars, NO second background, "
            "NO letterboxing. The person sits on one continuous canvas (like Aman Gupta / "
            "standard speaker banners on the site)."
        )

    prompt = f"""Convert IMAGE 1 into the same FULL-BLEED layout quality as IMAGE 2 (target style).

WHAT IS WRONG IN IMAGE 1 (you must eliminate ALL of these):
- Dark gray or black vertical bars on the left and right edges.
- A visible inner rectangle — the portrait looks pasted inside a larger frame.
- Flat, solid-colour padding bands that do not match the inner gradient.
- Any "two backgrounds" or picture-in-picture seam.

WHAT IMAGE 2 SHOWS (your output must match this structure):
- One seamless background from edge to edge on a {PROFILE_WIDTH}×{PROFILE_HEIGHT} canvas.
- Top: clean colour behind the head; bottom: smooth fade to black into clothing; corners: soft vignette.
- Background touches the left, right, top, and bottom edges — no empty bars anywhere.

STRICT — DO NOT CHANGE THE PERSON IN IMAGE 1:
- Same face, identity, pose, expression, clothing, hair, skin lighting.
- Face position is FIXED: hairline MUST stay at Y={GUIDE_TOP_Y}, chin at Y={GUIDE_BOTTOM_Y}, face center at X={GUIDE_CENTER_X} (designer grid). Do NOT move the face up/down/left/right.
- Do NOT zoom or recrop the subject. Do NOT replace the subject with anyone from IMAGE 2.

YOUR ONLY TASK: Repaint/extend the BACKGROUND of IMAGE 1 so it looks like IMAGE 2's full-bleed treatment — subject unchanged, borders gone.

Output exactly one {PROFILE_WIDTH}×{PROFILE_HEIGHT} landscape image. No text or logos."""

    contents.append(prompt)

    out_bytes = call_gemini_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="3:2"),
        ),
        context="profile background uniform",
    )
    return _profile_png_bytes(out_bytes)


def generate_gallery_image(
    images: list[tuple[bytes, str]],
    scene_prompt: str,
    profile_anchor_image: bytes | None = None,
    improvement_feedback: str | None = None,
    custom_prompt: str | None = None,
) -> bytes:
    """
    Generate one natural event/action gallery image for the artist.

    Args:
        images: User-uploaded subject reference photos.
        scene_prompt: The event scene description for this gallery variant.
        profile_anchor_image: The approved profile headshot (raw PNG bytes from Gemini).
            Used as the CANONICAL face reference for maximum identity consistency.
        improvement_feedback: Feedback from a prior failed face-match guardrail attempt.
        custom_prompt: User-supplied extra tags / preferences.
    """
    client = get_gemini_client()

    contents: list = []

    # 1. Style reference (lighting / polish only)
    style_ref = Image.open(STYLE_REFERENCE_PATH).convert("RGB")
    contents.append(style_ref)
    contents.append(
        "Above is a STYLE REFERENCE image. Use it only for realism, clean lighting, "
        "polish, and overall photographic quality. DO NOT copy this person's face — "
        "this is not the subject."
    )

    # 2. CANONICAL FACE ANCHOR — the approved profile headshot (if available)
    if profile_anchor_image:
        anchor_img = Image.open(io.BytesIO(profile_anchor_image)).convert("RGB")
        contents.append(anchor_img)
        contents.append(
            "Above is the CANONICAL FACE REFERENCE. This is the exact face you MUST "
            "reproduce in the output — same facial structure, same skin tone, same "
            "facial hair, same hair, same eyes, same nose, same lips. This image is "
            "the single source of truth for this person's identity. Treat every facial "
            "pixel in this reference as sacred — do not invent, interpolate, or stylise. "
            "If there is ANY conflict between this reference and the subject photos below, "
            "THIS reference wins."
        )

    # 3. Additional subject reference photos (for clothing + secondary identity cues)
    for img_bytes, _ in images:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contents.append(pil_img)
    contents.append(
        "Above are additional SUBJECT reference photos. Use them to confirm clothing "
        "colours/style and secondary identity cues. For the face, defer to the canonical "
        "face reference."
    )

    prompt_text = f"""Generate one natural, professional event-style gallery image of this person.

SCENE REQUIREMENT:
{scene_prompt}

CRITICAL REQUIREMENTS:

1. **FACIAL IDENTITY — HIGHEST PRIORITY (non-negotiable)**: The face in the output MUST be a pixel-faithful reproduction of the canonical face reference. This is the single most important requirement — an image with a non-matching face is an automatic failure. The generated person must be instantly recognisable as the same individual to anyone who sees both images side-by-side.

   Match ALL of these EXACTLY:
   - Eye shape, eye spacing, eye colour, eyelid crease, and distance between the eyes.
   - Eyebrow shape, thickness, arch, and position relative to the eyes.
   - Nose shape, tip, width, bridge height, and nostril flare.
   - Lip shape, lip fullness (upper vs lower), cupid's bow curvature, and mouth width.
   - Jawline angle, chin shape and prominence, cheekbone height and definition.
   - Skin tone, undertone, and skin colour distribution (do NOT lighten, darken, or smooth).
   - Facial hair pattern: exact beard shape, length, density, moustache style, stubble zones.
   - Hair style, hair colour, parting, hairline, hair volume, and texture.
   - Any moles, freckles, scars, dimples, or distinguishing marks.
   - Head shape and proportions (forehead size, face length-to-width ratio).

   DO NOT invent or stylise the face. DO NOT merge with generic "speaker" stereotypes. The output face is a TRANSPLANT of the canonical reference into the scene — nothing more.

2. **Hyper-realistic face rendering**: Visible skin pores, subsurface scattering, individual eyebrow hairs and eyelashes, realistic iris detail with a specular highlight. Absolutely NO plastic, airbrushed, painted, or CGI look.

3. **Clothing**: Preserve clothing style and colours consistent with the subject reference photos (do not drastically change the outfit unless the scene demands it).

4. **Scene feel**: Image should feel candid/action-oriented, not a studio headshot. Natural event atmosphere with realistic lighting, no AI artifacts.

5. **Composition**: Horizontal banner suitable for a website gallery.

6. **Face visibility**: Keep the face clearly visible and sharp — the face is the focal point even in action/wide shots. No heavy occlusions, no motion blur on the face, no awkward cropping.

7. **Face sizing (strict)**:
   - The face (hairline to chin) MUST occupy at least 20-30% of the frame's vertical height.
   - Face pixels in the output must be large enough that every facial feature listed above is clearly recognisable.
   - If the scene would push the face too small (e.g. very wide establishing shot), pull the camera closer to the subject so the face stays detailed.

8. **Face-centering constraints**:
   - Keep the face near the centre area of the frame even in action shots.
   - Ensure full head visibility with clean headroom (no top/head crop).
   - Avoid aggressive edge-cropping of face or shoulders.

Output a single high-quality horizontal image. REMEMBER: face identity is non-negotiable. If the face does not match the canonical reference exactly, the image is a failure."""

    if custom_prompt:
        prompt_text += f"""

USER TAGS / PREFERENCES:
{custom_prompt}
Apply these preferences when they do not conflict with core quality, realism, or facial identity."""

    if improvement_feedback:
        prompt_text += f"""

IMPORTANT — The previous gallery attempt was rejected because the face did not match the canonical reference. Feedback:
{improvement_feedback}

Regenerate this gallery image with a face that EXACTLY matches the canonical face reference. All other requirements remain in force."""

    contents.append(prompt_text)

    return call_gemini_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="21:9"),
        ),
        context="gallery image",
    )


# ---------------------------------------------------------------------------
# GUARDRAIL 2 – Output quality check
# ---------------------------------------------------------------------------
def output_guardrail(generated_image: bytes, original_images: list[tuple[bytes, str]]) -> dict:
    client = get_openai_client()

    image_content = []
    for idx, (img_bytes, mime) in enumerate(original_images):
        image_content.append({"type": "text", "text": f"--- Original Reference Image {idx + 1} ---"})
        image_content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(img_bytes, mime), "detail": "high"},
        })

    image_content.append({"type": "text", "text": "--- Generated LinkedIn Profile Image ---"})
    image_content.append({
        "type": "image_url",
        "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
    })

    prompt = """You are a quality assurance reviewer for AI-generated LinkedIn profile headshots.

You are given the original reference photos of a person AND the AI-generated LinkedIn profile image.

You must ONLY check the criteria listed below. Do NOT invent additional criteria. Do NOT flag things like microphones, accessories, lanyards, logos, watermarks, or anything not explicitly listed. Be reasonable — a small artefact is fine, only fail on SEVERE problems.

ONLY evaluate these:
1. **Face resemblance**: Is the generated person clearly the same individual as in the reference photos? (Expression, angle, and lighting differences are fine — only fail if the face looks like a completely different person.)
2. **Severe artifacts**: Are there SEVERE AI-generation defects — extra limbs, missing eyes, melted/warped features, two faces, deformed hands visible near the face? (Minor imperfections are fine.)
3. **Appropriate content**: Is the content appropriate and professional? (No offensive, explicit, or violent imagery.)

Be lenient. The image should be REJECTED ONLY if:
- The face is clearly a different person, OR
- There are severe AI-generation defects that would make the image unusable, OR
- The content is offensive / inappropriate.

Do NOT reject for: background choice, cropping, minor focus/blur, posture, clothing preferences, presence of microphones/accessories, lighting preferences, or any other aesthetic judgement.

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "short description of the SEVERE issues only"
}

If none of the three rejection conditions apply, return is_approved=true and things_to_improve=false."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a LENIENT image reviewer. ONLY check what is explicitly asked. Do not invent extra criteria. Respond only with JSON, no markdown."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=1000,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# GUARDRAIL 2b – LinkedIn profile composition check (gradient / headroom / centering)
# ---------------------------------------------------------------------------
def headroom_guardrail(generated_image: bytes) -> dict:
    """
    Single-criterion guardrail: does the generated LinkedIn headshot have enough
    headroom above the subject's head? Anchored to the VIGNETTE reference image.

    Returns {"is_approved": bool, "things_to_improve": str | False}.
    """
    client = get_openai_client()

    with open(VIGNETTE_REFERENCE_PATH, "rb") as f:
        vignette_ref_bytes = f.read()

    image_content = [
        {"type": "text", "text": "--- IMAGE 1: CANONICAL REFERENCE (zakir-khan_Comedian.png) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(vignette_ref_bytes, "image/png"), "detail": "high"},
        },
        {"type": "text", "text": "--- IMAGE 2: GENERATED LINKEDIN IMAGE (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a STRICT headroom reviewer for AI-generated LinkedIn profile headshots.

You are given TWO images:
- IMAGE 1: the canonical reference (zakir-khan_Comedian.png).
- IMAGE 2: the generated LinkedIn image to evaluate.

Check ONE thing only: does IMAGE 2 have enough empty background ABOVE the top of the subject's hair, matching (or exceeding) the headroom shown in IMAGE 1?

Notice in IMAGE 1: the subject is framed at a wider distance — the head sits in the lower portion of the frame, and there is a clearly visible band of empty background occupying roughly the upper 20-25% of the frame above the hair.

PASS if the empty band above the hair in IMAGE 2 takes at least ~15% of the frame's vertical height.

FAIL if any of these are true:
- The hair in IMAGE 2 touches or nearly touches the top edge of the frame.
- The top of the hair sits in the upper 10% of the frame.
- IMAGE 2 has noticeably less headroom than IMAGE 1.
- The top of the head is cropped at all.

Do NOT evaluate anything else — not the person, clothing, background colour, lighting, vignette, or centring. Headroom ONLY.

If the rule fails, write the fix instruction using direct image comparison. The actionable fix is always: make the SUBJECT IN IMAGE 2 smaller (zoom out / wider crop) so its size matches the subject in IMAGE 1. Do NOT simply add empty pixels above the existing subject — that would crop the shoulders.

Always phrase the fix this way (template, fill in the bracketed parts):
"Reduce the size of the subject in image 2 (the generated image) and make it equivalent to the size of the subject in image 1 (zakir-khan_Comedian.png) — currently the head in image 2 [describe how it's too large or too close to the top edge in one short phrase]. Pull the camera back / use a wider crop so the head and shoulders in image 2 occupy the same proportion of the frame as in image 1, leaving a generous band of empty background above the hair. Do not zoom in tighter; do not simply translate the subject down (that would crop the shoulders)."

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "fix instruction following the template above if it failed"
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict single-criterion headroom reviewer. When instructing fixes, always tell the generator to shrink the subject in image 2 to match the size of the subject in image 1. Respond only with JSON, no markdown fences."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=400,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


def bottom_gradient_guardrail(generated_image: bytes) -> dict:
    """
    Single-criterion guardrail: does the generated LinkedIn headshot have a dark
    bottom gradient that blends UP into the subject's clothing, matching the
    canonical reference image (zakir-khan_Comedian.png)?

    Returns {"is_approved": bool, "things_to_improve": str | False}.
    """
    client = get_openai_client()

    with open(VIGNETTE_REFERENCE_PATH, "rb") as f:
        ref_bytes = f.read()

    image_content = [
        {"type": "text", "text": "--- IMAGE 1: CANONICAL REFERENCE (zakir-khan_Comedian.png) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(ref_bytes, "image/png"), "detail": "high"},
        },
        {"type": "text", "text": "--- IMAGE 2: GENERATED LINKEDIN IMAGE (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a STRICT bottom-gradient reviewer for AI-generated LinkedIn profile headshots.

You are given TWO images:
- IMAGE 1: the canonical reference (zakir-khan_Comedian.png).
- IMAGE 2: the generated LinkedIn image to evaluate.

Check ONE thing only: does IMAGE 2 have the same dark-bottom-gradient-merging-into-subject effect as IMAGE 1?

Notice in IMAGE 1: the bottom of the frame fades into a deep dark band, and that darkness extends UPWARD into the lower portion of the subject — the bottom of the clothing/shoulders is partially absorbed into the darkness, with NO sharp visible edge between the subject and the bottom of the frame.

PASS only if IMAGE 2 has BOTH of these (matching IMAGE 1):
(a) A deep dark band along the bottom edge of the frame.
(b) That dark band extends UPWARD into the lower portion of the subject's clothing so the bottom of the jacket/shirt dissolves into the darkness with no hard visible edge.

FAIL if any of these are true in IMAGE 2:
- The background is a flat colour with no dark bottom band.
- Only a corner vignette is present, without a dark bottom band.
- The bottom of the subject's clothing is fully lit and sits cleanly above a visible edge instead of dissolving into black.
- A dark band exists but does NOT extend up into the subject's clothing (there's a hard visible edge where the subject ends).

Do NOT evaluate anything else — not the person, clothing colour, background hue, lighting, vignette, headroom, or centring. Bottom-gradient ONLY.

If the rule fails, write the fix instruction using direct image comparison. Always phrase the fix this way (template, fill in the bracketed parts):
"Replicate the bottom-gradient pattern from image 1 (zakir-khan_Comedian.png) in image 2 (the generated image) — currently in image 2 [describe what's wrong in one short phrase, e.g. 'the background is a flat colour' or 'the bottom of the jacket has a hard visible edge']. Add a smooth dark gradient across the bottom of the background AND let that darkness extend upward into the lower portion of the subject's clothing so the bottom of the jacket dissolves into the dark area, exactly as in image 1."

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "fix instruction following the template above if it failed"
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict single-criterion bottom-gradient reviewer. When instructing fixes, always reference image 1 (zakir-khan_Comedian.png) vs image 2 (the generated image) directly. Respond only with JSON, no markdown fences."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=400,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


def vignette_guardrail(generated_image: bytes) -> dict:
    """
    Single-criterion guardrail: does the generated LinkedIn headshot show a soft
    radial vignette matching the canonical reference (zakir-khan_Comedian.png)?

    Returns {"is_approved": bool, "things_to_improve": str | False}.
    """
    client = get_openai_client()

    with open(VIGNETTE_REFERENCE_PATH, "rb") as f:
        ref_bytes = f.read()

    image_content = [
        {"type": "text", "text": "--- IMAGE 1: CANONICAL REFERENCE (zakir-khan_Comedian.png) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(ref_bytes, "image/png"), "detail": "high"},
        },
        {"type": "text", "text": "--- IMAGE 2: GENERATED LINKEDIN IMAGE (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a STRICT vignette reviewer for AI-generated LinkedIn profile headshots.

You are given TWO images:
- IMAGE 1: the canonical reference (zakir-khan_Comedian.png).
- IMAGE 2: the generated LinkedIn image to evaluate.

Check ONE thing only: does IMAGE 2 have the same soft radial vignette effect as IMAGE 1?

Notice in IMAGE 1: the area immediately behind/around the subject's head is the brightest part of the background, and the corners and edges of the frame are noticeably darker — a clear, smooth radial darkening from the centre outward.

PASS if the corners of the frame in IMAGE 2 are clearly darker than the area directly behind the subject's head, with a smooth fade and no visible banding.

FAIL if any of these are true in IMAGE 2:
- The background is a flat colour from edge to edge with no radial darkening.
- The corners are the same brightness as the centre.
- The vignette is so heavy or hard-edged that the subject is silhouetted, or a visible dark border ring appears.
- The fade has visible banding or hard transitions.

Do NOT evaluate anything else — not the person, clothing, background colour, lighting style, headroom, centring, or bottom gradient. Radial vignette ONLY.

If the rule fails, write the fix instruction using direct image comparison. Always phrase the fix this way (template, fill in the bracketed parts):
"Replicate the radial vignette pattern from image 1 (zakir-khan_Comedian.png) in image 2 (the generated image) — currently in image 2 [describe what's wrong in one short phrase, e.g. 'the background is uniformly bright from edge to edge' or 'the corners are not noticeably darker than the centre']. Apply a soft, smooth radial darkening so the area behind the head stays the brightest part of the background and the corners and edges fade gradually into noticeably darker tones, matching the gentle vignette in image 1. Avoid a hard mask or a visible dark border."

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "fix instruction following the template above if it failed"
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict single-criterion vignette reviewer. When instructing fixes, always reference image 1 (zakir-khan_Comedian.png) vs image 2 (the generated image) directly. Respond only with JSON, no markdown fences."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=400,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# REFINEMENT — per-image feedback analyzer (OpenAI) + Gemini editor.
# Used by the "Refine gallery images" client-feedback flow.
# ---------------------------------------------------------------------------
def analyze_gallery_image_for_feedback(
    gallery_image: bytes,
    client_feedback: str,
) -> dict:
    """
    Ask GPT-4o to look at ONE gallery image and produce a specific, actionable
    instruction string that captures what needs to change in THIS particular image,
    grounded in the client's general feedback.

    Returns {"needs_refinement": bool, "instruction": str | None}.
    If the image doesn't exhibit the problem described, needs_refinement is False.
    """
    client = get_openai_client()

    image_content = [
        {"type": "text", "text": "--- GALLERY IMAGE TO ANALYZE ---"},
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image_for_openai(gallery_image, "image/webp"),
                "detail": "high",
            },
        },
    ]

    prompt = f"""You are an image quality reviewer for a website's artist gallery.

The client has shared the following general feedback about the gallery images and wants them refined:

CLIENT FEEDBACK:
\"\"\"{client_feedback}\"\"\"

Look at the image above and decide whether it exhibits the problem(s) described. If yes, produce ONE concise, specific, ACTIONABLE instruction (a single short paragraph) that an image editing model can follow to fix the issue in THIS image. Be concrete — point at WHERE in the image the issue is (e.g., "the upper-left corner has visible text 'Comedy Night 2024' on the backdrop", "a logo of XYZ Network is in the bottom-right corner") and what should replace it (e.g., "remove the text and replace with a clean continuation of the dark backdrop").

If the image does NOT exhibit the problem the client described, return needs_refinement=false and instruction=null. Do not invent issues.

Do not change the subject, their face, their clothing, their pose, the lighting, or the overall scene. Refinement is ONLY to address the client's specific complaint.

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{{
    "needs_refinement": true or false,
    "instruction": "specific actionable instruction" or null
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You analyze a single image against client feedback. Output only valid JSON, no markdown.",
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=400,
        temperature=0,
    )
    return parse_json_response(response.choices[0].message.content)


def refine_gallery_image_with_feedback(
    gallery_image: bytes,
    instruction: str,
) -> bytes:
    """
    Ask Gemini (Nano Banana) to edit a single gallery image guided by the
    OpenAI-generated instruction. Returns refined image bytes at the same
    aspect/size as the input (caller is responsible for re-cropping/encoding).

    NOTE: Gemini's image-edit path on real-person photos often refuses with
    IMAGE_OTHER. The retry helper handles transient failures; permanent
    refusals will raise RuntimeError, which the caller should catch and treat
    as "keep original image".
    """
    client = get_gemini_client()

    src_img = Image.open(io.BytesIO(gallery_image)).convert("RGB")

    prompt = (
        "Refine this image with a minimal, targeted edit.\n\n"
        f"Specific instruction:\n{instruction}\n\n"
        "Keep the person, their face, their clothing, their pose, the lighting, "
        "and the overall composition EXACTLY the same. Only address the specific "
        "instruction above. Return a single high-quality landscape image at the "
        "same aspect ratio as the input."
    )

    edited_bytes = call_gemini_with_retry(
        client,
        model=NANO_BANANA_REGEN_MODEL,
        contents=[src_img, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
        context="gallery refinement",
    )
    return edited_bytes


# ---------------------------------------------------------------------------
# GUARDRAIL 3 – Gallery face-match check
# ---------------------------------------------------------------------------
def gallery_face_match_guardrail(
    gallery_image: bytes,
    profile_anchor_image: bytes,
) -> dict:
    """
    Compares the face in a generated gallery image against the approved profile
    anchor. Returns {is_approved: bool, things_to_improve: str | False}.

    Only checks facial identity — not scene composition, lighting, etc.
    """
    client = get_openai_client()

    image_content = [
        {"type": "text", "text": "--- CANONICAL Profile Face (ground truth) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(profile_anchor_image, "image/png"), "detail": "high"},
        },
        {"type": "text", "text": "--- Generated Gallery Image (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(gallery_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a strict facial-identity reviewer. You are given:
1. A canonical profile photo of a person (the ground truth).
2. A generated gallery image that should contain the SAME person.

Your ONLY job is to determine whether the face in the gallery image matches the canonical face closely enough that a stranger would recognise them as the same person.

Compare these features between the two images:
- Eye shape, spacing, colour
- Eyebrow shape and thickness
- Nose shape, bridge, nostrils
- Lip shape and fullness
- Jawline, chin, cheekbones
- Skin tone and undertone
- Facial hair pattern (beard, moustache, stubble)
- Hair style, colour, hairline

Be REASONABLY STRICT — different expression, angle, or scene lighting is fine. But the underlying facial identity must clearly be the same individual. If the face looks like a different person, or a generic "speaker" stereotype that doesn't match, REJECT.

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "specific description of which facial features do NOT match and how they differ"
}

If the face clearly matches, return is_approved=true and things_to_improve=false.
If not, return is_approved=false and describe the specific mismatches (e.g. "nose is wider and shorter than reference; jawline is rounder; beard is longer than reference")."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict facial-identity reviewer. ONLY judge whether the face is the same person. Respond only with JSON."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=600,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Gemini-based Zakir Khan identity check (no reference image).
# Sends ONLY the generated image to Gemini and asks whether the person depicted
# is Zakir Khan, relying on the model's prior knowledge of the comedian.
# ---------------------------------------------------------------------------
def gemini_zakir_check(generated_image: bytes) -> dict:
    """
    Returns {"is_zakir": bool, "explanation": str | None}.
    is_zakir == True means the generated person LOOKS LIKE Zakir Khan.
    Uses gemini-2.5-flash (vision-capable text model) for the classification.
    """
    client = get_gemini_client()
    src_img = Image.open(io.BytesIO(generated_image)).convert("RGB")

    prompt = (
        "Look carefully at the person in this image. Decide whether this is "
        "ZAKIR KHAN, the Indian stand-up comedian.\n\n"
        "Use your prior knowledge of what Zakir Khan looks like — his face, "
        "beard, eyes, build, and overall identity. Don't be tricked by the "
        "background, the clothing, or the studio look — focus only on the "
        "person's face and identity.\n\n"
        "Return is_zakir=true ONLY if you are reasonably confident the person "
        "shown is Zakir Khan. Otherwise return is_zakir=false. If you are not "
        "sure, return false.\n\n"
        "Respond with valid JSON ONLY (no markdown fences) in this exact schema:\n"
        "{\n"
        '  "is_zakir": true or false,\n'
        '  "explanation": "one short sentence explaining the verdict"\n'
        "}"
    )

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=[src_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    # Pull text out of the response — text-mode response, not image.
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return {"is_zakir": False, "explanation": "No candidates returned by Gemini."}

    raw_text = None
    for part in getattr(candidates[0].content, "parts", []) or []:
        t = getattr(part, "text", None)
        if t:
            raw_text = t
            break

    if not raw_text:
        return {"is_zakir": False, "explanation": "Empty text response from Gemini."}

    try:
        parsed = parse_json_response(raw_text)
    except Exception as e:
        return {"is_zakir": False, "explanation": f"Could not parse Gemini JSON ({e})."}

    return {
        "is_zakir": bool(parsed.get("is_zakir")),
        "explanation": parsed.get("explanation"),
    }


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.set_page_config(page_title="Engage4more Profile and Gallery Image Generator", page_icon="📸", layout="wide")

st.title("Engage4more Profile and Gallery Image Generator")
st.caption(
    "Enter artist name, optional profile tags, and upload 2-3 reference photos. "
    "One click generates profile + final top 10 gallery images."
)

_align_ok, _align_msg = check_profile_align_ready()
if _align_ok:
    try:
        from profile_align import _use_mediapipe_detection

        _mp_note = (
            "MediaPipe + YuNet"
            if _use_mediapipe_detection()
            else "YuNet + Haar (Render — PROFILE_ALIGN_USE_MEDIAPIPE=0)"
        )
    except Exception:
        _mp_note = "OpenCV"
    st.caption(f"Face alignment: ready (guides Y 74–303, center X 382.5) — {_mp_note}.")
else:
    st.warning(f"Face alignment not ready — {_align_msg}")

artist_name = st.text_input("Artist name", placeholder="e.g., Ankur Warikoo")
optional_tags = st.text_area(
    "Optional tags/prompts",
    placeholder='e.g., "corporate keynote", "motivational speaker on stage", "stand-up comedy show"',
    help="These tags guide only the main profile image generation style.",
)

if "result_data" not in st.session_state:
    st.session_state.result_data = None
if "reference_pool_rows" not in st.session_state:
    st.session_state.reference_pool_rows = None
if "reference_top6_indices" not in st.session_state:
    st.session_state.reference_top6_indices = None
if "reference_rank_note" not in st.session_state:
    st.session_state.reference_rank_note = None
if "reference_regenerated" not in st.session_state:
    st.session_state.reference_regenerated = None
# Profile-selection phase state — 3-candidate flow.
if "pipeline_phase" not in st.session_state:
    st.session_state.pipeline_phase = "idle"  # idle | awaiting_selection | running_gallery | done
if "profile_candidates" not in st.session_state:
    st.session_state.profile_candidates = None  # list[bytes] of 3 PNGs
if "pipeline_inputs" not in st.session_state:
    st.session_state.pipeline_inputs = None  # cached form inputs across reruns

# ---- Google Images API: reference pool (gallery aspect) ----
st.divider()
st.subheader(f"Reference image pool (~{REFERENCE_IMAGE_POOL_SIZE} via Google API)")
st.caption(
    f"Google API searches favour **live event / action** shots (stage, audience, candid, expressive). "
    f"Then **GPT-4o** picks **{REFERENCE_TOP_K_OPENAI}** using **four event types**, single clear subject, visible face, low text, "
    f"plus banner aspect ~{GALLERY_WIDTH}×{GALLERY_HEIGHT}. "
    "Set `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ID`, and `OPENAI_API_KEY` in `.env`."
)
if st.button(f"Fetch {REFERENCE_IMAGE_POOL_SIZE} images (Google API)", key="btn_google_pool", width="content"):
    st.session_state.reference_pool_rows = None
    st.session_state.reference_top6_indices = None
    st.session_state.reference_rank_note = None
    st.session_state.reference_regenerated = None
    if not artist_name.strip():
        st.error("Enter an artist / celebrity name above first.")
    else:
        api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
        cx = os.getenv("GOOGLE_CSE_ID", "").strip()
        if not api_key or not cx:
            st.error("Missing `GOOGLE_CSE_API_KEY` or `GOOGLE_CSE_ID` in environment (.env).")
        else:
            with st.spinner(
                f"Google API: fetching up to {REFERENCE_IMAGE_POOL_SIZE} images, "
                f"then OpenAI: selecting top {REFERENCE_TOP_K_OPENAI}…"
            ):
                try:
                    candidates = fetch_google_image_candidates(
                        artist_name.strip(),
                        api_key,
                        cx,
                        target_aspect=GALLERY_WIDTH / GALLERY_HEIGHT,
                        target_count=REFERENCE_IMAGE_POOL_SIZE,
                    )
                except Exception as e:
                    st.error(str(e))
                    candidates = []

                rows: list[dict] = []
                for pool_idx, c in enumerate(candidates):
                    data = None
                    err = None
                    for url in (c.get("link"), c.get("thumbnail_link")):
                        if not url:
                            continue
                        try:
                            data = download_image_bytes(url)
                            if data and len(data) > 500:
                                data = _downscale_for_pool(data)
                                break
                        except Exception as ex:
                            err = str(ex)
                            data = None
                    rows.append(
                        {
                            "pool_idx": pool_idx,
                            "index": pool_idx + 1,
                            "title": c.get("title", ""),
                            "link": c.get("link", ""),
                            "width": c.get("width"),
                            "height": c.get("height"),
                            "aspect_ratio": c.get("aspect_ratio"),
                            "bytes": data,
                            "error": err if not data else None,
                        }
                    )
                st.session_state.reference_pool_rows = rows

                ta = GALLERY_WIDTH / GALLERY_HEIGHT
                try:
                    top6, rank_err = rank_reference_images_openai(
                        rows,
                        artist_name.strip(),
                        ta,
                        top_k=REFERENCE_TOP_K_OPENAI,
                    )
                except Exception as e:
                    top6 = _fallback_top_indices_by_aspect(rows, ta, REFERENCE_TOP_K_OPENAI)
                    rank_err = f"OpenAI ranking failed ({e}); used aspect-ratio fallback."
                st.session_state.reference_top6_indices = top6
                st.session_state.reference_rank_note = rank_err

if st.session_state.reference_pool_rows:
    st.success(
        f"Pool: {len(st.session_state.reference_pool_rows)} images "
        f"(target {REFERENCE_IMAGE_POOL_SIZE})."
    )
    gcols = st.columns(5)
    for idx, row in enumerate(st.session_state.reference_pool_rows):
        with gcols[idx % 5]:
            cap_parts = [f"#{row.get('index', idx + 1)}"]
            if row.get("width") and row.get("height"):
                cap_parts.append(f"{row['width']}×{row['height']}")
            if row.get("aspect_ratio"):
                cap_parts.append(f"r={row['aspect_ratio']:.2f}")
            caption = " · ".join(cap_parts)
            if row.get("bytes"):
                st.image(row["bytes"], caption=caption, width="stretch")
            else:
                st.warning(f"{caption}\nCould not load: {row.get('error', 'unknown')}")
            if row.get("link"):
                st.markdown(f"[Open source]({row['link']})")

    top6 = st.session_state.reference_top6_indices
    if top6:
        st.divider()
        st.subheader(f"Top {REFERENCE_TOP_K_OPENAI} (OpenAI ranked)")
        if st.session_state.reference_rank_note:
            st.caption(st.session_state.reference_rank_note)
        pool = st.session_state.reference_pool_rows
        cols6 = st.columns(3)
        for j, pool_idx in enumerate(top6):
            if pool_idx < 0 or pool_idx >= len(pool):
                continue
            row = pool[pool_idx]
            with cols6[j % 3]:
                cap = f"Pick #{j + 1} · pool idx {pool_idx}"
                if row.get("bytes"):
                    st.image(row["bytes"], caption=cap, width="stretch")
                else:
                    st.warning(f"{cap} — missing bytes")
                if row.get("link"):
                    st.markdown(f"[Open source]({row['link']})")

        st.caption(
            f"Gemini re-renders each pick: removes text/watermarks, keeps people unchanged, "
            f"then exports **{GALLERY_WIDTH}×{GALLERY_HEIGHT}** `.webp` (target **≤{GALLERY_MAX_SIZE_KB} KB** each)."
        )
        if st.button(
            f"Regenerate top {REFERENCE_TOP_K_OPENAI} with Gemini (clean + WebP)",
            key="btn_regen_top6_gemini",
            width="content",
        ):
            st.session_state.reference_regenerated = None
            pool = st.session_state.reference_pool_rows
            regen_out: list[dict] = []
            with st.spinner(f"Gemini + WebP: processing {len(top6)} images…"):
                for j, pool_idx in enumerate(top6):
                    if pool_idx < 0 or pool_idx >= len(pool):
                        regen_out.append(
                            {
                                "slot": j + 1,
                                "pool_idx": pool_idx,
                                "webp_bytes": None,
                                "within_limit": False,
                                "error": "Invalid pool index",
                            }
                        )
                        continue
                    src = pool[pool_idx].get("bytes")
                    if not src:
                        regen_out.append(
                            {
                                "slot": j + 1,
                                "pool_idx": pool_idx,
                                "webp_bytes": None,
                                "within_limit": False,
                                "error": "No source bytes",
                            }
                        )
                        continue
                    try:
                        raw = regenerate_reference_clean_gemini(src)
                        webp_b, ok = prepare_gallery_webp(raw)
                        if not webp_b:
                            raise RuntimeError("WebP export failed")
                        regen_out.append(
                            {
                                "slot": j + 1,
                                "pool_idx": pool_idx,
                                "webp_bytes": webp_b,
                                "within_limit": ok,
                                "error": None,
                            }
                        )
                    except Exception as ex:
                        regen_out.append(
                            {
                                "slot": j + 1,
                                "pool_idx": pool_idx,
                                "webp_bytes": None,
                                "within_limit": False,
                                "error": str(ex),
                            }
                        )
            st.session_state.reference_regenerated = regen_out

    regen = st.session_state.reference_regenerated
    if regen:
        st.divider()
        st.subheader("Gemini-cleaned gallery exports")
        st.caption(
            f"Each image: **{GALLERY_WIDTH}×{GALLERY_HEIGHT}** WebP. "
            f"Green caption = within {GALLERY_MAX_SIZE_KB} KB; otherwise best-effort compression."
        )
        rcols = st.columns(3)
        for j, item in enumerate(regen):
            with rcols[j % 3]:
                slot = item.get("slot", j + 1)
                if item.get("webp_bytes"):
                    kb = len(item["webp_bytes"]) / 1024
                    cap = f"Regen #{slot} · {kb:.1f} KB"
                    if item.get("within_limit"):
                        st.success(f"{cap} (≤{GALLERY_MAX_SIZE_KB} KB)")
                    else:
                        st.warning(f"{cap} (over {GALLERY_MAX_SIZE_KB} KB target)")
                    st.image(
                        Image.open(io.BytesIO(item["webp_bytes"])),
                        caption=f"pool idx {item.get('pool_idx')}",
                        width="stretch",
                    )
                    st.download_button(
                        label=f"Download regen_{slot}.webp",
                        data=item["webp_bytes"],
                        file_name=f"gallery_regen_{slot}.webp",
                        mime="image/webp",
                        key=f"dl_regen_{slot}",
                        width="stretch",
                    )
                else:
                    st.error(f"Regen #{slot} failed: {item.get('error', 'unknown')}")

# ---- Sidebar ----
with st.sidebar:
    st.header("How it works")
    st.markdown(
        f"""
1. Upload **{MIN_IMAGES}-{MAX_IMAGES}** clear reference photos
2. Generate **profile image** (Nano Banana + output review)
3. Fetch SerpApi pool, rank top **{REFERENCE_TOP_K_OPENAI}** with GPT-4o
4. Regenerate those top **{REFERENCE_TOP_K_OPENAI}** with Nano Banana
5. Download single ZIP (`profile.webp` + `gallery1.webp` ... `gallery10.webp`)
"""
    )
    st.divider()
    st.markdown("**Models used**")
    st.markdown("- Profile output review / ranking: `gpt-4o`")
    st.markdown(f"- Profile + gallery regeneration: `{GEMINI_MODEL}`")

# ---- File uploader ----
uploaded_files = st.file_uploader(
    f"Upload {MIN_IMAGES}-{MAX_IMAGES} person photos (JPEG / PNG / WebP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) < MIN_IMAGES or len(uploaded_files) > MAX_IMAGES:
        st.error(f"Please upload between {MIN_IMAGES} and {MAX_IMAGES} images. You uploaded {len(uploaded_files)}.")
        st.stop()

    st.subheader("Uploaded Images")
    cols = st.columns(min(len(uploaded_files), MAX_IMAGES))
    for i, f in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            try:
                f.seek(0)
                img_bytes = f.read()
                f.seek(0)
                st.image(img_bytes, caption=f.name, width="stretch")
            except Exception as e:
                st.warning(f"Cannot preview {f.name}: {e}")

NUM_PROFILE_CANDIDATES = 3

# =====================================================================
# PHASE 1 — User clicks "Generate Image".
# Validate inputs, run input guardrail, generate 3 profile candidates
# (no output/composition guardrails — the user chooses the best one).
# =====================================================================
if st.button("Generate Image", type="primary", width="stretch"):
    # Reset any prior run.
    st.session_state.result_data = None
    st.session_state.profile_candidates = None
    st.session_state.pipeline_inputs = None
    st.session_state.pipeline_phase = "idle"

    if not artist_name.strip():
        st.error("Please enter an artist name.")
        st.stop()
    if not uploaded_files:
        st.error(f"Please upload {MIN_IMAGES}-{MAX_IMAGES} reference images.")
        st.stop()
    if len(uploaded_files) < MIN_IMAGES or len(uploaded_files) > MAX_IMAGES:
        st.error(f"Please upload between {MIN_IMAGES} and {MAX_IMAGES} images.")
        st.stop()

    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        st.error("Missing `SERPAPI_API_KEY` in `.env`.")
        st.stop()

    artist_folder = slugify_artist_name(artist_name)
    custom_prompt = optional_tags.strip() or None

    images: list[tuple[bytes, str]] = []
    for f in uploaded_files:
        f.seek(0)
        images.append((f.read(), mime_from_name(f.name)))

    # Step 1: input check (lightweight quality gate on the uploads).
    with st.status("Step 1 — Input quality check...", expanded=True) as status:
        try:
            input_check = input_guardrail(images)
        except Exception as e:
            st.error(f"Input guardrail failed: {e}")
            st.stop()
        if not input_check.get("is_approved"):
            status.update(label="Step 1 — Input guardrail failed", state="error")
            st.error("Some uploaded images failed quality checks.")
            for issue in input_check.get("issues", []):
                idx = issue.get("image_index", "?")
                problems = issue.get("problems", [])
                file_name = (
                    uploaded_files[idx].name
                    if isinstance(idx, int) and idx < len(uploaded_files)
                    else f"Image {idx}"
                )
                for p in problems:
                    st.write(f"- **{file_name}**: {p}")
            st.stop()
        status.update(label="Step 1 — Input guardrail passed", state="complete")

    # Step 2: generate N profile candidates. Each candidate is independently
    # validated by a Gemini-based "is this Zakir Khan?" check. If the candidate
    # IS Zakir Khan, regenerate (up to ZAKIR_MAX_RETRIES per candidate).
    # Worst case: NUM_PROFILE_CANDIDATES * (1 + ZAKIR_MAX_RETRIES) generations.
    ZAKIR_MAX_RETRIES = 3
    ZAKIR_REGEN_FEEDBACK = (
        "The previous attempt generated a person who looks like Zakir Khan (the "
        "Indian stand-up comedian from the composition reference image) instead "
        "of the actual subject from the subject photos. Regenerate using the "
        "SUBJECT PHOTOS as the identity source. The person in the output must "
        "be the user-provided subject — NOT Zakir Khan. Do NOT copy Zakir Khan's "
        "face, beard, hair, ethnicity, or any visual feature from the composition "
        "reference. Keep the designer face grid: hairline at Y=74, chin at Y=303, "
        "face centred at X=382.5 on a 765×480 canvas."
    )

    profile_candidates: list[bytes] = []
    with st.status(
        f"Step 2 — Generating {NUM_PROFILE_CANDIDATES} profile candidates "
        f"(Gemini → align → background uniform; Zakir check + up to {ZAKIR_MAX_RETRIES} retries each)...",
        expanded=True,
    ) as status:
        for i in range(1, NUM_PROFILE_CANDIDATES + 1):
            st.markdown(f"**Candidate {i}/{NUM_PROFILE_CANDIDATES}**")
            candidate_img: bytes | None = None
            feedback_for_next = None

            for attempt in range(1, ZAKIR_MAX_RETRIES + 2):  # 1 initial + up to N retries
                st.write(
                    f"  Attempt {attempt}/{ZAKIR_MAX_RETRIES + 1} — generating..."
                )
                try:
                    candidate_img = generate_linkedin_image(
                        images, feedback_for_next, custom_prompt
                    )
                except Exception as e:
                    st.warning(f"  Generation failed: {e}")
                    candidate_img = None
                    break

                # Gemini-based Zakir identity check on the freshly generated image
                # (no reference image — Gemini decides from prior knowledge).
                try:
                    check = gemini_zakir_check(candidate_img)
                except Exception as e:
                    st.warning(
                        f"  Zakir check errored ({e}) — treating as not-Zakir for safety."
                    )
                    check = {"is_zakir": False, "explanation": None}

                explanation = check.get("explanation") or ""
                if not check.get("is_zakir"):
                    st.write(
                        f"  ✅ Not Zakir Khan — accepted "
                        + (f"({explanation})" if explanation else "")
                    )
                    break

                # Is Zakir — log and retry if budget remains.
                st.write(
                    f"  ❌ Looks like Zakir Khan "
                    + (f"({explanation})" if explanation else "")
                )
                if attempt > ZAKIR_MAX_RETRIES:
                    st.warning(
                        f"  Reached max retries ({ZAKIR_MAX_RETRIES}) — accepting "
                        "as best-effort."
                    )
                    break
                feedback_for_next = ZAKIR_REGEN_FEEDBACK

            if candidate_img is not None:
                with st.spinner(f"  Aligning candidate {i} (face guides, keep Gemini background)..."):
                    aligned_bytes, aligned_ok, align_err = apply_profile_alignment(
                        candidate_img
                    )
                if aligned_ok:
                    if align_err and "center-crop fallback" in align_err:
                        st.write(f"  ⚠ Partial align: {align_err}")
                    else:
                        try:
                            from profile_align import format_guide_errors, is_face_on_guides

                            _al = Image.open(io.BytesIO(aligned_bytes))
                            if is_face_on_guides(_al):
                                st.write("  ✅ Face on guides (765×480)")
                            else:
                                st.write(
                                    f"  ⚠ Face still off guides: {format_guide_errors(_al)}"
                                )
                        except Exception:
                            st.write("  ✅ Face aligned (765×480)")
                    profile_candidates.append(aligned_bytes)
                    preview_bytes = aligned_bytes
                else:
                    st.warning(
                        "  ⚠ Alignment failed — showing raw Gemini output for this candidate. "
                        f"**Reason:** {align_err or 'unknown'}"
                    )
                    profile_candidates.append(candidate_img)
                    preview_bytes = candidate_img
                st.image(
                    profile_image_for_display(preview_bytes),
                    caption=(
                        f"Candidate {i} preview — cyan: Y={GUIDE_TOP_Y}, Y={GUIDE_BOTTOM_Y}, "
                        f"center X={GUIDE_CENTER_X:.0f}"
                    ),
                    width="stretch",
                )
            else:
                st.warning(f"Candidate {i} produced no image.")

        if not profile_candidates:
            status.update(
                label=f"Step 2 — All {NUM_PROFILE_CANDIDATES} candidate generations failed",
                state="error",
            )
            st.error("All profile candidate generations failed. Try again.")
            st.stop()

        # Step 2b: Gemini pass on all candidates — seamless background (no double frame).
        if PROFILE_UNIFORM_BACKGROUND:
            st.markdown("**Background uniforming (Gemini)**")
            uniformed_candidates: list[bytes] = []
            for j, cand_bytes in enumerate(profile_candidates, 1):
                with st.spinner(
                    f"  Candidate {j}/{len(profile_candidates)} — seamless background..."
                ):
                    try:
                        fixed = _profile_png_bytes(
                            uniform_profile_background_gemini(cand_bytes)
                        )
                        uniformed_candidates.append(fixed)
                        try:
                            from profile_align import format_guide_errors, is_face_on_guides

                            _u = Image.open(io.BytesIO(fixed))
                            if is_face_on_guides(_u):
                                st.write(
                                    f"  ✅ Candidate {j} background unified (face on guides)"
                                )
                            else:
                                st.write(
                                    f"  ✅ Candidate {j} background unified "
                                    f"(face may be slightly off guides — no re-align to avoid borders: "
                                    f"{format_guide_errors(_u)})"
                                )
                        except Exception:
                            st.write(f"  ✅ Candidate {j} background unified")
                    except Exception as e:
                        st.warning(
                            f"  ⚠ Candidate {j} background uniform failed ({e}) "
                            "— using aligned version"
                        )
                        uniformed_candidates.append(_profile_png_bytes(cand_bytes))
                st.image(
                    profile_image_for_display(uniformed_candidates[-1]),
                    caption=(
                        f"Candidate {j} after background uniform — "
                        f"Y={GUIDE_TOP_Y}, Y={GUIDE_BOTTOM_Y}, X={GUIDE_CENTER_X:.0f}"
                    ),
                    width="stretch",
                )
            profile_candidates = uniformed_candidates

        status.update(
            label=f"Step 2 — Generated {len(profile_candidates)}/{NUM_PROFILE_CANDIDATES} candidates",
            state="complete",
        )

    # Cache inputs + candidates for the next rerun, then move to selection phase.
    st.session_state.profile_candidates = profile_candidates
    st.session_state.pipeline_inputs = {
        "images": images,
        "artist_name": artist_name.strip(),
        "artist_folder": artist_folder,
        "custom_prompt": custom_prompt,
        "api_key": api_key,
    }
    st.session_state.pipeline_phase = "awaiting_selection"
    st.rerun()


# =====================================================================
# PHASE 2 — Candidate selection.
# Show all generated profile candidates and let the user pick one.
# =====================================================================
if (
    st.session_state.pipeline_phase == "awaiting_selection"
    and st.session_state.profile_candidates
):
    st.divider()
    st.subheader("Pick your profile image")
    guide_caption = (
        f" — cyan: Y={GUIDE_TOP_Y}, Y={GUIDE_BOTTOM_Y}, center X={GUIDE_CENTER_X:.0f}"
        if PROFILE_SHOW_GUIDE_LINES
        else ""
    )
    st.caption(
        f"Generated {len(st.session_state.profile_candidates)} candidate profile images "
        "(align once → Gemini background uniform only; no re-align after uniform). "
        "Choose the one you like best — it becomes profile.webp and the gallery face anchor."
        + guide_caption
    )

    candidates = st.session_state.profile_candidates
    cols = st.columns(len(candidates))
    for i, img_bytes in enumerate(candidates):
        with cols[i]:
            st.image(
                profile_image_for_display(img_bytes),
                caption=f"Candidate {i + 1}{guide_caption}",
                width="stretch",
            )

    selected_index = st.radio(
        "Select a candidate",
        options=list(range(len(candidates))),
        format_func=lambda i: f"Candidate {i + 1}",
        horizontal=True,
        key="profile_candidate_selection",
    )

    if st.button("Continue with this profile", type="primary", width="stretch"):
        st.session_state.pipeline_inputs["selected_profile_image"] = candidates[selected_index]
        st.session_state.pipeline_phase = "running_gallery"
        # Free the unselected candidates to keep memory bounded.
        st.session_state.profile_candidates = None
        gc.collect()
        st.rerun()


# =====================================================================
# PHASE 3 — Gallery pipeline (Steps 3-5) using the selected profile.
# =====================================================================
if st.session_state.pipeline_phase == "running_gallery" and st.session_state.pipeline_inputs:
    inputs = st.session_state.pipeline_inputs
    images = inputs["images"]
    artist_name_resolved = inputs["artist_name"]
    artist_folder = inputs["artist_folder"]
    custom_prompt = inputs["custom_prompt"]
    api_key = inputs["api_key"]
    generated_image = inputs["selected_profile_image"]

    profile_webp_bytes, profile_within_limit = prepare_profile_webp(generated_image)
    if not profile_webp_bytes:
        st.error("Failed to prepare profile.webp.")
        st.session_state.pipeline_phase = "idle"
        st.stop()

    # Step 3: fetch SerpApi pool
    with st.status("Step 3 — Fetching gallery references from SerpApi...", expanded=False) as status:
        try:
            serp_candidates = fetch_serpapi_image_candidates(
                artist_name_resolved,
                api_key,
                target_aspect=GALLERY_WIDTH / GALLERY_HEIGHT,
                target_count=REFERENCE_IMAGE_POOL_SIZE,
            )
        except Exception as e:
            st.error(f"SerpApi fetch failed: {e}")
            st.stop()
        rows: list[dict] = []
        for pool_idx, c in enumerate(serp_candidates):
            data = None
            err = None
            for url in (c.get("link"), c.get("thumbnail_link")):
                if not url:
                    continue
                try:
                    cand_bytes = download_image_bytes(url)
                    ok, decode_err = is_valid_image_bytes(cand_bytes) if cand_bytes else (False, "No data")
                    if cand_bytes and len(cand_bytes) > 500 and ok:
                        data = _downscale_for_pool(cand_bytes)
                        break
                    err = decode_err or "Downloaded bytes are not a valid image"
                except Exception as ex:
                    err = str(ex)
                    data = None
            rows.append(
                {
                    "pool_idx": pool_idx,
                    "index": pool_idx + 1,
                    "title": c.get("title", ""),
                    "link": c.get("link", ""),
                    "width": c.get("width"),
                    "height": c.get("height"),
                    "aspect_ratio": c.get("aspect_ratio"),
                    "bytes": data,
                    "error": err if not data else None,
                }
            )
        if not rows:
            st.error("No gallery references found from SerpApi.")
            st.stop()
        status.update(label="Step 3 — SerpApi references fetched", state="complete")

    # Step 4: rank top 10
    with st.status("Step 4 — Ranking top 10 gallery references...", expanded=False) as status:
        ta = GALLERY_WIDTH / GALLERY_HEIGHT
        try:
            top_indices, rank_err = rank_reference_images_openai(
                rows,
                artist_name_resolved,
                ta,
                top_k=REFERENCE_TOP_K_OPENAI,
            )
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.stop()
        if rank_err:
            st.warning(rank_err)
        if len(top_indices) < REFERENCE_TOP_K_OPENAI:
            st.error(
                f"Ranking produced only {len(top_indices)} images; expected {REFERENCE_TOP_K_OPENAI}."
            )
            st.stop()

        # Free non-selected image payloads — they account for ~half of pool memory.
        top_set = set(top_indices)
        for i, row in enumerate(rows):
            if i not in top_set:
                row["bytes"] = None
        gc.collect()

        status.update(label="Step 4 — Top 10 selected", state="complete")

    # Step 5: regenerate top 10 (serial, no retry)
    gallery_outputs: list[tuple[str, bytes, bool]] = []
    with st.status("Step 5 — Regenerating final 10 gallery images...", expanded=True):
        used_urls_initial: set[str] = set()
        progress = st.progress(0, text=f"Processing 0/{REFERENCE_TOP_K_OPENAI}")
        for j, pool_idx in enumerate(top_indices, start=1):
            if pool_idx < 0 or pool_idx >= len(rows):
                st.error(f"Gallery {j} failed: invalid ranked index.")
                st.stop()
            src = rows[pool_idx].get("bytes")
            if not src:
                st.error(f"Gallery {j} failed: missing source bytes.")
                st.stop()
            try:
                png_banner = regenerate_gallery_image_nano_banana(src)
                webp_b, ok = prepare_gallery_webp(png_banner)
            except Exception as ex:
                st.error(f"Gallery {j} failed: {ex}")
                st.stop()
            if not webp_b:
                st.error(f"Gallery {j} failed: WebP export failed.")
                st.stop()
            gallery_outputs.append((f"gallery{j}.webp", webp_b, ok))
            # Capture the source URL so a follow-up "Fetch 10 more" batch
            # can de-duplicate against what we already used.
            row_link = rows[pool_idx].get("link") or ""
            if row_link:
                used_urls_initial.add(row_link)
            progress.progress(
                int((j / REFERENCE_TOP_K_OPENAI) * 100),
                text=f"Processing {j}/{REFERENCE_TOP_K_OPENAI}",
            )

            # Free the processed source + transient PIL/numpy objects before the
            # next iteration so peak memory stays bounded.
            rows[pool_idx]["bytes"] = None
            del png_banner, src
            gc.collect()

    # Build zip output
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{artist_folder}/profile.webp", profile_webp_bytes)
        for name, data, _ in gallery_outputs:
            zip_file.writestr(f"{artist_folder}/{name}", data)

    st.session_state.result_data = {
        "profile_webp_bytes": profile_webp_bytes,
        "profile_within_limit": profile_within_limit,
        "gallery_outputs": gallery_outputs,
        "zip_bytes": zip_buffer.getvalue(),
        "artist_folder": artist_folder,
    }

    # Release the heavy SerpApi pool; keep pipeline_inputs around so the user
    # can click "Fetch 10 more gallery images" later without re-entering inputs.
    # (pipeline_inputs is small — just the artist name, slug, api key, and
    # cached uploaded image bytes — kilobytes, not megabytes.)
    st.session_state.reference_pool_rows = None
    # Track URLs already used so the "Fetch 10 more" button skips duplicates.
    st.session_state.used_gallery_urls = used_urls_initial
    st.session_state.pipeline_phase = "done"
    rows = None
    serp_candidates = None
    gc.collect()
    st.rerun()

result_data = st.session_state.result_data
if result_data:
    st.divider()
    st.subheader("Final Output")

    profile_webp_bytes = result_data["profile_webp_bytes"]
    profile_within_limit = result_data["profile_within_limit"]
    gallery_outputs = result_data["gallery_outputs"]
    zip_bytes = result_data["zip_bytes"]
    artist_folder = result_data["artist_folder"]

    if profile_within_limit:
        st.success(f"`profile.webp` meets size target (<= {PROFILE_MAX_SIZE_KB} KB).")
    else:
        actual_kb = len(profile_webp_bytes) / 1024
        st.warning(f"`profile.webp` is best-effort at {actual_kb:.1f} KB (target <= {PROFILE_MAX_SIZE_KB} KB).")

    st.caption("Showing only final generated assets.")
    preview_cols = st.columns(3)
    all_outputs = [("profile.webp", profile_webp_bytes)] + [(name, data) for name, data, _ in gallery_outputs]
    for idx, (name, data) in enumerate(all_outputs):
        with preview_cols[idx % 3]:
            if name == "profile.webp":
                preview_img = profile_image_for_display(data)
                cap = (
                    f"{name} ({len(data)/1024:.1f} KB) — cyan: Y={GUIDE_TOP_Y}, "
                    f"Y={GUIDE_BOTTOM_Y}, center X={GUIDE_CENTER_X:.0f}"
                )
            else:
                preview_img = Image.open(io.BytesIO(data))
                cap = f"{name} ({len(data)/1024:.1f} KB)"
            st.image(preview_img, caption=cap, width="stretch")

    st.download_button(
        label="Download All Images (ZIP)",
        data=zip_bytes,
        file_name=f"{artist_folder}_images.zip",
        mime="application/zip",
        width="stretch",
    )

    # ── Fetch 10 more gallery images ─────────────────────────────────────────
    # Runs the same SerpApi → OpenAI rerank → Nano Banana banner pipeline on a
    # fresh batch and appends the new images to result_data.gallery_outputs.
    # Available only when pipeline_inputs is still cached (i.e. didn't reset).
    st.divider()
    if st.session_state.pipeline_inputs is None:
        st.caption(
            "Inputs were cleared. To fetch more gallery images, regenerate the profile first."
        )
    else:
        already_count = len(gallery_outputs)
        if st.button(
            f"Fetch {EXTRA_GALLERY_BATCH_SIZE} more gallery images",
            type="secondary",
            width="stretch",
            key="btn_fetch_more_gallery",
        ):
            inputs = st.session_state.pipeline_inputs
            artist_name_resolved = inputs["artist_name"]
            api_key = inputs["api_key"]
            already_used: set[str] = set(st.session_state.get("used_gallery_urls") or [])
            extra_new_outputs: list[tuple[str, bytes, bool]] = []

            # Step 3 (extra) — fetch a fresh SerpApi pool
            with st.status(
                f"Fetching {REFERENCE_IMAGE_POOL_SIZE} more references from SerpApi...",
                expanded=False,
            ) as status:
                try:
                    extra_candidates = fetch_serpapi_image_candidates(
                        artist_name_resolved,
                        api_key,
                        target_aspect=GALLERY_WIDTH / GALLERY_HEIGHT,
                        target_count=REFERENCE_IMAGE_POOL_SIZE,
                    )
                except Exception as e:
                    st.error(f"SerpApi fetch failed: {e}")
                    st.stop()
                extra_rows: list[dict] = []
                for pool_idx, c in enumerate(extra_candidates):
                    link = c.get("link") or ""
                    # Skip URLs already used in the initial batch.
                    if link and link in already_used:
                        continue
                    data = None
                    for url in (c.get("link"), c.get("thumbnail_link")):
                        if not url:
                            continue
                        try:
                            cand_bytes = download_image_bytes(url)
                            ok, _ = is_valid_image_bytes(cand_bytes) if cand_bytes else (False, "No data")
                            if cand_bytes and len(cand_bytes) > 500 and ok:
                                data = _downscale_for_pool(cand_bytes)
                                break
                        except Exception:
                            data = None
                    extra_rows.append({
                        "pool_idx": pool_idx,
                        "index": pool_idx + 1,
                        "title": c.get("title", ""),
                        "link": link,
                        "width": c.get("width"),
                        "height": c.get("height"),
                        "aspect_ratio": c.get("aspect_ratio"),
                        "bytes": data,
                    })
                if not extra_rows:
                    st.error("No additional references found (all were duplicates of the first batch).")
                    st.stop()
                status.update(
                    label=f"Fetched {len(extra_rows)} additional references from SerpApi",
                    state="complete",
                )

            # Step 4 (extra) — rank top N via OpenAI
            with st.status(
                f"Ranking top {EXTRA_GALLERY_BATCH_SIZE} of the new references...",
                expanded=False,
            ) as status:
                try:
                    extra_top_indices, extra_rank_err = rank_reference_images_openai(
                        extra_rows,
                        artist_name_resolved,
                        GALLERY_WIDTH / GALLERY_HEIGHT,
                        top_k=EXTRA_GALLERY_BATCH_SIZE,
                    )
                except Exception as e:
                    st.error(f"Ranking failed: {e}")
                    st.stop()
                if extra_rank_err:
                    st.warning(extra_rank_err)
                if not extra_top_indices:
                    st.error("Ranker returned no usable images.")
                    st.stop()
                # Free non-selected payloads.
                top_set_extra = set(extra_top_indices)
                for i, row in enumerate(extra_rows):
                    if i not in top_set_extra:
                        row["bytes"] = None
                gc.collect()
                status.update(
                    label=f"Selected {len(extra_top_indices)} additional references",
                    state="complete",
                )

            # Step 5 (extra) — regenerate banners
            with st.status("Regenerating additional gallery banners...", expanded=True):
                progress = st.progress(0, text=f"Processing 0/{len(extra_top_indices)}")
                next_name_start = already_count + 1
                for j, pool_idx in enumerate(extra_top_indices, start=1):
                    if pool_idx < 0 or pool_idx >= len(extra_rows):
                        st.warning(f"Extra gallery {j} skipped: invalid index.")
                        continue
                    src = extra_rows[pool_idx].get("bytes")
                    if not src:
                        st.warning(f"Extra gallery {j} skipped: missing source bytes.")
                        continue
                    try:
                        png_banner = regenerate_gallery_image_nano_banana(src)
                        webp_b, ok = prepare_gallery_webp(png_banner)
                    except Exception as ex:
                        st.warning(f"Extra gallery {j} failed: {ex}")
                        continue
                    if not webp_b:
                        st.warning(f"Extra gallery {j} skipped: WebP export failed.")
                        continue
                    fname = f"gallery{next_name_start + len(extra_new_outputs)}.webp"
                    extra_new_outputs.append((fname, webp_b, ok))
                    link = extra_rows[pool_idx].get("link") or ""
                    if link:
                        already_used.add(link)
                    progress.progress(
                        int((j / len(extra_top_indices)) * 100),
                        text=f"Processing {j}/{len(extra_top_indices)}",
                    )
                    extra_rows[pool_idx]["bytes"] = None
                    del png_banner, src
                    gc.collect()

            if not extra_new_outputs:
                st.error("No additional gallery images were produced.")
                st.stop()

            # Merge into result_data and rebuild ZIP.
            merged_outputs = list(gallery_outputs) + extra_new_outputs
            new_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(new_zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{artist_folder}/profile.webp", profile_webp_bytes)
                for name, data, _ in merged_outputs:
                    zf.writestr(f"{artist_folder}/{name}", data)

            st.session_state.result_data = {
                **result_data,
                "gallery_outputs": merged_outputs,
                "zip_bytes": new_zip_buffer.getvalue(),
            }
            st.session_state.used_gallery_urls = already_used

            extra_rows = None
            extra_candidates = None
            gc.collect()
            st.success(
                f"Added {len(extra_new_outputs)} new gallery image(s). Total now: {len(merged_outputs)}."
            )
            st.rerun()

    # ── Refine gallery images with client feedback ───────────────────────────
    # Each gallery image is processed one-by-one:
    #   1. OpenAI looks at the image + client's general feedback → outputs a
    #      specific actionable instruction for THIS image.
    #   2. Gemini (Nano Banana) edits the image using that instruction.
    #   3. The refined image replaces the original. If Gemini refuses (common
    #      for real-person photo edits), the original is kept and a warning is
    #      shown for that image.
    st.divider()
    st.subheader("Refine gallery images with client feedback")
    st.caption(
        "Describe what the client wants changed across the gallery (e.g., "
        "\"remove text/logos from the background\", \"make the lighting warmer\"). "
        "Each image will be analysed individually by OpenAI to produce a specific "
        "fix, then refined by Gemini one image at a time."
    )

    client_feedback_text = st.text_area(
        "Client feedback",
        placeholder='e.g., "There is visible text on the backdrop in some images. '
                    'Please remove any banners, signage, or logos from the background '
                    'while keeping the person and scene unchanged."',
        key="gallery_refine_feedback",
        height=100,
    )

    if st.button(
        "Refine all gallery images",
        type="secondary",
        width="stretch",
        key="btn_refine_gallery",
        disabled=not client_feedback_text.strip(),
    ):
        feedback_clean = client_feedback_text.strip()
        if not feedback_clean:
            st.warning("Please enter the client's feedback first.")
        else:
            refined_outputs: list[tuple[str, bytes, bool]] = []
            n_total = len(gallery_outputs)
            n_refined = 0
            n_skipped = 0
            n_failed = 0

            with st.status(
                f"Refining {n_total} gallery image(s) one at a time...",
                expanded=True,
            ) as status:
                progress = st.progress(0, text=f"0/{n_total}")

                for idx, (name, original_bytes, original_within_limit) in enumerate(gallery_outputs, start=1):
                    st.write(f"**{name} ({idx}/{n_total})**")

                    # Step A: OpenAI per-image analysis
                    try:
                        analysis = analyze_gallery_image_for_feedback(
                            original_bytes, feedback_clean
                        )
                    except Exception as e:
                        st.warning(f"  ⚠ OpenAI analysis failed for {name}: {e} — keeping original.")
                        refined_outputs.append((name, original_bytes, original_within_limit))
                        n_failed += 1
                        progress.progress(int((idx / n_total) * 100), text=f"{idx}/{n_total}")
                        gc.collect()
                        continue

                    needs_refine = bool(analysis.get("needs_refinement"))
                    instruction = analysis.get("instruction") or ""

                    if not needs_refine or not instruction.strip():
                        st.write(f"  • No refinement needed for {name} — keeping original.")
                        refined_outputs.append((name, original_bytes, original_within_limit))
                        n_skipped += 1
                        progress.progress(int((idx / n_total) * 100), text=f"{idx}/{n_total}")
                        gc.collect()
                        continue

                    with st.expander(f"OpenAI instruction for {name}"):
                        st.code(instruction)

                    # Step B: Gemini edit using the instruction
                    try:
                        edited_raw = refine_gallery_image_with_feedback(original_bytes, instruction)
                    except Exception as e:
                        st.warning(
                            f"  ⚠ Gemini refused or failed to refine {name} ({e}) — keeping original."
                        )
                        refined_outputs.append((name, original_bytes, original_within_limit))
                        n_failed += 1
                        progress.progress(int((idx / n_total) * 100), text=f"{idx}/{n_total}")
                        gc.collect()
                        continue

                    # Step C: re-crop / re-encode to gallery WebP spec
                    try:
                        new_webp, new_within_limit = prepare_gallery_webp(edited_raw)
                    except Exception as e:
                        st.warning(
                            f"  ⚠ WebP export failed for refined {name} ({e}) — keeping original."
                        )
                        refined_outputs.append((name, original_bytes, original_within_limit))
                        n_failed += 1
                        progress.progress(int((idx / n_total) * 100), text=f"{idx}/{n_total}")
                        gc.collect()
                        continue

                    if not new_webp:
                        st.warning(f"  ⚠ Empty refined WebP for {name} — keeping original.")
                        refined_outputs.append((name, original_bytes, original_within_limit))
                        n_failed += 1
                    else:
                        st.write(f"  ✓ Refined {name} successfully.")
                        refined_outputs.append((name, new_webp, new_within_limit))
                        n_refined += 1

                    del edited_raw
                    progress.progress(int((idx / n_total) * 100), text=f"{idx}/{n_total}")
                    gc.collect()

                status.update(
                    label=(
                        f"Refinement complete — refined: {n_refined}, "
                        f"skipped (no issue): {n_skipped}, kept original (failed): {n_failed}"
                    ),
                    state="complete",
                )

            # Rebuild ZIP with the refined gallery + same profile.
            new_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(new_zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{artist_folder}/profile.webp", profile_webp_bytes)
                for name, data, _ in refined_outputs:
                    zf.writestr(f"{artist_folder}/{name}", data)

            st.session_state.result_data = {
                **result_data,
                "gallery_outputs": refined_outputs,
                "zip_bytes": new_zip_buffer.getvalue(),
            }
            st.success(
                f"Gallery refined. Refined {n_refined}, kept original {n_skipped + n_failed}."
            )
            gc.collect()
            st.rerun()
