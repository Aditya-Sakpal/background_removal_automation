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
from PIL import Image

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
GEMINI_MODEL = "gemini-2.5-flash-image"  # Nano Banana
MIN_IMAGES = 2
MAX_IMAGES = 3
MAX_RETRY = 3
STYLE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "1768222411704.jpg")
HEAD_SPACE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "head_space_reference.png")
GRADIENT_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "gradient_reference.jpg")
COMPOSITION_GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "netanyahu's image.jpg")
VIGNETTE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "zakir-khan_Comedian.png")
PROFILE_WIDTH = 765
PROFILE_HEIGHT = 480
PROFILE_MAX_SIZE_KB = 50
GALLERY_WIDTH = 1176
GALLERY_HEIGHT = 516
GALLERY_MAX_SIZE_KB = 50
# Google image pool → OpenAI picks best gallery-style references
REFERENCE_TOP_K_OPENAI = 10
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


def prepare_profile_webp(image_bytes: bytes) -> tuple[bytes, bool]:
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

    # 1. GOLD STANDARD composition reference (Netanyahu) — black bottom gradient anchor.
    gold_standard_ref = Image.open(COMPOSITION_GOLD_STANDARD_PATH).convert("RGB")
    contents.append(gold_standard_ref)
    contents.append(
        "The first image is the GOLD STANDARD COMPOSITION REFERENCE — this is exactly "
        "what the client wants the final output to look like in terms of composition. "
        "Do NOT copy the person, their clothing, the suit, the tie, the pin, or the "
        "background hue. Copy ONLY the composition pattern, which has THREE features "
        "you MUST replicate:\n"
        "  (a) HEADROOM — there is a generous band of empty background above the top "
        "of the hair, occupying roughly the upper 20-25% of the frame.\n"
        "  (b) BLACK BOTTOM GRADIENT — the background fades smoothly into a deep black "
        "band along the bottom edge of the frame, AND this black band extends UPWARD "
        "into the lower portion of the subject's clothing so the bottom of the jacket "
        "blends into the darkness with no sharp visible edge. The subject appears to "
        "merge into black at the bottom — there is no hard boundary between the "
        "subject's lower clothing and the frame bottom.\n"
        "  (c) HORIZONTAL CENTRING — the subject sits roughly centred with similar "
        "amounts of background on the left and the right."
    )

    # 2. VIGNETTE reference (Zakir Khan) — anchors radial vignette + headroom + centring.
    vignette_ref = Image.open(VIGNETTE_REFERENCE_PATH).convert("RGB")
    contents.append(vignette_ref)
    contents.append(
        "The second image is the VIGNETTE COMPOSITION REFERENCE. Do NOT copy the "
        "person, the maroon/red hue, the microphone, the blazer, or the clothing. "
        "Copy ONLY these three composition features:\n"
        "  (a) RADIAL VIGNETTE — there is a clear radial darkening from a brighter, "
        "softly-lit region behind/around the subject's head fading into noticeably "
        "darker corners and edges. The corners of the frame are distinctly darker "
        "than the area immediately behind the subject's head. This is a SOFT, smooth "
        "radial fade — not a hard mask, not a black border, not a vignette so heavy "
        "that the subject is silhouetted.\n"
        "  (b) HEADROOM — match the empty background above the top of the hair: "
        "roughly the upper 20-25% of the frame is empty background, the hair does "
        "NOT touch the top edge.\n"
        "  (c) HORIZONTAL CENTRING — the subject's head and shoulders sit roughly "
        "centred horizontally with similar amounts of background on the left and "
        "the right of the person."
    )

    # 3. Style reference image
    style_ref = Image.open(STYLE_REFERENCE_PATH).convert("RGB")
    contents.append(style_ref)
    contents.append(
        "The third image is a style reference for cohesive, polished portrait look. "
        "Use it for inspiration on lighting and how the subject sits naturally in the "
        "frame. The final background should combine the gold-standard pattern (clean "
        "colour at top → black at bottom) with the vignette reference's radial "
        "darkening at the corners and edges."
    )

    # 4. Head-space / framing reference — reinforces the headroom rule
    head_space_ref = Image.open(HEAD_SPACE_REFERENCE_PATH).convert("RGB")
    contents.append(head_space_ref)
    contents.append(
        "The fourth image reinforces the HEAD-SPACE rule. Match this much (or more) "
        "empty background above the head. Do NOT copy the person."
    )

    # 5. Gradient reference — reinforces the black-bottom gradient rule
    gradient_ref = Image.open(GRADIENT_REFERENCE_PATH).convert("RGB")
    contents.append(gradient_ref)
    contents.append(
        "The fifth image reinforces the BLACK BOTTOM GRADIENT rule. Match the "
        "top-colour-to-black vertical fade. Do NOT copy the person or the orange hue."
    )

    # 6. User's input photos
    for img_bytes, _ in images:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contents.append(pil_img)
    contents.append(
        "The remaining images are the subject photos. Use them as inspiration for the "
        "person's appearance — facial features, skin tone, hair, and clothing."
    )

    # 7. Generation prompt — short, conversational, no trigger words
    prompt_text = """Create a professional LinkedIn-style headshot inspired by the subject photos. The composition must satisfy ALL FOUR rules: (1) generous headroom, (2) black bottom gradient that blends into the subject's lower clothing, (3) horizontal centring, AND (4) a soft radial vignette darkening the corners — matching the gold-standard reference (first image) and the vignette reference (second image).

TOP PRIORITY — HEADROOM (most important rule, do not violate):
- There MUST be a large, obvious band of empty background ABOVE the top of the hair.
- The top of the hair must sit roughly 25-30% of the way DOWN from the top edge of the frame. The upper ~25-30% of the image is pure background, with NO part of the subject in it.
- Match the headroom shown in the gold-standard reference (first image) — that much space above the head, or more.
- The eyes should sit at or slightly BELOW the horizontal midline of the frame, NOT in the upper third. This pushes the head down and forces visible space above it.
- FORBIDDEN: hair touching or close to the top edge of the frame; head filling the upper portion of the frame; any cropping of the top of the head. If the top of the hair is anywhere in the upper 20% of the frame, the image is WRONG.

Style notes:
- A polished, photorealistic portrait with a warm, confident expression and a slight smile.
- The person's clothing should match what they wear in the subject photos (same outfit, same colours).
- Soft, diffused studio lighting with natural shadows under the chin and a subtle catch-light in the eyes.
- Render natural skin texture (subtle pores, fine details) — avoid an airbrushed or plastic look.
- Hair rendered with natural strands and volume, not a flat mass.

Background — vertical gradient + radial vignette (match gold-standard reference for the bottom gradient AND the vignette reference for the radial darkening):
- The TOP portion of the background is a clean, vivid colour that complements the clothing — for example a deep royal blue, teal, or polished grey-blue.
- The background transitions smoothly DOWNWARD into a rich, deep BLACK band across the bottom of the frame. The lower ~30-40% of the background fades into black, and this black band extends UPWARD into the lower portion of the subject's clothing so the bottom of the jacket dissolves into black with no hard visible edge.
- ON TOP OF the vertical gradient, apply a SOFT RADIAL VIGNETTE: the area immediately behind and around the subject's head is the brightest part of the background, and the corners and edges of the frame are noticeably darker. The vignette is smooth and subtle — no hard mask, no heavy black border, the subject is never silhouetted.
- The transitions (both vertical gradient and radial vignette) must be smooth — no banding, no hard lines.
- Net effect: the corners are dark, the bottom is darker still (fading into the subject's lower clothing), and the area behind the head has a softly-lit "halo" that draws the eye to the face.

Composition (landscape 3:2):
- The face takes around 25-35% of the frame's vertical height (smaller than you might default to — this is what creates the headroom).
- The subject is centred horizontally — equal background on left and right.
- Both shoulders are fully visible with a small margin, and the collar/neckline of the clothing is visible at the bottom.
- A slightly angled pose works well; avoid a straight-on stare.

Self-check before finishing: cover the lower half of your output with your hand — the upper half should show clear background above the head, with the head sitting mostly in the lower half of the frame. If the head dominates the upper half, regenerate with the camera pulled back further.

Output one bright, polished landscape headshot at 3:2."""

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
        {"type": "text", "text": "--- HEADROOM REFERENCE (the empty band above the head is what we want to match) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(vignette_ref_bytes, "image/png"), "detail": "high"},
        },
        {"type": "text", "text": "--- GENERATED LINKEDIN IMAGE (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a STRICT headroom reviewer for AI-generated LinkedIn profile headshots.

You are given TWO images:
1. HEADROOM REFERENCE — anchors the headroom rule.
2. The generated LinkedIn image to evaluate.

Check ONE thing only: is there enough empty background ABOVE the top of the subject's hair in the generated image, matching (or exceeding) the reference?

In the reference image, observe the gap between the top of the hair and the top edge of the frame — there is a clearly visible band of empty background occupying roughly the upper 20-25% of the frame.

PASS if the empty band above the hair in the generated image takes at least ~15% of the frame's vertical height.

FAIL if any of these are true:
- The hair touches or nearly touches the top edge of the frame.
- The top of the hair sits in the upper 10% of the frame.
- There is noticeably less headroom than the reference.
- The top of the head is cropped at all.

Do NOT evaluate anything else — not the person, not the clothing, not the background colour, not the lighting, not the vignette, not centering. Headroom ONLY.

If the rule fails, write a SHORT, SPECIFIC, ACTIONABLE instruction that can be fed directly back to the image generator to fix it.

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "actionable fix instruction if it failed"
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict single-criterion headroom reviewer. Respond only with JSON, no markdown fences."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=300,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


def bottom_gradient_guardrail(generated_image: bytes) -> dict:
    """
    Single-criterion guardrail: does the generated LinkedIn headshot have a black
    bottom gradient that blends UP into the subject's clothing? Anchored to the
    GOLD-STANDARD reference image.

    Returns {"is_approved": bool, "things_to_improve": str | False}.
    """
    client = get_openai_client()

    with open(COMPOSITION_GOLD_STANDARD_PATH, "rb") as f:
        gold_standard_bytes = f.read()

    image_content = [
        {"type": "text", "text": "--- BOTTOM-GRADIENT REFERENCE (the black band fading up into the jacket is what we want to match) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(gold_standard_bytes, "image/jpeg"), "detail": "high"},
        },
        {"type": "text", "text": "--- GENERATED LINKEDIN IMAGE (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a STRICT bottom-gradient reviewer for AI-generated LinkedIn profile headshots.

You are given TWO images:
1. BOTTOM-GRADIENT REFERENCE — anchors the rule.
2. The generated LinkedIn image to evaluate.

Check ONE thing only: does the generated image have the same black-bottom-gradient effect as the reference?

In the reference image, observe the bottom of the frame. TWO things are true and BOTH must be matched in the generated image:
(a) The background fades into a deep BLACK band along the bottom edge of the frame.
(b) That black band extends UPWARD into the lower portion of the subject — the bottom of the jacket/shirt is partially absorbed into the darkness, with NO sharp visible edge between the subject's clothing and the bottom of the frame. The subject appears to merge into the black at the bottom.

PASS only if BOTH (a) and (b) are clearly present.

FAIL if any of these are true:
- The background is a flat colour with no dark bottom band.
- Only a corner vignette is present, without a dark bottom band.
- The bottom of the subject's clothing is fully lit and sits cleanly above a visible edge instead of dissolving into black.
- The black band exists but does NOT extend up into the subject's clothing (i.e. there's a hard visible edge where the subject ends).

Do NOT evaluate anything else — not the person, not the clothing colour, not the background hue, not the lighting, not the vignette, not headroom, not centering. Bottom-gradient ONLY.

If the rule fails, write a SHORT, SPECIFIC, ACTIONABLE instruction that can be fed directly back to the image generator to fix it.

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "actionable fix instruction if it failed"
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict single-criterion bottom-gradient reviewer. Respond only with JSON, no markdown fences."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=300,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


def vignette_guardrail(generated_image: bytes) -> dict:
    """
    Single-criterion guardrail: does the generated LinkedIn headshot show a soft
    radial vignette that darkens the corners relative to the area behind the head?
    Anchored to the VIGNETTE reference image.

    Returns {"is_approved": bool, "things_to_improve": str | False}.
    """
    client = get_openai_client()

    with open(VIGNETTE_REFERENCE_PATH, "rb") as f:
        vignette_ref_bytes = f.read()

    image_content = [
        {"type": "text", "text": "--- VIGNETTE REFERENCE (the soft radial darkening of the corners is what we want to match) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(vignette_ref_bytes, "image/png"), "detail": "high"},
        },
        {"type": "text", "text": "--- GENERATED LINKEDIN IMAGE (to evaluate) ---"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_for_openai(generated_image, "image/png"), "detail": "high"},
        },
    ]

    prompt = """You are a STRICT vignette reviewer for AI-generated LinkedIn profile headshots.

You are given TWO images:
1. VIGNETTE REFERENCE — anchors the rule.
2. The generated LinkedIn image to evaluate.

Check ONE thing only: does the generated image have the same soft radial vignette effect as the reference?

In the reference image, observe how the area immediately behind/around the subject's head is the brightest part of the background, and the corners and edges of the frame are noticeably darker — a clear, smooth radial darkening from the centre outward.

PASS if the corners of the frame in the generated image are clearly darker than the area directly behind the subject's head, with a smooth fade and no visible banding.

FAIL if any of these are true:
- The background is a flat colour from edge to edge with no radial darkening.
- The corners are the same brightness as the centre.
- The vignette is so heavy or hard-edged that the subject is silhouetted, or a visible black border ring appears.
- The fade has visible banding or hard transitions.

Do NOT evaluate anything else — not the person, not the clothing, not the background colour, not the lighting style, not headroom, not centering, not the bottom gradient. Radial vignette ONLY.

If the rule fails, write a SHORT, SPECIFIC, ACTIONABLE instruction that can be fed directly back to the image generator to fix it.

Respond ONLY with valid JSON (no markdown fences) in this EXACT schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "actionable fix instruction if it failed"
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict single-criterion vignette reviewer. Respond only with JSON, no markdown fences."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=300,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


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


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.set_page_config(page_title="Engage4more Profile and Gallery Image Generator", page_icon="📸", layout="wide")

st.title("Engage4more Profile and Gallery Image Generator")
st.caption(
    "Enter artist name, optional profile tags, and upload 2-3 reference photos. "
    "One click generates profile + final top 10 gallery images."
)

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

# ---- Google Images API: reference pool (gallery aspect) ----
st.divider()
st.subheader(f"Reference image pool (~{REFERENCE_IMAGE_POOL_SIZE} via Google API)")
st.caption(
    f"Google API searches favour **live event / action** shots (stage, audience, candid, expressive). "
    f"Then **GPT-4o** picks **{REFERENCE_TOP_K_OPENAI}** using **four event types**, single clear subject, visible face, low text, "
    f"plus banner aspect ~{GALLERY_WIDTH}×{GALLERY_HEIGHT}. "
    "Set `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ID`, and `OPENAI_API_KEY` in `.env`."
)
if st.button(f"Fetch {REFERENCE_IMAGE_POOL_SIZE} images (Google API)", key="btn_google_pool", use_container_width=False):
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
                st.image(row["bytes"], caption=caption, use_container_width=True)
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
                    st.image(row["bytes"], caption=cap, use_container_width=True)
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
            use_container_width=False,
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
                        use_container_width=True,
                    )
                    st.download_button(
                        label=f"Download regen_{slot}.webp",
                        data=item["webp_bytes"],
                        file_name=f"gallery_regen_{slot}.webp",
                        mime="image/webp",
                        key=f"dl_regen_{slot}",
                        use_container_width=True,
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
                st.image(img_bytes, caption=f.name, use_container_width=True)
            except Exception as e:
                st.warning(f"Cannot preview {f.name}: {e}")

if st.button("Generate Image", type="primary", use_container_width=True):
    st.session_state.result_data = None

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

    # Step 1: input check
    with st.status("Step 1/5 — Input quality check...", expanded=True) as status:
        try:
            input_check = input_guardrail(images)
        except Exception as e:
            st.error(f"Input guardrail failed: {e}")
            st.stop()
        if not input_check.get("is_approved"):
            status.update(label="Step 1/5 — Input guardrail failed", state="error")
            st.error("Some uploaded images failed quality checks.")
            for issue in input_check.get("issues", []):
                idx = issue.get("image_index", "?")
                problems = issue.get("problems", [])
                file_name = uploaded_files[idx].name if isinstance(idx, int) and idx < len(uploaded_files) else f"Image {idx}"
                for p in problems:
                    st.write(f"- **{file_name}**: {p}")
            st.stop()
        status.update(label="Step 1/5 — Input guardrail passed", state="complete")

    # Step 2: profile generation with THREE independent staged guardrail loops.
    #
    # 1) Generate the initial image.
    # 2) Run output_guardrail (face / artifacts / appropriate) — hard fail aborts.
    # 3) Stage A — Headroom loop:  up to MAX_RETRY regenerations until headroom passes.
    # 4) Stage B — Bottom-gradient loop: up to MAX_RETRY regenerations until gradient passes.
    # 5) Stage C — Vignette loop: up to MAX_RETRY regenerations until vignette passes.
    #
    # Each stage independently retries with feedback specific to its own criterion.
    # If a stage exhausts its retries, the latest image is accepted and we move on.
    # The base prompt always enforces all rules, so later stages shouldn't normally
    # regress earlier ones — but if it happens, we accept best-effort.
    generated_image = None
    output_check = None

    def _run_single_guardrail(name: str, fn, *args) -> tuple[bool, str | None]:
        """Wrap a guardrail call so a failure inside it doesn't kill the pipeline."""
        try:
            result = fn(*args)
        except Exception as e:
            st.warning(f"{name} guardrail errored ({e}) — treating as passed.")
            return True, None
        passed = bool(result.get("is_approved"))
        feedback = result.get("things_to_improve") if not passed else None
        return passed, feedback if isinstance(feedback, str) else None

    def _run_stage(stage_name: str, guardrail_fn, current_image: bytes) -> bytes:
        """
        Run one composition stage with up to MAX_RETRY regenerations.
        Checks the current image first; only regenerates if the guardrail fails.
        Returns the final image (passed or best-effort).

        Reads `images` and `custom_prompt` from the enclosing module scope.
        """
        image = current_image
        for retry in range(1, MAX_RETRY + 1):
            passed, feedback = _run_single_guardrail(stage_name, guardrail_fn, image)
            st.write(
                f"  • {stage_name} guardrail (check {retry}/{MAX_RETRY}): "
                f"{'PASS' if passed else 'FAIL'}"
                + (f" — {feedback}" if feedback else "")
            )
            if passed:
                return image
            if retry == MAX_RETRY:
                st.warning(
                    f"{stage_name} guardrail still failing after {MAX_RETRY} attempts — "
                    "accepting the latest image as-is and moving on."
                )
                return image
            # Regenerate with this stage's specific feedback for the next check.
            stage_feedback = feedback or f"{stage_name} composition rule failed."
            st.write(f"  Regenerating with {stage_name.lower()} feedback...")
            try:
                image = generate_linkedin_image(images, stage_feedback, custom_prompt)
            except Exception as e:
                st.error(f"{stage_name} retry generation failed: {e}")
                return image
        return image

    with st.status("Step 2/5 — Generating profile image...", expanded=True) as status:
        # ── Initial generation ──────────────────────────────────────
        st.write("Generating initial profile image...")
        try:
            generated_image = generate_linkedin_image(images, None, custom_prompt)
        except Exception as e:
            st.error(f"Profile generation failed: {e}")
            st.stop()

        # ── Output guardrail (face / artifacts / appropriate) ───────
        try:
            output_check = output_guardrail(generated_image, images)
        except Exception as e:
            st.warning(f"Output guardrail errored ({e}) — treating profile as approved.")
            output_check = {"is_approved": True, "things_to_improve": False}
        if not output_check.get("is_approved"):
            st.error(
                f"Profile failed output guardrail: "
                f"{output_check.get('things_to_improve', 'unknown reason')}"
            )
            st.stop()

        # ── Stage A: Headroom (independent retry loop) ──────────────
        st.write("**Stage A — Headroom check**")
        generated_image = _run_stage("Headroom", headroom_guardrail, generated_image)

        # ── Stage B: Bottom gradient (independent retry loop) ───────
        st.write("**Stage B — Bottom-gradient check**")
        generated_image = _run_stage("Bottom gradient", bottom_gradient_guardrail, generated_image)

        # ── Stage C: Vignette (independent retry loop) ──────────────
        st.write("**Stage C — Vignette check**")
        generated_image = _run_stage("Vignette", vignette_guardrail, generated_image)

        if generated_image is None:
            st.error("Profile generation failed.")
            st.stop()
        status.update(label="Step 2/5 — Profile approved", state="complete")

    profile_webp_bytes, profile_within_limit = prepare_profile_webp(generated_image)
    if not profile_webp_bytes:
        st.error("Failed to prepare profile.webp.")
        st.stop()

    # Step 3: fetch SerpApi pool
    with st.status("Step 3/5 — Fetching gallery references from SerpApi...", expanded=False) as status:
        try:
            candidates = fetch_serpapi_image_candidates(
                artist_name.strip(),
                api_key,
                target_aspect=GALLERY_WIDTH / GALLERY_HEIGHT,
                target_count=REFERENCE_IMAGE_POOL_SIZE,
            )
        except Exception as e:
            st.error(f"SerpApi fetch failed: {e}")
            st.stop()
        rows: list[dict] = []
        for pool_idx, c in enumerate(candidates):
            data = None
            err = None
            for url in (c.get("link"), c.get("thumbnail_link")):
                if not url:
                    continue
                try:
                    candidate = download_image_bytes(url)
                    ok, decode_err = is_valid_image_bytes(candidate) if candidate else (False, "No data")
                    if candidate and len(candidate) > 500 and ok:
                        data = _downscale_for_pool(candidate)
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
        status.update(label="Step 3/5 — SerpApi references fetched", state="complete")

    # Step 4: rank top 10
    with st.status("Step 4/5 — Ranking top 10 gallery references...", expanded=False) as status:
        ta = GALLERY_WIDTH / GALLERY_HEIGHT
        try:
            top_indices, rank_err = rank_reference_images_openai(
                rows,
                artist_name.strip(),
                ta,
                top_k=REFERENCE_TOP_K_OPENAI,
            )
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.stop()
        if rank_err:
            st.warning(rank_err)
        if len(top_indices) < REFERENCE_TOP_K_OPENAI:
            st.error(f"Ranking produced only {len(top_indices)} images; expected {REFERENCE_TOP_K_OPENAI}.")
            st.stop()

        # Free non-selected image payloads — they account for ~half of pool memory.
        top_set = set(top_indices)
        for i, row in enumerate(rows):
            if i not in top_set:
                row["bytes"] = None
        gc.collect()

        status.update(label="Step 4/5 — Top 10 selected", state="complete")

    # Step 5: regenerate top 10 (serial, no retry)
    gallery_outputs: list[tuple[str, bytes, bool]] = []
    with st.status("Step 5/5 — Regenerating final 10 gallery images...", expanded=True):
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
            progress.progress(int((j / REFERENCE_TOP_K_OPENAI) * 100), text=f"Processing {j}/{REFERENCE_TOP_K_OPENAI}")

            # Free the processed source + any transient PIL/numpy objects before
            # the next iteration so peak memory stays bounded.
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

    # Release the pool + rows so Streamlit doesn't hold onto raw image bytes
    # across reruns. Final output is now fully in result_data + the zip buffer.
    st.session_state.reference_pool_rows = None
    rows = None
    candidates = None
    gc.collect()

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
            st.image(Image.open(io.BytesIO(data)), caption=f"{name} ({len(data)/1024:.1f} KB)", use_container_width=True)

    st.download_button(
        label="Download All Images (ZIP)",
        data=zip_bytes,
        file_name=f"{artist_folder}_images.zip",
        mime="application/zip",
        use_container_width=True,
    )
