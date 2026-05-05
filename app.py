import base64
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

from serpapi_images import (
    DEFAULT_TARGET_ASPECT,
    REFERENCE_IMAGE_POOL_SIZE,
    download_image_bytes,
    fetch_serpapi_image_candidates,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash-image"  # Nano Banana
MIN_IMAGES = 2
MAX_IMAGES = 3
MAX_RETRY = 3
STYLE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "1768222411704.jpg")
PROFILE_WIDTH = 765
PROFILE_HEIGHT = 480
PROFILE_MAX_SIZE_KB = 50
GALLERY_WIDTH = 1176
GALLERY_HEIGHT = 516
GALLERY_MAX_SIZE_KB = 50
# SerpApi pool → OpenAI picks best gallery-style references
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
    model: str = NANO_BANANA_REGEN_MODEL,
) -> bytes:
    """
    Regenerate a gallery candidate with Nano Banana while preserving identity/scene:
    - remove visible text/watermarks/logos
    - keep person and scene otherwise unchanged
    - return PNG bytes resized/cropped to exact 1176x516
    """
    client = get_gemini_client()

    src_img = Image.open(io.BytesIO(source_image_bytes)).convert("RGB")
    src_w, src_h = src_img.size
    src_ratio = src_w / max(src_h, 1)
    target_ratio = GALLERY_WIDTH / GALLERY_HEIGHT
    horizontal_focus = 0.5
    vertical_focus = 0.5
    if src_ratio < target_ratio:
        # Wider crop from portrait-ish inputs: bias higher to protect head/face framing.
        vertical_focus = 0.25

    prompt = """Edit the provided image with minimal changes.

STRICT REQUIREMENTS:
1. Keep the exact same person and identity (face, body, pose, clothing, skin tone, hairstyle).
2. Keep the same scene, camera angle, lighting, colors, and composition.
3. Remove ALL visible text, logo, caption, subtitle, lower-third, and watermark from every area of the image.
4. Text/watermark removal is mandatory. Replace removed regions naturally so no text fragments, ghosting, or logo artifacts remain.
5. Do not add/remove people or objects unless needed for text/watermark cleanup.
6. Do not stylize. Keep the result photorealistic and as close as possible to the source.

Return one high-quality horizontal image."""

    response = client.models.generate_content(
        model=model,
        contents=[src_img, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="21:9"),
        ),
    )

    edited_bytes: bytes | None = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            edited_bytes = part.inline_data.data
            break

    if not edited_bytes:
        raise RuntimeError("Nano Banana did not return an image payload.")

    edited_img = Image.open(io.BytesIO(edited_bytes)).convert("RGB")
    final_img = resize_and_crop_to_fill(
        edited_img,
        GALLERY_WIDTH,
        GALLERY_HEIGHT,
        horizontal_focus=horizontal_focus,
        vertical_focus=vertical_focus,
    )
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

    # 1. Style reference image first
    style_ref = Image.open(STYLE_REFERENCE_PATH).convert("RGB")
    contents.append(style_ref)
    contents.append(
        "Above is the STYLE REFERENCE image. The final output must match this "
        "exact style — notice how the person, lighting, and gradient background "
        "all feel like a single cohesive photograph, not a cutout pasted on a background."
    )

    # 2. User's input photos
    for img_bytes, _ in images:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contents.append(pil_img)
    contents.append(
        "Above are the SUBJECT photos. Study this person's facial features, "
        "skin tone, hair style, hair colour, and clothing carefully."
    )

    # 3. Generation prompt
    prompt_text = """Now generate a BRAND NEW professional LinkedIn profile headshot of the person from the subject photos. Do NOT simply cut out the person and paste them onto a background — that looks fake. Instead, CREATE a completely new portrait from scratch that captures the person's likeness.

CRITICAL REQUIREMENTS:

1. **Create, don't copy**: Generate a fresh portrait that captures the person's facial features, skin tone, hair, and likeness. Do NOT reuse or edit the input photos directly.

2. **Natural blending**: The person and background must look like they belong together — as if this photo was taken in a professional studio. The lighting on the person's face and body must match the ambient light of the background. Edges around hair, shoulders, and clothing must blend softly and naturally into the background with no harsh cutout lines.

3. **Background**: A smooth radial or linear gradient background. The gradient colours should complement the person's clothing. Dark edges fading to a lighter centre, or a rich colour that harmonises with their outfit (similar to the style reference image).

4. **Lighting**: Soft, diffused studio lighting with natural shadows under the chin and along the jawline. A subtle rim light or catch-light in the eyes to add depth. The lighting direction must be consistent between the person and the background glow.

5. **Framing — MUST show full head-to-shoulders WITH GENEROUS HEADROOM**: This is a MEDIUM HEADSHOT, NOT a face close-up. The frame MUST include:
   - Full head with SIGNIFICANT empty space ABOVE the hair — the top of the head must NOT be near the top edge of the frame. There must be a clear, generous band of background visible above the hair (roughly 20-30% of the frame's vertical height should be empty background above the head).
   - Full face from forehead to chin
   - Full neck
   - BOTH shoulders entirely visible within the frame
   - The top portion of the clothing (collar, neckline, shirt/blazer top) clearly visible
   The face should occupy roughly 30-40% of the frame vertically — NOT fill the entire frame. Think classic LinkedIn profile picture composition with airy, breathing room above the head: you can clearly see the shoulders, collar, AND a noticeable amount of background sky/gradient above the hair. Slightly angled pose, not straight-on.

6. **Expression**: A natural, warm, confident expression — a slight smile is ideal.

7. **Hyper-realistic skin and face detail**: This is the MOST IMPORTANT requirement. The face must be PHOTOREALISTIC at the level of a high-end DSLR portrait shot at f/2.8. Include:
   - Visible skin pores, fine texture, and micro-wrinkles appropriate to the person's age.
   - Natural subsurface scattering — skin should have warmth and translucency, especially around the ears, nose bridge, and cheeks.
   - Individual eyebrow hairs and eyelashes clearly defined.
   - Realistic iris detail with natural colour variation, visible pupil, and a sharp specular highlight.
   - Natural lip texture with subtle moisture/shine.
   - Stubble, beard grain, or clean-shaven smoothness matching the subject photos exactly.
   - Slight natural skin colour variation (mild redness on nose/cheeks, undertones) — do NOT make the skin a uniform flat tone.
   - Absolutely NO plastic, airbrushed, painted, or CGI look. If it looks like a video game character or a wax figure, it is WRONG. It must be indistinguishable from a real DSLR photograph.

8. **Hair detail**: Render individual hair strands with realistic flyaways and volume. Hair should have natural shine and light interaction, not look like a solid mass or a helmet.

9. **Clothing — DO NOT CHANGE**: The person MUST wear the EXACT same clothing as in the subject photos. Same colour, same style, same neckline, same pattern, same fabric. Do NOT replace, alter, or upgrade the clothing in any way. If the person is wearing a t-shirt, keep the t-shirt — do NOT swap it for a blazer or formal shirt. Render the clothing with visible fabric weave, proper folds, creases, and shadows.

10. **Final composition rule**: Highlight the face area using soft, natural shading as per the style reference. Keep the subject centered in the frame, ensure the full head is visible with no head cropping, and maintain a horizontal orientation suitable for a website profile image.
11. **Face-centering constraints (strict)**:
   - Position the eyes around the MIDDLE horizontal band of the frame (around the 50% vertical line), NOT the upper third — this pushes the head DOWN in the frame and leaves substantial empty space above the head.
   - Leave approximately 20-30% HEADROOM above the hair/top of head — this is a hard requirement. The top of the hair should sit roughly 20-30% down from the top edge of the frame, with clean gradient background filling that space above. A LinkedIn profile picture with the head pressed against the top of the frame is WRONG.
   - Do NOT crop hair, forehead, chin, jawline, neck, or shoulders.
   - Both shoulders must be fully inside the frame with a small margin on each side.
   - The subject should occupy roughly 50-60% of frame width — leaving visible gradient background on both sides of the person.
   - Avoid off-center framing unless needed for natural pose balance.

Output a single LANDSCAPE image at a 3:2 aspect ratio (wider than tall) at the highest possible quality and resolution. The full head and both shoulders MUST fit within this landscape frame with visible gradient background on left and right sides. The final image must look like it was taken by a professional photographer with a high-end camera — not generated by AI."""

    if custom_prompt:
        prompt_text += f"""

10. **User intent tags/prompts**: Incorporate these preferences where reasonable:
{custom_prompt}
"""

    if improvement_feedback:
        prompt_text += f"""

IMPORTANT — The previous generation was rejected. Here is the feedback on what to fix:
{improvement_feedback}

Please regenerate the image addressing ALL of the above issues while keeping all the original requirements."""

    contents.append(prompt_text)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="3:2"),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini did not return an image in its response.")


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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio="21:9"),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini did not return a gallery image in its response.")


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
            st.image(f, caption=f.name, use_container_width=True)

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

    # Step 2: profile generation (single attempt, no retry)
    generated_image = None
    output_check = None
    with st.status("Step 2/5 — Generating profile image...", expanded=True) as status:
        try:
            generated_image = generate_linkedin_image(images, None, custom_prompt)
        except Exception as e:
            st.error(f"Profile generation failed: {e}")
            st.stop()
        try:
            output_check = output_guardrail(generated_image, images)
        except Exception as e:
            st.warning(f"Output guardrail errored ({e}) — treating profile as approved.")
            output_check = {"is_approved": True, "things_to_improve": False}
        if output_check.get("is_approved"):
            status.update(label="Step 2/5 — Profile approved", state="complete")
        else:
            st.error(f"Profile failed output guardrail: {output_check.get('things_to_improve', 'unknown reason')}")
            st.stop()
        if generated_image is None:
            st.error("Profile generation failed.")
            st.stop()

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
                        data = candidate
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
