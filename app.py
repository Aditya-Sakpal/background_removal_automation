import base64
import io
import json
import os
import re
import zipfile

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
MAX_IMAGES = 5
MAX_RETRY = 3
STYLE_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "1768222411704.jpg")
PROFILE_WIDTH = 765
PROFILE_HEIGHT = 480
PROFILE_MAX_SIZE_KB = 50
GALLERY_WIDTH = 1176
GALLERY_HEIGHT = 516
GALLERY_MAX_SIZE_KB = 50
GALLERY_COUNT = 6
# SerpApi pool → OpenAI picks best gallery-style references
REFERENCE_TOP_K_OPENAI = 6
GALLERY_SCENES = [
    "motivational speaker delivering a keynote on stage with a clean conference backdrop",
    "speaker interacting with an engaged audience during a live session",
    "candid on-stage moment with expressive hand gestures and confident presence",
    "corporate event speaking moment with natural lighting and professional ambience",
    "comedy performance moment with expressive face and audience energy",
    "off-stage candid portrait in an event environment with natural professional look",
]


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


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()
    return json.loads(raw)


def mime_from_name(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")


def slugify_artist_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return cleaned or "artist_name"


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
- **One clear subject**: Exactly **one** dominant person (or one clear performer); not group collages or multi-panel images.
- **Face**: Face **clearly visible** and sharp enough (not tiny silhouette, not heavy occlusion).
- **Clarity**: Not extremely blurry, dark, or low-res.
- **Text**: Reject heavy overlaid text, meme text, news banners, big watermarks; tiny corner logos OK.

## Banner shape (secondary)
Prefer width ÷ height near **{target_ratio_str}** for a wide gallery strip; avoid extreme vertical crops if you have alternatives.

## Your task
Pick exactly **{top_k}** distinct indices that **best** match the **event/action** brief (Types 1–4) while passing the hard checks. Prefer TYPE 1–4 over plain studio portraits.

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
                "content": "You are a strict event-photography editor. Output only valid JSON as requested.",
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=600,
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


def resize_and_crop_to_fill(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resize while preserving aspect ratio, then center-crop to exact target size.
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

    left = max(0, (new_w - target_width) // 2)
    top = max(0, (new_h - target_height) // 2)
    right = left + target_width
    bottom = top + target_height

    return resized.crop((left, top, right, bottom))


def prepare_webp_with_constraints(
    image_bytes: bytes,
    width: int,
    height: int,
    max_size_kb: int,
) -> tuple[bytes, bool]:
    """
    Convert generated image to exact output spec (size + format constraints).

    Returns:
      (webp_bytes, is_within_size_limit)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = resize_and_crop_to_fill(img, width, height)
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
    )


def prepare_gallery_webp(image_bytes: bytes) -> tuple[bytes, bool]:
    return prepare_webp_with_constraints(
        image_bytes=image_bytes,
        width=GALLERY_WIDTH,
        height=GALLERY_HEIGHT,
        max_size_kb=GALLERY_MAX_SIZE_KB,
    )


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

5. **Framing**: Head and upper shoulders, slightly angled pose (not straight-on), classic LinkedIn headshot crop.

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
   - Keep both eyes around the upper-middle horizontal band of the frame.
   - Leave approximately 8-12% headroom above the hair/top of head.
   - Do not crop hair, forehead, chin, or jawline.
   - Keep the face as the visual focal point, with the subject occupying roughly 60-70% of frame width.
   - Avoid off-center framing unless needed for natural pose balance.

Output a single square image (1:1 aspect ratio) at the highest possible quality and resolution. The final image must look like it was taken by a professional photographer with a high-end camera — not generated by AI."""

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
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini did not return an image in its response.")


def generate_gallery_image(
    images: list[tuple[bytes, str]],
    scene_prompt: str,
    custom_prompt: str | None = None,
) -> bytes:
    """
    Generate one natural event/action gallery image for the artist.
    """
    client = get_gemini_client()

    contents: list = []

    style_ref = Image.open(STYLE_REFERENCE_PATH).convert("RGB")
    contents.append(style_ref)
    contents.append(
        "Above is a STYLE REFERENCE image for realism, clean lighting, and polish."
    )

    for img_bytes, _ in images:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contents.append(pil_img)
    contents.append(
        "Above are SUBJECT reference photos. Keep facial features, skin tone, hair, and clothing consistent."
    )

    prompt_text = f"""Generate one natural, professional event-style gallery image of this person.

SCENE REQUIREMENT:
{scene_prompt}

CRITICAL REQUIREMENTS:
1. Keep the same person identity and close resemblance to reference photos.
2. Preserve clothing style and colors (do not drastically change outfit).
3. Image should feel candid/action-oriented, not a studio headshot.
4. Natural event atmosphere with realistic lighting, no AI artifacts.
5. Composition should be horizontal and suitable for a website gallery banner.
6. Keep face clearly visible; no heavy occlusions or awkward cropping.
7. Face-centering constraints:
   - Keep the face near the center area of the frame even in action shots.
   - Ensure full head visibility with clean headroom (no top/head crop).
   - Avoid aggressive edge-cropping of face or shoulders.

Output a single high-quality horizontal image."""

    if custom_prompt:
        prompt_text += f"""

USER TAGS / PREFERENCES:
{custom_prompt}
Apply these preferences when they do not conflict with core quality and realism."""

    contents.append(prompt_text)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini did not return a gallery image in its response.")


def regenerate_reference_clean_gemini(source_image_bytes: bytes) -> bytes:
    """
    Gemini image model: faithful cleanup of one reference photo — remove text/watermarks,
    keep people and scene unchanged. Caller applies exact WebP export (1176×516, size cap).
    """
    client = get_gemini_client()
    pil_img = Image.open(io.BytesIO(source_image_bytes)).convert("RGB")
    contents: list = [
        pil_img,
        """You are given ONE photograph below.

Produce a cleaned version of THE SAME photograph for a wide website gallery banner.

STRICT RULES (highest priority):
1. **Identity lock**: Keep every person identical — same face, age, skin tone, hair, expression, pose, and body proportions. Do NOT beautify, de-age, slim, or change facial features.
2. **Scene lock**: Keep clothing, environment, lighting direction, and overall colour faithful. This is restoration, not a re-shoot or art-style change.
3. **Text removal only**: Remove overlaid text, captions, subtitles, news banners, channel logos, stock watermarks, meme text, and UI typography. Inpaint those regions so textures match surroundings naturally.
4. **No text output**: The final image must contain **no readable letters, numbers, words, or logos** anywhere (including corners). Do NOT add watermarks or branding.
5. **No new elements**: Do not add objects, people, or text. Do not crop off heads or key parts of the subject.
6. **Format intent**: Single wide horizontal photorealistic image at high resolution (will be resized to ~1176×516 later — keep content suitable for a wide banner).

If the source has no text, return a near-identical, slightly sharpened horizontal version only.""",
    ]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates (blocked or empty response).")

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini did not return a cleaned image.")


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

Evaluate the generated image on ALL of the following criteria:
1. **Face resemblance**: Does the generated face closely match the person in the reference photos?
2. **Professional quality**: Is it suitable for a LinkedIn profile? Clean, polished, high-resolution?
3. **Background**: Is the background a smooth gradient that complements the clothing?
4. **No artifacts**: No distortions, extra limbs, warped features, blurriness, or other AI artifacts.
5. **Appropriate**: The image is appropriate and professional (no offensive content).
6. **Focus**: The image is properly focused on the face and upper shoulders.

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{
    "is_approved": true or false,
    "things_to_improve": false or "detailed description of ALL issues that need fixing"
}

If the image passes ALL criteria, set "is_approved" to true and "things_to_improve" to false.
If ANY criterion fails, set "is_approved" to false and describe ALL issues in "things_to_improve"."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict image quality reviewer. Respond only with JSON."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=1000,
        temperature=0,
    )

    return parse_json_response(response.choices[0].message.content)


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.set_page_config(page_title="LinkedIn Profile Generator", page_icon="📸", layout="wide")

st.title("LinkedIn Profile Image Generator")
st.caption("Upload photos of a person → AI generates a professional LinkedIn headshot with a matching gradient background.")

artist_name = st.text_input("Artist name", placeholder="e.g., Ankur Warikoo")
optional_tags = st.text_area(
    "Optional tags/prompts",
    placeholder='e.g., "corporate keynote", "motivational speaker on stage", "stand-up comedy show"',
    help="These tags guide profile and gallery generation style.",
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

# ---- SerpApi Google Images: reference pool (gallery aspect) ----
st.divider()
st.subheader(f"Reference image pool (~{REFERENCE_IMAGE_POOL_SIZE} via SerpApi)")
st.caption(
    f"SerpApi searches favour **live event / action** shots (stage, audience, candid, expressive). "
    f"Then **GPT-4o** picks **{REFERENCE_TOP_K_OPENAI}** using **four event types**, single clear subject, visible face, low text, "
    f"plus banner aspect ~{GALLERY_WIDTH}×{GALLERY_HEIGHT}. "
    "Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in `.env`."
)
if st.button(f"Fetch {REFERENCE_IMAGE_POOL_SIZE} images (SerpApi)", key="btn_serpapi_pool", use_container_width=False):
    st.session_state.reference_pool_rows = None
    st.session_state.reference_top6_indices = None
    st.session_state.reference_rank_note = None
    st.session_state.reference_regenerated = None
    if not artist_name.strip():
        st.error("Enter an artist / celebrity name above first.")
    else:
        api_key = os.getenv("SERPAPI_API_KEY", "").strip()
        if not api_key:
            st.error("Missing `SERPAPI_API_KEY` in environment (.env).")
        else:
            with st.spinner(
                f"SerpApi: fetching up to {REFERENCE_IMAGE_POOL_SIZE} images, "
                f"then OpenAI: selecting top {REFERENCE_TOP_K_OPENAI}…"
            ):
                try:
                    candidates = fetch_serpapi_image_candidates(
                        artist_name.strip(),
                        api_key,
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
    st.markdown("""
1. **Upload** 1–5 clear photos showing face & shoulders
2. **Input Guardrail** checks visibility & quality (OpenAI GPT-4o)
3. **Nano Banana** (Gemini) generates a LinkedIn headshot
4. **Output Guardrail** validates the result (up to 3 retries)
5. **Download** your final image
""")
    st.divider()
    st.markdown("**Models used**")
    st.markdown("- Input/Output guardrail: `gpt-4o`")
    st.markdown(f"- Image generation: `{GEMINI_MODEL}`")

# ---- File uploader ----
uploaded_files = st.file_uploader(
    "Upload person photos (JPEG / PNG / WebP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"Maximum {MAX_IMAGES} images allowed. You uploaded {len(uploaded_files)}.")
        st.stop()

    # Show uploaded previews
    st.subheader("Uploaded Images")
    cols = st.columns(min(len(uploaded_files), 5))
    for i, f in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            st.image(f, caption=f.name, use_container_width=True)

    # ---- Generate button ----
    if st.button("Generate LinkedIn Profile Image", type="primary", use_container_width=True):
        st.session_state.result_data = None
        artist_folder = slugify_artist_name(artist_name)
        custom_prompt = optional_tags.strip() or None

        if not artist_name.strip():
            st.error("Please enter an artist name to generate structured output.")
            st.stop()

        # Read all files into (bytes, mime) tuples
        images: list[tuple[bytes, str]] = []
        for f in uploaded_files:
            f.seek(0)
            images.append((f.read(), mime_from_name(f.name)))

        # ── STEP 1: Input guardrail ──────────────────────────────
        with st.status("Step 1/3 — Checking image quality...", expanded=True) as status:
            st.write("Analyzing face visibility, hair, shoulders, and image quality...")
            try:
                input_check = input_guardrail(images)
            except Exception as e:
                st.error(f"Input guardrail failed: {e}")
                st.stop()

            if input_check.get("is_approved"):
                st.write("All images passed quality checks.")
                status.update(label="Step 1/3 — Input guardrail passed", state="complete")
            else:
                status.update(label="Step 1/3 — Input guardrail FAILED", state="error")
                st.error("Some images did not pass the quality check:")
                for issue in input_check.get("issues", []):
                    idx = issue.get("image_index", "?")
                    problems = issue.get("problems", [])
                    file_name = uploaded_files[idx].name if isinstance(idx, int) and idx < len(uploaded_files) else f"Image {idx}"
                    for p in problems:
                        st.write(f"- **{file_name}**: {p}")
                st.info("Please upload clearer photos where the person's face, hair, and shoulders are fully visible.")
                st.stop()

        # ── STEP 2 + 3: Generate + output guardrail (retry loop) ─
        generated_image = None
        output_check = None
        improvement_feedback = None
        attempt = 0

        with st.status("Step 2/3 — Generating LinkedIn headshot...", expanded=True) as status:
            for attempt in range(1, MAX_RETRY + 1):
                # Generation
                st.write(f"**Attempt {attempt}/{MAX_RETRY}** — Generating image with Gemini Nano Banana...")
                try:
                    generated_image = generate_linkedin_image(images, improvement_feedback, custom_prompt)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                    st.stop()

                st.write(f"**Attempt {attempt}/{MAX_RETRY}** — Validating output quality...")
                try:
                    output_check = output_guardrail(generated_image, images)
                except Exception as e:
                    st.error(f"Output guardrail failed: {e}")
                    st.stop()

                if output_check.get("is_approved"):
                    st.write("Output guardrail approved the image.")
                    status.update(label=f"Steps 2–3 — Image approved (attempt {attempt})", state="complete")
                    break

                # Not approved — show feedback and retry
                improvement_feedback = output_check.get("things_to_improve", "")
                st.warning(f"Attempt {attempt} rejected: {improvement_feedback}")

                if attempt == MAX_RETRY:
                    status.update(label=f"Steps 2–3 — Best effort after {MAX_RETRY} attempts", state="complete")

        if generated_image:
            profile_webp_bytes, within_limit = prepare_profile_webp(generated_image)
            if not profile_webp_bytes:
                st.error("Failed to prepare final profile image in WebP format.")
                st.stop()

            # ---- Gallery generation (6 variants) ----
            st.divider()
            st.subheader("Gallery Images")
            st.caption(
                f"Generating {GALLERY_COUNT} gallery variants ({GALLERY_WIDTH}x{GALLERY_HEIGHT}, .webp, target <= {GALLERY_MAX_SIZE_KB} KB each)."
            )

            gallery_outputs: list[tuple[str, bytes, bool]] = []
            with st.status("Generating gallery images...", expanded=True) as gallery_status:
                for i, scene in enumerate(GALLERY_SCENES[:GALLERY_COUNT], start=1):
                    st.write(f"Generating gallery image {i}/{GALLERY_COUNT}...")
                    try:
                        raw_gallery = generate_gallery_image(images, scene, custom_prompt)
                        gallery_webp_bytes, gallery_within_limit = prepare_gallery_webp(raw_gallery)
                    except Exception as e:
                        st.warning(f"Gallery image {i} failed: {e}")
                        continue

                    if not gallery_webp_bytes:
                        st.warning(f"Gallery image {i} could not be prepared in WebP.")
                        continue

                    gallery_outputs.append((f"gallery{i}.webp", gallery_webp_bytes, gallery_within_limit))

                if len(gallery_outputs) == GALLERY_COUNT:
                    gallery_status.update(label="Gallery generation complete", state="complete")
                elif gallery_outputs:
                    gallery_status.update(label="Gallery generation partially complete", state="complete")
                else:
                    gallery_status.update(label="Gallery generation failed", state="error")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f"{artist_folder}/profile.webp", profile_webp_bytes)
                for name, data, _ in gallery_outputs:
                    zip_file.writestr(f"{artist_folder}/{name}", data)

            st.session_state.result_data = {
                "output_check": output_check,
                "attempt": attempt,
                "profile_webp_bytes": profile_webp_bytes,
                "profile_within_limit": within_limit,
                "gallery_outputs": gallery_outputs,
                "zip_bytes": zip_buffer.getvalue(),
                "artist_folder": artist_folder,
            }

    result_data = st.session_state.result_data
    if result_data:
        st.divider()
        st.subheader("Result")

        output_check = result_data["output_check"]
        attempt = result_data["attempt"]
        profile_webp_bytes = result_data["profile_webp_bytes"]
        profile_within_limit = result_data["profile_within_limit"]
        gallery_outputs = result_data["gallery_outputs"]
        zip_bytes = result_data["zip_bytes"]
        artist_folder = result_data["artist_folder"]

        if output_check and output_check.get("is_approved"):
            st.success(f"Image approved on attempt {attempt}.")
        else:
            st.warning(
                f"Image was not fully approved after {MAX_RETRY} attempts. "
                "Showing the best result. You can try again with different photos."
            )
            if output_check and output_check.get("things_to_improve"):
                with st.expander("Remaining issues"):
                    st.write(output_check["things_to_improve"])

        if profile_within_limit:
            st.success(f"Profile output meets size target (<= {PROFILE_MAX_SIZE_KB} KB).")
        else:
            actual_kb = len(profile_webp_bytes) / 1024
            st.warning(
                f"Could not reach <= {PROFILE_MAX_SIZE_KB} KB without heavy quality loss. "
                f"Using best effort: {actual_kb:.1f} KB."
            )

        result_img = Image.open(io.BytesIO(profile_webp_bytes))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(result_img, caption="Generated LinkedIn Profile Image", use_container_width=True)

        st.download_button(
            label="Download Image",
            data=profile_webp_bytes,
            file_name="profile.webp",
            mime="image/webp",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Gallery Images")
        if gallery_outputs:
            preview_cols = st.columns(3)
            for idx, (name, data, gallery_within_limit) in enumerate(gallery_outputs):
                with preview_cols[idx % 3]:
                    st.image(
                        Image.open(io.BytesIO(data)),
                        caption=f"{name} ({len(data)/1024:.1f} KB)",
                        use_container_width=True,
                    )
                    if gallery_within_limit:
                        st.caption("Meets <= 50 KB")
                    else:
                        st.caption("Best effort (over 50 KB)")
                    st.download_button(
                        label=f"Download {name}",
                        data=data,
                        file_name=name,
                        mime="image/webp",
                        key=f"download_{name}",
                        use_container_width=True,
                    )
        else:
            st.warning("No gallery images were successfully generated.")

        st.download_button(
            label="Download All Images (ZIP)",
            data=zip_bytes,
            file_name=f"{artist_folder}_images.zip",
            mime="application/zip",
            use_container_width=True,
        )
