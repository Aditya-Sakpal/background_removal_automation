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

5. **Framing — MUST show full head-to-shoulders**: This is a MEDIUM HEADSHOT, NOT a face close-up. The frame MUST include:
   - Full head with visible headroom above the hair (do NOT crop the top of the head)
   - Full face from forehead to chin
   - Full neck
   - BOTH shoulders entirely visible within the frame
   - The top portion of the clothing (collar, neckline, shirt/blazer top) clearly visible
   The face should occupy roughly 35-45% of the frame vertically — NOT fill the entire frame. Think classic LinkedIn profile picture composition: you can clearly see the shoulders and collar, not just the face. Slightly angled pose, not straight-on.

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
   - Keep both eyes around the upper-third horizontal band of the frame (rule of thirds).
   - Leave approximately 8-12% headroom above the hair/top of head.
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
                    # Don't crash — we already have a generated image; trust it.
                    st.warning(f"Output guardrail errored ({e}) — treating image as approved.")
                    output_check = {"is_approved": True, "things_to_improve": False}

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
            GALLERY_MAX_ATTEMPTS = 2  # 1 retry on face-match failure
            with st.status("Generating gallery images...", expanded=True) as gallery_status:
                for i, scene in enumerate(GALLERY_SCENES[:GALLERY_COUNT], start=1):
                    st.write(f"**Gallery image {i}/{GALLERY_COUNT}** — scene: _{scene}_")

                    raw_gallery = None
                    face_feedback = None
                    face_approved = False

                    for g_attempt in range(1, GALLERY_MAX_ATTEMPTS + 1):
                        st.write(f"Attempt {g_attempt}/{GALLERY_MAX_ATTEMPTS} — generating with face anchor...")
                        try:
                            raw_gallery = generate_gallery_image(
                                images=images,
                                scene_prompt=scene,
                                profile_anchor_image=generated_image,
                                improvement_feedback=face_feedback,
                                custom_prompt=custom_prompt,
                            )
                        except Exception as e:
                            st.warning(f"Gallery image {i} generation failed: {e}")
                            raw_gallery = None
                            break

                        # Face-match guardrail
                        st.write(f"Attempt {g_attempt}/{GALLERY_MAX_ATTEMPTS} — checking face match...")
                        try:
                            face_check = gallery_face_match_guardrail(raw_gallery, generated_image)
                        except Exception as e:
                            st.warning(f"Face-match guardrail failed for image {i}: {e} — keeping best effort.")
                            face_check = {"is_approved": True, "things_to_improve": False}

                        if face_check.get("is_approved"):
                            st.write(f"Face match approved on attempt {g_attempt}.")
                            face_approved = True
                            break

                        face_feedback = face_check.get("things_to_improve", "")
                        st.warning(f"Attempt {g_attempt} face mismatch: {face_feedback}")

                    if raw_gallery is None:
                        continue

                    if not face_approved:
                        st.info(f"Gallery image {i}: kept best-effort result after {GALLERY_MAX_ATTEMPTS} attempts.")

                    try:
                        gallery_webp_bytes, gallery_within_limit = prepare_gallery_webp(raw_gallery)
                    except Exception as e:
                        st.warning(f"Gallery image {i} webp conversion failed: {e}")
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
