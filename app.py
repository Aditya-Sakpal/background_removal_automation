import base64
import io
import json
import os

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

Output a single square image (1:1 aspect ratio) at the highest possible quality and resolution. The final image must look like it was taken by a professional photographer with a high-end camera — not generated by AI."""

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
                    generated_image = generate_linkedin_image(images, improvement_feedback)
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

        # ── RESULT ───────────────────────────────────────────────
        st.divider()
        st.subheader("Result")

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

        if generated_image:
            result_img = Image.open(io.BytesIO(generated_image))
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(result_img, caption="Generated LinkedIn Profile Image", use_container_width=True)

            # Download button
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="linkedin_profile.png",
                mime="image/png",
                use_container_width=True,
            )
