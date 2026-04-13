import base64
import io
import json
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image

load_dotenv()

app = FastAPI(title="LinkedIn Profile Image Generator")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-2.5-flash-image"  # Nano Banana - native image generation

SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGES = 5


# ---------------------------------------------------------------------------
# Helper: encode bytes to base64 data URL for OpenAI Vision
# ---------------------------------------------------------------------------
def encode_image_for_openai(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------------------------
# GUARDRAIL 1 – Input quality & visibility check (OpenAI Vision)
# ---------------------------------------------------------------------------
async def input_guardrail(images: list[tuple[bytes, str]]) -> dict:
    """
    Checks every uploaded image for:
      - Face fully visible
      - Hair visible
      - Head-to-shoulder region visible
      - Acceptable image quality (not blurry, well-lit, reasonable resolution)

    Returns:
        {
            "is_approved": true/false,
            "issues": [
                {"image_index": 0, "problems": ["face partially occluded", ...]},
                ...
            ]
        }
    """
    image_content = []
    for idx, (img_bytes, mime) in enumerate(images):
        image_content.append(
            {"type": "text", "text": f"--- Image {idx + 1} ---"}
        )
        image_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_for_openai(img_bytes, mime),
                    "detail": "high",
                },
            }
        )

    prompt = """You are an image quality inspector for professional headshot / LinkedIn profile photos.

For EACH image provided, evaluate ALL of the following criteria:
1. **Face visibility**: The person's full face (eyes, nose, mouth, chin) must be clearly visible and not occluded.
2. **Hair visibility**: The person's hair (or head if bald) must be fully visible, not cropped out.
3. **Head-to-shoulder visibility**: The frame must include the full head down to at least the shoulders. The top of the person's clothing should be visible.
4. **Image quality**: The image must not be blurry, must have adequate lighting, and must have a reasonable resolution (not extremely pixelated).

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

If ALL images pass ALL criteria, set "is_approved" to true and "issues" to an empty array.
If ANY image fails ANY criterion, set "is_approved" to false and list the specific problems for each failing image."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict image quality inspector. Respond only with JSON."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=1000,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# IMAGE GENERATION – Gemini Nano Banana (LinkedIn headshot)
# ---------------------------------------------------------------------------
async def generate_linkedin_image(
    images: list[tuple[bytes, str]],
    improvement_feedback: str | None = None,
) -> bytes:
    """
    Send all reference images + a detailed prompt to Gemini to produce
    a professional LinkedIn-style headshot with a gradient background
    that matches the person's clothing.

    Returns the generated image as PNG bytes.
    """
    # Build content parts: reference images + text prompt
    contents: list = []
    for img_bytes, mime in images:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contents.append(pil_img)

    prompt_text = """Using the reference photos of this person, generate a single professional LinkedIn profile headshot with the following requirements:

1. **Focus**: The image should be tightly focused on the person's face and upper shoulders — a classic headshot crop.
2. **Expression**: Keep the person's natural expression, or give a slight confident smile.
3. **Background**: Create a smooth, soft gradient background. The gradient colours must complement and harmonise with the colours of the clothing the person is wearing in the reference photos. The gradient should go from a slightly darker shade to a lighter shade, creating depth.
4. **Lighting**: Professional studio-quality lighting — soft, even, with a subtle catch-light in the eyes.
5. **Style**: Clean, polished, high-resolution look suitable for a LinkedIn profile picture.
6. **Resemblance**: The generated face MUST closely match the person in the reference photos — same facial features, skin tone, hair style, and hair colour.
7. **Clothing**: Keep the same clothing the person is wearing in the reference images, or a clean professional top if ambiguous.

Output a single square image (1:1 aspect ratio) at high quality."""

    if improvement_feedback:
        prompt_text += f"""

IMPORTANT — The previous generation was rejected. Here is the feedback on what to fix:
{improvement_feedback}

Please regenerate the image addressing ALL of the above issues."""

    contents.append(prompt_text)

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Extract generated image from response
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini did not return an image in its response.")


# ---------------------------------------------------------------------------
# GUARDRAIL 2 – Output quality check (OpenAI Vision)
# ---------------------------------------------------------------------------
async def output_guardrail(
    generated_image: bytes,
    original_images: list[tuple[bytes, str]],
) -> dict:
    """
    Validates the generated LinkedIn headshot:
      - Professional quality
      - Face matches the person in original images
      - Background is a clean gradient
      - Appropriate for LinkedIn (no artifacts, distortions, etc.)

    Returns:
        {
            "is_approved": true/false,
            "things_to_improve": false | "description of issues"
        }
    """
    image_content = []

    # Original reference images
    for idx, (img_bytes, mime) in enumerate(original_images):
        image_content.append(
            {"type": "text", "text": f"--- Original Reference Image {idx + 1} ---"}
        )
        image_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_for_openai(img_bytes, mime),
                    "detail": "high",
                },
            }
        )

    # Generated image
    image_content.append(
        {"type": "text", "text": "--- Generated LinkedIn Profile Image ---"}
    )
    image_content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image_for_openai(generated_image, "image/png"),
                "detail": "high",
            },
        }
    )

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

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict image quality reviewer. Respond only with JSON."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content},
        ],
        max_tokens=1000,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# MAIN ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/generate-profile")
async def generate_profile(files: List[UploadFile] = File(...)):
    """
    Upload person images → input guardrail → generate LinkedIn headshot →
    output guardrail (retry up to 3×) → return final image.
    """
    # ---- Validate uploads ----
    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded.")
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed.")

    images: list[tuple[bytes, str]] = []
    for f in files:
        if f.content_type not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{f.content_type}' for file '{f.filename}'. "
                        f"Supported: JPEG, PNG, WebP.",
            )
        data = await f.read()
        images.append((data, f.content_type))

    # ---- GUARDRAIL 1: Input quality check ----
    input_check = await input_guardrail(images)
    if not input_check.get("is_approved"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Uploaded images did not pass the quality check.",
                "issues": input_check.get("issues", []),
            },
        )

    # ---- GENERATE + GUARDRAIL 2 (with retry loop) ----
    max_attempts = 3
    generated_image = None
    improvement_feedback = None
    output_check = None

    for attempt in range(1, max_attempts + 1):
        # Generate LinkedIn headshot
        generated_image = await generate_linkedin_image(images, improvement_feedback)

        # Output quality check
        output_check = await output_guardrail(generated_image, images)

        if output_check.get("is_approved"):
            break

        # Not approved — capture feedback for next attempt
        improvement_feedback = output_check.get("things_to_improve", "")
        if attempt < max_attempts:
            print(
                f"[Attempt {attempt}/{max_attempts}] Output guardrail rejected. "
                f"Feedback: {improvement_feedback}"
            )

    # Return the image (approved or best-effort after 3 attempts)
    return Response(
        content=generated_image,
        media_type="image/png",
        headers={
            "X-Approved": str(output_check.get("is_approved", False)).lower(),
            "X-Attempts": str(attempt),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
