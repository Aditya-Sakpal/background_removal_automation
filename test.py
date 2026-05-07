import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

os.environ["GEMINI_API_KEY"] = 'AIzaSyArmRC86JfEjgmWeOW2SyBrj1gzAjXHV2Y'

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=["A photorealistic product shot of a blue sneaker on a white seamless background, studio lighting"],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    ),
)

# Extract and save the image
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        img = Image.open(BytesIO(part.inline_data.data))
        img.save("output.png")
        img.show()  # opens in default image viewer
        print("Saved as output.png")
    elif part.text:
        print("Text:", part.text)
