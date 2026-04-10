import os
import sys
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable first.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

MODEL_ID = "gemini-2.5-flash"
IMAGE_PATH = "images/Dia-de-Muertos-1.jpg"
COUNTRY = "Mexico"
QUESTION = "What is shown in this image?"

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

system_prompt = (
    f"You are a helpful assistant with expertise in {COUNTRY} culture. "
    f"When answering questions, consider the cultural context of {COUNTRY}."
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        QUESTION,
    ],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=1024,
        temperature=0.7,
    ),
)

candidate = response.candidates[0]
print(f"Response: {response.text}\n")
print(f"Finish reason: {candidate.finish_reason}")
