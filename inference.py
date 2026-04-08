import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image
import requests

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT = "checkpoints/SFT-gemma-3-4b-it-CulturalGround/checkpoint-20000"

processor = AutoProcessor.from_pretrained(BASE_MODEL)

model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, CHECKPOINT)
model.eval()

# --- inference ---
# image = Image.open("image.jpg")
url = "https://davidguzmanr.github.io/assets/photos/Dia-de-Muertos-1.jpg"
image = Image.open(requests.get(url, stream=True).raw)
country = "mexico"

messages = [
    {
        "role": "system",
        "content": f"You are a helpful assistant with expertise in {country.title()} culture. "
                   f"When answering questions, consider the cultural context of {country.title()}.",
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

generated = output_ids[:, inputs["input_ids"].shape[1]:]
response = processor.decode(generated[0], skip_special_tokens=True)

image.show()

print("\n--- Prompt ---")
print(text)
print("\n--- Response ---")
print(response)