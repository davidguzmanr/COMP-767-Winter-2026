# -*- coding: utf-8 -*-
# Load model directly
import os
import torch
from datasets import load_dataset
import json
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login


HF_AUTH_TOKEN = "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"
if HF_AUTH_TOKEN:
    login(token=HF_AUTH_TOKEN)
else:
    print("Warning: No HF_TOKEN found. You may hit rate limits.")



base_model_id = "google/gemma-3-4b-it"
adapter_path = "/network/scratch/g/guzmand/Repositories/COMP-767-Winter-2026/checkpoints/SFT-gemma-3-4b-it-CulturalGround/checkpoint-1000"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, adapter_path)
processor = AutoProcessor.from_pretrained(base_model_id)
model.eval()
model = model.merge_and_unload()  # merge adapter weights into base model for inference

CACHE_DIR = "/network/scratch/m/maltaism/datasets/image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_image(example):
    filename = example["Image ID"]
    path = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(path):
        response = requests.get(example["URL"], timeout=10)
        response.raise_for_status()  # catch 4xx/5xx errors early

        # Validate it's actually an image before caching
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            raise ValueError(f'URL did not return a valid image: {example["URL"]}')

        with open(path, "wb") as f:
            f.write(response.content)

        return img  # already loaded, no need to re-open

    return Image.open(path).convert("RGB")


# inference info
output_file = "RESULTS_gemma_checkpoint_test_grounded_culture_baseline.jsonl"
baseline_prompt = "What cultural significance does the following image have?"
ds = load_dataset("Multimedia-SMU/seeingculture-benchmark")
shuffled_dataset = ds.shuffle(seed=42)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
counter = 0
results = []  # accumulate in memory
processed_files = set()  

if os.path.exists(output_file):
    try:
        with open(output_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    result = json.loads(line)
                    processed_files.add(result["Image ID"])
            print(f"  ↳ Loaded {len(processed_files)} existing results from {output_file}")
    except Exception as e:
        print(f"  ↳ Could not load existing results: {e}")

for example in shuffled_dataset["test"]:
    if example["Image ID"] in processed_files:
        print(f"  ↳ Skipping already processed file: {example['Image ID']}")
        continue
    grounded_prompt = f"What cultural significance does the following image have in {example['Country']}?"

    try:
        # image_url = example["URL"]
        image = load_image(example)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"What cultural significance does the following image have in {example['Country']}?"}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80
            )
        output_text = processor.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        print(output_text)
        results.append({
            "Image ID": example["Image ID"],
            "Prediction": output_text,
            "Ground Truth Rationale": example["Rationale"]
        })
        processed_files.add(example["Image ID"])  # ← ADD THIS
        counter += 1
        print("Processed:", example["Image ID"], "| Total processed:", counter)
        if counter % 10 == 0:
            with open(output_file, "a") as f:
                for r in results:
                    json.dump(r, f)
                    f.write("\n")
            results = []  # clear buffer

    except Exception as e:
        print("Skipping:", example["Image ID"], "|", e)
        processed_files.add(example["Image ID"]) 
        continue
    break

# Final flush for any remaining results
if results:
    with open(output_file, "a") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")
    print(f"  ↳ Final flush: {len(results)} results written to {output_file}")