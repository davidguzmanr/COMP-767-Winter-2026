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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()
adapter_path = args.checkpoint

base_model_id = "google/gemma-3-4b-it"
checkpoint_name = f"{base_model_id}{os.path.basename(adapter_path)}"
output_file = os.path.join("RESULTS", f"{checkpoint_name}.jsonl")

BATCH_SIZE = 8  # 4–16 depending on VRAM
HF_AUTH_TOKEN = "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"
if HF_AUTH_TOKEN:
    login(token=HF_AUTH_TOKEN)
else:
    print("Warning: No HF_TOKEN found. You may hit rate limits.")

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
# output_file = "RESULTS_gemma_checkpoint_test_grounded_culture_baseline.jsonl"
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

def process_batch(batch):
    texts = []
    images = []
    meta = []

    for example in batch:
        img_id = example["Image ID"]

        if img_id in processed_files:
            continue

        try:
            image = load_image(example)

            messages = [
                {"role": "user",
                    "content": 
                    [{"type": "image"},
                        {"type": "text",
                        "text": f"What cultural significance does the following image have in {example['Country']}?"}]}]
            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            texts.append(text)
            images.append(image)
            meta.append(example)

        except Exception as e:
            print("Skipping (load fail):", img_id, "|", e)
            processed_files.add(img_id)

    if len(texts) == 0:
        return []

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80
        )

    results = []

    for i, example in enumerate(meta):
        input_len = inputs.input_ids[i].shape[-1]

        decoded = processor.decode(
            outputs[i][input_len:],
            skip_special_tokens=True
        )

        results.append({
            "Image ID": example["Image ID"],
            "Prediction": decoded,
            "Ground Truth Rationale": example["Rationale"]
        })

        processed_files.add(example["Image ID"])

    return results


dataset = shuffled_dataset["test"]

for i in range(0, len(dataset), BATCH_SIZE):
    batch = dataset[i:i + BATCH_SIZE]

    batch_results = process_batch(batch)

    results.extend(batch_results)
    counter += len(batch_results)

    print(f"Processed {counter} examples")

    # periodic flush
    if counter % 50 == 0:
        with open(output_file, "a") as f:
            for r in results:
                json.dump(r, f)
                f.write("\n")
        results = []

# final flush
if results:
    with open(output_file, "a") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")
    print(f"  ↳ Final flush: {len(results)} results written to {output_file}")

# # Final flush for any remaining results
# if results:
#     with open(output_file, "a") as f:
#         for r in results:
#             json.dump(r, f)
#             f.write("\n")
#     print(f"  ↳ Final flush: {len(results)} results written to {output_file}")