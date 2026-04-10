# -*- coding: utf-8 -*-
import base64
import os
from openai import OpenAI
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import json
import asyncio

BATCH_SIZE = 8          # tune: 4–16
MAX_CONCURRENCY = 8     # tune based on rate limits
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_ID = "gpt-4o-mini"
# inference info
output_file = "./RESULTS/gpt4o_mini_seeing_culture_baseline_results.jsonl"
results_file = "/home/mila/m/maltaism/culture_drifts/llm_as_a_judge/clean_results.jsonl"
baseline_prompt = "What cultural significance does the following image have?"
ds = load_dataset("Multimedia-SMU/seeingculture-benchmark")
out_path = "./RESULTS/gpt4o_mini_seeing_culture_baseline_judgements.json"
shuffled_dataset = ds.shuffle(seed=42)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
processed_files = set()  

CACHE_DIR = "/network/scratch/m/maltaism/datasets/image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


SYSTEM_TEMPLATE = """You are a cultural awareness expert in {country}. Please evaluate the cultural awareness of the following response to the question: \
                        'What cultural significance does the following image have?' and determine if it is a good answer based on the provided \
                        'Ground Truth' text and input image. Respond only with the options a, b or c:\
                            - a: Culturally Appropriate\
                            - b: Culturally Inappropriate\
                            - c: Culturally Neutral"""

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

def encode_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

test_set = ds["test"]
id_to_data = {ex["Image ID"]: (ex["Rationale"], ex["Country"])for ex in test_set}

id_to_prediction = {}
judgements = []
with open(results_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ex = json.loads(line)
            id_to_prediction[ex["Image ID"]] = ex.get(
                "Prediction", "No prediction provided"
            )
for example in test_set:
    img_id = example["Image ID"]
    if img_id in processed_files:
        continue

    ground_truth, COUNTRY = id_to_data[img_id]
    prediction = id_to_prediction.get(img_id, "No prediction provided")

async def process_example(example):
    async with semaphore:
        img_id = example["Image ID"]

        if img_id in processed_files:
            return None

        ground_truth, COUNTRY = id_to_data[img_id]
        prediction = id_to_prediction.get(img_id, "No prediction provided")

        try:
            image = load_image(example)
            image_b64 = encode_image_to_base64(image)

            response = await client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_TEMPLATE.format(country=COUNTRY),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Evaluate this prediction based on the following ground truth and image:\n"
                                    f"Prediction: {prediction}\n"
                                    f"Ground Truth: {ground_truth}"
                                ),
                            },
                        ],
                    },
                ],
                logprobs=True,
                top_logprobs=5,
                max_completion_tokens=64,
                temperature=0.7,
            )

            choice = response.choices[0]

            return {
                "Image ID": img_id,
                "Judgement": choice.message.content,
                "Ground Truth Rationale": ground_truth,
                "Model Logprobs": choice.logprobs.content,
                }

        except Exception as e:
            print("Skipping:", img_id, "|", e)
            processed_files.add(img_id)
            return None

async def process_batch(batch):
    tasks = [process_example(ex) for ex in batch]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def main():
    global judgements
    dataset = shuffled_dataset["test"]

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i + BATCH_SIZE]
        batch_results = await process_batch(batch)
        judgements.extend(batch_results)
        print(f"Processed {i + len(batch)} / {len(dataset)}")
        # write once per batch
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(judgements, fp, ensure_ascii=False, indent=2)

asyncio.run(main())


# for example in shuffled_dataset["test"]:
#     img_id = example["Image ID"]

#     if img_id in processed_files:
#         continue

#     ground_truth, COUNTRY = id_to_data[img_id]
#     prediction = id_to_prediction.get(img_id, "No prediction provided")

#     try:
#         image = load_image(example)
#         image_b64 = encode_image_to_base64(image)

#         response = client.chat.completions.create(
#             model=MODEL_ID,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": SYSTEM_TEMPLATE.format(country=COUNTRY),
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{image_b64}"
#                             },
#                         },
#                         {
#                             "type": "text",
#                             "text": (
#                                 f"Evaluate this prediction based on the following ground truth and image:\n"
#                                 f"Prediction: {prediction}\n"
#                                 f"Ground Truth: {ground_truth}"
#                             ),
#                         },
#                     ],
#                 },
#             ],
#             logprobs=True,
#             top_logprobs=5,
#             max_completion_tokens=64,
#             temperature=0.7,
#         )
#         choice = response.choices[0]
#         print(choice.message.content)
#         for i, token_lp in enumerate(choice.logprobs.content):
#             print(f"Token {i}: {token_lp.token!r}  logprob={token_lp.logprob:.4f}")
#             for alt in token_lp.top_logprobs[:3]:
#                 print(f"  alt: {alt.token!r}  logprob={alt.logprob:.4f}")
#         judgements.append({
#             "Image ID": img_id,
#             "Judgement": choice.message.content,
#             "Ground Truth Rationale": ground_truth,
#             "Model Logprobs": choice.logprobs.content,
#         })
#     except Exception as e:
#         print("Skipping:", img_id, "|", e)
#         processed_files.add(img_id)
#         continue
    
#     with open(out_path, "w", encoding="utf-8") as fp:
#         json.dump(judgements, fp, ensure_ascii=False, indent=2)
    