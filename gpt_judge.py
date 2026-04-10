# -*- coding: utf-8 -*-
import argparse
import base64
import json
import os
import time
from io import BytesIO

from datasets import load_dataset
from openai import OpenAI, RateLimitError
from tqdm import tqdm

SYSTEM_TEMPLATE = """
You are a cultural awareness expert in {country}. Please evaluate the cultural awareness of the following response to the question: \
'What cultural significance does the following image have?' and determine if it is a good answer based on the provided \
'Ground Truth' text and input image. Respond only with:\
    - Culturally Appropriate\
    - Culturally Inappropriate\
    - Culturally Neutral
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="JSONL file with model predictions")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for judgements")
    parser.add_argument("--model_id", type=str, default="gpt-4o-mini")
    return parser.parse_args()


def encode_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    args = parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    ds = load_dataset("davidguzmanr/seeingculture-benchmark", split="test")
    idx_to_data = {ex["Unique Index"]: (ex["Rationale"], ex["Country"], ex["Image"]) for ex in ds}

    idx_to_prediction = {}
    with open(args.results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                idx_to_prediction[ex["Unique Index"]] = ex["Prediction"]

    judgements = []

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for example in tqdm(ds.select(range(10)), desc="Judging"):
        unique_idx = example["Unique Index"]

        ground_truth, country, pil_image = idx_to_data[unique_idx]
        prediction = idx_to_prediction[unique_idx]

        image_b64 = encode_image_to_base64(pil_image)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_TEMPLATE.format(country=country),
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
                            f"Evaluate this prediction based on the following:\n"
                            f"Prediction: {prediction}\n"
                            f"Ground Truth: {ground_truth}"
                        ),
                    },
                ],
            },
        ]

        time.sleep(1)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    logprobs=True,
                    top_logprobs=5,
                    max_completion_tokens=32,
                    temperature=0.7,
                )
                break
            except RateLimitError as e:
                wait = 2 ** attempt
                tqdm.write(f"Rate limit hit, retrying in {wait}s... ({e})")
                time.sleep(wait)
        else:
            raise RuntimeError("Failed after 8 retries due to rate limiting.")

        choice = response.choices[0]
        judgements.append({
            "Unique Index": unique_idx,
            "Image ID": example["Image ID"],
            "Judgement": choice.message.content,
            "Ground Truth Rationale": ground_truth,
            "Model Logprobs": [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top_logprobs": [{"token": a.token, "logprob": a.logprob} for a in t.top_logprobs],
                }
                for t in choice.logprobs.content
            ],
        })

        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(judgements, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
