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

EVALUATION_PROMPT = """\
You are given a question by a user and a response by an assistant about the culture of {country}.
Your task is to determine whether the response is culturally appropriate, based on the ground truth provided.

<conversation>
    Question: {question}
    Response: {prediction}
    Ground Truth: {ground_truth}
</conversation>

<instructions>
    Evaluate ONLY the cultural accuracy and relevance of the response relative to the ground truth.
    Do NOT reward or penalize based on response length, verbosity, or writing style.
    A short, precise answer that correctly identifies the cultural element is just as good as a long one.
    A long, verbose answer that adds no relevant cultural content should not be rated higher than a short one.
    Focus exclusively on whether the cultural facts in the response align with the ground truth.
</instructions>

<options>
    - (A) Culturally Appropriate: The response correctly identifies cultural elements and aligns with the ground truth.
    - (B) Culturally Neutral: The response does not address the cultural aspects of the question, or provides a generic answer without cultural grounding.
    - (C) Culturally Inappropriate: The response misidentifies cultural elements or contradicts the ground truth.
</options>

Output Format:
Return only the letter (A, B, or C). Do not include parentheses, option text, or any explanation."""


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

    ds = load_dataset("davidguzmanr/CulturalGround-test", name="all", split="test")
    id_to_image = {ex["id"]: ex["image"] for ex in ds}

    predictions = []
    with open(args.results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    judgements = []

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for pred_data in tqdm(predictions, desc="Judging"):
        sample_id = pred_data["id"]
        country = pred_data["country"]
        question = pred_data["question"]
        ground_truth = pred_data["answer"]
        prediction = pred_data["prediction"]
        model = pred_data["model"]
        checkpoint = pred_data["checkpoint"]
        system_prompt = pred_data["system_prompt"]

        pil_image = id_to_image[sample_id]
        image_b64 = encode_image_to_base64(pil_image)
        messages = [
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
                        "text": EVALUATION_PROMPT.format(
                            country=country,
                            question=question,
                            prediction=prediction,
                            ground_truth=ground_truth,
                        ),
                    },
                ],
            },
        ]

        # time.sleep(1)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    logprobs=True,
                    top_logprobs=5,
                    max_completion_tokens=32,
                    temperature=0.0,
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
            "id": sample_id,
            "country": country,
            "question": question,
            "prediction": prediction,
            "answer": ground_truth,
            "judgement": choice.message.content,
            "model": model,
            "checkpoint": checkpoint,
            "system_prompt": system_prompt,
            "judge_model": args.model_id,
            "model_logprobs": [
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
