"""
Run inference on the SeeingCulture benchmark using a base model or a fine-tuned PEFT checkpoint.

Examples:
    # Evaluate the base model
    python inference_at_checkpoint.py \
        --model_id google/gemma-3-4b-it \
        --output results/gemma-3-4b-it-base.jsonl

    # Evaluate a fine-tuned PEFT checkpoint (without system prompt)
    python inference_at_checkpoint.py \
        --model_id google/gemma-3-4b-it \
        --checkpoint /path/to/checkpoint \
        --output results/gemma-3-4b-it-finetuned.jsonl

    # Evaluate a fine-tuned PEFT checkpoint (with cultural system prompt, matches training format)
    python inference_at_checkpoint.py \
        --model_id google/gemma-3-4b-it \
        --checkpoint /path/to/checkpoint \
        --output results/gemma-3-4b-it-finetuned-sysprompt.jsonl \
        --use_system_prompt

    # Use a different base model with a larger batch size
    python inference_at_checkpoint.py \
        --model_id google/gemma-3-12b-it \
        --checkpoint /path/to/checkpoint \
        --output results/gemma-3-12b-it-finetuned.jsonl \
        --batch_size 8
"""
import os
import torch
from datasets import load_dataset
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a PEFT adapter checkpoint. If omitted, the base model is evaluated as-is.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_system_prompt", action="store_true",
                        help="Include cultural system prompt (use for finetuned models)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = args.output
    batch_size = args.batch_size
    base_model_id = args.model_id

    ds = load_dataset("davidguzmanr/seeingculture-benchmark", split="test")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )
    if args.checkpoint is not None:
        print(f"Loading PEFT adapter from {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()
    else:
        print("No checkpoint provided — evaluating base model.")
    processor = AutoProcessor.from_pretrained(base_model_id)
    model.eval()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    counter = 0
    results = []

    test_examples = list(ds)

    n_batches = (len(test_examples) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(test_examples), batch_size), total=n_batches, desc="Inference")

    for batch_start in pbar:
        batch = test_examples[batch_start: batch_start + batch_size]

        texts, images = [], []
        for example in batch:
            messages = []
            if args.use_system_prompt:
                messages.append({
                    "role": "system",
                    "content": (
                        f"You are a helpful assistant with expertise in {example['Country']} culture. "
                        f"When answering questions, consider the cultural context of {example['Country']}."
                    )
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["Question"]}
                ]
            })
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
            images.append(example["Image"].convert("RGB"))

        inputs = processor(
            text=texts,
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512
            )

        input_len = inputs.input_ids.shape[-1]
        for i, example in enumerate(batch):
            output_text = processor.decode(
                outputs[i][input_len:],
                skip_special_tokens=True
            )
            results.append({
                "Image ID": example["Image ID"],
                "Country": example["Country"],
                "Category": example["Category"],
                "Concept": example["Concept"],
                "Question": example["Question"],
                "Prediction": output_text,
                "Ground Truth Rationale": example["Rationale"],
                "Model": base_model_id,
                "Checkpoint": args.checkpoint,
                "System Prompt": args.use_system_prompt,
            })
            counter += 1

        pbar.set_postfix(processed=counter)

        if counter % 10 == 0:
            with open(output_file, "a") as f:
                for r in results:
                    json.dump(r, f)
                    f.write("\n")
            results = []

    if results:
        with open(output_file, "a") as f:
            for r in results:
                json.dump(r, f)
                f.write("\n")
        print(f" Final flush: {len(results)} results written to {output_file}")


if __name__ == "__main__":
    main()
