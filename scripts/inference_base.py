"""
Run inference on the SeeingCulture benchmark using a pre-trained (non-instruction-tuned)
base VLM.

Two prompt strategies are used depending on whether the base model ships with a chat
template:

  - No chat template (Gemma-3-pt, Llama-3.2-11B-Vision):
        "{image_token} {question}\nResponse:"
    Appending "Response:" follows the Value Drifts paper convention for base models.

  - Has chat template (Qwen3.5-4B-Base, which was pre-trained with <|im_start|> tokens):
        apply_chat_template(user message) — no system prompt added.

For instruction-tuned models or fine-tuned PEFT checkpoints use inference_at_checkpoint.py.

Examples:
    python scripts/inference_base.py \
        --model_id google/gemma-3-4b-pt \
        --output results/gemma-3-4b-pt-base.jsonl

    python scripts/inference_base.py \
        --model_id Qwen/Qwen3.5-4B-Base \
        --output results/Qwen3.5-4B-Base-base.jsonl

    python scripts/inference_base.py \
        --model_id meta-llama/Llama-3.2-11B-Vision \
        --output results/Llama-3.2-11B-Vision-base.jsonl
"""
import os
import torch
from datasets import load_dataset
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse
from tqdm import tqdm

_IMAGE_TOKEN = {
    "Gemma3Processor": "<start_of_image>",
    "MllamaProcessor": "<|image|><|begin_of_text|>",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID of the pre-trained base model.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for predictions.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--debug", action="store_true",
                        help="Load only the first 20 samples of the dataset for quick debugging.")
    return parser.parse_args()


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    output_file = args.output
    batch_size = args.batch_size
    model_id = args.model_id

    ds = load_dataset("davidguzmanr/CulturalGround-test", name="all", split="test")
    if args.debug:
        ds = ds.select(range(20))
        print("Debug mode: using first 20 samples.")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    processor_class = type(processor).__name__

    # Some base models (e.g. Qwen3.5-4B-Base) store their chat template on the tokenizer
    # but not on the processor itself. processor.apply_chat_template checks processor.chat_template,
    # so copy it over when needed (same fix used in sft_cultural_ground.py).
    if not getattr(processor, "chat_template", None):
        tokenizer_template = getattr(processor.tokenizer, "chat_template", None)
        if tokenizer_template:
            processor.chat_template = tokenizer_template

    has_chat_template = bool(getattr(processor, "chat_template", None))

    if has_chat_template:
        print(f"Processor: {processor_class} | prompt strategy: apply_chat_template")
    else:
        image_token = _IMAGE_TOKEN.get(processor_class) or getattr(processor, "image_token", None) or "<image>"
        print(f"Processor: {processor_class} | prompt strategy: raw continuation | image token: {repr(image_token)}")

    if getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"
    model.eval()


    if processor_class == "MllamaProcessor":
        # MllamaProcessor doesn't support repetition_penalty; use ngram blocking instead
        repetition_kwargs = {"no_repeat_ngram_size": 3, "top_p": 0.9}
    else:
        repetition_kwargs = {
            "repetition_penalty": 1.2,
            "top_p": 0.9,
        }

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
            question = example["question"] or ""
            if has_chat_template:
                # Base model with chat template (e.g. Qwen3.5-4B-Base): use
                # apply_chat_template without a system prompt.
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
            else:
                # Raw continuation prompt - no chat template.
                # Value Drifts paper: append "Response:" to signal the model to answer.
                text = f"{image_token} {question}\nResponse:"
            texts.append(text)
            images.append(example["image"].convert("RGB"))

        inputs = processor(
            text=texts,
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                **repetition_kwargs,
            )

        input_len = inputs.input_ids.shape[-1]
        for i, example in enumerate(batch):
            output_text = processor.decode(
                outputs[i][input_len:],
                skip_special_tokens=True,
            )
            results.append({
                "id": example["id"],
                "country": example["country"],
                "question": example["question"],
                "answer": example["answer"],
                "language": example["language"],
                "prediction": output_text,
                "model": model_id,
                "checkpoint": None,
                "system_prompt": False,
            })
            counter += 1

        pbar.set_postfix(processed=counter)

    with open(output_file, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")
    print(f"Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
