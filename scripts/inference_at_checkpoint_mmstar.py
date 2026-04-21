"""
Run inference on the MMStar benchmark using an instruction-tuned model
or a fine-tuned PEFT checkpoint built on top of one.

MMStar is a multiple-choice VQA benchmark. Predictions are the raw model output;
extract the chosen letter (A/B/C/D) during evaluation.

For pre-trained (non-instruction-tuned) base models use inference_base_mmstar.py instead.

Examples:
    # Evaluate an instruction-tuned base model (no checkpoint)
    python scripts/inference_at_checkpoint_mmstar.py \\
        --model_id google/gemma-3-4b-it \\
        --output results/mmstar/gemma-3-4b-it-base.jsonl

    # Evaluate a fine-tuned PEFT checkpoint (without system prompt)
    python scripts/inference_at_checkpoint_mmstar.py \\
        --model_id google/gemma-3-4b-it \\
        --checkpoint /path/to/checkpoint \\
        --output results/mmstar/gemma-3-4b-it-finetuned.jsonl

    # Evaluate a fine-tuned PEFT checkpoint (with cultural system prompt)
    python scripts/inference_at_checkpoint_mmstar.py \\
        --model_id google/gemma-3-4b-it \\
        --checkpoint /path/to/checkpoint \\
        --output results/mmstar/gemma-3-4b-it-finetuned-sysprompt.jsonl \\
        --use_system_prompt

    # SFT checkpoint trained from a base model — supply chat template source
    python scripts/inference_at_checkpoint_mmstar.py \\
        --model_id google/gemma-3-4b-pt \\
        --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-20000 \\
        --chat_template_source google/gemma-3-4b-it \\
        --output results/mmstar/gemma-3-4b-pt-sft-20000.jsonl
"""
import os
import torch
from datasets import load_dataset
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import argparse
from tqdm import tqdm

_LETTER_INSTRUCTION = "Answer with a single letter only (A, B, C, or D). Do not write anything else."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a PEFT adapter checkpoint. If omitted, the base model is evaluated as-is.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_system_prompt", action="store_true",
                        help="Include cultural system prompt (use for finetuned models)")
    parser.add_argument("--chat_template_source", type=str, default=None,
                        help=(
                            "Model ID to borrow the chat template from when the target model "
                            "has none (e.g. a pretrained base model). Example: pass "
                            "'google/gemma-3-4b-it' when evaluating 'google/gemma-3-4b-pt'."
                        ))
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
    base_model_id = args.model_id

    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if args.debug:
        ds = ds.select(range(20))
        print("Debug mode: using first 20 samples.")

    model = AutoModelForImageTextToText.from_pretrained(
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
    if not getattr(processor, "chat_template", None):
        tokenizer_template = getattr(processor.tokenizer, "chat_template", None)
        if tokenizer_template:
            processor.chat_template = tokenizer_template
        elif args.chat_template_source is None:
            raise ValueError(
                "The model's processor has no chat_template. For pretrained (base) models, "
                "pass --chat_template_source <it-model-id> (e.g. 'google/gemma-3-4b-it')."
            )
        else:
            _ct_processor = AutoProcessor.from_pretrained(args.chat_template_source)
            _chat_template = _ct_processor.tokenizer.chat_template
            processor.chat_template = _chat_template
            processor.tokenizer.chat_template = _chat_template
            print(f"Injected chat template from '{args.chat_template_source}' into base model processor.")
    if getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"
    model.eval()

    _eos_token_ids = [processor.tokenizer.eos_token_id]
    if "llama" in base_model_id.lower():
        eot_id = processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != processor.tokenizer.unk_token_id:
            _eos_token_ids.append(eot_id)

    processor_class = type(processor).__name__
    if processor_class == "MllamaProcessor":
        repetition_kwargs = {"top_p": 0.9}
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
            prompt_text = f"{question}\n{_LETTER_INSTRUCTION}"
            messages = []
            if args.use_system_prompt:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant. When answering multiple-choice questions, respond with only a single letter: A, B, C, or D. Do not include any explanation, punctuation, or extra text."
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ]
            })
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
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
                max_new_tokens=16,
                do_sample=False,
                eos_token_id=_eos_token_ids,
                **repetition_kwargs,
            )

        input_len = inputs.input_ids.shape[-1]
        for i, example in enumerate(batch):
            output_text = processor.decode(
                outputs[i][input_len:],
                skip_special_tokens=True
            )
            results.append({
                "index": example["index"],
                "category": example["category"],
                "l2_category": example["l2_category"],
                "question": example["question"],
                "answer": example["answer"],
                "prediction": output_text,
                "model": base_model_id,
                "checkpoint": args.checkpoint,
                "system_prompt": args.use_system_prompt,
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
