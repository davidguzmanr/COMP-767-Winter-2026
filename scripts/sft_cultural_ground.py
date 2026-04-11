"""
SFT pipeline for the davidguzmanr/CulturalGround-sft dataset.

Fine-tunes a VLM on the preprocessed CulturalGround dataset using Supervised
Fine-Tuning (SFT).  Checkpoints are saved at regular intervals so that
cultural alignment can be traced across training (following the Value Drifts
methodology).

The dataset (davidguzmanr/CulturalGround-sft) has one HuggingFace config per
country.  Use --countries to select a subset or 'all' for every country.
Each country split is concatenated into a single training dataset.

Quick test (single GPU, LoRA, bfloat16, small Qwen model):
    python scripts/sft_cultural_ground.py \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --output_dir checkpoints/sft-qwen2-5-vl-3b-cg \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_steps 50 \
        --dtype bfloat16 \
        --use_peft \
        --lora_r 64 \
        --lora_target_modules all-linear

Full run (7B model):
    python scripts/sft_cultural_ground.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --output_dir checkpoints/sft-qwen2-5-vl-7b-cg \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_steps 100 \
        --dtype bfloat16 \
        --use_peft \
        --lora_target_modules all-linear

Multi-GPU with accelerate + DeepSpeed ZeRO-3:
    accelerate launch \
        --config_file configs/accelerate/deepspeed_zero3.yaml \
        scripts/sft_cultural_ground.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --output_dir checkpoints/sft-qwen2-5-vl-7b-cg \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_steps 100 \
        --dtype bfloat16
"""
from dataclasses import dataclass, field

import io
import logging
import os

import torch
from datasets import load_dataset
from datasets import Image as HFImage
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


HF_DATASET = "davidguzmanr/CulturalGround-sft"


@dataclass
class CulturalGroundArguments:
    cg_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training."},
    )
    chat_template_source: str | None = field(
        default=None,
        metadata={
            "help": (
                "Model ID to borrow the chat template from when the target model has none "
                "(e.g. a pretrained base model). Example: pass 'google/gemma-3-4b-it' when "
                "training 'google/gemma-3-4b-pt'. IT models already have a chat template, so "
                "this argument is ignored for them."
            )
        },
    )
    image_max_pixels: int = field(
        default=1_280 * 1_280,
        metadata={
            "help": (
                "Maximum number of pixels (width × height) allowed per image. "
                "Examples containing any image that exceeds this limit are dropped "
                "before training to avoid CUDA OOM spikes. Default: 1638400 (1280×1280)."
            )
        },
    )


def _all_images_within_budget(example, max_pixels: int) -> bool:
    """Return True if every image in the example is within the pixel budget.

    Operates on raw bytes (decode=False) and reads only the image header via
    PIL.Image.open without calling .load(), so it is much faster than decoding
    the full pixel data.

    PIL's decompression-bomb guard is temporarily lifted inside this function
    because we apply our own stricter pixel cap via max_pixels.
    """
    images = example.get("images") or []
    old_limit = PILImage.MAX_IMAGE_PIXELS
    PILImage.MAX_IMAGE_PIXELS = None  # we do our own size check below
    try:
        for img_dict in images:
            if img_dict is None:
                continue
            raw = img_dict.get("bytes") if isinstance(img_dict, dict) else None
            if raw is None:
                continue
            with PILImage.open(io.BytesIO(raw)) as img:
                if img.width * img.height > max_pixels:
                    return False
    finally:
        PILImage.MAX_IMAGE_PIXELS = old_limit
    return True


def load_cultural_ground(args: CulturalGroundArguments):
    """Load davidguzmanr/CulturalGround-sft (single merged default config)."""
    print(f"Loading {HF_DATASET} ...")
    dataset = load_dataset(HF_DATASET, split=args.cg_split).select(range(100))
    print(f"Total training examples: {len(dataset)}")

    if args.image_max_pixels > 0:
        before = len(dataset)
        # Cast to decode=False so the filter reads only image headers (no full
        # pixel decode) — roughly 10–20× faster than operating on PIL objects.
        # num_proc=1 + keep_in_memory=True avoids Arrow cache-file collisions
        # between the two DDP ranks that both run this code simultaneously.
        dataset = (
            dataset
            .cast_column("images", [HFImage(decode=False)])
            .filter(
                _all_images_within_budget,
                fn_kwargs={"max_pixels": args.image_max_pixels},
                num_proc=1,
                keep_in_memory=True,
            )
            .cast_column("images", [HFImage(decode=True)])
        )
        dropped = before - len(dataset)
        print(
            f"Filtered {dropped} examples with images > {args.image_max_pixels:,} pixels "
            f"({len(dataset)} remaining)."
        )

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = TrlParser((CulturalGroundArguments, SFTConfig, ModelConfig))
    cg_args, training_args, model_args = parser.parse_args_and_config()

    # Critical for VLMs: do not truncate image tokens.
    training_args.max_length = None

    # Default to step-based saving for drift analysis.
    if training_args.save_strategy == "no":
        training_args.save_strategy = "steps"
        training_args.save_steps = training_args.save_steps or 100

    ################
    # Model
    ################
    dtype = (
        model_args.dtype
        if model_args.dtype in ("auto", None)
        else getattr(torch, model_args.dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    ################
    # Processor
    ################
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if not getattr(processor, "chat_template", None) and not getattr(processor.tokenizer, "chat_template", None):
        if cg_args.chat_template_source is None:
            raise ValueError(
                "The model's processor has no chat_template. For pretrained (base) models, "
                "pass --chat_template_source <it-model-id> (e.g. 'google/gemma-3-4b-it')."
            )
        _ct_processor = AutoProcessor.from_pretrained(
            cg_args.chat_template_source,
            trust_remote_code=model_args.trust_remote_code,
        )
        _chat_template = _ct_processor.tokenizer.chat_template
        processor.chat_template = _chat_template
        processor.tokenizer.chat_template = _chat_template
        print(f"Injected chat template from '{cg_args.chat_template_source}' into base model processor.")

    ################
    # Dataset
    ################
    train_dataset = load_cultural_ground(cg_args)

    ################
    # Training
    ################
    # Mllama (Llama 3.2 Vision) uses cross-attention image tokens whose IDs can
    # exceed the text vocab size.  When those IDs appear in the labels tensor the
    # NLL loss kernel throws an out-of-range assertion.  Masking them to -100
    # (the standard ignore index) before the forward pass fixes this.
    # See: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/31
    if model.config.model_type == "mllama":
        class VisionSFTTrainer(SFTTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                if "labels" in inputs:
                    cfg = getattr(model.config, "text_config", model.config)
                    vocab_size = cfg.vocab_size
                    labels = inputs["labels"]
                    inputs["labels"] = labels.masked_fill(labels >= vocab_size, -100)
                return super().compute_loss(
                    model, inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
        trainer_cls = VisionSFTTrainer
    else:
        trainer_cls = SFTTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=processor,
    )

    # Resume from the latest checkpoint if one exists in output_dir (handles
    # preemption: the job simply re-queues and picks up where it left off).
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            logger.info(f"Output directory '{training_args.output_dir}' exists but contains no checkpoint; starting from scratch.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=HF_DATASET)
