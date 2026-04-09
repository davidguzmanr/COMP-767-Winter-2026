"""
DPO pipeline for the davidguzmanr/CulturalGround-dpo dataset.

Runs Direct Preference Optimization (DPO) on a VLM, starting from either a
base instruct model or an SFT checkpoint (scripts/sft_cultural_ground.py).
Checkpoints are saved at regular intervals so that cultural alignment can be
traced across the preference-optimization stage.

The dataset (davidguzmanr/CulturalGround-dpo) contains prompt/chosen/rejected
triples where:
  - chosen  → culturally-grounded answer from the SFT model with cultural conditioning
  - rejected → generic answer from the same model without cultural conditioning

Quick test (single GPU, LoRA, bfloat16, small Qwen model):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/dpo_cultural_ground.py \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --output_dir checkpoints/dpo-qwen2-5-vl-3b-cg \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing \
        --save_steps 200 \
        --beta 0.1 \
        --dtype bfloat16 \
        --attn_implementation sdpa \
        --use_peft \
        --lora_r 64 \
        --lora_target_modules all-linear

From SFT checkpoint:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/dpo_cultural_ground.py \
        --model_name_or_path checkpoints/sft-qwen2-5-vl-3b-cg \
        --output_dir checkpoints/dpo-qwen2-5-vl-3b-cg \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing \
        --save_steps 200 \
        --beta 0.1 \
        --dtype bfloat16 \
        --attn_implementation sdpa \
        --use_peft \
        --lora_r 64 \
        --lora_target_modules all-linear

Multi-GPU with accelerate + DeepSpeed ZeRO-3:
    accelerate launch \
        --config_file configs/accelerate/deepspeed_zero3.yaml \
        scripts/dpo_cultural_ground.py \
        --model_name_or_path checkpoints/sft-qwen2-5-vl-7b-cg \
        --output_dir checkpoints/dpo-qwen2-5-vl-7b-cg \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --gradient_checkpointing \
        --save_steps 200 \
        --beta 0.1 \
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
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


HF_DATASET = "davidguzmanr/CulturalGround-dpo"


@dataclass
class CulturalGroundArguments:
    cg_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": (
                "Fix for torch distributed when model contains bool buffers "
                "(e.g. Qwen2.5-VL). Set to True when training with DDP."
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
    """Load davidguzmanr/CulturalGround-dpo (single merged default config)."""
    print(f"Loading {HF_DATASET} ...")
    dataset = load_dataset(HF_DATASET, split=args.cg_split)
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

    parser = TrlParser((CulturalGroundArguments, DPOConfig, ModelConfig))
    cg_args, training_args, model_args = parser.parse_args_and_config()

    # Critical for VLMs: do not truncate image tokens.
    training_args.max_length = None

    # Default to step-based saving for drift analysis.
    if training_args.save_strategy == "no":
        training_args.save_strategy = "steps"
        training_args.save_steps = training_args.save_steps or 200

    ################
    # Model & Processor
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

    # If the checkpoint is a LoRA adapter directory, load the base model first
    # and apply the adapter manually to avoid a transformers PEFT integration
    # bug with qwen2_vl (KeyError in _MOE_TARGET_MODULE_MAPPING).
    is_peft_checkpoint = os.path.exists(
        os.path.join(model_args.model_name_or_path, "adapter_config.json")
    )
    if is_peft_checkpoint:
        peft_cfg = PeftConfig.from_pretrained(model_args.model_name_or_path)
        base_model_path = peft_cfg.base_model_name_or_path
        print(f"LoRA checkpoint detected. Loading base model from {base_model_path} ...")
    else:
        base_model_path = model_args.model_name_or_path

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    if is_peft_checkpoint:
        print(f"Applying LoRA adapter from {model_args.model_name_or_path} ...")
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=True)

    # Required for torch distributed when model contains bool buffers (e.g. Qwen2.5-VL).
    if cg_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # DPOTrainer needs the processor to handle image inputs in the preference pairs.
    # Use the base model path so the processor is always loadable (adapters don't
    # ship a tokenizer/processor config).
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=model_args.trust_remote_code,
        do_image_splitting=False,
    )

    ################
    # Dataset
    ################
    train_dataset = load_cultural_ground(cg_args)

    ################
    # Training
    ################
    # Only pass a new peft_config when we are NOT loading from an existing
    # LoRA checkpoint (which is already applied as a PeftModel above).
    peft_config = None if is_peft_checkpoint else get_peft_config(model_args)

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=peft_config,
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
