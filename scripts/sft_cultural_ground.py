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

# /// script
# dependencies = [
#     "trl[peft]",
#     "Pillow>=9.4.0",
#     "datasets>=2.20",
# ]
# ///

from dataclasses import dataclass, field

import logging
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText
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


def load_cultural_ground(args: CulturalGroundArguments):
    """Load davidguzmanr/CulturalGround-sft (single merged default config)."""
    print(f"Loading {HF_DATASET} ...")
    dataset = load_dataset(HF_DATASET, split=args.cg_split)
    print(f"Total training examples: {len(dataset)}")
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
    # Dataset
    ################
    train_dataset = load_cultural_ground(cg_args)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
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
