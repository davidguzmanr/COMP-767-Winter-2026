# COMP-767-Winter-2026
Project for COMP-767: Large Language Models

## Culture Drifts: How Post-Training Shapes Cultural Awareness in Vision-Language Models

### Preprocessing

CulturalGround is preprocessed with `scripts/data/preprocess_cultural_ground.py`.

```bash
# SFT - all valid examples except the last 250 per country
python scripts/data/preprocess_cultural_ground.py \
    --countries all \
    --output_format sft \
    --dpo_holdout_size 250 \
    --push_to_hub \
    --repo_id davidguzmanr/CulturalGround-sft

# DPO - last 250 valid examples per country
python scripts/data/preprocess_cultural_ground.py \
    --countries all \
    --output_format dpo \
    --reference_model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dpo_rejected_batch_size 16 \
    --dpo_holdout_size 250 \
    --regenerate_chosen \
    --push_to_hub \
    --repo_id davidguzmanr/CulturalGround-dpo
```

The SFT and DPO splits are non-overlapping (`--dpo_holdout_size` reserves the last N valid examples per country for DPO).

### Datasets

#### SFT / Evaluation
| Dataset | HF ID | Size | Coverage |
|---|---|---|---|
| CulturalGround | [neulab/CulturalGround](https://huggingface.co/datasets/neulab/CulturalGround) | 22M VQA | 42 countries, 39 languages |
| CVQA | [afaji/cvqa](https://huggingface.co/datasets/afaji/cvqa) | 10K | 30 countries, 31 languages |
| CulturalVQA | [mair-lab/CulturalVQA](https://huggingface.co/datasets/mair-lab/CulturalVQA) | 2.4K | 11 countries, 5 continents |
| Culture Affordance Atlas | None | 367 obj-func pairs | 63 countries (eval only) |
| Seeing Culture | [Multimedia-SMU/seeingculture-benchmark](https://huggingface.co/datasets/Multimedia-SMU/seeingculture-benchmark) | 3K Q | 7 SE Asian countries (eval only) |

### Training

#### SFT

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/sft_cultural_ground.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir checkpoints/sft-qwen2-5-vl-3b-culturalground \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --save_steps 200 \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --use_peft \
    --lora_r 64 \
    --lora_target_modules all-linear
```

#### DPO

Pass a local SFT checkpoint or any HF model ID as `--model_name_or_path`. LoRA adapters in the checkpoint directory are detected and loaded automatically.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/dpo_cultural_ground.py \
    --model_name_or_path checkpoints/sft-qwen2-5-vl-3b-culturalground \
    --output_dir checkpoints/dpo-qwen2-5-vl-3b-culturalground \
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
```

#### DPO (preference pairs)
Synthetic preference dataset constructed from CVQA/CulturalVQA: chosen = culturally grounded answer, rejected = response from a base model prompted without cultural conditioning. See `scripts/data/preprocess_datasets.py`.

General-purpose VLM preference datasets (pipeline validation):
| Dataset | HF ID | Size | Notes |
|---|---|---|---|
| RLAIF-V (formatted) | [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) | 83K | Pre-formatted for TRL DPOTrainer |
| RLAIF-V (raw) | [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) | ~100K | Sources: COCO, TextVQA, GQA, ShareGPT4V |
| RLHF-V | [openbmb/RLHF-V-Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset) | 5.7K | Fine-grained human feedback |
| MM-RLHF | [yifanzhang114/MM-RLHF](https://huggingface.co/datasets/yifanzhang114/MM-RLHF) | 16K | Expert-ranked; diverse visual domains |
