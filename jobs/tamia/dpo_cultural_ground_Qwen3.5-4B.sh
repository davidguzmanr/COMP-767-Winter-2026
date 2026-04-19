#!/usr/bin/env bash
#SBATCH --job-name=dpo_cultural_ground_Qwen3.5-4B
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=0-24:00:00
#SBATCH --account=aip-davlan
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load python/3.10.13
module load StdEnv/2023 scipy-stack/2023b
module load gcc arrow/23.0.1
module load cuda/12.2

export HF_HOME=/home/d/davidguz/links/scratch/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/d/davidguz/links/scratch/Repositories/COMP-767-Winter-2026
source COMP-767/bin/activate

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi

START_TIME=$(date +%s)
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo $HF_HOME
##################################################################
# Run
##################################################################
accelerate launch --num_processes 4 --mixed_precision bf16 scripts/dpo_cultural_ground.py \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --output_dir checkpoints/DPO-Qwen3.5-4B-CulturalGround \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_steps 250 \
    --beta 0.1 \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear \
    --ignore_bias_buffers True \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --report_to tensorboard \
    --logging_steps 10

##################################################################
# Print ending datetime and total duration
##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"