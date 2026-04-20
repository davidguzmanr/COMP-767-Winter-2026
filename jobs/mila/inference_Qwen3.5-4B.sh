#!/usr/bin/env bash
#SBATCH --job-name=inference_Qwen3.5-4B
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load miniconda/3
module load gcc/9.3.0

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate COMP-767

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

cd /home/mila/g/guzmand/scratch/Repositories/COMP-767-Winter-2026

##################################################################
# Run
##################################################################

##################################################################
# Base IT model
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id Qwen/Qwen3.5-4B \
    --output results/Qwen/Qwen3.5-4B/Qwen/Qwen3.5-4B.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 250 steps of DPO
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/DPO-Qwen3.5-4B-CulturalGround/checkpoint-250 \
    --output results/Qwen3.5-4B/Qwen3.5-4B-dpo-250.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 500 steps of DPO
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/DPO-Qwen3.5-4B-CulturalGround/checkpoint-500 \
    --output results/Qwen3.5-4B/Qwen3.5-4B-dpo-500.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 750 steps of DPO
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/DPO-Qwen3.5-4B-CulturalGround/checkpoint-750 \
    --output results/Qwen3.5-4B/Qwen3.5-4B-dpo-750.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 1000 steps of DPO
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/DPO-Qwen3.5-4B-CulturalGround/checkpoint-1000 \
    --output results/Qwen3.5-4B/Qwen3.5-4B-dpo-1000.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 1250 steps of DPO
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/DPO-Qwen3.5-4B-CulturalGround/checkpoint-1250 \
    --output results/Qwen3.5-4B/Qwen3.5-4B-dpo-1250.jsonl \
    --batch_size 32 \
    --use_system_prompt

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
