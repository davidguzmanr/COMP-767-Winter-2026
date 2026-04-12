#!/usr/bin/env bash
#SBATCH --job-name=inference_gemma-3-4b-pt
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
# Base model (no system prompt - pretrained model, no IT format)
##################################################################
python scripts/inference_base.py \
    --model_id google/gemma-3-4b-pt \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-base.jsonl \
    --batch_size 8

##################################################################
# 4000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-4000 \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-4000.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# 8000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-8000 \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-8000.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# 12000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-12000 \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-12000.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# 16000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-16000 \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-16000.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# 20000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-20000 \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-20000.jsonl \
    --batch_size 8 \
    --use_system_prompt

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
