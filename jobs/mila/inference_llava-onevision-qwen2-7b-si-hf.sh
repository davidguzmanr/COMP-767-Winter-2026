#!/usr/bin/env bash
#SBATCH --job-name=inference_llava-onevision-qwen2-7b-si-hf
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
# Base model
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id llava-hf/llava-onevision-qwen2-7b-si-hf \
    --output results/llava-onevision-qwen2-7b-si-hf-base.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# 5000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id llava-hf/llava-onevision-qwen2-7b-si-hf \
    --checkpoint checkpoints/SFT-llava-onevision-qwen2-7b-si-hf-CulturalGround/checkpoint-5000 \
    --output results/llava-onevision-qwen2-7b-si-hf-sft-5000.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 10000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id llava-hf/llava-onevision-qwen2-7b-si-hf \
    --checkpoint checkpoints/SFT-llava-onevision-qwen2-7b-si-hf-CulturalGround/checkpoint-10000 \
    --output results/llava-onevision-qwen2-7b-si-hf-sft-10000.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 15000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id llava-hf/llava-onevision-qwen2-7b-si-hf \
    --checkpoint checkpoints/SFT-llava-onevision-qwen2-7b-si-hf-CulturalGround/checkpoint-15000 \
    --output results/llava-onevision-qwen2-7b-si-hf-sft-15000.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# 20000 steps of SFT
##################################################################
python scripts/inference_at_checkpoint.py \
    --model_id llava-hf/llava-onevision-qwen2-7b-si-hf \
    --checkpoint checkpoints/SFT-llava-onevision-qwen2-7b-si-hf-CulturalGround/checkpoint-20000 \
    --output results/llava-onevision-qwen2-7b-si-hf-sft-20000.jsonl \
    --batch_size 32 \
    --use_system_prompt

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
