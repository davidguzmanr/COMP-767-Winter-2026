#!/usr/bin/env bash
#SBATCH --job-name=inference_mmstar
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
python scripts/inference_base_mmstar.py \
    --model_id google/gemma-3-4b-pt \
    --output results/mmstar/gemma-3-4b-pt/gemma-3-4b-pt-base.jsonl \
    --batch_size 32

##################################################################
# SFT checkpoints
##################################################################
python scripts/inference_at_checkpoint_mmstar.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-20000 \
    --output results/mmstar/gemma-3-4b-pt/gemma-3-4b-pt-sft-20000.jsonl \
    --batch_size 32 \
    --use_system_prompt
    
##################################################################
# DPO checkpoints
##################################################################
python scripts/inference_at_checkpoint_mmstar.py \
    --model_id google/gemma-3-4b-pt \
    --chat_template_source google/gemma-3-4b-it \
    --checkpoint checkpoints/DPO-gemma-3-4b-pt-CulturalGround/checkpoint-1250 \
    --output results/mmstar/gemma-3-4b-pt/gemma-3-4b-pt-dpo-1250.jsonl \
    --batch_size 32 \
    --use_system_prompt

##################################################################
# meta-llama/Llama-3.2-11B-Vision
##################################################################

##################################################################
# Base model
##################################################################
python scripts/inference_base_mmstar.py \
    --model_id meta-llama/Llama-3.2-11B-Vision \
    --output results/mmstar/Llama-3.2-11B-Vision/Llama-3.2-11B-Vision-base.jsonl \
    --batch_size 8

##################################################################
# SFT checkpoints
##################################################################
python scripts/inference_at_checkpoint_mmstar.py \
    --model_id meta-llama/Llama-3.2-11B-Vision \
    --chat_template_source meta-llama/Llama-3.2-11B-Vision-Instruct \
    --checkpoint checkpoints/SFT-Llama-3.2-11B-Vision-CulturalGround/checkpoint-20000 \
    --output results/mmstar/Llama-3.2-11B-Vision/Llama-3.2-11B-Vision-sft-20000.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# DPO checkpoints
##################################################################
python scripts/inference_at_checkpoint_mmstar.py \
    --model_id meta-llama/Llama-3.2-11B-Vision \
    --chat_template_source meta-llama/Llama-3.2-11B-Vision-Instruct \
    --checkpoint checkpoints/DPO-Llama-3.2-11B-Vision-CulturalGround/checkpoint-1250 \
    --output results/mmstar/Llama-3.2-11B-Vision/Llama-3.2-11B-Vision-dpo-1250.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# Qwen/Qwen3.5-4B-Base
##################################################################

##################################################################
# Base model
##################################################################
python scripts/inference_base_mmstar.py \
    --model_id Qwen/Qwen3.5-4B-Base \
    --output results/mmstar/Qwen3.5-4B-Base/Qwen3.5-4B-Base-base.jsonl \
    --batch_size 8

##################################################################
# SFT checkpoints
##################################################################
python scripts/inference_at_checkpoint_mmstar.py \
    --model_id Qwen/Qwen3.5-4B-Base \
    --chat_template_source Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/SFT-Qwen3.5-4B-Base-CulturalGround/checkpoint-20000 \
    --output results/mmstar/Qwen3.5-4B-Base/Qwen3.5-4B-Base-sft-20000.jsonl \
    --batch_size 8 \
    --use_system_prompt

##################################################################
# DPO checkpoints
##################################################################
python scripts/inference_at_checkpoint_mmstar.py \
    --model_id Qwen/Qwen3.5-4B-Base \
    --chat_template_source Qwen/Qwen3.5-4B \
    --checkpoint checkpoints/DPO-Qwen3.5-4B-Base-CulturalGround/checkpoint-1250 \
    --output results/mmstar/Qwen3.5-4B-Base/Qwen3.5-4B-Base-dpo-1250.jsonl \
    --batch_size 8 \
    --use_system_prompt

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
