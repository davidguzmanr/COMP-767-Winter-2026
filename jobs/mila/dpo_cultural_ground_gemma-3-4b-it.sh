#!/usr/bin/env bash
#SBATCH --job-name=dpo_cultural_ground_gemma-3-4b-it
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
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
accelerate launch --num_processes 2 --mixed_precision bf16 scripts/dpo_cultural_ground.py \
    --model_name_or_path checkpoints/SFT-gemma-3-4b-it-CulturalGround/checkpoint-20000 \
    --output_dir checkpoints/DPO-gemma-3-4b-it-CulturalGround \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --save_steps 100 \
    --beta 0.1 \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --learning_rate 1e-5 \
    --warmup_steps 63

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
