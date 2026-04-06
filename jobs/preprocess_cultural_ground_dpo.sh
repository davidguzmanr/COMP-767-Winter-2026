#!/usr/bin/env bash
#SBATCH --job-name=preprocess_cultural_ground_dpo
#SBATCH --partition=long	
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
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
python scripts/data/preprocess_cultural_ground.py \
    --countries all \
    --output_format dpo \
    --reference_model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dpo_rejected_batch_size 16 \
    --dpo_holdout_size 250 \
    --regenerate_chosen \
    --push_to_hub \
    --repo_id davidguzmanr/CulturalGround-dpo

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"