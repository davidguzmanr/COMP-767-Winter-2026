#!/usr/bin/env bash
#SBATCH --job-name=judge_gemma-3-4b-it
#SBATCH --partition=long-cpu	
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
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
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-it/gemma-3-4b-it.jsonl \
    --output results/gemma-3-4b-it/gemma-3-4b-it-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 250 steps of DPO
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-it/gemma-3-4b-it-dpo-250.jsonl \
    --output results/gemma-3-4b-it/gemma-3-4b-it-dpo-250-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 500 steps of DPO
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-it/gemma-3-4b-it-dpo-500.jsonl \
    --output results/gemma-3-4b-it/gemma-3-4b-it-dpo-500-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 750 steps of DPO
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-it/gemma-3-4b-it-dpo-750.jsonl \
    --output results/gemma-3-4b-it/gemma-3-4b-it-dpo-750-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 1000 steps of DPO
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-it/gemma-3-4b-it-dpo-1000.jsonl \
    --output results/gemma-3-4b-it/gemma-3-4b-it-dpo-1000-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 1250 steps of DPO
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-it/gemma-3-4b-it-dpo-1250.jsonl \
    --output results/gemma-3-4b-it/gemma-3-4b-it-dpo-1250-judge.json \
    --model_id gpt-4o-mini

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"