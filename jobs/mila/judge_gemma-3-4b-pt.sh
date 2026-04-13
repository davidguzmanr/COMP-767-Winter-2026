#!/usr/bin/env bash
#SBATCH --job-name=judge_gemma-3-4b-pt
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
# Base model
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-pt/gemma-3-4b-pt-base.jsonl \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-base-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 4000 steps of SFT
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-pt/gemma-3-4b-pt-sft-4000.jsonl \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-4000-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 8000 steps of SFT
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-pt/gemma-3-4b-pt-sft-8000.jsonl \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-8000-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 12000 steps of SFT
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-pt/gemma-3-4b-pt-sft-12000.jsonl \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-12000-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 16000 steps of SFT
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-pt/gemma-3-4b-pt-sft-16000.jsonl \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-16000-judge.json \
    --model_id gpt-4o-mini

##################################################################
# 20000 steps of SFT
##################################################################
python scripts/gpt_judge.py \
    --results_file results/gemma-3-4b-pt/gemma-3-4b-pt-sft-20000.jsonl \
    --output results/gemma-3-4b-pt/gemma-3-4b-pt-sft-20000-judge.json \
    --model_id gpt-4o-mini

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
