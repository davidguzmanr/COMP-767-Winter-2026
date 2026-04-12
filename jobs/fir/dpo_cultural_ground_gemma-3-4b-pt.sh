#!/usr/bin/env bash
#SBATCH --job-name=dpo_cultural_ground_gemma-3-4b-pt
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --account=def-davlan
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

export HF_HOME=/home/davidguz/scratch/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/davidguz/scratch/Repositories/COMP-767-Winter-2026
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
# Start the main job
##################################################################
accelerate launch --num_processes 2 --mixed_precision bf16 scripts/dpo_cultural_ground.py \
    --model_name_or_path checkpoints/SFT-gemma-3-4b-pt-CulturalGround/checkpoint-20000 \
    --chat_template_source google/gemma-3-4b-it \
    --output_dir checkpoints/DPO-gemma-3-4b-pt-CulturalGround \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_steps 250 \
    --beta 0.1 \
    --dtype bfloat16 \
    --attn_implementation sdpa \
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