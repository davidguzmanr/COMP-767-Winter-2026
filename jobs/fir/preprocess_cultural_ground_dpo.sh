#!/usr/bin/env bash
#SBATCH --job-name=preprocess_cultural_ground_dpo-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
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

countries=(
    bangladesh
    brazil
    bulgaria
    china
    czechia
    egypt
    ethiopia
    france
    germany
    greece
    india
    indonesia
    iran
    ireland
    israel
    italy
    japan
    kenya
    malaysia
    mexico
    mongolia
    netherlands
    nigeria
    norway
    pakistan
    poland
    portugal
    romania
    russia
    rwanda
    saudi_arabia
    singapore
    south_korea
    spain
    sri_lanka
    taiwan
    tanzania
    thailand
    turkey
    ukraine
    united_kingdom
    vietnam
)

for country in "${countries[@]}"; do
    python scripts/data/preprocess_cultural_ground.py \
        --countries "$country" \
        --output_format dpo \
        --reference_model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
        --dpo_rejected_batch_size 2 \
        --dpo_holdout_size 250 \
        --regenerate_chosen \
        --push_to_hub \
        --repo_id davidguzmanr/CulturalGround-dpo \
        --merge_countries False
done

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