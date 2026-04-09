#!/bin/bash
#SBATCH --job-name=gemma_ckpt_eval
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Activate environment
module load miniconda
conda activate COMP-767

# Directory containing checkpoints
CKPT_DIR="/network/scratch/g/guzmand/Repositories/COMP-767-Winter-2026/checkpoints/SFT-gemma-3-4b-it-CulturalGround"

# Get list of checkpoints
CHECKPOINTS=($(ls -d ${CKPT_DIR}/checkpoint-* | sort))

# Select checkpoint for this job
CKPT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

# Safety check
if [ -z "$CKPT" ]; then
    echo "No checkpoint found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running checkpoint: $CKPT"

# Output file
OUT_DIR="RESULTS"
mkdir -p $OUT_DIR
CKPT_NAME=$(basename $CKPT)
OUT_FILE="${OUT_DIR}/${CKPT_NAME}.jsonl"

# Run script
python3 inference_at_checkpoint.py \
    --checkpoint $CKPT \
    --output $OUT_FILE