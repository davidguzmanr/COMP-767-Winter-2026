#!/bin/bash
#SBATCH --job-name=ckpt_inference
#SBATCH --output=job_output_ckpt_inference.txt
#SBATCH --error=job_error_ckpt_inference.txt
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=main

SCRATCH_CACHE="/network/scratch/m/maltaism/"
mkdir -p $SCRATCH_CACHE

export HF_HOME=$SCRATCH_CACHE
export HF_DATASETS_CACHE=$SCRATCH_CACHE/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH_CACHE/hub
export XET_CACHE_DIR=$SCRATCH_CACHE/xet

module purge
module load cudatoolkit/12.1.1

# IMPORTANT: no module python
module load miniconda
conda activate COMP-767

# force correct python
# export PATH=/home/mila/m/maltaism/culture_drifts/.venv/bin:$PATH

which python
python -c "import sys; print(sys.executable)"

# CUDA setup
export CUDA_HOME=$EBROOTCUDATOOLKIT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME=$CUDA_HOME"
which nvcc

python -c "import torch; print(torch.__version__, torch.version.cuda)"

python /home/mila/m/maltaism/culture_drifts/sft_dpo/COMP-767-Winter-2026/inference_at_checkpoint.py