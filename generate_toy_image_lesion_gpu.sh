#!/bin/bash
#SBATCH --job-name=generate_toy_lesion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/generate_toy_image_lesion_%j.log
#SBATCH --error=outputs/generate_toy_image_lesion_%j.err
#SBATCH --account=eecs  # Change this to your account name if needed

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules if needed
# module load python/3.9
# module load cuda/11.8

# Activate conda environment if needed
# source activate inversebench
# or
# conda activate inversebench

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Check if model exists
if [ ! -f "toy_image_lesion_diffusion.pt" ]; then
    echo "Error: Model file toy_image_lesion_diffusion.pt not found!"
    echo "Please train the model first using: sbatch train_toy_image_lesion_gpu.sh"
    exit 1
fi

# Run generation
echo "Starting generation..."
python scripts/generate_toy_image_lesion.py

echo "Generation completed at: $(date)"
