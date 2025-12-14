#!/bin/bash
#SBATCH --job-name=train_toy_lesion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/train_toy_image_lesion_%j.log
#SBATCH --error=outputs/train_toy_image_lesion_%j.err
ed

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules if needed (adjust based on your cluster setup)
# module load python/3.9
# module load cuda/11.8

# Activate conda environment if needed
# source activate inversebench
# or
# conda activate inversebench

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "Starting training..."
python train_toy_image_lesion.py

echo "Training completed at: $(date)"
