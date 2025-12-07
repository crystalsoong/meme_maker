#!/bin/bash
#
# --- SLURM DIRECTIVES (Resource Allocation) ---

#SBATCH --job-name=MemeViT_DDP        # Name for your job (updated for DDP)
#SBATCH --nodes=1                     # Request 1 node 
#SBATCH --ntasks=1                   # CRITICAL DDP CHANGE: Launch 2 tasks/processes (1 per GPU)
#SBATCH --gpus-per-task=1             # Each task is explicitly assigned 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores per task (16 total, recommended for DDP)
#SBATCH --mem=32G                     # Request memory
#SBATCH --time=04:00:00               # Set maximum runtime
#SBATCH --output=slurm_train_%j.out   # Send output/logs to this file

# The gres flag can be kept or simplified, but ntasks/gpus-per-task are stronger directives:
#SBATCH --gres=gpu:1 
# unset CUDA_VISIBLE_DEVICES if present, as torchrun handles visibility via SLURM variables
unset CUDA_VISIBLE_DEVICES 


# --- ENVIRONMENT SETUP ---

# Load any necessary modules for CUDA or your Python environment/conda
# Example: module load cuda/12.1
# Example: module load anaconda3

# --- EXECUTION ---

# Change directory to where you run sbatch (recommended)
cd $SLURM_SUBMIT_DIR

# CRITICAL DDP CHANGE: Use torchrun to launch and coordinate 2 processes
# --nproc_per_node=2 matches our #SBATCH --ntasks=2
python3 models/real_train_vit.py