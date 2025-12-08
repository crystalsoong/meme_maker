#!/bin/bash
#
# --- SLURM DIRECTIVES (Resource Allocation) ---

#SBATCH --job-name=MemeViT_SingleGPU  # Name for your job
#SBATCH --nodes=1                     # Request 1 node 
#SBATCH --ntasks=1                    # Launch 1 process/task
#SBATCH --cpus-per-task=16            # Request high CPU count for DataLoader workers
#SBATCH --mem=32G                     # Request memory
#SBATCH --time=04:00:00               # Set maximum runtime
#SBATCH --output=slurm_train_%j.out   # Send output/logs to this file

# CRITICAL: Request 1 GPU resource
#SBATCH --gres=gpu:1 


# --- ENVIRONMENT SETUP (The Fix for 'Device: cpu') ---

# Ensure all steps are correctly sourced and loaded BEFORE python runs.

# 1. 🔑 Load the necessary CUDA module
# This command is CRUCIAL for PyTorch to see the GPU drivers.
# !!! REPLACE '12.1' with the exact CUDA version used to install your PyTorch !!!
echo "Loading CUDA module..."
# module load cuda/12.1 

# 2. 🐍 Load your Python environment (Conda/Venv)
# !!! UNCOMMENT and replace with your actual environment path/name !!!
# source /path/to/your/venv/bin/activate
# OR if using modules:
# module load anaconda3/latest
# conda activate my_pytorch_env


# --- EXECUTION ---

# Change directory to where you ran sbatch
cd $SLURM_SUBMIT_DIR

# Run a quick check right before execution (optional but helpful)
echo "Verifying CUDA visibility inside job..."
python3 -c "import torch; print(f'PyTorch sees CUDA: {torch.cuda.is_available()}')"
echo "--- Starting Training ---"

# Run your training script directly
python3 models/real_train_vit.py

echo "Job finished."