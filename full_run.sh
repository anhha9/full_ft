#!/bin/bash
#SBATCH --job-name=full_ft
#SBATCH --time=0:20:00
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=tc062-pool1
#SBATCH --partition=gpu
#SBATCH --output=slurm_out/run_jul29_1.out

export HF_HOME=/work/tc062/tc062/haanh/.cache/hugging_face_cache
export OMP_NUM_THREADS=1

# Enable wandb logging
export WANDB_PROJECT=full_fine_tune  # Replace with your wandb project name
export WANDB_ENTITY=anhha9-university-of-edinburgh    # Replace with your wandb entity/team name if applicable
export WANDB_API_KEY=475636ecc7553641495f7731aa39d130588a0c92      # Optionally, set your wandb API key if it's not already configured

# Avoids cuda driver problems
export TMPDIR=$(pwd)/tmp
mkdir -p $TMPDIR

echo "Starting job on $(date)"
source ./load_conda_env.sh

# Verify the Python script path
PYTHON_SCRIPT_PATH="/work/tc062/tc062/haanh/full_ft/full_ft.py"
if [ ! -f $PYTHON_SCRIPT_PATH ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT_PATH"
    exit 1
fi

# Start nvidia-smi monitoring in the background
nvidia-smi --loop=10 --filename=outfile_1.txt &

# Directory where checkpoints are saved
CHECKPOINT_DIR="/work/tc062/tc062/haanh/full_ft/checkpoints"

# Check for existing checkpoints in subfolders
latest_checkpoint=$(find $CHECKPOINT_DIR -type d -name "checkpoint-*" | sort -V | tail -n 1)

if [ -z "$latest_checkpoint" ]; then
    echo "No checkpoints found. Training from scratch."
    python $PYTHON_SCRIPT_PATH --checkpoint_dir $CHECKPOINT_DIR --train_from_scratch
else
    echo "Found checkpoint: $latest_checkpoint"
    python $PYTHON_SCRIPT_PATH --checkpoint_dir $latest_checkpoint
fi

echo "Job finished on $(date)"
