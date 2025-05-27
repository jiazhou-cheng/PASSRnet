#!/bin/sh

#SBATCH --job-name=train-X
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000051
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=~/train_X_2.out
#SBATCH --error=~/train_X_2.err

# Load modules
module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75
module load python/3.10  # Adjust to the right Python module for your cluster

# Activate virtual environment (adjust this path)
source ~/.virtualenvs/sfo/bin/activate

# Go to the working directory (where train.py is)
cd /users/chengjz/PASSRnet/

# Run your script
python train_x.py --device cuda:0 --batch_size 2 --tb_dir tensorboard_log_X/batch_2_lr_2e-4_epoch_20_step_10  # Add any other arguments as needed