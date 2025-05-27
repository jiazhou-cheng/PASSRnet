#!/bin/sh

#SBATCH --job-name=YR-pre
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000051
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=~/train_YR_fixed_pre.out
#SBATCH --error=~/train_YR_fixed_pre.err

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
python train_right.py --device cuda:0 --batch_size 8 --tb_dir tensorboard_log_Y/batch_8_lr_2e-4_epoch_25_step_10_fixed_rp  # Add any other arguments as needed