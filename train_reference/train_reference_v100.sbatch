#!/bin/bash        

# script
#SBATCH --job-name=rho-1                # Job name
#SBATCH --partition=v100                # Partition name
#SBATCH --gres=gpu:v100:1               # Number of GPUs
#SBATCH --mem=40G                       # Memory total in MB (for all cores)
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --output=./job_log/%j.log       # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen6704@umn.edu

# module load gcc/8.2.0 python3/3.10.9_anaconda2023.03_libmamba cuda/11.8.0-gcc-7.2.0-xqzqlf2

# Check Python and GPU setup
python --version

nvidia-smi

echo "Current Conda environment"
conda info --envs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_reference.py