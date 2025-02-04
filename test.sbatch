#!/bin/bash        

# script
#SBATCH --job-name=test                 # Job name
#SBATCH --nodes=1                       # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=4             # This needs to match Fabric(devices=...)
#SBATCH --partition=a100-4              # Partition name
#SBATCH --gres=gpu:a100:4               # Number of GPUs
#SBATCH --mem=80G                       # Memory total in MB (for all cores)
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --output=./job_log/%j.log       # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen6704@umn.edu

# module load gcc/8.2.0 python3/3.10.9_anaconda2023.03_libmamba cuda/11.8.0-gcc-7.2.0-xqzqlf2

source activate rho-1
# Check Python and GPU setup
python --version

nvidia-smi

echo "Current Conda environment"
conda info --envs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python -m litgpt pretrain --config config_hub/pretrain/rho-1-test.yaml