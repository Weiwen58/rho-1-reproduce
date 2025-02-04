#!/bin/bash        

# script
#SBATCH --job-name=rho-1-v0             # Job name
#SBATCH --nodes=1                       # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=8             # This needs to match Fabric(devices=...)
#SBATCH --partition=a100-8              # Partition name
#SBATCH --gres=gpu:a100:8               # Number of GPUs
#SBATCH --mem-per-cpu=7G                # Allocate memory per core
#SBATCH --cpus-per-task=16              # Number of cores
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --output=./job_log/%j.log       # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen6704@umn.edu

# module load gcc/8.2.0 python3/3.10.9_anaconda2023.03_libmamba cuda/11.8.0-gcc-7.2.0-xqzqlf2

source activate rho-1
# Check Python and GPU setup
python --version

nvidia-smi
htop        # To monitor CPU and RAM usage
free -h     # To check available system RAM

echo "Current Conda environment"
conda info --envs

# Set NCCL environment variables
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python -m litgpt pretrain --config config_hub/pretrain/rho-1-v0.yaml