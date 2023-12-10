#!/bin/bash 
#SBATCH --job-name=OptunaPointNet
#SBATCH -o ./logs/OptunaPointNet++-%j.out
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo the job id is $SLURM_JOB_ID
python3 -u pointnet_optim.py
