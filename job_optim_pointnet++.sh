#!/bin/bash 
#SBATCH --job-name=OptimPointNet
#SBATCH -o ./logs/OptimPointNet-%j.out
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH --array 85250-86250:1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

args_file=hyperparameters.txt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo the job id is $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}

arg=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" $args_file)
arg=$(echo $arg | sed "s/--radius_list '//;s/']'/]/")
echo hyperparameters: $arg

# Pass the parameters to the Python script
python3 -u train_pointnet++.py $arg



