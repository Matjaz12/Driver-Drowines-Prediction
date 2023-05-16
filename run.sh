#!/bin/sh
#SBATCH --job-name=dl_proj # job name
#SBATCH --output=slurm_out.log
#SBATCH --time=02:00:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH --ntasks=1 # number of tasks
#SBATCH --mem-per-gpu=16G # memory allocation
#SBATCH --partition=gpu # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12 # number of allocated cores

## source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
## conda activate dl_env # activate the previously created environment
source /d/hpc/projects/FRI/DL/mm1706/miniconda3/etc/profile.d/conda.sh # init. conda
conda activate /d/hpc/projects/FRI/DL/mm1706/miniconda3/envs/dl2_env
srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 python $1 $2 $3