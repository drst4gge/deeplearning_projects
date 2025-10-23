#!/bin/bash

#SBATCH -J mnist_cnn1
#SBATCH -p gpu-interactive
#SBATCH -A workshop
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00

module load python/gpu/3.11.5

export OMP_NUM_THREADS=10

cd $SLURM_SUBMIT_DIR

srun python mnist_cnn.py
