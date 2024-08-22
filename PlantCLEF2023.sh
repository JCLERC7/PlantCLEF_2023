#!/bin/bash

#SBATCH --job-name=multiGPU-PlantCLEF_2023
#SBATCH --partition=private-ruch-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=3
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/torchrun_%j.log
#SBATCH --error=logs/torchrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joel.clerc1@hes-so.ch

mkdir -p logs
ml load GCCcore/11.3.0
ml load Python/3.10.4


echo "Environment Information:"
which python
python --version
which torchrun

echo "GPU Information:"
nvidia-smi

NUM_PROCESSES=3

torchrun --standalone --nproc_per_node=$NUM_PROCESSES main.py -e 50
