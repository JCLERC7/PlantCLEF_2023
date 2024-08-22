#!/bin/bash

#SBATCH --job-name=class_map
#SBATCH --partition=private-ruch-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/torchrun_%j.log
#SBATCH --error=logs/torchrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joel.clerc1@hes-so.ch

mkdir -p logs
ml load GCCcore/11.3.0
ml load Python/3.10.4

python3 id_class_map.py