#!/bin/bash
#SBATCH --job-name=single
#SBATCH --nodelist=ceashpc-05
#SBATCH --partition=ceashpc
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
nvidia-smi
nvcc --version

