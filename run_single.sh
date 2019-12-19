#!/bin/bash
#SBATCH --job-name=single
#SBATCH --nodelist=ceashpc-05
#SBATCH --partition=ceashpc
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/sparse-auc/logs/array_%A_%02a.out
/network/rit/home/bz383376/anaconda3/bin/python test_single.py
