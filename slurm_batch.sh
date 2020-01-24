#!/bin/bash
#SBATCH --job-name=05-sto-iht
#SBATCH --nodelist=rhea-09
#SBATCH --partition=batch
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=80
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/sparse-auc/logs/array-%A.out
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python cv_simu_01.py run_ms graph_am 15 18 80
