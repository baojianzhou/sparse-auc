#!/bin/bash
# $1: array start number
# $2: array end number
# $3: partition name
# $4: node id
# $5: python_file
#SBATCH --job-name=sparse-run
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/sparse-auc/logs/array_%A_%a.out
#SBATCH --array=00-24
#SBATCH --nodelist=rhea-08
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --mem=2G
#SBATCH --ntasks=1
# %A_%a notation is filled in with the master job id (%A) and the array task id (%a)
# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_13_realsim.py 
