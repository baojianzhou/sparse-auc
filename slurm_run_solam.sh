#!/bin/bash
#SBATCH --job-name=slurm-run-sparse
#SBATCH --output=array_%A_%a.out
#SBATCH --array=0-2
#SBATCH --time=01:00:00
#SBATCH --partition=ceashpc
#SBATCH --ntasks=1

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
PYTHON="/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python"
${PYTHON} slurm_run_solam.py
