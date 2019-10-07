#!/bin/bash
# $1: array start number
# $2: array end number
# $3: partition name
# $4: node id
# $5: python_file
#SBATCH --job-name=09-spam-l1
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/sparse-auc/logs/array_%A_%02a.out
#SBATCH --array=00-00
#SBATCH --nodelist=ceashpc-01
#SBATCH --time=96:00:00
#SBATCH --partition=ceashpc
#SBATCH --mem=8G
#SBATCH --ntasks=1
# %A_%a notation is filled in with the master job id (%A) and the array task id (%a)
# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
# /network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_13_realsim.py 
# /network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_00_simu.py 
# /network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_09_sector.py 
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_high_dim_data.py 
# /network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_16_bc.py graph_am 
