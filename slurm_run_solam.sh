#!/bin/bash
#SBATCH -p batch # We can choose batch,ceashpc,kratos,minerva,snow
#SBATCH --cpus-per-task=30
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bzhou6@albany.edu
#SBATCH -o /network/rit/home/bz383376/slurm-%j.out
