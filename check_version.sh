#!/bin/bash
#SBATCH --job-name=gpu-compile
#SBATCH --nodelist=ceashpc-04
#SBATCH --partition=ceashpc
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
PYTHON_PATH=/network/rit/home/bz383376/anaconda3/include/python3.7m/
NUMPY_PATH=/network/rit/home/bz383376/anaconda3/lib/python3.7/site-packages/numpy/core/include
FLAGS="-g -shared -lcuda -lcudart -O3"
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH}"
LIB="-L${OPENBLAS_LIB} -L${PYTHON_LIB} -L/usr/local/cuda/lib64"
SRC="algo_wrapper/main_gpu.c"
nvcc -o cuda_spam -shared -c algo_wrapper/cuda_spam.cu -Xcompiler -fPIC -L/usr/local/cuda/lib64
gcc ${FLAGS} ${INCLUDE} ${LIB} ${SRC} -o main_gpu.so -lm
