#!/bin/bash
PYTHON_PATH=/network/rit/home/bz383376/anaconda3/include/python3.7m/
NUMPY_PATH=/network/rit/home/bz383376/anaconda3/lib/python3.7/site-packages/numpy/core/include
FLAGS="-g -shared  -Wall -fPIC -lcuda -lcudart -std=c11 -O3"
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH}"
LIB="-L${OPENBLAS_LIB} -L${PYTHON_LIB} -L/usr/local/cuda/lib64"
SRC="algo_wrapper/main_gpu.c"
nvcc ${FLAGS} ${INCLUDE} ${LIB} ${SRC} -o main_gpu.so -lm
