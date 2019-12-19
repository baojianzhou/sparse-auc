#!/bin/bash
PYTHON_PATH=/home/baojian/anaconda3/include/python3.7m/
NUMPY_PATH=/home/baojian/git/sparse-auc/venv/lib/python3.7/site-packages/numpy/core/include/
OPENBLAS_PATH=${ROOT_PATH}openblas-0.3.1/include/
OPENBLAS_LIB=${ROOT_PATH}openblas-0.3.1/lib/
PYTHON_LIB=/home/baojian/anaconda3/lib/
FLAGS="-g -shared  -Wall -fPIC -std=c11 -O3 "
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH} -I${OPENBLAS_PATH} "
LIB="-L${OPENBLAS_LIB} -L${PYTHON_LIB} "
SRC="algo_wrapper/main_wrapper.c algo_wrapper/sort.h algo_wrapper/sort.c
             algo_wrapper/auc_opt_methods.h algo_wrapper/auc_opt_methods.c 
	     algo_wrapper/fast_pcst.c algo_wrapper/fast_pcst.h"
gcc ${FLAGS} ${INCLUDE} ${LIB} ${SRC} -o sparse_module.so -lopenblas -lm
