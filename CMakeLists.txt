cmake_minimum_required(VERSION 2.8)
include_directories(algo_wrapper/)
project(sparse-auc C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC -lpython2.7 -pthread")
set(ROOT /network/rit/lab/ceashpc/bz383376/opt/)
set(NUMPY env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/)
set(OPENBLAS_INCLUDE_DIR "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_INCLUDE "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_LIB "${ROOT}openblas-0.3.1/lib/")
set(NUMPY_INCLUDE "/network/rit/lab/ceashpc/bz383376/opt/${NUMPY}")
set(PYTHON_LIBRARY "${ROOT}python-2.7.14/lib/libpython2.7.so")
set(PYTHON_INCLUDE "${ROOT}python-2.7.14/include/python2.7/")
include_directories(${OPENBLAS_INCLUDE_DIR})
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(test_kth_selection test_kth_selection.c
<<<<<<< HEAD
        algo_wrapper/kth_selection.c algo_wrapper/kth_selection.h)
=======
        algo_wrapper/kth_selection.c algo_wrapper/kth_selection.h algo_wrapper/algo_base.h test_auc_curve.c)
>>>>>>> 823b15f01d6bf052b332376080afdcd3d7da58ba
target_link_libraries(test_kth_selection -Wall -O3 -lm)

add_executable(test_main main.c algo_wrapper/sort.h algo_wrapper/sort.c algo_wrapper/auc_opt_methods.h algo_wrapper/auc_opt_methods.c)
target_link_libraries(test_main python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE} -L${OPENBLAS_LIB}
        -lm -lpython2.7 -lopenblas -lpthread)

<<<<<<< HEAD
add_library(sparse_module SHARED algo_wrapper/main_wrapper.c
        algo_wrapper/sort.c algo_wrapper/sort.h)
=======
add_library(sparse_module SHARED algo_wrapper/main_wrapper.c algo_wrapper/algo_solam.c
        algo_wrapper/algo_solam.h algo_wrapper/algo_stoht_am.c
        algo_wrapper/algo_stoht_am.h algo_wrapper/algo_da_solam.c algo_wrapper/algo_da_solam.h
        algo_wrapper/sort.c algo_wrapper/sort.h algo_wrapper/auc_curve.c algo_wrapper/auc_curve.h)
>>>>>>> 823b15f01d6bf052b332376080afdcd3d7da58ba
target_link_libraries(sparse_module python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${OPENBLAS_LIB}
        -lm -lpython2.7 -lopenblas -lpthread)