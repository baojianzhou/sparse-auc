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

set(LOSS algo_wrapper/loss.c algo_wrapper/loss.h)
include_directories(${OPENBLAS_INCLUDE_DIR})
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(test_loss algo_wrapper/loss_test.c ${LOSS})
target_link_libraries(test_loss python2.7 -std=c11 -Wall -O3
        -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)

add_executable(test_l1ballproj algo_wrapper/condat_l1ballproj.c)
target_link_libraries(test_l1ballproj -Wall -O3 -lm)

add_executable(test_kth_selection algo_wrapper/test_kth_selection.c
        algo_wrapper/kth_selection.c algo_wrapper/kth_selection.h)
target_link_libraries(test_kth_selection -Wall -O3 -lm)

add_executable(test_case algo_wrapper/test_case.c)
target_link_libraries(test_case -Wall -O3 -lm)

add_library(sparse_module SHARED algo_wrapper/main_wrapper.c
        algo_wrapper/auc_opt_methods.h algo_wrapper/auc_opt_methods.c
        algo_wrapper/fast_pcst.c algo_wrapper/fast_pcst.h algo_wrapper/loss.c algo_wrapper/loss.h)
target_link_libraries(sparse_module python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${OPENBLAS_LIB}
        -lm -lpython2.7 -lopenblas -lpthread)