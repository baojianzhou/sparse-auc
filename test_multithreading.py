# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import multiprocessing

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_test
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')


def test_single_thread(para):
    n = para
    x = np.reshape(np.random.rand(n), (10, 10))
    print('test')
    return c_test(x)


def test_multi_process(cpus):
    para_space = []
    for i in range(1000):
        para_space.append(100)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(test_single_thread, para_space)
    pool.close()
    pool.join()
    return np.sum(ms_res)


def main():
    print(test_multi_process(cpus=5))


if __name__ == '__main__':
    main()
