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


def test_single_thread():
    x = np.reshape(np.random.rand(100), (10, 10))


def main():
    print('test')


if __name__ == '__main__':
    main()
