# -*- coding: utf-8 -*-
__all__ = ['algo_test',
           'algo_solam']
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import c_test
        from sparse_module import c_algo_solam
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')


def algo_test():
    x = np.arange(1, 13).reshape(3, 4)
    sum_x = c_test(np.asarray(x, dtype=np.double))
    print('sum: %.2f' % sum_x)


def algo_solam(x_tr, y_tr, para_rand_ind, para_r, para_xi, para_n_pass, verbose):
    """
    The enhanced l1-RDA method. Shown in Algorithm 2 of reference [1].
    That is Equation (10) is equivalent to Equation (30) if rho=0.0.
    :param x_tr: training samples (n,p) dimension.
    :param y_tr: training labels (n,1) dimension.
    :param para_r: radius of w
    :param para_xi: parameter to control the learning rate.
    :param para_n_pass: number of pass
    :param para_rand_ind: random shuffle the training samples.
    :param verbose: print out information.
    :return: statistical results
    """
    re = c_algo_solam(x_tr, y_tr, para_rand_ind, para_r, para_xi, para_n_pass, verbose)
    wt = np.asarray(re[0])
    a = np.asarray(re[1])
    b = np.asanyarray(re[2])
    return wt, a, b
