# -*- coding: utf-8 -*-
__all__ = ['algo_test', 'algo_solam', 'fn_ep_solam_py']
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
    :param x_tr: training samples (n,p) dimension. --double
    :param y_tr: training labels (n,1) dimension. --double
    :param para_rand_ind: random shuffle the training samples. --np.int32
    :param para_r: radius of w. --double
    :param para_xi: parameter to control the learning rate. --double
    :param para_n_pass: number of pass. --int
    :param verbose: print out information. --int
    :return: statistical results
    """
    re = c_algo_solam(x_tr, y_tr, para_rand_ind, para_r, para_xi, para_n_pass, verbose)
    wt = np.asarray(re[0])
    a = np.asarray(re[1])
    b = np.asanyarray(re[2])
    return wt, a, b


def fn_ep_solam_py(x_train, y_train, options, rand_id):
    sr = options['sr']
    sc = options['sc']
    n_pass = options['n_pass']
    n_dim = x_train.shape[1]
    n_p0_ = 0.  # number of positive
    n_v0_ = np.zeros(n_dim + 2)
    n_a_p0_ = 0.
    n_g_a0_ = 0.
    # initial vector
    n_v0 = np.zeros(n_dim + 2)
    n_v0[:n_dim] = np.zeros(n_dim) + np.sqrt(sr * sr / float(n_dim))
    print(np.linalg.norm(n_v0[:n_dim]))
    n_v0[n_dim] = sr
    n_v0[n_dim + 1] = sr
    n_a_p0 = 2. * sr
    # iteration time.
    n_t = 1.
    n_cnt = 1
    wt = np.zeros(n_dim)
    while True:
        if n_cnt > n_pass:
            break
        for j in range(len(rand_id)):
            id_ = rand_id[j]
            t_feat = x_train[id_]
            t_label = y_train[id_]
            n_ga = sc / np.sqrt(n_t)
            if t_label > 0:  # if it is positive case
                n_p1_ = ((n_t - 1.) * n_p0_ + 1.) / n_t
                v_wt = n_v0[:n_dim]
                n_a = n_v0[n_dim]
                v_p_dv = np.zeros_like(n_v0)
                vt_dot = np.dot(v_wt, t_feat)
                v_p_dv[:n_dim] = 2. * (1. - n_p1_) * (vt_dot - n_a) * t_feat
                v_p_dv[:n_dim] -= 2. * (1. + n_a_p0) * (1. - n_p1_) * t_feat
                v_p_dv[n_dim] = - 2. * (1. - n_p1_) * (vt_dot - n_a)
                v_p_dv[n_dim + 1] = 0.
                v_p_dv = n_v0 - n_ga * v_p_dv
                v_p_da = -2. * (1. - n_p1_) * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0
                v_p_da = n_a_p0 + n_ga * v_p_da
            else:
                n_p1_ = ((n_t - 1.) * n_p0_) / n_t
                v_wt = n_v0[:n_dim]
                n_b = n_v0[n_dim + 1]
                v_p_dv = np.zeros_like(n_v0)
                vt_dot = np.dot(v_wt, t_feat)
                v_p_dv[:n_dim] = 2. * n_p1_ * (vt_dot - n_b) * t_feat
                v_p_dv[:n_dim] += 2. * (1. + n_a_p0) * n_p1_ * t_feat
                v_p_dv[n_dim] = 0.
                v_p_dv[n_dim + 1] = - 2. * n_p1_ * (vt_dot - n_b)
                v_p_dv = n_v0 - n_ga * v_p_dv
                v_p_da = 2. * n_p1_ * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0
                v_p_da = n_a_p0 + n_ga * v_p_da

            # normalization -- the projection step.
            n_rv = np.linalg.norm(v_p_dv[:n_dim])
            if n_rv > sr:
                v_p_dv[:n_dim] = v_p_dv[:n_dim] / n_rv * sr
            if v_p_dv[n_dim] > sr:
                v_p_dv[n_dim] = sr
            if v_p_dv[n_dim + 1] > sr:
                v_p_dv[n_dim + 1] = sr
            n_v1 = v_p_dv
            n_ra = np.linalg.norm(v_p_da)
            if n_ra > 2. * sr:
                n_a_p1 = v_p_da / n_ra * (2. * sr)
            else:
                n_a_p1 = v_p_da
            # update gamma_
            n_g_a1_ = n_g_a0_ + n_ga
            # update v_
            n_v1_ = (n_g_a0_ * n_v0_ + n_ga * n_v0) / n_g_a1_
            # update alpha_
            n_a_p1_ = (n_g_a0_ * n_a_p0_ + n_ga * n_a_p0) / n_g_a1_
            # update the information
            n_p0_ = n_p1_
            n_v0_ = n_v1_
            n_a_p0_ = n_a_p1_
            n_g_a0_ = n_g_a1_
            n_v0 = n_v1
            n_a_p0 = n_a_p1
            # update the counts
            n_t = n_t + 1.
            wt = n_v1_[:n_dim]
        n_cnt += 1
    return wt
