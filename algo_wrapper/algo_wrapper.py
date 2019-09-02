# -*- coding: utf-8 -*-
__all__ = ['algo_test',
           'algo_solam',
           'algo_solam_cv',
           'algo_solam_py',
           'algo_sparse_solam',
           'algo_sparse_solam_cv',
           'algo_da_solam',
           'algo_da_solam_cv',
           'fpr_tpr_auc']
import time
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

try:
    import sparse_module

    try:
        from sparse_module import c_test
        from sparse_module import c_algo_solam
        from sparse_module import c_algo_sparse_solam
        from sparse_module import c_algo_da_solam
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

    SOLAM: Stochastic Online AUC Maximization.

    Parameters:
    :param x_tr: training samples (n,p) dimension. --double
    :param y_tr: training labels (n,1) dimension. --double
    :param para_rand_ind: random shuffle the training samples. --np.int32
    :param para_r: radius of w. --double
    :param para_xi: parameter to control the learning rate. --double
    :param para_n_pass: number of pass. --int
    :param verbose: print out information. --int
    :return: statistical results
    :return:

    Reference:
    @inproceedings{ying2016stochastic,
    title={Stochastic online AUC maximization},
    author={Ying, Yiming and Wen, Longyin and Lyu, Siwei},
    booktitle={Advances in neural information processing systems},
    pages={451--459},
    year={2016}}
    ---
    """
    t_start = time.time()
    re = c_algo_solam(np.asarray(x_tr, dtype=float), np.asarray(y_tr, dtype=float),
                      np.asarray(para_rand_ind, dtype=np.int32), float(para_r), float(para_xi),
                      int(para_n_pass), int(verbose))
    wt = np.asarray(re[0])
    a = np.asarray(re[1])
    b = np.asanyarray(re[2])
    return wt, a, b, time.time() - t_start


def algo_solam_cv(x_tr, y_tr, para_n_pass, para_n_cv, verbose):
    """
    Model selection of SOLAM by checking AUC score.
    Parameters:
    :param x_tr: training samples (n,p) dimension. --double
    :param y_tr: training labels (n,1) dimension. --double
    :param para_n_pass: how many passes need to run.
    :param para_n_cv: KFold cross validation of folding parameter.
    :param verbose: print out information. --int
    :return: model parameters.
    """
    max_auc = 0.0
    opt = dict()
    for para_xi in np.arange(1, 101, 9, dtype=float):
        for para_r in 10. ** np.arange(-1, 5, 1, dtype=float):
            cur_auc = np.zeros(para_n_cv)
            kf = KFold(n_splits=para_n_cv, shuffle=True)
            for ind, (tr_ind, te_ind) in enumerate(kf.split(x_tr)):
                x_train, x_test = x_tr[tr_ind], x_tr[te_ind]
                y_train, y_test = y_tr[tr_ind], y_tr[te_ind]
                re = c_algo_solam(np.asarray(x_train, dtype=float),
                                  np.asarray(y_train, dtype=float),
                                  np.asarray(range(len(x_train)), dtype=np.int32),
                                  float(para_r), float(para_xi), int(para_n_pass), int(verbose))
                wt = np.asarray(re[0])
                cur_auc[ind] = roc_auc_score(y_true=y_test, y_score=np.dot(x_test, wt))
            mean_auc = np.mean(cur_auc)
            if mean_auc > max_auc:
                max_auc = mean_auc
                opt['sc'] = para_xi
                opt['sr'] = para_r
                opt['n_pass'] = para_n_pass
    return opt


def algo_sparse_solam(x_tr, y_tr, para_rand_ind, para_r, para_xi, para_s, para_n_pass, verbose):
    """
    Sparsity Projection with: Stochastic Online AUC Maximization
    :param x_tr: training samples (n,p) dimension. --double
    :param y_tr: training labels (n,1) dimension. --double
    :param para_rand_ind: random shuffle the training samples. --np.int32
    :param para_r: radius of w. --double
    :param para_xi: parameter to control the learning rate. --double
    :param para_s: the sparsity parameter s.
    :param para_n_pass: number of pass. --int
    :param verbose: print out information. --int
    :return: statistical results
    :return:
    """
    t_start = time.time()
    re = c_algo_sparse_solam(np.asarray(x_tr, dtype=float), np.asarray(y_tr, dtype=float),
                             np.asarray(para_rand_ind, dtype=np.int32),
                             float(para_r), float(para_xi), int(para_s), int(para_n_pass),
                             int(verbose))
    wt = np.asarray(re[0])
    a = np.asarray(re[1])
    b = np.asanyarray(re[2])
    return wt, a, b, time.time() - t_start


def algo_sparse_solam_cv(x_tr, y_tr, para_n_pass, para_s, para_n_cv, verbose):
    """
    The enhanced l1-RDA method. Shown in Algorithm 2 of reference [1].
    That is Equation (10) is equivalent to Equation (30) if rho=0.0.
    :param x_tr: training samples (n,p) dimension. --double
    :param y_tr: training labels (n,1) dimension. --double
    :param para_s: the sparsity parameter.
    :param para_n_pass: radius of w. --double
    :param para_n_cv: parameter to control the learning rate. --double
    :param verbose: print out information. --int
    :return: statistical results
    """
    max_auc = 0.0
    opt = dict()
    for para_xi in np.arange(1, 101, 9, dtype=float):
        for para_r in 10. ** np.arange(-1, 5, 1, dtype=float):
            cur_auc = np.zeros(para_n_cv)
            kf = KFold(n_splits=para_n_cv, shuffle=True)
            for ind, (tr_ind, te_ind) in enumerate(kf.split(x_tr)):
                x_train, x_test = x_tr[tr_ind], x_tr[te_ind]
                y_train, y_test = y_tr[tr_ind], y_tr[te_ind]
                re = c_algo_sparse_solam(np.asarray(x_train, dtype=float),
                                         np.asarray(y_train, dtype=float),
                                         np.asarray(range(len(x_train)), dtype=np.int32),
                                         para_r, para_xi, para_s, para_n_pass, verbose)
                wt = np.asarray(re[0])
                y_score = np.dot(x_test, wt)
                cur_auc[ind] = roc_auc_score(y_true=y_test, y_score=y_score)
            mean_auc, std_auc = float(np.mean(cur_auc)), float(np.std(cur_auc))
            print('mean: %.4f std: %.4f' % (mean_auc, std_auc), cur_auc)
            if mean_auc > max_auc:
                max_auc = mean_auc
                opt['sc'] = para_xi
                opt['sr'] = para_r
                opt['n_pass'] = para_n_pass
                opt['s'] = para_s
    return opt


def algo_da_solam(x_tr, y_tr, para_rand_ind, para_r, para_xi, para_s, para_n_pass, verbose):
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
    re = c_algo_da_solam(x_tr, y_tr, para_rand_ind, para_r, para_xi, para_s, para_n_pass, verbose)
    wt = np.asarray(re[0])
    a = np.asarray(re[1])
    b = np.asanyarray(re[2])
    return wt, a, b


def algo_da_solam_cv(x_tr, y_tr, para_n_pass, para_s, para_n_cv, verbose):
    """
    The enhanced l1-RDA method. Shown in Algorithm 2 of reference [1].
    That is Equation (10) is equivalent to Equation (30) if rho=0.0.
    :param x_tr: training samples (n,p) dimension. --double
    :param y_tr: training labels (n,1) dimension. --double
    :param para_s: the sparsity parameter.
    :param para_n_pass: radius of w. --double
    :param para_n_cv: parameter to control the learning rate. --double
    :param verbose: print out information. --int
    :return: statistical results
    """
    max_auc = 0.0
    opt = dict()
    for m in range(1, 101, 9):
        para_xi = float(m * 1.0)
        for n in np.arange(-1, 5):
            para_r = float(10. ** n)
            cur_auc = np.zeros(para_n_cv)
            kf = KFold(n_splits=para_n_cv, shuffle=True)
            for ind, (tr_ind, te_ind) in enumerate(kf.split(x_tr)):
                x_train, x_test = x_tr[tr_ind], x_tr[te_ind]
                y_train, y_test = y_tr[tr_ind], y_tr[te_ind]
                re = c_algo_da_solam(np.asarray(x_train, dtype=float),
                                     np.asarray(y_train, dtype=float),
                                     np.asarray(range(len(x_train)), dtype=np.int32),
                                     para_r, para_xi, para_s, para_n_pass, verbose)
                wt = np.asarray(re[0])
                cur_auc[ind] = roc_auc_score(y_true=y_test, y_score=np.dot(x_test, wt))
            mean_auc = np.mean(cur_auc)
            if mean_auc > max_auc:
                max_auc = mean_auc
                opt['sc'] = m
                opt['sr'] = 10. ** n
                opt['n_pass'] = para_n_pass
            print(cur_auc)
    return opt


def algo_solam_py(x_train, y_train, options, rand_id):
    """
    This is a Python version of SOLAM. It is pretty slow for large scale datasets.
    :param x_train:
    :param y_train:
    :param options:
    :param rand_id:
    :return:
    """
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


def fpr_tpr_auc(x_te, y_te, wt):
    v_py = np.dot(x_te, wt)
    v_fpr, v_tpr, _ = roc_curve(y_true=y_te, y_score=v_py)
    n_auc = roc_auc_score(y_true=y_te, y_score=v_py)
    n_auc_alter = auc(v_fpr, v_tpr)
    assert n_auc == n_auc_alter
    return v_fpr, v_tpr, n_auc
