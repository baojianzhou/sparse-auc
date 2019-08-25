# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy import sparse
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from algo_wrapper.algo_wrapper import algo_solam


def load_results():
    import scipy.io
    results = scipy.io.loadmat('baselines/nips16_solam/EP_a9a_SOLAM.mat')['data']
    re = {'auc': np.asarray(results['AUC'])[0][0],
          'mean_auc': np.asarray(results['meanAUC'])[0][0][0][0],
          'std_auc': np.asarray(results['stdAUC'])[0][0][0][0],
          'run_time': np.asarray(results['RT'])[0][0],
          'wt': np.asarray(results['wt'])[0][0],
          'kfold_ind': np.asarray(results['vIndices'])[0][0]}
    k_fold_ind = re['kfold_ind']
    for i in range(len(k_fold_ind)):
        for j in range(len(k_fold_ind[i])):
            k_fold_ind[i][j] -= 1
    re['kfold_ind'] = np.asarray(k_fold_ind, dtype=int)
    return re


def demo():
    results = load_results()
    g_pass = 1
    g_seq_num = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    opt_solam = {'sc': 100, 'sr': 10, 'n_pass': g_pass}
    res_solam = {'AUC': np.zeros(shape=(g_iters, g_cv)),
                 'mean_AUC': 0.0,
                 'std_AUC': 0.0,
                 'run_time': np.zeros(shape=(g_iters, g_cv)),
                 'mean_run_time': 0.0}
    import scipy.io
    mat = scipy.io.loadmat('baselines/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    u_lab = np.unique(org_label)
    u_num = len(u_lab)
    if u_num > 2:
        u_sort = np.random.permutation(u_num)

    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim']))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    for m in range(g_iters):
        kfold_ind = results['kfold_ind'][m]
        kf_split = []
        for i in range(g_cv):
            train_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] != i]
            test_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] == i]
            kf_split.append((train_ind, test_ind))
        for j, (train_index, test_index) in enumerate(kf_split):
            x_tr, x_te = pp_feat[train_index], pp_feat[test_index]
            y_tr, y_te = pp_label[train_index], pp_label[test_index]
            rand_id = range(len(x_tr))
            run_time, n_auc, n_auc_2, wt = fn_ep_solam(x_tr, y_tr, x_te, y_te, opt_solam, rand_id)
            print('run time: %.6f, auc: %.6f' % (run_time, n_auc)),
            print('run time: %.6f, auc: %.6f' % (
                results['run_time'][m][j], results['auc'][m][j])),
            print('speed up: %.4f', results['run_time'][m][j] / run_time)


def fn_ep_solam(x_train, y_train, x_test, y_test, options, rand_id):
    """
    SOLAM: Stochastic Online AUC Maximization
    :param x_train: the training instances
    :param y_train: the vector of labels for x_train
    :param x_test: the testing instances
    :param y_test: the vector of labels for X_test
    :param options: a struct containing rho, sigma, C, n_label and n_tick
    :param rand_id:
    :return:
    """
    t_start = time.time()
    wt_py = fn_ep_solam_py(x_train, y_train, options, rand_id)
    run_time_py = time.time() - t_start
    t_start = time.time()
    # TODO be careful, the rand_ind type should be np.int32
    wt_c, a, b = algo_solam(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float),
                            para_rand_ind=np.asarray(range(len(x_train)), dtype=np.int32),
                            para_r=options['sr'], para_xi=options['sc'],
                            para_n_pass=options['n_pass'], verbose=0)
    run_time_c = time.time() - t_start
    v_fpr, v_tpr, n_auc, n_auc_2 = evaluate(x_test, y_test, wt_py)
    v_fpr, v_tpr, n_auc, n_auc_2 = evaluate(x_test, y_test, wt_c)
    print(np.linalg.norm(wt_py - wt_c))
    return run_time_c, n_auc, n_auc_2, wt_c


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


def evaluate(x_te, y_te, wt):
    v_py = np.dot(x_te, wt)
    v_fpr, v_tpr, _ = roc_curve(y_true=y_te, y_score=v_py)
    n_auc = roc_auc_score(y_true=y_te, y_score=v_py)
    n_auc_2 = auc(v_fpr, v_tpr)
    return v_fpr, v_tpr, n_auc, n_auc_2


def main():
    demo()


if __name__ == '__main__':
    main()
