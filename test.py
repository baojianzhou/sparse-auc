# -*- coding: utf-8 -*-
import os
import sys
import time
import ctypes
import pickle as pkl
import numpy as np
import scipy.io
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

try:
    sys.path.append(os.getcwd())
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


def fpr_tpr_auc(x_te, y_te, wt):
    v_py = np.dot(x_te, wt)
    v_fpr, v_tpr, _ = roc_curve(y_true=y_te, y_score=v_py)
    n_auc = roc_auc_score(y_true=y_te, y_score=v_py)
    n_auc_alter = auc(v_fpr, v_tpr)
    assert n_auc == n_auc_alter
    return v_fpr, v_tpr, n_auc


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


def test():
    # task_id = os.environ['SLURM_ARRAY_TASK_ID']
    task_id = 10
    print('test')
    print(len(load_results()), task_id)
    results_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
    results = load_results()
    g_pass = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    mat = scipy.io.loadmat('baselines/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim'] + 1000))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        fake_features = np.zeros(1000)
        nonzeros_ind = np.where(np.random.rand(1000) <= 0.05)
        fake_features[nonzeros_ind] = 1.0
        t_dat = np.concatenate((t_dat, fake_features))
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    results_mat = dict()
    para_s = 10 * int(task_id) + 123
    print(para_s)
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
            wt_ = []
            for _ in list(results['wt'][m][j]):
                wt_.append(_[0])
            wt_ = np.asarray(wt_)
            print(np.linalg.norm(wt_))
            s_t = time.time()
            opt_solam = algo_sparse_solam_cv(
                x_tr=x_tr, y_tr=y_tr, para_s=para_s,
                para_n_pass=g_pass, para_n_cv=5, verbose=0)
            print('run time for model selection: %.4f' % (time.time() - s_t))
            wt, a, b, run_time = algo_sparse_solam(x_tr=x_tr, y_tr=y_tr,
                                                   para_rand_ind=range(len(x_tr)),
                                                   para_r=opt_solam['sr'],
                                                   para_xi=opt_solam['sc'],
                                                   para_s=opt_solam['s'],
                                                   para_n_pass=opt_solam['n_pass'], verbose=0)
            v_fpr, v_tpr, n_auc = fpr_tpr_auc(x_te=x_te, y_te=y_te, wt=wt)
            print('run time: (%.6f, %.6f) ' % (run_time, results['run_time'][m][j])),
            print('auc: (%.6f, %.6f) ' % (n_auc, results['auc'][m][j])),
            print('norm(wt-wt_): %.6f' % (np.linalg.norm(wt[:123] - wt_))),
            print('speed up: %.2f' % (results['run_time'][m][j] / run_time))
            results_mat[(para_s, m, j)] = {'fpr': v_fpr, 'tpr': v_tpr, 'auc': n_auc,
                                           'optimal_opts': opt_solam}
    pkl.dump(results_mat, results_path + 'results_mat_a9a_%d.pkl' % int(task_id))


if __name__ == '__main__':
    cur = os.path.abspath(os.path.dirname(__file__))
    lib = ctypes.cdll.LoadLibrary(os.path.join(cur, ".", "sparse_module.so"))
    print(lib.main())
    print lib.c_test(np.random.rand(12).reshape((3, 4)))
    pass
