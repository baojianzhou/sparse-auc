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
from algo_wrapper.algo_wrapper import fn_ep_solam_py


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
    g_pass = 3
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
            wt_ = []
            for _ in list(results['wt'][m][j]):
                wt_.append(_[0])
            wt_ = np.asarray(wt_)
            run_time, n_auc, n_auc_2, wt = fn_ep_solam(x_tr, y_tr, x_te, y_te, opt_solam, rand_id)
            print('run time: (%.6f, %.6f) ' % (run_time, results['run_time'][m][j])),
            print('auc: (%.6f, %.6f) ' % (n_auc, results['auc'][m][j])),
            print('norm(wt-wt_): %.6f' % (np.linalg.norm(wt - wt_))),
            print('speed up: %.2f' % (results['run_time'][m][j] / run_time))


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
    wt_c, a, b = algo_solam(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float),
                            para_rand_ind=np.asarray(range(len(x_train)), dtype=np.int32),
                            para_r=options['sr'], para_xi=options['sc'],
                            para_n_pass=options['n_pass'], verbose=0)
    run_time_c = time.time() - t_start
    v_fpr, v_tpr, n_auc, n_auc_2 = evaluate(x_test, y_test, wt_c)
    return run_time_c, n_auc, n_auc_2, wt_c


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
