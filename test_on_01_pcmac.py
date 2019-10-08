# -*- coding: utf-8 -*-
import os
import time
import scipy.io
import numpy as np
import pickle as pkl
from scipy.sparse import csc_matrix
from sklearn.model_selection import KFold
from algo_wrapper.algo_wrapper import fpr_tpr_auc
from algo_wrapper.algo_wrapper import algo_sparse_solam
from algo_wrapper.algo_wrapper import algo_sparse_solam_cv

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/01-pcmac/'


def load_results():
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


def get_para_by_task_id(task_id):
    s_para = 123
    return 10 * int(task_id) + s_para


def test_single(task_id, results_path):
    results = load_results()
    g_pass = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    import scipy.io
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
    para_s = get_para_by_task_id(task_id)
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
    pkl.dump(results_mat, open(results_path + 'results_mat_a9a_%d.pkl' % int(task_id), 'wb'))


def test_run():
    print('test loading ...')
    results_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
    test_single(task_id=os.environ['SLURM_ARRAY_TASK_ID'], results_path=results_path)


def main():
    data = load_dataset()
    pass


if __name__ == '__main__':
    main()
