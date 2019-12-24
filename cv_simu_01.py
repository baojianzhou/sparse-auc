# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import pickle as pkl
import multiprocessing
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_solam
        from sparse_module import c_algo_opauc
        from sparse_module import c_algo_sht_am
        from sparse_module import c_algo_sht_am_old
        from sparse_module import c_algo_sto_iht
        from sparse_module import c_algo_graph_am
        from sparse_module import c_algo_fsauc
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        pass
except ImportError:
    print('cannot find the module: sparse_module')
    pass

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'


def cv_solam(para):
    """ SOLAM algorithm. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_xi = np.arange(1, 101, 9, dtype=float)
    list_r = 10 ** np.arange(-1, 6, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_xi), len(list_r)))
    for fold_id, (ind_xi, para_xi), (ind_r, para_r) in product(range(k_fold), enumerate(list_xi), enumerate(list_r)):
        # only run sub-tasks for parallel
        algo_para = (trial_id, fold_id, num_passes, para_xi, para_r, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            re = c_algo_solam(sub_x_tr, sub_y_tr, para_xi, para_r, num_passes, 1000000, 0)
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_xi, ind_r] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        print(para_xi, para_r, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' solam ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_solam(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'solam'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        _, _, _, para_xi, para_r, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        step_len, verbose = 100, 0
        wt, wt_bar, auc, rts = c_algo_solam(np.asarray(data['x_tr'][tr_index], dtype=float),
                                            np.asarray(data['y_tr'][tr_index], dtype=float),
                                            para_xi, para_r, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_xi, para_r],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    return results


def cv_spam_l1(para):
    """ SPAM algorithm with l1-regularization. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_c), len(list_l1)))
    for fold_id, (ind_c, para_c), (ind_l1, para_l1) in product(range(k_fold), enumerate(list_c), enumerate(list_l1)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_l1, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (trial_id, fold_id) not in auc_wt:
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            reg_opt, step_len, verbose, para_l2_reg = 0, 1000000, 0, 0.0
            re = c_algo_spam(sub_x_tr, sub_y_tr, para_c, para_l1, para_l2_reg, reg_opt, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_c, ind_l1] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        # print(para_c, para_l1, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' spam-l1 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_spam_l1(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'spam_l1'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        _, _, _, para_c, para_l1, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        reg_opt, step_len, l2_reg, verbose = 0, 100, 0.0, 0
        wt, wt_bar, auc, rts = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                                           np.asarray(data['y_tr'][tr_index], dtype=float),
                                           para_c, para_l1, l2_reg, reg_opt, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_l1],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    return results


def cv_spam_l2(para):
    """ SPAM algorithm with l2-regularization. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_c), len(list_l2)))
    s_time = time.time()
    for fold_id, (ind_c, para_c), (ind_l2, para_l2) in product(range(k_fold), enumerate(list_c), enumerate(list_l2)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_l2, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            l1_reg, reg_opt, step_len, verbose = 0.0, 1, 1000000, 0
            re = c_algo_spam(sub_x_tr, sub_y_tr, para_c, l1_reg, para_l2, reg_opt, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_c, ind_l2] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        # print(para_c, para_l2, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' spam-l2 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_spam_l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'spam_l2'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        _, _, _, para_c, para_l2, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        reg_opt, step_len, l1_reg, verbose = 1, 100, 0.0, 0
        wt, wt_bar, auc, rts = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                                           np.asarray(data['y_tr'][tr_index], dtype=float), para_c,
                                           l1_reg, para_l2, reg_opt, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_l2],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    return results


def cv_spam_l1l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_c), len(list_l1), len(list_l2)))
    s_time = time.time()
    for fold_id, (ind_c, para_c), (ind_l1, para_l1), (ind_l2, para_l2) in \
            product(range(k_fold), enumerate(list_c), enumerate(list_l1), enumerate(list_l2)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_l1, para_l2, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            reg_opt, step_len, verbose = 0, 1000000, 0
            re = c_algo_spam(sub_x_tr, sub_y_tr, para_c, para_l1, para_l2, reg_opt, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_c, ind_l1, ind_l2] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        # print(para_c, para_l1, para_l2, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' spam-l1l2 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_spam_l1l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'spam_l1l2'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        _, _, _, para_c, para_l1, para_l2, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        reg_opt, step_len, verbose = 0, 100, 0
        wt, wt_bar, auc, rts = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                                           np.asarray(data['y_tr'][tr_index], dtype=float),
                                           para_c, para_l1, para_l2, reg_opt, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_l1, para_l2],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def cv_opauc(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_eta = 2. ** np.arange(-12, -3, 1, dtype=float)
    list_lambda = 2. ** np.arange(-10, 1, 1, dtype=float)
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_eta), len(list_lambda)))
    s_time = time.time()
    for fold_id, (ind_eta, para_eta), (ind_lambda, para_lambda) in \
            product(range(k_fold), enumerate(list_eta), enumerate(list_lambda)):
        algo_para = (trial_id, fold_id, num_passes, para_eta, para_lambda, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            re = c_algo_opauc(sub_x_tr, sub_y_tr, para_eta, para_lambda, num_passes, 1000000, 0)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_eta, ind_lambda] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        print(para_eta, para_lambda, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' opauc ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def cv_fsauc(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    list_g = 2. ** np.arange(-10, 11, 1, dtype=float)
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_r), len(list_g)))
    s_time = time.time()
    for fold_id, (ind_r, para_r), (ind_g, para_g) in product(range(k_fold), enumerate(list_r), enumerate(list_g)):
        algo_para = (trial_id, fold_id, num_passes, para_r, para_g, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            verbose, step_len = 0, 10000000
            re = c_algo_fsauc(sub_x_tr, sub_y_tr, para_r, para_g, num_passes, step_len, verbose)
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_r, ind_g] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        # print(para_r, para_g, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' opauc ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def cv_sht_am(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(20, 140, 2)
    list_c = 10. ** np.arange(-3, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_c), len(list_s)))
    for fold_id, (ind_c, para_c), (ind_s, para_s) in product(range(k_fold), enumerate(list_c), enumerate(list_s)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_s, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            b, para_l2, step_len, verbose = 50, 0.0, 1000000, 0
            re = c_algo_sht_am(sub_x_tr, sub_y_tr, para_s, b, para_c, para_l2, num_passes, step_len, verbose)
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_c, ind_s] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        print(para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_sht_am(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'sht_am'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        _, _, _, para_c, para_s, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        b, l2_reg, step_len, verbose = 50, 0.0, 100, 0
        wt, wt_bar, auc, rts = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                                             np.asarray(data['y_tr'][tr_index], dtype=float),
                                             para_s, b, para_c, l2_reg, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_s],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    return results


def cv_sto_iht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(20, 140, 2)
    list_c = 10. ** np.arange(-3, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_c), len(list_s)))
    for fold_id, (ind_c, para_c), (ind_s, para_s) in product(range(k_fold), enumerate(list_c), enumerate(list_s)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_s, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            b, para_l2, step_len, verbose = 50, 0.0, 1000000, 0
            re = c_algo_sto_iht(sub_x_tr, sub_y_tr, para_s, b, para_c, para_l2, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_c, ind_s] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        print(para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def cv_graph_am(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(20, 140, 2)
    list_c = 10. ** np.arange(-3, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for fold_id, (ind_c, para_c), (ind_s, para_s) in product(range(k_fold), enumerate(list_c), enumerate(list_s)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_s, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            edges, weights = np.asarray(data['edges'], dtype=np.int32), np.asarray(data['weights'], dtype=float)
            b, para_l2, step_len, verbose = 50, 0.0, 1000000, 0
            wt, wt_bar, _, _ = c_algo_graph_am(sub_x_tr, sub_y_tr, edges, weights, para_s, b, para_c, para_l2,
                                               num_passes, step_len, verbose)
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, np.asarray(wt)))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, np.asarray(wt_bar)))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(np.asarray(wt_bar))
            # print(para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' graph-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def test_graph_am(trial_id, fold_id, para_c, sparsity, b, num_passes, data):
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    step_len, verbose = len(tr_index), 0
    re = c_algo_graph_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                         np.asarray(data['y_tr'][tr_index], dtype=float),
                         np.asarray(data['edges'], dtype=np.int32),
                         np.asarray(data['weights'], dtype=float),
                         sparsity, b, para_c, 0.0, num_passes, step_len, verbose)
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    t_auc = np.asarray(re[3])
    return {'algo_para': [trial_id, fold_id, para_c, sparsity],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            't_auc': t_auc,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_sto_iht(trial_id, fold_id, para_c, para_l2_reg, sparsity, b, num_passes, data):
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    step_len, verbose = 1000 * len(tr_index), 0
    re = c_algo_sto_iht(np.asarray(data['x_tr'][tr_index], dtype=float),
                        np.asarray(data['y_tr'][tr_index], dtype=float),
                        sparsity, b, para_c, para_l2_reg, num_passes, step_len, verbose)
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    t_auc = np.asarray(re[2])
    return {'algo_para': [trial_id, fold_id, para_c],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            't_auc': t_auc,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_opauc(trial_id, fold_id, para_eta, para_lambda, data):
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    re = c_algo_opauc(np.asarray(data['x_tr'][tr_index], dtype=float),
                      np.asarray(data['y_tr'][tr_index], dtype=float),
                      para_eta, para_lambda, 1, 1000000, 0)
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    return {'algo_para': [trial_id, fold_id, para_eta, para_lambda],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_fsauc(trial_id, fold_id, num_passes, para_r, para_g, data):
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    re = c_algo_fsauc(np.asarray(data['x_tr'][tr_index], dtype=float),
                      np.asarray(data['y_tr'][tr_index], dtype=float),
                      para_r, para_g, num_passes, 10000000, 0)
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    return {'algo_para': [trial_id, fold_id, para_r, para_g],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_ms(method_name, num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 20, 20, [1000], [0.3]
    posi_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    results = dict()
    para_space, ms_res = [], []
    for trial_id, num_tr, mu, posi_ratio in product(range(num_trials), tr_list, mu_list, posi_ratio_list):
        for fig_i in fig_list:
            para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    if method_name == 'solam':
        ms_res = pool.map(cv_solam, para_space)
    elif method_name == 'spam_l1':
        ms_res = pool.map(cv_spam_l1, para_space)
    elif method_name == 'spam_l2':
        ms_res = pool.map(cv_spam_l2, para_space)
    elif method_name == 'spam_l1l2':
        ms_res = pool.map(cv_spam_l1l2, para_space)
    elif method_name == 'fsauc':
        ms_res = pool.map(cv_fsauc, para_space)
    elif method_name == 'sht_am':
        ms_res = pool.map(cv_sht_am, para_space)
    elif method_name == 'graph_am':
        ms_res = pool.map(cv_graph_am, para_space)
    elif method_name == 'opauc':
        ms_res = pool.map(cv_opauc, para_space)
    elif method_name == 'sto_iht':
        ms_res = pool.map(cv_sto_iht, para_space)
    pool.close()
    pool.join()
    for para, auc_wt, auc_wt_bar, cv_wt_results in ms_res:
        results[para] = dict()
        results[para][method_name] = {'auc_wt': auc_wt, 'auc_wt_bar': auc_wt_bar, 'cv_wt': cv_wt_results}
    pkl.dump(results, open(data_path + 'ms_%s.pkl' % method_name, 'wb'))


def run_testing(method_name, num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 20, 20, [1000], [0.3]
    posi_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    para_space, ms_res, results = [], [], dict()
    for trial_id, num_tr, mu, posi_ratio in product(range(num_trials), tr_list, mu_list, posi_ratio_list):
        for fig_i in fig_list:
            para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    if method_name == 'solam':
        ms_res = pool.map(test_solam, para_space)
    elif method_name == 'spam_l1':
        ms_res = pool.map(test_spam_l1, para_space)
    elif method_name == 'spam_l2':
        ms_res = pool.map(test_spam_l2, para_space)
    elif method_name == 'spam_l1l2':
        ms_res = pool.map(test_spam_l1l2, para_space)
    elif method_name == 'fsauc':
        ms_res = pool.map(cv_fsauc, para_space)
    elif method_name == 'sht_am':
        ms_res = pool.map(test_sht_am, para_space)
    elif method_name == 'graph_am':
        ms_res = pool.map(test_graph_am, para_space)
    elif method_name == 'opauc':
        ms_res = pool.map(cv_opauc, para_space)
    elif method_name == 'sto_iht':
        ms_res = pool.map(cv_sto_iht, para_space)
    pool.close()
    pool.join()
    for para, auc_wt, auc_wt_bar, cv_wt_results in ms_res:
        results[para] = dict()
        results[para][method_name] = {'auc_wt': auc_wt, 'auc_wt_bar': auc_wt_bar, 'cv_wt': cv_wt_results}
    pkl.dump(results, open(data_path + 're_%s.pkl' % method_name, 'wb'))
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            # -----------------------
            method = 'graph_am'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_c, sparsity, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = run_graph_am(trial_id, fold_id, para_c, sparsity, len(tr_index), passes,
                              data[fig_i])
            results[key][method] = re
            print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
            # -----------------------
            method = 'opauc'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_eta, para_lambda, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = run_opauc(trial_id, fold_id, para_eta, para_lambda, data[fig_i])
            results[key][method] = re
            print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
            # -----------------------
            method = 'fsauc'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_eta, para_lambda, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = run_fsauc(trial_id, fold_id, passes, para_eta, para_lambda, data[fig_i])
            results[key][method] = re
            print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
    f_name = 'results_trial_%02d.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % trial_id), 'wb'))


def run_para_sparsity():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.1, 0.5]
    fig_list = ['fig_2']
    results = dict()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        item = (trial_id, passes, num_tr, mu, posi_ratio, fig_i)
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            # -----------------------
            method = 'sht_am'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_c, _, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = []
            for sparsity in [22, 28, 34, 40, 46, 52, 58, 66, 72]:
                _ = run_sht_am(trial_id, fold_id, para_c, sparsity, 800, passes, data[fig_i])
                re.append(_['auc_wt'])
            results[key][method] = re
            print(fold_id, method, ' '.join('%.4f' % _ for _ in re))
            # -----------------------
            method = 'graph_am'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_c, _, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = []
            for sparsity in [22, 28, 34, 40, 46, 52, 58, 66, 72]:
                _ = run_graph_am(trial_id, fold_id, para_c, sparsity, 800, passes, data[fig_i])
                re.append(_['auc_wt'])
            results[key][method] = re
            print(fold_id, method, ' '.join('%.4f' % _ for _ in re))
    f_name = 'results_trial_%02d_sparsity.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % trial_id), 'wb'))


def run_para_blocksize():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.1, 0.5]
    fig_list = ['fig_2']
    results = dict()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        item = (trial_id, passes, num_tr, mu, posi_ratio, fig_i)
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            # -----------------------
            method = 'sht_am'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_c, sparsity, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = []
            for b in [16, 32, 40, 100, 200, 400, 800]:
                _ = run_sht_am(trial_id, fold_id, para_c, sparsity, b, passes, data[fig_i])
                re.append(_['auc_wt'])
            results[key][method] = re
            print(fold_id, method, ' '.join('%.4f' % _ for _ in re))
            # -----------------------
            method = 'graph_am'
            ms = pkl.load(open(data_path + 'ms_trial_%02d_%s.pkl' % (trial_id, method), 'rb'))
            _, _, _, para_c, sparsity, _ = ms[item][method][0][(trial_id, fold_id)]['para']
            re = []
            for b in [16, 32, 40, 100, 200, 400, 800]:
                _ = run_graph_am(trial_id, fold_id, para_c, sparsity, b, passes, data[fig_i])
                re.append(_['auc_wt'])
            results[key][method] = re
            print(fold_id, method, ' '.join('%.4f' % _ for _ in re))
    f_name = 'results_trial_%02d_blocksize.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % trial_id), 'wb'))


def test_solam_simu():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.5]
    fig_list = ['fig_4']
    results = dict()
    s_time = time.time()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            method = 'solam'
            list_xi = np.arange(1, 101, 9, dtype=float)
            list_r = 10 ** np.arange(-1, 6, 1, dtype=float)
            best_auc = None
            for para_xi, para_r in product(list_xi, list_r):
                re = run_solam(trial_id, fold_id, para_xi, para_r, passes, data[fig_i])
                if best_auc is None or best_auc['auc_wt'] < re['auc_wt']:
                    best_auc = re
            results[key][method] = best_auc
            print(fold_id, method, best_auc['auc_wt'],
                  best_auc['auc_wt_bar'], time.time() - s_time)


def test_spaml1_simu():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.5]
    fig_list = ['fig_2']
    results = dict()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            method = 'spam'
            list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
            list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
            best_auc, best_wt = 0.0, None
            for para_c, para_l1 in product(list_c, list_l1):
                tr_index = data[fig_i]['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
                te_index = data[fig_i]['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
                reg_opt, step_len, verbose, num_passes = 0, 10000000, 0, 10
                wt, wt_bar, auc, rts = c_algo_spam(
                    np.asarray(data[fig_i]['x_tr'][tr_index], dtype=float),
                    np.asarray(data[fig_i]['y_tr'][tr_index], dtype=float),
                    para_c, para_l1, 0.0, reg_opt, num_passes, step_len,
                    verbose)
                wt_ = []
                for _ in wt:
                    if abs(_) < 1e-2:
                        wt_.append(0.0)
                    else:
                        wt_.append(_)
                wt_ = np.asarray(wt_)
                auc1 = roc_auc_score(y_true=data[fig_i]['y_tr'][te_index],
                                     y_score=np.dot(data[fig_i]['x_tr'][te_index], wt))
                auc2 = roc_auc_score(y_true=data[fig_i]['y_tr'][te_index],
                                     y_score=np.dot(data[fig_i]['x_tr'][te_index], wt_))
                print(auc1, auc2, np.count_nonzero(wt), np.count_nonzero(wt_))
                if best_auc < auc1:
                    best_auc = auc1
                    best_wt = wt
            results[key][method] = best_auc
            import matplotlib.pyplot as plt
            plt.plot(np.sort(np.abs(best_wt))[::-1])
            plt.show()
            break


def test_spaml2_simu():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.5]
    fig_list = ['fig_2']
    results = dict()
    s_time = time.time()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            method = 'spam'
            list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
            list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
            best_auc = None
            for para_c, para_beta in product(list_c, list_beta):
                re = run_spam_l2(trial_id, fold_id, para_c, para_beta, passes, data[fig_i])
                if best_auc is None or best_auc['auc_wt'] < re['auc_wt']:
                    best_auc = re
            results[key][method] = best_auc
            print(fold_id, method, best_auc['auc_wt'],
                  best_auc['auc_wt_bar'], time.time() - s_time)


def test_sht_am_simu():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.5]
    fig_list = ['fig_2']
    results = dict()
    s_time = time.time()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        for fold_id in range(k_fold):
            key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
            results[key] = dict()
            method = 'spam'
            list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
            list_sparsity = [46]
            best_auc = None
            for para_c, para_sparsity in product(list_c, list_sparsity):
                re = run_sht_am(trial_id, fold_id, para_c, para_sparsity, 100, passes, data[fig_i])
                if best_auc is None or best_auc['auc_wt'] < re['auc_wt']:
                    best_auc = re
            results[key][method] = best_auc
            print(fold_id, method, best_auc['auc_wt'],
                  best_auc['auc_wt_bar'], time.time() - s_time)


def test_fsauc_simu():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        trial_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        trial_id = 0
    k_fold, passes = 5, 10
    tr_list, mu_list, posi_ratio_list, fig_list = [1000], [0.3], [0.5], ['fig_4']
    results = dict()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))
        fold_id = 0
        key = (trial_id, fold_id, passes, num_tr, mu, posi_ratio, fig_i)
        results[key] = dict()
        list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
        list_g = 2. ** np.arange(-10, -2, 1, dtype=float)
        for para_r, para_g in product(list_r, list_g):
            re = run_fsauc(trial_id, fold_id, passes, para_r, para_g, data[fig_i])
            print(para_r, para_g, re['auc_wt'])


def cv_sht_am_v1():
    k_fold = 5
    num_passes = 20
    trial_id = 0
    data = pkl.load(open(
        os.path.join(data_path, 'data_trial_%02d_tr_1000_mu_0.3_p-ratio_0.3.pkl' % trial_id), 'r'))
    data = data['fig_1']
    list_c = 2. ** np.arange(-2., 1., 0.2)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for fold_id, para_c, sparsity, b in product(
            range(k_fold), list_c, [20, 30, 40, 50, 60], [20, 40]):
        # only run sub-tasks for parallel
        algo_para = (trial_id, fold_id, num_passes, para_c, sparsity, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (trial_id, fold_id) not in auc_wt:
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        step_len, verbose = 100000000, 0
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            re = c_algo_sht_am(np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float),
                               np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                               sparsity, b, para_c, 0.0, num_passes, step_len, verbose)
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(
                np.mean(list_num_nonzeros_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def main():
    run_testing(method_name=sys.argv[1], num_cpus=int(sys.argv[2]))


if __name__ == '__main__':
    main()
