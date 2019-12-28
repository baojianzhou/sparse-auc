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
        from sparse_module import c_algo_solam
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_sht_am
        from sparse_module import c_algo_graph_am
        from sparse_module import c_algo_opauc
        from sparse_module import c_algo_sto_iht
        from sparse_module import c_algo_hsg_ht
        from sparse_module import c_algo_fsauc
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        pass
except ImportError:
    print('cannot find the module: sparse_module')
    pass

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'


def test_solam(para):
    def get_ms_file():
        if 0 <= trial_id < 5:
            return '00_05'
        elif 5 <= trial_id < 10:
            return '05_10'
        elif 10 <= trial_id < 15:
            return '10_15'
        else:
            return '15_20'

    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s_solam.pkl' % (get_ms_file()), 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_xi, para_r, _ = ms[para]['solam']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        step_len, verbose, none_arr = 100, 0, np.asarray([0.0], dtype=np.int32)
        wt, wt_bar, auc, rts = c_algo_solam(x_tr, none_arr, none_arr, none_arr, y_tr, 0, data['p'],
                                            para_xi, para_r, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_xi, para_r],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    sys.stdout.flush()
    return results


def cv_solam(para):
    """ SOLAM algorithm. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_xi = np.arange(1, 101, 9, dtype=float)
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_xi), len(list_r)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_r, para_r) in product(range(k_fold), enumerate(list_xi), enumerate(list_r)):
        s_time = time.time()
        algo_para = (para_xi, para_r, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_solam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_r)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_r] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
            auc_wt[(trial_id, fold_id)]['epochs'] = float(np.mean(list_epochs))
        print("trial-%d fold-%d para_xi: %.1e para_r: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_r, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def cv_spam_l1(para):
    """ SPAM algorithm with l1-regularization. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_c), len(list_l1)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_l1, para_l1) in product(range(k_fold), enumerate(list_c), enumerate(list_l1)):
        s_time = time.time()
        algo_para = (para_xi, para_l1, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_l1, 0.0)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_l1] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
            auc_wt[(trial_id, fold_id)]['epochs'] = float(np.mean(list_epochs))
        print("trial-%d fold-%d para_xi: %.1e para_l1: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_l1, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_spam_l1(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'spam_l1'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_c, para_l1, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        reg_opt, step_len, para_l2, verbose = 0, 100, 0.0, 0
        wt, wt_bar, auc, rts = c_algo_spam(x_tr, None, None, None, y_tr, 0, data['p'],
                                           para_c, para_l1, para_l2, reg_opt, num_passes, step_len, verbose)
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
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_c), len(list_l2)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_l2, para_l2) in product(range(k_fold), enumerate(list_c), enumerate(list_l2)):
        s_time = time.time()
        algo_para = (para_xi, para_l2, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_l2] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
            auc_wt[(trial_id, fold_id)]['epochs'] = float(np.mean(list_epochs))
        print("trial-%d fold-%d para_xi: %.1e para_l2: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_l2, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_spam_l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'spam_l2'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_c, para_l2, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        reg_opt, step_len, l1_reg, verbose = 1, 100, 0.0, 0
        wt, wt_bar, auc, rts = c_algo_spam(x_tr, None, None, None, y_tr, 0, data['p'], para_c,
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
    """SPAM algorithm with elastic-net """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_c), len(list_l1), len(list_l2)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_l1, para_l1), (ind_l2, para_l2) in \
            product(range(k_fold), enumerate(list_c), enumerate(list_l1), enumerate(list_l2)):
        s_time = time.time()
        algo_para = (para_xi, para_l1, para_l2, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_l1, para_l2)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_l1, ind_l2] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d para_xi: %.1e para_l2: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_l2, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_spam_l1l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'spam_l1l2'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_c, para_l1, para_l2, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        reg_opt, step_len, verbose = 0, 100, 0
        wt, wt_bar, auc, rts = c_algo_spam(x_tr, None, None, None, y_tr, 0, data['p'],
                                           para_c, para_l1, para_l2, reg_opt, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_l1, para_l2],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    sys.stdout.flush()
    return results


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
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    list_g = 2. ** np.arange(-10, 11, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_r), len(list_g)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_r, para_r), (ind_g, para_g) in product(range(k_fold), enumerate(list_r), enumerate(list_g)):
        s_time = time.time()
        algo_para = (para_r, para_g, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_fsauc(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_r, para_g)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_r, ind_g] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d para_r: %.1e para_g: %.3e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_r, para_g, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_fsauc(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'fsauc'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_r, para_g, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        step_len, verbose = 100, 0
        wt, wt_bar, auc, rts = c_algo_fsauc(np.asarray(data['x_tr'][tr_index], dtype=float),
                                            np.asarray(data['y_tr'][tr_index], dtype=float),
                                            para_r, para_g, num_passes, step_len, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_r, para_g],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    sys.stdout.flush()
    return results


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
            b, para_l2, record_aucs, verbose = 50, 0.0, 0, 0
            re = c_algo_sht_am(sub_x_tr, sub_y_tr, para_s, b, para_c, para_l2, num_passes, record_aucs, verbose)
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
        print(trial_id, fold_id, fig_i)
        _, _, _, para_c, para_s, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        b, l2_reg, record_aucs, verbose = 50, 0.0, 1, 0
        wt, wt_bar, auc, rts = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                                             np.asarray(data['y_tr'][tr_index], dtype=float),
                                             para_s, b, para_c, l2_reg, num_passes, record_aucs, verbose)
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
            b, para_l2, is_sparse, record_aucs, verbose = 50, 0.0, 0, 0, 0
            re = c_algo_sto_iht(sub_x_tr, sub_y_tr, para_s, b, 0, 0, para_c, para_l2, num_passes, verbose)
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
        print(para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar), time.time() - s_time)
    run_time = time.time() - s_time
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_sto_iht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'sto_iht'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_c, para_s, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        b, para_l2, record_aucs, verbose = 50, 0.0, 1, 0
        wt, wt_bar, auc, rts = c_algo_sto_iht(x_tr, y_tr, para_s, b, 0, 1, para_c, para_l2, num_passes, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_s],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    sys.stdout.flush()
    return results


def cv_hsg_ht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(100, 501, 50)
    list_c = [1e-3, 1e-2, 1e-1, 1e0, 3e0, 1e1]
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
            para_l2, is_sparse, record_aucs, verbose = 0.0, 0, 0, 0
            para_tau, para_zeta = 1.0, 1.033
            re = c_algo_hsg_ht(sub_x_tr, sub_y_tr, para_s, is_sparse, record_aucs,
                               para_tau, para_zeta, para_c, para_l2, num_passes, verbose)
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
        print(para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar), time.time() - s_time)
    run_time = time.time() - s_time
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_hsg_ht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    method = 'hsg_ht'
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    ms = pkl.load(open(data_path + 'ms_%s.pkl' % method, 'rb'))
    results = dict()
    for fold_id in range(k_fold):
        print(trial_id, fold_id, fig_i)
        _, _, _, para_c, para_s, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        b, para_l2, is_sparse, record_aucs, verbose = 50, 0.0, 0, 1, 0
        para_tau, para_zeta = 1.0, 1.033
        wt, wt_bar, auc, rts = c_algo_hsg_ht(x_tr, y_tr, para_s, is_sparse, record_aucs,
                                             para_tau, para_zeta, para_c, para_l2, num_passes, verbose)
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        results[item] = {'algo_para': [trial_id, fold_id, para_c, para_s],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                                     y_score=np.dot(data['x_tr'][te_index], wt_bar)),
                         'auc': auc, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt),
                         'nonzero_wt_bar': np.count_nonzero(wt_bar)}
    sys.stdout.flush()
    return results


def cv_graph_am(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(20, 140, 2)
    list_s = [26, 46, 92, 132]
    list_c = 10. ** np.arange(-2, 1, 1, dtype=float)
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
            edges, weights = np.asarray(data['edges'], dtype=np.int32), np.asarray(data['weights'], dtype=float)
            none_arr = np.asarray([0.0], dtype=np.int32)
            b, para_l2, verbose = 50, 0.0, 0
            wt, wt_bar, _, _ = c_algo_graph_am(sub_x_tr, none_arr, none_arr, none_arr, sub_y_tr, edges, weights, 0, 0,
                                               data['p'], para_s, b, para_c, para_l2, num_passes, verbose)
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, np.asarray(wt)))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, np.asarray(wt_bar)))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(np.asarray(wt_bar))
        cv_wt_results[ind_c, ind_s] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        print(trial_id, fold_id, para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar), time.time() - s_time)
        sys.stdout.flush()
    run_time = time.time() - s_time
    print('-' * 40 + ' graph-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    sys.stdout.flush()
    return para, auc_wt, auc_wt_bar, cv_wt_results


def test_graph_am(trial_id, fold_id, para_c, sparsity, b, num_passes, data):
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    edges, weights = np.asarray(data['edges'], dtype=np.int32), np.asarray(data['weights'], dtype=float)
    step_len, verbose = len(tr_index), 0
    re = c_algo_graph_am(x_tr, None, None, None, y_tr, edges, weights, 0, 0,
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


def run_ms(method_name, trial_id_low, trial_id_high, num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 5, 50, [1000], [0.3]
    posi_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    results = dict()
    para_space, ms_res = [], []
    for trial_id in range(trial_id_low, trial_id_high):
        for fig_i in fig_list:
            for num_tr, mu, posi_ratio in product(tr_list, mu_list, posi_ratio_list):
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
    elif method_name == 'hsg_ht':
        ms_res = pool.map(cv_hsg_ht, para_space)
    pool.close()
    pool.join()
    for para, auc_wt, cv_wt_results in ms_res:
        results[para] = dict()
        results[para][method_name] = {'auc_wt': auc_wt, 'cv_wt': cv_wt_results}
    f_name = 'ms_%02d_%02d_%s.pkl' % (trial_id_low, trial_id_high, method_name)
    pkl.dump(results, open(data_path + f_name, 'wb'))


def run_testing(method_name, num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 20, 20, [1000], [0.3]
    posi_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    para_space, test_res, results = [], [], dict()
    for trial_id, num_tr, mu, posi_ratio in product(range(num_trials), tr_list, mu_list, posi_ratio_list):
        for fig_i in fig_list:
            para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    if method_name == 'solam':
        test_res = pool.map(test_solam, para_space)
    elif method_name == 'spam_l1':
        test_res = pool.map(test_spam_l1, para_space)
    elif method_name == 'spam_l2':
        test_res = pool.map(test_spam_l2, para_space)
    elif method_name == 'spam_l1l2':
        test_res = pool.map(test_spam_l1l2, para_space)
    elif method_name == 'fsauc':
        test_res = pool.map(test_fsauc, para_space)
    elif method_name == 'sht_am':
        test_res = pool.map(test_sht_am, para_space)
    elif method_name == 'graph_am':
        test_res = pool.map(test_graph_am, para_space)
    elif method_name == 'opauc':
        test_res = pool.map(cv_opauc, para_space)
    elif method_name == 'sto_iht':
        test_res = pool.map(test_sto_iht, para_space)
    elif method_name == 'hsg_ht':
        test_res = pool.map(test_hsg_ht, para_space)
    pool.close()
    pool.join()
    results = {key: val for d in test_res for key, val in d.items()}
    pkl.dump(results, open(data_path + 're_%s.pkl' % method_name, 'wb'))


def run_para_s(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(20, 140, 2)
    list_c = 10. ** np.arange(-3, 3, 1, dtype=float)
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
            b, para_l2, record_aucs, verbose = 50, 0.0, 0, 0
            re = c_algo_sht_am(sub_x_tr, sub_y_tr, para_s, b, para_c, para_l2, num_passes, record_aucs, verbose)
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


def run_para_sparsity(num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list, = 5, 20, 20, [1000], [0.3]
    posi_ratio_list, fig_list = [0.3], ['fig_2']
    para_space = []
    for trial_id, num_tr, mu, posi_ratio in product(range(num_trials), tr_list, mu_list, posi_ratio_list):
        for fig_i in fig_list:
            para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    test_res = pool.map(run_para_s, para_space)
    pool.close()
    pool.join()


def run_para_b(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    list_s = range(20, 140, 2)
    list_c = 10. ** np.arange(-3, 3, 1, dtype=float)
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
            b, para_l2, record_aucs, verbose = 50, 0.0, 0, 0
            re = c_algo_sht_am(sub_x_tr, sub_y_tr, para_s, b, para_c, para_l2, num_passes, record_aucs, verbose)
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


def run_para_blocksize(num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list, = 5, 20, 20, [1000], [0.3]
    posi_ratio_list, fig_list = [0.3], ['fig_2']
    para_space = []
    for trial_id, num_tr, mu, posi_ratio in product(range(num_trials), tr_list, mu_list, posi_ratio_list):
        for fig_i in fig_list:
            para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    test_res = pool.map(run_para_s, para_space)
    pool.close()
    pool.join()


def show_result_01():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 8
    fig, ax = plt.subplots(2, 2, sharex=True)
    for ii, jj in product(range(2), range(2)):
        ax[ii, jj].grid(color='lightgray', linestyle='dotted', axis='both')
        ax[ii, jj].spines['right'].set_visible(False)
        ax[ii, jj].spines['top'].set_visible(False)

    color_list = ['r', 'g', 'm', 'b', 'y', 'k', 'orangered', 'olive', 'dogdeblue', 'darkgray', 'darkorange']
    marker_list = ['s', 'o', 'P', 'X', 'H', '*', 'x', 'v', '^', '+', '>']
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'fsauc', 'spam_l1l2', 'solam', 'sto_iht', 'hsg_ht']
    method_label_list = ['SHT-AM', r"SPAM-$\displaystyle \ell^1$", r"SPAM-$\displaystyle \ell^2$", 'FSAUC',
                         r"SPAM-$\displaystyle \ell^1/\ell^2$", r"SOLAM", r"StoIHT", 'HSG-HT']
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    posi_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for ind_method, method in enumerate(method_list):
        results = pkl.load(open(os.path.join(data_path, 're_%s.pkl' % method)))
        for ind_fig, fig_i in enumerate(fig_list):
            re = []
            for posi_ratio in posi_ratio_list:
                tmp = [results[key]['auc_wt'] for key in results if key[-1] == fig_i and key[-2] == posi_ratio]
                re.append(np.mean(tmp))
            ax[ind_fig / 2, ind_fig % 2].plot(posi_ratio_list, re,
                                              marker=marker_list[ind_method],
                                              color=color_list[ind_method],
                                              label=method_label_list[ind_method])
    for ind_fig in range(4):
        ax[ind_fig / 2, ind_fig % 2].set_title(r"Network %d" % (ind_fig + 1))
    ax[0, 0].set_ylabel('AUC')
    ax[1, 0].set_ylabel('AUC')
    ax[1, 0].set_xlabel('Positive Ratio')
    ax[1, 1].set_xlabel('Positive Ratio')
    ax[1, 1].legend(loc='lower right', framealpha=1., bbox_to_anchor=(1.0, 0.0),
                    fontsize=14., frameon=True, borderpad=0.1,
                    labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax[0, 0].set_xticks(posi_ratio_list)
    ax[0, 1].set_xticks(posi_ratio_list)
    ax[1, 0].set_xticks(posi_ratio_list)
    ax[1, 1].set_xticks(posi_ratio_list)
    ax[0, 0].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    ax[0, 1].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    for i in range(2):
        ax[0, i].set_ylim([0.56, .96])
    for i in range(2):
        ax[1, i].set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax[1, i].set_ylim([0.81, 1.01])
    # ax[1, 1].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    plt.savefig(root_path + 'simu-result-01.pdf', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def show_result_02():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 8
    fig, ax = plt.subplots(2, 2, sharex=True)
    for ii, jj in product(range(2), range(2)):
        ax[ii, jj].grid(color='lightgray', linestyle='dotted', axis='both')
        ax[ii, jj].spines['right'].set_visible(False)
        ax[ii, jj].spines['top'].set_visible(False)

    color_list = ['b', 'g', 'm', 'r', 'y']
    marker_list = ['X', 'o', 'P', 's', 'H']
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'fsauc', 'spam_l1l2']
    method_label_list = ['SHT-AM', r"SPAM-$\displaystyle \ell^1$", r"SPAM-$\displaystyle \ell^2$", 'FSAUC',
                         r"SPAM-$\displaystyle \ell^1/\ell^2$"]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    posi_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for ind_method, method in enumerate(method_list):
        results = pkl.load(open(os.path.join(data_path, 're_%s.pkl' % method)))
        for ind_fig, fig_i in enumerate(fig_list):
            re = []
            for posi_ratio in posi_ratio_list:
                tmp = [results[key]['auc_wt'] for key in results if key[-1] == fig_i and key[-2] == posi_ratio]
                re.append(np.mean(tmp))
            ax[ind_fig / 2, ind_fig % 2].plot(posi_ratio_list, re,
                                              marker=marker_list[ind_method],
                                              color=color_list[ind_method],
                                              label=method_label_list[ind_method])
    for ind_fig in range(4):
        ax[ind_fig / 2, ind_fig % 2].set_title(r"Network %d" % (ind_fig + 1))
    ax[0, 0].set_ylabel('AUC')
    ax[1, 0].set_ylabel('AUC')
    ax[1, 0].set_xlabel('Positive Ratio')
    ax[1, 1].set_xlabel('Positive Ratio')
    ax[1, 1].legend(loc='lower right', framealpha=1., bbox_to_anchor=(1.0, 0.0),
                    fontsize=14., frameon=True, borderpad=0.1,
                    labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax[0, 0].set_xticks(posi_ratio_list)
    ax[0, 1].set_xticks(posi_ratio_list)
    ax[1, 0].set_xticks(posi_ratio_list)
    ax[1, 1].set_xticks(posi_ratio_list)
    ax[0, 0].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    ax[0, 1].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    for i in range(2):
        ax[0, i].set_ylim([0.56, .96])
    for i in range(2):
        ax[1, i].set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax[1, i].set_ylim([0.81, 1.01])
    # ax[1, 1].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    plt.savefig(root_path + 'simu-result-01', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def show_result_03():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 8
    fig, ax = plt.subplots(2, 2, sharex=True)
    for ii, jj in product(range(2), range(2)):
        ax[ii, jj].grid(color='lightgray', linestyle='dotted', axis='both')
        ax[ii, jj].spines['right'].set_visible(False)
        ax[ii, jj].spines['top'].set_visible(False)

    color_list = ['b', 'g', 'm', 'r', 'y']
    marker_list = ['X', 'o', 'P', 's', 'H']
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'fsauc', 'spam_l1l2']
    method_label_list = ['SHT-AM', r"SPAM-$\displaystyle \ell^1$", r"SPAM-$\displaystyle \ell^2$", 'FSAUC',
                         r"SPAM-$\displaystyle \ell^1/\ell^2$"]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    posi_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for ind_method, method in enumerate(method_list):
        results = pkl.load(open(os.path.join(data_path, 're_%s.pkl' % method)))
        for ind_fig, fig_i in enumerate(fig_list):
            re = []
            for posi_ratio in posi_ratio_list:
                tmp = [results[key]['auc_wt'] for key in results if key[-1] == fig_i and key[-2] == posi_ratio]
                re.append(np.mean(tmp))
            ax[ind_fig / 2, ind_fig % 2].plot(posi_ratio_list, re,
                                              marker=marker_list[ind_method],
                                              color=color_list[ind_method],
                                              label=method_label_list[ind_method])
    for ind_fig in range(4):
        ax[ind_fig / 2, ind_fig % 2].set_title(r"Network %d" % (ind_fig + 1))
    ax[0, 0].set_ylabel('AUC')
    ax[1, 0].set_ylabel('AUC')
    ax[1, 0].set_xlabel('Positive Ratio')
    ax[1, 1].set_xlabel('Positive Ratio')
    ax[1, 1].legend(loc='lower right', framealpha=1., bbox_to_anchor=(1.0, 0.0),
                    fontsize=14., frameon=True, borderpad=0.1,
                    labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax[0, 0].set_xticks(posi_ratio_list)
    ax[0, 1].set_xticks(posi_ratio_list)
    ax[1, 0].set_xticks(posi_ratio_list)
    ax[1, 1].set_xticks(posi_ratio_list)
    ax[0, 0].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    ax[0, 1].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    for i in range(2):
        ax[0, i].set_ylim([0.56, .96])
    for i in range(2):
        ax[1, i].set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax[1, i].set_ylim([0.81, 1.01])
    # ax[1, 1].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    plt.savefig(root_path + 'simu-result-01', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def merge_ms(method):
    results = dict()
    if method == 'sto_iht':
        for ii, jj in zip(['00', '02', '04', '06', '08', '10', '12', '14', '16', '18'],
                          ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20']):
            print(ii, jj)
            re = pkl.load(open(data_path + 'ms_%s_%s_sto_iht.pkl' % (ii, jj)))
            for key in re:
                results[key] = re[key]
        pkl.dump(results, open(data_path + 'ms_%s.pkl' % method, 'wb'))
    if method == 'hsg_ht':
        for ii, jj in zip(['00', '05', '10', '15', '17', '19'],
                          ['05', '10', '15', '17', '19', '20']):
            print(ii, jj)
            re = pkl.load(open(data_path + 'ms_%s_%s_hsg_ht.pkl' % (ii, jj)))
            for key in re:
                results[key] = re[key]
        pkl.dump(results, open(data_path + 'ms_%s.pkl' % method, 'wb'))


def main(run_option):
    if run_option == 'run_ms':
        run_ms(method_name=sys.argv[2], trial_id_low=int(sys.argv[3]),
               trial_id_high=int(sys.argv[4]), num_cpus=int(sys.argv[5]))
    elif run_option == 'run_test':
        run_testing(method_name=sys.argv[2], num_cpus=int(sys.argv[3]))
    elif run_option == 'run_para_s':
        run_para_sparsity(num_cpus=int(sys.argv[2]))
    elif run_option == 'run_para_b':
        run_para_blocksize(num_cpus=int(sys.argv[2]))
    elif run_option == 'show_01':
        show_result_01()
    elif run_option == 'show_02':
        show_result_02()
    elif run_option == 'merge_ms':
        merge_ms(method=sys.argv[2])


if __name__ == '__main__':
    main(run_option=sys.argv[1])
