# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import numpy as np
import pickle as pkl
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam_sparse
        from sparse_module import c_algo_solam_sparse
        from sparse_module import c_algo_sht_am_sparse
        from sparse_module import c_algo_fsauc_sparse
        from sparse_module import c_algo_opauc_sparse
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector/'


def get_data_by_ind(data, tr_index, sub_tr_ind):
    sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens = [], [], [], []
    prev_posi = 0
    for index in tr_index[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_posis'][index]
        sub_x_vals.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_inds.extend(data['x_tr_indices'][cur_posi:cur_posi + cur_len])
        sub_x_lens.append(cur_len)
        sub_x_posis.append(prev_posi)
        prev_posi += cur_len
    sub_x_vals = np.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_posis = np.asarray(sub_x_posis, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_tr


def pred_auc(data, tr_index, sub_te_ind, wt):
    _ = get_data_by_ind(data, tr_index, sub_te_ind)
    sub_x_te_values, sub_x_te_indices, sub_x_te_positions, sub_x_te_len_list, _ = _
    y_pred_wt = np.zeros_like(sub_te_ind, dtype=float)
    for i in range(len(sub_te_ind)):
        cur_posi = sub_x_te_positions[i]
        cur_len = sub_x_te_len_list[i]
        cur_x = sub_x_te_values[cur_posi:cur_posi + cur_len]
        cur_ind = sub_x_te_indices[cur_posi:cur_posi + cur_len]
        y_pred_wt[i] = np.sum([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
    sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
    auc_wt = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
    return auc_wt


def cv_spam_l1(run_id, fold_id, k_fold, num_passes, data):
    s_time, models = time.time(), {'auc': 0.0, 'para_c': 1e-5, 'para_l1': 1e-5}
    for para_c, para_l1 in product(10. ** np.arange(-5, 6, 1, dtype=float),
                                   10. ** np.arange(-5, 6, 1, dtype=float)):
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        cur_auc = 0.0
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                KFold(n_splits=k_fold, shuffle=False).split(np.zeros(shape=(len(tr_index), 1)))):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_spam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, para_l1, 0.0, 0, num_passes, 10000000, 0)
            cur_auc += pred_auc(data, tr_index, sub_te_ind, wt)
        print('run_%02d_fold_%d para_c: %.4f para-l1: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_l1, cur_auc / (k_fold * 1.), time.time() - s_time))
        if models['auc'] < cur_auc / (k_fold * 1.):
            models['auc'] = cur_auc / (k_fold * 1.)
            models['para_c'] = para_c
            models['para_l1'] = para_l1
    print('total run_time: %.4f selected para:(%.6f,%.6f) best_auc: %.4f' %
          (time.time() - s_time, models['para']['para_c'], models['para']['para_l1'],
           models['auc']))
    return models


def cv_spam_l2(run_id, fold_id, k_fold, num_passes, data):
    s_time, models = time.time(), {'auc': 0.0, 'para_c': 1e-5, 'para_l1': 1e-5}
    for para_c, para_beta in product(10. ** np.arange(-5, 6, 1, dtype=float),
                                     10. ** np.arange(-5, 6, 1, dtype=float)):
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        cur_auc = 0.0
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                KFold(n_splits=k_fold, shuffle=False).split(np.zeros(shape=(len(tr_index), 1)))):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_spam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, 0.0, para_beta, 1, num_passes, 10000000, 0)
            cur_auc += pred_auc(data, tr_index, sub_te_ind, wt)
        print('run_%02d_fold_%d para_c: %.4f para-beta: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_beta, cur_auc / (k_fold * 1.), time.time() - s_time))
        if models['auc'] < cur_auc / (k_fold * 1.):
            models['auc'] = cur_auc / (k_fold * 1.)
            models['para_c'] = para_c
            models['para_beta'] = para_beta
    print('total run_time: %.4f selected para:(%.6f,%.6f) best_auc: %.4f' %
          (time.time() - s_time, models['para']['para_c'], models['para']['para_l1'],
           models['auc']))
    return models


def cv_spam_l1l2(run_id, fold_id, k_fold, num_passes, data):
    s_time, models = time.time(), {'auc': 0.0, 'para_c': 1e-5, 'para_l1': 1e-5}
    for para_c, para_l1, para_beta in product(10. ** np.arange(-2, 3, 1, dtype=float),
                                              10. ** np.arange(-5, 2, 1, dtype=float),
                                              10. ** np.arange(-5, 2, 1, dtype=float)):
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        cur_auc = 0.0
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                KFold(n_splits=k_fold, shuffle=False).split(np.zeros(shape=(len(tr_index), 1)))):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_spam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, para_l1, para_beta, 1, num_passes, 10000000, 0)
            cur_auc += pred_auc(data, tr_index, sub_te_ind, wt)
        print('run_%02d_fold_%d para_c: %.4f para-beta: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_beta, cur_auc / (k_fold * 1.), time.time() - s_time))
        if models['auc'] < cur_auc / (k_fold * 1.):
            models['auc'] = cur_auc / (k_fold * 1.)
            models['para_c'] = para_c
            models['para_l1'] = para_l1
            models['para_beta'] = para_beta
    print('total run_time: %.4f selected para:(%.6f,%.6f) best_auc: %.4f' %
          (time.time() - s_time, models['para']['para_c'], models['para']['para_l1'],
           models['auc']))
    return models


def cv_solam(run_id, fold_id, num_passes, data):
    s_time = time.time()
    list_xi = np.arange(1, 101, 9, dtype=float)
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    auc_wt, auc_wt_bar = dict(), dict()
    for para_xi, para_r in product(list_xi, list_r):
        algo_para = (run_id, fold_id, num_passes, para_xi, para_r)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(data['num_k_fold'])
        list_auc_wt_bar = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt_bar = np.zeros(data['num_k_fold'])
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_data_by_ind(data, tr_index, sub_tr_ind)
            sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens = _
            re = c_algo_solam_sparse(
                np.asarray(sub_x_vals, dtype=float),
                np.asarray(sub_x_inds, dtype=np.int32),
                np.asarray(sub_x_posis, dtype=np.int32),
                np.asarray(sub_x_lens, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_xi, para_r, num_passes, 0)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred_auc(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('para_xi: %.4f para_r: %.4f AUC-wt: %.4f AUC-wt-bar: %.4f run_time: %.2f' %
              (para_xi, para_r, float(np.mean(list_auc_wt)),
               float(np.mean(list_auc_wt_bar)), time.time() - s_time))
        if auc_wt[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(run_id, fold_id)]['para'] = algo_para
            auc_wt[(run_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(run_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(run_id, fold_id)]['num_nonzeros'] = float(
                np.mean(list_num_nonzeros_wt_bar))

    run_time = time.time() - s_time

    print('-' * 40 + ' solam ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def cv_opauc(run_id, fold_id, num_passes, data):
    list_tau = [50]
    list_eta = 2. ** np.arange(-12, -4, 1, dtype=float)
    list_lambda = 2. ** np.arange(-10, -2, 1, dtype=float)
    auc_wt, auc_wt_bar = dict(), dict()
    s_time = time.time()
    step_len = 1000000
    for para_tau, para_eta, para_lambda in product(list_tau, list_eta, list_lambda):
        # only run sub-tasks for parallel
        algo_para = (run_id, fold_id, num_passes, para_tau, para_eta, para_lambda)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(data['num_k_fold'])
        list_auc_wt_bar = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt_bar = np.zeros(data['num_k_fold'])
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_data_by_ind(data, tr_index, sub_tr_ind)
            sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens = _
            wt, wt_bar, aucs, rts = c_algo_opauc_sparse(
                np.asarray(sub_x_vals, dtype=float), np.asarray(sub_x_inds, dtype=np.int32),
                np.asarray(sub_x_posis, dtype=np.int32), np.asarray(sub_x_lens, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_tau, para_eta, para_lambda, num_passes, step_len, 0)
            y_pred_wt, y_pred_wt_bar = pred_auc(data, tr_index, sub_te_ind, wt, wt_bar)
            list_auc_wt[ind] = roc_auc_score(data['y_tr'][tr_index[sub_te_ind]], y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(data['y_tr'][tr_index[sub_te_ind]], y_pred_wt_bar)
            print(ind, list_auc_wt[ind], list_auc_wt_bar[ind], np.linalg.norm(wt),
                  np.linalg.norm(wt_bar), time.time() - s_time)
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('fold-id:%d AUC-wt: %.4f AUC-wt-bar: %.4f run_time: %.4f' %
              (fold_id, float(np.mean(list_auc_wt)), float(np.mean(list_auc_wt_bar)),
               time.time() - s_time))
        if auc_wt[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(run_id, fold_id)]['para'] = algo_para
            auc_wt[(run_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(run_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(run_id, fold_id)]['num_nonzeros'] = float(
                np.mean(list_num_nonzeros_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' opauc ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def cv_fsauc(run_id, fold_id, num_passes, data):
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    list_g = 2. ** np.arange(-10, -2, 1, dtype=float)
    auc_wt, auc_wt_bar = dict(), dict()
    s_time = time.time()
    for para_r, para_g in product(list_r, list_g):
        # only run sub-tasks for parallel
        algo_para = (run_id, fold_id, num_passes, para_r, para_g)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        if (run_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(data['num_k_fold'])
        list_auc_wt_bar = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt_bar = np.zeros(data['num_k_fold'])
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        step_len, verbose = 10000000, 0
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_data_by_ind(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            wt, wt_bar, aucs, rts = c_algo_fsauc_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_r, para_g, num_passes, step_len, verbose)
            y_pred_wt, y_pred_wt_bar = pred_auc(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            if (not np.isnan(y_pred_wt).any()) and (not np.isinf(y_pred_wt).any()):
                list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            else:
                list_auc_wt[ind] = 0.0
            if (not np.isnan(y_pred_wt_bar).any()) and (not np.isinf(y_pred_wt_bar).any()):
                list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            else:
                list_auc_wt_bar[ind] = 0.0
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('para_r: %.4f para_g: %.4f AUC-wt: %.4f AUC-wt-bar: %.4f run_time: %.2f' %
              (para_r, para_g, float(np.mean(list_auc_wt)),
               float(np.mean(list_auc_wt_bar)), time.time() - s_time))
        if auc_wt[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(run_id, fold_id)]['para'] = algo_para
            auc_wt[(run_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(run_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(run_id, fold_id)]['num_nonzeros'] = float(
                np.mean(list_num_nonzeros_wt_bar))

    run_time = time.time() - s_time
    print('-' * 40 + ' opauc ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def cv_sht_am(run_id, fold_id, num_passes, data):
    para_b = 108
    list_c = list(10. ** np.arange(-5, 3, 1, dtype=float))
    list_sparsity = [10000, 15000, 20000, 25000, 30000, 35000, 40000]
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for para_c, para_sparsity in product(list_c, list_sparsity):
        algo_para = (run_id, fold_id, num_passes, para_c, para_sparsity)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        step_len, verbose = 10000000, 0
        list_auc_wt = np.zeros(data['num_k_fold'])
        list_auc_wt_bar = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt_bar = np.zeros(data['num_k_fold'])
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_data_by_ind(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_sht_am_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_sparsity, para_b, para_c, 0.0, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred_auc(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('para_c: %.4f AUC-wt: %.4f AUC-wt-bar: %.4f run_time: %.2f' %
              (para_c, float(np.mean(list_auc_wt)),
               float(np.mean(list_auc_wt_bar)), time.time() - s_time))
        if auc_wt[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(run_id, fold_id)]['para'] = algo_para
            auc_wt[(run_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(run_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(run_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(run_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(run_id, fold_id)]['num_nonzeros'] = float(
                np.mean(list_num_nonzeros_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_spam_l1l2(task_id, fold_id, para_c, para_beta, para_l1, num_passes, step_len, data):
    tr_index = data['run_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (task_id, fold_id)]['te_index']
    _ = get_data_by_ind(data, tr_index, range(len(tr_index)))
    sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
    re = c_algo_spam_sparse(
        np.asarray(sub_x_values, dtype=float), np.asarray(sub_x_indices, dtype=np.int32),
        np.asarray(sub_x_positions, dtype=np.int32), np.asarray(sub_x_len_list, dtype=np.int32),
        np.asarray(data['y_tr'][tr_index], dtype=float), data['p'],
        len(tr_index), para_c, para_l1, para_beta, 0, num_passes, step_len, 0)
    wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
    y_pred_wt, y_pred_wt_bar = pred_auc(data, te_index, range(len(te_index)), wt, wt_bar)
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt_bar),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_ms(method_name):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    run_id, fold_id, num_passes = task_id / 5, task_id / 5, 5
    data = pkl.load(open(data_path + 'processed_sector_normalized.pkl', 'rb'))
    results, key = dict(), (run_id, fold_id)
    if method_name == 'spam_l1':
        results[key] = dict()
        results[key][method_name] = cv_spam_l1(run_id, fold_id, num_passes, data)
    elif method_name == 'spam_l2':
        results[key] = dict()
        results[key][method_name] = cv_spam_l2(run_id, fold_id, num_passes, data)
    elif method_name == 'spam_l1l2':
        results[key] = dict()
        results[key][method_name] = cv_spam_l1l2(run_id, fold_id, num_passes, data)
    elif method_name == 'solam':
        results[key] = dict()
        results[key][method_name] = cv_solam(run_id, fold_id, num_passes, data)
    elif method_name == 'sht_am':
        results[key] = dict()
        results[key][method_name] = cv_sht_am(run_id, fold_id, num_passes, data)
    elif method_name == 'fsauc':
        results[key] = dict()
        results[key][method_name] = cv_fsauc(run_id, fold_id, num_passes, data)
    elif method_name == 'opauc':
        results[key] = dict()
        results[key][method_name] = cv_opauc(run_id, fold_id, num_passes, data)
    pkl.dump(results, open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method_name), 'wb'))


def pred_results(wt, wt_bar, auc, rts, para_list, te_index, data):
    return {'auc_wt': pred_auc(data, te_index, range(len(te_index)), wt),
            'nonzero_wt': np.count_nonzero(wt),
            'algo_para': para_list,
            'auc': auc,
            'rts': rts,
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def test():
    # -----------------------
    method = 'opauc'
    ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
    _, _, _, para_eta, para_lambda, _ = ms[key][method][0][(task_id, fold_id)]['para']
    re = run_opauc(task_id, fold_id, para_eta, para_lambda, data)
    results[key][method] = re
    print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
    # -----------------------
    method = 'spam_l1l2'
    ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
    _, _, _, para_c, para_beta, para_l1 = ms[key][method][0][(task_id, fold_id)]['para']
    re = run_spam_l1l2(task_id, fold_id, para_c, para_beta, para_l1, num_passes, data)
    results[key][method] = re
    print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])


def run_testing():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    run_id, fold_id, num_passes, step_len = task_id / 5, task_id / 5, 20, 10000000
    data = pkl.load(open(data_path + 'processed_sector_normalized.pkl', 'rb'))

    ms_f_name = data_path + 'ms_task_%02d_%s.pkl'
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
    _ = get_data_by_ind(data, tr_index, range(len(tr_index)))
    x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr = _
    results, key = dict(), (run_id, fold_id)
    results[key] = dict()
    # -----------------------
    method = 'spam_l1'
    ms = pkl.load(open(ms_f_name % (task_id, method), 'rb'))
    _, _, _, para_c, para_l1, _ = ms[key][method][0][(run_id, fold_id)]['para']
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
        data['p'], para_c, para_l1, 0.0, 0, num_passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'spam_l2'
    ms = pkl.load(open(ms_f_name % (task_id, method), 'rb'))
    _, _, _, para_c, para_beta = ms[key][method][0][(run_id, fold_id)]['para']
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
        data['p'], para_c, 0.0, para_beta, 1, num_passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'sht_am'
    ms = pkl.load(open(ms_f_name % (task_id, method), 'rb'))
    _, _, _, para_c, sparsity = ms[key][method][0][(run_id, fold_id)]['para']
    b = 135
    wt, wt_bar, auc, rts = c_algo_sht_am_sparse(
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
        data['p'], sparsity, b, para_c, 0.0, num_passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'fsauc'
    ms = pkl.load(open(ms_f_name % (task_id, method), 'rb'))
    _, _, _, para_r, para_g, _ = ms[key][method][0][(run_id, fold_id)]['para']
    wt, wt_bar, auc, rts = c_algo_fsauc_sparse(
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
        data['p'], para_r, para_g, num_passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'solam'
    ms = pkl.load(open(ms_f_name % (task_id, method), 'rb'))
    _, _, _, para_xi, para_r = ms[key][method][0][(task_id, fold_id)]['para']
    wt, wt_bar, auc, rts = c_algo_solam_sparse(
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
        data['p'], para_xi, para_r, num_passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    f_name = 'results_task_%02d_passes_%02d.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % (task_id, num_passes)), 'wb'))


def main():
    run_testing()


if __name__ == '__main__':
    main()
