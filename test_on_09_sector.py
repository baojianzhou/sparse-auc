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


def get_sub_data_by_indices(data, tr_index, sub_tr_ind):
    sub_x_values = []
    sub_x_indices = []
    sub_x_len_list = []
    sub_x_positions = []
    prev_posi = 0
    for index in tr_index[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_posis'][index]
        sub_x_values.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_indices.extend(data['x_tr_indices'][cur_posi:cur_posi + cur_len])
        sub_x_len_list.append(cur_len)
        sub_x_positions.append(prev_posi)
        prev_posi += cur_len
    return sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list


def pred(data, tr_index, sub_te_ind, wt, wt_bar):
    _ = get_sub_data_by_indices(data, tr_index, sub_te_ind)
    sub_x_te_values, sub_x_te_indices, sub_x_te_positions, sub_x_te_len_list = _
    y_pred_wt = np.zeros_like(sub_te_ind, dtype=float)
    y_pred_wt_bar = np.zeros_like(sub_te_ind, dtype=float)
    for i in range(len(sub_te_ind)):
        cur_posi = sub_x_te_positions[i]
        cur_len = sub_x_te_len_list[i]
        cur_x = sub_x_te_values[cur_posi:cur_posi + cur_len]
        cur_ind = sub_x_te_indices[cur_posi:cur_posi + cur_len]
        y_pred_wt[i] = np.sum([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
        y_pred_wt_bar[i] = np.sum([cur_x[_] * wt_bar[cur_ind[_]] for _ in range(cur_len)])
    return y_pred_wt, y_pred_wt_bar


def cv_spam_l1(run_id, fold_id, num_passes, data):
    s_time = time.time()
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    k_fold = data['num_k_fold']
    auc_wt, auc_wt_bar = dict(), dict()
    for para_c, para_l1 in product(list_c, list_l1):
        algo_para = (run_id, fold_id, num_passes, para_c, para_l1, k_fold)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        # cross validate based on tr_index
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            reg_opt, step_len, is_sparse, verbose, para_beta = 0, 10000000, 0, 0, 0.0
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_spam_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], len(sub_tr_ind), para_c, para_l1, para_beta, reg_opt,
                num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('para_c: %.4f para-l1: %.4f AUC-wt: %.4f AUC-wt-bar: %.4f run_time: %.2f' %
              (para_c, para_l1, float(np.mean(list_auc_wt)),
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
        # print(para_c, para_beta, para_l1, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' spam-l1l2 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def cv_spam_l2(run_id, fold_id, num_passes, data):
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, auc_wt_bar = dict(), dict()
    s_time = time.time()
    for para_c, para_beta in product(list_c, list_beta):
        algo_para = (run_id, fold_id, num_passes, para_c, para_beta)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(data['num_k_fold'])
        list_auc_wt_bar = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt_bar = np.zeros(data['num_k_fold'])
        reg_opt, step_len, is_sparse, verbose = 1, 10000000, 0, 0
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_spam_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_c, 0.0, para_beta, reg_opt, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('para_c: %.4f para-beta: %.4f AUC-wt: %.4f AUC-wt-bar: %.4f run_time: %.2f' %
              (para_c, para_beta, float(np.mean(list_auc_wt)),
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
    print('-' * 40 + ' spam-l2 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def cv_spam_l1l2(run_id, fold_id, num_passes, data):
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for para_c, para_beta, para_l1 in product(list_c, list_beta, list_l1):
        algo_para = (run_id, fold_id, num_passes, para_c, para_beta, para_l1)
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(run_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        # cross validate based on tr_index
        list_auc_wt = np.zeros(data['num_k_fold'])
        list_auc_wt_bar = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt = np.zeros(data['num_k_fold'])
        list_num_nonzeros_wt_bar = np.zeros(data['num_k_fold'])
        reg_opt, step_len, is_sparse, verbose = 0, 10000000, 0, 0
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_spam_sparse(np.asarray(sub_x_values, dtype=float),
                                    np.asarray(sub_x_indices, dtype=np.int32),
                                    np.asarray(sub_x_positions, dtype=np.int32),
                                    np.asarray(sub_x_len_list, dtype=np.int32),
                                    np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                                    data['p'], len(sub_tr_ind), para_c, para_l1, para_beta,
                                    reg_opt, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        print('para_c: %.4f para-beta: %.4f para-l1: %.4f AUC-wt: %.4f '
              'AUC-wt-bar: %.4f run_time: %.2f' %
              (para_c, para_beta, para_l1, float(np.mean(list_auc_wt)),
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
        # print(para_c, para_beta, para_l1, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
    run_time = time.time() - s_time
    print('-' * 40 + ' spam-l1l2 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


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
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens = _
            re = c_algo_solam_sparse(
                np.asarray(sub_x_vals, dtype=float),
                np.asarray(sub_x_inds, dtype=np.int32),
                np.asarray(sub_x_posis, dtype=np.int32),
                np.asarray(sub_x_lens, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_xi, para_r, num_passes, 0)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
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
    list_eta = 2. ** np.arange(-12, -4, 1, dtype=float)
    list_lambda = 2. ** np.arange(-10, -2, 1, dtype=float)
    auc_wt, auc_wt_bar = dict(), dict()
    s_time = time.time()
    for para_eta, para_lambda in product(list_eta, list_lambda):
        if fold_id == 1:
            break
        # only run sub-tasks for parallel
        algo_para = (run_id, fold_id, num_passes, para_eta, para_lambda)
        tr_index = data['task_%d_fold_%d' % (run_id, fold_id)]['tr_index']
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
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_opauc_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_eta, para_lambda, num_passes, 0)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
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
    # to save some model selection time.
    for fold_id in range(data['num_k_fold']):
        if (run_id, fold_id) not in auc_wt:
            auc_wt[(run_id, fold_id)] = {
                'auc': auc_wt[(run_id, 0)]['auc'],
                'para': auc_wt[(run_id, 0)]['para'],
                'num_nonzeros': auc_wt[(run_id, 0)]['num_nonzeros']}
            auc_wt_bar[(run_id, fold_id)] = {
                'auc': auc_wt_bar[(run_id, 0)]['auc'],
                'para': auc_wt_bar[(run_id, 0)]['para'],
                'num_nonzeros': auc_wt_bar[(run_id, 0)]['num_nonzeros']}
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
        algo_para = (run_id, fold_id, num_passes, para_r, para_g, data['num_k_fold'])
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
        step_len, verbose = 10000000, 0
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_fsauc_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], len(sub_tr_ind), num_passes, para_r, para_g, step_len, 0)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
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
    sparsity = 20000
    list_c = list(10. ** np.arange(-5, 3, 1, dtype=float))
    list_c.extend([150., 200., 300.])
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for para_c in list_c:
        algo_para = (run_id, fold_id, num_passes, para_c, sparsity)
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
            b, para_beta = 108, 0.0
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
            re = c_algo_sht_am_sparse(
                np.asarray(sub_x_values, dtype=float),
                np.asarray(sub_x_indices, dtype=np.int32),
                np.asarray(sub_x_positions, dtype=np.int32),
                np.asarray(sub_x_len_list, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], sparsity, b, para_c, para_beta, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
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


def cv_graph_am(task_id, k_fold, num_passes, data):
    sparsity = 1 * len(data['subgraph'])
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for fold_id, para_c in product(range(k_fold), list_c):
        # only run sub-tasks for parallel
        algo_para = (task_id, fold_id, num_passes, para_c, sparsity, k_fold)
        tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (task_id, fold_id) not in auc_wt:
            auc_wt[(task_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(task_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        step_len, is_sparse, verbose = 10000, 0, 0
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            b, para_beta = len(sub_x_tr), 0.0
            re = c_algo_graph_am(sub_x_tr, sub_y_tr, sparsity, b, para_c, para_beta, num_passes,
                                 step_len, is_sparse, verbose,
                                 np.asarray(data['edges'], dtype=np.int32),
                                 np.asarray(data['weights'], dtype=float),
                                 np.asarray(data['subgraph'], dtype=np.int32))
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
            # print(np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
        if auc_wt[(task_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(task_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(task_id, fold_id)]['para'] = algo_para
            auc_wt[(task_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(task_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(task_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(task_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(task_id, fold_id)]['num_nonzeros'] = float(
                np.mean(list_num_nonzeros_wt_bar))
    run_time = time.time() - s_time

    print('-' * 40 + ' graph-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_spam_l1(task_id, fold_id, para_c, para_l1, num_passes, data):
    tr_index = data['run_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (task_id, fold_id)]['te_index']
    reg_opt, step_len, verbose = 0, 1000000, 0
    _ = get_sub_data_by_indices(data, tr_index, range(len(tr_index)))
    sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
    re = c_algo_spam_sparse(
        np.asarray(sub_x_values, dtype=float),
        np.asarray(sub_x_indices, dtype=np.int32),
        np.asarray(sub_x_positions, dtype=np.int32),
        np.asarray(sub_x_len_list, dtype=np.int32),
        np.asarray(data['y_tr'][tr_index], dtype=float),
        data['p'], len(tr_index), para_c, para_l1, 0.0, reg_opt, num_passes, step_len, verbose)
    wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
    y_pred_wt, y_pred_wt_bar = pred(data, te_index, range(len(te_index)), wt, wt_bar)
    return {'algo_para': [task_id, fold_id, para_c, para_l1],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt_bar),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_spam_l2(task_id, fold_id, para_c, para_beta, num_passes, data):
    tr_index = data['run_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (task_id, fold_id)]['te_index']
    reg_opt, step_len, verbose = 1, 1000000, 0
    _ = get_sub_data_by_indices(data, tr_index, range(len(tr_index)))
    sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
    re = c_algo_spam_sparse(
        np.asarray(sub_x_values, dtype=float),
        np.asarray(sub_x_indices, dtype=np.int32),
        np.asarray(sub_x_positions, dtype=np.int32),
        np.asarray(sub_x_len_list, dtype=np.int32),
        np.asarray(data['y_tr'][tr_index], dtype=float),
        data['p'], len(tr_index), para_c, 0.0, para_beta, reg_opt, num_passes, step_len, verbose)
    wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
    y_pred_wt, y_pred_wt_bar = pred(data, te_index, range(len(te_index)), wt, wt_bar)
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt_bar),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_spam_l1l2(task_id, fold_id, para_c, para_beta, para_l1, num_passes, data):
    tr_index = data['run_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (task_id, fold_id)]['te_index']
    reg_opt, step_len, verbose = 1, 1000000, 0
    _ = get_sub_data_by_indices(data, tr_index, range(len(tr_index)))
    sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
    re = c_algo_spam_sparse(
        np.asarray(sub_x_values, dtype=float),
        np.asarray(sub_x_indices, dtype=np.int32),
        np.asarray(sub_x_positions, dtype=np.int32),
        np.asarray(sub_x_len_list, dtype=np.int32),
        np.asarray(data['y_tr'][tr_index], dtype=float), data['p'],
        len(tr_index), para_c, para_l1, para_beta, reg_opt, num_passes, step_len, verbose)
    wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
    y_pred_wt, y_pred_wt_bar = pred(data, te_index, range(len(te_index)), wt, wt_bar)
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt_bar),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_solam(task_id, fold_id, para_xi, para_r, num_passes, data):
    tr_index = data['run_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (task_id, fold_id)]['te_index']
    reg_opt, step_len, verbose = 1, 1000000, 0
    _ = get_sub_data_by_indices(data, tr_index, range(len(tr_index)))
    sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
    re = c_algo_solam_sparse(
        np.asarray(sub_x_values, dtype=float),
        np.asarray(sub_x_indices, dtype=np.int32),
        np.asarray(sub_x_positions, dtype=np.int32),
        np.asarray(sub_x_len_list, dtype=np.int32),
        np.asarray(data['y_tr'][tr_index], dtype=float), data['p'],
        para_xi, para_r, num_passes, verbose)
    wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
    y_pred_wt, y_pred_wt_bar = pred(data, te_index, range(len(te_index)), wt, wt_bar)
    return {'algo_para': [task_id, fold_id, para_xi, para_r],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt_bar),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_sht_am(run_id, fold_id, para_c, sparsity, num_passes, data):
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
    b, para_beta, step_len, verbose = 135, 0.0, 1000000, 0
    _ = get_sub_data_by_indices(data, tr_index, range(len(tr_index)))
    sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list = _
    re = c_algo_sht_am_sparse(
        np.asarray(sub_x_values, dtype=float),
        np.asarray(sub_x_indices, dtype=np.int32),
        np.asarray(sub_x_positions, dtype=np.int32),
        np.asarray(sub_x_len_list, dtype=np.int32),
        np.asarray(data['y_tr'][tr_index], dtype=float), data['p'], len(tr_index), b,
        para_c, para_beta, sparsity, num_passes, step_len, verbose)
    wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
    y_pred_wt, y_pred_wt_bar = pred(data, te_index, range(len(te_index)), wt, wt_bar)
    return {'algo_para': [run_id, fold_id, para_c, sparsity],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_pred_wt_bar),
            't_auc': 0.0,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def get_run_fold_index_by_task_id(method, task_start, task_end):
    if method == 'solam':
        para_space = []
        for run_id in range(5):
            for fold_id in range(5):
                for para_xi in np.arange(1, 101, 9, dtype=float):
                    for para_r in 10. ** np.arange(-1, 6, 1, dtype=float):
                        para_space.append((run_id, fold_id, para_xi, para_r))
        return para_space[task_start:task_end]
    if method == 'stoht_am':
        para_space = []
        for run_id in range(5):
            for fold_id in range(5):
                for s in range(2000, 20001, 2000):
                    for para_xi in np.arange(1, 50, 9, dtype=float):
                        for para_r in 10. ** np.arange(-1, 4, 1, dtype=float):
                            para_space.append((run_id, fold_id, s, para_xi, para_r))
        return para_space[task_start:task_end]


def sparse_dot(x_indices, x_values, wt):
    y_score = np.zeros(len(x_values))
    for i in range(len(x_values)):
        for j in range(1, x_indices[i][0] + 1):
            y_score[i] += wt[x_indices[i][j]] * x_values[i][j]
    return y_score


def test_single_model_select_solam(run_id, fold_id, para_xi, para_r):
    s_time = time.time()
    data = load_dataset_normalized()
    para_spaces = {'global_pass': 5,
                   'global_runs': 5,
                   'global_cv': 5,
                   'data_id': 5,
                   'data_name': '09_sector',
                   'data_dim': data['p'],
                   'data_num': data['n'],
                   'verbose': 0}
    x_tr_indices = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=np.int32)
    x_tr_values = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=float)
    for i in range(data['n']):
        indices = [_[0] for _ in data['x_tr'][i]]
        values = np.asarray([_[1] for _ in data['x_tr'][i]], dtype=float)
        x_tr_indices[i][0] = len(indices)  # the first entry is to save len of nonzeros.
        x_tr_indices[i][1:len(indices) + 1] = indices
        x_tr_values[i][0] = len(values)  # the first entry is to save len of nonzeros.
        x_tr_values[i][1:len(indices) + 1] = values
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']

    # cross validate based on tr_index
    list_auc = np.zeros(para_spaces['global_cv'])
    kf = KFold(n_splits=para_spaces['global_cv'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        sub_x_tr_indices = x_tr_indices[tr_index[sub_tr_ind]]
        sub_x_tr_values = x_tr_values[tr_index[sub_tr_ind]]
        sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]

        sub_x_te_indices = x_tr_indices[tr_index[sub_te_ind]]
        sub_x_te_values = x_tr_values[tr_index[sub_te_ind]]
        sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
        re = c_algo_solam_sparse(np.asarray(sub_x_tr_indices, dtype=np.int32),
                                 np.asarray(sub_x_tr_values, dtype=float),
                                 np.asarray(sub_y_tr, dtype=float),
                                 np.asarray(range(len(sub_x_tr_indices)), dtype=np.int32),
                                 int(data['p']), float(para_r), float(para_xi),
                                 int(para_spaces['global_pass']),
                                 int(para_spaces['verbose']))
        wt = np.asarray(re[0])
        y_score = sparse_dot(sub_x_te_indices, sub_x_te_values, wt)
        list_auc[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_score)
    print('run_id, fold_id, para_xi, para_r: ', run_id, fold_id, para_xi, para_r)
    print('list_auc:', list_auc)
    run_time = time.time() - s_time
    return {'algo_para': [run_id, fold_id, para_xi, para_r],
            'para_spaces': para_spaces,
            'list_auc': list_auc, 'run_time': run_time}


def test_single_model_select_stoht_am(run_id, fold_id, s, para_xi, para_r):
    s_time = time.time()
    data = load_dataset_normalized()
    para_spaces = {'global_pass': 5,
                   'global_runs': 5,
                   'global_cv': 5,
                   'data_id': 5,
                   'data_name': '09_sector',
                   'data_dim': data['p'],
                   'data_num': data['n'],
                   'verbose': 0}
    x_indices = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=np.int32)
    x_values = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=float)
    for i in range(data['n']):
        indices = [_[0] for _ in data['x_tr'][i]]
        values = np.asarray([_[1] for _ in data['x_tr'][i]], dtype=float)
        values /= np.linalg.norm(values)
        x_indices[i][0] = len(indices)  # the first entry is to save len of nonzeros.
        x_indices[i][1:len(indices) + 1] = indices
        x_values[i][0] = len(values)  # the first entry is to save len of nonzeros.
        x_values[i][1:len(indices) + 1] = values
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']

    # cross validate based on tr_index
    list_auc = np.zeros(para_spaces['global_cv'])
    kf = KFold(n_splits=para_spaces['global_cv'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        sub_x_tr_indices = x_indices[tr_index[sub_tr_ind]]
        sub_x_tr_values = x_values[tr_index[sub_tr_ind]]
        sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]

        sub_x_te_indices = x_indices[tr_index[sub_te_ind]]
        sub_x_te_values = x_values[tr_index[sub_te_ind]]
        sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
        re = c_algo_stoht_am_sparse(np.asarray(sub_x_tr_indices, dtype=np.int32),
                                    np.asarray(sub_x_tr_values, dtype=float),
                                    np.asarray(sub_y_tr, dtype=float),
                                    np.asarray(range(len(sub_x_tr_indices)), dtype=np.int32),
                                    int(data['p']), float(para_r), float(para_xi), int(s),
                                    int(para_spaces['global_pass']),
                                    int(para_spaces['verbose']))
        wt = np.asarray(re[0])
        y_score = sparse_dot(sub_x_te_indices, sub_x_te_values, wt)
        list_auc[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_score)
        print(list_auc[ind])
    print('run_id, fold_id, s, para_xi, para_r: ', run_id, fold_id, s, para_xi, para_r)
    print('list_auc:', list_auc)
    run_time = time.time() - s_time
    return {'algo_para': [run_id, fold_id, s, para_xi, para_r],
            'para_spaces': para_spaces, 'list_auc': list_auc, 'run_time': run_time}


def result_summary():
    all_results = []
    for task_id in range(100):
        task_start, task_end = int(task_id) * 21, int(task_id) * 21 + 21
        f_name = data_path + 'model_select_%04d_%04d.pkl' % (task_start, task_end)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)
    file_name = data_path + 'model_select_0000_2100_2.pkl'
    pkl.dump(all_results, open(file_name, 'wb'))


def run_model_selection_solam():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1
    num_sub_tasks = 21
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = get_run_fold_index_by_task_id('solam', task_start, task_end)
    list_results = []
    for task_para in list_tasks:
        (run_id, fold_id, para_xi, para_r) = task_para
        result = test_single_model_select_solam(run_id, fold_id, para_xi, para_r)
        list_results.append(result)
    file_name = data_path + 'model_select_solam_%04d_%04d_5.pkl' % (task_start, task_end)
    pkl.dump(list_results, open(file_name, 'wb'))


def run_model_selection_stoht_am():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1
    num_sub_tasks = 75
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = get_run_fold_index_by_task_id('stoht_am', task_start, task_end)
    list_results = []
    for task_para in list_tasks:
        (run_id, fold_id, s, para_xi, para_r) = task_para
        result = test_single_model_select_stoht_am(run_id, fold_id, s, para_xi, para_r)
        list_results.append(result)
    file_name = data_path + 'model_select_sht_am_%04d_%04d_5.pkl' % (task_start, task_end)
    pkl.dump(list_results, open(file_name, 'wb'))


def model_result_analysis():
    results = pkl.load(open(data_path + 'model_select_0000_2100_5.pkl', 'rb'))
    max_auc_dict = dict()
    for result in results:
        run_id, fold_id, para_xi, para_r = result['algo_para']
        mean_auc = np.mean(result['list_auc'])
        if (run_id, fold_id) not in max_auc_dict:
            max_auc_dict[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
        if mean_auc > max_auc_dict[(run_id, fold_id)][0]:
            max_auc_dict[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
    list_best_auc = []
    for key in max_auc_dict:
        print(key, max_auc_dict[key])
        list_best_auc.append(max_auc_dict[key][0])
    print('mean_auc: %.4f std_auc: %.4f' % (np.mean(list_best_auc), np.std(list_best_auc)))


def run_solam_by_selected_model():
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1
    all_results = []
    for i in range(100):
        task_start, task_end = int(i) * 21, int(i) * 21 + 21
        f_name = data_path + 'model_select_solam_%04d_%04d_5.pkl' % (task_start, task_end)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)

    # selected model
    selected_model = dict()
    for result in all_results:
        run_id, fold_id, para_xi, para_r = result['algo_para']
        mean_auc = np.mean(result['list_auc'])
        if (run_id, fold_id) not in selected_model:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
        if mean_auc > selected_model[(run_id, fold_id)][0]:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)

    # select run_id and fold_id by task_id
    selected_run_id, selected_fold_id = selected_model[(task_id / 5, task_id % 5)][1:3]
    selected_para_xi, selected_para_r = selected_model[(task_id / 5, task_id % 5)][3:5]
    print(selected_run_id, selected_fold_id, selected_para_xi, selected_para_r)
    # to test it
    data = load_dataset_normalized()
    para_spaces = {'global_pass': 5,
                   'global_runs': 5,
                   'global_cv': 5,
                   'data_id': 5,
                   'data_name': '09_sector',
                   'data_dim': data['p'],
                   'data_num': data['n'],
                   'verbose': 0}
    x_indices = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=np.int32)
    x_values = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=float)
    for i in range(data['n']):
        indices = [_[0] for _ in data['x_tr'][i]]
        values = np.asarray([_[1] for _ in data['x_tr'][i]], dtype=float)
        x_indices[i][0] = len(indices)  # the first entry is to save len of nonzeros.
        x_indices[i][1:len(indices) + 1] = indices
        x_values[i][0] = len(values)  # the first entry is to save len of nonzeros.
        x_values[i][1:len(indices) + 1] = values
    tr_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['te_index']

    re = c_algo_solam_sparse(np.asarray(x_indices[tr_index], dtype=np.int32),
                             np.asarray(x_values[tr_index], dtype=float),
                             np.asarray(data['y_tr'][tr_index], dtype=float),
                             np.asarray(range(len(tr_index)), dtype=np.int32),
                             int(data['p']), float(selected_para_r), float(selected_para_xi),
                             int(para_spaces['global_pass']),
                             int(para_spaces['verbose']))
    wt = np.asarray(re[0])
    y_score = sparse_dot(x_indices[te_index], x_values[te_index], wt)
    auc = roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_score)
    print('run_id, fold_id, para_xi, para_r: ',
          selected_run_id, selected_fold_id, selected_para_xi, selected_para_r)
    run_time = time.time() - s_time
    print('auc:', auc, 'run_time', run_time)
    re = {'algo_para': [selected_run_id, selected_fold_id, selected_para_xi, selected_para_r],
          'para_spaces': para_spaces, 'auc': auc, 'run_time': run_time}
    pkl.dump(re, open(data_path + 'result_solam_%d_%d_passes_5.pkl' %
                      (selected_run_id, selected_fold_id), 'wb'))


def get_paras():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1
    all_results = []
    for _ in range(100):
        task_start, task_end = int(_) * 21, int(_) * 21 + 21
        f_name = data_path + 'model_select_solam_%04d_%04d_5.pkl' % (task_start, task_end)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)

    # selected model
    selected_model = dict()
    for result in all_results:
        run_id, fold_id, para_xi, para_r = result['algo_para']
        mean_auc = np.mean(result['list_auc'])
        if (run_id, fold_id) not in selected_model:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
        if mean_auc > selected_model[(run_id, fold_id)][0]:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
    paras = []
    for run_id, fold_id, s in product(range(5), range(5), range(2000, 40001, 2000)):
        paras.append((run_id, fold_id, s, selected_model[(run_id, fold_id)]))
    return task_id, paras[task_id * 5:task_id * 5 + 5]


def run_sht_am_by_selected_model():
    # to test it
    data = load_dataset_normalized()
    para_spaces = {'global_pass': 5,
                   'global_runs': 5,
                   'global_cv': 5,
                   'data_id': 5,
                   'data_name': '09_sector',
                   'data_dim': data['p'],
                   'data_num': data['n'],
                   'verbose': 0}
    x_indices = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=np.int32)
    x_values = np.zeros(shape=(data['n'], data['max_nonzero'] + 1), dtype=float)
    for i in range(data['n']):
        indices = [_[0] for _ in data['x_tr'][i]]
        values = np.asarray([_[1] for _ in data['x_tr'][i]], dtype=float)
        x_indices[i][0] = len(indices)  # the first entry is to save len of nonzeros.
        x_indices[i][1:len(indices) + 1] = indices
        x_values[i][0] = len(values)  # the first entry is to save len of nonzeros.
        x_values[i][1:len(indices) + 1] = values
    task_id, list_paras = get_paras()
    results = dict()
    for selected_run_id, selected_fold_id, s, _ in list_paras:
        selected_para_xi, selected_para_r = _[3:5]
        tr_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['tr_index']
        te_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['te_index']
        s_time = time.time()
        re = c_algo_stoht_am_sparse(np.asarray(x_indices[tr_index], dtype=np.int32),
                                    np.asarray(x_values[tr_index], dtype=float),
                                    np.asarray(data['y_tr'][tr_index], dtype=float),
                                    np.asarray(range(len(tr_index)), dtype=np.int32),
                                    int(data['p']), float(selected_para_r),
                                    float(selected_para_xi), int(s),
                                    int(para_spaces['global_pass']),
                                    int(para_spaces['verbose']))
        run_time = time.time() - s_time
        wt = np.asarray(re[0])
        y_score = sparse_dot(x_indices[te_index], x_values[te_index], wt)
        auc = roc_auc_score(y_true=data['y_tr'][te_index], y_score=y_score)
        print('run_id, fold_id, para_xi, para_r: ',
              selected_run_id, selected_fold_id, selected_para_xi, selected_para_r)
        results[(selected_run_id, selected_fold_id, s)] = {
            'algo_para': (selected_para_xi, selected_para_r, s),
            'auc': auc, 'run_time': run_time, 'para_spaces': para_spaces, 's': s}
    pkl.dump(results, open(data_path + 'result_sht_am_%03d_passes_5.pkl' % task_id, 'wb'))


def final_result_analysis_solam():
    list_auc = []
    list_time = []
    for (run_id, fold_id) in product(range(5), range(5)):
        re = pkl.load(open(data_path + 'result_solam_%d_%d_passes_5.pkl' %
                           (run_id, fold_id), 'rb'))
        print(re['auc'], re['run_time'])
        list_auc.append(re['auc'])
        list_time.append(re['run_time'])
    print('mean: %.4f, std: %.4f' % (np.mean(list_auc), np.std(list_auc)))
    print('total run time in aveage: %.4f run time per-iteration: %.4f' %
          (np.mean(list_time), np.mean(list_time) / 5.))


def final_result_analysis_sht_am():
    auc_matrix = np.zeros(shape=(25, 10))
    for ind, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        re = pkl.load(open(data_path + 'result_sht_am_%d_%d_passes_5.pkl' %
                           (run_id, fold_id), 'rb'))
        auc_matrix[ind] = re['auc_list']
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14})
    xx = ["{0:.0%}".format(_ / 55197.) for _ in np.asarray(range(2000, 20001, 2000))]
    print(xx)
    plt.figure(figsize=(5, 5))
    plt.plot(range(10), np.mean(auc_matrix, axis=0), color='r', marker='D',
             label='StoIHT+AUC')
    plt.plot(range(10), [0.9601] * 10, color='b', marker='*', label='SOLAM')
    plt.xticks(range(10), xx)
    plt.ylim([0.95, 0.97])
    plt.title('Sector Dataset')
    plt.xlabel('Sparsity Level=k/d')
    plt.legend()
    plt.show()


def test():
    auc_matrix = np.zeros(shape=(25, 20))
    for _ in range(100):
        re = pkl.load(open(data_path + 'result_sht_am_%3d_passes_5.pkl' % _, 'rb'))
        for key in re:
            run_id, fold_id, s = key
            auc_matrix[run_id * 5 + fold_id][s / 2000 - 1] = re[key]['auc']
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14})
    xx = ["{0:.0%}".format(_ / 55197.) for _ in np.asarray(range(2000, 40001, 2000))]
    print(xx)
    plt.figure(figsize=(5, 10))
    plt.plot(range(20), np.mean(auc_matrix, axis=0), color='r', marker='D',
             label='StoIHT+AUC')
    plt.plot(range(20), [0.9601] * 20, color='b', marker='*', label='SOLAM')
    plt.xticks(range(20), xx)
    plt.ylim([0.95, 0.97])
    plt.title('Sector Dataset')
    plt.xlabel('Sparsity Level=k/d')
    plt.legend()
    plt.show()


def run_ms(method_name):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    run_id, fold_id, num_passes = task_id / 5, task_id / 5, 5
    data = pkl.load(open(data_path + 'processed_sector_normalized.pkl', 'rb'))
    results, key = dict(), (run_id, fold_id)
    if method_name == 'spam_l1':
        results[key] = dict()
        results[key][method_name] = cv_spam_l1(run_id, fold_id, num_passes, data)
    elif method_name == 'solam':
        results[key] = dict()
        results[key][method_name] = cv_solam(run_id, fold_id, num_passes, data)
    elif method_name == 'sht_am':
        results[key] = dict()
        results[key][method_name] = cv_sht_am(run_id, fold_id, num_passes, data)
    elif method_name == 'spam_l2':
        results[key] = dict()
        results[key][method_name] = cv_spam_l2(run_id, fold_id, num_passes, data)
    elif method_name == 'spam_l1l2':
        results[key] = dict()
        results[key][method_name] = cv_spam_l1l2(run_id, fold_id, num_passes, data)
    elif method_name == 'fsauc':
        results[key] = dict()
        results[key][method_name] = cv_fsauc(run_id, fold_id, num_passes, data)
    elif method_name == 'opauc':
        results[key] = dict()
        results[key][method_name] = cv_opauc(run_id, fold_id, num_passes, data)
    pkl.dump(results, open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method_name), 'wb'))


def run_testing():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    num_passes = 20
    run_id, fold_id = task_id / 5, task_id / 5
    data = pkl.load(open(data_path + 'processed_sector_normalized.pkl', 'rb'))
    results, key = dict(), (run_id, fold_id)
    results[key] = dict()

    if False:
        # -----------------------
        method = 'opauc'
        ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
        _, _, _, para_eta, para_lambda, _ = ms[key][method][0][(task_id, fold_id)]['para']
        re = run_opauc(task_id, fold_id, para_eta, para_lambda, data)
        results[key][method] = re
        print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
        # -----------------------
        method = 'fsauc'
        ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
        _, _, _, para_eta, para_lambda, _ = ms[key][method][0][(task_id, fold_id)]['para']
        re = run_fsauc(task_id, fold_id, num_passes, para_eta, para_lambda, data)
        results[key][method] = re
        print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])

        # -----------------------
        method = 'spam_l1l2'
        ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
        _, _, _, para_c, para_beta, para_l1 = ms[key][method][0][(task_id, fold_id)]['para']
        re = run_spam_l1l2(task_id, fold_id, para_c, para_beta, para_l1, num_passes, data)
        results[key][method] = re
        print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
        # -----------------------
        method = 'solam'
        ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
        _, _, _, para_xi, para_r = ms[item][method][0][(task_id, fold_id)]['para']
        re = run_solam(task_id, fold_id, para_xi, para_r, num_passes, data)
        results[key][method] = re
        print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
    # -----------------------
    method = 'spam_l1'
    ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
    _, _, _, para_c, para_l1, _ = ms[key][method][0][(run_id, fold_id)]['para']
    re = run_spam_l1(run_id, fold_id, para_c, para_l1, num_passes, data)
    results[key][method] = re
    print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
    # -----------------------
    method = 'spam_l2'
    ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
    _, _, _, para_c, para_beta = ms[key][method][0][(run_id, fold_id)]['para']
    re = run_spam_l2(run_id, fold_id, para_c, para_beta, num_passes, data)
    results[key][method] = re
    print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
    # -----------------------
    method = 'sht_am'
    ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
    _, _, _, para_c, sparsity = ms[key][method][0][(run_id, fold_id)]['para']
    re = run_sht_am(run_id, fold_id, para_c, sparsity, num_passes, data)
    results[key][method] = re
    print(fold_id, method, re['auc_wt'], re['auc_wt_bar'])
    f_name = 'results_task_%02d_passes_%02d.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % (task_id, num_passes)), 'wb'))


def main():
    run_ms(method_name='sht_am')


if __name__ == '__main__':
    main()
