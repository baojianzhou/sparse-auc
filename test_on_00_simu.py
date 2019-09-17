# -*- coding: utf-8 -*-
import os
import sys
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
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_sht_am
        from sparse_module import c_algo_graph_am
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        pass
except ImportError:
    print('cannot find the module: sparse_module')
    pass

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'


def get_run_fold_index_by_task_id(method, task_start, task_end, num_passes, num_runs, k_fold):
    if method == 'spam':
        para_space = []
        for run_id in range(num_runs):
            for fold_id in range(k_fold):
                for para_xi in np.arange(1, 61, 5, dtype=float):
                    for para_beta in 10. ** np.arange(-5, 1, 1, dtype=float):
                        para_space.append((run_id, fold_id, para_xi, para_beta,
                                           num_passes, num_runs, k_fold))
        return para_space[task_start:task_end]
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


def run_spam_l2(task_id, fold_id, para_c, para_beta, num_passes, data):
    """

    :param task_id:
    :param fold_id:
    :param para_c:
    :param para_beta:
    :param num_passes:
    :param data:
    :return:
    """
    tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['task_%d_fold_%d' % (task_id, fold_id)]['te_index']
    l1_reg, reg_opt, step_len, is_sparse, verbose = 0.0, 1, 10000, 0, 0
    re = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                     np.asarray(data['y_tr'][tr_index], dtype=float), para_c,
                     l1_reg, para_beta, reg_opt, num_passes, step_len, is_sparse, verbose)
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    run_time = re[3]
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            'run_time': run_time,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_spam_l2_cv(task_id, k_fold, num_passes, data):
    """
    Model selection of SPAM-l2
    :param task_id:
    :param k_fold:
    :param num_passes:
    :param data:
    :return:
    """
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, auc_wt_bar = dict(), dict()
    s_time = time.time()
    for fold_id, para_c, para_beta in product(range(k_fold), list_c, list_beta):
        algo_para = (task_id, fold_id, num_passes, para_c, para_beta, k_fold)
        tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (task_id, fold_id) not in auc_wt:
            auc_wt[(task_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(task_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            l1_reg, reg_opt, step_len, is_sparse, verbose = 0.0, 1, 10000, 0, 0
            re = c_algo_spam(sub_x_tr, sub_y_tr, para_c, l1_reg, para_beta, reg_opt, num_passes,
                             step_len, is_sparse, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        # print(para_c, para_beta, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))
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
    print('-' * 40 + ' spam-l2 ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_spam_l1l2(task_id, fold_id, para_c, para_beta, para_l1, num_passes, data):
    """

    :param task_id:
    :param fold_id:
    :param para_c:
    :param para_beta:
    :param para_l1:
    :param num_passes:
    :param data:
    :return:
    """
    tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['task_%d_fold_%d' % (task_id, fold_id)]['te_index']
    reg_opt, step_len, is_sparse, verbose = 0, 10000, 0, 0
    re = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                     np.asarray(data['y_tr'][tr_index], dtype=float),
                     para_c, para_l1, para_beta, reg_opt, num_passes,
                     step_len, is_sparse, verbose)
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    run_time = re[3]
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            'run_time': run_time,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_spam_l1l2_cv(task_id, k_fold, num_passes, data):
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for fold_id, para_c, para_beta, para_l1 in product(range(k_fold), list_c, list_beta, list_l1):
        algo_para = (task_id, fold_id, num_passes, para_c, para_beta, para_l1, k_fold)
        tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
        # cross validate based on tr_index
        if (task_id, fold_id) not in auc_wt:
            auc_wt[(task_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(task_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        # cross validate based on tr_index
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(
                kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            reg_opt, step_len, is_sparse, verbose = 0, 10000, 0, 0
            re = c_algo_spam(sub_x_tr, sub_y_tr, para_c, para_l1, para_beta, reg_opt, num_passes,
                             step_len, is_sparse, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        if auc_wt[(task_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(task_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(task_id, fold_id)]['para'] = algo_para
            auc_wt[(task_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(task_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(task_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(task_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(task_id, fold_id)]['num_nonzeros'] = float(
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


def run_sht_am(task_id, fold_id, para_c, para_beta, sparsity, num_passes, data):
    """
    :param task_id:
    :param fold_id:
    :param para_c:
    :param para_beta:
    :param sparsity:
    :param num_passes:
    :param data:
    :return:
    """
    tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['task_%d_fold_%d' % (task_id, fold_id)]['te_index']
    step_len, is_sparse, verbose, b = 10000, 0, 0, len(tr_index)
    re = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                       np.asarray(data['y_tr'][tr_index], dtype=float),
                       sparsity, b, para_c, para_beta, num_passes, step_len,
                       is_sparse, verbose, np.asarray(data['subgraph'], dtype=np.int32))
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    run_time = re[3]
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            'run_time': run_time,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_sht_am_cv(task_id, k_fold, num_passes, data):
    sparsity, b = 1 * len(data['subgraph']), 640
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for fold_id, para_c, para_beta in product(range(k_fold), list_c, list_beta):
        # only run sub-tasks for parallel
        algo_para = (task_id, fold_id, num_passes, para_c, para_beta, sparsity, k_fold)
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
            re = c_algo_sht_am(sub_x_tr, sub_y_tr, sparsity, b, para_c, para_beta, num_passes,
                               step_len, is_sparse, verbose,
                               np.asarray(data['subgraph'], dtype=np.int32))
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
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
    print('-' * 40 + ' sht-am ' + '-' * 40)
    print('run_time: %.4f' % run_time)
    print('AUC-wt: ' + ' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print('AUC-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('nonzeros-wt: ' + ' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print('nonzeros-wt-bar: ' + ' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros']
                                          for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_graph_am(task_id, fold_id, para_c, para_beta, sparsity, num_passes, data):
    """
    :param task_id:
    :param fold_id:
    :param para_c:
    :param para_beta:
    :param sparsity:
    :param num_passes:
    :param data:
    :return:
    """
    tr_index = data['task_%d_fold_%d' % (task_id, fold_id)]['tr_index']
    te_index = data['task_%d_fold_%d' % (task_id, fold_id)]['te_index']
    step_len, is_sparse, verbose, b = 10000, 0, 0, len(tr_index)
    re = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                       np.asarray(data['y_tr'][tr_index], dtype=float),
                       sparsity, b, para_c, para_beta, num_passes, step_len,
                       is_sparse, verbose, np.asarray(data['subgraph'], dtype=np.int32))
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    run_time = re[3]
    return {'algo_para': [task_id, fold_id, para_c, para_beta],
            'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                    y_score=np.dot(data['x_tr'][te_index], wt)),
            'auc_wt_bar': roc_auc_score(y_true=data['y_tr'][te_index],
                                        y_score=np.dot(data['x_tr'][te_index], wt_bar)),
            'run_time': run_time,
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def run_graph_am_cv(task_id, k_fold, num_passes, data):
    sparsity, b = 1 * len(data['subgraph']), 640
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_beta = 10. ** np.arange(-5, 3, 1, dtype=float)
    s_time = time.time()
    auc_wt, auc_wt_bar = dict(), dict()
    for fold_id, para_c, para_beta in product(range(k_fold), list_c, list_beta):
        # only run sub-tasks for parallel
        algo_para = (task_id, fold_id, num_passes, para_c, para_beta, sparsity, k_fold)
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


def run_model_selection():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    k_fold, num_passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.3]
    fig_list = ['fig_4']
    results = dict()
    for num_tr, mu, posi_ratio, fig_i in product(tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (task_id, num_tr, mu, posi_ratio), 'rb'))
        item = (num_tr, mu, posi_ratio, fig_i, num_passes)
        results[item] = dict()
        results[item]['spam_l2'] = run_spam_l2_cv(task_id, k_fold, num_passes, data[fig_i])
        results[item]['spam_l1l2'] = run_spam_l1l2_cv(task_id, k_fold, num_passes, data[fig_i])
        results[item]['sht_am'] = run_sht_am_cv(task_id, k_fold, num_passes, data[fig_i])
        results[item]['graph_am'] = run_graph_am_cv(task_id, k_fold, num_passes, data[fig_i])
        f_name = os.path.join(data_path, 'ms_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f_%s.pkl' %
                              (task_id, num_tr, mu, posi_ratio, fig_i))
        pkl.dump({task_id: results}, open(f_name, 'wb'))


def run_testing():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    k_fold, num_passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.1, 0.3, 0.5]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    results = dict()
    for num_tr, mu, posi_ratio, fig_i in product(
            tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = data_path + 'data_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (task_id, num_tr, mu, posi_ratio), 'rb'))
        f_name = os.path.join(data_path, 'ms_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f_%s.pkl' %
                              (task_id, num_tr, mu, posi_ratio, fig_i))
        models = pkl.load(open(f_name, 'rb'))[task_id]
        for fold_id in range(k_fold):
            item = (num_tr, mu, posi_ratio, fig_i, num_passes)
            print(item)
            item_ext = (task_id, fold_id, num_tr, mu, posi_ratio, fig_i, num_passes)
            results[item_ext] = dict()
            key = (task_id, fold_id)
            # -------
            print(models[item]['spam_l2'][0][key]['para'])
            _, _, _, para_c, para_beta, _ = models[item]['spam_l2'][0][key]['para']
            re = run_spam_l2(task_id, fold_id, para_c, para_beta, num_passes, data[fig_i])
            results[item_ext]['spam_l2'] = re
            # -------
            _, _, _, para_c, para_beta, para_l1, _ = models[item]['spam_l1l2'][0][key]['para']
            re = run_spam_l1l2(task_id, fold_id, para_c, para_beta, para_l1, num_passes,
                               data[fig_i])
            results[item_ext]['spam_l1l2'] = re
            # -------
            _, _, _, para_c, para_beta, s, _ = models[item]['sht_am'][0][key]['para']
            re = run_sht_am(task_id, fold_id, para_c, para_beta, s, num_passes, data[fig_i])
            results[item_ext]['sht_am'] = re
            # -------
            _, _, _, para_c, para_beta, s, _ = models[item]['sht_am'][0][key]['para']
            re = run_graph_am(task_id, fold_id, para_c, para_beta, s, num_passes, data[fig_i])
            results[item_ext]['graph_am'] = re
    f_name = 'results_task_%02d.pkl'
    pkl.dump(results, open(f_name % task_id, 'wb'))


def main():
    run_testing()


if __name__ == '__main__':
    main()
