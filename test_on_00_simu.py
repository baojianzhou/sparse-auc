# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import random
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
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

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


def run_cv_spam_l2(task_id, k_fold, num_passes, data):
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
    print('spam_l2 run time: %.4f' % (time.time() - s_time))
    print('-------- AUC --------')
    print(' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print(' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('-------- nonzeros --------')
    print(' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print(' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_cv_spam_l1l2(task_id, k_fold, num_passes, data):
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
    print('spam_l1l2 run_time: %.4f' % run_time)
    print('-------- AUC --------')
    print(' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print(' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('-------- nonzeros --------')
    print(' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print(' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_ms_sht_am(task_id, k_fold, num_passes, data):
    sparsity, b = 1 * len(data['subgraph']), 128
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
    print('sht_am run_time: %.4f' % run_time)
    print('-------- AUC --------')
    print(' '.join(['%.4f' % auc_wt[_]['auc'] for _ in auc_wt]))
    print(' '.join(['%.4f' % auc_wt_bar[_]['auc'] for _ in auc_wt_bar]))
    print('-------- nonzeros --------')
    print(' '.join(['%.4f' % auc_wt[_]['num_nonzeros'] for _ in auc_wt]))
    print(' '.join(['%.4f' % auc_wt_bar[_]['num_nonzeros'] for _ in auc_wt_bar]))
    return auc_wt, auc_wt_bar


def run_spam_l1l2_by_sm(model, num_passes):
    """
    25 tasks to finish
    :return:
    """
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1

    num_runs, k_fold = 5, 5
    all_results = pkl.load(open(data_path + 'ms_task_%02d.pkl' % task_id, 'rb'))
    # selected model
    selected_model = dict()
    for result in all_results['spam_elastic_net'][num_passes]:
        run_id, fold_id, para_xi, para_beta, para_l1, num_passes, num_runs, k_fold = result[
            'algo_para']
        mean_auc = np.mean(result['list_auc_%s' % model])
        if (run_id, fold_id) not in selected_model:
            selected_model[(run_id, fold_id)] = (
                mean_auc, run_id, fold_id, para_xi, para_beta, para_l1)
        if mean_auc > selected_model[(run_id, fold_id)][0]:
            selected_model[(run_id, fold_id)] = (
                mean_auc, run_id, fold_id, para_xi, para_beta, para_l1)

    # select run_id and fold_id by task_id
    selected_run_id, selected_fold_id = selected_model[(task_id / 5, task_id % 5)][1:3]
    selected_para_xi, selected_para_beta, selected_para_l1 = selected_model[
                                                                 (task_id / 5, task_id % 5)][3:6]
    print(selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    # to test it
    data = load_data(width=33, height=33, num_tr=1000, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'], task_id=task_id)
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_beta': selected_para_beta,
                   'para_l1_reg': selected_para_l1,  # no l1 regularization needed.
                   'para_xi': selected_para_xi,
                   'para_fold_id': selected_fold_id,
                   'para_run_id': selected_run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 0,
                   'para_step_len': 2000,
                   'para_reg_opt': 1,
                   'data_id': 00,
                   'data_name': '00_simu'}
    tr_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['te_index']
    re = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                     np.asarray(data['y_tr'][tr_index], dtype=float),
                     para_spaces['para_xi'],
                     para_spaces['para_l1_reg'],
                     para_spaces['para_beta'],
                     para_spaces['para_reg_opt'],
                     para_spaces['para_num_passes'],
                     para_spaces['para_step_len'],
                     para_spaces['para_is_sparse'],
                     para_spaces['para_verbose'])
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    auc_wt = roc_auc_score(y_true=data['y_tr'][te_index],
                           y_score=np.dot(data['x_tr'][te_index], wt))
    auc_wt_bar = roc_auc_score(y_true=data['y_tr'][te_index],
                               y_score=np.dot(data['x_tr'][te_index], wt_bar))
    print('run_id, fold_id, para_xi, para_r: ',
          selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    run_time = time.time() - s_time
    print('auc_wt:', auc_wt, 'auc_wt_bar:', auc_wt_bar, 'run_time', run_time)
    re = {'algo_para': [selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta],
          'para_spaces': para_spaces, 'auc_wt': auc_wt, 'auc_wt_bar': auc_wt_bar,
          'run_time': run_time}
    return re


def run_spam_l2_by_sm(model, num_passes):
    """
    25 tasks to finish
    :return:
    """
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1

    num_runs, k_fold = 5, 5
    all_results = pkl.load(open(data_path + 'ms_task_%02d.pkl' % task_id, 'rb'))
    # selected model
    selected_model = dict()
    for result in all_results['spam_l2'][num_passes]:
        run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold = result['algo_para']
        mean_auc = np.mean(result['list_auc_%s' % model])
        if (run_id, fold_id) not in selected_model:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_beta)
        if mean_auc > selected_model[(run_id, fold_id)][0]:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_beta)

    # select run_id and fold_id by task_id
    selected_run_id, selected_fold_id = selected_model[(task_id / 5, task_id % 5)][1:3]
    selected_para_xi, selected_para_beta = selected_model[(task_id / 5, task_id % 5)][3:5]
    print(selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    # to test it
    data = load_data(width=33, height=33, num_tr=1000, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'], task_id=task_id)
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_beta': selected_para_beta,
                   'para_l1_reg': 0.0,  # no l1 regularization needed.
                   'para_xi': selected_para_xi,
                   'para_fold_id': selected_fold_id,
                   'para_run_id': selected_run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 0,
                   'para_step_len': 2000,
                   'para_reg_opt': 1,
                   'data_id': 00,
                   'data_name': '00_simu'}
    tr_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['te_index']
    re = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                     np.asarray(data['y_tr'][tr_index], dtype=float),
                     para_spaces['para_xi'],
                     para_spaces['para_l1_reg'],
                     para_spaces['para_beta'],
                     para_spaces['para_reg_opt'],
                     para_spaces['para_num_passes'],
                     para_spaces['para_step_len'],
                     para_spaces['para_is_sparse'],
                     para_spaces['para_verbose'])
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    auc_wt = roc_auc_score(y_true=data['y_tr'][te_index],
                           y_score=np.dot(data['x_tr'][te_index], wt))
    auc_wt_bar = roc_auc_score(y_true=data['y_tr'][te_index],
                               y_score=np.dot(data['x_tr'][te_index], wt_bar))
    print('run_id, fold_id, para_xi, para_r: ',
          selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    run_time = time.time() - s_time
    print('auc_wt:', auc_wt, 'auc_wt_bar:', auc_wt_bar, 'run_time', run_time)
    re = {'algo_para': [selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta],
          'para_spaces': para_spaces, 'auc_wt': auc_wt, 'auc_wt_bar': auc_wt_bar,
          'run_time': run_time}
    return re


def run_sht_am_by_sm(model, num_passes, global_sparsity):
    """
    25 tasks to finish
    :return:
    """
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1
    num_runs, k_fold = 5, 5
    all_results = pkl.load(open(data_path + 'ms_task_%02d.pkl' % task_id, 'rb'))
    # selected model
    selected_model = dict()
    for result in all_results['sht_am'][num_passes][global_sparsity]:
        run_id, fold_id, para_sparsity, para_xi, para_beta, num_passes, num_runs, k_fold = result[
            'algo_para']
        mean_auc = np.mean(result['list_auc_%s' % model])
        if (run_id, fold_id) not in selected_model:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_beta)
        if mean_auc > selected_model[(run_id, fold_id)][0]:
            selected_model[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_beta)

    # select run_id and fold_id by task_id
    selected_run_id, selected_fold_id = selected_model[(task_id / 5, task_id % 5)][1:3]
    selected_para_xi, selected_para_beta = selected_model[(task_id / 5, task_id % 5)][3:5]
    print(selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    # to test it
    data = load_data(width=33, height=33, num_tr=1000, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'], task_id=task_id)
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_sparsity': global_sparsity,
                   'para_beta': selected_para_beta,
                   'para_l1_reg': 0.0,  # no l1 regularization needed.
                   'para_xi': selected_para_xi,
                   'para_fold_id': selected_fold_id,
                   'para_run_id': selected_run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 0,
                   'para_step_len': 2000,
                   'para_reg_opt': 1,
                   'data_id': 00,
                   'data_name': '00_simu'}
    tr_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['te_index']
    re = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                       np.asarray(data['y_tr'][tr_index], dtype=float),
                       para_spaces['para_sparsity'],
                       para_spaces['para_xi'],
                       para_spaces['para_beta'],
                       para_spaces['para_num_passes'],
                       para_spaces['para_step_len'],
                       para_spaces['para_is_sparse'],
                       para_spaces['para_verbose'])
    wt = np.asarray(re[0])
    wt_bar = np.asarray(re[1])
    auc_wt = roc_auc_score(y_true=data['y_tr'][te_index],
                           y_score=np.dot(data['x_tr'][te_index], wt))
    auc_wt_bar = roc_auc_score(y_true=data['y_tr'][te_index],
                               y_score=np.dot(data['x_tr'][te_index], wt_bar))
    print('run_id, fold_id, para_xi, para_r: ',
          selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    run_time = time.time() - s_time
    print('auc_wt:', auc_wt, 'auc_wt_bar:', auc_wt_bar, 'run_time', run_time)
    re = {'algo_para': [selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta],
          'para_spaces': para_spaces, 'auc_wt': auc_wt, 'auc_wt_bar': auc_wt_bar,
          'run_time': run_time}
    return re


def run_model_selection():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    k_fold = 5
    results = dict()
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.1]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    for num_tr, mu, posi_ratio in product(tr_list, mu_list, posi_ratio_list):
        f_name = data_path + 'data_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
        data = pkl.load(open(f_name % (task_id, num_tr, mu, posi_ratio), 'rb'))
        for fig_i in fig_list:
            if fig_i != 'fig_2':
                continue
            item = (num_tr, mu, posi_ratio, fig_i)
            results[item] = dict()
            results[item]['spam_l2'] = dict()
            results[item]['spam_l1l2'] = dict()
            results[item]['spam_sht_am'] = dict()
            for num_passes in [20, 10, 15, 20]:
                s_time = time.time()
                re = run_cv_spam_l2(task_id, k_fold, num_passes, data[fig_i])
                results[item]['spam_l2'][num_passes] = re
                if False:
                    re = run_cv_spam_l1l2(task_id, k_fold, num_passes, data[fig_i])
                    results[item]['spam_l1l2'][num_passes] = re
                re = run_ms_sht_am(task_id, k_fold, num_passes, data[fig_i])
                results[item]['sht_am'] = re
            print(time.time() - s_time)
    f_name = os.path.join(data_path, 'ms_task_%02d.pkl' % task_id)
    pkl.dump({task_id: results}, open(f_name, 'wb'))


def run_testing():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    results_sht_am = dict()
    results_spam_l2 = dict()
    results_spam_l1l2 = dict()
    for num_passes in [1, 5, 10, 15, 20]:
        results_spam_l2[num_passes] = run_spam_l2_by_sm(model='wt', num_passes=num_passes)
        results_spam_l1l2[num_passes] = run_spam_l1l2_by_sm(model='wt', num_passes=num_passes)
        results_sht_am[num_passes] = dict()
        for sparsity in [26]:
            re = run_sht_am_by_sm(model='wt', num_passes=num_passes, global_sparsity=sparsity)
            results_sht_am[num_passes][sparsity] = re
    file_name = 're_task_%02d.pkl' % task_id
    pkl.dump({'spam_l2': results_spam_l2,
              'sht_am': results_sht_am,
              'spam_elastic_net': results_spam_l1l2},
             open(os.path.join(data_path, file_name), 'wb'))


def main():
    run_model_selection()


if __name__ == '__main__':
    main()
