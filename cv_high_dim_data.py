# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pickle as pkl
import multiprocessing
from os.path import join
from os.path import exists
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'


def get_data_by_ind(data, tr_ind, sub_tr_ind):
    sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens = [], [], [], []
    prev_posi = 0
    for index in tr_ind[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_poss'][index]
        sub_x_vals.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_inds.extend(data['x_tr_inds'][cur_posi:cur_posi + cur_len])
        sub_x_lens.append(cur_len)
        sub_x_poss.append(prev_posi)
        prev_posi += cur_len
    sub_x_vals = np.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_poss = np.asarray(sub_x_poss, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data['y_tr'][tr_ind[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr


def pred_auc(data, tr_index, sub_te_ind, wt):
    if np.isnan(wt).any() or np.isinf(wt).any():  # not a valid score function.
        return 0.0
    _ = get_data_by_ind(data, tr_index, sub_te_ind)
    sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_te = _
    y_pred_wt = np.zeros_like(sub_te_ind, dtype=float)
    for i in range(len(sub_te_ind)):
        cur_posi = sub_x_posis[i]
        cur_len = sub_x_lens[i]
        cur_x = sub_x_vals[cur_posi:cur_posi + cur_len]
        cur_ind = sub_x_inds[cur_posi:cur_posi + cur_len]
        y_pred_wt[i] = np.sum([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
    return roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)


def run_single_spam_l1(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_l1, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_c': para_c, 'para_l1': para_l1,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, auc, rts = c_algo_spam_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_c, para_l1, 0.0, 0, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_c, para_l1, results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_spam_l1(data_name, method, task_id, k_fold, passes, step_len, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    list_c = np.arange(1, 101, 9, dtype=float)
    list_l1 = 10. ** np.arange(-8, 3, 1, dtype=float)
    # by adding this step, we can reduce some redundant model space.
    if exists(join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))):
        f = join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))
        re_list_c, re_list_l1 = set(), set()
        for item in pkl.load(open(f, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_c.add(item['para_c'])
                re_list_l1.add(item['para_l1'])
        list_c, list_l1 = np.sort(list(re_list_c)), np.sort(list(re_list_l1))
    print('space size: %d' % (len(list_c) * len(list_l1)))
    sys.stdout.flush()
    para_space = []
    for index, (para_c, para_l1) in enumerate(product(list_c, list_l1)):
        para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l1, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_spam_l1, para_space)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_single_spam_l2(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_l2, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_c': para_c, 'para_l2': para_l2,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    data = pkl.load(open(join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id)), 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    print(len(tr_index), len(results))
    sys.stdout.flush()
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, auc, rts = c_algo_spam_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_c, 0.0, para_l2, 1, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_c, para_l2, results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_spam_l2(data_name, method, task_id, k_fold, passes, step_len, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    list_c = np.arange(1, 101, 9, dtype=float)
    list_l2 = 10. ** np.arange(-5, 6, 1, dtype=float)
    # by adding this step, we can reduce some redundant model space.
    if exists(join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))):
        f = join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))
        re_list_c, re_list_l2 = set(), set()
        for item in pkl.load(open(f, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_c.add(item['para_c'])
                re_list_l2.add(item['para_l2'])
        list_c, list_l2 = list(re_list_c), list(re_list_l2)
    print('space size: %d' % (len(list_c) * len(list_l2)))
    para_space = []
    for index, (para_c, para_l2) in enumerate(product(list_c, list_l2)):
        para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l2, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_spam_l2, para_space)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_single_spam_l1l2(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_l1, para_l2, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_c': para_c, 'para_l1': para_l1, 'para_l2': para_l2,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    f_name = join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, auc, rts = c_algo_spam_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_c, para_l1, para_l2, 0, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_c, para_l2, results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_spam_l1l2(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    # original para space is too large, we can reduce them based on spam_l1, spam_l2
    list_c = np.arange(1, 101, 9, dtype=float)
    list_l1 = 10. ** np.arange(-5, 6, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 6, 1, dtype=float)
    # pre-select parameters that have AUC=0.8+
    if os.path.exists(join(data_path, '%s/ms_run_0_fold_0_spam_l1.pkl' % data_name)) and \
            os.path.exists(join(data_path, '%s/ms_run_0_fold_0_spam_l2.pkl' % data_name)):
        f1 = join(data_path, '%s/ms_run_0_fold_0_spam_l1.pkl' % data_name)
        f2 = join(data_path, '%s/ms_run_0_fold_0_spam_l2.pkl' % data_name)
        re_list_c, re_list_l1, re_list_l2 = set(), set(), set()
        for item in pkl.load(open(f1, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_c.add(item['para_c'])
                re_list_l1.add(item['para_l1'])
        for item in pkl.load(open(f2, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_c.add(item['para_c'])
                re_list_l2.add(item['para_l2'])
        list_c = np.sort(list(re_list_c))
        list_l1, list_l2 = np.sort(list(re_list_l1)), np.sort(list(re_list_l2))
    print('space size: %d' % (len(list_c) * len(list_l1) * len(list_l2)))
    para_space = []
    for index, (para_c, para_l1, para_l2) in enumerate(product(list_c, list_l1, list_l2)):
        para = (run_id, fold_id, k_fold, passes, step, para_c, para_l1, para_l2, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_spam_l1l2, para_space)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_single_solam(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_r, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_c': para_c, 'para_r': para_r,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, auc, rts = c_algo_solam_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_c, para_r, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_c, para_r, results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_solam(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    list_c = np.arange(1, 101, 9, dtype=float)
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    # by adding this step, we can reduce some redundant model space.
    if os.path.exists(join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))):
        f = join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))
        re_list_c, re_list_r = set(), set()
        for item in pkl.load(open(f, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_c.add(item['para_c'])
                re_list_r.add(item['para_r'])
        list_c, list_r = np.sort(list(re_list_c)), np.sort(list(re_list_r))
    print('space size: %d' % (len(list_c) * len(list_r)))
    para_space = []
    for index, (para_c, para_r) in enumerate(product(list_c, list_r)):
        para = (run_id, fold_id, k_fold, passes, step, para_c, para_r, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_solam, para_space)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_single_fsauc(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_r, para_g, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_r': para_r, 'para_g': para_g,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    f_name = join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, aucs, rts = c_algo_fsauc_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_r, para_g, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_r, para_g, results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_fsauc(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    list_g = 2. ** np.arange(-10, 11, 1, dtype=float)
    # by adding this step, we can reduce some redundant model space.
    if exists(join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))):
        f = join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))
        re_list_r, re_list_g = set(), set()
        for item in pkl.load(open(f, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_r.add(item['para_r'])
                re_list_g.add(item['para_g'])
        list_r, list_g = np.sort(list(re_list_r)), np.sort(list(re_list_g))
    print('space size: %d' % (len(list_r) * len(list_g)))
    para_space = []
    for index, (para_r, para_g) in enumerate(product(list_r, list_g)):
        para = (run_id, fold_id, k_fold, passes, step, para_r, para_g, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_fsauc, para_space)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_single_opauc(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_tau, para_eta, para_lam, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_tau': para_tau, 'para_eta': para_eta, 'para_lambda': para_lam,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, aucs, rts = c_algo_opauc_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_tau, para_eta, para_lam, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_eta, para_lam, results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_opauc(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    list_eta = 2. ** np.arange(-12, -4, 1, dtype=float)
    list_lambda = 2. ** np.arange(-10, -2, 1, dtype=float)
    # by adding this step, we can reduce some redundant model space.
    if os.path.exists(join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))):
        f = join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))
        re_list_eta, re_list_lambda = set(), set()
        for item in pkl.load(open(f, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_eta.add(item['para_eta'])
                re_list_lambda.add(item['para_lambda'])
        list_eta, list_lambda = np.sort(list(re_list_eta)), np.sort(list(re_list_lambda))
    print('space size: %d' % (len(list_eta) * len(list_lambda)))
    para_space = []
    for index, (para_tau, para_eta, para_lam) in enumerate(product([50], list_eta, list_lambda)):
        para = (run_id, fold_id, k_fold, passes, step, para_tau, para_eta, para_lam, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_opauc, para_space)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_single_sht_am(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_s, para_b, para_c, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float), 'run_time': np.zeros(k_fold, dtype=float),
               'para': para, 'para_s': para_s, 'para_b': para_b, 'para_c': para_c,
               'run_id': run_id, 'fold_id': fold_id, 'k_fold': k_fold,
               'num_passes': num_passes, 'step_len': step_len, 'data_name': data_name}
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, aucs, rts = c_algo_sht_am_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_s, para_b, para_c, 0.0, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_s, para_b, para_c,
              results['auc_arr'][ind], time.time() - s_time)
        sys.stdout.flush()
    return results


def cv_sht_am(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh):
    run_id, fold_id = task_id / 5, task_id % 5
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    list_s = [int(_ * data['p']) for _ in np.arange(0.1, 1.01, 0.1)]
    list_b = [10, 20, 40]
    list_c = np.arange(1., 101., 9)
    # by adding this step, we can reduce some redundant model space.
    if os.path.exists(join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))):
        f = join(data_path, '%s/ms_run_0_fold_0_%s.pkl' % (data_name, method))
        re_list_s, re_list_b, re_list_c = set(), set(), set()
        for item in pkl.load(open(f, 'rb')):
            if np.mean(item['auc_arr']) >= auc_thresh:
                re_list_s.add(item['para_s'])
                re_list_b.add(item['para_b'])
                re_list_c.add(item['para_c'])
        list_s, list_b = np.sort(list(re_list_s)), np.sort(list(re_list_b))
        list_c = np.sort(list(re_list_c))
    print('space size: %d' % (len(list_s) * len(list_b) * len(list_c)))
    para_space = []
    for index, (para_s, para_b, para_c) in enumerate(product(list_s, list_b, list_c)):
        para = (run_id, fold_id, k_fold, passes, step, para_s, para_b, para_c, data_name)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    ms_res = pool.map(run_single_sht_am, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    pkl.dump(ms_res, open(f_name, 'wb'))


def main():
    data_name, method, cpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) \
        if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    k_fold, passes, step, auc_thresh = 5, 20, 10000000, 0.8
    if method == 'spam_l1':
        cv_spam_l1(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    elif method == 'spam_l2':
        cv_spam_l2(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    elif method == 'spam_l1l2':
        cv_spam_l1l2(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    elif method == 'solam':
        cv_solam(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    elif method == 'fsauc':
        cv_fsauc(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    elif method == 'opauc':
        cv_opauc(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    elif method == 'sht_am':
        cv_sht_am(data_name, method, task_id, k_fold, passes, step, cpus, auc_thresh)
    else:
        print('other method ?')


if __name__ == '__main__':
    main()
