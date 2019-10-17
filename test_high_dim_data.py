# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
from os.path import join
from itertools import product

import numpy as np
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

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'


def get_data_by_ind(data, tr_ind, sub_tr_ind):
    sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens = [], [], [], []
    prev_posi = 0
    for index in tr_ind[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_poss'][index]
        sub_x_vals.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_inds.extend(data['x_tr_inds'][cur_posi:cur_posi + cur_len])
        sub_x_lens.append(cur_len)
        sub_x_posis.append(prev_posi)
        prev_posi += cur_len
    sub_x_vals = np.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_posis = np.asarray(sub_x_posis, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data['y_tr'][tr_ind[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_tr


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


def pred_results(wt, wt_bar, auc, rts, para_list, te_index, data):
    return {'auc_wt': pred_auc(data, te_index, range(len(te_index)), wt),
            'auc_wt_bar': pred_auc(data, te_index, range(len(te_index)), wt_bar),
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar),
            'algo_para': para_list,
            'auc': auc,
            'rts': rts}


def get_model_para(data_name, method, run_id, fold_id):
    f_name = join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' % (data_name, run_id, fold_id, method))
    ms = pkl.load(open(f_name, 'rb'))
    sm = {'aver_auc': 0.0}
    for re_row in ms:
        mean_auc = np.mean(re_row['auc_arr'])
        if method == 'sht_am':
            if sm['aver_auc'] < mean_auc:
                sm['para'] = re_row['para']
                sm['aver_auc'] = mean_auc
        else:
            if sm['aver_auc'] < mean_auc:
                sm['para'] = re_row['para']
                sm['aver_auc'] = mean_auc
    if method == 'spam_l1':
        para_c, para_l1 = sm['para'][5], sm['para'][6]
        return para_c, para_l1
    elif method == 'spam_l2':
        para_c, para_l2 = sm['para'][5], sm['para'][6]
        return para_c, para_l2
    elif method == 'spam_l1l2':
        para_c, para_l1, para_l2 = sm['para'][5], sm['para'][6], sm['para'][7]
        return para_c, para_l1, para_l2
    elif method == 'solam':
        para_c, para_r = sm['para'][5], sm['para'][6]
        return para_c, para_r
    elif method == 'fsauc':
        para_r, para_g = sm['para'][5], sm['para'][6]
        return para_r, para_g
    elif method == 'opauc':
        para_tau, para_eta, para_lambda = sm['para'][5], sm['para'][6], sm['para'][7]
        return para_tau, para_eta, para_lambda
    elif method == 'sht_am':
        para_s, para_b, para_c, para_l2 = sm['para'][5:9]
        return para_s, para_b, para_c, para_l2
    return sm


def run_single_spam_l1(para):
    run_id, fold_id, k_fold, passes, step_len, para_c, para_l1, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, para_l1, 0.0, 0, passes, step_len, 0)
    res = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_c, para_l1, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_spam_l1(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_c, para_l1 = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l1, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_spam_l1, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def run_single_spam_l2(para):
    run_id, fold_id, k_fold, passes, step_len, para_c, para_l2, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, 0.0, para_l2, 1, passes, step_len, 0)
    res = pred_results(wt, wt_bar, auc, rts, (para_c, para_l2), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_c, para_l2, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_spam_l2(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_c, para_l2 = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l2, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_spam_l2, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def run_single_spam_l1l2(para):
    run_id, fold_id, k_fold, passes, step_len, para_c, para_l1, para_l2, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, para_l1, para_l2, 0, passes, step_len, 0)
    res = pred_results(wt, wt_bar, auc, rts, (para_c, para_l2), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_c, para_l2, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_spam_l1l2(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_c, para_l1, para_l2 = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len,
                para_c, para_l1, para_l2, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_spam_l1l2, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def run_single_solam(para):
    run_id, fold_id, k_fold, passes, step_len, para_c, para_r, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_solam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, para_r, passes, step_len, 0)
    res = pred_results(wt, wt_bar, auc, rts, (para_c, para_r), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_c, para_r, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_solam(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_c, para_r = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_r, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_solam, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def run_single_fsauc(para):
    run_id, fold_id, k_fold, passes, step_len, para_r, para_g, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_fsauc_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_r, para_g, passes, step_len, 0)
    res = pred_results(wt, wt_bar, auc, rts, (para_r, para_g), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_r, para_g, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_fsauc(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_r, para_g = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len, para_r, para_g, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_fsauc, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def run_single_sht_am(para):
    run_id, fold_id, k_fold, passes, step, \
    para_s, para_b, para_c, para_l2, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_sht_am_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_s, para_b, para_c, 0.0, passes, step, 1)
    res = pred_results(wt, wt_bar, auc, rts, (para_s, para_b, para_c), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_s, para_b, para_c, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_sht_am(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_s, para_b, para_c, para_l2, = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len,
                para_s, para_b, para_c, para_l2, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_sht_am, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def run_single_opauc(para):
    run_id, fold_id, k_fold, passes, step, para_tau, para_eta, para_lam, data_name, method = para
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_opauc_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_tau, para_eta, para_lam, passes, step, 0)
    res = pred_results(wt, wt_bar, auc, rts, (para_tau, para_eta, para_lam), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_tau, para_eta, para_lam, auc, run_time)
    sys.stdout.flush()
    return {(run_id, fold_id): res}


def test_opauc(data_name, method, k_fold, passes, step_len, cpus):
    para_space = []
    for index, (run_id, fold_id) in enumerate(product(range(5), range(5))):
        para_tau, para_eta, para_lambda = get_model_para(data_name, method, run_id, fold_id)
        para = (run_id, fold_id, k_fold, passes, step_len,
                para_tau, para_eta, para_lambda, data_name, method)
        para_space.append(para)
    pool = multiprocessing.Pool(processes=cpus)
    test_res = pool.map(run_single_opauc, para_space)
    pool.close()
    pool.join()
    f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
    pkl.dump(test_res, open(f_name, 'wb'))


def main():
    data_name, method, cpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
    k_fold, passes, step = 5, 20, 20
    if method == 'spam_l1':
        test_spam_l1(data_name, method, k_fold, passes, step, cpus)
    elif method == 'spam_l2':
        test_spam_l2(data_name, method, k_fold, passes, step, cpus)
    elif method == 'spam_l1l2':
        test_spam_l1l2(data_name, method, k_fold, passes, step, cpus)
    elif method == 'solam':
        test_solam(data_name, method, k_fold, passes, step, cpus)
    elif method == 'fsauc':
        test_fsauc(data_name, method, k_fold, passes, step, cpus)
    elif method == 'opauc':
        test_opauc(data_name, method, k_fold, passes, step, cpus)
    elif method == 'sht_am':
        test_sht_am(data_name, method, k_fold, passes, step, cpus)
    else:
        print('other method ?')


if __name__ == '__main__':
    main()
