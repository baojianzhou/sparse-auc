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


def cv_spam_l1(run_id, fold_id, k_fold, num_passes, step_len, data):
    list_c = np.arange(1, 101, 9, dtype=float)
    list_l1 = 10. ** np.arange(-5, 6, 1, dtype=float)
    s_time, models = time.time(), dict()
    for index, (para_c, para_l1) in enumerate(product(list_c, list_l1)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_spam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, para_l1, 0.0, 0, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_c, para_l1, auc_arr[ind])
        models[index] = {'para_c': para_c, 'para_l1': para_l1, 'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_c: %.4f para_l1: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_l1, float(np.mean(auc_arr)), time.time() - s_time))
    return models


def cv_spam_l2(run_id, fold_id, k_fold, num_passes, step_len, data):
    list_c = np.arange(1, 101, 9, dtype=float)
    list_l2 = 10. ** np.arange(-5, 6, 1, dtype=float)
    s_time, models = time.time(), dict()
    for index, (para_c, para_beta) in enumerate(product(list_c, list_l2)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_spam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, 0.0, para_beta, 1, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_c, para_beta, auc_arr[ind])
        models[index] = {'para_c': para_c, 'para_beta': para_beta, 'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_c: %.4f para_beta: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_beta, float(np.mean(auc_arr)), time.time() - s_time))
    return models


def cv_spam_l1l2(run_id, fold_id, k_fold, num_passes, step_len, data):
    s_time, models = time.time(), dict()
    list_c = np.arange(1, 101, 9, dtype=float)
    list_l1 = 10. ** np.arange(-5, 6, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 6, 1, dtype=float)
    for index, (para_c, para_l1, para_l2) in enumerate(product(list_c, list_l1, list_l2)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_spam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, para_l1, para_l2, 0, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_c, para_l1, para_l2, auc_arr[ind])
        models[index] = {'para_c': para_c, 'para_l2': para_l2, 'para_l1': para_l1,
                         'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_c: %.4f para_beta: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_l2, float(np.mean(auc_arr)), time.time() - s_time))
    return models


def cv_solam(run_id, fold_id, k_fold, num_passes, step_len, data):
    s_time, models = time.time(), dict()
    list_c = np.arange(1, 101, 9, dtype=float)
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    for index, (para_c, para_r) in enumerate(product(list_c, list_r)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, auc, rts = c_algo_solam_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_c, para_r, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_c, para_r, auc_arr[ind])
        models[index] = {'para_c': para_c, 'para_r': para_r, 'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_xi: %.4f para_r: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_c, para_r, float(np.mean(auc_arr)), time.time() - s_time))
    return models


def cv_fsauc(run_id, fold_id, k_fold, num_passes, step_len, data):
    s_time, models = time.time(), dict()
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    list_g = 2. ** np.arange(-10, 11, 1, dtype=float)
    for index, (para_r, para_g) in enumerate(product(list_r, list_g)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, aucs, rts = c_algo_fsauc_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_r, para_g, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_r, para_g, auc_arr[ind])
        models[index] = {'para_r': para_r, 'para_g': para_g, 'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_r: %.4f para_g: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_r, para_g, float(np.mean(auc_arr)), time.time() - s_time))
    return models


def cv_opauc(run_id, fold_id, k_fold, num_passes, step_len, data):
    s_time, models = time.time(), dict()
    list_tau = [50]
    list_eta = 2. ** np.arange(-12, -4, 1, dtype=float)
    list_lambda = 2. ** np.arange(-10, -2, 1, dtype=float)
    for index, (para_tau, para_eta, para_lambda) in enumerate(
            product(list_tau, list_eta, list_lambda)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, aucs, rts = c_algo_opauc_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_tau, para_eta, para_lambda, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_tau, para_eta, para_lambda, auc_arr[ind])
        models[index] = {'para_tau': para_tau, 'para_eta': para_eta, 'para_lambda': para_lambda,
                         'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_eta: %.4f para_lambda: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_eta, para_lambda,
               float(np.mean(auc_arr)), time.time() - s_time))
    return models


def cv_sht_am(run_id, fold_id, k_fold, num_passes, step_len, data):
    s_time, models = time.time(), dict()
    list_s = [10000, 15000, 20000, 25000, 30000, 35000, 40000]
    list_b = [108]
    list_c = np.arange(1, 101, 9, dtype=float)
    for index, (para_s, para_b, para_c) in enumerate(product(list_s, list_b, list_c)):
        tr_index = data['fold_%d' % fold_id]['tr_index']
        auc_arr, fold = np.zeros(k_fold, dtype=float), KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
            x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
            wt, wt_bar, aucs, rts = c_algo_sht_am_sparse(
                x_vals, x_inds, x_posis, x_lens, y_tr,
                data['p'], para_s, para_b, para_c, 0.0, num_passes, step_len, 0)
            auc_arr[ind] = pred_auc(data, tr_index, sub_te_ind, wt)
            print(para_s, para_b, para_c, auc_arr[ind])
        models[index] = {'para_s': para_s, 'para_b': para_b, 'para_c': para_c, 'auc_arr': auc_arr}
        print('run_%02d_fold_%d para_s: %.4f para_b: %.4f AUC: %.4f run_time: %.2f' %
              (run_id, fold_id, para_s, para_b, float(np.mean(auc_arr)), time.time() - s_time))
    return models


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
    run_id, fold_id, k_fold, passes, step_len = task_id / 5, task_id % 5, 5, 20, 10000000
    data = pkl.load(open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'rb'))
    results, key = dict(), (run_id, fold_id)
    if method_name == 'spam_l1':
        results[key] = dict()
        results[key][method_name] = cv_spam_l1(run_id, fold_id, k_fold, passes, step_len, data)
    elif method_name == 'spam_l2':
        results[key] = dict()
        results[key][method_name] = cv_spam_l2(run_id, fold_id, k_fold, passes, step_len, data)
    elif method_name == 'spam_l1l2':
        results[key] = dict()
        results[key][method_name] = cv_spam_l1l2(run_id, fold_id, k_fold, passes, step_len, data)
    elif method_name == 'solam':
        results[key] = dict()
        results[key][method_name] = cv_solam(run_id, fold_id, k_fold, passes, step_len, data)
    elif method_name == 'fsauc':
        results[key] = dict()
        results[key][method_name] = cv_fsauc(run_id, fold_id, k_fold, passes, step_len, data)
    elif method_name == 'opauc':
        results[key] = dict()
        results[key][method_name] = cv_opauc(run_id, fold_id, k_fold, passes, step_len, data)
    elif method_name == 'sht_am':
        results[key] = dict()
        results[key][method_name] = cv_sht_am(run_id, fold_id, k_fold, passes, step_len, data)
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


def test():
    exit()
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
    _ = ms[key][method][0][(run_id, fold_id)]['para']
    wt, wt_bar, auc, rts = c_algo_fsauc_sparse(
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
        data['p'], _[3], _[4], num_passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    f_name = 'results_task_%02d_passes_%02d.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % (task_id, num_passes)), 'wb'))


def run_testing():
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    for task_id in range(25):
        run_id, fold_id, num_passes, step_len = task_id / 5, task_id % 5, 50, 10000000
        data = pkl.load(open(data_path + 'processed_sector_normalized.pkl', 'rb'))

        ms_f_name = data_path + 'ms_task_%02d_%s.pkl'
        tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
        te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
        _ = get_data_by_ind(data, tr_index, range(len(tr_index)))
        x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr = _
        results, key = dict(), (run_id, fold_id)
        results[key] = dict()
        # -----------------------
        method = 'solam'
        ms = pkl.load(open(ms_f_name % (task_id, method), 'rb'))
        for item in ms:
            _, _, _, para_xi, para_r = ms[item][method][0][(run_id, task_id / 5)]['para']
        wt, wt_bar, auc, rts = c_algo_solam_sparse(
            x_tr_vals, x_tr_inds, x_tr_posis, x_tr_lens, y_tr,
            data['p'], para_xi, para_r, num_passes, step_len, 0)
        results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_xi, para_r), te_index,
                                            data)
        print(fold_id, method, results[key][method]['auc_wt'], time.time() - s_time)


def main():
    run_ms(method_name='solam')


if __name__ == '__main__':
    main()
