# -*- coding: utf-8 -*-
import os
import sys
import csv
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


def run_single_spam_l1(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_l1, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float),
               'run_time': np.zeros(k_fold, dtype=float),
               'para': para}
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
    return results


def run_single_spam_l2(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_l2, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float),
               'run_time': np.zeros(k_fold, dtype=float),
               'para': para}
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    fold = KFold(n_splits=k_fold, shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(fold.split(tr_index)):
        s_time = time.time()
        x_vals, x_inds, x_posis, x_lens, y_tr = get_data_by_ind(data, tr_index, sub_tr_ind)
        wt, wt_bar, auc, rts = c_algo_spam_sparse(
            x_vals, x_inds, x_posis, x_lens, y_tr,
            data['p'], para_c, 0.0, para_l2, 1, num_passes, step_len, 0)
        results['auc_arr'][ind] = pred_auc(data, tr_index, sub_te_ind, wt)
        results['run_time'][ind] = time.time() - s_time
        print(run_id, fold_id, para_c, para_l2, results['auc_arr'][ind], time.time() - s_time)
    return results


def run_single_spam_l1l2(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_l1, para_l2, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float),
               'run_time': np.zeros(k_fold, dtype=float),
               'para': para}
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
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
    return results


def run_single_solam(para):
    run_id, fold_id, k_fold, num_passes, step_len, para_c, para_r, data_name = para
    results = {'auc_arr': np.zeros(k_fold, dtype=float),
               'run_time': np.zeros(k_fold, dtype=float),
               'para': para}
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
    return results


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


def load_data(data_name, run_id):
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    data = pkl.load(open(f_name, 'rb'))
    return data


def run_ms(data_name, method_name):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    run_id, fold_id, k_fold, passes, step_len = task_id / 5, task_id % 5, 5, 20, 10000000
    data = load_data(data_name=data_name, run_id=run_id)
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
    f_name = os.path.join(data_path, '%s/ms_task_%02d_%s.pkl' % (data_name, task_id, method_name))
    pkl.dump(results, open(f_name, 'wb'))


def pred_results(wt, wt_bar, auc, rts, para_list, te_index, data):
    return {'auc_wt': pred_auc(data, te_index, range(len(te_index)), wt),
            'nonzero_wt': np.count_nonzero(wt),
            'algo_para': para_list,
            'auc': auc,
            'rts': rts,
            'nonzero_wt_bar': np.count_nonzero(wt_bar)}


def get_model(method, task_id, run_id, fold_id):
    ms = pkl.load(open(data_path + 'ms_task_%02d_%s.pkl' % (task_id, method), 'rb'))
    selected_model = {'aver_auc': 0.0}
    for index in ms[(run_id, fold_id)][method]:
        mean_auc = np.mean(ms[(run_id, fold_id)][method]['auc_arr'])
        if selected_model['aver_auc'] < mean_auc:
            selected_model['model'] = ms[(run_id, fold_id)][method][index]
            selected_model['aver_auc'] = mean_auc
    return selected_model


def run_testing():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    run_id, fold_id, k_fold, passes, step_len = task_id / 5, task_id % 5, 5, 20, 10000000
    data = pkl.load(open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'rb'))
    tr_index = data['fold_%d' % fold_id]['tr_index']
    te_index = data['fold_%d' % fold_id]['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))

    results, key = dict(), (run_id, fold_id)
    # -----------------------
    method = 'spam_l1'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_c, para_l1 = model['model']['para_c'], model['model']['para_l1']
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, para_l1, 0.0, 0, passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'spam_l2'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_c, para_l2 = model['model']['para_c'], model['model']['para_l2']
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, 0.0, para_l2, 1, passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l2), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'sht_am'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_s, para_c, b = model['model']['para_s'], model['model']['para_c'], 135
    wt, wt_bar, auc, rts = c_algo_sht_am_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_s, b, para_c, 0.0, passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_s, para_c, b), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'fsauc'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_r, para_g = model['model']['para_r'], model['model']['para_g']
    wt, wt_bar, auc, rts = c_algo_fsauc_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_r, para_g, passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'solam'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_c, para_r = model['model']['para_c'], model['model']['para_r']
    wt, wt_bar, auc, rts = c_algo_solam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, para_r, passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, (para_c, para_r), te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'opauc'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_tau, para_eta = model['model']['para_tau'], model['model']['para_eta']
    para_lambda = model['model']['para_lambda']
    wt, wt_bar, aucs, rts = c_algo_opauc_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_tau, para_eta, para_lambda, passes, step_len, 0)
    para_list = (para_tau, para_eta, para_lambda)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, para_list, te_index, data)
    print(fold_id, method, results[key][method]['auc_wt'])
    # -----------------------
    method = 'spam_l1l2'
    model = get_model(method=method, task_id=task_id, run_id=run_id, fold_id=fold_id)
    para_l1, para_l2 = model['model']['para_l1'], model['model']['para_l2']
    para_c = model['model']['para_c']
    para_list = (para_c, para_l1, para_l2)
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr,
        data['p'], para_c, para_l1, para_l2, 0, passes, step_len, 0)
    results[key][method] = pred_results(wt, wt_bar, auc, rts, para_list, te_index, data)

    f_name = 'results_task_%02d_passes_%02d.pkl'
    pkl.dump(results, open(os.path.join(data_path, f_name % (task_id, passes)), 'wb'))


def para_spaces():
    method_list = ['spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'opauc',
                   'solam', 'sht_am', 'graph_am']
    data_list = ['09-sector', '13-realsim']
    para = {'method_list': method_list, 'data_list': data_list}
    return para


def main():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    run_id, fold_id, k_fold, passes, step_len = task_id / 5, task_id % 5, 5, 20, 10000000
    data_name, method_name, num_cpus = sys.argv[1], sys.argv[2], 25
    if method_name == 'spam_l1':
        list_c = np.arange(1, 101, 9, dtype=float)
        list_l1 = 10. ** np.arange(-5, 6, 1, dtype=float)
        input_data_list = []
        for index, (para_c, para_l1) in enumerate(product(list_c, list_l1)):
            para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l1, data_name)
            input_data_list.append(para)
        pool = multiprocessing.Pool(processes=num_cpus)
        results = pool.map(run_single_spam_l1, input_data_list)
        pool.close()
        pool.join()
        f_name = os.path.join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' %
                              (data_name, run_id, fold_id, method_name))
        pkl.dump(results, open(f_name, 'wb'))
    elif method_name == 'spam_l2':
        list_c = np.arange(1, 101, 9, dtype=float)
        list_l2 = 10. ** np.arange(-5, 6, 1, dtype=float)
        input_data_list = []
        for index, (para_c, para_l2) in enumerate(product(list_c, list_l2)):
            para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l2, data_name)
            input_data_list.append(para)
        pool = multiprocessing.Pool(processes=num_cpus)
        results = pool.map(run_single_spam_l2, input_data_list)
        pool.close()
        pool.join()
        f_name = os.path.join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' %
                              (data_name, run_id, fold_id, method_name))
        pkl.dump(results, open(f_name, 'wb'))
    elif method_name == 'spam_l1l2':
        list_c = np.arange(1, 101, 9, dtype=float)
        list_l1 = 10. ** np.arange(-5, 6, 1, dtype=float)
        list_l2 = 10. ** np.arange(-5, 6, 1, dtype=float)
        input_data_list = []
        for index, (para_c, para_l1, para_l2) in enumerate(product(list_c, list_l1, list_l2)):
            para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_l1, para_l2, data_name)
            input_data_list.append(para)
        pool = multiprocessing.Pool(processes=num_cpus)
        results = pool.map(run_single_spam_l1l2, input_data_list)
        pool.close()
        pool.join()
        f_name = os.path.join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' %
                              (data_name, run_id, fold_id, method_name))
        pkl.dump(results, open(f_name, 'wb'))
    elif method_name == 'solam':
        list_c = np.arange(1, 101, 9, dtype=float)
        list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
        input_data_list = []
        for index, (para_c, para_r) in enumerate(product(list_c, list_r)):
            para = (run_id, fold_id, k_fold, passes, step_len, para_c, para_r, data_name)
            input_data_list.append(para)
        pool = multiprocessing.Pool(processes=num_cpus)
        results = pool.map(run_single_solam, input_data_list)
        pool.close()
        pool.join()
        f_name = os.path.join(data_path, '%s/ms_run_%d_fold_%d_%s.pkl' %
                              (data_name, run_id, fold_id, method_name))
        pkl.dump(results, open(f_name, 'wb'))


if __name__ == '__main__':
    main()
