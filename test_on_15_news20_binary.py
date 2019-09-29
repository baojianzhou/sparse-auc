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

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/15-news20-binary/'


def get_sub_data_by_indices(data, tr_index, sub_tr_ind):
    sub_x_tr_vals = []
    sub_x_tr_inds = []
    sub_x_tr_lens = []
    sub_x_tr_posis = []
    prev_posi = 0
    for index in tr_index[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_posis'][index]
        sub_x_tr_vals.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_tr_inds.extend(data['x_tr_inds'][cur_posi:cur_posi + cur_len])
        sub_x_tr_lens.append(cur_len)
        sub_x_tr_posis.append(prev_posi)
        prev_posi += cur_len
    return sub_x_tr_vals, sub_x_tr_inds, sub_x_tr_posis, sub_x_tr_lens


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


def data_processing():
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_posis'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    words = dict()
    prev_posi = 0
    with open(os.path.join(data_path, 'news20.binary'), 'rb') as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(float(items[0]))
            data['x_tr_inds'].extend([int(_.split(':')[0]) - 1 for _ in items[1:]])
            data['x_tr_vals'].extend([float(_.split(':')[1]) for _ in items[1:]])
            data['x_tr_posis'].append(prev_posi)
            data['x_tr_lens'].append(len(items[1:]))
            prev_posi += len(items[1:])
            for _ in [int(_.split(':')[0]) - 1 for _ in items[1:]]:
                if _ not in words:
                    words[_] = 0
                words[_] += 1
    print(len(data['y_tr']))
    print(len(data['x_tr_inds']))
    print(len(words))
    print(len([_ for _ in words if words[_] <= 1]))
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_posis'] = np.asarray(data['x_tr_posis'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert 19996 == len(data['y_tr'])
    assert 1355191 == len(words)
    print(min(words.keys()), max(words.keys()))
    assert 0 == min(words.keys())  # make sure the feature index starts from 0.
    data['n'] = 19996
    data['p'] = 1355191
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    # randomly permute the datasets 25 times for future use.
    data['num_runs'] = 5
    data['num_k_fold'] = 5
    for run_index in range(data['num_runs']):
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        # just need the number of training samples.
        fake_x = np.zeros(shape=(data['n'], 1))
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['run_%d_fold_%d' % (run_index, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'processed_new20_binary.pkl'), 'wb'))
    return data


def cv_sht_am(run_id, fold_id, num_passes, data):
    para_b = 28
    list_c = list(10. ** np.arange(1, 3, 1, dtype=float))
    list_sparsity = [800000]
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
            _ = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
            sub_x_tr_vals, sub_x_tr_inds, sub_x_tr_posis, sub_x_tr_lens = _
            re = c_algo_sht_am_sparse(
                np.asarray(sub_x_tr_vals, dtype=float),
                np.asarray(sub_x_tr_inds, dtype=np.int32),
                np.asarray(sub_x_tr_posis, dtype=np.int32),
                np.asarray(sub_x_tr_lens, dtype=np.int32),
                np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float),
                data['p'], para_sparsity, para_b, para_c, 0.0, num_passes, step_len, verbose)
            wt, wt_bar = np.asarray(re[0]), np.asarray(re[1])
            y_pred_wt, y_pred_wt_bar = pred(data, tr_index, sub_te_ind, wt, wt_bar)
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
            print(list_auc_wt[ind], list_auc_wt_bar[ind], time.time() - s_time)
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


def run_ms(method_name):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0
    run_id, fold_id, num_passes = task_id / 5, task_id / 5, 10
    data = pkl.load(open(data_path + 'processed_new20_binary.pkl', 'rb'))
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


def main():
    run_ms(method_name='sht_am')


if __name__ == '__main__':
    main()
