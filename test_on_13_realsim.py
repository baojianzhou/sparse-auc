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

from data_preprocess import load_dataset

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_spam_sparse
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

root_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'


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


def sparse_dot(x_indices, x_values, wt):
    y_score = np.zeros(len(x_values))
    for i in range(len(x_values)):
        for j in range(1, x_indices[i][0] + 1):
            y_score[i] += wt[x_indices[i][j]] * x_values[i][j]
    return y_score


def get_sub_data_by_indices(data, tr_index, sub_tr_ind):
    sub_x_values = []
    sub_x_indices = []
    sub_x_len_list = []
    sub_x_positions = []
    prev_posi = 0
    for index in tr_index[sub_tr_ind]:
        cur_len = data['x_len_list'][index]
        cur_posi = data['x_positions'][index]
        sub_x_values.extend(data['x_values'][cur_posi:cur_posi + cur_len])
        sub_x_indices.extend(data['x_indices'][cur_posi:cur_posi + cur_len])
        sub_x_len_list.append(cur_len)
        sub_x_positions.append(prev_posi)
        prev_posi += cur_len
    return sub_x_values, sub_x_indices, sub_x_positions, sub_x_len_list


def test_single_ms_spam_l2(para):
    """
    Model selection of SPAM-l2
    :return:
    """
    run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold = para
    s_time = time.time()
    data = load_dataset(root_path=root_path, name='realsim')
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_beta': para_beta,
                   'para_l1_reg': 0.0,  # no l1 regularization needed.
                   'para_xi': para_xi,
                   'para_fold_id': fold_id,
                   'para_run_id': run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 1,
                   'para_step_len': 400000000,
                   'para_reg_opt': 1,
                   'data_id': 13,
                   'data_name': '13_realsim'}
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
    print('number of tr: %d number of te: %d' % (len(tr_index), len(te_index)))
    # cross validate based on tr_index
    list_auc_wt = np.zeros(para_spaces['conf_k_fold'])
    list_auc_wt_bar = np.zeros(para_spaces['conf_k_fold'])
    kf = KFold(n_splits=para_spaces['conf_k_fold'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        re = get_sub_data_by_indices(data, tr_index, sub_tr_ind)
        sub_x_tr_values, sub_x_tr_indices, sub_x_tr_positions, sub_x_tr_len_list = re
        sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
        re = c_algo_spam_sparse(np.asarray(sub_x_tr_values, dtype=float),
                                np.asarray(sub_x_tr_indices, dtype=np.int32),
                                np.asarray(sub_x_tr_positions, dtype=np.int32),
                                np.asarray(sub_x_tr_len_list, dtype=np.int32),
                                np.asarray(sub_y_tr, dtype=float),
                                len(sub_y_tr),
                                data['p'],
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
        re = get_sub_data_by_indices(data, tr_index, sub_te_ind)
        sub_x_te_values, sub_x_te_indices, sub_x_te_positions, sub_x_te_len_list = re
        sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
        y_pred_wt, y_pred_wt_bar = np.zeros_like(sub_y_te), np.zeros_like(sub_y_te)
        for i in range(len(sub_te_ind)):
            cur_posi = sub_x_te_positions[i]
            cur_len = sub_x_te_len_list[i]
            cur_x = sub_x_te_values[cur_posi:cur_posi + cur_len]
            cur_ind = sub_x_te_indices[cur_posi:cur_posi + cur_len]
            y_pred_wt[i] = np.sum([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
            y_pred_wt_bar[i] = np.sum([cur_x[_] * wt_bar[cur_ind[_]] for _ in range(cur_len)])
        list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)
        list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt_bar)
    print('run_id, fold_id, para_xi, para_beta: ', run_id, fold_id, para_xi, para_beta)
    print('list_auc_wt:', list_auc_wt)
    print('list_auc_wt_bar:', list_auc_wt_bar)
    run_time = time.time() - s_time
    return {'algo_para_name_list': ['run_id', 'fold_id', 'para_xi', 'para_beta',
                                    'num_passes', 'num_runs', 'k_fold'],
            'algo_para': para, 'para_spaces': para_spaces,
            'list_auc_wt': list_auc_wt,
            'list_auc_wt_bar': list_auc_wt_bar,
            'run_time': run_time}


def run_model_selection_spam_l2():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    num_runs, k_fold, num_tasks, global_passes = 5, 5, 25, 80
    all_para_space = []
    for run_id, fold_id in product(range(num_runs), range(k_fold)):
        for num_passes in [global_passes]:
            for para_xi in 10. ** np.arange(-7, -2., 0.5, dtype=float):
                for para_beta in 10. ** np.arange(-6, 1, 1, dtype=float):
                    para_row = (run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold)
                    all_para_space.append(para_row)
    # only run sub-tasks for parallel
    num_sub_tasks = len(all_para_space) / num_tasks
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = all_para_space[task_start:task_end]
    list_results = []
    for task_para in list_tasks:
        result = test_single_ms_spam_l2(task_para)
        list_results.append(result)
    file_name = 'ms_spam_l2_task_%02d_passes_%03d.pkl' % (task_id, global_passes)
    pkl.dump(list_results, open(os.path.join(root_path, file_name), 'wb'))


def run_model_selection_sht_am():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    num_runs, k_fold, num_tasks, global_passes = 5, 5, 25, 10
    all_para_space = []
    for run_id, fold_id in product(range(num_runs), range(k_fold)):
        for num_passes in [global_passes]:
            for para_xi in 10. ** np.arange(-7, -2., 0.5, dtype=float):
                for para_beta in 10. ** np.arange(-6, 1, 1, dtype=float):
                    para_row = (run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold)
                    all_para_space.append(para_row)
    # only run sub-tasks for parallel
    num_sub_tasks = len(all_para_space) / num_tasks
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = all_para_space[task_start:task_end]
    list_results = []
    for task_para in list_tasks:
        result = test_single_ms_spam_l2(task_para)
        list_results.append(result)
    file_name = 'ms_spam_l2_task_%02d_passes_%03d.pkl' % (task_id, global_passes)
    pkl.dump(list_results, open(os.path.join(root_path, file_name), 'wb'))


def run_spam_l2_by_selected_model(id_=None, model='wt', num_passes=1):
    """
    25 tasks to finish
    :return:
    """
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        if id_ is None:
            task_id = 1
        else:
            task_id = id_

    all_results, num_runs, k_fold = [], 5, 5
    for task_id in range(num_runs * k_fold):
        f_name = root_path + 'ms_spam_l2_task_%02d_passes_%03d.pkl' % (task_id, num_passes)
        all_results.extend(pkl.load(open(f_name, 'rb')))
    # selected model
    selected_model = dict()
    for result in all_results:
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
    data = load_dataset()
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
                   'data_id': 02,
                   'data_name': '02_usps'}
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
    pkl.dump(re, open(data_path + 're_spam_l2_%d_%d_%04d_%s.pkl' %
                      (selected_run_id, selected_fold_id, num_passes, model), 'wb'))


def final_result_analysis_spam_l2(num_passes=1, model='wt'):
    list_auc = []
    list_time = []
    for (run_id, fold_id) in product(range(5), range(5)):
        re = pkl.load(open(data_path + 're_spam_l2_%d_%d_%04d_%s.pkl' %
                           (run_id, fold_id, num_passes, model), 'rb'))
        list_auc.append(re['auc_%s' % model])
        list_time.append(re['run_time'])
    print('num_passes: %d %s mean: %.4f, std: %.4f' %
          (num_passes, model, float(np.mean(list_auc)), float(np.std(list_auc))))


def show_graph():
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


def run_test():
    for num_passes in [1, 5, 10, 20, 30, 40, 50]:
        run_spam_l2_by_selected_model(id_=None, model='wt', num_passes=num_passes)
        run_spam_l2_by_selected_model(id_=None, model='wt_bar', num_passes=num_passes)


def run_test_result():
    for i in [1, 5, 10, 20, 30, 40, 50]:
        final_result_analysis_spam_l2(i, 'wt')
        final_result_analysis_spam_l2(i, 'wt_bar')


def main():
    run_model_selection_sht_am()


if __name__ == '__main__':
    main()
