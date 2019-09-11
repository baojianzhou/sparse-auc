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
        from sparse_module import c_algo_spam
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/'


def load_dataset():
    """
    number of samples: 9,298
    number of features: 256
    :return:
    """
    if os.path.exists(data_path + 'processed_usps.pkl'):
        return pkl.load(open(data_path + 'processed_usps.pkl', 'rb'))
    data = dict()
    data['x_tr'] = []
    data['y_tr'] = []
    with open(data_path + 'processed_usps.txt') as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            cur_x = [float(_.split(':')[1]) for _ in items[1:]]
            # normalize the data.
            data['x_tr'].append(cur_x / np.linalg.norm(cur_x))
    data['x_tr'] = np.asarray(data['x_tr'], dtype=float)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert len(data['y_tr']) == 9298  # total samples in train
    data['n'] = 9298
    data['p'] = 256
    assert len(np.unique(data['y_tr'])) == 10  # we have total 10 classes.
    rand_ind = np.random.permutation(len(np.unique(data['y_tr'])))
    posi_classes = rand_ind[:len(np.unique(data['y_tr'])) / 2]
    nega_classes = rand_ind[len(np.unique(data['y_tr'])) / 2:]
    posi_indices = [ind for ind, _ in enumerate(data['y_tr']) if _ in posi_classes]
    nega_indices = [ind for ind, _ in enumerate(data['y_tr']) if _ in nega_classes]
    data['y_tr'][posi_indices] = 1
    data['y_tr'][nega_indices] = -1
    print('number of positive: %d' % len(posi_indices))
    print('number of negative: %d' % len(nega_indices))
    data['num_posi'] = len(posi_indices)
    data['num_nega'] = len(nega_indices)
    # randomly permute the datasets 25 times for future use.
    data['num_runs'] = 5
    data['num_k_fold'] = 5
    for run_index in range(data['num_runs']):
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        # just need the number of training samples
        fake_x = np.zeros(shape=(data['n'], 1))
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['run_%d_fold_%d' % (run_index, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                'te_index': rand_perm[test_index]}
    pkl.dump(data, open(data_path + 'processed_usps.pkl', 'wb'))
    return data


def get_run_fold_index_by_task_id(method, task_start, task_end):
    if method == 'spam':
        num_passes, num_runs, k_fold = 50, 5, 5
        para_space = []
        for run_id in range(5):
            for fold_id in range(5):
                for para_xi in np.arange(1, 61, 5, dtype=float):
                    for para_beta in 10. ** np.arange(-5, 5, 1, dtype=float):
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


def test_single_ms_spam_l2(para):
    """
    Model selection of SPAM-l2
    :return:
    """
    run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold = para

    s_time = time.time()
    data = load_dataset()
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_beta': para_beta,
                   'para_l1_reg': 0.0,  # no l1 regularization needed.
                   'para_xi': para_xi,
                   'para_fold_id': fold_id,
                   'para_run_id': run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 0,
                   'para_step_len': 2000,
                   'para_reg_opt': 1,
                   'data_id': 02,
                   'data_name': '02_usps'}
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
    print('number of tr: %d number of te: %d' % (len(tr_index), len(te_index)))
    # cross validate based on tr_index
    list_auc = np.zeros(para_spaces['global_cv'])
    kf = KFold(n_splits=para_spaces['global_cv'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
        sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
        sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
        sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
        re = c_algo_spam(np.asarray(sub_x_tr, dtype=float),
                         np.asarray(sub_y_tr, dtype=float),
                         para_spaces['para_xi'],
                         para_spaces['para_l1_reg'],
                         para_spaces['para_beta'],
                         para_spaces['para_reg_opt'],
                         para_spaces['para_num_passes'],
                         para_spaces['para_step_len'],
                         para_spaces['para_is_sparse'],
                         para_spaces['para_verbose'])
        wt_bar = np.asarray(re[0])  # use wt_bar as the output
        list_auc[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
    print('run_id, fold_id, para_xi, para_beta: ', run_id, fold_id, para_xi, para_beta)
    print('list_auc:', list_auc)
    run_time = time.time() - s_time
    return {'algo_para': [run_id, fold_id, num_passes, para_xi, para_beta],
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


def run_model_selection_spam_l2():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    num_sub_tasks = 30
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = get_run_fold_index_by_task_id('spam', task_start, task_end)
    list_results = []
    for task_para in list_tasks:
        (run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold) = task_para
        result = test_single_ms_spam_l2(run_id, fold_id, para_xi, para_beta, num_passes, num_runs,
                                        k_fold)
        list_results.append(result)
    # model selection of spam-l2
    file_name = data_path + 'ms_spam_l2_%04d_%04d_%04d.pkl' % (task_start, task_end, num_passes)
    pkl.dump(list_results, open(file_name, 'wb'))


def model_result_analysis():
    all_results = []
    for task_id in range(100):
        task_start, task_end = int(task_id) * 30, int(task_id) * 30 + 30
        f_name = data_path + 'ms_spam_l2_%04d_%04d_%04d.pkl' % (task_start, task_end, 50)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)
    results = all_results
    max_auc_dict = dict()
    for result in results:
        run_id, fold_id, num_passes, para_xi, para_r = result['algo_para']
        mean_auc = np.mean(result['list_auc'])
        if (run_id, fold_id) not in max_auc_dict:
            max_auc_dict[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
        if mean_auc > max_auc_dict[(run_id, fold_id)][0]:
            max_auc_dict[(run_id, fold_id)] = (mean_auc, run_id, fold_id, para_xi, para_r)
    list_best_auc = []
    for key in max_auc_dict:
        print(key, max_auc_dict[key])
        list_best_auc.append(max_auc_dict[key][0])
    print('mean_auc: %.4f std_auc: %.4f' % (
        float(np.mean(list_best_auc)), float(np.std(list_best_auc))))


def run_spam_l2_by_selected_model(id_=None):
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
    num_passes = 50
    all_results = []
    for i in range(100):
        task_start, task_end = int(i) * 30, int(i) * 30 + 30
        f_name = data_path + 'ms_spam_l2_%04d_%04d_%04d.pkl' % (task_start, task_end, num_passes)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)

    # selected model
    selected_model = dict()
    for result in all_results:
        run_id, fold_id, num_passes, para_xi, para_beta = result['algo_para']
        mean_auc = np.mean(result['list_auc'])
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
    para_spaces = {'global_pass': num_passes,
                   'global_runs': 5,
                   'global_cv': 5,
                   'data_id': 5,
                   'data_name': '09_sector',
                   'data_dim': data['p'],
                   'data_num': data['n'],
                   'verbose': 0}
    tr_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (selected_run_id, selected_fold_id)]['te_index']
    verbose, step_len, l1_reg, l2_reg, = 0, 2000, 0.0, selected_para_beta
    reg_opt, is_sparse = 1, 0
    re = c_algo_spam(np.asarray(data['x_tr'][tr_index], dtype=float),
                     np.asarray(data['y_tr'][tr_index], dtype=float),
                     float(selected_para_xi), float(l1_reg), float(selected_para_beta),
                     int(reg_opt), int(num_passes), int(step_len), int(is_sparse), int(verbose))
    wt_bar = np.asarray(re[1])
    auc = roc_auc_score(y_true=data['y_tr'][te_index],
                        y_score=np.dot(data['x_tr'][te_index], wt_bar))
    print('run_id, fold_id, para_xi, para_r: ',
          selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta)
    run_time = time.time() - s_time
    print('auc:', auc, 'run_time', run_time)
    re = {'algo_para': [selected_run_id, selected_fold_id, selected_para_xi, selected_para_beta],
          'para_spaces': para_spaces, 'auc': auc, 'run_time': run_time}
    pkl.dump(re, open(data_path + 'result_spam_l2_%d_%d_passes_%02d.pkl' %
                      (selected_run_id, selected_fold_id, num_passes), 'wb'))


def final_result_analysis_spam_l2():
    list_auc = []
    list_time = []
    for (run_id, fold_id) in product(range(5), range(5)):
        re = pkl.load(open(data_path + 'result_spam_l2_%d_%d_passes_%02d.pkl' %
                           (run_id, fold_id, 50), 'rb'))
        print(re['auc'], re['run_time'])
        list_auc.append(re['auc'])
        list_time.append(re['run_time'])
    print('mean: %.4f, std: %.4f' % (np.mean(list_auc), np.std(list_auc)))
    print('total run time in aveage: %.4f run time per-iteration: %.4f' %
          (np.mean(list_time), np.mean(list_time) / 5.))


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


def main():
    run_model_selection_spam_l2()


if __name__ == '__main__':
    run_model_selection_spam_l2()
