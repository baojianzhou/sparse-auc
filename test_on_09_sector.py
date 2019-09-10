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
        from sparse_module import c_algo_solam_sparse
        from sparse_module import c_algo_stoht_am_sparse
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector/lib-svm-data/'


def load_dataset():
    """
    number of samples: 9,619
    number of features: 55,197 (notice: some features are all zeros.)
    :return:
    """
    # TODO there is a bug in the index.
    if os.path.exists(data_path + 'processed_sector.pkl'):
        return pkl.load(open(data_path + 'processed_sector.pkl', 'rb'))
    data = dict()
    data['x_tr'] = []
    data['y_tr'] = []
    max_id, max_nonzero = 0, 0
    words_freq = dict()
    # training part
    with open(data_path + 'sector', 'rb') as csv_file:
        raw_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in raw_reader:
            raw_row = [int(_) for _ in row if _ != '']
            data['y_tr'].append(raw_row[1])
            indices = range(2, len(raw_row), 2)
            # each feature value pair.
            data['x_tr'].append([(raw_row[i], raw_row[i + 1]) for i in indices])
            max_id = max(max([raw_row[i] for i in indices]), max_id)
            max_nonzero = max(len(data['x_tr'][-1]), max_nonzero)
            for i in indices:
                word = raw_row[i]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1

    assert len(data['y_tr']) == 6412  # total samples in train
    # testing part
    with open(data_path + 'sector.t', 'rb') as csv_file:
        raw_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in raw_reader:
            raw_row = [int(_) for _ in row if _ != '']
            data['y_tr'].append(raw_row[1])
            indices = range(0, len(raw_row[2:]), 2)
            # each feature value pair.
            data['x_tr'].append([(raw_row[i], raw_row[i + 1]) for i in indices])
            max_id = max(max([raw_row[i] for i in indices]), max_id)
            max_nonzero = max(len(data['x_tr'][-1]), max_nonzero)
            for i in indices:
                word = raw_row[i]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1

    print('maximal length is: %d' % max_nonzero)
    data['y_tr'] = np.asarray(data['y_tr'])
    assert len(data['y_tr']) == 9619  # total samples in the dataset
    data['n'] = 9619
    data['p'] = 55197
    data['max_nonzero'] = max_nonzero  # maximal number of nonzero features.
    assert (max_id + 1) == data['p']  # to make sure the number of features is p
    assert len(np.unique(data['y_tr'])) == 105  # we have total 105 classes.
    rand_ind = np.random.permutation(len(np.unique(data['y_tr'])))
    posi_classes = rand_ind[:len(np.unique(data['y_tr'])) / 2]
    nega_classes = rand_ind[len(np.unique(data['y_tr'])) / 2:]
    posi_indices = [ind for ind, _ in enumerate(data['y_tr']) if _ in posi_classes]
    nega_indices = [ind for ind, _ in enumerate(data['y_tr']) if _ in nega_classes]
    data['y_tr'][posi_indices] = 1
    data['y_tr'][nega_indices] = -1
    print('number of positive: %d' % len(posi_indices))
    print('number of negative: %d' % len(nega_indices))
    print('%d features frequency is less than 10!' %
          len([word for word in words_freq if words_freq[word] <= 10]))
    data['num_posi'] = len(posi_indices)
    data['num_nega'] = len(nega_indices)
    # randomly permute the datasets 25 times for future use.
    data['num_runs'] = 5
    data['num_k_fold'] = 5
    for run_index in range(data['num_runs']):
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        fake_x = np.zeros(shape=(data['n'], 1))  # just need the number of training samples
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['run_%d_fold_%d' % (run_index, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                'te_index': rand_perm[test_index]}
    pkl.dump(data, open(data_path + 'processed_sector.pkl', 'wb'))
    return data


def load_dataset_normalized():
    """
    number of samples: 9,619
    number of features: 55,197 (notice: some features are all zeros.)
    :return:
    """
    if os.path.exists(data_path + 'processed_sector_normalized.pkl'):
        return pkl.load(open(data_path + 'processed_sector_normalized.pkl', 'rb'))
    data = dict()
    data['x_tr'] = []
    data['y_tr'] = []
    max_id, max_nonzero = 0, 0
    words_freq = dict()
    # training part
    with open(data_path + 'sector.scale', 'rb') as f:
        for row in f.readlines():
            items = row.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            items = [(int(_.split(':')[0]) - 1, float(_.split(':')[1])) for _ in items[1:]]
            # each feature value pair.
            data['x_tr'].append(items)
            max_id = max(max([item[0] for item in items]), max_id)
            max_nonzero = max(len(data['x_tr'][-1]), max_nonzero)
            for item in items:
                word = item[0]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1

    assert len(data['y_tr']) == 6412  # total samples in train
    # testing part
    with open(data_path + 'sector.t.scale', 'rb') as f:
        for row in f.readlines():
            items = row.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            items = [(int(_.split(':')[0]) - 1, float(_.split(':')[1])) for _ in items[1:]]
            # each feature value pair.
            data['x_tr'].append(items)
            max_id = max(max([item[0] for item in items]), max_id)
            max_nonzero = max(len(data['x_tr'][-1]), max_nonzero)
            for item in items:
                word = item[0]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1

    print('maximal length is: %d' % max_nonzero)
    data['y_tr'] = np.asarray(data['y_tr'])
    assert len(data['y_tr']) == 9619  # total samples in the dataset
    data['n'] = 9619
    data['p'] = 55197
    data['max_nonzero'] = max_nonzero  # maximal number of nonzero features.
    print(max_id)
    assert (max_id + 1) == data['p']  # to make sure the number of features is p
    assert len(np.unique(data['y_tr'])) == 105  # we have total 105 classes.
    rand_ind = np.random.permutation(len(np.unique(data['y_tr'])))
    posi_classes = rand_ind[:len(np.unique(data['y_tr'])) / 2]
    nega_classes = rand_ind[len(np.unique(data['y_tr'])) / 2:]
    posi_indices = [ind for ind, _ in enumerate(data['y_tr']) if _ in posi_classes]
    nega_indices = [ind for ind, _ in enumerate(data['y_tr']) if _ in nega_classes]
    data['y_tr'][posi_indices] = 1
    data['y_tr'][nega_indices] = -1
    print('number of positive: %d' % len(posi_indices))
    print('number of negative: %d' % len(nega_indices))
    print('%d features frequency is less than 10!' %
          len([word for word in words_freq if words_freq[word] <= 10]))
    data['num_posi'] = len(posi_indices)
    data['num_nega'] = len(nega_indices)
    # randomly permute the datasets 100 times for future use.
    data['num_runs'] = 5
    data['num_k_fold'] = 5
    for run_index in range(data['num_runs']):
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        fake_x = np.zeros(shape=(data['n'], 1))  # just need the number of training samples
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['run_%d_fold_%d' % (run_index, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                'te_index': rand_perm[test_index]}
    pkl.dump(data, open(data_path + 'processed_sector_normalized.pkl', 'wb'))
    return data


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


def main():
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


if __name__ == '__main__':
    main()
