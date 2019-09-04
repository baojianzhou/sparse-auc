# -*- coding: utf-8 -*-
import os
import csv
import time
import numpy as np
import pickle as pkl
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from algo_wrapper.algo_wrapper import fpr_tpr_auc
from algo_wrapper.algo_wrapper import algo_sparse_solam
from algo_wrapper.algo_wrapper import algo_sparse_solam_cv

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector/lib-svm-data/'


def load_dataset():
    """
    number of samples: 9,619
    number of features: 55,197 (notice: some features are all zeros.)
    :return:
    """

    # if os.path.exists(data_path + 'processed_sector.pkl'):
    #    return pkl.load(open(data_path + 'processed_sector.pkl', 'rb'))
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
    print('maximal length is: %d' % max_nonzero)
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
    pkl.dump(data, open(data_path + 'processed_sector.pkl', 'wb'))
    return data


def get_run_fold_index_by_task_id(task_id):
    para_space = dict()
    index = 0
    for run_id in range(5):
        for fold_id in range(5):
            for para_xi in np.arange(1, 101, 9, dtype=float):
                for para_r in 10. ** np.arange(-1, 5, 1, dtype=float):
                    para_space[index] = (run_id, fold_id, para_xi, para_r)
                    index += 1
    return para_space[int(task_id)]


def transform_tr():


def test_single_model_selection(task_id):
    data = load_dataset()
    exit()
    para_spaces = {'global_pass': 1,
                   'global_runs': 5,
                   'global_cv': 5,
                   'data_id': 5,
                   'data_name': '09_sector',
                   'data_dim': data['p'],
                   'data_num': data['n']}
    run_id, fold_id, para_ix, para_r = get_run_fold_index_by_task_id(task_id)
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']

    # cross validate based on tr_index
    cur_auc = np.zeros(para_spaces['global_cv'])
    kf = KFold(n_splits=para_spaces['global_cv'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        sub_x_train, sub_x_test = tr_index[sub_tr_ind], tr_index[sub_te_ind]
        sub_y_train, sub_y_test = tr_index[sub_tr_ind], tr_index[sub_te_ind]
        x_tr_indices = np.zeros(shape=(len(sub_tr_ind), data['max_nonzero'] + 1), dtype=np.int32)
        x_tr_values = np.zeros(shape=(len(sub_tr_ind), data['max_nonzero'] + 1), dtype=np.int32)
        re = c_algo_solam(np.asarray(x_train, dtype=float),
                          np.asarray(y_train, dtype=float),
                          np.asarray(range(len(x_train)), dtype=np.int32),
                          float(para_r), float(para_xi), int(para_n_pass), int(verbose))
        wt = np.asarray(re[0])
        cur_auc[ind] = roc_auc_score(y_true=y_test, y_score=np.dot(x_test, wt))


def main():
    # os.environ['SLURM_ARRAY_TASK_ID']
    test_single_model_selection(task_id=1)
    pass


if __name__ == '__main__':
    main()
