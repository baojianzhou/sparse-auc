# -*- coding: utf-8 -*-
__all__ = ['load_dataset']
import os
import csv
import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold


def data_process_13_ad():
    """
    ---
    Introduction

    The task is to predict whether an image is an advertisement ("ad") or not ("nonad"). There are
    3 continuous attributes(1st:height,2nd:width,3rd:as-ratio)-28% data is missing for each
    continuous attribute. Clearly, the third attribute does not make any contribution since it is
    the 2nd:width./1st:height.

    Number of positive labels: 459
    Number of negative labels: 2820

    ---
    Data Processing
    Need to fill the missing values. The missing values only happened in the first three features.


    ---
    BibTeX:
    @inproceedings{kushmerick1999learning,
        title={Learning to remove internet advertisements},
        author={KUSHMERICK, N},
        booktitle={the third annual conference on Autonomous Agents, May 1999},
        pages={175--181},
        year={1999}
        }

    ---
    Data Format:
    each feature is either np.nan ( missing values) or a real value.
    label: +1: positive label(ad.) -1: negative label(nonad.)
    :return:
    """
    path = '/home/baojian/Desktop/ad-dataset/ad.data'
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        raw_data = [row for row in reader]
        data_x = []
        data_y = []
        for row in raw_data:
            values = []
            for ind, _ in enumerate(row[:-1]):
                if str(_).find('?') == -1:  # handle missing values.
                    values.append(float(_))
                else:
                    values.append(np.nan)
            data_x.append(values)
            if row[-1] == 'ad.':
                data_y.append(+1)
            else:
                data_y.append(-1)
        print('number of positive labels: %d' % len([_ for _ in data_y if _ > 0]))
        print('number of negative labels: %d' % len([_ for _ in data_y if _ < 0]))
        pkl.dump({'data_x': np.asarray(data_x),
                  'data_y': np.asarray(data_y)}, open('processed-13-ad.pkl', 'wb'))


def _load_dataset_realsim(data_path):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    if os.path.exists(os.path.join(data_path, 'processed_realsim.pkl')):
        return pkl.load(open(os.path.join(data_path, 'processed_realsim.pkl'), 'rb'))
    data = dict()
    # sparse data to make it linear
    data['x_values'] = []
    data['x_indices'] = []
    data['x_positions'] = []
    data['x_len_list'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, non_zeros, max_len = 0, np.inf, 0, 0, 0
    with open(os.path.join(data_path, 'real-sim')) as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            # do not need to normalize the data.
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_values'].extend(cur_values)
            data['x_indices'].extend(cur_indices)
            data['x_positions'].append(prev_posi)
            data['x_len_list'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
            non_zeros += len(cur_indices)
        print(min_id, max_id, max_len, non_zeros)
    data['x_values'] = np.asarray(data['x_values'], dtype=float)
    data['x_indices'] = np.asarray(data['x_indices'], dtype=np.int32)
    data['x_positions'] = np.asarray(data['x_positions'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert len(data['y_tr']) == 72309  # total samples in train
    data['n'] = 72309
    data['p'] = 20958
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
        # just need the number of training samples
        fake_x = np.zeros(shape=(data['n'], 1))
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['run_%d_fold_%d' % (run_index, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'processed_realsim.pkl'), 'wb'))
    return data


def load_dataset(root_path, name=None):
    if name is None:
        print('Unknown dataset name')
    if name == 'realsim':
        _load_dataset_realsim(data_path=os.path.join(root_path, '13_%s' % name))
