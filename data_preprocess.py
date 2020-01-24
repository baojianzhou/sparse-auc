# -*- coding: utf-8 -*-

import os
import csv
import sys
import time
import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold


def __simu_grid_graph(width, height, rand_weight=False):
    """ Generate a grid graph with size, width x height. Totally there will be
            width x height number of nodes in this generated graph.
    :param width:       the width of the grid graph.
    :param height:      the height of the grid graph.
    :param rand_weight: the edge costs in this generated grid graph.
    :return:            1.  list of edges
                        2.  list of edge costs
    """
    np.random.seed()
    if width < 0 and height < 0:
        print('Error: width and height should be positive.')
        return [], []
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    # random generate costs of the graph
    if rand_weight:
        weights = []
        while len(weights) < len(edges):
            weights.append(np.random.uniform(1., 2.0))
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def _gen_dataset_00_simu(data_path, num_tr, trial_id, mu, posi_ratio, noise_mu=0.0, noise_std=1.0):
    """
    number of classes: 2
    number of samples: 1,000
    number of features: 1,000
    ---
    :param data_path:
    :param num_tr:
    :param trial_id:
    :param mu:
    :param posi_ratio:
    :param noise_mu:
    :param noise_std:
    :return:
    """
    posi_label, nega_label, k_fold, p = +1, -1, 5, 1000
    all_data = dict()
    for s in [20, 40, 60, 80]:
        perm = np.random.permutation(p)
        subset_nodes = perm[:s]
        n = num_tr
        num_posi, num_nega = int(n * posi_ratio), int(n * (1. - posi_ratio))
        assert (num_posi + num_nega) == n
        # generate training samples and labels
        labels = [posi_label] * num_posi + [nega_label] * num_nega
        y_labels = np.asarray(labels, dtype=np.float64)
        x_data = np.random.normal(noise_mu, noise_std, n * p).reshape(n, p)
        anomalous_data = np.random.normal(mu, noise_std, s * num_posi).reshape(num_posi, s)
        x_data[:num_posi, subset_nodes] = anomalous_data
        rand_indices = np.random.permutation(len(y_labels))
        x_tr, y_tr = x_data[rand_indices], y_labels[rand_indices]
        print(trial_id, posi_ratio, s, np.linalg.norm(x_tr), subset_nodes[:5])
        # normalize data by z-score
        x_mean = np.tile(np.mean(x_tr, axis=0), (len(x_tr), 1))
        x_std = np.tile(np.std(x_tr, axis=0), (len(x_tr), 1))
        x_tr = np.nan_to_num(np.divide(x_tr - x_mean, x_std))

        # normalize samples to unit length.
        for i in range(len(x_tr)):
            x_tr[i] = x_tr[i] / np.linalg.norm(x_tr[i])
        data = {'x_tr': x_tr,
                'y_tr': y_tr,
                'subset': subset_nodes,
                'mu': mu,
                'p': p,
                'n': num_tr,
                's': len(subset_nodes),
                'noise_mu': noise_mu,
                'noise_std': noise_std,
                'trial_id': trial_id,
                'num_k_fold': k_fold,
                'posi_ratio': posi_ratio}
        # randomly permute the datasets 25 times for future use.
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        fake_x = np.zeros(shape=(data['n'], 1))  # just need the number of training samples
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['trial_%d_fold_%d' % (trial_id, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                 'te_index': rand_perm[test_index]}
        all_data[s] = data
    pkl.dump(all_data, open(data_path + '/data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
                            % (trial_id, num_tr, mu, posi_ratio), 'wb'))


def _gen_dataset_01_pcmac(run_id, data_path):
    """
    number of samples: 1,946
    number of features: 7,511 (the last feature is useless)
    http://vikas.sindhwani.org/datasets/lskm/svml/
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/pcmac.svml'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            # do not need to normalize the data.
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert len(data['y_tr']) == 1946  # total samples in train
    data['n'] = 1946
    data['p'] = 7511
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    # randomly permute the datasets 25 times for future use.
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '01_realsim'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))


def _gen_dataset_02_pcmacs(run_id, data_path):
    """
    number of samples: 7,50
    number of features: 7,511 (the last feature is useless)
    http://vikas.sindhwani.org/datasets/lskm/svml/
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/pcmac.svml'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            # do not need to normalize the data.
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
            if index == 749:
                break
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert len(data['y_tr']) == 750  # total samples in train
    data['n'] = 750
    data['p'] = 7511
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    # randomly permute the datasets 25 times for future use.
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '01_realsim'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))


def _gen_dataset_02_usps(data_path):
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


def _gen_dataset_09_sector(run_id, data_path):
    """
    number of samples: 9,619
    number of features: 55,197 (notice: some features are all zeros.)
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    max_id, max_nonzero = 0, 0
    words_freq = dict()
    # training part
    with open(os.path.join(data_path, 'raw_data/sector.scale'), 'rb') as f:
        for row in f.readlines():
            items = row.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            items = [(int(_.split(':')[0]) - 1, float(_.split(':')[1])) for _ in items[1:]]
            # each feature value pair.
            data['x_tr_inds'].extend([_[0] for _ in items])
            data['x_tr_vals'].extend([_[1] for _ in items])
            data['x_tr_lens'].append(len(items))
            max_id = max(max([item[0] for item in items]), max_id)
            max_nonzero = max(len(items), max_nonzero)
            for item in items:
                word = item[0]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1
    assert len(data['y_tr']) == 6412  # total samples in train
    # testing part
    with open(data_path + '/raw_data/sector.t.scale', 'rb') as f:
        for row in f.readlines():
            items = row.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            items = [(int(_.split(':')[0]) - 1, float(_.split(':')[1])) for _ in items[1:]]
            # each feature value pair.
            data['x_tr_inds'].extend([_[0] for _ in items])
            data['x_tr_vals'].extend([_[1] for _ in items])
            data['x_tr_lens'].append(len(items))
            max_id = max(max([item[0] for item in items]), max_id)
            max_nonzero = max(len(items), max_nonzero)
            for item in items:
                word = item[0]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1
    # update positions
    prev_posi = 0
    for i in range(len(data['y_tr'])):
        data['x_tr_poss'].append(prev_posi)
        prev_posi += data['x_tr_lens'][i]
    print('maximal length is: %d' % max_nonzero)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    assert len(data['y_tr']) == 9619  # total samples in the dataset
    data['n'] = 9619
    data['p'] = 55197
    data['max_nonzero'] = max_nonzero  # maximal number of nonzero features.
    print(max_id)
    assert (max_id + 1) == data['p']  # to make sure the number of features is p
    assert len(np.unique(data['y_tr'])) == 105  # we have total 105 classes.
    rand_ind = np.random.permutation(len(np.unique(data['y_tr'])))
    print(rand_ind[:10])
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
    data['num_nonzeros'] = len(data['x_tr_vals'])
    # randomly permute the datasets 100 times for future use.
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '09_sector'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))


def _gen_dataset_10_farmads(run_id, data_path):
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/farm-ads-vect'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_vals = np.asarray([float(_.split(':')[1]) for _ in items[1:]])
            cur_vals = cur_vals / np.linalg.norm(cur_vals)
            cur_inds = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_vals)
            data['x_tr_inds'].extend(cur_inds)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_inds))
            prev_posi += len(cur_inds)
            if len(cur_inds) != 0:
                min_id = min(min(cur_inds), min_id)
                max_id = max(max(cur_inds), max_id)
                max_len = max(len(cur_inds), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = 4143
    data['p'] = 54877
    non_zero_ratio = float(len(data['x_tr_vals']) * 100) / (float(data['n']) * float(data['p']))
    print(min_id, max_id, max_len, len(data['y_tr']), '%.4f' % non_zero_ratio)
    assert data['n'] == len(data['y_tr'])
    assert data['p'] == max_id + 1
    assert 0 == min_id
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '10_farmads'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))
    return data


def _gen_dataset_12_news20(run_id, data_path):
    """
    number of samples: 19,928
    number of features: 62,061 (notice: some features are all zeros.)
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    max_id, min_id, max_nonzero = 0, np.inf, 0
    words_freq = dict()
    # training part
    with open(os.path.join(data_path, 'raw_data/news20.scale'), 'rb') as f:
        for row in f.readlines():
            items = row.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            items = [(int(_.split(':')[0]) - 1, float(_.split(':')[1])) for _ in items[1:]]
            # each feature value pair.
            data['x_tr_inds'].extend([_[0] for _ in items])
            data['x_tr_vals'].extend([_[1] for _ in items])
            data['x_tr_lens'].append(len(items))
            max_id = max(max([item[0] for item in items]), max_id)
            min_id = min(min([item[0] for item in items]), min_id)
            max_nonzero = max(len(items), max_nonzero)

            for item in items:
                word = item[0]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1
    print(min_id, max_id)
    assert len(data['y_tr']) == 15935  # total samples in train
    # testing part
    with open(data_path + '/raw_data/news20.t.scale', 'rb') as f:
        for row in f.readlines():
            items = row.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]) - 1)
            items = [(int(_.split(':')[0]) - 1, float(_.split(':')[1])) for _ in items[1:]]
            # each feature value pair.
            data['x_tr_inds'].extend([_[0] for _ in items])
            data['x_tr_vals'].extend([_[1] for _ in items])
            data['x_tr_lens'].append(len(items))
            max_id = max(max([item[0] for item in items]), max_id)
            min_id = min(min([item[0] for item in items]), min_id)
            max_nonzero = max(len(items), max_nonzero)
            for item in items:
                word = item[0]
                if word not in words_freq:
                    words_freq[word] = 0
                words_freq[word] += 1
    print(min_id, max_id)
    # update positions
    prev_posi = 0
    for i in range(len(data['y_tr'])):
        data['x_tr_poss'].append(prev_posi)
        prev_posi += data['x_tr_lens'][i]
    print('maximal length is: %d' % max_nonzero)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    assert len(data['y_tr']) == 19928  # total samples in the dataset
    data['n'] = 19928
    data['p'] = 62061
    data['max_nonzero'] = max_nonzero  # maximal number of nonzero features.
    print(max_id)
    assert (max_id + 1) == data['p']  # to make sure the number of features is p
    assert len(np.unique(data['y_tr'])) == 20  # we have total 105 classes.
    rand_ind = np.random.permutation(len(np.unique(data['y_tr'])))
    print(rand_ind[:10])
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
    data['num_nonzeros'] = len(data['x_tr_vals'])
    # randomly permute the datasets 100 times for future use.
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '12_news20'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))


def _gen_dataset_13_realsim(run_id, data_path):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/real-sim'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert len(data['y_tr']) == 72309  # total samples in train
    data['n'] = 72309
    data['p'] = 20958
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '13_realsim'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))


def _gen_dataset_14_news20b(run_id, data_path):
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/news20.binary'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)

    data['n'] = 19996
    data['p'] = 1355191
    assert data['n'] == len(data['y_tr'])
    assert data['p'] == max_id + 1
    assert 0 == min_id
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '14_news20b'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))
    return data


def _gen_dataset_15_rcv1b(run_id, data_path):
    """
    n: 20,242
    p: 47,236
    :param run_id:
    :param data_path:
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/rcv1_train.binary'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_vals = [float(_.split(':')[1]) for _ in items[1:]]
            cur_inds = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_vals)
            data['x_tr_inds'].extend(cur_inds)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_inds))
            prev_posi += len(cur_inds)
            if len(cur_inds) != 0:
                min_id = min(min(cur_inds), min_id)
                max_id = max(max(cur_inds), max_id)
                max_len = max(len(cur_inds), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = 20242
    data['p'] = 47236
    non_zero_ratio = float(len(data['x_tr_vals']) * 100) / (float(data['n']) * float(data['p']))
    print(min_id, max_id, max_len, len(data['y_tr']), '%.4f' % non_zero_ratio)
    assert data['n'] == len(data['y_tr'])
    assert data['p'] == max_id + 1
    assert 0 == min_id
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '15_rcv1b'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))
    return data


def _gen_dataset_17_gisette(run_id, data_path):
    """
    number of classes: 2
    number of samples: 7,000
    number of features: 5,000
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    np.random.seed(int(time.time()))
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(os.path.join(data_path, 'raw_data/gisette_scale'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            # do not need to normalize the data.
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
        print(min_id, max_id, max_len)
    with open(os.path.join(data_path, 'raw_data/gisette_scale.t'), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            # do not need to normalize the data.
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    assert len(data['y_tr']) == 7000  # total samples in train
    data['n'] = 7000
    data['p'] = 5000
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    # randomly permute the datasets 25 times for future use.
    data['run_id'] = run_id
    data['k_fold'] = 5
    data['name'] = '17_gisette'
    kf = KFold(n_splits=data['k_fold'], shuffle=False)
    for fold_index, (train_index, test_index) in enumerate(kf.split(range(data['n']))):
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        data['fold_%d' % fold_index] = {'tr_index': rand_perm[train_index],
                                        'te_index': rand_perm[test_index]}
    pkl.dump(data, open(os.path.join(data_path, 'data_run_%d.pkl' % run_id), 'wb'))


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


def main(dataset):
    if dataset == '00_simu':
        root_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
        for trial_id in range(20):
            for posi_ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                _gen_dataset_00_simu(data_path=os.path.join(root_path, '00_simu'),
                                     num_tr=1000, trial_id=trial_id, mu=0.3, posi_ratio=posi_ratio)
    elif dataset == '01_pcmac':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/01_pcmac'
        for run_id in range(5):
            _gen_dataset_01_pcmac(run_id=run_id, data_path=data_path)
    elif dataset == '02_pcmacs':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/02_pcmacs'
        for run_id in range(5):
            _gen_dataset_02_pcmacs(run_id=run_id, data_path=data_path)
    elif dataset == '09_sector':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector'
        for run_id in range(5):
            _gen_dataset_09_sector(run_id=run_id, data_path=data_path)
    elif dataset == '10_farmads':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/10_farmads'
        for run_id in range(5):
            _gen_dataset_10_farmads(run_id=run_id, data_path=data_path)
    elif dataset == '12_news20':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/12_news20'
        for run_id in range(5):
            _gen_dataset_12_news20(run_id=run_id, data_path=data_path)
    elif dataset == '13_realsim':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/13_realsim'
        for run_id in range(5):
            _gen_dataset_13_realsim(run_id=run_id, data_path=data_path)
    elif dataset == '14_news20b':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/14_news20b'
        for run_id in range(5):
            _gen_dataset_14_news20b(run_id=run_id, data_path=data_path)
    elif dataset == '15_rcv1b':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/15_rcv1b'
        for run_id in range(5):
            _gen_dataset_15_rcv1b(run_id=run_id, data_path=data_path)
    elif dataset == '17_gisette':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/17_gisette'
        for run_id in range(5):
            _gen_dataset_17_gisette(run_id=run_id, data_path=data_path)


if __name__ == '__main__':
    main(dataset=sys.argv[1])
