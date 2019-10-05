# -*- coding: utf-8 -*-

import os
import csv
import time
import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold


def _generate_dataset_simu(data_path, num_tr, task_id, mu,
                           posi_ratio=0.1, noise_mu=0.0, noise_std=1.0):
    """
    number of classes: 2
    number of samples: 1,000
    number of features: 1,089
    ---
    @article{arias2011detection,
    title={Detection of an anomalous cluster in a network},
    author={Arias-Castro, Ery and Candes, Emmanuel J and Durand, Arnaud and others},
    journal={The Annals of Statistics},
    volume={39},
    number={1},
    pages={278--304},
    year={2011},
    publisher={Institute of Mathematical Statistics}}
    ---
    :return:
    """

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

    bench_data = {
        # figure 1 in [1], it has 26 nodes.
        'fig_1': [475, 505, 506, 507, 508, 509, 510, 511, 512, 539, 540, 541, 542,
                  543, 544, 545, 576, 609, 642, 643, 644, 645, 646, 647, 679, 712],
        # figure 2 in [1], it has 46 nodes.
        'fig_2': [439, 440, 471, 472, 473, 474, 504, 505, 506, 537, 538, 539, 568,
                  569, 570, 571, 572, 600, 601, 602, 603, 604, 605, 633, 634, 635,
                  636, 637, 666, 667, 668, 698, 699, 700, 701, 730, 731, 732, 733,
                  763, 764, 765, 796, 797, 798, 830],
        # figure 3 in [1], it has 92 nodes.
        'fig_3': [151, 183, 184, 185, 217, 218, 219, 251, 252, 285, 286, 319, 320,
                  352, 353, 385, 386, 405, 406, 407, 408, 409, 419, 420, 437, 438,
                  439, 440, 441, 442, 443, 452, 453, 470, 471, 475, 476, 485, 486,
                  502, 503, 504, 507, 508, 509, 518, 519, 535, 536, 541, 550, 551,
                  568, 569, 583, 584, 601, 602, 615, 616, 617, 635, 636, 648, 649,
                  668, 669, 670, 680, 681, 702, 703, 704, 711, 712, 713, 736, 737,
                  738, 739, 740, 741, 742, 743, 744, 745, 771, 772, 773, 774, 775,
                  776],
        # figure 4 in [1], it has 132 nodes.
        'fig_4': [244, 245, 246, 247, 248, 249, 254, 255, 256, 277, 278, 279, 280,
                  281, 282, 283, 286, 287, 288, 289, 290, 310, 311, 312, 313, 314,
                  315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 343, 344, 345,
                  346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 377,
                  378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
                  411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                  448, 449, 450, 451, 452, 453, 454, 455, 456, 481, 482, 483, 484,
                  485, 486, 487, 488, 489, 514, 515, 516, 517, 518, 519, 520, 521,
                  547, 548, 549, 550, 551, 552, 553, 579, 580, 581, 582, 583, 584,
                  585, 586, 613, 614, 615, 616, 617, 618, 646, 647, 648, 649, 650,
                  680, 681],
        # grid size (length).
        'height': 33,
        # grid width (width).
        'width': 33,
        # the dimension of grid graph is 33 x 33.
        'p': 33 * 33,
        # sparsity list of these 4 figures.
        's': {'fig_1': 26, 'fig_2': 46, 'fig_3': 92, 'fig_4': 132},
        # sparsity list
        's_list': [26, 46, 92, 132],
        'graph': __simu_grid_graph(height=33, width=33)}
    p = int(bench_data['width'] * bench_data['height'])
    posi_label, nega_label, k_fold = +1, -1, 5
    all_data = dict()
    for fig_id in ['fig_1', 'fig_2', 'fig_3', 'fig_4']:
        sub_graph = bench_data[fig_id]
        s, n = len(sub_graph), num_tr
        num_posi, num_nega = int(n * posi_ratio), int(n * (1. - posi_ratio))
        assert (num_posi + num_nega) == n
        # generate training samples and labels
        labels = [posi_label] * num_posi + [nega_label] * num_nega
        y_labels = np.asarray(labels, dtype=np.float64)
        x_data = np.random.normal(noise_mu, noise_std, n * p).reshape(n, p)
        _ = s * num_posi
        anomalous_data = np.random.normal(mu, noise_std, _).reshape(num_posi, s)
        x_data[:num_posi, sub_graph] = anomalous_data
        rand_indices = np.random.permutation(len(y_labels))
        x_tr, y_tr = x_data[rand_indices], y_labels[rand_indices]
        # normalize data by z-score
        x_mean = np.tile(np.mean(x_tr, axis=0), (len(x_tr), 1))
        x_std = np.tile(np.std(x_tr, axis=0), (len(x_tr), 1))
        x_tr = np.nan_to_num(np.divide(x_tr - x_mean, x_std))
        # normalize samples to unit length.
        for i in range(len(x_tr)):
            x_tr[i] = x_tr[i] / np.linalg.norm(x_tr[i])
        edges, weights = bench_data['graph']
        data = {'x_tr': x_tr,
                'y_tr': y_tr,
                'subgraph': sub_graph,
                'edges': edges,
                'weights': weights,
                'mu': mu,
                'p': p,
                'n': num_tr,
                'noise_mu': noise_mu,
                'noise_std': noise_std,
                'task_id': task_id,
                'num_k_fold': k_fold,
                'posi_ratio': posi_ratio}
        # randomly permute the datasets 25 times for future use.
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        fake_x = np.zeros(shape=(data['n'], 1))  # just need the number of training samples
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['task_%d_fold_%d' % (task_id, fold_index)] = {'tr_index': rand_perm[train_index],
                                                               'te_index': rand_perm[test_index]}
        all_data[fig_id] = data
    pkl.dump(all_data, open(data_path + '/data_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f.pkl'
                            % (task_id, num_tr, mu, posi_ratio), 'wb'))


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
    data['x_tr_lens'] = []
    data['x_tr_poss'] = []
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


def _load_dataset_13_realsim(data_path):
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


def generate_dataset():
    root_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
    for task_id in range(25):
        print('generate data: %02d' % task_id)
        _generate_dataset_simu(data_path=os.path.join(root_path, '00_%s' % 'simu'),
                               num_tr=1000, task_id=task_id, mu=0.3, posi_ratio=0.3)


def main(dataset):
    if dataset == 'sector':
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector'
        for run_id in range(5):
            _gen_dataset_09_sector(run_id=run_id, data_path=data_path)


if __name__ == '__main__':
    main(dataset='sector')
