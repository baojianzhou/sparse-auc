# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import random
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
        from sparse_module import c_algo_sht_am
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'


def simu_grid_graph(width, height, rand_weight=False):
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
            weights.append(random.uniform(1., 2.0))
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
    'graph': simu_grid_graph(height=33, width=33)
}


def load_data(width, height, num_tr, noise_mu, noise_std, mu, sub_graph, task_id):
    if os.path.exists(data_path + 'processed_%02d.pkl' % task_id):
        return pkl.load(open(data_path + 'processed_%02d.pkl' % task_id, 'rb'))
    p = int(width * height)
    posi_label = +1
    nega_label = -1
    edges, weis = simu_grid_graph(height=height, width=width)  # get grid graph
    s, n = len(sub_graph), num_tr
    num_posi, num_nega = n / 2, n / 2
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
    for i in range(len(x_tr)):
        x_tr[i] = x_tr[i] / np.linalg.norm(x_tr[i])
    data = {'x_tr': x_tr[:num_tr], 'y_tr': y_tr[:num_tr], 'subgraph': sub_graph, 'edges': edges,
            'weights': weis, 'mu': mu, 'noise_mu': noise_mu, 'noise_std': noise_std,
            'task_id': task_id, 'p': p, 'num_runs': 5, 'num_k_fold': 5, 'n': num_tr}
    # randomly permute the datasets 25 times for future use.
    for run_index in range(data['num_runs']):
        kf = KFold(n_splits=data['num_k_fold'], shuffle=False)
        fake_x = np.zeros(shape=(data['n'], 1))  # just need the number of training samples
        for fold_index, (train_index, test_index) in enumerate(kf.split(fake_x)):
            # since original data is ordered, we need to shuffle it!
            rand_perm = np.random.permutation(data['n'])
            data['run_%d_fold_%d' % (run_index, fold_index)] = {'tr_index': rand_perm[train_index],
                                                                'te_index': rand_perm[test_index]}
    pkl.dump(data, open(data_path + 'processed_%02d.pkl' % task_id, 'wb'))
    return data


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


def run_single_ms_spam_l2(para):
    """
    Model selection of SPAM-l2
    :return:
    """
    run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold = para
    s_time = time.time()
    data = load_data(width=33, height=33, num_tr=1200, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'],
                     task_id=(run_id * 5 + fold_id))
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_beta': para_beta,
                   'para_l1_reg': 0.0,  # no l1 regularization needed.
                   'para_xi': para_xi,
                   'para_fold_id': fold_id,
                   'para_run_id': run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 0,  # not sparse data
                   'para_step_len': 400000,
                   'para_reg_opt': 1,
                   'data_id': 00,
                   'data_name': '00_simu'}
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
    print('number of tr: %d number of te: %d' % (len(tr_index), len(te_index)))
    # cross validate based on tr_index
    list_auc_wt = np.zeros(para_spaces['conf_k_fold'])
    list_auc_wt_bar = np.zeros(para_spaces['conf_k_fold'])
    kf = KFold(n_splits=para_spaces['conf_k_fold'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
        sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
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
        wt = np.asarray(re[0])
        wt_bar = np.asarray(re[1])
        sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
        sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
        list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
        list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
    run_time = time.time() - s_time
    print('run_id, fold_id, para_xi, para_beta: ', run_id, fold_id, para_xi, para_beta)
    print('list_auc_wt:', list_auc_wt)
    print('list_auc_wt_bar:', list_auc_wt_bar)
    print('run_time: %.4f' % run_time)
    return {'algo_para_name_list': ['run_id', 'fold_id', 'para_xi', 'para_beta',
                                    'num_passes', 'num_runs', 'k_fold'],
            'algo_para': para, 'para_spaces': para_spaces,
            'list_auc_wt': list_auc_wt,
            'list_auc_wt_bar': list_auc_wt_bar,
            'run_time': run_time}


def run_ms_spam_l2(global_passes):
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    num_runs, k_fold = 5, 5
    all_para_space = []
    for run_id, fold_id in product(range(num_runs), range(k_fold)):
        for num_passes in [global_passes]:
            for para_xi in 10. ** np.arange(-7, -2., .5, dtype=float):
                for para_beta in 10. ** np.arange(-6, 1, 1, dtype=float):
                    para_row = (run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold)
                    all_para_space.append(para_row)
    # only run sub-tasks for parallel
    num_sub_tasks = len(all_para_space) / (num_runs * k_fold)
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = all_para_space[task_start:task_end]
    list_results = []
    for task_para in list_tasks:
        result = run_single_ms_spam_l2(task_para)
        list_results.append(result)
    return list_results


def run_single_ms_sht_am(para):
    """
    Model selection of SPAM-l2
    :return:
    """
    run_id, fold_id, para_sparsity, para_xi, para_beta, num_passes, num_runs, k_fold = para
    s_time = time.time()
    data = load_data(width=33, height=33, num_tr=1000, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'],
                     task_id=(run_id * 5 + fold_id))
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_beta': para_beta,
                   'para_l1_reg': 0.0,  # no l1 regularization needed.
                   'para_sparsity': para_sparsity,
                   'para_xi': para_xi,
                   'para_fold_id': fold_id,
                   'para_run_id': run_id,
                   'para_verbose': 0,
                   'para_is_sparse': 0,
                   'para_step_len': 400000000,
                   'data_id': 0,
                   'data_name': '00_simu'}
    tr_index = data['run_%d_fold_%d' % (run_id, fold_id)]['tr_index']
    te_index = data['run_%d_fold_%d' % (run_id, fold_id)]['te_index']
    print('number of tr: %d number of te: %d' % (len(tr_index), len(te_index)))
    # cross validate based on tr_index
    list_auc_wt = np.zeros(para_spaces['conf_k_fold'])
    list_auc_wt_bar = np.zeros(para_spaces['conf_k_fold'])
    kf = KFold(n_splits=para_spaces['conf_k_fold'], shuffle=False)
    for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
        sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
        sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
        re = c_algo_sht_am(np.asarray(sub_x_tr, dtype=float),
                           np.asarray(sub_y_tr, dtype=float),
                           para_spaces['para_sparsity'],
                           para_spaces['para_xi'],
                           para_spaces['para_beta'],
                           para_spaces['para_num_passes'],
                           para_spaces['para_step_len'],
                           para_spaces['para_is_sparse'],
                           para_spaces['para_verbose'])
        wt = np.asarray(re[0])
        wt_bar = np.asarray(re[1])
        sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
        sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
        list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
        list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
    run_time = time.time() - s_time
    print('run_id, fold_id, para_xi, para_beta: ', run_id, fold_id, para_xi, para_beta)
    print('list_auc_wt:', list_auc_wt)
    print('list_auc_wt_bar:', list_auc_wt_bar)
    print('run time: %.4f' % run_time)
    return {'algo_para_name_list': ['run_id', 'fold_id', 'para_xi', 'para_beta',
                                    'num_passes', 'num_runs', 'k_fold'],
            'algo_para': para, 'para_spaces': para_spaces,
            'list_auc_wt': list_auc_wt,
            'list_auc_wt_bar': list_auc_wt_bar,
            'run_time': run_time}


def run_ms_sht_am(global_passes, global_sparsity):
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    num_runs, k_fold = 5, 5
    all_para_space = []
    list_sparsity = [global_sparsity]
    list_xi = 10. ** np.arange(-7, -2., .5, dtype=float)
    list_beta = 10. ** np.arange(-6, 1, 1, dtype=float)
    for run_id, fold_id in product(range(num_runs), range(k_fold)):
        for num_passes in [global_passes]:
            for para_sparsity, para_xi, para_beta in product(list_sparsity, list_xi, list_beta):
                para_row = (run_id, fold_id, para_sparsity, para_xi, para_beta, num_passes,
                            num_runs, k_fold)
                all_para_space.append(para_row)
    # only run sub-tasks for parallel
    num_sub_tasks = len(all_para_space) / (num_runs * k_fold)
    task_start = int(task_id) * num_sub_tasks
    task_end = int(task_id) * num_sub_tasks + num_sub_tasks
    list_tasks = all_para_space[task_start:task_end]
    list_results = []
    for task_para in list_tasks:
        result = run_single_ms_sht_am(task_para)
        list_results.append(result)
    return list_results


def run_spam_l2_by_sm(model='wt', num_passes=1):
    """
    25 tasks to finish
    :return:
    """
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1

    all_results, num_runs, k_fold = [], 5, 5
    for ind in range(num_runs * k_fold):
        f_name = data_path + 'ms_spam_l2_task_%02d_passes_%03d.pkl' % (ind, num_passes)
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
    data = load_data(width=33, height=33, num_tr=1000, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'], task_id=task_id)
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
    pkl.dump(re, open(data_path + 're_spam_l2_%02d_passes_%03d.pkl' % (task_id, num_passes), 'wb'))


def run_sht_am_by_sm(model='wt', num_passes=1, global_sparsity=100):
    """
    25 tasks to finish
    :return:
    """
    s_time = time.time()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 1
    all_results, num_runs, k_fold = [], 5, 5
    for ind in range(num_runs * k_fold):
        f_name = data_path + 'ms_sht_am_l2_task_%02d_passes_%03d_sparsity_%04d.pkl' % \
                 (ind, num_passes, global_sparsity)
        all_results.extend(pkl.load(open(f_name, 'rb')))
    # selected model
    selected_model = dict()
    for result in all_results:
        run_id, fold_id, para_sparsity, para_xi, para_beta, num_passes, num_runs, k_fold = result[
            'algo_para']
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
    data = load_data(width=33, height=33, num_tr=1000, noise_mu=0.0,
                     noise_std=1.0, mu=0.3, sub_graph=bench_data['fig_1'], task_id=task_id)
    para_spaces = {'conf_num_runs': num_runs,
                   'conf_k_fold': k_fold,
                   'para_num_passes': num_passes,
                   'para_sparsity': global_sparsity,
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
    re = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float),
                       np.asarray(data['y_tr'][tr_index], dtype=float),
                       para_spaces['para_sparsity'],
                       para_spaces['para_xi'],
                       para_spaces['para_beta'],
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
    pkl.dump(re, open(data_path + 're_sht_am_%02d_passes_%03d_sparsity_%04d.pkl' %
                      (task_id, num_passes, global_sparsity), 'wb'))


def run_test_result():
    for num_passes in [1, 5]:
        run_sht_am_by_sm(num_passes=num_passes, global_sparsity=100)
        run_spam_l2_by_sm(num_passes=num_passes)


def run_model_selection():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        task_id = 0
    results_spam_l2 = dict()
    results_sht_am = dict()
    for num_passes in [1, 5, 10, 15, 20, 25, 30]:
        s_time = time.time()
        results_spam_l2[num_passes] = run_ms_spam_l2(num_passes)
        results_sht_am[num_passes] = dict()
        for sparsity in [50, 100, 150, 200, 250, 300]:
            re = run_ms_sht_am(num_passes, global_sparsity=sparsity)
            results_sht_am[num_passes][sparsity] = re
        print(time.time() - s_time)
    file_name = 'ms_task_%02d.pkl' % task_id
    pkl.dump({'spam_l2': results_spam_l2,
              'sht_am': results_sht_am},
             open(os.path.join(data_path, file_name), 'wb'))


def main():
    run_model_selection()


if __name__ == '__main__':
    main()
