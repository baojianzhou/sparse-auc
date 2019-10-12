# -*- coding: utf-8 -*-
import time
import scipy.io
import numpy as np
import sklearn as sk
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from itertools import product
import os
import pickle as pkl


def show_ht_run_time():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    plt.rcParams.update({'font.size': 12})
    names = ['QuickSelect-V1', 'QuickSelect-V2', 'MaxHeap-V1', 'Floyd-Rivest-V1', 'QuickSelect-V3',
             'Wirth', 'QuickSelect-V4', 'Floyd-Rivest-V2', 'QuickSort']
    colors = ['b', 'b', 'm', 'green', 'b', 'y', 'b', 'green', 'r']
    alpha_list = [1., .8, 1., 1., .6, 1., .4, .5, 1.]
    run_time = {1: np.asarray(
        [621, 416, 977.0, 164, 403, 440, 549, 214, 9716]),
        10: np.asarray([5023, 4497, 9906.0, 1279, 3939, 3641, 3854, 1895, 95396]),
        100: np.asarray([46165, 38435, 98872.0, 12727, 38959, 36181, 38824, 19763, 956011]),
        1000: np.asarray(
            [462480, 387161, 990644.0, 125644, 383959, 356307, 390857, 196206, 9606209]),
        2000: np.asarray(
            [973674, 799438, 2076245.0, 266914, 813301, 745576, 804968, 410895, 20250160]),
        5000: np.asarray(
            [2496992, 2028612, 5242117.0, 684493, 2058850, 1871607, 2055310, 1049421, 51434902]),
        8000: np.asarray(
            [3862229, 3189205, 8228015.0, 1058340, 3214370, 2938456, 3207977, 1640777, 80098740]),
        10000: np.asarray([4662248, 3909561, 10003924.0, 1271100, 3909567, 3583082, 3919146,
                           1989976, 97095163]),
        100000: np.asarray(
            [48050835, 39885553, 102413292.0, 13181102, 40028054, 36642196, 39946229, 20375441,
             999201785])}
    x = [1, 10, 100, 1000, 2000, 5000, 8000, 10000]
    fig, ax = plt.subplots(1, 2)
    for ind, name in enumerate(names):
        vals = np.asarray([run_time[_][ind] for _ in x],
                          dtype=float) / 1e6
        ax[0].plot(x, vals, linewidth=2., label=name, color=colors[ind], alpha=alpha_list[ind],
                   marker='.')
        ax[0].set_xlabel('iterations')
        ax[0].set_ylabel('run time (seconds)')
        ax[0].set_title('Total run time w.r.t iterations')
        ax[0].grid(True)
        ax[0].legend()
    for ind, name in enumerate(names):
        if name == 'QuickSort':
            continue
        vals = np.asarray([run_time[_][ind] for _ in x],
                          dtype=float) / 1e6
        ax[1].plot(x, vals, linewidth=2., label=name, color=colors[ind], alpha=alpha_list[ind],
                   marker='.')
        ax[1].set_xlabel('iterations')
        ax[1].set_ylabel('run time (seconds)')
        ax[1].set_title('Total run time w.r.t iterations')
        ax[1].grid(True)
        ax[1].legend()
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    plt.savefig('/home/baojian/ht_run_time.png', dpi=400, bbox_inches='tight', pad_inches=0,
                format='png')
    plt.show()


def load_results():
    import scipy.io
    results = scipy.io.loadmat('baselines/nips16_solam/EP_a9a_SOLAM.mat')['data']
    re = {'auc': np.asarray(results['AUC'])[0][0],
          'mean_auc': np.asarray(results['meanAUC'])[0][0][0][0],
          'std_auc': np.asarray(results['stdAUC'])[0][0][0][0],
          'run_time': np.asarray(results['RT'])[0][0],
          'wt': np.asarray(results['wt'])[0][0],
          'kfold_ind': np.asarray(results['vIndices'])[0][0]}
    k_fold_ind = re['kfold_ind']
    for i in range(len(k_fold_ind)):
        for j in range(len(k_fold_ind[i])):
            k_fold_ind[i][j] -= 1
    re['kfold_ind'] = np.asarray(k_fold_ind, dtype=int)
    return re


def app_compare_solam():
    """
    To run the MATLAB version of the SOLAM algorithm. Then we compare the results obtained
    from MATLAB's version and C's version.
    """
    results = load_results()
    g_pass = 3
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561, 'g_seq_num': 1}
    opt_solam = {'sc': 100, 'sr': 10, 'n_pass': g_pass}
    res_solam = {'AUC': np.zeros(shape=(g_iters, g_cv)),
                 'mean_AUC': 0.0,
                 'std_AUC': 0.0,
                 'run_time': np.zeros(shape=(g_iters, g_cv)),
                 'mean_run_time': 0.0}

    mat = scipy.io.loadmat('baselines/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    u_lab = np.unique(org_label)
    u_num = len(u_lab)
    if u_num > 2:
        u_sort = np.random.permutation(u_num)

    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim']))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    for m in range(g_iters):
        kfold_ind = results['kfold_ind'][m]
        kf_split = []
        for i in range(g_cv):
            train_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] != i]
            test_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] == i]
            kf_split.append((train_ind, test_ind))
        for j, (train_index, test_index) in enumerate(kf_split):
            x_tr, x_te = pp_feat[train_index], pp_feat[test_index]
            y_tr, y_te = pp_label[train_index], pp_label[test_index]
            wt_ = []
            for _ in list(results['wt'][m][j]):
                wt_.append(_[0])
            wt_ = np.asarray(wt_)
            wt, a, b, run_time = algo_solam(x_tr=x_tr, y_tr=y_tr, para_rand_ind=range(len(x_tr)),
                                            para_r=opt_solam['sr'], para_xi=opt_solam['sc'],
                                            para_n_pass=opt_solam['n_pass'], verbose=0)
            v_fpr, v_tpr, n_auc = fpr_tpr_auc(x_te=x_te, y_te=y_te, wt=wt)
            print('run time: (%.6f, %.6f) ' % (run_time, results['run_time'][m][j])),
            print('auc: (%.6f, %.6f) ' % (n_auc, results['auc'][m][j])),
            print('norm(wt-wt_): %.6f' % (np.linalg.norm(wt - wt_))),
            print('speed up: %.2f' % (results['run_time'][m][j] / run_time))


def app_sparse_cv():
    results = load_results()
    g_pass = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    import scipy.io
    mat = scipy.io.loadmat('baselines/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim'] + 1000))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        fake_features = np.zeros(1000)
        nonzeros_ind = np.where(np.random.rand(1000) <= 0.05)
        fake_features[nonzeros_ind] = 1.0
        t_dat = np.concatenate((t_dat, fake_features))
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    for m in range(g_iters):
        kfold_ind = results['kfold_ind'][m]
        kf_split = []
        for i in range(g_cv):
            train_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] != i]
            test_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] == i]
            kf_split.append((train_ind, test_ind))
        for j, (train_index, test_index) in enumerate(kf_split):
            x_tr, x_te = pp_feat[train_index], pp_feat[test_index]
            y_tr, y_te = pp_label[train_index], pp_label[test_index]
            wt_ = []
            for _ in list(results['wt'][m][j]):
                wt_.append(_[0])
            wt_ = np.asarray(wt_)
            s_t = time.time()
            opt_solam = algo_sparse_solam_cv(
                x_tr=x_tr, y_tr=y_tr, para_s=500, para_n_pass=g_pass, para_n_cv=5, verbose=0)
            print('run time for model selection: %.4f' % (time.time() - s_t))
            wt, a, b, run_time = algo_sparse_solam(x_tr=x_tr, y_tr=y_tr,
                                                   para_rand_ind=range(len(x_tr)),
                                                   para_r=opt_solam['sr'],
                                                   para_xi=opt_solam['sc'],
                                                   para_s=opt_solam['s'],
                                                   para_n_pass=opt_solam['n_pass'], verbose=0)
            v_fpr, v_tpr, n_auc = fpr_tpr_auc(x_te=x_te, y_te=y_te, wt=wt)
            print('run time: (%.6f, %.6f) ' % (run_time, results['run_time'][m][j])),
            print('auc: (%.6f, %.6f) ' % (n_auc, results['auc'][m][j])),
            print('norm(wt-wt_): %.6f' % (np.linalg.norm(wt[:123] - wt_))),
            print('speed up: %.2f' % (results['run_time'][m][j] / run_time))


def demo_cv():
    results = load_results()
    g_pass = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    import scipy.io
    mat = scipy.io.loadmat('baselines/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim'] + 1000))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        fake_features = np.zeros(1000)
        nonzeros_ind = np.where(np.random.rand(1000) <= 0.05)
        fake_features[nonzeros_ind] = 1.0
        t_dat = np.concatenate((t_dat, fake_features))
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    for m in range(g_iters):
        kfold_ind = results['kfold_ind'][m]
        kf_split = []
        for i in range(g_cv):
            train_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] != i]
            test_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] == i]
            kf_split.append((train_ind, test_ind))
        for j, (train_index, test_index) in enumerate(kf_split):
            x_tr, x_te = pp_feat[train_index], pp_feat[test_index]
            y_tr, y_te = pp_label[train_index], pp_label[test_index]
            rand_id = np.asarray(range(len(x_tr)), dtype=np.int32)
            wt_ = []
            for _ in list(results['wt'][m][j]):
                wt_.append(_[0])
            wt_ = np.asarray(wt_)
            s_t = time.time()
            opt_solam = algo_solam_cv(x_tr, y_tr, para_n_pass=g_pass, para_n_cv=5, verbose=0)

            run_time, n_auc, n_auc_2, wt = algo_solam(x_tr, y_tr, x_te, y_te, opt_solam, rand_id)
            # print('run time: (%.6f, %.6f) ' % (run_time, results['run_time'][m][j])),
            print('%.6f, %.6f ' % (n_auc, results['auc'][m][j])),
            # print('norm(wt-wt_): %.6f' % (np.linalg.norm(wt[:123] - wt_))),
            # print('speed up: %.2f' % (results['run_time'][m][j] / run_time))


def demo_da_cv():
    results = load_results()
    g_pass = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    import scipy.io
    mat = scipy.io.loadmat('baselines/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim'] + 1000))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        fake_features = np.zeros(1000)
        nonzeros_ind = np.where(np.random.rand(1000) <= 0.05)
        fake_features[nonzeros_ind] = 1.0
        t_dat = np.concatenate((t_dat, fake_features))
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    for m in range(g_iters):
        kfold_ind = results['kfold_ind'][m]
        kf_split = []
        for i in range(g_cv):
            train_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] != i]
            test_ind = [_ for _ in range(len(pp_feat)) if kfold_ind[_] == i]
            kf_split.append((train_ind, test_ind))
        for j, (train_index, test_index) in enumerate(kf_split):
            x_tr, x_te = pp_feat[train_index], pp_feat[test_index]
            y_tr, y_te = pp_label[train_index], pp_label[test_index]
            rand_id = np.asarray(range(len(x_tr)), dtype=np.int32)
            wt_ = []
            for _ in list(results['wt'][m][j]):
                wt_.append(_[0])
            wt_ = np.asarray(wt_)
            s_t = time.time()
            opt_solam = algo_da_solam_cv(x_tr=x_tr, y_tr=y_tr, para_s=123,
                                         para_n_pass=g_pass, para_n_cv=5, verbose=0)
            print('run time for model selection: %.4f' % (time.time() - s_t))
            run_time, n_auc, n_auc_2, wt = fn_ep_da_solam(x_tr, y_tr, x_te, y_te, opt_solam,
                                                          rand_id)
            print('run time: (%.6f, %.6f) ' % (run_time, results['run_time'][m][j])),
            print('auc: (%.6f, %.6f) ' % (n_auc, results['auc'][m][j])),
            print('norm(wt-wt_): %.6f' % (np.linalg.norm(wt[:123] - wt_))),
            print('speed up: %.2f' % (results['run_time'][m][j] / run_time))


def fn_ep_da_solam(x_train, y_train, x_test, y_test, options, rand_id):
    """
    SOLAM: Stochastic Online AUC Maximization
    :param x_train: the training instances
    :param y_train: the vector of labels for x_train
    :param x_test: the testing instances
    :param y_test: the vector of labels for X_test
    :param options: a struct containing rho, sigma, C, n_label and n_tick
    :param rand_id:
    :return:
    """
    t_start = time.time()
    wt_c, a, b = algo_da_solam(x_tr=np.asarray(x_train, dtype=float),
                               y_tr=np.asarray(y_train, dtype=float),
                               para_rand_ind=np.asarray(rand_id, dtype=np.int32),
                               para_r=options['sr'], para_xi=options['sc'], para_s=123,
                               para_n_pass=options['n_pass'], verbose=0)
    run_time_c = time.time() - t_start
    v_fpr, v_tpr, n_auc, n_auc_2 = fpr_tpr_auc(x_test, y_test, wt_c)
    return run_time_c, n_auc, n_auc_2, wt_c


def evaluate(x_te, y_te, wt):
    v_py = np.dot(x_te, wt)
    v_fpr, v_tpr, _ = roc_curve(y_true=y_te, y_score=v_py)
    n_auc = roc_auc_score(y_true=y_te, y_score=v_py)
    n_auc_2 = auc(v_fpr, v_tpr)
    return v_fpr, v_tpr, n_auc, n_auc_2


def test():
    results = []
    with open('results.txt', 'rb') as f:
        for each_line in f.readlines():
            if len(each_line.lstrip().rstrip()) != 0:
                results.append([float(_) for _ in each_line.split(', ')])
    with open('results_2.txt', 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            if len(each_line.lstrip().rstrip()) != 0:
                results[index].append(float(each_line.split(', ')[0]))
    print(results)
    re = [0.0, 0.0, 0.0]
    for i in range(10):
        print('%.4f & %.4f & %.4f & - \\\\' % (results[i][1], results[i][2], results[i][0]))
        re[0] += results[i][1]
        re[1] += results[i][2]
        re[2] += results[i][0]
    print(re[0] / 10., re[1] / 10., re[2] / 10.)


def result_summary(num_passes):
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/'
    num_runs, k_fold = 5, 5
    all_para_space = []
    for run_id, fold_id in product(range(num_runs), range(k_fold)):
        for para_xi in 10. ** np.arange(-7, -2., 0.5, dtype=float):
            for para_beta in 10. ** np.arange(-6, 1, 1, dtype=float):
                para_row = (run_id, fold_id, para_xi, para_beta, num_passes, num_runs, k_fold)
                all_para_space.append(para_row)
    # only run sub-tasks for parallel
    num_sub_tasks = len(all_para_space) / 100
    all_results = []
    for i in range(100):
        task_start, task_end = int(i) * num_sub_tasks, int(i) * num_sub_tasks + num_sub_tasks
        f_name = data_path + 'ms_spam_l2_%04d_%04d_%04d.pkl' % (task_start, task_end, num_passes)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)
    pkl.dump(all_results, open(data_path + 'ms_spam_l2_passes_%04d.pkl' % num_passes, 'wb'))


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


def result_summary_00_simu():
    import matplotlib.pyplot as plt
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    num_runs, k_fold = 5, 5
    passes_list = [1, 5, 10, 15, 20]
    y, yerr = [], []
    for num_passes in passes_list:
        auc_wt, auc_wt_bar = [], []
        for ind in range(num_runs * k_fold):
            f_name = data_path + 're_task_%02d.pkl' % ind
            auc_wt.append(pkl.load(open(f_name, 'rb'))['spam_l2'][num_passes]['auc_wt'])
            auc_wt_bar.append(pkl.load(open(f_name, 'rb'))['spam_l2'][num_passes]['auc_wt_bar'])
        print(np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar))
        y.append(np.mean(auc_wt))
        yerr.append(np.std(auc_wt))

    print('test')
    plt.errorbar(x=passes_list, y=y, yerr=yerr, c='blue', label='SPAM-L2')
    plt.ylim([0.5, 1.])

    y, yerr = [], []
    for num_passes in passes_list:
        auc_wt, auc_wt_bar = [], []
        for ind in range(num_runs * k_fold):
            f_name = data_path + 're_task_%02d_elastic_net.pkl' % ind
            auc_wt.append(pkl.load(open(f_name, 'rb'))['spam_elastic_net'][num_passes]['auc_wt'])
            auc_wt_bar.append(
                pkl.load(open(f_name, 'rb'))['spam_elastic_net'][num_passes]['auc_wt_bar'])
        print(np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar))
        y.append(np.mean(auc_wt))
        yerr.append(np.std(auc_wt))

    print('test')
    plt.errorbar(x=passes_list, y=y, yerr=yerr, linestyle='--', c='green', label='SPAM-L1/L2')
    plt.ylim([0.5, 1.])

    for sparsity in [100, 200, 400, 800]:
        y, yerr = [], []
        for num_passes in [1, 5, 10, 15, 20]:
            auc_wt, auc_wt_bar = [], []
            for ind in range(num_runs * k_fold):
                f_name = data_path + 're_task_%02d.pkl' % ind
                auc_wt.append(
                    pkl.load(open(f_name, 'rb'))['sht_am'][num_passes][sparsity]['auc_wt'])
                auc_wt_bar.append(
                    pkl.load(open(f_name, 'rb'))['sht_am'][num_passes][sparsity]['auc_wt_bar'])
            print(np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar))
            y.append(np.mean(auc_wt))
            yerr.append(np.std(auc_wt))
        plt.errorbar(x=passes_list, y=y, yerr=yerr, label='HT(k=%03d)' % sparsity)
        print('\n')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel('num_passes')
    plt.show()


def result_summary_13_realsim():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/13_realsim/'
    num_runs, k_fold = 5, 5
    auc_wt, auc_wt_bar = [], []
    for ind in range(num_runs * k_fold):
        for num_passes in [5]:
            f_name = data_path + 're_spam_l2_%02d_%04d.pkl' % (ind, num_passes)
            re = pkl.load(open(f_name, 'rb'))
            print(re['auc_wt'])
            auc_wt.append(re['auc_wt'])
            auc_wt_bar.append(re['auc_wt_bar'])
    print('%.4f %.4f %.4f %.4f ' % (
        np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar)))
    global_sparsity = 6000
    auc_wt, auc_wt_bar = [], []
    for ind in range(num_runs * k_fold):
        for num_passes in [5]:
            f_name = data_path + 're_sht_am_%02d_%04d_sparsity_%04d.pkl' % \
                     (ind, num_passes, global_sparsity)
            re = pkl.load(open(f_name, 'rb'))
            print(re['auc_wt'])
            auc_wt.append(re['auc_wt'])
            auc_wt_bar.append(re['auc_wt_bar'])
    print('%.4f %.4f %.4f %.4f ' % (
        np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar)))


def result_summary_00_simu_ms():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    k_fold, num_passes = 5, 10
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.3, 0.5]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    models = dict()
    for task_id, num_tr, mu, posi_ratio, fig_i in product(
            range(25), tr_list, mu_list, posi_ratio_list, fig_list):
        f_name = os.path.join(data_path, 'ms_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f_%s.pkl' %
                              (task_id, num_tr, mu, posi_ratio, fig_i))
        re = pkl.load(open(f_name, 'rb'))[task_id]
        for key in re:
            (num_tr, mu, posi_ratio, fig_i, num_passes) = key
            models[(task_id, num_tr, mu, posi_ratio, fig_i, num_passes)] = re[key]
    pkl.dump(models, open(data_path + 'models.pkl', 'wb'))


def result_summary_00_simu_re():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    models = dict()
    for task_id in product(range(25)):
        f_name = os.path.join(data_path, 'results_task_%02d.pkl' % task_id)
        re = pkl.load(open(f_name, 'rb'))
        for key in re:
            models[key] = re[key]
    results = dict()
    for num_tr, mu, p_ratio, fig_i, num_passes in product([1000], [0.3], [0.3],
                                                          ['fig_1', 'fig_2', 'fig_3', 'fig_4'],
                                                          [10]):
        print('-' * 50 + '\nposi_ratio:%.1f fig_i: %s' % (p_ratio, fig_i))
        print('\t\t\t'.join(['spam_l2', 'spam_l1l2', 'sht_am', 'graph_am', 'solam', 'opauc']))
        for model in ['auc_wt', 'nonzero_wt']:
            for method in ['spam_l2', 'spam_l1l2', 'sht_am', 'graph_am', 'solam', 'opauc']:
                re1 = np.mean(
                    [models[(i, j, num_tr, mu, p_ratio, fig_i, num_passes)][method][model]
                     for i, j in product(range(25), range(5))])
                re2 = np.std([models[(i, j, num_tr, mu, p_ratio, fig_i, num_passes)][method][model]
                              for i, j in product(range(25), range(5))])
                print('%.4f$\pm$%.4f &' % (re1, re2)),
            print('')


def show_test():
    import matplotlib.pyplot as plt
    from pylab import rcParams
    color_list = ['b', 'g', 'm', 'r', 'y', 'k', 'orange', 'Aqua']
    marker_list = ['X', 'o', 'P', 's', '*', '<', '>', 'v']
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    models = dict()
    for task_id in product(range(25)):
        f_name = os.path.join(data_path, 'results_task_%02d.pkl' % task_id)
        re = pkl.load(open(f_name, 'rb'))
        for key in re:
            models[key] = re[key]
    rcParams['figure.figsize'] = 16, 4

    fig, ax = plt.subplots(1, 4)
    for ii in range(4):
        ax[ii].grid(b=True, which='both', color='lightgray',
                    linestyle='dotted', axis='both')
    for ind, simu in enumerate(['Simu1', 'Simu2', 'Simu3', 'Simu4']):
        ax[ind].set_title(simu)
    results = dict()
    method_list = ['opauc', 'solam', 'fsauc', 'spam_l2', 'spam_l1', 'spam_l1l2', 'sht_am',
                   'graph_am']
    for fig_i in ['fig_1', 'fig_2', 'fig_3', 'fig_4']:
        results[fig_i] = dict()
        for method in method_list:
            results[fig_i][method] = np.zeros(5)
            for ind, p_ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
                for num_tr, mu, num_passes in product([1000], [0.3], [10]):
                    for model in ['auc_wt']:
                        re1 = np.mean(
                            [models[(i, j, num_passes, num_tr, mu, p_ratio, fig_i)][method][model]
                             for i, j in product(range(25), range(5))])
                        results[fig_i][method][ind] = re1
        for item in results[fig_i]:
            print(item, results[fig_i][item])
        print('-' * 50)

    title_list = ['OPAUC', 'SOLAM', 'FSAUC', 'SPAM-L2', 'SPAM-L1',
                  'spam-L1L2', 'SHT-AM', 'GRAPH-AM']
    for fig_ind, fig_i in enumerate(['fig_1', 'fig_2', 'fig_3', 'fig_4']):
        for method_ind, method in enumerate(method_list):
            ax[fig_ind].plot([0.1, 0.2, 0.3, 0.4, 0.5],
                             results[fig_i][method],
                             c=color_list[method_ind], linestyle='-',
                             markerfacecolor='none',
                             marker=marker_list[method_ind], markersize=8.,
                             markeredgewidth=1.2, linewidth=1.2,
                             label=title_list[method_ind])
    ax[0].set_ylabel(r'AUC score')
    for i in range(4):
        ax[i].set_xlabel(r'Imbalanced Ratio')
        ax[i].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
    ax[3].legend(loc='lower right', fontsize=15., borderpad=0.1,
                 labelspacing=0.2, handletextpad=0.1)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[2].get_yticklabels(), visible=False)
    plt.setp(ax[3].get_yticklabels(), visible=False)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = '/results_exp_simu.png'
    print('save fig to: %s' % f_name)
    plt.savefig(data_path + f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()


def combine_results():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    p_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    method_list = ['opauc', 'solam', 'fsauc', 'spam_l2', 'spam_l1', 'spam_l1l2', 'sht_am',
                   'graph_am']
    results = dict()
    for task_id in product(range(25)):
        re = pkl.load(open(os.path.join(data_path, 'results_task_%02d.pkl' % task_id), 'rb'))
        for key in re:
            results[key] = re[key]
    print('          opauc   solam    fsauc  spam_l2 spam_l1 spam_l1l2 sht_am graph_am')
    for p_ratio, fig_i in product(p_ratio_list, fig_list):
        re = {_: [] for _ in method_list}
        for key in results:
            if key[5] == p_ratio and key[6] == fig_i:
                for method in results[key]:
                    re[method].append(results[key][method]['auc_wt'])
        for method in method_list:
            re[method] = float(np.mean(re[method]))
        print('%.1f %s ' % (p_ratio, fig_i) + '  '.join(
            ['%.4f' % re[method] for method in method_list]))

    for p_ratio, fig_i in product(p_ratio_list, fig_list):
        re = {_: [] for _ in method_list}
        for key in results:
            if key[5] == p_ratio and key[6] == fig_i:
                for method in results[key]:
                    re[method].append(results[key][method]['auc_wt'])
        for method in method_list:
            re[method] = float(np.mean(re[method]))
        print(' & '.join(['%.4f' % re[method] for method in method_list]))

    for p_ratio, fig_i in product(p_ratio_list, fig_list):
        re = {_: [] for _ in method_list}
        for key in results:
            if key[5] == p_ratio and key[6] == fig_i:
                for method in results[key]:
                    re[method].append(results[key][method]['nonzero_wt'])
        for method in method_list:
            re[method] = float(np.mean(re[method]))
        print(' & '.join(['%.4f' % re[method] for method in method_list]))


def show_blocksize():
    import matplotlib.pyplot as plt
    from pylab import rcParams
    color_list = ['b', 'g', 'm', 'r', 'y', 'k', 'orange', 'Aqua']
    marker_list = ['X', 'o', 'P', 's', '*', '<', '>', 'v']
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    models = dict()
    for task_id in product(range(25)):
        f_name = os.path.join(data_path, 'results_task_%02d_blocksize.pkl' % task_id)
        re = pkl.load(open(f_name, 'rb'))
        for key in re:
            models[key] = re[key]
    rcParams['figure.figsize'] = 8, 4

    fig, ax = plt.subplots(1, 2)
    for ii in range(2):
        ax[ii].grid(b=True, which='both', color='lightgray',
                    linestyle='dotted', axis='both')
    for ind, simu in enumerate(['p-ratio=0.1', 'p-ratio=0.5']):
        ax[ind].set_title(simu)
    results = dict()
    method_list = ['sht_am', 'graph_am']
    for fig_i in ['fig_2']:
        results[fig_i] = dict()
        for method in method_list:
            results[fig_i][method] = dict()
            for p_ratio in [0.1, 0.5]:
                for num_tr, mu, num_passes in product([1000], [0.3], [10]):
                    print(models[(0, 0, num_passes, num_tr, mu, p_ratio, fig_i)][method])
                    re1 = np.mean([models[(i, j, num_passes, num_tr, mu, p_ratio, fig_i)][method]
                                   for i, j in product(range(25), range(5))], axis=0)
                    results[fig_i][method][p_ratio] = re1
        for item in results[fig_i]:
            print(item, results[fig_i][item])
        print('-' * 50)

    title_list = ['SHT-AM', 'GRAPH-AM']
    for p_ratio_index, p_ratio in enumerate([0.1, 0.5]):
        for method_ind, method in enumerate(method_list):
            ax[p_ratio_index].plot([16, 32, 40, 100, 200, 400, 800],
                                   results['fig_2'][method][p_ratio],
                                   c=color_list[method_ind], linestyle='-',
                                   markerfacecolor='none',
                                   marker=marker_list[method_ind], markersize=8.,
                                   markeredgewidth=1.2, linewidth=1.2,
                                   label=title_list[method_ind])

    ax[0].set_ylabel(r'AUC score')
    for i in range(2):
        ax[i].set_xlabel(r'Block size')
    ax[1].legend(loc='lower right', fontsize=15., borderpad=0.1,
                 labelspacing=0.2, handletextpad=0.1)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f_name = '/results_exp_simu_blocksize.png'
    print('save fig to: %s' % f_name)
    plt.savefig('/home/baojian/Dropbox/Apps/ShareLaTeX/online-sparse-auc/09-27-2019' +
                f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()


def show_sparsity():
    import matplotlib.pyplot as plt
    from pylab import rcParams
    color_list = ['b', 'g', 'm', 'r', 'y', 'k', 'orange', 'Aqua']
    marker_list = ['X', 'o', 'P', 's', '*', '<', '>', 'v']
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    models = dict()
    for task_id in product(range(25)):
        f_name = os.path.join(data_path, 'results_task_%02d_sparsity.pkl' % task_id)
        re = pkl.load(open(f_name, 'rb'))
        for key in re:
            models[key] = re[key]
    rcParams['figure.figsize'] = 10, 4

    fig, ax = plt.subplots(1, 2)
    for ii in range(2):
        ax[ii].grid(b=True, which='both', color='lightgray',
                    linestyle='dotted', axis='both')
    for ind, simu in enumerate(['p-ratio=0.1', 'p-ratio=0.5']):
        ax[ind].set_title(simu)
    results = dict()
    method_list = ['sht_am', 'graph_am']
    for fig_i in ['fig_2']:
        results[fig_i] = dict()
        for method in method_list:
            results[fig_i][method] = dict()
            for p_ratio in [0.1, 0.5]:
                for num_tr, mu, num_passes in product([1000], [0.3], [10]):
                    print(models[(0, 0, num_passes, num_tr, mu, p_ratio, fig_i)][method])
                    re1 = np.mean([models[(i, j, num_passes, num_tr, mu, p_ratio, fig_i)][method]
                                   for i, j in product(range(25), range(5))], axis=0)
                    results[fig_i][method][p_ratio] = re1
        for item in results[fig_i]:
            print(item, results[fig_i][item])
        print('-' * 50)

    title_list = ['SHT-AM', 'GRAPH-AM']
    for p_ratio_index, p_ratio in enumerate([0.1, 0.5]):
        for method_ind, method in enumerate(method_list):
            ax[p_ratio_index].plot([22, 28, 34, 40, 46, 52, 58, 66, 72],
                                   results['fig_2'][method][p_ratio],
                                   c=color_list[method_ind], linestyle='-',
                                   markerfacecolor='none',
                                   marker=marker_list[method_ind], markersize=8.,
                                   markeredgewidth=1.2, linewidth=1.2,
                                   label=title_list[method_ind])

    ax[0].set_ylabel(r'AUC score')
    for i in range(2):
        ax[i].set_xlabel(r'Sparsity')
    ax[1].legend(loc='lower right', fontsize=15., borderpad=0.1,
                 labelspacing=0.2, handletextpad=0.1)
    plt.subplots_adjust(wspace=0.2, hspace=0.0)
    f_name = '/results_exp_simu_sparsity.png'
    print('save fig to: %s' % f_name)
    plt.savefig('/home/baojian/Dropbox/Apps/ShareLaTeX/online-sparse-auc/09-27-2019' +
                f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()


def results_16_bc():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/16_bc/'
    all_results = dict()
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'solam', 'opauc',
                   'graph_am']
    method_list = ['opauc', 'solam', 'spam_l2', 'fsauc', 'spam_l1', 'spam_l1l2', 'sht_am',
                   'graph_am']
    print(' ' * 10 + '         '.join(method_list))
    for task_id in range(25):
        all_results[task_id] = dict()
        list_means = []
        print('fold-%02d & ' % task_id),
        for method_name in method_list:
            f_path = os.path.join(data_path, 'ms_task_%02d_%s.pkl' % (task_id, method_name))
            results = pkl.load(open(f_path, 'rb'))
            aucs = [results[(task_id, fold_id)][method_name]['auc_wt'] for fold_id in range(5)]
            all_results[task_id][method_name] = np.mean(aucs)
            list_means.append('%.4f ' % float(np.mean(aucs)))
        print(' & '.join(list_means)),
        print('\\\\')
    print('aver:  & '),
    list_means = []
    for method in method_list:
        mean_aucs = [all_results[task_id][method] for task_id in range(25)]
        list_means.append('%.4f ' % float(np.mean(mean_aucs)))
    print(' & '.join(list_means)),


def results_09_sector():
    num_passes = 20
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector/'
    method_list = ['spam_l2', 'spam_l1', 'fsauc', 'solam', 'sht_am']
    print(' ' * 10 + '         '.join(method_list))
    for method_name in method_list:
        aucs = []
        for run_id in range(5):
            for fold_id in range(5):
                task_id = run_id * 5 + fold_id
                xx = pkl.load(
                    open(os.path.join(data_path,
                                      'results_task_%02d_passes_%s.pkl' % (task_id, num_passes)),
                         'rb'))
                for item in xx:
                    aucs.append(xx[item][method_name]['auc_wt'])
        print(method_name, '%.4f %.4f' % (np.mean(aucs), np.std(aucs)))


def test_graph():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    c1 = [21, 15, 23, 16, 19, 33, 34, 27, 24, 30, 31, 9]
    c2 = [26, 25, 28, 29, 32, 10, 3]
    c3 = [14, 20, 1, 22, 18, 4, 2, 8, 12, 13]
    c4 = [5, 7, 6, 17, 11]
    c_dict = dict()
    for ci, __ in zip([c1, c2, c3, c4], ['c1', 'c2', 'c3', 'c4']):
        for _ in ci:
            c_dict[_] = __

    with open('/home/baojian/git/deepwalk-c/deepwalk/karate.embeddings', 'rb') as f:
        for each_row in f.readlines()[1:]:
            items = each_row.lstrip().rstrip().split(' ')
            print('%02d %.6f %.6f' % (int(items[0]), float(items[1]), float(items[2])))
            representation[int(items[0]) - 1][0] = float(items[1])
            representation[int(items[0]) - 1][1] = float(items[2])

    for ci, __ in zip([c1, c2, c3, c4], ['c1', 'c2', 'c3', 'c4']):
        nodes = np.zeros(shape=(len(ci), 2))
        for i, _ in enumerate(ci):
            nodes[i] = representation[_ - 1]
        plt.scatter(nodes[:, 0], nodes[:, 1], label=__)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    representation = np.zeros(shape=(34, 2))
    data_name = '09_sector'
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
    method_list = ['opauc', 'spam_l2', 'solam', 'fsauc', 'spam_l1', 'spam_l1l2', 'sht_am']
    results_auc = {_: [] for _ in method_list}
    import matplotlib.pyplot as plt

    for run_id, fold_id in product(range(5), range(5)):
        task_id = run_id * 5 + fold_id
        f_name = os.path.join(data_path, '%s/results_task_%02d_passes_%02d.pkl'
                              % (data_name, task_id, 20))
        re = pkl.load(open(f_name, 'rb'))
        plt.figure()
        for _ in re[(run_id, fold_id)]:
            results_auc[_].append(re[(run_id, fold_id)][_]['auc_wt'])
            x = re[(run_id, fold_id)][_]['rts']
            print(len(x), len(re[(run_id, fold_id)][_]['auc']))
            if _ != 'opauc':
                plt.plot(x, re[(run_id, fold_id)][_]['auc'], label=_)
            else:
                plt.plot(x, re[(run_id, fold_id)][_]['auc'], label=_)
        plt.legend()
        plt.show()
    str_list = []
    for method in method_list:
        aver_auc = '{:.4f}'.format(float(np.mean(results_auc[method]))).lstrip('0')
        std_auc = '{:.4f}'.format(float(np.std(results_auc[method]))).lstrip('0')
        print(aver_auc + '$\pm$' + std_auc + ' & '),
