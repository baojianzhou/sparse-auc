# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
from itertools import product


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
    passes_list = [1, 5, 10, 15, 20, 25, 30]
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

    for sparsity in [50, 100, 150, 200, 250, 300]:
        y, yerr = [], []
        for num_passes in [1, 5, 10, 15, 20, 25, 30]:
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
    global_sparsity = 4000
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


result_summary_00_simu()
