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


def result_summary_00_simu():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    num_runs, k_fold = 5, 5
    global_sparsity = 100

    for num_passes in [10, 20, 30, 40, 50]:
        auc_wt, auc_wt_bar = [], []
        for ind in range(num_runs * k_fold):
            f_name = data_path + 're_spam_l2_%02d_passes_%03d.pkl' % (ind, num_passes)
            auc_wt.append(pkl.load(open(f_name, 'rb'))['auc_wt'])
            auc_wt_bar.append(pkl.load(open(f_name, 'rb'))['auc_wt_bar'])
        print(np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar))

    for num_passes in [10, 20, 30, 40, 50]:
        # model selection
        all_results, num_runs, k_fold = [], 5, 5
        for ind in range(num_runs * k_fold):
            f_name = data_path + 'ms_sht_am_l2_task_%02d_passes_%03d_sparsity_%04d.pkl' % \
                     (ind, num_passes, global_sparsity)
            all_results.extend(pkl.load(open(f_name, 'rb')))
        for model in ['wt', 'wt_bar']:
            selected_model = dict()
            for result in all_results:
                run_id, fold_id, para_sparsity, para_xi, para_beta, num_passes, num_runs, k_fold = \
                    result['algo_para']
                mean_auc = np.mean(result['list_auc_%s' % model])
                if (run_id, fold_id) not in selected_model:
                    selected_model[(run_id, fold_id)] = (
                        mean_auc, run_id, fold_id, para_xi, para_beta)
                if mean_auc > selected_model[(run_id, fold_id)][0]:
                    selected_model[(run_id, fold_id)] = (
                        mean_auc, run_id, fold_id, para_xi, para_beta)
            for item in selected_model:
                print(item, selected_model[item])

        auc_wt, auc_wt_bar = [], []
        for ind in range(num_runs * k_fold):
            f_name = data_path + 're_sht_am_%02d_passes_%03d_sparsity_%04d.pkl' % \
                     (ind, num_passes, 100)
            auc_wt.append(pkl.load(open(f_name, 'rb'))['auc_wt'])
            auc_wt_bar.append(pkl.load(open(f_name, 'rb'))['auc_wt_bar'])
        print(np.mean(auc_wt), np.std(auc_wt), np.mean(auc_wt_bar), np.std(auc_wt_bar))


result_summary_00_simu()
