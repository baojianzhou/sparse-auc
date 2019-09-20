# -*- coding: utf-8 -*-
import os
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


def combine_results():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    tr_list = [1000]
    mu_list = [0.3]
    posi_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    for task_id, num_tr, mu, posi_ratio, fig_i in product(
            range(25), tr_list, mu_list, posi_ratio_list, fig_list):
        print(task_id, num_tr, mu, posi_ratio, fig_i)
        f_name1 = os.path.join(data_path, 'ms_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f_%s.pkl' %
                               (task_id, num_tr, mu, posi_ratio, fig_i))
        re1 = pkl.load(open(f_name1, 'rb'))
        f_name = os.path.join(data_path, 'ms_task_%02d_tr_%03d_mu_%.1f_p-ratio_%.1f_%s_opauc.pkl' %
                              (task_id, num_tr, mu, posi_ratio, fig_i))
        re2 = pkl.load(open(f_name, 'rb'))[task_id]
        re1[re1.keys()[0]].update(re2[re2.keys()[0]])
        pkl.dump(re1, open(f_name1, 'wb'))


def test():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    models = pkl.load(open(data_path + 'models.pkl', 'rb'))
    print(len(models))


if __name__ == '__main__':
    result_summary_00_simu_re()
    exit()
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    dd = pkl.load(open(data_path + 'ms_task_22_tr_1000_mu_0.3_p-ratio_0.3_fig_3.pkl', 'rb'))
    pass
