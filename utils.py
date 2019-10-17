# -*- coding: utf-8 -*-
import os
import time
import scipy.io
import numpy as np
import pickle as pkl
from os.path import join
from itertools import product
from sklearn.metrics import auc
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from test_high_dim_data import get_model_para
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 15
from pylab import rcParams


def converge_curve(method_list, data_name, passes):
    rcParams['figure.figsize'] = 16, 14
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
    if method_list is None:
        method_list = ['opauc', 'spam_l2', 'solam', 'fsauc', 'spam_l1', 'spam_l1l2', 'sht_am']
    color_dict = {'opauc': 'black',
                  'spam_l2': 'blue',
                  'solam': 'green',
                  'fsauc': 'yellow',
                  'spam_l1': 'purple',
                  'spam_l1l2': 'brown',
                  'sht_am': 'red'}
    model_results = {(run_id, fold_id): {} for run_id, fold_id in product(range(5), range(5))}
    for ind, method in enumerate(method_list):
        f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
        re = pkl.load(open(f_name, 'rb'))
        for item in re:
            model_results[item.keys()[0]][method] = item.values()[0]
    fig, ax = plt.subplots(5, 5)
    for i in range(0, 5):
        for j in range(0, 4):
            ax[j, i].set_xticks([])
    for i in range(1, 5):
        for j in range(5):
            ax[j, i].set_yticks([])
    for i in range(5):
        ax[i, 0].set_ylabel('AUC')
        ax[i, 0].set_yticks([0.4, 0.6, 0.8])
    for i in range(5):
        ax[4, i].set_xlabel('run time (seconds)')
        ax[4, i].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])

    averaged = {method: {'x': [], 'y': []} for method in method_list}
    min_len = np.inf
    for run_id, fold_id in product(range(5), range(5)):
        re = model_results[(run_id, fold_id)]
        for method in method_list:
            len_x = len([x for x in re[method]['rts'] if x < 5.])
            aver_run_time = re[method]['rts'][-1] / float(passes)
            ax[run_id, fold_id].plot(re[method]['rts'][:len_x], re[method]['auc'][:len_x],
                                     label='%-10s : %2.2f(sec)' % (method, aver_run_time),
                                     color=color_dict[method], linewidth=2)
            averaged[method]['x'].append(re[method]['rts'])
            averaged[method]['y'].append(re[method]['auc'])
            min_len = min(min_len, len(re[method]['rts']))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig('/home/baojian/10_farmads.pdf', dpi=600, bbox_inches='tight',
                pad_inches=0, format='pdf')
    rcParams['figure.figsize'] = 8, 8
    fig, ax = plt.subplots(1, 1)
    for method in method_list:
        x = np.mean(np.asarray([_[:min_len] for _ in averaged[method]['x']]), axis=0)
        y = np.mean(np.asarray([_[:min_len] for _ in averaged[method]['y']]), axis=0)
        len_x = len([_ for _ in x if _ < 5.])
        ax.plot(x[:len_x], y[:len_x], label='%-10s' % method,
                color=color_dict[method], linewidth=2.5)
        ax.set_ylabel('AUC')
        ax.set_xlabel('Run time (seconds)')
    ax.legend(loc='lower right', fontsize=18.)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig('/home/baojian/10_averaged.pdf', dpi=600, bbox_inches='tight',
                pad_inches=0, format='pdf')


def average_scores(method_list, data_name, passes):
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'
    if method_list is None:
        method_list = ['opauc', 'spam_l2', 'solam', 'fsauc', 'spam_l1', 'spam_l1l2', 'sht_am']
    for metric in ['auc_wt', 'nonzero_wt']:
        results_auc = {_: [] for _ in method_list}
        for method in method_list:
            f_name = join(data_path, '%s/results_%s_%02d.pkl' % (data_name, method, passes))
            re = [_.values()[0] for _ in pkl.load(open(f_name, 'rb'))]
            results_auc[method] = [_[metric] for _ in re]
        for method in method_list:
            aver_auc = '{:.4f}'.format(float(np.mean(results_auc[method]))).lstrip('0')
            std_auc = '{:.4f}'.format(float(np.std(results_auc[method]))).lstrip('0')
            print(aver_auc + '$\pm$' + std_auc + ' & '),
        print('')
        if metric == 'auc_wt':
            for method in method_list:
                if method == 'sht_am':
                    continue
                x1, x2 = results_auc[method], results_auc['sht_am']
                t_score, p_value = ttest_ind(a=x1, b=x2)
                color = 'none'  # comparable
                if np.mean(x1) > np.mean(x2) and p_value <= 0.05:
                    color = 'white'  # worse
                if np.mean(x1) < np.mean(x2) and p_value <= 0.05:
                    color = 'black'  # better
                ll = ['%10s' % method, '%.4f' % np.mean(x1), '%.4f' % np.mean(x2),
                      '%.4f' % p_value, color]
                print(' '.join(ll))


def get_selected_paras(data_name, method_list):
    for method in method_list:
        for run_id, fold_id in product(range(5), range(5)):
            print(method, get_model_para(data_name, method, run_id, fold_id))


if __name__ == '__main__':
    converge_curve(method_list=['spam_l1', 'spam_l2', 'solam', 'fsauc', 'sht_am'],
                   data_name='10_farmads', passes=20)
    get_selected_paras(data_name='10_farmads',
                       method_list=['spam_l1', 'spam_l2', 'solam', 'fsauc', 'sht_am'])
    average_scores(method_list=['spam_l1', 'spam_l2', 'solam', 'fsauc', 'sht_am'],
                   data_name='10_farmads', passes=20)
