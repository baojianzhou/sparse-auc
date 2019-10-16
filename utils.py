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
    get_selected_paras(data_name='10_farmads', method_list=['spam_l1', 'spam_l2', 'sht_am'])
    average_scores(method_list=['spam_l1', 'spam_l2', 'sht_am'],
                   data_name='10_farmads', passes=20)
