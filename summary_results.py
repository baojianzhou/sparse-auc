# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import pickle as pkl

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector/lib-svm-data/'


def result_summary():
    all_results = []
    for task_id in range(100):
        task_start, task_end = int(task_id) * 21, int(task_id) * 21 + 21
        f_name = data_path + 'model_select_%04d_%04d_5.pkl' % (task_start, task_end)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)
    file_name = data_path + 'model_select_0000_2100_5.pkl'
    pkl.dump(all_results, open(file_name, 'wb'))


result_summary()
