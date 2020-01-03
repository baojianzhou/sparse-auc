# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import pickle as pkl
import multiprocessing
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_solam
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_sht_am
        from sparse_module import c_algo_graph_am
        from sparse_module import c_algo_opauc
        from sparse_module import c_algo_sto_iht
        from sparse_module import c_algo_hsg_ht
        from sparse_module import c_algo_fsauc
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        pass
except ImportError:
    print('cannot find the module: sparse_module')
    pass


def process_data_20_colon():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    data = {'feature_ids': None, 'x_tr': [], 'y_tr': [], 'feature_names': []}
    import csv
    with open(data_path + 'colon_x.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                data['feature_ids'] = [int(_) for _ in row[1:]]
                line_count += 1
            elif 1 <= line_count <= 62:
                data['x_tr'].append([float(_) for _ in row[1:]])

                line_count += 1
        data['x_tr'] = np.asarray(data['x_tr'])
        for i in range(len(data['x_tr'])):
            data['x_tr'][i] = data['x_tr'][i] / np.linalg.norm(data['x_tr'][i])
    with open(data_path + 'colon_y.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            elif 1 <= line_count <= 62:
                line_count += 1
                if row[1] == 't':
                    data['y_tr'].append(1.)
                else:
                    data['y_tr'].append(-1.)
        data['y_tr'] = np.asarray(data['y_tr'])
    with open(data_path + 'colon_names.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            elif 1 <= line_count <= 2000:
                line_count += 1
                if row[1] == 't':
                    data['feature_names'].append(row[1])
                else:
                    data['feature_names'].append(row[1])
    data['n'] = 62
    data['p'] = 2000
    data['num_trials'] = 20
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ == 1.])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ == -1.])
    trial_i = 0
    while True:
        # since original data is ordered, we need to shuffle it!
        rand_perm = np.random.permutation(data['n'])
        train_ind, test_ind = rand_perm[:50], rand_perm[50:]
        if len([_ for _ in data['y_tr'][train_ind] if _ == 1.]) == 33 or \
                len([_ for _ in data['y_tr'][train_ind] if _ == 1.]) == 32:
            data['trial_%d' % trial_i] = {'tr_index': rand_perm[train_ind], 'te_index': rand_perm[test_ind]}
            print(len([_ for _ in data['y_tr'][train_ind] if _ == 1.]),
                  len([_ for _ in data['y_tr'][train_ind] if _ == -1.]),
                  len([_ for _ in data['y_tr'][test_ind] if _ == 1.]),
                  len([_ for _ in data['y_tr'][test_ind] if _ == -1.])),
            success = True
            kf = KFold(n_splits=5, shuffle=False)
            for fold_index, (train_index, test_index) in enumerate(kf.split(range(len(train_ind)))):
                if len([_ for _ in data['y_tr'][train_ind[test_index]] if _ == -1.]) < 3:
                    success = False
                    break
                data['trial_%d_fold_%d' % (trial_i, fold_index)] = {'tr_index': train_ind[train_index],
                                                                    'te_index': train_ind[test_index]}
                print(len([_ for _ in data['y_tr'][train_ind[train_index]] if _ == 1.]),
                      len([_ for _ in data['y_tr'][train_ind[train_index]] if _ == -1.]),
                      len([_ for _ in data['y_tr'][train_ind[test_index]] if _ == 1.]),
                      len([_ for _ in data['y_tr'][train_ind[test_index]] if _ == -1.])),
            print(trial_i)
            if success:
                trial_i += 1
        if trial_i >= data['num_trials']:
            break
    pkl.dump(data, open(data_path + 'colon_data.pkl', 'wb'))


def test_on_leukemia():
    """
    This dataset is from:
    https://github.com/ramhiser/datamicroarray/blob/master/data/golub.RData
    :return:
    """
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
    import pyreadr
    result = pyreadr.read_r(data_path + 'golub')
    print(result.keys())
    pass


def cv_sht_am(para):
    data, para_s = para
    num_passes, k_fold, step_len, verbose, record_aucs, stop_eps = 100, 5, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    results = dict()
    for trial_id, fold_id in product(range(data['num_trials']), range(k_fold)):
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        selected_b, best_auc = None, None
        for para_b in range(1, 40, 1):
            _ = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float), np.empty(shape=(1,), dtype=float),
                              np.empty(shape=(1,), dtype=float), np.empty(shape=(1,), dtype=float),
                              np.asarray(data['y_tr'][tr_index], dtype=float),
                              0, data['p'], global_paras, 0, para_s, para_b, 1., 0.0)
            wt, aucs, rts, epochs = _
            re = {'algo_para': [trial_id, fold_id, para_s, para_b],
                  'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                          y_score=np.dot(data['x_tr'][te_index], wt)),
                  'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
            if selected_b is None or best_auc is None or best_auc < re['auc_wt']:
                selected_b = para_b
                best_auc = re['auc_wt']
        print('selected b: %d best_auc: %.4f' % (selected_b, best_auc))
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        _ = c_algo_sht_am(np.asarray(data['x_tr'][tr_index], dtype=float), np.empty(shape=(1,), dtype=float),
                          np.empty(shape=(1,), dtype=float), np.empty(shape=(1,), dtype=float),
                          np.asarray(data['y_tr'][tr_index], dtype=float),
                          0, data['p'], global_paras, 0, para_s, selected_b, 1., 0.0)
        wt, aucs, rts, epochs = _
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, para_s, selected_b],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
    print(para_s, '%.5f' % np.mean(np.asarray([results[_]['auc_wt'] for _ in results])))
    return para_s, results


def cv_sto_iht(para):
    data, para_s = para
    num_passes, k_fold, step_len, verbose, record_aucs, stop_eps = 100, 5, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    for trial_id, fold_id in product(range(data['num_trials']), range(k_fold)):
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        selected_b, best_auc = None, None
        for para_b in range(1, 40, 1):
            _ = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, para_b, 1., 0.0)
            wt, aucs, rts, epochs = _
            re = {'algo_para': [trial_id, fold_id, para_s, para_b],
                  'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                          y_score=np.dot(data['x_tr'][te_index], wt)),
                  'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
            if selected_b is None or best_auc is None or best_auc < re['auc_wt']:
                selected_b = para_b
                best_auc = re['auc_wt']
        print('selected b: %d best_auc: %.4f' % (selected_b, best_auc))
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, selected_b, 1., 0.0)
        wt, aucs, rts, epochs = _
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, para_s, selected_b],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt}
    print(para_s, '%.5f' % np.mean(np.asarray([results[_]['auc_wt'] for _ in results])))
    return para_s, results


def cv_spam_l1(para):
    data, para_l1 = para
    num_passes, k_fold, step_len, verbose, record_aucs, stop_eps = 100, 5, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    for trial_id, fold_id in product(range(data['num_trials']), range(k_fold)):
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        selected_xi, best_auc = None, None
        aver_nonzero = []
        for para_xi in 10. ** np.arange(-5, 3, 1, dtype=float):
            _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_l1, 0.0)
            wt, aucs, rts, epochs = _
            aver_nonzero.append(np.count_nonzero(wt))
            re = {'algo_para': [trial_id, fold_id, para_xi, para_l1],
                  'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                          y_score=np.dot(data['x_tr'][te_index], wt)),
                  'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
            if best_auc is None or best_auc < re['auc_wt']:
                selected_xi = para_xi
                best_auc = re['auc_wt']
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, selected_xi, para_l1, 0.0)
        wt, aucs, rts, epochs = _
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, selected_xi, para_l1],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt}
        print('selected xi: %.4e l1: %.4e nonzero: %.4e test_auc: %.4f' %
              (selected_xi, para_l1, float(np.mean(aver_nonzero)), results[(trial_id, fold_id)]['auc_wt']))
    print(para_l1, '%.5f' % np.mean(np.asarray([results[_]['auc_wt'] for _ in results])))
    return para_l1, results


def cv_spam_l2(para):
    data, para_ind = para
    num_passes, k_fold, step_len, verbose, record_aucs, stop_eps = 100, 5, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    for trial_id, fold_id in product(range(data['num_trials']), range(k_fold)):
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        selected_xi, select_l2, best_auc = None, None, None
        aver_nonzero = []
        for para_xi, para_l2, in product(10. ** np.arange(-5, 3, 1, dtype=float),
                                         10. ** np.arange(-5, 3, 1, dtype=float)):
            _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            aver_nonzero.append(np.count_nonzero(wt))
            re = {'algo_para': [trial_id, fold_id, para_xi, para_l2],
                  'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                          y_score=np.dot(data['x_tr'][te_index], wt)),
                  'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
            if best_auc is None or best_auc < re['auc_wt']:
                selected_xi = para_xi
                select_l2 = para_l2
                best_auc = re['auc_wt']
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, selected_xi, 0.0, select_l2)
        wt, aucs, rts, epochs = _
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, selected_xi, select_l2],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt}
        print('selected xi: %.4e l2: %.4e nonzero: %.4e test_auc: %.4f' %
              (selected_xi, select_l2, float(np.mean(aver_nonzero)), results[(trial_id, fold_id)]['auc_wt']))
    print(para_ind, '%.5f' % np.mean(np.asarray([results[_]['auc_wt'] for _ in results])))
    return para_ind, results


def run_methods(method):
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    data = pkl.load(open(data_path + 'colon_data.pkl'))
    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    if method == 'sht_am':
        para_list = []
        for para_s in range(10, 101, 5):
            para_list.append((data, para_s))
        ms_res = pool.map(cv_sht_am, para_list)
    elif method == 'sto_iht':
        para_list = []
        for para_s in range(10, 101, 5):
            para_list.append((data, para_s))
        ms_res = pool.map(cv_sto_iht, para_list)
    elif method == 'spam_l1':
        para_list = []
        for para_l1 in [1e-5, 3e-5, 5e-5, 7e-5, 9e-5,
                        1e-4, 3e-4, 5e-4, 7e-4, 9e-4,
                        1e-3, 3e-3, 5e-3, 7e-3, 9e-3,
                        1e-2, 3e-2, 5e-2, 7e-2]:
            para_list.append((data, para_l1))
        ms_res = pool.map(cv_spam_l1, para_list)
    elif method == 'spam_l2':
        para_list = []
        for para_ind in range(19):
            para_list.append((data, para_ind))
        ms_res = pool.map(cv_spam_l2, para_list)
    else:
        ms_res = None
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(data_path + 're_%s.pkl' % method, 'wb'))


def show_results():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    re_sht_am = {_[0]: _[1] for _ in pkl.load(open(data_path + 're_sht_am_v1.pkl'))}
    re_sto_iht = {_[0]: _[1] for _ in pkl.load(open(data_path + 're_sto_iht.pkl'))}
    re_spam_l1 = {_[0]: _[1] for _ in pkl.load(open(data_path + 're_spam_l1.pkl'))}
    auc_sht_am = []
    auc_sto_iht = []
    auc_spam_l1 = []
    for para_s in range(10, 101, 5):
        auc_sht_am.append(np.mean([re_sht_am[para_s][_]['auc_wt'] for _ in re_sht_am[para_s]]))
        auc_sto_iht.append(np.mean([re_sto_iht[para_s][_]['auc_wt'] for _ in re_sto_iht[para_s]]))
    for para_l1 in [1e-5, 3e-5, 5e-5, 7e-5, 9e-5,
                    1e-4, 3e-4, 5e-4, 7e-4, 9e-4,
                    1e-3, 3e-3, 5e-3, 7e-3, 9e-3,
                    1e-2, 3e-2, 5e-2, 7e-2][::-1]:
        auc_spam_l1.append(np.mean([re_spam_l1[para_l1][_]['auc_wt'] for _ in re_spam_l1[para_l1]]))
    import matplotlib.pyplot as plt
    plt.plot(range(10, 101, 5), auc_sht_am, label='SHT-AUC')
    plt.plot(range(10, 101, 5), auc_sto_iht, label='Sto-IHT')
    plt.plot(range(10, 101, 5), auc_spam_l1, label='SPAM-L1')
    plt.legend()
    plt.show()


def main():
    if sys.argv[1] == 'run_sht_am':
        run_methods(method='sht_am')
    elif sys.argv[1] == 'run_sto_iht':
        run_methods(method='sto_iht')
    elif sys.argv[1] == 'run_spam_l1':
        run_methods(method='spam_l1')
    elif sys.argv[1] == 'run_spam_l2':
        run_methods(method='spam_l2')
    elif sys.argv[1] == 'show_01':
        show_results()


if __name__ == '__main__':
    main()
