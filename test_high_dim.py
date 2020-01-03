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
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    for trial_id, fold_id in product(range(data['num_trials']), range(k_fold)):
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        selected_b, best_auc = None, None
        for para_b in range(1, 40, 1):
            _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1., 0.0)
            wt, aucs, rts, epochs = _
            re = {'algo_para': [trial_id, fold_id, para_s, para_b],
                  'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                          y_score=np.dot(data['x_tr'][te_index], wt)),
                  'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
            if selected_b is None or best_auc is None or best_auc < re['auc_wt']:
                selected_b = para_b
                best_auc = re['auc_wt']
        print('selected b: %d best_auc: %.4f' % (selected_b, best_auc))
        _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, selected_b, 1., 0.0)
        wt, aucs, rts, epochs = _
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, para_s, selected_b],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
    print(para_s, '%.5f' % np.mean(np.asarray([results[_]['auc_wt'] for _ in results])))
    return para_s, results


def run_sht_am():
    """ https://www.genome.jp/kegg/tool/conv_id.html """
    # process_data_20_colon()
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    data = pkl.load(open(data_path + 'colon_data.pkl'))
    __ = np.empty(shape=(1,), dtype=float)
    para_list = []
    for para_s in range(10, 101, 5):
        para_list.append((data, para_s))
    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    ms_res = pool.map(cv_sht_am, para_list)
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(data_path + 're_sht_am_v1.pkl', 'wb'))


def show_results():
    results = dict()
    k_fold, step_len, verbose, record_aucs, stop_eps = 5, 1e2, 0, 1, 1e-7
    import matplotlib.pyplot as plt
    all_res = []
    selected_features = dict()
    for num_passes in [50]:
        global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
        re_aucs = []
        for trial_id, fold_id in product(range(5), range(k_fold)):
            tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
            te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
            para_s, para_b = 20, 7
            x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
            y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
            _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 2., 0.0)
            wt, aucs, rts, epochs = _
            results = {'algo_para': [trial_id, fold_id, para_s, para_b],
                       'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                               y_score=np.dot(data['x_tr'][te_index], wt)),
                       'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
            for ii in np.nonzero(wt)[0]:
                if ii not in selected_features:
                    selected_features[ii] = 0
                selected_features[ii] += 1
            print('trial-%d fold-%d auc: %.4f para_s:%03d para_b:%03d' %
                  (trial_id, fold_id, results['auc_wt'], para_s, para_b))
            re_aucs.append(results['auc_wt'])
            plt.plot(rts, aucs)
        all_res.append(np.mean(re_aucs))
        plt.show()
        plt.close()
    xxxxx = set(selected_features.keys()).intersection(set(
        [295, 451, 608, 1041, 1043, 1165, 1279, 918, 1352, 1386, 1393, 1870]))
    print(xxxxx, len(xxxxx))
    print(len(selected_features))

    xxxxx = set(selected_features.keys()).intersection(set(
        [554, 268, 250, 146, 1463, 112, 29, 829, 325, 137, 209, 158, 1143, 316, 225, 207, 860, 163, 154, 701, 534, 510,
         512, 628, 295, 451, 608, 1041, 1043, 1165, 1279, 918, 1352, 1386, 1393, 1870]))
    print(xxxxx, len(xxxxx))
    print(len(selected_features))
    exit()
    xx = [(k, v) for k, v in sorted(selected_features.items(), key=lambda item: item[1], reverse=True)]
    xxx = []
    for x in xx[:25]:
        print(x[0], data['feature_names'][x[0]])
        xxx.append(str(data['feature_names'][x[0]]).split('Hsa.')[1])
    for item in xxx:
        print(item)
    plt.plot(all_res, marker='D')
    plt.show()
    plt.show()


def main():
    if sys.argv[1] == 'run_sht_am':
        run_sht_am()


if __name__ == '__main__':
    main()
