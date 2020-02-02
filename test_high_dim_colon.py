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

"""
Related genes are found by the following paper:
Agarwal, Shivani, and Shiladitya Sengupta.
"Ranking genes by relevance to a disease."
Proceedings of the 8th annual international 
conference on computational systems bioinformatics. 2009.
"""
related_genes = {683: "01 -- Hsa.1036 -- Phospholipase A2",
                 1235: "01 -- Hsa.290 -- Phospholipase A2",
                 295: "01 -- Hsa.994 -- Phospholipase A2",
                 451: "02 -- Hsa.3328 -- Keratin 6 isoform",
                 608: "03 -- Hsa.24944 -- Protein-tyrosine phosphatase PTP-H1",
                 1041: "04 -- Hsa.549 -- Transcription factor IIIA",
                 1043: "05 -- Hsa.13522 -- Viral (v-raf) oncogene homolog 1",
                 1165: "06 -- Hsa.7348 -- Dual specificity mitogen-activated protein kinase kinase 1",
                 1279: "07 -- Hsa.1280 -- Transmembrane carcinoembryonic antigen",
                 917: "07 -- Hsa.3068 -- Transmembrane carcinoembryonic antigen",
                 1352: "08 -- Hsa.2957 -- Oncoprotein 18",
                 1386: "09 -- Hsa.1902 -- Phosphoenolpyruvate carboxykinase",
                 1870: "10 -- Hsa.865 -- Extracellular signal-regulated kinase 1",
                 1393: "10 -- Hsa.42746 -- Extracellular signal-regulated kinase 1",
                 554: "11 -- Hsa.1098 -- 26 kDa cell surface protein TAPA-1",
                 268: "12 -- Hsa.2806 -- Id1",
                 146: "13 -- Hsa.558 -- Interferon-inducible protein 9-27",
                 1463: "14 -- Hsa.558 -- Nonspecific crossreacting antigen",
                 112: "15 -- Hsa.68 -- cAMP response element regulatory protein (CREB2)",
                 325: "16 -- Hsa.256 -- Splicing factor (CC1.4)",
                 137: "17 -- Hsa.957 -- Nucleolar protein (B23)",
                 209: "18 -- Hsa.2846 -- Lactate dehydrogenase-A (LDH-A)",
                 158: "19 -- Hsa.45604 -- Guanine nucleotide-binding protein G(OLF)",
                 170: "19 -- Hsa.45604 -- Guanine nucleotide-binding protein G(OLF)",
                 175: "19 -- Hsa.25451 -- Guanine nucleotide-binding protein G(OLF)",
                 1143: "20 -- Hsa.393 -- LI-cadherin",
                 316: "21 -- Hsa.891 -- Lysozyme",
                 225: "22 -- Hsa.3295 -- Prolyl 4-hydroxylase (P4HB)",
                 207: "23 -- Hsa.338 -- Eukaryotic initiation factor 4AII",
                 163: "24 -- Hsa.5821 -- Interferon-inducible protein 1-8D",
                 701: "25 -- Hsa.109 -- Dipeptidase",
                 1790: "26 -- Hsa.2794 -- Heat shock 27 kDa protein",
                 534: "27 -- Hsa.5633 -- Tyrosine-protein kinase receptor TIE-1 precursor",
                 512: "28 -- Hsa.831 -- Mitochondrial matrix protein P1 precursor",
                 1: "29 -- Hsa.13491 -- Eukaryotic initiation factor EIF-4A homolog",
                 2: "29 -- Hsa.13491 -- Eukaryotic initiation factor EIF-4A homolog",
                 282: "29 -- Hsa.80 -- Eukaryotic initiation factor EIF-4A homolog",
                 613: "29 -- Hsa.9251 -- Eukaryotic initiation factor EIF-4A homolog"}


def process_data_20_colon():
    """
    https://github.com/ramhiser/datamicroarray
    http://genomics-pubs.princeton.edu/oncology/affydata/index.html
    :return:
    """
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


def cv_sht_auc(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    all_results = dict()
    s_list = range(5, 101, 2)
    s_list.extend([120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    for para_s in s_list:
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        results = dict()
        best_b, best_auc = None, None
        for para_b in range(1, 40, 1):
            wt, _, _, _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1., 0.0)
            auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
            if best_b is None or best_auc is None or best_auc < auc_score:
                best_b, best_auc = para_b, auc_score
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        wt, aucs, rts, epochs = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras,
                                              0, para_s, best_b, 1., 0.0)
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, para_s, best_b],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)), 'wt': wt}
        print('best_b: %02d nonzero: %.4e test_auc: %.4f' %
              (best_b, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
        all_results[para_s] = results
    return trial_id, fold_id, all_results


def cv_sto_iht(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    all_results = dict()
    s_list = range(1, 101, 2)
    s_list.extend([120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    for para_s in s_list:
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        results = dict()
        best_b, best_auc = None, None
        for para_b in range(1, 40, 1):
            wt, _, _, _ = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, para_b, 1., 0.0)
            auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
            if best_b is None or best_auc is None or best_auc < auc_score:
                best_b, best_auc = para_b, auc_score
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        wt, aucs, rts, epochs = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'],
                                               global_paras, para_s, best_b, 1., 0.0)
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, para_s, best_b],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)), 'wt': wt}
        print('best_b: %02d nonzero: %.4e test_auc: %.4f' %
              (best_b, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
        all_results[para_s] = results
    return trial_id, fold_id, all_results


def cv_hsg_ht(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    all_results = dict()
    s_list = range(1, 101, 2)
    s_list.extend([120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    for para_s in s_list:
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        results = dict()
        best_c, best_auc = None, None
        for para_tau in [1., 10., 100., 1000.]:
            para_c, para_zeta = 3.0, 1.033
            wt, _, _, _ = c_algo_hsg_ht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras,
                                        para_s, para_tau, para_zeta, para_c, 0.0)
            auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
            if best_auc is None or best_auc < auc_score:
                best_c, best_auc = para_c, auc_score
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        wt, aucs, rts, epochs = c_algo_hsg_ht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras,
                                              para_s, para_tau, para_zeta, best_c, 0.0)
        results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, best_c, para_s],
                                        'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                                y_score=np.dot(data['x_tr'][te_index], wt)), 'wt': wt}
        print('best_c: %02d nonzero: %.4e test_auc: %.4f' %
              (best_c, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
        all_results[para_s] = results
    return trial_id, fold_id, all_results


def cv_spam_l1(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    best_xi, best_l1, best_auc = None, None, None
    for para_xi, para_l1 in product(10. ** np.arange(-5, 3, 1, dtype=float),
                                    10. ** np.arange(-5, 3, 1, dtype=float)):
        wt, _, _, _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_l1, 0.0)
        auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
        if best_auc is None or best_auc < auc_score:
            best_xi, best_l1, best_auc = para_xi, para_l1, auc_score
    tr_index = data['trial_%d' % trial_id]['tr_index']
    te_index = data['trial_%d' % trial_id]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    wt, aucs, rts, epochs = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, best_xi, best_l1, 0.0)
    results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, best_xi, best_l1],
                                    'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                            y_score=np.dot(data['x_tr'][te_index], wt)),
                                    'aucs': aucs, 'rts': rts, 'wt': wt}
    print('best_xi: %.1e best_l1: %.1e nonzero: %.4e test_auc: %.4f' %
          (best_xi, best_l1, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
    return trial_id, fold_id, results


def cv_spam_l2(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    best_xi, best_l2, best_auc = None, None, None
    for para_xi, para_l2, in product(10. ** np.arange(-5, 3, 1, dtype=float),
                                     10. ** np.arange(-5, 3, 1, dtype=float)):
        wt, _, _, _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
        auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
        if best_auc is None or best_auc < auc_score:
            best_xi, best_l2, best_auc = para_xi, para_l2, auc_score
    tr_index = data['trial_%d' % trial_id]['tr_index']
    te_index = data['trial_%d' % trial_id]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    wt, aucs, rts, epochs = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, best_xi, 0.0, best_l2)
    results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, best_xi, best_l2],
                                    'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                            y_score=np.dot(data['x_tr'][te_index], wt)),
                                    'aucs': aucs, 'rts': rts, 'wt': wt}
    print('best_xi: %.1e best_l2: %.1e nonzero: %.4e test_auc: %.4f' %
          (best_xi, best_l2, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
    return trial_id, fold_id, results


def cv_spam_l1l2(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    best_xi, best_l1, best_l2, best_auc = None, None, None, None
    for para_xi, para_l1, para_l2, in product(10. ** np.arange(-5, 3, 1, dtype=float),
                                              10. ** np.arange(-5, 3, 1, dtype=float),
                                              10. ** np.arange(-5, 3, 1, dtype=float)):
        wt, _, _, _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_l1, para_l2)
        auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
        if best_auc is None or best_auc < auc_score:
            best_xi, best_l1, best_l2, best_auc = para_xi, para_l1, para_l2, auc_score
    tr_index = data['trial_%d' % trial_id]['tr_index']
    te_index = data['trial_%d' % trial_id]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    wt, aucs, rts, epochs = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, best_xi, best_l1, best_l2)
    results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, best_xi, best_l1, best_l2],
                                    'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                            y_score=np.dot(data['x_tr'][te_index], wt)),
                                    'aucs': aucs, 'rts': rts, 'wt': wt}
    print('best_xi: %.1e best_l1: %.1e best_l2: %.1e nonzero: %.4e test_auc: %.4f' %
          (best_xi, best_l1, best_l2, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
    return trial_id, fold_id, results


def cv_solam(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    best_xi, best_r, best_auc = None, None, None
    for para_xi, para_r, in product(np.arange(1, 101, 9, dtype=float),
                                    10. ** np.arange(-1, 6, 1, dtype=float)):
        wt, _, _, _ = c_algo_solam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_r)
        auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
        if best_auc is None or best_auc < auc_score:
            best_xi, best_r, best_auc = para_xi, para_r, auc_score
    tr_index = data['trial_%d' % trial_id]['tr_index']
    te_index = data['trial_%d' % trial_id]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    wt, aucs, rts, epochs = c_algo_solam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, best_xi, best_r)
    results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, best_xi, best_r],
                                    'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                            y_score=np.dot(data['x_tr'][te_index], wt)),
                                    'aucs': aucs, 'rts': rts, 'wt': wt}
    print('best_xi: %.1e best_r: %.1e nonzero: %.4e test_auc: %.4f' %
          (best_xi, best_r, float(np.count_nonzero(wt)), results[(trial_id, fold_id)]['auc_wt']))
    return trial_id, fold_id, results


def cv_fsauc(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    results = dict()
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    best_g, best_r, best_auc = None, None, None
    aver_nonzero = []
    for para_g, para_r in product(2. ** np.arange(-10, 11, 1, dtype=float),
                                  10. ** np.arange(-1, 6, 1, dtype=float)):
        wt, _, _, _ = c_algo_fsauc(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_r, para_g)
        auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
        if best_auc is None or best_auc < auc_score:
            best_g, best_r, best_auc = para_g, para_r, auc_score
    tr_index = data['trial_%d' % trial_id]['tr_index']
    te_index = data['trial_%d' % trial_id]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    wt, aucs, rts, epochs = c_algo_fsauc(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, best_r, best_g)
    results[(trial_id, fold_id)] = {'algo_para': [trial_id, fold_id, best_g, best_r],
                                    'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                            y_score=np.dot(data['x_tr'][te_index], wt)),
                                    'aucs': aucs, 'rts': rts, 'wt': wt}
    print('best_g: %.1e best_r: %.1e nonzero: %.4e test_auc: %.4f' %
          (best_g, best_r, float(np.mean(aver_nonzero)), results[(trial_id, fold_id)]['auc_wt']))
    return trial_id, fold_id, results


def summary_auc_results():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    all_aus = dict()
    for method_ind, method in enumerate(['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc']):
        re_summary = pkl.load(open(data_path + 're_%s.pkl' % method, 'rb'))
        all_aus[method] = dict()
        for trial_id, fold_id, re in re_summary:
            all_aus[method][(trial_id, fold_id)] = re[(trial_id, fold_id)]['auc_wt']
        print(method, np.mean(np.asarray(all_aus[method].values())))
    for method_ind, method in enumerate(['sht_am', 'sto_iht', 'hsg_ht']):
        re_summary = pkl.load(open(data_path + 're_%s.pkl' % method, 'rb'))
        all_aus[method] = dict()
        for trial_id, fold_id, re in re_summary:
            for s in re:
                if s not in all_aus[method]:
                    all_aus[method][s] = dict()
                all_aus[method][s][(trial_id, fold_id)] = re[s][(trial_id, fold_id)]['auc_wt']
        for s in all_aus[method].keys():
            print(method, s, np.mean(np.asarray(all_aus[method][s].values())))
    pkl.dump(all_aus, open(data_path + 're_summary_all_aucs.pkl', 'wb'))


def summary_feature_results():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    all_features = dict()
    for method_ind, method in enumerate(['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc']):
        re_summary = pkl.load(open(data_path + 're_%s.pkl' % method, 'rb'))
        all_features[method] = dict()
        for trial_id, fold_id, re in re_summary:
            re_features = set(np.nonzero(re[(trial_id, fold_id)]['wt'])[0])
            inter = set(related_genes.keys()).intersection(re_features)
            all_features[method][(trial_id, fold_id)] = float(len(inter)) / float(len(re_features))
        print(method, np.mean(np.asarray(all_features[method].values())))
    for method_ind, method in enumerate(['sht_am', 'sto_iht', 'hsg_ht']):
        re_summary = pkl.load(open(data_path + 're_%s.pkl' % method, 'rb'))
        all_features[method] = dict()
        for trial_id, fold_id, re in re_summary:
            for s in re:
                if s not in all_features[method]:
                    all_features[method][s] = dict()
                re_features = set(np.nonzero(re[s][(trial_id, fold_id)]['wt'])[0])
                inter = set(related_genes.keys()).intersection(re_features)
                if len(re_features) == 0:
                    all_features[method][s][(trial_id, fold_id)] = 0.0
                else:
                    all_features[method][s][(trial_id, fold_id)] = float(len(inter)) / float(len(re_features))
        for s in all_features[method].keys():
            print(method, s, np.mean(np.asarray(all_features[method][s].values())))
    pkl.dump(all_features, open(data_path + 're_summary_all_features.pkl', 'wb'))


def show_auc():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 8
    fig, ax = plt.subplots(1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    s_list = range(1, 101, 2)
    s_list.extend([120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    all_aucs = pkl.load(open(data_path + 're_summary_all_aucs.pkl'))
    method_list = ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'sht_am', 'sto_iht', 'hsg_ht']
    method_label_list = ['SOLAM', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'FSAUC', 'SHT-AM', 'StoIHT', 'HSG-HT']
    color_list = ['b', 'y', 'k', 'orangered', 'olive', 'r', 'g', 'm']
    for method_ind, method in enumerate(method_list):
        if method in ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc']:
            plt.plot(s_list, [np.mean(all_aucs[method].values())] * len(s_list),
                     label=method_label_list[method_ind], color=color_list[method_ind], linewidth=1.5)
        if method in ['sht_am', 'sto_iht', 'hsg_ht']:
            aucs = []
            for s in s_list:
                aucs.append(np.mean(all_aucs[method][s].values()))
            plt.plot(s_list, aucs, label=method_label_list[method_ind], color=color_list[method_ind], linewidth=1.5)
    ax.legend(loc='center right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('AUC Score')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_colon_auc.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_auc_scores():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 8
    fig, ax = plt.subplots(1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    all_aucs = pkl.load(open(data_path + 're_summary_all_aucs.pkl'))
    method_list = ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'sht_am', 'sto_iht', 'hsg_ht']
    method_label_list = ['SOLAM', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'FSAUC', 'SHT-AM', 'StoIHT', 'HSG-HT']
    color_list = ['b', 'y', 'k', 'c', 'olive', 'r', 'g', 'm']
    s_list = range(1, 101, 2)
    s_list.extend([120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    for method_ind, method in enumerate(method_list):
        if method in ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc']:
            plt.plot(range(100), np.sort(all_aucs[method].values())[::-1],
                     label=method_label_list[method_ind], color=color_list[method_ind], linewidth=1.5)
        if method in ['sht_am', 'sto_iht', 'hsg_ht']:
            best_auc, aucs = None, []
            for s in s_list:
                if best_auc is None or best_auc < np.mean(all_aucs[method][s].values()):
                    best_auc = np.mean(all_aucs[method][s].values())
                    aucs = all_aucs[method][s].values()
            plt.plot(range(100), np.sort(aucs)[::-1],
                     label=method_label_list[method_ind], color=color_list[method_ind], linewidth=1.5)
    plt.plot(range(100), [0.5] * 100, linewidth=1.0, linestyle='--', color='lightgray')
    ax.legend(loc='lower left', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('i-th highest AUC score among 20 trials of 5-fold')
    ax.set_ylabel('AUC Score')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_colon_auc_scores.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_features():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 8, 8
    fig, ax = plt.subplots(1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    s_list = range(1, 101, 2)
    s_list.extend([120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    all_features = pkl.load(open(data_path + 're_summary_all_features.pkl'))
    method_list = ['sht_am', 'sto_iht', 'hsg_ht']
    method_label_list = ['SHT-AM', 'StoIHT', 'HSG-HT']
    color_list = ['r', 'g', 'm']
    for method_ind, method in enumerate(method_list):
        ratio_features = []
        for s in s_list:
            ratio_features.append(np.mean(all_features[method][s].values()))
        plt.plot(s_list, ratio_features, label=method_label_list[method_ind],
                 color=color_list[method_ind], linewidth=1.5)
    ax.legend(loc='center right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Ratio of Related Features')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_colon_features.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def main():
    if sys.argv[1] == 'run':
        method = sys.argv[2]
        data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
        data = pkl.load(open(data_path + 'colon_data.pkl'))
        pool = multiprocessing.Pool(processes=int(sys.argv[3]))
        para_list = []
        for trial_id, fold_id in product(range(data['num_trials']), range(5)):
            para_list.append((data, trial_id, fold_id))
        if method == 'sht_auc':
            ms_res = pool.map(cv_sht_auc, para_list)
        elif method == 'sto_iht':
            ms_res = pool.map(cv_sto_iht, para_list)
        elif method == 'spam_l1':
            ms_res = pool.map(cv_spam_l1, para_list)
        elif method == 'spam_l2':
            ms_res = pool.map(cv_spam_l2, para_list)
        elif method == 'spam_l1l2':
            ms_res = pool.map(cv_spam_l1l2, para_list)
        elif method == 'fsauc':
            ms_res = pool.map(cv_fsauc, para_list)
        elif method == 'hsg_ht':
            ms_res = pool.map(cv_hsg_ht, para_list)
        elif method == 'solam':
            ms_res = pool.map(cv_solam, para_list)
        else:
            ms_res = None
        pool.close()
        pool.join()
        pkl.dump(ms_res, open(data_path + 're_%s.pkl' % method, 'wb'))
    elif sys.argv[1] == 'show':
        if sys.argv[2] == 'auc':
            # summary_auc_results()
            show_auc()
        elif sys.argv[2] == 'features':
            # summary_feature_results()
            show_features()
        elif sys.argv[2] == 'auc_scores':
            show_auc_scores()


if __name__ == '__main__':
    main()
