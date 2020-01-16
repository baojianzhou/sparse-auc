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


def cv_sht_am(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    all_results = dict()
    for para_s in range(1, 101, 3):
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
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
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
    for para_s in range(1, 101, 3):
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
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt}
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
    for para_s in range(1, 101, 3):
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        results = dict()
        best_c, best_auc = None, None
        for para_c in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]:
            para_tau, para_zeta = 1.0, 1.033
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
                                                                y_score=np.dot(data['x_tr'][te_index], wt)),
                                        'aucs': aucs, 'rts': rts, 'wt': wt}
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


def run_methods(method):
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    data = pkl.load(open(data_path + 'colon_data.pkl'))
    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    para_list = []
    for trial_id, fold_id in product(range(data['num_trials']), range(5)):
        para_list.append((data, trial_id, fold_id))
    if method == 'sht_am':
        ms_res = pool.map(cv_sht_am, para_list)
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


def preprocess_results():
    results = dict()
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/20_colon/'
    for method in ['sht_am', 'sto_iht', 'hsg_ht', 'fsauc']:
        print(method)
        results[method] = []
        re = pkl.load(open(data_path + 're_%s.pkl' % method))
        for item in sorted(re):
            results[method].append({'auc': None, 'nonzeros': None})
            results[method][-1]['auc'] = {_: item[1][_]['auc_wt'] for _ in item[1]}
            results[method][-1]['nonzeros'] = {_: np.nonzero(item[1][_]['wt'])[0] for _ in item[1]}
    for method in ['spam_l1', 'spam_l2', 'spam_l1l2', 'solam']:
        print(method)
        results[method] = []
        re = pkl.load(open(data_path + 're_%s.pkl' % method))
        for item in sorted(re)[::-1]:
            results[method].append({'auc': None, 'nonzeros': None})
            results[method][-1]['auc'] = {_: item[1][_]['auc_wt'] for _ in item[1]}
            results[method][-1]['nonzeros'] = {_: np.nonzero(item[1][_]['wt'])[0] for _ in item[1]}
    pkl.dump(results, open(data_path + 're_summary.pkl', 'wb'))


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
    re_summary = pkl.load(open(data_path + 're_summary.pkl', 'rb'))
    color_list = ['r', 'g', 'm', 'b', 'y', 'k', 'orangered', 'olive', 'blue', 'darkgray', 'darkorange']
    marker_list = ['s', 'o', 'P', 'X', 'H', '*', 'x', 'v', '^', '+', '>']
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'fsauc', 'spam_l1l2', 'solam', 'sto_iht', 'hsg_ht']
    method_label_list = ['SHT-AUC', r"SPAM-$\displaystyle \ell^1$", r"SPAM-$\displaystyle \ell^2$",
                         'FSAUC', r"SPAM-$\displaystyle \ell^1/\ell^2$", r"SOLAM", r"StoIHT", 'HSG-HT']
    for method_ind, method in enumerate(method_list):
        plt.plot(range(10, 101, 5),
                 [float(np.mean(np.asarray([_['auc'][key] for key in _['auc']]))) for _ in re_summary[method]],
                 label=method_label_list[method_ind], color=color_list[method_ind],
                 marker=marker_list[method_ind], linewidth=2.)
    ax.legend(loc='center right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel('Sparse parameter')
    ax.set_ylabel('AUC Score')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_colon_auc.pdf'
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
    re_summary = pkl.load(open(data_path + 're_summary.pkl', 'rb'))
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'fsauc', 'spam_l1l2', 'solam', 'sto_iht', 'hsg_ht']
    summary_genes = dict()
    summary_len = dict()
    for method_ind, method in enumerate(method_list):
        summary_genes[method] = []
        summary_len[method] = []
        for each_para in re_summary[method]:
            re = []
            re_len = []
            for trial_i in range(20):
                selected_genes = []
                for (_, fold_i) in each_para['nonzeros']:
                    if _ == trial_i:
                        selected_genes.extend(each_para['nonzeros'][(trial_i, fold_i)])
                re.extend(list(set(selected_genes).intersection(set(related_genes.keys()))))
                re_len.extend(selected_genes)
            summary_genes[method].append(set(re))
            summary_len[method].append(set(re_len))
    ratio_sht_am = []
    ratio_sto_iht = []
    ratio_hsg_ht = []
    for s_ind, para_s in enumerate(range(10, 101, 5)):
        print(para_s),
        for method in ['sto_iht', 'hsg_ht', 'sht_am']:
            x1 = len(summary_genes[method][s_ind])
            x2 = len(summary_len[method][s_ind])
            if method == 'sht_am':
                ratio_sht_am.append(float(x1) / float(x2))
            elif method == 'sto_iht':
                ratio_sto_iht.append(float(x1) / float(x2))
            elif method == 'hsg_ht':
                ratio_hsg_ht.append(float(x1) / float(x2))
            print('%02d/%03d-%.4f' % (x1, x2, float(x1) / float(x2))),
        print('')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(range(10, 101, 5), ratio_sht_am, label='SHT-AUC', marker='D', color='r',
             linewidth=2., markersize=10., markerfacecolor='white', markeredgewidth=2.)
    plt.plot(range(10, 101, 5), ratio_hsg_ht, label='HSG-HT', marker='P', color='b',
             linewidth=2., markersize=10., markerfacecolor='white', markeredgewidth=2.)
    plt.plot(range(10, 101, 5), ratio_sto_iht, label='StoIHT', marker='X', color='g',
             linewidth=2., markersize=10., markerfacecolor='white', markeredgewidth=2.)
    ax.legend(loc='center right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('Sparse parameter')
    ax.set_ylabel('Percentage of related genes')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_colon_feature.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def main():
    if sys.argv[1] == 'run_sht_am':
        run_methods(method='sht_am')
    elif sys.argv[1] == 'run_sto_iht':
        run_methods(method='sto_iht')
    elif sys.argv[1] == 'run_spam_l1':
        run_methods(method='spam_l1')
    elif sys.argv[1] == 'run_spam_l2':
        run_methods(method='spam_l2')
    elif sys.argv[1] == 'run_spam_l1l2':
        run_methods(method='spam_l1l2')
    elif sys.argv[1] == 'run_fsauc':
        run_methods(method='fsauc')
    elif sys.argv[1] == 'run_hsg_ht':
        run_methods(method='hsg_ht')
    elif sys.argv[1] == 'run_solam':
        run_methods(method='solam')
    elif sys.argv[1] == 'show_auc':
        show_auc()
    elif sys.argv[1] == 'show_features':
        show_features()
    elif sys.argv[1] == 'summary':
        preprocess_results()


if __name__ == '__main__':
    main()
