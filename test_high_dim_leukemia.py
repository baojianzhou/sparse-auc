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
        from sparse_module import c_algo_sht_auc
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
related_genes = {
    # markers for AML
    # 01: Myeloperoxidase
    1778: '"773","1779","MPO Myeloperoxidase","M19507_at"',
    # 02: CD13
    1816: '"792","1817","ANPEP Alanyl (membrane) aminopeptidase (aminopeptidase N, '
          'aminopeptidase M, microsomal aminopeptidase, CD13)","M22324_at"',
    # 03: CD33
    1833: '"808","1834","CD33 CD33 antigen (differentiation antigen)","M23197_at"',
    # 04: HOXA9 Homeo box A9
    3188: '"1391","3189","HOXA9 Homeo box A9","U41813_at"',
    # 05: MYBL2
    4129: '"1788","4130","MYBL2 V-myb avian myeloblastosis viral oncogene homolog-like 2","X13293_at"',
    # markers for ALL
    # 06: CD19
    6224: '"2673","6225","CD19 gene","M84371_rna1_s_at"',
    # 07: CD10 (CALLA)
    1085: '"493","1086","MME Membrane metallo-endopeptidase '
          '(neutral endopeptidase, enkephalinase, CALLA, CD10)","J03779_at"',
    # 08: TCL1 (T cell leukemia)
    4679: '"2065","4680","TCL1 gene (T cell leukemia) extracted from '
          'H.sapiens mRNA for Tcell leukemia/lymphoma 1","X82240_rna1_at"',
    # 09: C-myb
    5771: '"2489","5772","C-myb gene extracted from Human (c-myb) gene, complete primary cds, '
          'and five complete alternatively spliced cds","U22376_cds2_s_at"',
    # 10: Deoxyhypusine synthase
    6514: '"2801","6515","DHPS Deoxyhypusine synthase","U26266_s_at"',
    # 10: Deoxyhypusine synthase
    3763: '"1633","3764","DHPS Deoxyhypusine synthase","U79262_at"',
    # 12: G-gamma globin
    2344: '"1034","2345","G-gamma globin gene extracted from H.sapiens '
          'G-gamma globin and A-gamma globin genes","M91036_rna1_at",',
    # 13: Delta-globin
    6883: '"2945","6884","Delta-globin gene extracted from Human beta'
          ' globin region on chromosome 11","U01317_cds4_at"',
    # 14: Brain-expressed HHCPA78 homolog
    2481: '"1103","2482","Brain-expressed HHCPA78 homolog [human, HL-60 acute '
          'promyelocytic leukemia cells, mRNA, 2704 nt]","S73591_at"',
    # 15:
    6214: '"2669","6215","MPO from  Human myeloperoxidase gene, '
          'exons 1-4./ntype=DNA /annot=exon","M19508_xpt3_s_at"',
    # 16: Probable protein disulfide isomerase ER-60 precursor
    535: '"257","536","PROBABLE PROTEIN DISULFIDE ISOMERASE ER-60 PRECURSOR","D63878_at"',
    # 16: Probable protein disulfide isomerase ER-60 precursor
    5577: '"2415","5578","PROBABLE PROTEIN DISULFIDE ISOMERASE ER-60 PRECURSOR","Z49835_s_at"',
    # 16: Probable protein disulfide isomerase ER-60 precursor
    6167: '"2646","6168","PROBABLE PROTEIN DISULFIDE ISOMERASE ER-60 PRECURSOR","M13560_s_at"',
    # 17: NPM1 Nucleophosmin
    3577: '"1549","3578","NPM1 Nucleophosmin (nucleolar phosphoprotein B23, numatrin)","U66559_at"',
    # 18
    2441: '"1087","2442","CD34 CD34 antigen (hemopoietic progenitor cell antigen)","S53911_at"',
    # 20
    5687: '"2459","5688","CD24 signal transducer mRNA and 3 region","L33930_s_at"',
    # 21
    281: '"124","282","60S RIBOSOMAL PROTEIN L23","D21260_at"',
    # 22
    5224: '"2273","5225","5-aminolevulinic acid synthase gene extracted from Human DNA sequence '
          'from PAC 296K21 on chromosome X contains cytokeratin exon, delta-aminolevulinate synthase '
          '(erythroid); 5-aminolevulinic acid synthase.(EC 2.3.1.37). '
          '6-phosphofructo-2-kinase/fructose-2,6-bisphosphatase (EC 2.7.1.105, EC 3.1.3.46),'
          ' ESTs and STS","Z83821_cds2_at"',
    # 23
    4016: '"1736","4017","HLA CLASS II HISTOCOMPATIBILITY ANTIGEN, DR ALPHA CHAIN PRECURSOR","X00274_at"',
    # 24
    2878: '"1257","2879","Epstein-Barr virus-induced protein mRNA","U19261_at"',
    # 25
    4629: '"2042","4630","HNRPA1 Heterogeneous nuclear ribonucleoprotein A1","X79536_at"',
    # 26
    2401: '"1069","2402","Azurocidin gene","M96326_rna1_at"',
    # 27
    4594: '"2026","4595","Red cell anion exchanger (EPB3, AE1, Band 3) 3 non-coding region","X77737_at"',
    # 28
    5500: '"2386","5501","TOP2B Topoisomerase (DNA) II beta (180kD)","Z15115_at"',
    # 30
    5551: '"2402","5552","PROBABLE G PROTEIN-COUPLED RECEPTOR LCR1 HOMOLOG","L06797_s_at"',
    # 32
    3520: '"1527","3521","Int-6 mRNA","U62962_at"',
    # 33
    1173: '"534","1174","Alpha-tubulin isotype H2-alpha gene, last exon","K03460_at"',
    # 33
    4467: '"1957","4468","Alpha-tubulin mRNA","X68277_at"',
    # 33
    4909: '"2156","4910","Alpha-tubulin mRNA","X99325_at"',
    # 33
    6914: '"2957","6915","Alpha-tubulin mRNA","X01703_at"',
    # 34
    1684: '"738","1685","Terminal transferase mRNA","M11722_at"',
    # 35:
    5951: '"2561","5952","GLYCOPHORIN B PRECURSOR","U05255_s_at"'
}


def process_data_21_leukemia():
    """
    https://github.com/ramhiser/datamicroarray
    http://genomics-pubs.princeton.edu/oncology/affydata/index.html
    :return:
    """
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
    data = {'feature_ids': None, 'x_tr': [], 'y_tr': [], 'feature_names': []}
    import csv
    with open(data_path + 'golub_x.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                data['feature_ids'] = [str(_) for _ in row[1:]]
                line_count += 1
            elif 1 <= line_count <= 72:
                data['x_tr'].append([float(_) for _ in row[1:]])
                line_count += 1
        data['x_tr'] = np.asarray(data['x_tr'])
        for i in range(len(data['x_tr'])):
            data['x_tr'][i] = data['x_tr'][i] / np.linalg.norm(data['x_tr'][i])
    # AML: 急性粒细胞白血病 ALL:急性淋巴细胞白血病
    with open(data_path + 'golub_y.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            elif 1 <= line_count <= 72:
                line_count += 1
                if row[1] == 'ALL':
                    data['y_tr'].append(1.)
                else:
                    data['y_tr'].append(-1.)
        data['y_tr'] = np.asarray(data['y_tr'])
    data['n'] = 72
    data['p'] = 7129
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
    pkl.dump(data, open(data_path + 'leukemia_data.pkl', 'wb'))


def cv_sht_am(para):
    data, trial_id, fold_id = para
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 1e2, 0, 1, 1e-6
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    all_results = dict()
    s_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    for para_s in s_list:
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        results = dict()
        best_b, best_auc = None, None
        for para_b in range(1, 40, 1):
            wt, _, _, _ = c_algo_sht_auc(
                x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1., 0.0)
            auc_score = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
            if best_b is None or best_auc is None or best_auc < auc_score:
                best_b, best_auc = para_b, auc_score
        tr_index = data['trial_%d' % trial_id]['tr_index']
        te_index = data['trial_%d' % trial_id]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        wt, aucs, rts, epochs = c_algo_sht_auc(x_tr, __, __, __, y_tr, 0, data['p'], global_paras,
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
    s_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    all_results = dict()
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
    s_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
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
    for para_xi, para_r, in product(np.arange(1, 101, 9, dtype=float), 10. ** np.arange(-1, 6, 1, dtype=float)):
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
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
    data = pkl.load(open(data_path + 'leukemia_data.pkl'))
    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    para_list = []
    for trial_id, fold_id in product(range(data['num_trials']), range(5)):
        para_list.append((data, trial_id, fold_id))
    if method == 'sht_am':
        ms_res = pool.map(cv_sht_am, para_list)
    elif method == 'sto_iht':
        ms_res = pool.map(cv_sto_iht, para_list)
    elif method == 'hsg_ht':
        ms_res = pool.map(cv_hsg_ht, para_list)
    elif method == 'spam_l1':
        ms_res = pool.map(cv_spam_l1, para_list)
    elif method == 'spam_l2':
        ms_res = pool.map(cv_spam_l2, para_list)
    elif method == 'spam_l1l2':
        ms_res = pool.map(cv_spam_l1l2, para_list)
    elif method == 'fsauc':
        ms_res = pool.map(cv_fsauc, para_list)
    elif method == 'solam':
        ms_res = pool.map(cv_solam, para_list)
    else:
        ms_res = None
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(data_path + 're_%s.pkl' % method, 'wb'))


def summary_auc_results():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
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
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
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
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
    s_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    all_aucs = pkl.load(open(data_path + 're_summary_all_aucs.pkl'))
    method_list = ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'sht_am', 'sto_iht', 'hsg_ht']
    method_label_list = ['SOLAM', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'FSAUC', 'SHT-AM', 'StoIHT', 'HSG-HT']
    color_list = ['b', 'y', 'k', 'orangered', 'olive', 'r', 'g', 'm']
    for method_ind, method in enumerate(method_list):
        if method in ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc']:
            print(method, '%.4f %.4f' % (np.mean(all_aucs[method].values()), np.std(all_aucs[method].values())))
            plt.plot(s_list, [np.mean(all_aucs[method].values())] * len(s_list),
                     label=method_label_list[method_ind], color=color_list[method_ind], linewidth=1.5)
        if method in ['sht_am', 'sto_iht', 'hsg_ht']:
            aucs, aucs_std = [], []
            for s in s_list:
                aucs.append(np.mean(all_aucs[method][s].values()))
                aucs_std.append(np.std(all_aucs[method][s].values()))
            import operator
            index, value = max(enumerate(aucs), key=operator.itemgetter(1))
            print(method, '%.4f %.4f' % (value, aucs_std[index]))
            plt.plot(s_list, aucs, label=method_label_list[method_ind], color=color_list[method_ind], linewidth=1.5)
    ax.legend(loc='center right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('AUC Score')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_leukemia_auc.pdf'
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
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
    all_aucs = pkl.load(open(data_path + 're_summary_all_aucs.pkl'))
    method_list = ['solam', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'sht_am', 'sto_iht', 'hsg_ht']
    method_label_list = ['SOLAM', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'FSAUC', 'SHT-AM', 'StoIHT', 'HSG-HT']
    color_list = ['b', 'y', 'k', 'c', 'olive', 'r', 'g', 'm']
    s_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
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
    f_name = root_path + 'real_leukemia_auc_scores.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_figure3_b():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 5, 4
    fig, ax = plt.subplots(1, 1)
    ax.grid(color='lightgray', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/21_leukemia/'
    s_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    all_features = pkl.load(open(data_path + 're_summary_all_features.pkl'))
    method_list = ['sht_am', 'sto_iht', 'hsg_ht']
    method_label_list = ['SHT-AUC', 'StoIHT', 'HSG-HT']
    color_list = ['r', 'b', 'g']
    marker_list = ['D', 's', 'o']
    for method_ind, method in enumerate(method_list):
        ratio_features = []
        for s in s_list:
            ratio_features.append(np.mean(all_features[method][s].values()))
        plt.plot(s_list, ratio_features, label=method_label_list[method_ind], marker=marker_list[method_ind],
                 markerfacecolor='w', color=color_list[method_ind], linewidth=1.5, markersize=5.)
    ax.legend(loc='upper right', framealpha=1., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('Sparsity $\displaystyle k$')
    ax.set_ylabel('Ratio of Selected Genes')
    ax.set_xticks([100, 200, 300, 400, 500])
    ax.set_xticklabels([100, 200, 300, 400, 500])
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'real_leukemia_features.pdf'
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
    elif sys.argv[1] == 'run_fsauc':
        run_methods(method='fsauc')
    elif sys.argv[1] == 'run_hsg_ht':
        run_methods(method='hsg_ht')
    elif sys.argv[1] == 'run_solam':
        run_methods(method='solam')
    elif sys.argv[1] == 'show_auc':
        # summary_auc_results()
        show_auc()
    elif sys.argv[1] == 'show_figure3_b':
        # summary_feature_results()
        show_figure3_b()
    elif sys.argv[1] == 'show_auc_scores':
        show_auc_scores()


if __name__ == '__main__':
    main()
