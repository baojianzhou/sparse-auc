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
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_solam
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_sht_am
        from sparse_module import c_algo_graph_am_v1
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


def get_example_data_non_sparse():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'
    trial_id, num_tr, mu, posi_ratio, fold_id = 0, 1000, 0.3, 0.5, 0
    data = pkl.load(open(data_path + 'data_trial_00_tr_1000_mu_0.3_p-ratio_0.50.pkl', 'rb'))['fig_1']
    tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
    te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
    x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
    y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
    x_te = np.asarray(data['x_tr'][te_index], dtype=float)
    y_te = np.asarray(data['y_tr'][te_index], dtype=float)
    __ = np.empty(shape=(1,), dtype=float)
    is_sparse = 0
    return x_tr, y_tr, x_te, y_te, data['p'], __, is_sparse


def get_data_by_ind(data, tr_ind, sub_tr_ind):
    sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens = [], [], [], []
    prev_posi = 0
    for index in tr_ind[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_poss'][index]
        sub_x_vals.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_inds.extend(data['x_tr_inds'][cur_posi:cur_posi + cur_len])
        sub_x_lens.append(cur_len)
        sub_x_poss.append(prev_posi)
        prev_posi += cur_len
    sub_x_vals = np.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_poss = np.asarray(sub_x_poss, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data['y_tr'][tr_ind[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr


def pred_auc(data, tr_index, sub_te_ind, wt):
    if np.isnan(wt).any() or np.isinf(wt).any():  # not a valid score function.
        return 0.0
    _ = get_data_by_ind(data, tr_index, sub_te_ind)
    sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_te = _
    y_pred_wt = np.zeros_like(sub_te_ind, dtype=float)
    for i in range(len(sub_te_ind)):
        cur_posi = sub_x_posis[i]
        cur_len = sub_x_lens[i]
        cur_x = sub_x_vals[cur_posi:cur_posi + cur_len]
        cur_ind = sub_x_inds[cur_posi:cur_posi + cur_len]
        y_pred_wt[i] = np.sum([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
    return roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)


def test_non_sparse():
    # data
    x_tr, y_tr, x_te, y_te, p, __, is_sparse = get_example_data_non_sparse()
    # common parameters
    num_passes, step_len, verbose, record_aucs, stop_eps = 50, 100, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    # run algorithm

    if False:
        best_auc = None
        for para_xi, para_r in product(np.arange(1, 51, 9, dtype=float), 10 ** np.arange(-2, 2, 1, dtype=float)):
            re = c_algo_solam(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, para_xi, para_r)
            wt, wt_bar, aucs, rts = re
            auc_score = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
            if best_auc is None or best_auc[-1] < auc_score:
                print(para_xi, para_r, aucs[-1], auc_score)
                best_auc = [aucs, rts, para_xi, para_r, auc_score]
        plt.plot(best_auc[1], best_auc[0], label='SOLAM')
        best_auc = None
        for para_xi, para_l1 in product(10. ** np.arange(-5, 2, 1, dtype=float),
                                        10. ** np.arange(-5, 0, 1, dtype=float)):
            re = c_algo_spam(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, para_xi, para_l1, 0.0)
            wt, wt_bar, aucs, rts = re
            auc_score = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
            if best_auc is None or best_auc[-1] < auc_score:
                print(para_xi, para_l1, aucs[-1], auc_score)
                best_auc = [aucs, rts, para_xi, para_l1, auc_score]
        plt.plot(best_auc[1], best_auc[0], label='SPAM-L1')
        best_auc = None
        for para_xi, para_l2 in product(10. ** np.arange(-5, 3, 1, dtype=float),
                                        10. ** np.arange(-5, 3, 1, dtype=float)):
            re = c_algo_spam(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, para_xi, 0.0, para_l2)
            wt, wt_bar, aucs, rts = re
            auc_score = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
            if best_auc is None or best_auc[-1] < auc_score:
                print(para_xi, para_l2, aucs[-1], auc_score)
                best_auc = [aucs, rts, para_xi, para_l2, auc_score]
        plt.plot(best_auc[1], best_auc[0], label='SPAM-L2')
        best_auc = None
        for para_xi, para_l1, para_l2 in product(10. ** np.arange(-5, 3, 1, dtype=float),
                                                 10. ** np.arange(-5, 3, 1, dtype=float),
                                                 10. ** np.arange(-5, 3, 1, dtype=float)):
            re = c_algo_spam(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, para_xi, para_l1, para_l2)
            wt, wt_bar, aucs, rts = re
            auc_score = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
            if best_auc is None or best_auc[-1] < auc_score:
                print(para_xi, para_l2, aucs[-1], auc_score)
                best_auc = [aucs, rts, para_xi, para_l2, auc_score]
        plt.plot(best_auc[1], best_auc[0], label='SPAM-L1L2')
        best_auc = None
        for para_s, para_b, para_c in product(range(40, 60, 2), [100], 10. ** np.arange(-3, 3, 1, dtype=float)):
            re = c_algo_sht_am_v1(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, para_s, para_b, para_c, 0.0)
            wt, wt_bar, aucs, rts = re
            auc_score = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
            if best_auc is None or best_auc[-1] < auc_score:
                print(para_s, para_b, para_c, aucs[-1], auc_score)
                best_auc = [aucs, rts, para_s, para_b, para_c, auc_score]
        plt.plot(best_auc[1], best_auc[0], label='SHT-AM-V1')
        best_auc = None
        for para_s, para_b, para_c in product(range(40, 50, 2), [800], 10. ** np.arange(-3, 3, 1, dtype=float)):
            re = c_algo_sht_am_v2(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, para_s, para_b, para_c, 0.0)
            wt, wt_bar, aucs, rts = re
            auc_score = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
            if best_auc is None or best_auc[-1] < auc_score:
                print(para_s, para_b, para_c, aucs[-1], auc_score)
                best_auc = [aucs, rts, para_s, para_b, para_c, auc_score]
        plt.plot(best_auc[1], best_auc[0], label='SHT-AM-V2')
    for para_s, para_b, para_c in product([26], [200], 10. ** np.arange(-3, 3, 1, dtype=float)):
        re = c_algo_sht_am(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, 0, para_s, para_b, para_c, 0.0)
        wt, wt_bar, aucs, rts = re
        plt.plot(rts, aucs, label='SHT-AM-V1')
        auc_score_0 = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
        re = c_algo_sht_am(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, 1, para_s, para_b, para_c, 0.0)
        wt, wt_bar, aucs, rts = re
        plt.plot(rts, aucs, label='SHT-AM-V2')
        auc_score_1 = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
        re = c_algo_sht_am(x_tr, __, __, __, y_tr, is_sparse, p, global_paras, 2, para_s, para_b, para_c, 0.0)
        wt, wt_bar, aucs, rts = re
        plt.plot(rts, aucs, label='SHT-AM-V3')
        auc_score_2 = roc_auc_score(y_true=y_te, y_score=np.dot(x_te, wt))
        print(auc_score_0, auc_score_1, auc_score_2)
        plt.legend()
        plt.show()
    exit()
    plt.legend()
    plt.show()
    plt.close()


def test_solam_sparse():
    # data
    f_name = os.path.join('/network/rit/lab/ceashpc/bz383376/data/icml2020/', '01_pcmac/data_run_0.pkl')
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_0']['tr_index']
    te_index = data['fold_0']['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    is_sparse = 1
    # common parameters
    num_passes, step_len, verbose, record_aucs, stop_eps = 50, 100, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    best_auc = None
    for para_xi, para_r in product(np.arange(1, 101, 9, dtype=float), 10 ** np.arange(-1, 6, 1, dtype=float)):
        re = c_algo_solam(x_vals, x_inds, x_poss, x_lens, y_tr, is_sparse, data['p'], global_paras, para_xi, para_r)
        wt, wt_bar, aucs, rts = re
        auc_score = pred_auc(data, te_index, range(len(te_index)), wt)
        print(para_xi, para_r, aucs[-1], auc_score)
        if best_auc is None or best_auc[-1] < auc_score:
            best_auc = [wt, wt_bar, aucs, rts, para_xi, para_r, auc_score]
    wt, wt_bar, aucs, rts, para_xi, para_r, auc_score = best_auc
    print(para_xi, para_r, auc_score)
    import matplotlib.pyplot as plt
    plt.plot(rts, aucs)
    plt.plot(rts, [1.] * len(rts), color='lightgray', linestyle='dotted')
    plt.show()
    plt.close()


def test_sht_am_sparse():
    # data
    f_name = os.path.join('/network/rit/lab/ceashpc/bz383376/data/icml2020/', '01_pcmac/data_run_0.pkl')
    f_name = os.path.join('/network/rit/lab/ceashpc/bz383376/data/icml2020/', '10_farmads/data_run_0.pkl')
    data = pkl.load(open(f_name, 'rb'))
    tr_index = data['fold_0']['tr_index']
    te_index = data['fold_0']['te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    is_sparse, p = 1, data['p']
    # common parameters
    num_passes, step_len, verbose, record_aucs, stop_eps = 100, 100, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for para_s, para_b, para_c in product([10000], [100], 10. ** np.arange(0, 1, 1, dtype=float)):
        re = c_algo_sht_am(x_vals, x_inds, x_poss, x_lens, y_tr, is_sparse, p,
                           global_paras, 0, para_s, para_b, para_c, 0.0)
        wt, wt_bar, aucs, rts = re
        plt.plot(rts, aucs, label='SHT-AM-V1')
        auc_score_0 = pred_auc(data, te_index, range(len(te_index)), wt)
        re = c_algo_sht_am(x_vals, x_inds, x_poss, x_lens, y_tr, is_sparse, p,
                           global_paras, 1, para_s, para_b, para_c, 0.0)
        wt, wt_bar, aucs, rts = re
        plt.plot(rts, aucs, label='SHT-AM-V2')
        auc_score_1 = pred_auc(data, te_index, range(len(te_index)), wt)
        re = c_algo_sht_am(x_vals, x_inds, x_poss, x_lens, y_tr, is_sparse, p,
                           global_paras, 2, para_s, para_b, para_c, 0.0)
        wt, wt_bar, aucs, rts = re
        plt.plot(rts, aucs, label='SHT-AM-V3')
        auc_score_2 = pred_auc(data, te_index, range(len(te_index)), wt)
        print(auc_score_0, auc_score_1, auc_score_2)
        plt.legend()
        plt.show()


def main():
    # test_non_sparse()
    test_sht_am_sparse()


if __name__ == '__main__':
    main()
