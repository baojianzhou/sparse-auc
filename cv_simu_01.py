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

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'


def node_pre_rec_fm(true_nodes, pred_nodes):
    """ Return the precision, recall and f-measure.
    :param true_nodes:
    :param pred_nodes:
    :return: precision, recall and f-measure """
    true_nodes, pred_nodes = set(true_nodes), set(pred_nodes)
    pre, rec, fm = 0.0, 0.0, 0.0
    if len(pred_nodes) != 0:
        pre = len(true_nodes & pred_nodes) / float(len(pred_nodes))
    if len(true_nodes) != 0:
        rec = len(true_nodes & pred_nodes) / float(len(true_nodes))
    if (pre + rec) > 0.:
        fm = (2. * pre * rec) / (pre + rec)
    return [pre, rec, fm]


def cv_solam(para):
    """ SOLAM algorithm. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_xi = np.arange(1, 101, 9, dtype=float)
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_xi), len(list_r)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_r, para_r) in product(range(k_fold), enumerate(list_xi), enumerate(list_r)):
        s_time = time.time()
        algo_para = (para_xi, para_r, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_solam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_r)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_r] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
            auc_wt[(trial_id, fold_id)]['epochs'] = float(np.mean(list_epochs))
        print("trial-%d fold-%d para_xi: %.1e para_r: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_r, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_solam(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_solam.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_xi, para_r, _ = ms[para]['solam']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_solam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_r)
        wt, aucs, rts, epochs = _
        # indices = np.argsort(np.abs(wt))[::-1]
        wt = np.asarray(wt)
        wt[np.where(np.abs(wt) < 1e-3)] = 0.0
        indices = np.nonzero(wt)[0]
        xx = set(indices).intersection(set(data['subset']))
        if float(len(indices)) != 0.0:
            pre = float(len(xx)) / float(len(indices))
        else:
            pre = 0.0
        rec = float(len(xx)) / float(len(data['subset']))
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
        results[item] = {'algo_para': [trial_id, fold_id, s, para_xi, para_r],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_xi:%.4f para_r:%.4f' %
              (trial_id, fold_id, s, posi_ratio, results[item]['auc_wt'], para_xi, para_r))
    sys.stdout.flush()
    return results


def cv_spam_l1(para):
    """ SPAM algorithm with l1-regularization. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_c), len(list_l1)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_l1, para_l1) in product(range(k_fold), enumerate(list_c), enumerate(list_l1)):
        s_time = time.time()
        algo_para = (para_xi, para_l1, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_l1, 0.0)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_l1] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
            auc_wt[(trial_id, fold_id)]['epochs'] = float(np.mean(list_epochs))
        print("trial-%d fold-%d para_xi: %.1e para_l1: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_l1, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_spam_l1(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_spam_l1.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_xi, para_l1, _ = ms[para]['spam_l1']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_l1, 0.0)
        wt, aucs, rts, epochs = _
        wt = np.asarray(wt)
        wt[np.where(np.abs(wt) < 1e-3)] = 0.0
        indices = np.nonzero(wt)[0]
        xx = set(indices).intersection(set(data['subset']))
        if float(len(indices)) != 0.0:
            pre = float(len(xx)) / float(len(indices))
        else:
            pre = 0.0
        rec = float(len(xx)) / float(len(data['subset']))
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
        results[item] = {'algo_para': [trial_id, fold_id, s, para_xi, para_l1],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_xi:%.4f para_l1:%.4f' %
              (trial_id, fold_id, s, posi_ratio, results[item]['auc_wt'], para_xi, para_l1))
    sys.stdout.flush()
    return results


def cv_spam_l2(para):
    """ SPAM algorithm with l2-regularization. """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_c), len(list_l2)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_l2, para_l2) in product(range(k_fold), enumerate(list_c), enumerate(list_l2)):
        s_time = time.time()
        algo_para = (para_xi, para_l2, (trial_id, fold_id, s, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_l2] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
            auc_wt[(trial_id, fold_id)]['epochs'] = float(np.mean(list_epochs))
        print("trial-%d fold-%d para_xi: %.1e para_l2: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_l2, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_spam_l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_spam_l2.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_xi, para_l2, _ = ms[para]['spam_l2']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
        wt, aucs, rts, epochs = _
        wt = np.asarray(wt)
        wt[np.where(np.abs(wt) < 1e-3)] = 0.0
        indices = np.nonzero(wt)[0]
        xx = set(indices).intersection(set(data['subset']))
        if float(len(indices)) != 0.0:
            pre = float(len(xx)) / float(len(indices))
        else:
            pre = 0.0
        rec = float(len(xx)) / float(len(data['subset']))
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
        results[item] = {'algo_para': [trial_id, fold_id, s, para_xi, para_l2],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_xi:%.4f para_l2:%.4f' %
              (trial_id, fold_id, s, posi_ratio, results[item]['auc_wt'], para_xi, para_l2))
    sys.stdout.flush()
    return results


def cv_spam_l1l2(para):
    """SPAM algorithm with elastic-net """
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l1 = 10. ** np.arange(-5, 3, 1, dtype=float)
    list_l2 = 10. ** np.arange(-5, 3, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_c), len(list_l1), len(list_l2)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_xi, para_xi), (ind_l1, para_l1), (ind_l2, para_l2) in \
            product(range(k_fold), enumerate(list_c), enumerate(list_l1), enumerate(list_l2)):
        s_time = time.time()
        algo_para = (para_xi, para_l1, para_l2, (trial_id, fold_id, s, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_l1, para_l2)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_xi, ind_l1, ind_l2] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d para_xi: %.1e para_l2: %.1e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_xi, para_l2, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_spam_l1l2(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_spam_l1l2.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_xi, para_l1, para_l2, _ = ms[para]['spam_l1l2']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_spam(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_xi, para_l1, para_l2)
        wt, aucs, rts, epochs = _
        wt = np.asarray(wt)
        wt[np.where(np.abs(wt) < 1e-3)] = 0.0
        indices = np.nonzero(wt)[0]
        xx = set(indices).intersection(set(data['subset']))
        if float(len(indices)) != 0.0:
            pre = float(len(xx)) / float(len(indices))
        else:
            pre = 0.0
        rec = float(len(xx)) / float(len(data['subset']))
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
        results[item] = {'algo_para': [trial_id, fold_id, s, para_xi, para_l1, para_l2],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_xi:%.4f para_l1:%.4f para_l2:%.4f' %
              (trial_id, fold_id, s, posi_ratio, results[item]['auc_wt'], para_xi, para_l1, para_l2))
    sys.stdout.flush()
    return results


def cv_fsauc(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_r = 10. ** np.arange(-1, 6, 1, dtype=float)
    list_g = 2. ** np.arange(-10, 11, 1, dtype=float)
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_r), len(list_g)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_r, para_r), (ind_g, para_g) in product(range(k_fold), enumerate(list_r), enumerate(list_g)):
        s_time = time.time()
        algo_para = (para_r, para_g, (trial_id, fold_id, s, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_fsauc(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_r, para_g)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_r, ind_g] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d para_r: %.1e para_g: %.3e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_r, para_g, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_fsauc(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_fsauc.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_r, para_g, _ = ms[para]['fsauc']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_fsauc(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_r, para_g)
        wt, aucs, rts, epochs = _
        wt = np.asarray(wt)
        wt[np.where(np.abs(wt) < 1e-3)] = 0.0
        indices = np.nonzero(wt)[0]
        xx = set(indices).intersection(set(data['subset']))
        if float(len(indices)) != 0.0:
            pre = float(len(xx)) / float(len(indices))
        else:
            pre = 0.0
        rec = float(len(xx)) / float(len(data['subset']))
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
        results[item] = {'algo_para': [trial_id, fold_id, s, para_r, para_g],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_r:%.4f para_g:%.4f' %
              (trial_id, fold_id, s, posi_ratio, results[item]['auc_wt'], para_r, para_g))
    sys.stdout.flush()
    return results


def cv_sht_am(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_s, list_b = range(10, 101, 10), [640 / _ for _ in [1, 2, 4, 8, 10]]
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_s), len(list_b)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_s, para_s), (ind_b, para_b) in product(range(k_fold), enumerate(list_s), enumerate(list_b)):
        s_time = time.time()
        algo_para = (para_s, para_b, (trial_id, fold_id, s, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_sht_am(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, 0,
                              para_s, para_b, 1.0, 0.1)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_s, ind_b] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d s: %d para_s:%03d para_b:%03d auc:%.4f epochs:%02d run_time: %.6f" %
              (trial_id, fold_id, s, para_s, para_b, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_sht_am(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_sht_am.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_s, para_b, _ = ms[para]['sht_am']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1.0, 0.0)
        wt, aucs, rts, epochs = _
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
        xx = set(np.nonzero(wt)[0]).intersection(set(data['subset']))
        pre, rec = float(len(xx)) * 1. / float(len(np.nonzero(wt)[0])), float(len(xx)) / float(len(data['subset']))
        results[item] = {'algo_para': [trial_id, fold_id, s, para_s, para_b],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d p-ratio:%.2f s: %d para_s: %d para_b: %d auc: %.4f para_s:%03d para_b:%03d' %
              (trial_id, fold_id, posi_ratio, s, para_s, para_b, results[item]['auc_wt'], para_s, para_b))
    sys.stdout.flush()
    return results


def cv_sto_iht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    # get data
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_s, list_b = range(10, 101, 10), [640 / _ for _ in [1, 2, 4, 8, 10]]
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_s), len(list_b)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_s, para_s), (ind_b, para_b) in product(range(k_fold), enumerate(list_s), enumerate(list_b)):
        s_time = time.time()
        algo_para = (para_s, para_b, (trial_id, fold_id, fig_i, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_sto_iht(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_s, para_b, 1., 0.0)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_s, ind_b] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d para_s: %03d para_b: %03d auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_s, para_b, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_sto_iht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_sto_iht.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_s, para_b, _ = ms[para]['sto_iht']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, para_b, 1., 0.0)
        wt, aucs, rts, epochs = _
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        xx = set(np.nonzero(wt)[0]).intersection(set(data['subset']))
        pre, rec = float(len(xx)) * 1. / float(len(np.nonzero(wt)[0])), float(len(xx)) / float(len(data['subset']))
        results[item] = {'algo_para': [trial_id, fold_id, para_s, para_b],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_s:%03d para_b:%03d' %
              (trial_id, fold_id, fig_i, posi_ratio, results[item]['auc_wt'], para_s, para_b))
    sys.stdout.flush()
    return results


def cv_hsg_ht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_s = range(10, 101, 10)
    list_tau = [1., 10., 100., 1000.]
    auc_wt, cv_wt_results = dict(), np.zeros((len(list_s), len(list_tau)))
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id, (ind_s, para_s), (ind_c, para_tau) in product(range(k_fold), enumerate(list_s), enumerate(list_tau)):
        s_time = time.time()
        algo_para = (para_s, para_tau, (trial_id, fold_id, s, num_passes, posi_ratio, stop_eps))
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_epochs = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            para_c, para_zeta = 3.0, 1.033
            _ = c_algo_hsg_ht(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras,
                              para_s, para_tau, para_zeta, para_c, 0.0)
            wt, aucs, rts, epochs = _
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_epochs[ind] = epochs[0]
        cv_wt_results[ind_s, ind_c] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        print("trial-%d fold-%d para_s: %03d para_c: %.3e auc: %.4f epochs: %02d run_time: %.6f" %
              (trial_id, fold_id, para_s, para_tau, float(np.mean(list_auc_wt)),
               float(np.mean(list_epochs)), time.time() - s_time))
    sys.stdout.flush()
    return para, auc_wt, cv_wt_results


def test_hsg_ht(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    ms = pkl.load(open(data_path + 'ms_00_05_hsg_ht.pkl', 'rb'))
    results = dict()
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_s, para_tau, _ = ms[para]['hsg_ht']['auc_wt'][(trial_id, fold_id)]['para']
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        para_c, para_zeta = 3.0, 1.033
        _ = c_algo_hsg_ht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, para_tau, para_zeta, para_c, 0.0)
        wt, aucs, rts, epochs = _
        item = (trial_id, fold_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
        xx = set(np.nonzero(wt)[0]).intersection(set(data['subset']))
        pre, rec = float(len(xx)) * 1. / float(len(np.nonzero(wt)[0])), float(len(xx)) / float(len(data['subset']))
        results[item] = {'algo_para': [trial_id, fold_id, para_s, para_tau],
                         'auc_wt': roc_auc_score(y_true=data['y_tr'][te_index],
                                                 y_score=np.dot(data['x_tr'][te_index], wt)),
                         'f1_score': 2. * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0,
                         'aucs': aucs, 'rts': rts, 'wt': wt, 'nonzero_wt': np.count_nonzero(wt)}
        print('trial-%d fold-%d %s p-ratio:%.2f auc: %.4f para_s:%03d para_c:%.2e' %
              (trial_id, fold_id, fig_i, posi_ratio, results[item]['auc_wt'], para_s, para_tau))
    sys.stdout.flush()
    return results


def run_ms(method_name, trial_id_low, trial_id_high, num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 5, 50, [1000], [0.3]
    posi_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    s_list = [20, 40, 60, 80]
    para_space, ms_res, results = [], [], dict()
    for trial_id in range(trial_id_low, trial_id_high):
        for s in s_list:
            for num_tr, mu, posi_ratio in product(tr_list, mu_list, posi_ratio_list):
                para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s))
    pool = multiprocessing.Pool(processes=num_cpus)
    if method_name == 'solam':
        ms_res = pool.map(cv_solam, para_space)
    elif method_name == 'spam_l1':
        ms_res = pool.map(cv_spam_l1, para_space)
    elif method_name == 'spam_l2':
        ms_res = pool.map(cv_spam_l2, para_space)
    elif method_name == 'spam_l1l2':
        ms_res = pool.map(cv_spam_l1l2, para_space)
    elif method_name == 'fsauc':
        ms_res = pool.map(cv_fsauc, para_space)
    elif method_name == 'sht_am':
        ms_res = pool.map(cv_sht_am, para_space)
    elif method_name == 'sto_iht':
        ms_res = pool.map(cv_sto_iht, para_space)
    elif method_name == 'hsg_ht':
        ms_res = pool.map(cv_hsg_ht, para_space)
    pool.close()
    pool.join()
    for para, auc_wt, cv_wt_results in ms_res:
        results[para] = dict()
        results[para][method_name] = {'auc_wt': auc_wt, 'cv_wt': cv_wt_results}
    f_name = 'ms_%02d_%02d_%s.pkl' % (trial_id_low, trial_id_high, method_name)
    pkl.dump(results, open(data_path + f_name, 'wb'))


def run_testing(method_name, num_cpus):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 5, 50, [1000], [0.3]
    posi_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    s_list = [20, 40, 60, 80]
    para_space, test_res, results = [], [], dict()
    for trial_id, num_tr, mu, posi_ratio in product(range(num_trials), tr_list, mu_list, posi_ratio_list):
        for s in s_list:
            para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s))
    pool = multiprocessing.Pool(processes=num_cpus)
    if method_name == 'solam':
        test_res = pool.map(test_solam, para_space)
    elif method_name == 'spam_l1':
        test_res = pool.map(test_spam_l1, para_space)
    elif method_name == 'spam_l2':
        test_res = pool.map(test_spam_l2, para_space)
    elif method_name == 'spam_l1l2':
        test_res = pool.map(test_spam_l1l2, para_space)
    elif method_name == 'fsauc':
        test_res = pool.map(test_fsauc, para_space)
    elif method_name == 'sht_am':
        test_res = pool.map(test_sht_am, para_space)
    elif method_name == 'sto_iht':
        test_res = pool.map(test_sto_iht, para_space)
    elif method_name == 'hsg_ht':
        test_res = pool.map(test_hsg_ht, para_space)
    pool.close()
    pool.join()
    results = {key: val for d in test_res for key, val in d.items()}
    pkl.dump(results, open(data_path + 're_%s.pkl' % method_name, 'wb'))


def run_para_s(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i = para
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
    __ = np.empty(shape=(1,), dtype=float)
    list_s = range(20, 76)
    # c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1.0, 0.0)
    list_c = 10. ** np.arange(-3, 3, 1, dtype=float)
    auc_wt, auc_wt_bar, cv_wt_results = dict(), dict(), np.zeros((len(list_c), len(list_s)))
    for fold_id, (ind_c, para_c), (ind_s, para_s) in product(range(k_fold), enumerate(list_c), enumerate(list_s)):
        algo_para = (trial_id, fold_id, num_passes, para_c, para_s, k_fold)
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        if (trial_id, fold_id) not in auc_wt:  # cross validate based on tr_index
            auc_wt[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
            auc_wt_bar[(trial_id, fold_id)] = {'auc': 0.0, 'para': algo_para, 'num_nonzeros': 0.0}
        list_auc_wt = np.zeros(k_fold)
        list_auc_wt_bar = np.zeros(k_fold)
        list_num_nonzeros_wt = np.zeros(k_fold)
        list_num_nonzeros_wt_bar = np.zeros(k_fold)
        kf = KFold(n_splits=k_fold, shuffle=False)
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            b, para_l2, record_aucs, verbose = 50, 0.0, 0, 0
            # c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1.0, 0.0)
            re = c_algo_sht_am(sub_x_tr, sub_y_tr, para_s, b, para_c, para_l2, num_passes, record_aucs, verbose)
            wt = np.asarray(re[0])
            wt_bar = np.asarray(re[1])
            list_auc_wt[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            list_auc_wt_bar[ind] = roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt_bar))
            list_num_nonzeros_wt[ind] = np.count_nonzero(wt)
            list_num_nonzeros_wt_bar[ind] = np.count_nonzero(wt_bar)
        cv_wt_results[ind_c, ind_s] = np.mean(list_auc_wt)
        if auc_wt[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt):
            auc_wt[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt))
            auc_wt[(trial_id, fold_id)]['para'] = algo_para
            auc_wt[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt))
        if auc_wt_bar[(trial_id, fold_id)]['auc'] < np.mean(list_auc_wt_bar):
            auc_wt_bar[(trial_id, fold_id)]['auc'] = float(np.mean(list_auc_wt_bar))
            auc_wt_bar[(trial_id, fold_id)]['para'] = algo_para
            auc_wt_bar[(trial_id, fold_id)]['num_nonzeros'] = float(np.mean(list_num_nonzeros_wt_bar))
        print(para_c, para_s, np.mean(list_auc_wt), np.mean(list_auc_wt_bar))


def run_diff_ratio(method):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 5, 50, [1000], [0.3]
    posi_ratio_list = [0.10, 0.20, 0.30, 0.40, 0.50]
    trial_id, num_tr, mu, fig_i, para_b = 0, 1000, 0.3, 'fig_3', 50
    rts_list = np.zeros(shape=(25, 100))
    aucs_list = np.zeros(shape=(25, 100))
    results = dict()
    ms = pkl.load(open(data_path + 'ms_00_05_%s.pkl' % method, 'rb'))
    if method == 'sht_am_v1':
        version = 0
    elif method == 'sht_am_v2':
        version = 2
    else:
        version = 1
    for posi_ratio in posi_ratio_list:
        for trial_id in range(5):
            f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
            data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[fig_i]
            __ = np.empty(shape=(1,), dtype=float)
            step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
            global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
            for fold_id in range(k_fold):
                para = (trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, fig_i)
                para_s, _, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
                tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
                x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
                y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
                _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'],
                                  global_paras, version, para_s, para_b, 1.0, 0.0)
                wt, aucs, rts, epochs = _
                rts_list[trial_id * 5 + fold_id] = rts[:100]
                aucs_list[trial_id * 5 + fold_id] = aucs[:100]
        results[posi_ratio] = np.mean(aucs_list, axis=0)
    pkl.dump(results, open(data_path + 're_diff_p_ratio-%s.pkl' % method, 'wb'))


def run_diff_s(para_s):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 5, 50, [1000], [0.3]
    posi_ratio, num_tr, mu, s = 0.1, 1000, 0.3, 40
    __ = np.empty(shape=(1,), dtype=float)
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    aucs_list = dict()
    for method in ['sht_am', 'sto_iht', 'hsg_ht']:
        aucs_list[method] = np.zeros(25)
        ms = pkl.load(open(data_path + 'ms_00_05_%s.pkl' % method, 'rb'))
        for trial_id, fold_id in product(range(5), range(5)):
            f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
            data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
            para = (trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
            tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
            te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
            x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
            y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
            if method == 'sht_am':
                _, para_b, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
                wt, aucs, rts, epochs = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'],
                                                      global_paras, 0, para_s, para_b, 1.0, 0.0)
            elif method == 'sto_iht':
                _, para_b, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
                wt, aucs, rts, epochs = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras,
                                                       para_s, para_b, 1., 0.0)
            else:
                para_tau, para_zeta = 1.0, 1.0003
                _, para_c, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
                wt, aucs, rts, epochs = c_algo_hsg_ht(x_tr, __, __, __, y_tr, 0, data['p'],
                                                      global_paras, para_s, para_tau, para_zeta, para_c, 0.0)
            re = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
            aucs_list[method][trial_id * 5 + fold_id] = re
            print(trial_id, fold_id, method, aucs_list[method][trial_id * 5 + fold_id])
    return para_s, aucs_list


def show_diff_s():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 6, 5
    para_s_list = range(20, 61, 5)
    fig, ax = plt.subplots(1, 1)
    ax.grid(color='lightgray', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    results = pkl.load(open(data_path + 're_diff_s.pkl', 'rb'))
    results = [_[1] for _ in results]
    method_label = ['SHT-AUC', 'StoIHT', 'HSG-HT']
    marker_list = ['D', 's', 'o']
    color_list = ['r', 'g', 'b']
    for method_ind, method in enumerate(['sht_am', 'sto_iht', 'hsg_ht']):
        ax.plot(para_s_list, [np.mean(_[method]) for _ in results], label=method_label[method_ind],
                marker=marker_list[method_ind], markersize=6., markerfacecolor='white', color=color_list[method_ind],
                linewidth=2., markeredgewidth=2.)
    ax.legend(loc='center right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.5, handletextpad=0.1, markerfirst=True)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('AUC Score')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'simu_diff_s.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def run_diff_b(para_b):
    k_fold, num_trials, num_passes, tr_list, mu_list = 5, 5, 50, [1000], [0.3]
    posi_ratio, num_tr, mu, s = 0.1, 1000, 0.3, 20
    __ = np.empty(shape=(1,), dtype=float)
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    aucs_list = dict()
    for method in ['sht_am', 'sto_iht']:
        aucs_list[method] = np.zeros(25)
        ms = pkl.load(open(data_path + 'ms_00_05_%s.pkl' % method, 'rb'))
        for trial_id, fold_id in product(range(5), range(5)):
            f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
            data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
            para = (trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s)
            tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
            te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
            x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
            y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
            if method == 'sht_am':
                para_s, _, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
                wt, aucs, rts, epochs = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'],
                                                      global_paras, 0, para_s, para_b, 1.0, 0.0)
            elif method == 'sto_iht':
                para_s, _, _ = ms[para][method]['auc_wt'][(trial_id, fold_id)]['para']
                wt, aucs, rts, epochs = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras,
                                                       para_s, para_b, 1., 0.0)
            else:
                wt = np.empty(shape=1)
            re = roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt))
            aucs_list[method][trial_id * 5 + fold_id] = re
            print(trial_id, fold_id, method, aucs_list[method][trial_id * 5 + fold_id])
    return para_b, aucs_list


def show_diff_b():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 6, 5
    para_b_list = [800 / _ for _ in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20][::-1]]
    fig, ax = plt.subplots(1, 1)
    ax.grid(color='lightgray', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    results = pkl.load(open(data_path + 're_diff_b.pkl', 'rb'))
    results = [_[1] for _ in results]
    method_label = ['SHT-AUC', 'StoIHT']
    marker_list = ['D', 's']
    color_list = ['r', 'g']
    for method_ind, method in enumerate(['sht_am', 'sto_iht']):
        y = [np.mean(_[method]) for _ in results]
        ax.plot(range(len(para_b_list)), y, label=method_label[method_ind],
                marker=marker_list[method_ind], markersize=6., markerfacecolor='white',
                color=color_list[method_ind], linewidth=2., markeredgewidth=2.)
    ax.legend(loc='lower right', framealpha=0., frameon=True, borderpad=0.1,
              labelspacing=0.5, handletextpad=0.1, markerfirst=True)
    ax.set_xticks(range(len([40, 44, 50, 57, 66, 80, 100, 133, 200, 400, 800])))
    ax.set_xticklabels([40, 44, 50, 57, 66, 80, 100, 133, 200, 400, 800])
    ax.set_xlabel('Block Size (b)')
    ax.set_ylabel('AUC Score')
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'simu_diff_b.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def show_diff_ratio(method):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 6, 5
    results = pkl.load(open(data_path + 're_diff_p_ratio-%s.pkl' % method, 'rb'))
    fig, ax = plt.subplots(1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    color_list = ['r', 'b', 'g', 'm', 'c']
    for ind_p, p_ratio in enumerate([0.10, 0.20, 0.30, 0.40, 0.50]):
        aucs = results[p_ratio]
        ax.plot(aucs[:50], color=color_list[ind_p], linewidth=2., label=r'$\displaystyle \hat{p}_n=%.1f$' % p_ratio)
    ax.legend(loc='lower right', framealpha=0., bbox_to_anchor=(1.0, 0.0), frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_ylabel('AUC')
    ax.set_xlabel('Epochs')
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    f_name = root_path + 'simu_diff_p_ratio-%s.pdf' % method
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()
    plt.show()


def show_sparsity():
    method_list = ['sht_am', 'spam_l1', 'spam_l2', 'fsauc', 'solam', 'sto_iht', 'hsg_ht']
    s_list = [20, 40, 60, 80]
    posi_ratio_list = [0.05]
    results = {method: pkl.load(open(os.path.join(data_path, 're_%s.pkl' % method))) for method in method_list}
    print('-' * 100)
    for (ind_p, posi_ratio) in enumerate(posi_ratio_list):
        for ind_method, method in enumerate(method_list):
            print(method),
            str_list = []
            for ind_fig, s in enumerate(s_list):
                re_auc = [results[method][key]['auc_wt'] for key in results[method] if
                          key[-1] == s and key[-2] == posi_ratio]
                str_list.append('%.3f$\pm$%.3f' % (float(np.mean(re_auc)), float(np.std(re_auc))))
            print(' & '.join(str_list))
    print('---')
    for (ind_p, posi_ratio) in enumerate(posi_ratio_list):
        for ind_method, method in enumerate(method_list):
            print(method),
            str_list = []
            for ind_fig, s in enumerate(s_list):
                re_fm = [results[method][key]['f1_score'] for key in results[method] if
                         key[-1] == s and key[-2] == posi_ratio]
                str_list.append('%.3f$\pm$%.3f' % (float(np.mean(re_fm)), float(np.std(re_fm))))
            print(' & '.join(str_list))


def show_result_01_2():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 14
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 8
    fig, ax = plt.subplots(2, 2, sharex=True)
    for ii, jj in product(range(2), range(2)):
        ax[ii, jj].grid(color='lightgray', linestyle='dotted', axis='both')
        ax[ii, jj].spines['right'].set_visible(False)
        ax[ii, jj].spines['top'].set_visible(False)
    color_list = ['r', 'g', 'm', 'b', 'y', 'k', 'orangered', 'olive', 'blue', 'darkgray', 'darkorange']
    marker_list = ['s', 'o', 'P', 'X', 'H', '*', 'x', 'v', '^', '+', '>']
    method_list = ['sht_am_v1', 'sht_am_v2', 'graph_am_v1', 'graph_am_v2',
                   'spam_l1', 'spam_l2', 'fsauc', 'spam_l1l2', 'solam', 'sto_iht', 'hsg_ht']
    method_label_list = ['SHT-AM-V1', 'SHT-AM-V2', 'Graph-AM-V1', 'Graph-AM-V2',
                         r"SPAM-$\displaystyle \ell^1$", r"SPAM-$\displaystyle \ell^2$",
                         'FSAUC', r"SPAM-$\displaystyle \ell^1/\ell^2$", r"SOLAM", r"StoIHT", 'HSG-HT']
    fig_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4']
    posi_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for ind_method, method in enumerate(method_list):
        results = pkl.load(open(os.path.join(data_path, 're_%s.pkl' % method)))
        for ind_fig, fig_i in enumerate(fig_list):
            re = []
            for posi_ratio in posi_ratio_list:
                re.append(np.mean([results[key]['auc_wt'] for key in results
                                   if key[-1] == fig_i and key[-2] == posi_ratio]))
            ax[ind_fig / 2, ind_fig % 2].plot(posi_ratio_list, re,
                                              marker=marker_list[ind_method],
                                              color=color_list[ind_method],
                                              label=method_label_list[ind_method])
    for ind_fig in range(4):
        ax[ind_fig / 2, ind_fig % 2].set_title(r"Network %d" % (ind_fig + 1))
    ax[0, 0].set_ylabel('AUC')
    ax[1, 0].set_ylabel('AUC')
    ax[1, 0].set_xlabel('Positive Ratio')
    ax[1, 1].set_xlabel('Positive Ratio')
    ax[1, 1].legend(loc='lower right', framealpha=1., bbox_to_anchor=(1.0, 0.0),
                    fontsize=14., frameon=True, borderpad=0.1,
                    labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax[0, 0].set_xticks(posi_ratio_list)
    ax[0, 1].set_xticks(posi_ratio_list)
    ax[1, 0].set_xticks(posi_ratio_list)
    ax[1, 1].set_xticks(posi_ratio_list)
    ax[0, 0].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    ax[0, 1].set_yticks([0.55, 0.65, 0.75, 0.85, 0.95])
    for i in range(2):
        ax[0, i].set_ylim([0.56, .96])
    for i in range(2):
        ax[1, i].set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax[1, i].set_ylim([0.81, 1.01])
    ax[1, 1].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    plt.savefig(root_path + 'simu-result-01.pdf', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def show_result_01():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 12, 8
    fig, ax = plt.subplots(1, 1)
    ax.grid(color='lightgray', linestyle='dotted', axis='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    color_list = ['r', 'g', 'm', 'b', 'y', 'k', 'orangered', 'olive', 'darkorange']
    marker_list = ['s', 'o', 'P', 'X', 'H', '*', 'x', 'v', '^', '+', '>']
    method_list = ['sht_am', 'fsauc', 'solam', 'sto_iht', 'hsg_ht', 'spam_l1', 'spam_l2', 'spam_l1l2']
    method_label_list = ['SHT-AUC', 'FSAUC', r"SOLAM", r"StoIHT", 'HSG-HT', r"SPAM-$\displaystyle \ell^1$",
                         r"SPAM-$\displaystyle \ell^2$", r"SPAM-$\displaystyle \ell^1/\ell^2$", ]
    method_list = ['sht_am', 'fsauc']
    method_label_list = ['SHT-AUC', 'FSAUC']
    s_list = [20]
    posi_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for ind_method, method in enumerate(method_list):
        results = pkl.load(open(os.path.join(data_path, 're_%s.pkl' % method)))
        if method == 'fsauc':
            pass
        for ind_fig, s in enumerate(s_list):
            re = []
            for posi_ratio in posi_ratio_list:
                re.append(np.mean([results[key]['auc_wt'] for key in results
                                   if key[-1] == s and key[-2] == posi_ratio]))
            ax.plot(posi_ratio_list, re, marker=marker_list[ind_method], label=method_label_list[ind_method],
                    markersize=6., markerfacecolor='white', color=color_list[ind_method], linewidth=2.,
                    markeredgewidth=2., )
    ax.set_title(r"Network 1")
    ax.set_ylabel('AUC')
    ax.set_xlabel('Positive Ratio')
    ax.legend(loc='lower right', framealpha=1., bbox_to_anchor=(1.0, 0.0),
              fontsize=14., frameon=True, borderpad=0.1,
              labelspacing=0.1, handletextpad=0.1, markerfirst=True)
    ax.set_xticks(posi_ratio_list)
    # ax.set_yticks([0.5, 0.55, 0.65, 0.75, 0.85, 0.95])
    # ax.set_ylim([0.5, .99])
    root_path = '/home/baojian/Dropbox/Apps/ShareLaTeX/icml20-sht-auc/figs/'
    plt.savefig(root_path + 'simu-result-01-01.pdf', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()
    plt.show()


def main(run_option):
    if run_option == 'run_ms':
        run_ms(method_name=sys.argv[2], trial_id_low=int(sys.argv[3]),
               trial_id_high=int(sys.argv[4]), num_cpus=int(sys.argv[5]))
    elif run_option == 'run_test':
        run_testing(method_name=sys.argv[2], num_cpus=int(sys.argv[3]))
    elif run_option == 'run_diff_ratio':
        run_diff_ratio(method='sht_am_v1')
        run_diff_ratio(method='sht_am_v2')
    elif run_option == 'run_diff_s':
        para_s_list = range(20, 61, 5)
        pool = multiprocessing.Pool(processes=int(sys.argv[2]))
        ms_res = pool.map(run_diff_s, para_s_list)
        pool.close()
        pool.join()
        pkl.dump(ms_res, open(data_path + 're_diff_s.pkl', 'wb'))
    elif run_option == 'show_diff_s':
        show_diff_s()
    elif run_option == 'run_diff_b':
        para_b_list = [800 / _ for _ in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20][::-1]]
        pool = multiprocessing.Pool(processes=int(sys.argv[2]))
        ms_res = pool.map(run_diff_b, para_b_list)
        pool.close()
        pool.join()
        pkl.dump(ms_res, open(data_path + 're_diff_b.pkl', 'wb'))
    elif run_option == 'show_diff_b':
        show_diff_b()
    elif run_option == 'show_diff_ratio':
        show_diff_ratio(method='sht_am_v1')
    elif run_option == 'show_sparsity':
        show_sparsity()
    elif run_option == 'show_01':
        show_result_01()


def test_single():
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = 0, 5, 50, 1000, 0.3, 0.5, 80
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_s, para_c, _ = 80, 1., None
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_sto_iht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, 100, 1., 0.0)
        wt, aucs, rts, epochs = _
        import matplotlib.pyplot as plt
        plt.plot(rts, aucs)
        plt.show()
        print(roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt)))
        break


def test_single_2():
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = 0, 5, 50, 1000, 0.3, 0.5, 80
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_s, para_c, _ = 100, 3., None
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        para_tau, para_zeta = 1000., 1.033
        _ = c_algo_hsg_ht(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, para_s, para_tau, para_zeta, para_c, 0.0)
        wt, aucs, rts, epochs = _
        import matplotlib.pyplot as plt
        plt.plot(rts, aucs)
        plt.show()
        print(roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt)))
        break


def test_single_3(trial_id):
    k_fold, num_passes, num_tr, mu, posi_ratio, s = 5, 50, 1000, 0.3, 0.1, 60
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)

    for para_s in [20, 30, 40, 50, 60, 70, 80, 90]:
        aver_auc = []
        for fold_id in range(k_fold):
            para_b = 50
            tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
            te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
            x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
            y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
            _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1., 0.0)
            wt, aucs, rts, epochs = _
            aver_auc.append(roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt)))
        print(para_s, np.mean(aver_auc))


def run_all_model_selection():
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio = 0, 5, 50, 1000, 0.3, 0.2
    step_len, verbose, record_aucs, stop_eps = 1e8, 0, 0, 1e-4
    kf = KFold(n_splits=k_fold, shuffle=False)
    __ = np.empty(shape=(1,), dtype=float)
    for (s_ind, s), fold_id in product(enumerate([20, 40, 60, 80]), range(k_fold)):
        f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
        data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        # solam
        for (ind_xi, para_xi), (ind_r, para_r) in product(
                enumerate(np.arange(1, 101, 9, dtype=float)),
                enumerate(10. ** np.arange(-1, 6, 1, dtype=float))):
            for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
                sub_x_tr = np.asarray(data['x_tr'][tr_index[sub_tr_ind]], dtype=float)
                sub_y_tr = np.asarray(data['y_tr'][tr_index[sub_tr_ind]], dtype=float)
                sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
                sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
                global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
                _ = c_algo_solam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, para_r)
                wt, aucs, rts, epochs = _
                print(roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt)))
            break
        break


if __name__ == '__main__':
    main(run_option=sys.argv[1])
