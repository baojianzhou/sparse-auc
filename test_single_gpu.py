# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam_sparse
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'


def get_data_by_ind(data, tr_ind, sub_tr_ind):
    sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens = [], [], [], []
    prev_posi = 0
    for index in tr_ind[sub_tr_ind]:
        cur_len = data[b'x_tr_lens'][index]
        cur_posi = data[b'x_tr_poss'][index]
        sub_x_vals.extend(data[b'x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_inds.extend(data[b'x_tr_inds'][cur_posi:cur_posi + cur_len])
        sub_x_lens.append(cur_len)
        sub_x_posis.append(prev_posi)
        prev_posi += cur_len
    sub_x_vals = np.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_posis = np.asarray(sub_x_posis, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data[b'y_tr'][tr_ind[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_tr


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


def pred_results(wt, wt_bar, auc, rts, para_list, te_index, data):
    return {'auc_wt': pred_auc(data, te_index, range(len(te_index)), wt),
            'auc_wt_bar': pred_auc(data, te_index, range(len(te_index)), wt_bar),
            'nonzero_wt': np.count_nonzero(wt),
            'nonzero_wt_bar': np.count_nonzero(wt_bar),
            'algo_para': para_list,
            'auc': auc,
            'rts': rts}


def main():
    method, para_c, para_l1, data_name, run_id, fold_id = 'spam_l1', 19., 1e-5, '12_news20', 0, 0
    passes, step_len = 5, 100
    s_time = time.time()
    f_name = os.path.join(data_path, '%s/data_run_%d.pkl' % (data_name, run_id))
    if sys.version_info[0] >= 3:
        with open(f_name, 'rb') as f:
            data = pkl.load(f, encoding="bytes")
    else:
        data = pkl.load(open(f_name, 'rb'))
    tr_index = data[b"fold_%d" % fold_id][b"tr_index"]
    te_index = data[b'fold_%d' % fold_id][b'te_index']
    x_vals, x_inds, x_poss, x_lens, y_tr = get_data_by_ind(data, tr_index, range(len(tr_index)))
    wt, wt_bar, auc, rts = c_algo_spam_sparse(
        np.asarray(x_vals, dtype=float), np.asarray(x_inds, dtype=np.int32),
        np.asarray(x_poss, dtype=np.int32), np.asarray(x_lens, dtype=np.int32),
        np.asarray(y_tr, dtype=float), int(data[b'p']), float(para_c), float(para_l1),
        float(0.0), int(0), int(passes), int(step_len), int(0))
    res = pred_results(wt, wt_bar, auc, rts, (para_c, para_l1), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_c, para_l1, auc, run_time)


if __name__ == '__main__':
    main()
