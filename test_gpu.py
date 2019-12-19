# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_auc_score
import cupy as cp

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/'


def c_algo_spam_sparse(x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr,
                       data_p, para_xi, para_l1_reg, para_l2_reg,
                       para_reg_opt, para_num_passes, para_step_len):
    start_time = time.time()
    posi_t, nega_t, data_n = 0.0, 0.0, len(data_y_tr)
    y_pred = np.zeros((len(data_y_tr),))
    re_wt, re_wt_bar = cp.zeros((data_p,)), cp.zeros((data_p,))
    posi_x_mean = cp.zeros((data_p,))
    nega_x_mean = cp.zeros((data_p,))
    re_len_auc = 0
    total_num_eval = int((data_n * (para_num_passes + 1)) / para_step_len)
    re_auc = cp.zeros((total_num_eval,))
    re_rts = cp.zeros((total_num_eval,))
    for i in range(len(data_y_tr)):
        if data_y_tr[i] > 0:
            posi_t += 1.
            for kk in range(x_tr_lens[i]):
                posi_x_mean[x_tr_inds[x_tr_poss[i] + kk]] += x_tr_vals[x_tr_poss[i] + kk]
        else:
            nega_t += 1.
            for kk in range(x_tr_lens[i]):
                nega_x_mean[x_tr_inds[x_tr_poss[i] + kk]] += x_tr_vals[x_tr_poss[i] + kk]
    posi_x_mean /= posi_t
    nega_x_mean /= nega_t
    prob_p, eta_t, t_eval = posi_t / (data_n * 1.), 0.0, 0.0
    for t in range(1, para_num_passes * data_n + 1):
        eta_t = para_xi / math.sqrt(t)
        a_wt = cp.inner(re_wt, posi_x_mean)
        b_wt = cp.inner(re_wt, nega_x_mean)
        alpha_wt = b_wt - a_wt
        wt_dot = 0.0
        for tt in range(x_tr_lens[(t - 1) % data_n]):
            val1 = re_wt[x_tr_inds[x_tr_poss[(t - 1) % data_n] + tt]]
            val2 = x_tr_vals[x_tr_poss[(t - 1) % data_n] + tt]
            wt_dot += (val1 * val2)
        term1 = 2. * (1.0 - prob_p) * (wt_dot - a_wt) - 2. * (1.0 + alpha_wt) * (1.0 - prob_p)
        term2 = 2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p
        weight = term1 if data_y_tr[(t - 1) % data_n] > 0 else term2
        for tt in range(x_tr_lens[(t - 1) % data_n]):
            val1 = x_tr_vals[x_tr_poss[(t - 1) % data_n] + tt]
            re_wt[x_tr_inds[x_tr_poss[(t - 1) % data_n] + tt]] += -eta_t * weight * val1
        if para_reg_opt == 0:
            tmp_demon = (eta_t * para_l2_reg + 1.)
            tmp_sign = cp.sign(re_wt) / tmp_demon
            re_wt = tmp_sign * cp.fmax(0.0, cp.abs(re_wt) - eta_t * para_l1_reg)
        else:
            re_wt = 1. / (eta_t * para_l2_reg + 1.) * re_wt
        re_wt_bar = re_wt + re_wt_bar
        if math.fmod(t, para_step_len) == 1.:
            t_eval = time.time()
            for q in range(data_n):
                y_pred[q] = 0.0
                for tt in range(x_tr_lens[q]):
                    val1 = x_tr_vals[x_tr_poss[q] + tt]
                    y_pred[q] += re_wt[x_tr_inds[x_tr_poss[q] + tt]] * val1
            re_auc[re_len_auc] = roc_auc_score(y_true=data_y_tr, y_score=y_pred)
            re_rts[re_len_auc] = time.time() - start_time - (time.time() - t_eval)
            re_len_auc += 1
    re_wt_bar = 1. / (para_num_passes * data_n) * re_wt_bar
    return re_wt, re_wt_bar


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
    sub_x_vals = cp.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_posis = np.asarray(sub_x_posis, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data[b'y_tr'][tr_ind[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_tr


def pred_auc(data, tr_index, sub_te_ind, wt):
    if cp.isnan(wt).any() or cp.isinf(wt).any():  # not a valid score function.
        return 0.0
    _ = get_data_by_ind(data, tr_index, sub_te_ind)
    sub_x_vals, sub_x_inds, sub_x_posis, sub_x_lens, sub_y_te = _
    y_pred_wt = cp.zeros_like(sub_te_ind, dtype=float)
    for i in range(len(sub_te_ind)):
        cur_posi = sub_x_posis[i]
        cur_len = sub_x_lens[i]
        cur_x = sub_x_vals[cur_posi:cur_posi + cur_len]
        cur_ind = sub_x_inds[cur_posi:cur_posi + cur_len]
        tmp = cp.asarray([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
        y_pred_wt[i] = cp.sum(tmp)
    return roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)


def pred_results(wt, wt_bar, auc, rts, para_list, te_index, data):
    return {'auc_wt': pred_auc(data, te_index, range(len(te_index)), wt),
            'auc_wt_bar': pred_auc(data, te_index, range(len(te_index)), wt_bar),
            'nonzero_wt': cp.count_nonzero(wt),
            'nonzero_wt_bar': cp.count_nonzero(wt_bar),
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
    wt, wt_bar = c_algo_spam_sparse(
        x_vals, x_inds, x_poss, x_lens, y_tr, data[b'p'], float(para_c), float(para_l1),
        float(0.0), int(0), int(passes), int(step_len))
    res = pred_results(wt, wt_bar, None, None, (para_c, para_l1), te_index, data)
    auc, run_time = res['auc_wt'], time.time() - s_time
    print(run_id, fold_id, method, para_c, para_l1, auc, run_time)


if __name__ == '__main__':
    main()
