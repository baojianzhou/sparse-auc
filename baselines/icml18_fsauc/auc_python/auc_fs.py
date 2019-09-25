# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:25:13 2018

@author: Yunwen

# -*- coding: utf-8 -*-
Spyder Editor

We apply the algorithm in Liu, 2018 ICML to do Fast AUC maximization

Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary 
        'ids' stores the indices of examples traversed, ids divided by number of training examples is the number of passes
        'eta' stores the initial step size
        'beta': the parameter R
        'n_pass': the number of passes
        'time_lim': optional argument, the maximal time allowed
Output:
    aucs: results on iterates indexed by res_idx
    time:
"""
import os
import time
import numpy as np
from itertools import product
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pickle as pkl
# from proj_l1ball import euclidean_proj_l1ball
import timeit
from scipy.sparse import isspmatrix

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/00_simu/'


def auc_fs(x_tr, y_tr, x_te, y_te, options):
    # options
    delta = 1e-1
    dd = np.log(12 / delta)
    ids = options['ids']
    n_ids = len(ids)
    eta = options['eta']
    R = options['beta']  # beta is the parameter R, we use beta for consistency
    n_tr, dim = x_tr.shape
    v_1, alpha_1 = np.zeros(dim + 2), 0
    sp = 0  # the estimate of probability with positive example
    t = 0  # the time iterate"
    time_s = 0
    sx_pos = np.zeros(dim)  # summation of positive instances
    sx_neg = np.zeros(dim)  # summation of negative instances
    m_pos = sx_pos
    m_neg = sx_neg
    # -------------------------------
    # for storing the results
    n_pass = options['n_pass']
    res_idx = 2 ** (np.arange(4, np.log2(n_pass * n_tr), options['rec']))
    n_idx = len(res_idx)
    aucs = np.zeros(n_idx)
    times = np.zeros(n_idx)
    i_res = 0
    # ------------------------------

    # we have normalized the data
    m = int(0.5 * np.log2(2 * n_ids / np.log2(n_ids))) - 1
    n_0 = int(n_ids / m)
    r = 2 * np.sqrt(3) * R
    beta = 9
    D = 2 * np.sqrt(2) * r

    res_idx[-1] = n_0 * m
    res_idx = [int(i) for i in res_idx]  # map(int, res_idx)
    pred = y_te
    gd = np.zeros(dim + 2)
    if isspmatrix(x_tr):
        x_tr = x_tr.toarray()  # to dense matrix for speeding up the computation
    start = timeit.default_timer()
    #    print('n_0 = %s m = %s R = %s dd = %.2f' % (n_0, m, R, dd))
    for k in range(m):
        #        print('D = %.2f beta = %.2f r = %.2f eta = %.2f' % (D, beta, r, eta))
        v_sum = np.zeros(dim + 2)
        v, alpha = v_1, alpha_1
        for kk in range(n_0):
            x_t = x_tr[ids[t], :]
            y_t = y_tr[ids[t]]
            wx = np.inner(x_t, v[:dim])
            if y_t == 1:
                sp = sp + 1
                p = sp / (t + 1)
                sx_pos = sx_pos + x_t
                gd[:dim] = (1 - p) * (wx - v[dim] - 1 - alpha) * x_t
                gd[dim] = (p - 1) * (wx - v[dim])
                gd[dim + 1] = 0
                gd_alpha = (p - 1) * (wx + p * alpha)
            else:
                p = sp / (t + 1)
                sx_neg = sx_neg + x_t
                gd[:dim] = p * (wx - v[dim + 1] + 1 + alpha) * x_t
                gd[dim] = 0
                gd[dim + 1] = p * (v[dim + 1] - wx)
                gd_alpha = p * (wx + (p - 1) * alpha)
            t = t + 1
            v = v - eta * gd
            alpha = alpha + eta * gd_alpha

            # some projection
            # ---------------------------------
            v[:dim] = ProjectOntoL1Ball(v[:dim], R)
            tnm = np.abs(v[dim])
            if tnm > R:
                v[dim] = v[dim] * (R / tnm)
            tnm = np.abs(v[dim + 1])
            if tnm > R:
                v[dim + 1] = v[dim + 1] * (R / tnm)
            tnm = np.abs(alpha)
            if tnm > 2 * R:
                alpha = alpha * (2 * R / tnm)

            vd = v - v_1
            tnm = np.linalg.norm(vd)
            if tnm > r:
                vd = vd * (r / tnm)
            v = v_1 + vd
            ad = alpha - alpha_1
            tnm = np.abs(ad)
            if tnm > D:
                ad = ad * (D / tnm)
            alpha = alpha_1 + ad
            # ---------------------------------

            v_sum = v_sum + v
            v_ave = v_sum / (kk + 1)
            if res_idx[i_res] == t:
                stop = timeit.default_timer()
                time_s += stop - start
                w_ave = v_ave[:dim]
                pred = (x_te.dot(w_ave.T)).ravel()
                if not np.all(np.isfinite(pred)):
                    aucs[i_res:] = aucs[i_res - 1]
                    times[i_res:] = time_s
                    break
                fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label=1)
                aucs[i_res] = metrics.auc(fpr, tpr)
                times[i_res] = time_s
                i_res = i_res + 1
                # this is slow we do not want to spend more time
                if 'time_lim' in options and time_s > options['time_lim']:
                    aucs[i_res:] = aucs[i_res - 1]
                    times[i_res:] = time_s
                    break
                start = timeit.default_timer()
        if not np.all(np.isfinite(pred)):
            break
        r = r / 2
        # update D and beta
        tmp1 = 12 * np.sqrt(2) * (2 + np.sqrt(2 * dd)) * R
        tmp2 = min(p, 1 - p) * n_0 - np.sqrt(2 * n_0 * dd)
        if tmp2 > 0:
            D = 2 * np.sqrt(2) * r + tmp1 / np.sqrt(tmp2)
        else:
            D = 1e7
        tmp1 = 288 * ((2 + np.sqrt(2 * dd)) ** 2)
        tmp2 = min(p, 1 - p) - np.sqrt(2 * dd / n_0)
        if tmp2 > 0:
            beta_new = 9 + tmp1 / tmp2
        else:
            beta_new = 1e7
        eta = min(np.sqrt(beta_new / beta) * eta / 2, eta)
        beta = beta_new
        if sp > 0:
            m_pos = sx_pos / sp
        if sp < t:
            m_neg = sx_neg / (t - sp)
        v_1 = v_ave
        alpha_1 = np.inner(m_neg - m_pos, v_ave[:dim])
    return aucs, times


def ProjectOntoL1Ball(v, b):
    nm = np.abs(v)
    if nm.sum() < b:
        w = v
    else:
        u = np.sort(nm)[::-1]
        sv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(v) + 1) > (sv - b))[0][-1]
        theta = (sv[rho] - b) / (rho + 1)
        w = np.sign(v) * np.maximum(nm - theta, 0)
    return w
