# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
import sklearn as sk

def demo():
    g_pass = 15
    g_seq_num = 1
    g_iters = 5
    g_cv = 5
    g_data = {'data_name': 'a9a', 'data_dim': 123, 'data_num': 32561}
    opt_solam = {'test': ''}
    res_solam = {'AUC': np.zeros(shape=(g_iters, g_cv)),
                 'mean_AUC': 0.0,
                 'std_AUC': 0.0,
                 'run_time': np.zeros(shape=(g_iters, g_cv)),
                 'mean_run_time': 0.0}
    import scipy.io
    mat = scipy.io.loadmat('/home/baojian/nips16_solam/a9a.mat')
    org_feat = mat['orgFeat'].toarray()
    org_label = np.asarray([_[0] for _ in mat['orgLabel']])
    print(mat['orgLabel'].shape, mat['orgFeat'].shape)
    pp_label = org_label
    u_lab = np.unique(org_label)
    u_num = len(u_lab)
    if u_num > 2:
        u_sort = np.random.permutation(u_num)

    # post-processing the data
    pp_feat = np.zeros(shape=(g_data['data_num'], g_data['data_dim']))
    for k in range(1, g_data['data_num']):
        t_dat = org_feat[k, :]
        if np.linalg.norm(t_dat) > 0:
            t_dat = t_dat / np.linalg.norm(t_dat)
        pp_feat[k] = t_dat

    # set the results to zeros
    for m in range(g_iters):
        v_indices = sk.utils.

def main():
    demo()


if __name__ == '__main__':
    main()
