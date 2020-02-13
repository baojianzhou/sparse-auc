# -*- coding: utf-8 -*-
import os
import sys
import time
import csv
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
        from sparse_module import c_algo_sht_am
    except ImportError:
        print('cannot find c_algo_sht_am function in sparse_module')
        pass
except ImportError:
    print('cannot find the module: sparse_module')
    pass

data_path = '/home/neyo/Projects/sparse-auc-master/data/00_simu/'

result_path = '/home/neyo/Projects/sparse-auc-master/results/conv/'

def pred_results(para):
    trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s = para
    results = dict()
    results['para'] = para
    results['aver_auc'] = []
    f_name = data_path + 'data_trial_%02d_tr_%03d_mu_%.1f_p-ratio_%.2f.pkl'
    data = pkl.load(open(f_name % (trial_id, num_tr, mu, posi_ratio), 'rb'))[s]
    __ = np.empty(shape=(1,), dtype=float)
    step_len, verbose, record_aucs, stop_eps = 1e2, 0, 1, 1e-4
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    for fold_id in range(k_fold):
        para_s = s
        para_b = 50 # batch size
        tr_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['tr_index']
        te_index = data['trial_%d_fold_%d' % (trial_id, fold_id)]['te_index']
        x_tr = np.asarray(data['x_tr'][tr_index], dtype=float)
        y_tr = np.asarray(data['y_tr'][tr_index], dtype=float)
        _ = c_algo_sht_am(x_tr, __, __, __, y_tr, 0, data['p'], global_paras, 0, para_s, para_b, 1., 0.0) # p is dimension
        wt, aucs, rts, epochs = _ # type are all list	
        # aver_auc.append(roc_auc_score(y_true=data['y_tr'][te_index], y_score=np.dot(data['x_tr'][te_index], wt)))
        results['aver_auc'].append(aucs)
    m = np.max(results['aver_auc'])	
    print('s: %d p-ratio: %.2f %f' % (s, posi_ratio,m))
    sys.stdout.flush()
    results['auc'] = np.mean(results['aver_auc'],0) / m
    pkl.dump(results, open(result_path + 's_%d_p-ratio_%.2f.pkl' % (s, posi_ratio), 'wb'))



def run(num_cpus):
    '''
    increase num_passes to see convergence
    '''
    trial_id, k_fold, num_passes, num_tr, mu = 0, 3, 300, 1000, 0.3
    posi_ratio_list = [0.05]
    s_list = [40,60,80]
    para_space = []
    for posi_ratio, s in product(posi_ratio_list, s_list):
        para_space.append((trial_id, k_fold, num_passes, num_tr, mu, posi_ratio, s))
    pool = multiprocessing.Pool(processes=num_cpus)
    conv_res = pool.map(pred_results, para_space)
    pool.close()
    pool.join()
    # for para, mrts, srts, maucs, saucs in conv_res:
        # f_name = result_path + 'test_result_s_%d_p-ratio_%.2f_%.6f.csv'
        # _, _, _, _, _, posi_ratio, s = para
        # with open(f_name % (s, posi_ratio, t), 'a') as csvfile:
            # writer = csv.writer(csvfile, delimiter=',')
            # writer.writerow([num_passes, round(mrts,3), round(srts,3), round(maucs,4), round(saucs,3)])
        # plt.plot(maucs)
        # g_name = result_path +  's_%d_p-ratio_%.2f_%.6f.pdf' % (s, posi_ratio, t)
        # plt.savefig(g_name, format='pdf')

def show():
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 6, 5
    s_list = [80]
    posi_ratio_list = [0.05, 0.25, 0.5]
    for s in s_list:
        mean_lines = []
        fig, ax = plt.subplots(1, 1)
        ax.grid(color='lightgray', linestyle='dotted', axis='both')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        marker_list = ['D', 's', 'o']
        color_list = ['r', 'g', 'b']
        for ind, posi_ratio in enumerate(posi_ratio_list):
            results = pkl.load(open(os.path.join(result_path, 's_%d_p-ratio_%.2f.pkl' % (s, posi_ratio))))
            mean_line = results['auc'] 
            mean_line_length = len(mean_line)
            iters = int(mean_line_length/300)
            mean_line_epochs = mean_line[0::20*iters]  
            ax.plot(range(15), mean_line_epochs, label='r = %.2f' % posi_ratio,  marker=marker_list[ind], markersize=6., markerfacecolor='white', color=color_list[ind], linewidth=2., markeredgewidth=2.)
        ax.legend(loc='lower right', framealpha=1., frameon=True, borderpad=0.1,
              labelspacing=0.5, handletextpad=0.1, markerfirst=True)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('AUC Score (scaled)')
        ax.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
        ax.set_xticklabels([0,50, 100, 150, 200, 250, 300])
        ax.set_yticklabels([])
        f_name = result_path +  's_%d.pdf' % (s)
        plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0.05, format='pdf')
        plt.close()

def main(run_option):
    if run_option == 'run':
        run(num_cpus = 12)
    elif run_option == 'show':
        show()
    
if __name__ == '__main__':
    main(run_option=sys.argv[1])
