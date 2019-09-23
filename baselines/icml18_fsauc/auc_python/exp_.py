# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:20:19 2019

@author: Yunwen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:14:30 2018

@author: Yunwen
"""
#from sklearn.preprocessing import Normalizer

import numpy as np
from scipy import io as spio

from sklearn.model_selection import RepeatedKFold
from auc_fs import auc_fs
from get_idx import get_idx
from sklearn import preprocessing 
from sklearn.datasets import load_svmlight_file
import scipy.sparse as sp
import matplotlib.pyplot as plt
#import multiprocessing as mp


options = dict()
data = 'usps'
method = 'auc_fs'  # auc_oam_seq
options['n_pass'] = 10
options['fast'] = True
options['rec'] = 0.5
options['buf_size_p'] = 100
options['buf_size_n'] = 100

x_tr, y_tr = load_svmlight_file('data/'+data)
x_te, y_te = load_svmlight_file('data/'+data + '_t')
x = sp.vstack((x_tr, x_te))
x = preprocessing.normalize(x)  #normalization
n_tr = len(y_tr)
x_tr = x[:n_tr, :]
x_te = x[n_tr:, :]
options['ids'] = get_idx(n_tr, options['n_pass'])
if method == 'auc_solam':  #
    options['eta'] = 1e-3
    options['beta'] = 1e5
    auc_ret, rt_ret = auc_solam(x_tr, y_tr, x_te, y_te, options)
elif method == 'auc_solam_L2':  #
    options['eta'] = 1e-3
    options['beta'] = 1
    auc_ret, rt_ret = auc_solam_L2(x_tr, y_tr, x_te, y_te, options)        
elif method == 'auc_op':  #
    options['eta'] = 1
    options['beta'] = 1
    auc_ret, rt_ret = auc_op(x_tr, y_tr, x_te, y_te, options)  
elif method == 'auc_oam_seq':  #
    options['eta'] = 1
    auc_ret, rt_ret = auc_oam_seq(x_tr, y_tr, x_te, y_te, options)  
elif method == 'auc_oam_gra':  #
    options['eta'] = 1
    auc_ret, rt_ret = auc_oam_gra(x_tr, y_tr, x_te, y_te, options)
elif method == 'auc_fs':  #
    options['eta'] = 1
    options['beta'] = 1e5
    auc_ret, rt_ret = auc_fs(x_tr, y_tr, x_te, y_te, options)    
plt.semilogx(rt_ret, auc_ret, color="blue", linewidth=2.5, linestyle="-", label=method)  #plt. plo-t   plt.semilog-x
print(auc_ret)
plt.xlabel('seconds')
plt.ylabel('AUC')
plt.legend(loc='lower right')
#print('res/' + data + '.png')
plt.savefig('res/' + data + '_' + method + '.png')
plt.show()
