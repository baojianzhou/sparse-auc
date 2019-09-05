# -*- coding: utf-8 -*-
import os
import sys
import csv
import time

data_path = '/network/rit/lab/ceashpc/bz383376/data/icml2020/09_sector/lib-svm-data/'

for i in range(1000):
    if os.path.exists(data_path + 'model_select_%04d.pkl' % i):
        os.remove(data_path + 'model_select_%04d.pkl' % i)
