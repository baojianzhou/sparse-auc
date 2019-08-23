# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle as pkl
import csv
import pandas as pd
import matplotlib.pyplot as plt


def data_process_13_ad():
    """
    ---
    Introduction

    The task is to predict whether an image is an advertisement ("ad") or not ("nonad"). There are
    3 continuous attributes(1st:height,2nd:width,3rd:as-ratio)-28% data is missing for each
    continuous attribute. Clearly, the third attribute does not make any contribution since it is
    the 2nd:width./1st:height.

    Number of positive labels: 459
    Number of negative labels: 2820

    ---
    Data Processing
    Need to fill the missing values. The missing values only happened in the first three features.


    ---
    BibTeX:
    @inproceedings{kushmerick1999learning,
        title={Learning to remove internet advertisements},
        author={KUSHMERICK, N},
        booktitle={the third annual conference on Autonomous Agents, May 1999},
        pages={175--181},
        year={1999}
        }

    ---
    Data Format:
    each feature is either np.nan ( missing values) or a real value.
    label: +1: positive label(ad.) -1: negative label(nonad.)
    :return:
    """
    path = '/home/baojian/Desktop/ad-dataset/ad.data'
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        raw_data = [row for row in reader]
        data_x = []
        data_y = []
        for row in raw_data:
            values = []
            for ind, _ in enumerate(row[:-1]):
                if str(_).find('?') == -1:  # handle missing values.
                    values.append(float(_))
                else:
                    values.append(np.nan)
            data_x.append(values)
            if row[-1] == 'ad.':
                data_y.append(+1)
            else:
                data_y.append(-1)
        print('number of positive labels: %d' % len([_ for _ in data_y if _ > 0]))
        print('number of negative labels: %d' % len([_ for _ in data_y if _ < 0]))
        pkl.dump({'data_x': np.asarray(data_x),
                  'data_y': np.asarray(data_y)}, open('processed-13-ad.pkl', 'wb'))


def main():
    data_process_13_ad()


if __name__ == '__main__':
    main()
