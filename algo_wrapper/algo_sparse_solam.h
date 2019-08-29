//
// Created by baojian on 8/29/19.
//

#ifndef SPARSE_AUC_ALGO_SPARSE_SOLAM_H
#define SPARSE_AUC_ALGO_SPARSE_SOLAM_H

#include "algo_base.h"

typedef struct {
    double *x_tr;
    double *y_tr;
    int p;
    int num_tr;
    int verbose;
    int *para_rand_ind; // to random shuffle the training samples
    double para_xi; // the parameter xi, to control the learning rate.
    double para_r; // the parameter R, to control the wt, the radius of the ball of wt.
    int para_s; // sparsity
    int para_num_pass; // number of passes, under online setting, it should be 1.
} sparse_solam_para;

typedef struct {
    double *wt;
    double a;
    double b;
    double alpha;
} sparse_solam_results;

bool algo_sparse_solam_func(sparse_solam_para *para, sparse_solam_results *results);

#endif //SPARSE_AUC_ALGO_SPARSE_SOLAM_H
