//
// Created by baojian on 8/29/19.
//

#ifndef SPARSE_AUC_ALGO_SPARSE_SOLAM_H
#define SPARSE_AUC_ALGO_SPARSE_SOLAM_H

#include "algo_base.h"

#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }

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
} stoht_am_para;

typedef struct {
    int *x_tr_indices;
    double *x_tr_values;
    double *y_tr;
    int max_nonzero; // the dimension of sparse matrix
    int p;
    int num_tr;
    int verbose;
    int *para_rand_ind; // to random shuffle the training samples
    double para_xi; // the parameter xi, to control the learning rate.
    double para_r; // the parameter R, to control the wt, the radius of the ball of wt.
    int para_s; // sparsity
    int para_num_pass; // number of passes, under online setting, it should be 1.
} stoht_am_sparse_para;

typedef struct {
    double *wt;
    double a;
    double b;
    double alpha;
} stoht_am_results;

bool algo_stoht_am_func(stoht_am_para *para, stoht_am_results *results);

bool algo_stoht_am_sparse_func(stoht_am_sparse_para *para, stoht_am_results *results);

#endif //SPARSE_AUC_ALGO_SPARSE_SOLAM_H
