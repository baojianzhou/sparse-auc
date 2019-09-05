//
// Created by baojian on 8/24/19.
//

#ifndef SPARSE_AUC_ALGO_SOLAM_H
#define SPARSE_AUC_ALGO_SOLAM_H

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
    int para_num_pass; // number of passes, under online setting, it should be 1.
} solam_para;

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
    int para_num_pass; // number of passes, under online setting, it should be 1.
} solam_para_sparse;

typedef struct {
    double *wt;
    double a;
    double b;
    double alpha;
} solam_results;

bool algo_solam_func(solam_para *para, solam_results *results);

bool algo_solam_sparse_func(solam_para_sparse *para, solam_results *results);

#endif //SPARSE_AUC_ALGO_SOLAM_H
