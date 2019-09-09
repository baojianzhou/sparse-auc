//
// Created by baojian on 9/9/19.
//

#ifndef SPARSE_AUC_AUC_OPT_METHODS_H
#define SPARSE_AUC_AUC_OPT_METHODS_H


#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// This is the only third part library needed.
#include <cblas.h>

#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }

/**
 * Define a sparse vector as a tuple of three elements.
 */
typedef struct {
    double *x_vals;
    int *x_indices;
    int x_len;
} sparse_vector;

sparse_vector *_malloc_sparse_vector(double *x_vals, int *x_indices, int x_len);

void _free_sparse_vector(sparse_vector *sv);

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


/**
 * A full vector y dot product with a sparse vector x.
 *
 * ---
 * x is presented as three elements:
 * 1. x_indices: the nonzero indices.
 * 2. x_values: the nonzeros.
 * 3. x_len: the number of nonzeros.
 * For example,
 * x = {0, 1.0, 0, 0, 0, 0.1, 0.5}, then
 * x_indices = {1,5,6}
 * x_values = {1.0,0.1,0.5}
 * x_len = 3.
 * ---
 *
 * @param x_indices: the nonzero indices of the sparse vector x. starts from 0 index.
 * @param x_values: the nonzeros values of the sparse vector x.
 * @param x_len: the number of nonzeros in sparse vector x.
 * @param y
 * @return
 */
double _sparse_dot(const int *x_indices, const double *x_values, int x_len, const double *y);


/**
 * A full vector y + a scaled sparse vector x, i.e., alpha*x + y --> y
 *
 * ---
 * x is presented as three elements:
 * 1. x_indices: the nonzero indices.
 * 2. x_values: the nonzeros.
 * 3. x_len: the number of nonzeros.
 * For example,
 * x = {0, 1.0, 0, 0, 0, 0.1, 0.5}, then
 * x_indices = {1,5,6}
 * x_values = {1.0,0.1,0.5}
 * x_len = 3.
 * ---
 *
 * @param x_indices: the nonzero indices of the sparse vector x. starts from 0 index.
 * @param x_values: the nonzeros values of the sparse vector x.
 * @param x_len: the number of nonzeros in sparse vector x.
 * @param y
 * @return
 */
void _sparse_cblas_daxpy(const int *x_indices, const double *x_values, int x_len,
                         double alpha, double *y);

/**
 * Given the unsorted array, we threshold this array by using Floyd-Rivest algorithm.
 * @param arr the unsorted array.
 * @param n, the number of elements in this array.
 * @param k, the number of k largest elements will be kept.
 * @return 0, successfully project arr to a k-sparse vector.
 */
int _hard_thresholding(double *arr, int n, int k);


/**
 * Please find the algorithm in the following paper:
 * ---
 * @article{floyd1975algorithm,
 * title={Algorithm 489: the algorithm SELECT—for finding the ith
 *        smallest of n elements [M1]},
 * author={Floyd, Robert W and Rivest, Ronald L},
 * journal={Communications of the ACM},
 * volume={18}, number={3}, pages={173},
 * year={1975},
 * publisher={ACM}}
 * @param array
 * @param l
 * @param r
 * @param k
 */
void _floyd_rivest_select(double *array, int l, int r, int k);

/**
 * SOLAM: Stochastic Online AUC Maximization
 * @para para: number of parameters of SOLAM
 * @para results: the results, wt, a, b of SOLAM
 *
 * ---
 * BibTEX:
 * @inproceedings{ying2016stochastic,
 * title={Stochastic online AUC maximization},
 * author={Ying, Yiming and Wen, Longyin and Lyu, Siwei},
 * booktitle={Advances in neural information processing systems},
 * pages={451--459},
 * year={2016}
 * }
 */
bool algo_solam(solam_para *para, solam_results *results);

/**
 * SOLAM: Stochastic Online AUC Maximization for sparse data
 * @para para: number of parameters of SOLAM
 * @para results: the results, wt, a, b of SOLAM
 *
 * ---
 * BibTEX:
 * @inproceedings{ying2016stochastic,
 * title={Stochastic online AUC maximization},
 * author={Ying, Yiming and Wen, Longyin and Lyu, Siwei},
 * booktitle={Advances in neural information processing systems},
 * pages={451--459},
 * year={2016}
 * }
 */
bool algo_solam_sparse(solam_para_sparse *para, solam_results *results);


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

bool algo_stoht_am(stoht_am_para *para, stoht_am_results *results);

bool algo_stoht_am_sparse(stoht_am_sparse_para *para, stoht_am_results *results);


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
} da_solam_para;

typedef struct {
    double *wt;
    double a;
    double b;
    double alpha;
} da_solam_results;

bool algo_da_solam_func(da_solam_para *para, da_solam_results *results);

typedef struct {
    double *x_tr;
    double *y_tr;
    int *para_rand_ind;


    ////////////////////////////////////
    /**
     * In some cases, the dataset is sparse.
     * We will use sparse representation to save memory.
     * sparse_x_values:
     *      matrix of nonzeros. Notice: first element of each row is the len
     * sparse_x_indices:
     *      matrix of nonzeros indices. Notice: first element of each row is the len
     * the number of columns in this sparse matrix.
     */
    double *sparse_x_values;
    int *sparse_x_indices;
    int sparse_p;
    bool is_sparse; // to check the data is sparse or not.
    ////////////////////////////////////

    int p;
    int num_tr;
    int num_passes; // number of epochs of the processing. default is one.
    double para_l2_reg; // regularization parameter for l2-norm
    double para_l1_reg; // regularization parameter for l1-norm
    double para_xi;     // the constant factor of the step size.
    int verbose;
} spam_para;

typedef struct {
    double *wt;
    double *wt_bar;
} spam_results;

/**
 *
 * This function implements the algorithm proposed in the following paper.
 * Stochastic Proximal Algorithms for AUC Maximization.
 * ---
 * @inproceedings{natole2018stochastic,
 * title={Stochastic proximal algorithms for AUC maximization},
 * author={Natole, Michael and Ying, Yiming and Lyu, Siwei},
 * booktitle={International Conference on Machine Learning},
 * pages={3707--3716},
 * year={2018}}
 * ---
 *
 *
 * Info
 * ---
 * Do not use the function directly. Instead, call it by Python Wrapper.
 *
 * @param para: related input parameters.
 * @param results
 * @author Baojian Zhou(Email: bzhou6@albany.edu)
 * @return
 */
bool algo_spam(spam_para *para, spam_results *results);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H
