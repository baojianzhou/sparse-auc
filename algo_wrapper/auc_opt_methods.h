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
#include "fast_pcst.h"

#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }

typedef struct {
    Array *nodes;
    Array *edges;
    double prize;
    double cost;
} Tree;

typedef struct {
    Array *re_nodes;
    Array *re_edges;
    double *prizes;
    double *costs;
    int num_pcst;
    double run_time;
    int num_iter;
} GraphStat;

typedef struct {
    double val;
    int index;
} data_pair;

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
    double *wt_bar;
    double a;
    double b;
    double alpha;
} solam_results;


GraphStat *make_graph_stat(int p, int m);

bool free_graph_stat(GraphStat *graph_stat);

bool head_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

bool head_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

bool tail_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

bool tail_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);


bool cluster_grid_pcst(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, double lambda,
        int root, PruningMethod pruning, int verbose,
        GraphStat *stat);

bool cluster_grid_pcst_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat);

bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat);


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
bool __solam(solam_para *para, solam_results *results);

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
bool __solam_sparse(solam_para_sparse *para, solam_results *results);


typedef struct {
    double *x_tr;
    double *y_tr;
    int p;
    int num_tr;
    int verbose;
    int *para_rand_ind; // to random shuffle the training samples
    double para_xi; // the parameter xi, to control the learning rate.
    double para_r; // the parameter R, to control the wt, the radius of the ball of wt.
    int para_s; // para_sparsity
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
    int *sparse_x_positions;
    int *sparse_x_len_list;
    bool is_sparse; // to check the data is sparse or not.
    ////////////////////////////////////

    int p;
    int num_tr;
    int num_classes;
    double para_xi;     // the constant factor of the step size.
    double para_l2_reg; // regularization parameter for l2-norm
    double para_l1_reg; // regularization parameter for l1-norm
    int para_num_passes; // number of epochs of the processing. default is one.
    int para_step_len;
    int para_reg_opt; // option of regularization: 0: l2^2 1: l1/l2 mixed norm.
    int verbose;
} spam_para;

typedef struct {
    double *wt;
    double *wt_bar;
    double *t_run_time;
    double *t_auc;
    double t_eval_time;
    int *t_indices;
    int t_index;
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


typedef struct {
    double *x_tr;
    double *y_tr;

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
    int *sparse_x_positions;
    int *sparse_x_len_list;
    bool is_sparse; // to check the data is sparse or not.
    ////////////////////////////////////

    int p;
    int para_b; // mini-batch size
    int num_tr;
    int num_classes;
    double para_xi;     // the constant factor of the step size.
    double para_l2_reg; // regularization parameter for l2-norm
    int para_sparsity; // the para_sparsity parameter
    int para_num_passes; // number of epochs of the processing. default is one.
    int para_step_len;
    int verbose;

    int *sub_nodes;
    int nodes_len;
} sht_am_para;

typedef struct {
    double *wt;
    double *wt_bar;
    double *t_run_time;
    double *t_auc;
    double t_eval_time;
    int *t_indices;
    int t_index;
} sht_am_results;

typedef struct {
    double *x_tr;
    double *y_tr;

    EdgePair *edges;
    double *weights;
    int m;

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
    int *sparse_x_positions;
    int *sparse_x_len_list;
    bool is_sparse; // to check the data is sparse or not.
    ////////////////////////////////////



    int p;
    int para_b; // mini-batch size
    int num_tr;
    int num_classes;
    double para_xi;     // the constant factor of the step size.
    double para_l2_reg; // regularization parameter for l2-norm
    int para_sparsity; // the para_sparsity parameter
    int para_num_passes; // number of epochs of the processing. default is one.
    int para_step_len;
    int verbose;

    int *sub_nodes;
    int nodes_len;
} graph_am_para;

typedef struct {
    double *wt;
    double *wt_bar;
    double *t_run_time;
    double *t_auc;
    double t_eval_time;
    int *t_indices;
    int t_index;
} graph_am_results;

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
bool algo_sht_am(sht_am_para *para, sht_am_results *results);


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
bool algo_graph_am(graph_am_para *para, graph_am_results *results);

void algo_opauc(const double *x_tr,
                const double *y_tr,
                int p,
                int n,
                double eta, double lambda, double *wt, double *wt_bar);

/**
 *
 * This function implements the algorithm, FSAUC, proposed in the following paper:
 * ---
 * @inproceedings{liu2018fast,
 * title={Fast stochastic AUC maximization with O (1/n)-convergence rate},
 * author={Liu, Mingrui and Zhang, Xiaoxuan and Chen, Zaiyi and Wang, Xiaoyu and Yang, Tianbao},
 * booktitle={International Conference on Machine Learning},
 * pages={3195--3203},
 * year={2018}}
 * ---
 *
 *
 * ---
 * Do not use the function directly. Instead, call it by Python Wrapper.
 * There are two main model parameters: para_g, para_r.
 * @param x_tr
 * @param y_tr
 * @param p : number of features.
 * @param n : number of samples, i.e. len(x_tr) == n.
 * @param para_g
 * @param para_r
 */
void algo_fsauc(const double *x_tr, const double *y_tr, int p, int n,
                double para_r, double para_g, int num_passes, double *wt, double *wt_bar);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H
