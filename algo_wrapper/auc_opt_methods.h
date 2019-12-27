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
#include <limits.h>

// These are the third part library needed.
#include <cblas.h>
#include "fast_pcst.h"
#include "loss.h"

#define PI 3.14159265358979323846
#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }
#define is_posi(x) ( x > 0.0 ? 1.0 : 0.0)
#define is_nega(x) ( x < 0.0 ? 1.0 : 0.0)

typedef struct {
    Array *re_nodes;
    Array *re_edges;
    double *prizes;
    double *costs;
    int num_pcst;
    double run_time;
    int num_iter;
} GraphStat;

GraphStat *make_graph_stat(int p, int m);

bool free_graph_stat(GraphStat *graph_stat);

typedef struct {
    double val;
    int index;
} data_pair;

typedef struct {
    double *wt;
    double *wt_bar;
    double *aucs;
    double *rts;
    int auc_len;
} AlgoResults;

typedef struct {
    const double *x_tr_vals;
    const int *x_tr_inds;
    const int *x_tr_poss;
    const int *x_tr_lens;
    const double *y_tr;
    bool is_sparse;
    int n;
    int p;
} Data;

typedef struct {
    int num_passes;
    int verbose;
    int step_len;
    int record_aucs;
} CommonParas;

AlgoResults *make_algo_results(int data_p, int total_num_eval);

bool free_algo_results(AlgoResults *re);

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
bool _algo_solam(Data *data,
                 CommonParas *paras,
                 AlgoResults *re,
                 double para_xi,
                 double para_r);

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
 * Do not use the function directly. Instead, call it by Python Wrapper.
 *
 * @param para: related input parameters.
 * @param results
 * @author Baojian Zhou(Email: bzhou6@albany.edu)
 * @return
 */
void _algo_spam(Data *data,
                CommonParas *paras,
                AlgoResults *re,
                double para_xi,
                double para_l1_reg,
                double para_l2_reg);

/**
 * Stochastic Hard Thresholding for AUC maximization.
 * @param x_tr_vals
 * @param x_tr_inds
 * @param x_tr_poss
 * @param x_tr_lens
 * @param data_y_tr
 * @param is_sparse
 * @param data_n
 * @param data_p
 * @param para_s
 * @param para_b
 * @param para_c
 * @param para_l2_reg
 * @param para_num_passes
 * @param para_step_len
 * @param para_verbose
 * @param re_wt
 * @param re_wt_bar
 * @param re_auc
 * @param re_rts
 * @param re_len_auc
 */
void _algo_sht_am_v1(Data *data,
                     CommonParas *paras,
                     AlgoResults *re,
                     int para_s,
                     int para_b,
                     double para_c,
                     double para_l2_re);


/**
 * Stochastic Hard Thresholding for AUC maximization.
 * @param data_x_tr
 * @param data_y_tr
 * @param data_n
 * @param data_p
 * @param para_s
 * @param para_b
 * @param para_c
 * @param para_l2_reg
 * @param para_num_passes
 * @param para_step_len
 * @param para_verbose
 * @param re_wt
 * @param re_wt_bar
 * @param re_auc
 */

void _algo_sht_am_v2(Data *data,
                     CommonParas *paras,
                     AlgoResults *re,
                     int para_s,
                     int para_b,
                     double para_c,
                     double para_l2_re);

void _algo_sto_iht(Data *data,
                   CommonParas *paras,
                   AlgoResults *re,
                   int para_s,
                   int para_b,
                   double para_xi,
                   double para_l2_reg);


void _algo_graph_am_v1(Data *data,
                       CommonParas *paras,
                       AlgoResults *re,
                       const EdgePair *edges,
                       const double *weights,
                       int data_m,
                       int para_s,
                       int para_b,
                       double para_c,
                       double para_l2_reg);


void _algo_graph_am_v2(Data *data,
                       CommonParas *paras,
                       AlgoResults *re,
                       const EdgePair *edges,
                       const double *weights,
                       int data_m,
                       int para_s,
                       int para_b,
                       double para_c,
                       double para_l2_reg);

/**
 *
 * @param data_x_tr
 * @param data_y_tr
 * @param data_n
 * @param data_p
 * @param para_s
 * @param is_sparse
 * @param record_aucs
 * @param para_tau
 * @param para_zeta
 * @param para_step_init
 * @param para_step_init
 * @param para_l2
 * @param para_num_passes
 * @param para_verbose
 * @param re_wt
 * @param re_wt_bar
 * @param re_auc
 * @param re_rts
 * @param re_len_auc
 */
void _algo_hsg_ht(Data *data,
                  CommonParas *paras,
                  AlgoResults *re,
                  int para_s,
                  double para_tau,
                  double para_zeta,
                  double para_step_init,
                  double para_l2);

/**
 *
 * @param x_tr_vals
 * @param x_tr_inds
 * @param x_tr_poss
 * @param x_tr_lens
 * @param data_y_tr
 * @param data_n
 * @param data_p
 * @param is_sparse
 * @param record_aucs
 * @param para_tau
 * @param para_eta
 * @param para_lambda
 * @param para_num_passes
 * @param para_step_len
 * @param para_verbose
 * @param re_wt
 * @param re_wt_bar
 * @param re_auc
 * @param re_rts
 * @param re_len_auc
 */
void _algo_opauc(Data *data,
                 CommonParas *paras,
                 AlgoResults *re,
                 int para_tau,
                 double para_eta,
                 double para_lambda);

/**
 * This function implements the algorithm, FSAUC, proposed in the following paper:
 * ---
 * @inproceedings{liu2018fast,
 * title={Fast stochastic AUC maximization with O (1/n)-convergence rate},
 * author={Liu, Mingrui and Zhang, Xiaoxuan and Chen, Zaiyi and Wang, Xiaoyu and Yang, Tianbao},
 * booktitle={International Conference on Machine Learning},
 * pages={3195--3203},
 * year={2018}}
 * ---
 * @param x_tr_vals
 * @param x_tr_inds
 * @param x_tr_poss
 * @param x_tr_lens
 * @param data_y_tr
 * @param data_n
 * @param data_p
 * @param is_sparse
 * @param record_aucs
 * @param para_r
 * @param para_g
 * @param para_num_passes
 * @param para_step_len
 * @param para_verbose
 * @param re_wt
 * @param re_wt_bar
 * @param re_auc
 * @param re_rts
 * @param re_len_auc
 */
void _algo_fsauc(Data *data,
                 CommonParas *paras,
                 AlgoResults *re,
                 double para_r,
                 double para_g);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H