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
#define is_posi(x) ( x > 0.0 ? 1.0 : 0.0)
#define is_nega(x) ( x < 0.0 ? 1.0 : 0.0)

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
    double *vals;
    int *indices;
    int len;
} sparse_arr;


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
bool _algo_solam(const double *data_x_tr,
                 const double *data_y_tr,
                 int data_n,
                 int data_p,
                 double para_xi,
                 double para_r,
                 int para_num_pass,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar);

bool _algo_solam_sparse(const double *x_tr_vals,
                        const int *x_tr_indices,
                        const int *x_tr_posis,
                        const int *x_tr_lens,
                        const double *data_y_tr,
                        int data_n,
                        int data_p,
                        double para_xi,
                        double para_r,
                        int para_num_pass,
                        int para_verbose,
                        double *re_wt,
                        double *re_wt_bar);


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
bool _algo_spam(const double *data_x_tr,
                const double *data_y_tr,
                int data_n,
                int data_p,
                double para_xi,
                double para_l1_reg,
                double para_l2_reg,
                int para_num_passes,
                int para_step_len,
                int para_reg_opt,
                int para_verbose,
                spam_results *results);

bool _algo_spam_sparse(const double *x_values,
                       const int *x_indices,
                       const int *x_positions,
                       const int *x_len_list,
                       const double *y_tr,
                       int data_n,
                       int data_p,
                       double para_xi,
                       double para_l1_reg,
                       double para_l2_reg,
                       int para_num_passes,
                       int para_step_len,
                       int para_reg_opt,
                       int para_verbose,
                       spam_results *results);

typedef struct {
    double *wt;
    double *wt_bar;
    double *t_run_time;
    double *t_auc;
    double t_eval_time;
    int *t_indices;
    int t_index;
} sht_am_results;


bool _algo_sht_am(const double *x_tr,
                  const double *y_tr,
                  int p,
                  int n,
                  int b,
                  double para_xi,
                  double para_l2_reg,
                  int para_sparsity,
                  int para_num_passes,
                  int para_step_len,
                  int para_verbose,
                  sht_am_results *results);

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
bool _algo_sht_am_sparse(const double *x_tr_vals,// the values of these nonzeros.
                         const int *x_tr_indices,  // the inidices of these nonzeros.
                         const int *x_tr_posis,// the start indices of these nonzeros.
                         const int *x_tr_lens, // the list of sizes of nonzeros.
                         const double *y_tr,    // the vector of training samples.
                         int p,                 // the dimension of the features of the dataset
                         int n,                 // the total number of training samples.
                         int b,
                         double para_xi,
                         double para_l2_reg,
                         int para_sparsity,
                         int para_num_passes,
                         int para_step_len,
                         int para_verbose,
                         sht_am_results *results);


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


typedef struct {
    double *wt;
    double *wt_bar;
    double *t_run_time;
    double *t_auc;
    double t_eval_time;
    int *t_indices;
    int t_index;
} graph_am_results;

bool _algo_graph_am(const double *x_tr,
                    const double *y_tr,
                    int p,
                    int n,
                    int b,
                    double para_xi,
                    double para_l2_reg,
                    int para_sparsity,
                    int para_num_passes,
                    int para_step_len,
                    int para_verbose,
                    const EdgePair *edges,
                    const double *weights,
                    int m,
                    graph_am_results *results);

bool _algo_graph_am_sparse(const double *x_values,// the values of these nonzeros.
                           const int *x_indices,  // the inidices of these nonzeros.
                           const int *x_positions,// the start indices of these nonzeros.
                           const int *x_len_list, // the list of sizes of nonzeros.
                           const double *y_tr,    // the vector of training samples.
                           int p,                 // the dimension of the features of the dataset
                           int n,                 // the total number of training samples.
                           int b,
                           int para_sparsity,
                           double para_xi,
                           double para_l2_reg,
                           int num_passes,
                           int step_len,
                           int verbose,
                           graph_am_results *results);

void _algo_opauc(const double *x_tr,
                 const double *y_tr,
                 int p,
                 int n,
                 double eta,
                 double lambda,
                 double *wt,
                 double *wt_bar);


void _algo_opauc_sparse(const double *x_tr_vals,
                        const int *x_tr_indices,
                        const int *x_tr_posis,
                        const int *x_tr_lens,
                        const double *y_tr,
                        int p,
                        int n,
                        double eta,
                        double lambda,
                        double *wt,
                        double *wt_bar);

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
void _algo_fsauc(const double *x_tr,
                 const double *y_tr,
                 int p,
                 int n,
                 double para_r,
                 double para_g,
                 int num_passes,
                 double *wt,
                 double *wt_bar);

void _algo_fsauc_sparse(const double *x_tr_vals,
                        const int *x_tr_indices,
                        const int *x_tr_posis,
                        const int *x_tr_lens,
                        const double *y_tr,
                        int p,
                        int n,
                        double para_r,
                        double para_g,
                        int num_passes,
                        double *wt,
                        double *wt_bar);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H
