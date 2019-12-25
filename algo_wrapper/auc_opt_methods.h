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

typedef struct {
    double val;
    int index;
} data_pair;

GraphStat *make_graph_stat(int p, int m);

bool free_graph_stat(GraphStat *graph_stat);

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
bool _algo_solam(const double *x_tr_vals,
                 const int *x_tr_inds,
                 const int *x_tr_poss,
                 const int *x_tr_lens,
                 const double *data_y_tr,
                 bool is_sparse,
                 int data_n,
                 int data_p,
                 double para_xi,
                 double para_r,
                 int para_num_passes,
                 int para_step_len,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar,
                 double *re_auc,
                 double *re_rts,
                 int *re_len_auc);

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
void _algo_spam(const double *data_x_tr,
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
                double *re_wt,
                double *re_wt_bar,
                double *re_auc,
                double *re_rts,
                int *re_len_auc);

void _algo_spam_sparse(const double *x_tr_vals,
                       const int *x_tr_inds,
                       const int *x_tr_poss,
                       const int *x_tr_lens,
                       const double *data_y_tr,
                       int data_n,
                       int data_p,
                       double para_c,
                       double para_l1_reg,
                       double para_l2_reg,
                       int para_num_passes,
                       int para_step_len,
                       int para_reg_opt,
                       int para_verbose,
                       double *re_wt,
                       double *re_wt_bar,
                       double *re_auc,
                       double *re_rts,
                       int *re_len_auc);

void _algo_sto_iht(const double *data_x_tr,
                   const double *data_y_tr,
                   int data_n,
                   int data_p,
                   int para_s,
                   int para_b,
                   double para_c,
                   double para_l2_reg,
                   int para_num_passes,
                   int para_step_len,
                   int para_verbose,
                   double *re_wt,
                   double *re_wt_bar,
                   double *re_auc,
                   double *re_rts,
                   int *re_len_auc);

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
void _algo_sht_am(const double *data_x_tr,
                  const double *data_y_tr,
                  int data_n,
                  int data_p,
                  int para_s,
                  int para_b,
                  double para_c,
                  double para_l2_reg,
                  int para_num_passes,
                  bool record_aucs,
                  int para_verbose,
                  double *re_wt,
                  double *re_wt_bar,
                  double *re_auc,
                  double *re_rts,
                  int *re_len_auc);

void _algo_sht_am_old(const double *data_x_tr,
                      const double *data_y_tr,
                      int data_n,
                      int data_p,
                      int para_s,
                      int para_b,
                      double para_c,
                      double para_l2_reg,
                      int para_num_passes,
                      int para_step_len,
                      int para_verbose,
                      double *re_wt,
                      double *re_wt_bar,
                      double *re_auc,
                      double *re_rts,
                      int *re_len_auc);

/**
 *
 * @param x_tr_vals
 * @param x_tr_inds
 * @param x_tr_poss
 * @param x_tr_lens
 * @param data_y_tr
 * @param data_tr_n
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
void _algo_sht_am_sparse(const double *x_tr_vals,
                         const int *x_tr_inds,
                         const int *x_tr_poss,
                         const int *x_tr_lens,
                         const double *data_y_tr,
                         const double *x_te_vals,
                         const int *x_te_inds,
                         const int *x_te_poss,
                         const int *x_te_lens,
                         const double *data_y_te,
                         int data_tr_n,
                         int data_te_n,
                         int data_p,
                         int para_s,
                         int para_b,
                         double para_c,
                         double para_l2_reg,
                         int para_num_passes,
                         int para_step_len,
                         int para_verbose,
                         double *re_wt,
                         double *re_wt_bar,
                         double *re_auc,
                         double *re_rts,
                         int *re_len_auc);

void _algo_sht_am_sparse_old(const double *x_tr_vals,
                             const int *x_tr_inds,
                             const int *x_tr_poss,
                             const int *x_tr_lens,
                             const double *data_y_tr,
                             const double *x_te_vals,
                             const int *x_te_inds,
                             const int *x_te_poss,
                             const int *x_te_lens,
                             const double *data_y_te,
                             int data_tr_n,
                             int data_te_n,
                             int data_p,
                             int para_s,
                             int para_b,
                             double para_c,
                             double para_l2_reg,
                             int para_num_passes,
                             int para_step_len,
                             int para_verbose,
                             double *re_wt,
                             double *re_wt_bar,
                             double *re_auc,
                             double *re_rts,
                             int *re_len_auc);


void _algo_graph_am(const double *data_x_tr,
                    const double *data_y_tr,
                    const EdgePair *edges,
                    const double *weights,
                    int data_m,
                    int data_n,
                    int data_p,
                    int para_sparsity,
                    int para_b,
                    double para_xi,
                    double para_l2_reg,
                    int para_num_passes,
                    int para_step_len,
                    int para_verbose,
                    double *re_wt,
                    double *re_wt_bar,
                    double *re_auc,
                    double *re_rts,
                    int *re_len_auc);

void _algo_graph_am_sparse(const double *x_values,
                           const int *x_tr_inds,
                           const int *x_tr_poss,
                           const int *x_len_list,
                           const double *data_y_tr,
                           const EdgePair *edges,
                           const double *weights,
                           int data_m,
                           int data_n,
                           int data_p,
                           int para_s,
                           int para_b,
                           double para_c,
                           double para_l2_reg,
                           int para_num_passes,
                           int para_step_len,
                           int para_verbose,
                           double *re_wt,
                           double *re_wt_bar,
                           double *re_auc,
                           double *re_rts,
                           int *re_len_auc);

void _algo_opauc(const double *data_x_tr,
                 const double *data_y_tr,
                 int data_n,
                 int data_p,
                 double para_eta,
                 double para_lambda,
                 int para_num_passes,
                 int para_step_len,
                 bool para_is_aucs,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar,
                 double *re_auc,
                 double *re_rts,
                 int *re_len_auc);


void _algo_opauc_sparse(const double *x_tr_vals,
                        const int *x_tr_inds,
                        const int *x_tr_poss,
                        const int *x_tr_lens,
                        const double *data_y_tr,
                        int data_n,
                        int data_p,
                        int para_tau,
                        double para_eta,
                        double para_lambda,
                        int para_num_passes,
                        int para_step_len,
                        int para_verbose,
                        double *re_wt,
                        double *re_wt_bar,
                        double *re_auc,
                        double *re_rts,
                        int *re_len_auc);

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
 * Do not use the function directly. Instead, call it by Python Wrapper.
 * There are two main model parameters: para_g, para_r.
 * @param data_x_tr
 * @param data_y_tr
 * @param data_p : number of features.
 * @param data_n : number of samples, i.e. len(x_tr) == n.
 * @param para_g
 * @param para_r
 */
void _algo_fsauc(const double *data_x_tr,
                 const double *data_y_tr,
                 int data_n,
                 int data_p,
                 double para_r,
                 double para_g,
                 int para_num_passes,
                 int para_step_len,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar,
                 double *re_auc,
                 double *re_rts,
                 int *re_len_auc);

void _algo_fsauc_sparse(const double *x_tr_vals,
                        const int *x_tr_inds,
                        const int *x_tr_poss,
                        const int *x_tr_lens,
                        const double *data_y_tr,
                        int data_n,
                        int data_p,
                        double para_r,
                        double para_g,
                        int para_num_passes,
                        int para_step_len,
                        int para_verbose,
                        double *re_wt,
                        double *re_wt_bar,
                        double *re_auc,
                        double *re_rts,
                        int *re_len_auc);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H
/**
 * ell_2 regularization option proposed in the following paper:
 *
 * @inproceedings{singer2009efficient,
 * title={Efficient learning using forward-backward splitting},
 * author={Singer, Yoram and Duchi, John C},
 * booktitle={Advances in Neural Information Processing Systems},
 * pages={495--503},
 * year={2009}}
 */