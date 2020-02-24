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
    double *wt;
    double *wt_prev;
    double *aucs;
    double *rts;
    int auc_len; // how many auc evaluated.
    int total_iterations; // total iterations
    int total_epochs; // total epochs executed.
} AlgoResults;


typedef struct {
    int num_passes;
    int verbose;
    int step_len;
    int record_aucs;
    double stop_eps;
} GlobalParas;

AlgoResults *make_algo_results(int data_p, int total_num_eval);

bool free_algo_results(AlgoResults *re);


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
    const double *x_tr_vals;
    const int *x_tr_inds;
    const int *x_tr_poss;
    const int *x_tr_lens;
    const double *y_tr;
    bool is_sparse;
    int n;
    int p;
    // this is only for the graph operator.
    bool is_graph;
    int m; // number of edges.
    EdgePair *edges;
    double *weights;
    int g;
    double *proj_prizes;
    GraphStat *graph_stat;
} Data;

GraphStat *make_graph_stat(int p, int m);

bool free_graph_stat(GraphStat *graph_stat);

typedef struct {
    double val;
    int index;
} data_pair;

bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat);

/**
 * SOLAM: Stochastic Online AUC Maximization
 * ---
 * BibTEX:
 * @inproceedings{ying2016stochastic,
 * title={Stochastic online AUC maximization},
 * author={Ying, Yiming and Wen, Longyin and Lyu, Siwei},
 * booktitle={Advances in neural information processing systems},
 * pages={451--459},
 * year={2016}
 * }
 * @param data
 * @param paras
 * @param re
 * @param para_xi
 * @param para_r
 * @author --- (Email: ---)
 * @return
 */
bool _algo_solam(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_xi,
                 double para_r);

/**
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
 * @param data
 * @param paras
 * @param re
 * @param para_xi
 * @param para_l1_reg
 * @param para_l2_reg
 * @author --- (Email: ---)
 */
void _algo_spam(Data *data,
                GlobalParas *paras,
                AlgoResults *re,
                double para_xi,
                double para_l1_reg,
                double para_l2_reg);

/**
 * Stochastic Hard Thresholding for AUC maximization.
 * @param data
 * @param paras
 * @param re
 * @param para_s
 * @param para_b
 * @param para_c
 * @param para_l2_reg
 */
void _algo_sht_auc(Data *data,
                   GlobalParas *paras,
                   AlgoResults *re,
                   int version,
                   int operator_id,
                   int para_s,
                   int para_b,
                   double para_c,
                   double para_l2_reg);

void _algo_sto_iht(Data *data,
                   GlobalParas *paras,
                   AlgoResults *re,
                   int para_s,
                   int para_b,
                   double para_xi,
                   double para_l2_reg);

/**
 *
 * @param data
 * @param paras
 * @param re
 * @param para_s
 * @param para_tau
 * @param para_zeta
 * @param para_step_init
 * @param para_l2
 */
void _algo_hsg_ht(Data *data,
                  GlobalParas *paras,
                  AlgoResults *re,
                  int para_s,
                  double para_tau,
                  double para_zeta,
                  double para_step_init,
                  double para_l2);

int _hard_thresholding(double *arr, int n, int k);

/**
 *
 * @param data
 * @param paras
 * @param re
 * @param para_tau
 * @param para_eta
 * @param para_lambda
 */
void _algo_opauc(Data *data,
                 GlobalParas *paras,
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
 * @param data
 * @param paras
 * @param re
 * @param para_r
 * @param para_g
 */
void _algo_fsauc(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_r,
                 double para_g);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H
