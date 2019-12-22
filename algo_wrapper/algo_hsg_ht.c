//
// Created by baojian on 12/20/19.
//
#include <cblas.h>
#include "algo_hsg_ht.h"
#include "loss.h"
#include "kth_selection.h"

void _algo_hsg_ht(const double *data_x_tr,
                  const double *data_y_tr,
                  int data_n,
                  int data_p,
                  double para_eta,
                  double para_gamma,
                  double para_step_init,
                  double para_step_alg,
                  double para_lambda,
                  double para_tol_optgap,
                  int para_batch_size,
                  int para_max_epoch,
                  double *para_w_init,
                  double para_f_opt,
                  bool para_permute_on,
                  bool para_verbose,
                  bool para_store_w) {
    int epoch = 0;
    int grad_calc_count = 0;
    int infos_iter = epoch;
    int infos_place = 0;
    int *infos_time = malloc(sizeof(int) * 10);
    int *infos_time2 = malloc(sizeof(int) * 10);
    int info_grad_calc_count = grad_calc_count;
    double f_val = 0.0;
    double optgap = f_val - para_f_opt;
    while ((optgap > para_tol_optgap) && (epoch < para_max_epoch)) {
        if(para_permute_on){

        }
    }

    free(infos_time);
    free(infos_time2);
}