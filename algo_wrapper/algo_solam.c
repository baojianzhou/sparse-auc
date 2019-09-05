//
// Created by baojian on 8/24/19.
//
#include "algo_base.h"
#include "algo_solam.h"


double _sparse_dot(const int *x_indices, const double *x_values, int x_len, const double *y) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x_values[i] * y[x_indices[i]];
    }
    return result;
}


void _sparse_cblas_daxpy(const int *x_indices, const double *x_values, int x_len,
                         double alpha, double *y) {
    for (int i = 0; i < x_len; i++) {
        y[x_indices[i]] += alpha * x_values[i];
    }
}


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
bool algo_solam_func(solam_para *para, solam_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    int *rand_id = para->para_rand_ind;
    int num_tr = para->num_tr;
    double *zero_v = malloc(sizeof(double) * (para->p + 2));
    double *one_v = malloc(sizeof(double) * (para->p + 2));
    for (int i = 0; i < para->p + 2; i++) {
        zero_v[i] = 0.0;
        one_v[i] = 1.0;
    }
    double *x_train = para->x_tr;
    double *y_train = para->y_tr;

    // start of the algorithm
    double sr = para->para_r;
    double sc = para->para_xi;
    int n_pass = para->para_num_pass;
    int n_dim = para->p;
    double n_p0_ = 0.; // number of positive
    double *n_v0_ = malloc(sizeof(double) * (n_dim + 2));
    cblas_dcopy(n_dim + 2, zero_v, 1, n_v0_, 1);
    double n_a_p0_ = 0.;
    double n_g_a0_ = 0.;
    // initial vector
    double *n_v0 = malloc(sizeof(double) * (n_dim + 2));
    cblas_dcopy(n_dim, one_v, 1, n_v0, 1);
    cblas_dscal(n_dim, sqrt(sr * sr / (n_dim * 1.0)), n_v0, 1);
    n_v0[n_dim] = sr;
    n_v0[n_dim + 1] = sr;
    double n_a_p0 = 2. * sr;
    // iteration time.
    double n_t = 1.;
    int n_cnt = 1;
    double *v_wt = malloc(sizeof(double) * n_dim);
    double *v_p_dv = malloc(sizeof(double) * (n_dim + 2));
    double *n_v1 = malloc(sizeof(double) * (n_dim + 2));
    double *n_v1_ = malloc(sizeof(double) * (n_dim + 2));
    double v_p_da;
    double n_a_p1;
    double n_a_p1_;
    double n_p1_;
    while (true) {
        if (n_cnt > n_pass) {
            break;
        }
        for (int j = 0; j < num_tr; j++) {
            double *t_feat = x_train + rand_id[j] * n_dim;
            double *t_label = y_train + rand_id[j];
            double n_ga = sc / sqrt(n_t);
            if (*t_label > 0) { // if it is positive case
                n_p1_ = ((n_t - 1.) * n_p0_ + 1.) / n_t;
                cblas_dcopy(n_dim, n_v0, 1, v_wt, 1);
                double n_a = n_v0[n_dim];
                cblas_dcopy(n_dim + 2, zero_v, 1, v_p_dv, 1);
                double vt_dot = cblas_ddot(n_dim, v_wt, 1, t_feat, 1);
                double weight =
                        2. * (1. - n_p1_) * (vt_dot - n_a) - 2. * (1. + n_a_p0) * (1. - n_p1_);
                cblas_daxpy(n_dim, weight, t_feat, 1, v_p_dv, 1);
                v_p_dv[n_dim] = -2. * (1. - n_p1_) * (vt_dot - n_a);
                v_p_dv[n_dim + 1] = 0.;
                cblas_dscal(n_dim + 2, -n_ga, v_p_dv, 1);
                cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, v_p_dv, 1);
                v_p_da = -2. * (1. - n_p1_) * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                v_p_da = n_a_p0 + n_ga * v_p_da;
            } else {
                n_p1_ = ((n_t - 1.) * n_p0_) / n_t;
                cblas_dcopy(n_dim, n_v0, 1, v_wt, 1);
                double n_b = n_v0[n_dim + 1];
                cblas_dcopy(n_dim + 2, zero_v, 1, v_p_dv, 1);
                double vt_dot = cblas_ddot(n_dim, v_wt, 1, t_feat, 1);
                double weight = 2. * n_p1_ * (vt_dot - n_b) + 2. * (1. + n_a_p0) * n_p1_;
                cblas_daxpy(n_dim, weight, t_feat, 1, v_p_dv, 1);
                v_p_dv[n_dim] = 0.;
                v_p_dv[n_dim + 1] = -2. * n_p1_ * (vt_dot - n_b);
                cblas_dscal(n_dim + 2, -n_ga, v_p_dv, 1);
                cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, v_p_dv, 1);
                v_p_da = 2. * n_p1_ * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                v_p_da = n_a_p0 + n_ga * v_p_da;
            }
            // normalization -- the projection step.
            double n_rv = sqrt(cblas_ddot(n_dim, v_p_dv, 1, v_p_dv, 1));
            if (n_rv > sr) {
                cblas_dscal(n_dim, 1. / n_rv * sr, v_p_dv, 1);
            }
            if (v_p_dv[n_dim] > sr) {
                v_p_dv[n_dim] = sr;
            }
            if (v_p_dv[n_dim + 1] > sr) {
                v_p_dv[n_dim + 1] = sr;
            }
            cblas_dcopy(n_dim + 2, v_p_dv, 1, n_v1, 1); //n_v1 = v_p_dv
            double n_ra = fabs(v_p_da);
            if (n_ra > 2. * sr) {
                n_a_p1 = v_p_da / n_ra * (2. * sr);
            } else {
                n_a_p1 = v_p_da;
            }
            // update gamma_
            double n_g_a1_ = n_g_a0_ + n_ga;
            // update v_
            cblas_dcopy(n_dim + 2, n_v0, 1, n_v1_, 1);
            cblas_dscal(n_dim + 2, n_ga / n_g_a1_, n_v1_, 1);
            cblas_daxpy(n_dim + 2, n_g_a0_ / n_g_a1_, n_v0_, 1, n_v1_, 1);
            // update alpha_
            n_a_p1_ = (n_g_a0_ * n_a_p0_ + n_ga * n_a_p0) / n_g_a1_;
            // update the information
            n_p0_ = n_p1_;
            cblas_dcopy(n_dim + 2, n_v1_, 1, n_v0_, 1); // n_v0_ = n_v1_;
            n_a_p0_ = n_a_p1_;
            n_g_a0_ = n_g_a1_;
            cblas_dcopy(n_dim + 2, n_v1, 1, n_v0, 1); // n_v0 = n_v1;
            n_a_p0 = n_a_p1;
            // update the counts
            n_t = n_t + 1.;
        }
        n_cnt += 1;
    }
    cblas_dcopy(n_dim, n_v1_, 1, results->wt, 1);
    results->a = n_v1_[n_dim];
    results->b = n_v1_[n_dim + 1];
    free(n_v1_);
    free(n_v1);
    free(n_v0);
    free(n_v0_);
    free(v_p_dv);
    free(v_wt);
    free(one_v);
    free(zero_v);
    return true;
}


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
bool algo_solam_sparse_func(solam_para_sparse *para, solam_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    int *rand_id = para->para_rand_ind;
    int num_tr = para->num_tr;
    double *zero_v = malloc(sizeof(double) * (para->p + 2));
    double *one_v = malloc(sizeof(double) * (para->p + 2));
    for (int i = 0; i < para->p + 2; i++) {
        zero_v[i] = 0.0;
        one_v[i] = 1.0;
    }
    int *x_train_indices = para->x_tr_indices;
    double *x_train_values = para->x_tr_values;
    double *y_train = para->y_tr;

    // start of the algorithm
    double sr = para->para_r;
    double sc = para->para_xi;
    int n_pass = para->para_num_pass;
    int n_dim = para->p;
    double n_p0_ = 0.; // number of positive
    double *n_v0_ = malloc(sizeof(double) * (n_dim + 2));
    cblas_dcopy(n_dim + 2, zero_v, 1, n_v0_, 1);
    double n_a_p0_ = 0.;
    double n_g_a0_ = 0.;
    // initial vector
    double *n_v0 = malloc(sizeof(double) * (n_dim + 2));
    cblas_dcopy(n_dim, one_v, 1, n_v0, 1);
    cblas_dscal(n_dim, sqrt(sr * sr / (n_dim * 1.0)), n_v0, 1);
    n_v0[n_dim] = sr;
    n_v0[n_dim + 1] = sr;
    printf("n_v0: %.4f\n", sqrt(cblas_ddot(n_dim + 2, n_v0, 1, n_v0, 1)));
    double n_a_p0 = 2. * sr;
    // iteration time.
    double n_t = 1.;
    int n_cnt = 1;
    double *v_wt = malloc(sizeof(double) * n_dim);
    double *v_p_dv = malloc(sizeof(double) * (n_dim + 2));
    double *n_v1 = malloc(sizeof(double) * (n_dim + 2));
    double *n_v1_ = malloc(sizeof(double) * (n_dim + 2));
    double v_p_da;
    double n_a_p1;
    double n_a_p1_;
    double n_p1_;
    //printf("n_dim: %d\n", n_dim);
    while (true) {
        if (n_cnt > n_pass) {
            break;
        }
        for (int j = 0; j < num_tr; j++) {
            int *t_feat_indices = x_train_indices + j * para->max_nonzero;
            double *t_feat_values = x_train_values + j * para->max_nonzero;
            int s_len = t_feat_indices[0];
            double *t_label = y_train + j;
            double n_ga = sc / sqrt(n_t);
            //printf("j: %d s_len: %d\n", j, s_len);
            if (*t_label > 0) { // if it is positive case
                n_p1_ = ((n_t - 1.) * n_p0_ + 1.) / n_t;
                cblas_dcopy(n_dim, n_v0, 1, v_wt, 1);
                double n_a = n_v0[n_dim];
                cblas_dcopy(n_dim + 2, zero_v, 1, v_p_dv, 1);
                double vt_dot = _sparse_dot(t_feat_indices + 1, t_feat_values + 1, s_len, v_wt);
                double weight = 2. * (1. - n_p1_) * (vt_dot - n_a);
                weight -= 2. * (1. + n_a_p0) * (1. - n_p1_);
                _sparse_cblas_daxpy(t_feat_indices + 1, t_feat_values + 1, s_len, weight, v_p_dv);
                v_p_dv[n_dim] = -2. * (1. - n_p1_) * (vt_dot - n_a);
                v_p_dv[n_dim + 1] = 0.;
                cblas_dscal(n_dim + 2, -n_ga, v_p_dv, 1);
                cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, v_p_dv, 1);
                v_p_da = -2. * (1. - n_p1_) * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                if(false){
                    printf("posi v_p_da: %.4f vt_dot: %.4f n_p1_:%.4f n_a_p0: %.4f n_t: %.4f,"
                           "weight: %.4f ||v||: %.4f na: %.4f\n",
                           v_p_da, vt_dot, n_p1_, n_a_p0, n_t, weight,
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)), n_a);
                }
                v_p_da = n_a_p0 + n_ga * v_p_da;
            } else {
                n_p1_ = ((n_t - 1.) * n_p0_) / n_t;
                cblas_dcopy(n_dim, n_v0, 1, v_wt, 1);
                double n_b = n_v0[n_dim + 1];
                cblas_dcopy(n_dim + 2, zero_v, 1, v_p_dv, 1);
                double vt_dot = _sparse_dot(t_feat_indices + 1, t_feat_values + 1, s_len, v_wt);
                double weight = 2. * n_p1_ * (vt_dot - n_b) + 2. * (1. + n_a_p0) * n_p1_;
                if(false){
                    printf("s_len: %d, cur_indices: %d cur_values: %.4f, ||v||: %.4f \n",
                           s_len, *(t_feat_indices + 1), *(t_feat_values + 1),
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
                }
                _sparse_cblas_daxpy(t_feat_indices + 1, t_feat_values + 1, s_len, weight, v_p_dv);
                if(false){
                    printf("s_len: %d, cur_indices: %d cur_values: %.4f, ||v||: %.4f \n",
                           s_len, *(t_feat_indices + 1), *(t_feat_values + 1),
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
                }
                v_p_dv[n_dim] = 0.;
                v_p_dv[n_dim + 1] = -2. * n_p1_ * (vt_dot - n_b);
                cblas_dscal(n_dim + 2, -n_ga, v_p_dv, 1);
                cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, v_p_dv, 1);
                v_p_da = 2. * n_p1_ * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                if(false){
                    printf("nega v_p_da: %.4f vt_dot: %.4f n_p1_:%.4f n_a_p0: %.4f n_t: %.4f,"
                           "weight: %.4f ||v||: %.4f nb: %.4f\n",
                           v_p_da, vt_dot, n_p1_, n_a_p0, n_t, weight,
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)), n_b);
                }
                v_p_da = n_a_p0 + n_ga * v_p_da;
            }

            if(false){
                printf("lr: %.4f alpha: %.4f sr: %.4f sc:%.4f n_p1: %.4f label: %.1f\n",
                       n_ga, v_p_da, sr, sc, n_p1_, *t_label);
                if (j == 10) {
                    return false;
                }
            }

            // normalization -- the projection step.
            double n_rv = sqrt(cblas_ddot(n_dim, v_p_dv, 1, v_p_dv, 1));
            if (n_rv > sr) {
                cblas_dscal(n_dim, 1. / n_rv * sr, v_p_dv, 1);
            }
            if (v_p_dv[n_dim] > sr) {
                v_p_dv[n_dim] = sr;
            }
            if (v_p_dv[n_dim + 1] > sr) {
                v_p_dv[n_dim + 1] = sr;
            }
            cblas_dcopy(n_dim + 2, v_p_dv, 1, n_v1, 1); //n_v1 = v_p_dv


            double n_ra = fabs(v_p_da);
            if (n_ra > 2. * sr) {
                n_a_p1 = v_p_da / n_ra * (2. * sr);
            } else {
                n_a_p1 = v_p_da;
            }

            // update gamma_
            double n_g_a1_ = n_g_a0_ + n_ga;

            // update v_
            cblas_dcopy(n_dim + 2, n_v0, 1, n_v1_, 1);
            cblas_dscal(n_dim + 2, n_ga / n_g_a1_, n_v1_, 1);
            cblas_daxpy(n_dim + 2, n_g_a0_ / n_g_a1_, n_v0_, 1, n_v1_, 1);

            // update alpha_
            n_a_p1_ = (n_g_a0_ * n_a_p0_ + n_ga * n_a_p0) / n_g_a1_;

            // update the information
            n_p0_ = n_p1_;
            cblas_dcopy(n_dim + 2, n_v1_, 1, n_v0_, 1); // n_v0_ = n_v1_;
            n_a_p0_ = n_a_p1_;
            n_g_a0_ = n_g_a1_;
            cblas_dcopy(n_dim + 2, n_v1, 1, n_v0, 1); // n_v0 = n_v1;
            n_a_p0 = n_a_p1;

            // update the counts
            n_t = n_t + 1.;
        }
        n_cnt += 1;
    }
    cblas_dcopy(n_dim, n_v1_, 1, results->wt, 1);
    results->a = n_v1_[n_dim];
    results->b = n_v1_[n_dim + 1];
    free(n_v1_);
    free(n_v1);
    free(n_v0);
    free(n_v0_);
    free(v_p_dv);
    free(v_wt);
    free(one_v);
    free(zero_v);
    return true;
}

