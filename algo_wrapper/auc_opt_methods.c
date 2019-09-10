//
// Created by baojian on 9/9/19.
//
#include "auc_opt_methods.h"


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
double _sparse_dot(const int *x_indices, const double *x_values, int x_len, const double *y) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x_values[i] * y[x_indices[i]];
    }
    return result;
}

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
                         double alpha, double *y) {
    for (int i = 0; i < x_len; i++) {
        y[x_indices[i]] += alpha * x_values[i];
    }
}


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
void _floyd_rivest_select(double *array, int l, int r, int k) {
    register int n, i, j, s, sd, ll, rr;
    register double z, t;
    while (r > l) {
        if (r - l > 600) {
            /**
             * use select() recursively on a sample of size s to get an
             * estimate for the (k-l+1)-th smallest element into array[k],
             * biased slightly so that the (k-l+1)-th element is expected to
             * lie in the smaller set after partitioning.
             */
            n = r - l + 1;
            i = k - l + 1;
            z = log(n);
            s = (int) (0.5 * exp(2 * z / 3));
            sd = (int) (0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2));
            ll = max(l, k - i * s / n + sd);
            rr = min(r, k + (n - i) * s / n + sd);
            _floyd_rivest_select(array, ll, rr, k);
        }
        t = array[k];
        /**
         * the following code partitions x[l:r] about t, it is similar to partition
         * but will run faster on most machines since subscript range checking on i
         * and j has been eliminated.
         */
        i = l;
        j = r;
        swap(array[l], array[k]);
        if (array[r] < t) {
            swap(array[r], array[l]);
        }
        while (i < j) {
            swap(array[i], array[j]);
            do i++; while (array[i] > t);
            do j--; while (array[j] < t);
        }
        if (array[l] == t) {
            swap(array[l], array[j]);
        } else {
            j++;
            swap(array[j], array[r]);
        }
        /**
         * New adjust l, r so they surround the subset containing the
         * (k-l+1)-th smallest element.
         */
        if (j <= k) {
            l = j + 1;
        }
        if (k <= j) {
            r = j - 1;
        }
    }
}


/**
 * Given the unsorted array, we threshold this array by using Floyd-Rivest algorithm.
 * @param arr the unsorted array.
 * @param n, the number of elements in this array.
 * @param k, the number of k largest elements will be kept.
 * @return 0, successfully project arr to a k-sparse vector.
 */
int _hard_thresholding(double *arr, int n, int k) {
    double *temp_arr = malloc(sizeof(double) * n), kth_largest;
    for (int i = 0; i < n; i++) {
        temp_arr[i] = fabs(arr[i]);
    }
    _floyd_rivest_select(temp_arr, 0, n - 1, k - 1);
    kth_largest = temp_arr[k - 1];
    bool flag = false;
    for (int i = 0; i < n; i++) {
        if (fabs(arr[i]) < kth_largest) {
            arr[i] = 0.0;
        } else if ((fabs(arr[i]) == kth_largest) && (flag == false)) {
            flag = true; // to handle the multiple cases.
        } else if ((fabs(arr[i]) == kth_largest) && (flag == true)) {
            arr[i] = 0.0;
        }
    }
    free(temp_arr);
    return 0;
}


bool algo_solam(solam_para *para, solam_results *results) {

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

bool algo_solam_sparse(solam_para_sparse *para, solam_results *results) {

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
    // printf("n_v0: %.4f\n", sqrt(cblas_ddot(n_dim + 2, n_v0, 1, n_v0, 1)));
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
                if (false) {
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
                if (false) {
                    printf("s_len: %d, cur_indices: %d cur_values: %.4f, ||v||: %.4f \n",
                           s_len, *(t_feat_indices + 1), *(t_feat_values + 1),
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
                }
                _sparse_cblas_daxpy(t_feat_indices + 1, t_feat_values + 1, s_len, weight, v_p_dv);
                if (false) {
                    printf("s_len: %d, cur_indices: %d cur_values: %.4f, ||v||: %.4f \n",
                           s_len, *(t_feat_indices + 1), *(t_feat_values + 1),
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
                }
                v_p_dv[n_dim] = 0.;
                v_p_dv[n_dim + 1] = -2. * n_p1_ * (vt_dot - n_b);
                cblas_dscal(n_dim + 2, -n_ga, v_p_dv, 1);
                cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, v_p_dv, 1);
                v_p_da = 2. * n_p1_ * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                if (false) {
                    printf("nega v_p_da: %.4f vt_dot: %.4f n_p1_:%.4f n_a_p0: %.4f n_t: %.4f,"
                           "weight: %.4f ||v||: %.4f nb: %.4f\n",
                           v_p_da, vt_dot, n_p1_, n_a_p0, n_t, weight,
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)), n_b);
                }
                v_p_da = n_a_p0 + n_ga * v_p_da;
            }

            if (false) {
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


bool algo_stoht_am(stoht_am_para *para, stoht_am_results *results) {

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
            //----- sparse projection step
            _hard_thresholding(v_p_dv, n_dim, para->para_s);
            //----- sparse projection step
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

bool algo_stoht_am_sparse(stoht_am_sparse_para *para, stoht_am_results *results) {

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
    // printf("n_v0: %.4f\n", sqrt(cblas_ddot(n_dim + 2, n_v0, 1, n_v0, 1)));
    double n_a_p0 = 2. * sr;
    // iteration time.
    double n_t = 1.;
    int n_cnt = 1;
    double *v_wt = malloc(sizeof(double) * n_dim);
    double *v_p_dv = malloc(sizeof(double) * (n_dim + 2));
    double *n_v1 = malloc(sizeof(double) * (n_dim + 2));
    double *n_v1_ = malloc(sizeof(double) * (n_dim + 2));
    double *block_grad_v = malloc(sizeof(double) * (n_dim + 2));
    double v_p_da;
    double n_a_p1;
    double n_a_p1_;
    double n_p1_ = n_p0_;
    int block_size = 1;
    while (true) {
        if (n_cnt > n_pass) {
            break;
        }
        for (int j = 0; j < num_tr / block_size; j++) {
            int *t_feat_indices = x_train_indices + j * para->max_nonzero * block_size;
            double *t_feat_values = x_train_values + j * para->max_nonzero * block_size;
            double n_ga = sc / sqrt(n_t);

            cblas_dcopy(n_dim, n_v0, 1, v_wt, 1);

            // update gradient
            cblas_dcopy(n_dim + 2, zero_v, 1, block_grad_v, 1);
            double block_grad_alpha = 0.0;
            for (int jj = 0; jj < block_size; jj++) {
                double *cur_label = y_train + j * block_size + jj;
                int *cur_indices = t_feat_indices + para->max_nonzero * jj;
                double *cur_values = t_feat_values + para->max_nonzero * jj;
                int s_len = cur_indices[0];
                double vt_dot = _sparse_dot(cur_indices + 1, cur_values + 1, s_len, v_wt);
                double weight;
                if (*cur_label > 0) {
                    cblas_dcopy(n_dim + 2, zero_v, 1, v_p_dv, 1);
                    n_p1_ = ((n_t - 1.) * n_p0_ + 1.) / n_t;
                    double n_a = n_v0[n_dim];
                    weight = 2. * (1. - n_p1_) * (vt_dot - n_a);
                    weight -= 2. * (1. + n_a_p0) * (1. - n_p1_);
                    // gradient of w
                    _sparse_cblas_daxpy(cur_indices + 1, cur_values + 1, s_len, weight, v_p_dv);
                    // gradient of a
                    v_p_dv[n_dim] = -2. * (1. - n_p1_) * (vt_dot - n_a);
                    // gradient of b
                    v_p_dv[n_dim + 1] = 0.;
                    // gradient of alpha
                    v_p_da = -2. * (1. - n_p1_) * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                    if (false) {
                        printf("posi v_p_da: %.4f vt_dot: %.4f n_p1_:%.4f n_a_p0: %.4f n_t: %.4f,"
                               "weight: %.4f ||v||: %.4f na: %.4f\n",
                               v_p_da, vt_dot, n_p1_, n_a_p0, n_t, weight,
                               sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)), n_a);
                    }
                } else {
                    cblas_dcopy(n_dim + 2, zero_v, 1, v_p_dv, 1);
                    n_p1_ = ((n_t - 1.) * n_p0_) / n_t;
                    double n_b = n_v0[n_dim + 1];
                    weight = 2. * n_p1_ * (vt_dot - n_b) + 2. * (1. + n_a_p0) * n_p1_;
                    // gradient of w
                    if (false) {
                        printf("s_len: %d, cur_indices: %d cur_values: %.4f, ||v||: %.4f \n",
                               s_len, *(cur_indices + 1), *(cur_values + 1),
                               sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
                    }
                    _sparse_cblas_daxpy(cur_indices + 1, cur_values + 1, s_len, weight, v_p_dv);
                    if (false) {
                        printf("s_len: %d, cur_indices: %d cur_values: %.4f, ||v||: %.4f \n",
                               s_len, *(cur_indices + 1), *(cur_values + 1),
                               sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
                    }
                    // gradient of a
                    v_p_dv[n_dim] = 0.;
                    // gradient of b
                    v_p_dv[n_dim + 1] = -2. * n_p1_ * (vt_dot - n_b);
                    // gradient of alpha
                    v_p_da = 2. * n_p1_ * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                }
                cblas_daxpy(n_dim + 2, 1., v_p_dv, 1, block_grad_v, 1);
                block_grad_alpha += v_p_da;
                // update the counts
                n_t = n_t + 1.;
            }
            //gradient descent step of alpha
            cblas_dscal(n_dim + 2, -n_ga, block_grad_v, 1);
            cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, block_grad_v, 1);
            cblas_dcopy(n_dim + 2, block_grad_v, 1, v_p_dv, 1);
            v_p_da = n_a_p0 + n_ga * block_grad_alpha;
            if (false) {
                if (j % 50 == 0) {
                    printf("lr: %.4f alpha: %.4f sr: %.4f sc:%.4f n_p1: %.4f label: %.1f, norm: %.4f\n",
                           n_ga, v_p_da, sr, sc, n_p1_, para->y_tr[j],
                           sqrt(cblas_ddot(n_dim + 2, v_p_dv, 1, v_p_dv, 1)));
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

            //----- sparse projection step
            _hard_thresholding(v_p_dv, n_dim, para->para_s);
            //----- sparse projection step

            cblas_dcopy(n_dim + 2, v_p_dv, 1, n_v1, 1); // n_v1 = v_p_dv

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
        }
        n_cnt += 1;
    }
    cblas_dcopy(n_dim, n_v1_, 1, results->wt, 1);
    results->a = n_v1_[n_dim];
    results->b = n_v1_[n_dim + 1];
    free(block_grad_v);
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


bool algo_da_solam_func(da_solam_para *para, da_solam_results *results) {

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
    double *temp = malloc(sizeof(double) * n_dim);
    double *dual_aver = malloc(sizeof(double) * (n_dim + 2));
    double *bt = malloc(sizeof(double) * (n_dim + 2));
    cblas_dcopy(n_dim + 2, zero_v, 1, dual_aver, 1);
    cblas_dcopy(n_dim + 2, zero_v, 1, bt, 1);
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
                cblas_daxpy(n_dim + 2, 1.0, v_p_dv, 1, dual_aver, 1);
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
                cblas_daxpy(n_dim + 2, 1.0, v_p_dv, 1, dual_aver, 1);
                cblas_dscal(n_dim + 2, -n_ga, v_p_dv, 1);
                cblas_daxpy(n_dim + 2, 1.0, n_v0, 1, v_p_dv, 1);
                v_p_da = 2. * n_p1_ * vt_dot - 2. * n_p1_ * (1. - n_p1_) * n_a_p0;
                v_p_da = n_a_p0 + n_ga * v_p_da;
            }
            // normalization -- the projection step.
            cblas_dcopy(n_dim + 2, dual_aver, 1, bt, 1);
            cblas_dscal(n_dim + 2, -sqrt(n_t) / sc, bt, 1);
            cblas_dcopy(n_dim + 2, bt, 1, v_p_dv, 1);
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

            //----- sparse projection step
            _hard_thresholding(v_p_dv, n_dim, para->para_s);
            //----- sparse projection step

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
    free(bt);
    free(dual_aver);
    free(temp);
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
 * SPAM algorithm.
 * @param x_tr: The matrix of data samples.
 * @param y_tr: We assume that each y_tr[i] is either +1.0 or -1.0.
 * @param p: >=1 (at least one feature).
 * @param n: >=2 (at least two samples).
 * @param num_passes: >=1 (at least pass dataset once)
 * @param para_xi: >0 (constant factor of learning rate).
 * @param para_l1_reg: >=0. (==0.0 without l1-regularization.)
 * @param para_l2_reg: >=0. (==0.0 without l2-regularization.)
 * @param results: wt/wt_bar.
 * @return
 */
bool _algo_spam(const double *x_tr,
                const double *y_tr,
                int p,
                int n,
                int num_passes,
                double para_xi,
                double para_l1_reg,
                double para_l2_reg,
                spam_results *results) {

    // zero vector
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, p * sizeof(double)); //set to zero.
    // unit vector
    double *unit_vector = malloc(sizeof(double) * p);
    for (int i = 0; i < p; i++) {
        unit_vector[i] = 1.0;
    }

    // initialize w1
    double *wt = malloc(sizeof(double) * p); // wt --> 0.0
    double *wt_bar = malloc(sizeof(double) * p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * p);

    //TODO need to update it accordingly.
    cblas_dcopy(p, unit_vector, 1, wt, 1);
    cblas_dscal(p, sqrt(1. / (p * 1.0)), wt, 1);
    cblas_dcopy(p, wt, 1, wt_bar, 1);

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = 0.0;

    // initialize the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt = 0.0; // initialize to zero.
    double *posi_x = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x, 1);


    // initialize the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt = 0.0; // initialize to zero.
    double *nega_x = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x, 1);

    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // learning rate
    double eta_t;

    //for each epoch i.
    double t = 1.0; // initial start time is zero=0.0
    double posi_t = 0.0;
    double nega_t = 0.0;
    for (int i = 0; i < num_passes; i++) {
        // for each training sample j
        for (int j = 0; j < n; j++) {
            //current learning rate
            eta_t = para_xi / sqrt(t);
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_tr + j * p;
            double cur_yt = y_tr[j];

            double weight;
            double dot_prod = cblas_ddot(p, wt, 1, cur_xt, 1);
            prob_p = ((t - 1.) * prob_p) / t;
            cblas_dscal(p, (t - 1.) / t, posi_x, 1);
            cblas_dscal(p, (t - 1.) / t, nega_x, 1);
            if (cur_yt > 0) {
                // update the probability: Pr(y=1)
                prob_p += 1. / t;
                // update positive estimate E[x|y=1]
                cblas_daxpy(p, 1. / t, cur_xt, 1, posi_x, 1);
                // compute a(wt), b(wt), and alpha(wt) by (8) and (9).
                a_wt = cblas_ddot(p, wt, 1, posi_x, 1);
                alpha_wt = b_wt - a_wt;
                weight = 2. * (1.0 - prob_p) * (dot_prod - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                // update negative estimate E[x|y=-1]
                cblas_daxpy(p, 1. / t, cur_xt, 1, nega_x, 1);
                b_wt = cblas_ddot(p, wt, 1, nega_x, 1);
                alpha_wt = b_wt - a_wt;
                weight = 2.0 * prob_p * (dot_prod - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            //calculate the gradient
            cblas_dcopy(p, cur_xt, 1, grad_wt, 1);
            cblas_dscal(p, weight, grad_wt, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, wt, 1);
            // currently, wt is the \hat{wt_{t+1}}, next is prox_operator
            // The following part of the code is the proximal operator for elastic norm.
            // Please see Equation (6.9) of Page 188 in
            // [1] N. Parikh and S. Boyd, Proximal Algorithms
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.0), wt, 1);
            double lambda = (eta_t * para_l1_reg) / (eta_t * para_l2_reg + 1.);
            for (int kk = 0; kk < p; kk++) {
                wt[kk] = fmax(0.0, wt[kk] - lambda) - fmax(0.0, -wt[kk] - lambda);
            }
            t++;
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(nega_x);
    free(posi_x);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(unit_vector);
    free(zero_vector);
    return true;
}

bool _algo_spam_sparse(const double *x_tr_vals,
                       const int *x_tr_indices,
                       int x_sparse_p,
                       const double *y_tr,
                       int p,
                       int n,
                       int num_passes,
                       double para_xi,
                       double para_l1_reg,
                       double para_l2_reg,
                       spam_results *results) {
    return true;
}

bool algo_spam(spam_para *para, spam_results *results) {
    if (para->is_sparse) {
        // non-sparse case
        return _algo_spam(para->x_tr, para->y_tr, para->p, para->num_tr, para->num_passes,
                          para->para_xi, para->para_l1_reg, para->para_l2_reg, results);
    } else {
        // sparse case (for sparse data).
        return _algo_spam_sparse(para->sparse_x_values, para->sparse_x_indices, para->sparse_p,
                                 para->y_tr, para->p, para->num_tr, para->num_passes,
                                 para->para_xi, para->para_l1_reg, para->para_l2_reg, results);
    }
}