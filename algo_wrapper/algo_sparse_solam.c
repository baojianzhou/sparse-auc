//
// Created by baojian on 8/29/19.
//
#include "sort.h"
#include "algo_sparse_solam.h"

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
void floyd_rivest_select(double *array, int l, int r, int k) {
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
            s = 0.5 * exp(2 * z / 3);
            sd = 0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2);
            ll = max(l, k - i * s / n + sd);
            rr = min(r, k + (n - i) * s / n + sd);
            floyd_rivest_select(array, ll, rr, k);
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
int hard_thresholding(double *arr, int n, int k) {
    double *temp_arr = malloc(sizeof(double) * n), kth_largest;
    for (int i = 0; i < n; i++) {
        temp_arr[i] = fabs(arr[i]);
    }
    floyd_rivest_select(temp_arr, 0, n - 1, k - 1);
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


bool algo_sparse_solam_func(sparse_solam_para *para, sparse_solam_results *results) {

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
            hard_thresholding(v_p_dv, n_dim, para->para_s);
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


