//
// Created by baojian on 9/9/19.
//
#include "auc_opt_methods.h"


static inline int __comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

void _arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

/**
 * calculate the TPR, FPR, and AUC score.
 *
 */

void _tpr_fpr_auc(const double *true_labels,
                  const double *scores, int n, double *tpr, double *fpr, double *auc) {
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < n; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * n);
    _arg_sort_descend(scores, sorted_indices, n);
    //TODO assume the score has no -infty
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < n; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    cblas_dscal(n, 1. / num_posi, tpr, 1);
    cblas_dscal(n, 1. / num_nega, fpr, 1);
    //AUC score by
    *auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < n; i++) {
        *auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
    }
    free(sorted_indices);
}

/**
 * Calculate the AUC score.
 * We assume true labels contain only +1,-1
 * We also assume scores are real numbers.
 * @param true_labels
 * @param scores
 * @param len
 * @return AUC score.
 */
double _auc_score(const double *true_labels, const double *scores, int len) {
    double *fpr = malloc(sizeof(double) * (len + 1));
    double *tpr = malloc(sizeof(double) * (len + 1));
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < len; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * len);
    _arg_sort_descend(scores, sorted_indices, len);
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < len; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    cblas_dscal(len, 1. / num_posi, tpr, 1);
    cblas_dscal(len, 1. / num_nega, fpr, 1);
    //AUC score
    double auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < len; i++) {
        auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
    }
    free(sorted_indices);
    free(fpr);
    free(tpr);
    return auc;
}

void _sparse_to_full(const double *sparse_v, const int *sparse_indices,
                     int sparse_len, double *full_v, int full_len) {
    cblas_dscal(full_len, 0.0, full_v, 1);
    for (int i = 0; i < sparse_len; i++) {
        full_v[sparse_indices[i]] = sparse_v[i];
    }
}

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

    //clock()
    long int time_start = clock();

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
 * Stochastic Proximal AUC Maximization with elastic net penalty
 * SPAM algorithm.
 * @param x_tr: The matrix of data samples.
 * @param y_tr: We assume that each y_tr[i] is either +1.0 or -1.0.
 * @param p: >=1 (at least one feature).
 * @param n: >=2 (at least two samples).
 * @param para_num_passes: >=1 (at least pass dataset once)
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
                double para_xi,
                double para_l1_reg,
                double para_l2_reg,
                int para_num_passes,
                int para_step_len,
                int para_reg_opt,
                int para_verbose,
                spam_results *results) {
    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);
    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    for (int i = 0; i < n; i++) {
        const double *cur_xt = x_tr + i * p;
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, cur_xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, cur_xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    for (int i = 0; i < para_num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_tr + j * p;
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = 2. / (para_xi * t + 1.);

            // update a(wt), b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (cblas_ddot(p, wt, 1, cur_xt, 1) - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (cblas_ddot(p, wt, 1, cur_xt, 1) - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (para_verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            // calculate the gradient
            cblas_dcopy(p, cur_xt, 1, grad_wt, 1);
            cblas_dscal(p, weight, grad_wt, 1);

            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

            // elastic net
            if (para_reg_opt == 0) {
                /**
                 * Currently, u is the \hat{wt_{t+1}}, next is to use prox_operator.
                 * The following part of the code is the proximal operator for elastic norm.
                 *
                 * @inproceedings{singer2009efficient,
                 * title={Efficient learning using forward-backward splitting},
                 * author={Singer, Yoram and Duchi, John C},
                 * booktitle={Advances in Neural Information Processing Systems},
                 * pages={495--503},
                 * year={2009}}
                 */
                double tmp_l2 = (eta_t * para_l2_reg + 1.);
                for (int k = 0; k < p; k++) {
                    wt[k] = sign(u[k]) * fmax(0.0, (fabs(u[k]) - eta_t * para_l1_reg) / tmp_l2);
                }
            } else {
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
                cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
                cblas_dcopy(p, u, 1, wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if (fmod(t, para_step_len) == 0.) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            n, p, 1., x_tr, p, wt_bar, 1, 0.0, y_pred, 1);
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (para_verbose > 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}

bool _algo_spam_sparse(const double *x_values,
                       const int *x_indices,
                       const int *x_positions,
                       const int *x_len_list,
                       const double *y_tr,
                       int p,
                       int n,
                       double para_xi,
                       double para_l1_reg,
                       double para_l2_reg,
                       int num_passes,
                       int step_len,
                       int reg_opt,
                       int verbose,
                       spam_results *results) {
    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);
    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    double *full_v = malloc(sizeof(double) * p);
    for (int i = 0; i < n; i++) {
        // get current sample
        const double *cur_xt = x_values + x_positions[i];
        const int *cur_indices = x_indices + x_positions[i];
        _sparse_to_full(cur_xt, cur_indices, x_len_list[i], full_v, p);
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, full_v, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, full_v, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    for (int i = 0; i < num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        double per_s_time = clock();
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_values + x_positions[j];
            const int *cur_indices = x_indices + x_positions[j];
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = 2. / (para_xi * t + 1.);

            // update a(wt), b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            double dot_prod = _sparse_dot(cur_indices, cur_xt, x_len_list[j], wt);
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (dot_prod - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (dot_prod - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            // calculate the gradient
            _sparse_to_full(cur_xt, cur_indices, x_len_list[j], full_v, p);
            cblas_dcopy(p, full_v, 1, grad_wt, 1);
            cblas_dscal(p, weight, grad_wt, 1);

            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

            // elastic net
            if (reg_opt == 0) {
                /**
                 * Currently, u is the \hat{wt_{t+1}}, next is to use prox_operator.
                 * The following part of the code is the proximal operator for elastic norm.
                 *
                 * @inproceedings{singer2009efficient,
                 * title={Efficient learning using forward-backward splitting},
                 * author={Singer, Yoram and Duchi, John C},
                 * booktitle={Advances in Neural Information Processing Systems},
                 * pages={495--503},
                 * year={2009}}
                 */
                double tmp_l2 = (eta_t * para_l2_reg + 1.);
                for (int k = 0; k < p; k++) {
                    wt[k] = sign(u[k]) * fmax(0.0, (fabs(u[k]) - eta_t * para_l1_reg) / tmp_l2);
                }
            } else {
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
                cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
                cblas_dcopy(p, u, 1, wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if (fmod(t, step_len) == 0.) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                for (int q = 0; q < n; q++) {
                    cur_xt = x_values + x_positions[q];
                    cur_indices = x_indices + x_positions[q];
                    y_pred[q] = _sparse_dot(cur_indices, cur_xt, x_len_list[q], wt_bar);
                }
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (verbose == 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
        if (verbose > 0) {
            printf("run time: %.4f\n", (clock() - per_s_time) / CLOCKS_PER_SEC);
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(full_v);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}

bool algo_spam(spam_para *para, spam_results *results) {
    if (para->is_sparse) {
        // sparse case (for sparse data).
        return _algo_spam_sparse(para->sparse_x_values,
                                 para->sparse_x_indices,
                                 para->sparse_x_positions,
                                 para->sparse_x_len_list,
                                 para->y_tr, para->p, para->num_tr, para->para_xi,
                                 para->para_l1_reg, para->para_l2_reg, para->para_num_passes,
                                 para->para_step_len, para->para_reg_opt, para->verbose, results);
    } else {
        // non-sparse case
        return _algo_spam(para->x_tr, para->y_tr, para->p, para->num_tr,
                          para->para_xi, para->para_l1_reg, para->para_l2_reg,
                          para->para_num_passes, para->para_step_len, para->para_reg_opt,
                          para->verbose, results);
    }
}

bool _algo_sht_am(const double *x_tr,
                  const double *y_tr,
                  int p,
                  int n,
                  double para_xi,
                  double para_l2_reg,
                  int para_sparsity,
                  int para_num_passes,
                  int para_step_len,
                  int para_verbose,
                  sht_am_results *results) {

    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);
    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    for (int i = 0; i < n; i++) {
        const double *cur_xt = x_tr + i * p;
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, cur_xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, cur_xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    for (int i = 0; i < para_num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_tr + j * p;
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = 2. / (para_xi * t + 1.);

            // update a(wt), b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (cblas_ddot(p, wt, 1, cur_xt, 1) - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (cblas_ddot(p, wt, 1, cur_xt, 1) - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (para_verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            // calculate the gradient
            cblas_dcopy(p, cur_xt, 1, grad_wt, 1);
            cblas_dscal(p, weight, grad_wt, 1);

            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

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
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, p, para_sparsity); // k-sparse step.
            cblas_dcopy(p, u, 1, wt, 1);

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if (fmod(t, para_step_len) == 0.) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            n, p, 1., x_tr, p, wt_bar, 1, 0.0, y_pred, 1);
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (para_verbose > 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}


bool _algo_sht_am_sparse(const double *x_values,
                         const int *x_indices,
                         const int *x_positions,
                         const int *x_len_list,
                         const double *y_tr,
                         int p,
                         int n,
                         int para_sparsity,
                         double para_xi,
                         double para_l2_reg,
                         int num_passes,
                         int step_len,
                         int verbose,
                         sht_am_results *results) {
    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);
    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    double *full_v = malloc(sizeof(double) * p);
    for (int i = 0; i < n; i++) {
        // get current sample
        const double *cur_xt = x_values + x_positions[i];
        const int *cur_indices = x_indices + x_positions[i];
        _sparse_to_full(cur_xt, cur_indices, x_len_list[i], full_v, p);
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, full_v, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, full_v, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    for (int i = 0; i < num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        double per_s_time = clock();
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_values + x_positions[j];
            const int *cur_indices = x_indices + x_positions[j];
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = 2. / (para_xi * t + 1.);

            // update a(wt), b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            double dot_prod = _sparse_dot(cur_indices, cur_xt, x_len_list[j], wt);
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (dot_prod - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (dot_prod - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            memset(grad_wt, 0, sizeof(double) * p);
            for (int kk = 0; kk < x_len_list[j]; kk++) {
                grad_wt[cur_indices[kk]] = weight * cur_xt[kk];
            }
            if (false) {
                // calculate the gradient
                _sparse_to_full(cur_xt, cur_indices, x_len_list[j], full_v, p);
                cblas_dcopy(p, full_v, 1, grad_wt, 1);
                cblas_dscal(p, weight, grad_wt, 1);
            }

            // gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

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
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, p, para_sparsity); // k-sparse step.
            cblas_dcopy(p, u, 1, wt, 1);

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if (fmod(t, step_len) == 0.) {
                printf("test!\n");
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                for (int q = 0; q < n; q++) {
                    cur_xt = x_values + x_positions[q];
                    cur_indices = x_indices + x_positions[q];
                    y_pred[q] = _sparse_dot(cur_indices, cur_xt, x_len_list[q], wt_bar);
                }
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (verbose == 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
        if (verbose == 0) {
            printf("run time: %.4f\n", (clock() - per_s_time) / CLOCKS_PER_SEC);
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(full_v);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}

bool algo_sht_am(sht_am_para *para, sht_am_results *results) {
    if (para->is_sparse) {
        // sparse case (for sparse data).
        return _algo_sht_am_sparse(para->sparse_x_values,
                                   para->sparse_x_indices,
                                   para->sparse_x_positions,
                                   para->sparse_x_len_list,
                                   para->y_tr,
                                   para->p,
                                   para->num_tr,
                                   para->para_sparsity,
                                   para->para_xi,
                                   para->para_l2_reg,
                                   para->para_num_passes,
                                   para->para_step_len,
                                   para->verbose,
                                   results);
    } else {
        // non-sparse case
        return _algo_sht_am(para->x_tr, para->y_tr, para->p, para->num_tr,
                            para->para_xi, para->para_l2_reg, para->para_sparsity,
                            para->para_num_passes, para->para_step_len, para->verbose, results);
    }
}