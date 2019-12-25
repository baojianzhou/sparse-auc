#include "auc_opt_methods.h"

/**
 * The Box–Muller method uses two independent random
 * numbers U and V distributed uniformly on (0,1).
 * Then the two random variables X and Y.
 * @param n
 * @param samples
 */
void std_normal(int n, double *samples) {
    double epsilon = 2.22507e-308, x, y;
    for (int i = 0; i < n; i++) {
        do {
            x = rand() / (RAND_MAX * 1.);
            y = rand() / (RAND_MAX * 1.);
        } while (x <= epsilon);
        samples[i] = sqrt(-2.0 * log(x)) * cos(2.0 * PI * y);
    }
}

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

GraphStat *make_graph_stat(int p, int m) {
    GraphStat *stat = malloc(sizeof(GraphStat));
    stat->num_pcst = 0;
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->run_time = 0;
    stat->costs = malloc(sizeof(double) * m);
    stat->prizes = malloc(sizeof(double) * p);
    return stat;
}

bool free_graph_stat(GraphStat *graph_stat) {
    free(graph_stat->re_nodes->array);
    free(graph_stat->re_nodes);
    free(graph_stat->re_edges->array);
    free(graph_stat->re_edges);
    free(graph_stat->costs);
    free(graph_stat->prizes);
    free(graph_stat);
    return true;
}


bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {

    // malloc: cur_costs, sorted_prizes, and sorted_indices
    // free: cur_costs, sorted_prizes, and sorted_indices
    double *cur_costs = malloc(sizeof(double) * m);
    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    _arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = fabs(2.0 * sorted_prizes[sorted_indices[guess_pos]]);
    bool using_sparsity_low = false;
    bool using_max_value = false;
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = fabs(2.0 * sorted_prizes[sorted_indices[guess_pos]]);
        if (lambda_high != 0.0) {
            using_sparsity_low = true;
        } else {
            using_max_value = true;
            lambda_high = fabs(prizes[0]);
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, fabs(prizes[ii]));
            }
            lambda_high *= 2.0;
        }
    }
    if (verbose >= 1) {
        const char *sparsity_low_text = "k_low";
        const char *sparsity_high_text = "k_high";
        const char *max_value_text = "max value";
        const char *guess_text = sparsity_high_text;
        if (using_sparsity_low) {
            guess_text = sparsity_low_text;
        } else if (using_max_value) {
            guess_text = max_value_text;
        }
        printf("n = %d  c: %d  k_low: %d  k_high: %d  l_low: %e  l_high: %e  "
               "max_num_iter: %d  (using %s for initial guess).\n",
               n, target_num_clusters, sparsity_low, sparsity_high,
               lambda_low, lambda_high, max_num_iter, guess_text);
    }
    stat->num_iter = 0;
    lambda_high /= 2.0;
    int cur_k;
    do {
        stat->num_iter += 1;
        lambda_high *= 2.0;
        if (lambda_high <= 0.0) { printf("lambda_high: %.6e\n", lambda_high); }
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second, cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) printf("N %d %.15f\n", ii, prizes[ii]);
            printf("\n");
            printf("lambda_high: %f\n", lambda_high);
            printf("target_num_clusters: %d\n", target_num_clusters);
        }
        PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                               1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
    } while (cur_k > sparsity_high && stat->num_iter < max_num_iter);

    if (stat->num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) printf("Found good lambda in exponential increase phase, returning.\n");
        free(cur_costs);
        free(sorted_prizes);
        free(sorted_indices);
        return true;
    }
    double lambda_mid;
    while (stat->num_iter < max_num_iter) {
        stat->num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        if (lambda_mid <= 0.0) { printf("lambda_mid: %.6e\n", lambda_mid); }
        for (int ii = 0; ii < m; ii++) { cur_costs[ii] = lambda_mid * costs[ii]; }

        PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters, 1e-10,
                               pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++)
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second, cur_costs[ii]);
            for (int ii = 0; ii < n; ii++) printf("N %d %.15f\n", ii, prizes[ii]);
            printf("bin_search: l_mid:  %e  k: %d  (lambda_low: %e  lambda_high: %e)\n",
                   lambda_mid, cur_k, lambda_low, lambda_high);
        }
        if (sparsity_low <= cur_k && cur_k <= sparsity_high) {
            if (verbose >= 1) {
                printf("Found good lambda in binary "
                       "search phase, returning.\n");
            }
            free(cur_costs);
            free(sorted_prizes);
            free(sorted_indices);
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    if (lambda_high <= 0.0) { printf("lambda_high: %.6e\n", lambda_high); }
    for (int ii = 0; ii < m; ++ii) { cur_costs[ii] = lambda_high * costs[ii]; }
    PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                           1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges);
    free_pcst(pcst);
    if (verbose >= 1) {
        for (int ii = 0; ii < m; ii++) {
            printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second, cur_costs[ii]);
        }
        printf("\n");
        for (int ii = 0; ii < n; ii++) printf("N %d %.15f\n", ii, prizes[ii]);
        printf("\nReached the maximum number of iterations, using the last l_high: %e  k: %d\n",
               lambda_high, stat->re_nodes->size);
    }
    free(cur_costs);
    free(sorted_prizes);
    free(sorted_indices);
    return true;
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
    for (int i = 0; i < n; i++) {
        if (fabs(arr[i]) <= kth_largest) { arr[i] = 0.0; }
    }
    free(temp_arr);
    return 0;
}

static void _l1ballproj_condat(double *y, double *x, int length, const double a) {
    // This code is implemented by Laurent Condat, PhD, CNRS research fellow in France.
    if (a <= 0.0) {
        if (a == 0.0) memset(x, 0, length * sizeof(double));
        return;
    }
    double *aux = (x == y ? (double *) malloc(length * sizeof(double)) : x);
    int aux_len = 1;
    int aux_len_hold = -1;
    double tau = (*aux = (*y >= 0.0 ? *y : -*y)) - a;
    int i = 1;
    for (; i < length; i++) {
        if (y[i] > 0.0) {
            if (y[i] > tau) {
                if ((tau += ((aux[aux_len] = y[i]) - tau) / (aux_len - aux_len_hold)) <=
                    y[i] - a) {
                    tau = y[i] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        } else if (y[i] != 0.0) {
            if (-y[i] > tau) {
                if ((tau += ((aux[aux_len] = -y[i]) - tau) / (aux_len - aux_len_hold))
                    <= aux[aux_len] - a) {
                    tau = aux[aux_len] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        }
    }
    if (tau <= 0) {    /* y is in the l1 ball => x=y */
        if (x != y) memcpy(x, y, length * sizeof(double));
        else free(aux);
    } else {
        double *aux0 = aux;
        if (aux_len_hold >= 0) {
            aux_len -= ++aux_len_hold;
            aux += aux_len_hold;
            while (--aux_len_hold >= 0)
                if (aux0[aux_len_hold] > tau)
                    tau += ((*(--aux) = aux0[aux_len_hold]) - tau) / (++aux_len);
        }
        do {
            aux_len_hold = aux_len - 1;
            for (i = aux_len = 0; i <= aux_len_hold; i++)
                if (aux[i] > tau)
                    aux[aux_len++] = aux[i];
                else
                    tau += (tau - aux[i]) / (aux_len_hold - i + aux_len);
        } while (aux_len <= aux_len_hold);
        for (i = 0; i < length; i++)
            x[i] = (y[i] - tau > 0.0 ? y[i] - tau : (y[i] + tau < 0.0 ? y[i] + tau : 0.0));
        if (x == y) free(aux0);
    }
}

bool _algo_solam(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, bool is_sparse, int data_n, int data_p, double para_xi, double para_r,
        int para_num_passes, int para_step_len, int para_verbose, double *re_wt, double *re_wt_bar,
        double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    double gamma_bar, gamma_bar_prev = 0.0, alpha_bar, alpha_bar_prev = 0.0, gamma, p_hat = 0.;
    double *v, *v_prev, *v_bar, *v_bar_prev, *y_pred, *grad_v, alpha, alpha_prev;
    double is_p_yt, is_n_yt, vt_dot, wei_posi, wei_nega, weight, t_eval, grad_alpha, norm_v;
    v = malloc(sizeof(double) * (data_p + 2));
    v_prev = malloc(sizeof(double) * (data_p + 2));
    for (int i = 0; i < data_p; i++) { v_prev[i] = sqrt((para_r * para_r) / data_p); }
    v_prev[data_p] = para_r, v_prev[data_p + 1] = para_r;
    alpha_prev = 2. * para_r;
    grad_v = malloc(sizeof(double) * (data_p + 2));
    v_bar = malloc(sizeof(double) * (data_p + 2));
    v_bar_prev = calloc(((unsigned) data_p + 2), sizeof(double));
    y_pred = calloc((size_t) data_n, sizeof(double));
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    if (para_verbose > 0) { printf("n: %d p: %d", data_n, data_p); }
    for (int t = 1; t <= (para_num_passes * data_n); t++) {
        int cur_ind = (t - 1) % data_n;
        const double *xt_vals = x_tr_vals + x_tr_poss[cur_ind];
        const int *xt_inds = x_tr_inds + x_tr_poss[cur_ind]; // current sample
        is_p_yt = is_posi(data_y_tr[cur_ind]);
        is_n_yt = is_nega(data_y_tr[cur_ind]);
        p_hat = ((t - 1.) * p_hat + is_p_yt) / t; // update p_hat
        gamma = para_xi / sqrt(t * 1.); // current learning rate
        if (is_sparse) {
            vt_dot = 0.0;
            memset(grad_v, 0, sizeof(double) * (data_p + 2)); // calculate the gradient w
            for (int kk = 0; kk < x_tr_lens[cur_ind]; kk++) {
                grad_v[xt_inds[kk]] = xt_vals[kk];
                vt_dot += (v_prev[xt_inds[kk]] * xt_vals[kk]);
            }
        } else {
            const double *xt = x_tr_vals + cur_ind * data_p; // current sample
            memcpy(grad_v, xt, sizeof(double) * data_p); // calculate the gradient w
            vt_dot = cblas_ddot(data_p, v_prev, 1, xt, 1);
        }
        wei_posi = 2. * (1. - p_hat) * (vt_dot - v_prev[data_p] - (1. + alpha_prev));
        wei_nega = 2. * p_hat * ((vt_dot - v_prev[data_p + 1]) + (1. + alpha_prev));
        weight = wei_posi * is_p_yt + wei_nega * is_n_yt;
        cblas_dscal(data_p, weight, grad_v, 1);
        grad_v[data_p] = -2. * (1. - p_hat) * (vt_dot - v_prev[data_p]) * is_p_yt; //grad of a
        grad_v[data_p + 1] = -2. * p_hat * (vt_dot - v_prev[data_p + 1]) * is_n_yt; //grad of b
        cblas_dscal(data_p + 2, -gamma, grad_v, 1); // gradient descent step of vt
        cblas_daxpy(data_p + 2, 1.0, v_prev, 1, grad_v, 1);
        memcpy(v, grad_v, sizeof(double) * (data_p + 2));
        wei_posi = -2. * (1. - p_hat) * vt_dot; // calculate the gradient of dual alpha
        wei_nega = 2. * p_hat * vt_dot;
        grad_alpha = wei_posi * is_p_yt + wei_nega * is_n_yt;
        grad_alpha += -2. * p_hat * (1. - p_hat) * alpha_prev;
        alpha = alpha_prev + gamma * grad_alpha; // gradient descent step of alpha
        norm_v = sqrt(cblas_ddot(data_p, v, 1, v, 1)); // projection w
        if (norm_v > para_r) { cblas_dscal(data_p, para_r / norm_v, v, 1); }
        v[data_p] = (v[data_p] > para_r) ? para_r : v[data_p]; // projection a
        v[data_p + 1] = (v[data_p + 1] > para_r) ? para_r : v[data_p + 1]; // projection b
        // projection alpha
        alpha = (fabs(alpha) > 2. * para_r) ? (2. * alpha * para_r) / fabs(alpha) : alpha;
        gamma_bar = gamma_bar_prev + gamma; // update gamma_
        memcpy(v_bar, v_prev, sizeof(double) * (data_p + 2)); // update v_bar
        cblas_dscal(data_p + 2, gamma / gamma_bar, v_bar, 1);
        cblas_daxpy(data_p + 2, gamma_bar_prev / gamma_bar, v_bar_prev, 1, v_bar, 1);
        // update alpha_bar
        alpha_bar = (gamma_bar_prev * alpha_bar_prev + gamma * alpha_prev) / gamma_bar;
        cblas_daxpy(data_p, 1., v_bar, 1, re_wt_bar, 1);
        alpha_prev = alpha, alpha_bar_prev = alpha_bar, gamma_bar_prev = gamma_bar;
        memcpy(v_bar_prev, v_bar, sizeof(double) * (data_p + 2));
        memcpy(v_prev, v, sizeof(double) * (data_p + 2));
        if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
            t_eval = clock();
            if (is_sparse) {
                for (int q = 0; q < data_n; q++) {
                    xt_inds = x_tr_inds + x_tr_poss[q];
                    xt_vals = x_tr_vals + x_tr_poss[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++) {
                        y_pred[q] += (v_bar[xt_inds[tt]] * xt_vals[tt]);
                    }
                }
            } else {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_n, data_p, 1.,
                            x_tr_vals, data_p, re_wt, 1, 0.0, y_pred, 1);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            memset(y_pred, 0, sizeof(double) * data_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    memcpy(re_wt, v_bar, sizeof(double) * data_p);
    cblas_dscal(data_p, 1. / (para_num_passes * data_n), re_wt_bar, 1);
    cblas_dscal(*(re_len_auc), 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(y_pred);
    free(v_bar_prev);
    free(v_bar);
    free(grad_v);
    free(v_prev);
    free(v);
    return true;
}

void _algo_spam(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, bool is_sparse, int data_n, int data_p, double para_xi, double para_l1_reg,
        double para_l2_reg, int para_num_passes, int para_step_len, int para_reg_opt, int para_verbose, double *re_wt,
        double *re_wt_bar, double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double a_wt, *posi_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double b_wt, *nega_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double alpha_wt, posi_t = 0.0, nega_t = 0.0;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    if (is_sparse) {
        for (int i = 0; i < data_n; i++) {
            const int *xt_inds = x_tr_inds + x_tr_poss[i];
            const double *xt_vals = x_tr_vals + x_tr_poss[i];
            if (data_y_tr[i] > 0) {
                posi_t++;
                for (int kk = 0; kk < x_tr_lens[i]; kk++)
                    posi_x_mean[xt_inds[kk]] += xt_vals[kk];
            } else {
                nega_t++;
                for (int kk = 0; kk < x_tr_lens[i]; kk++)
                    nega_x_mean[xt_inds[kk]] += xt_vals[kk];
            }
        }
    } else {
        for (int i = 0; i < data_n; i++) {
            if (data_y_tr[i] > 0) {
                posi_t++;
                cblas_daxpy(data_p, 1., (x_tr_vals + i * data_p), 1, posi_x_mean, 1);
            } else {
                nega_t++;
                cblas_daxpy(data_p, 1., (x_tr_vals + i * data_p), 1, nega_x_mean, 1);
            }
        }
    }
    cblas_dscal(data_p, 1. / posi_t, posi_x_mean, 1);
    cblas_dscal(data_p, 1. / nega_t, nega_x_mean, 1);
    double prob_p = posi_t / (data_n * 1.0), eta_t, t_eval;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    for (int t = 1; t <= (para_num_passes * data_n); t++) {
        const double *xt = x_tr_vals + ((t - 1) % data_n) * data_p;
        const int *xt_inds = x_tr_inds + x_tr_poss[(t - 1) % data_n]; // receive zt=(xt,yt)
        const double *xt_vals = x_tr_vals + x_tr_poss[(t - 1) % data_n];
        eta_t = para_xi / sqrt(t); // current learning rate
        a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
        b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // para_b(wt)
        alpha_wt = b_wt - a_wt; // alpha(wt)
        double wt_dot;
        if (is_sparse) {
            wt_dot = 0.0;
            for (int tt = 0; tt < x_tr_lens[(t - 1) % data_n]; tt++)
                wt_dot += (re_wt[xt_inds[tt]] * xt_vals[tt]);
        } else {
            wt_dot = cblas_ddot(data_p, re_wt, 1, xt, 1);
        }
        double weight = data_y_tr[(t - 1) % data_n] > 0 ?
                        2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                        2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                        2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
        if (is_sparse) {
            for (int tt = 0; tt < x_tr_lens[(t - 1) % data_n]; tt++) // gradient descent
                re_wt[xt_inds[tt]] += -eta_t * weight * xt_vals[tt];
        } else {
            cblas_daxpy(data_p, -eta_t * weight, xt, 1, re_wt, 1); // gradient descent
        }
        if (para_reg_opt == 0) { // elastic-net
            double tmp_demon = (eta_t * para_l2_reg + 1.);
            for (int k = 0; k < data_p; k++) {
                double tmp_sign = (double) sign(re_wt[k]) / tmp_demon;
                re_wt[k] = tmp_sign * fmax(0.0, fabs(re_wt[k]) - eta_t * para_l1_reg);
            }
        } else { // l2-regularization
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
        }
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if ((fmod(t, para_step_len) == 1.)) { // evaluate the AUC score
            t_eval = clock();
            if (is_sparse) {
                for (int q = 0; q < data_n; q++) {
                    xt_inds = x_tr_inds + x_tr_poss[q];
                    xt_vals = x_tr_vals + x_tr_poss[q];
                    y_pred[q] = 0.0;
                    for (int tt = 0; tt < x_tr_lens[q]; tt++)
                        y_pred[q] += re_wt[xt_inds[tt]] * xt_vals[tt];
                }
            } else {
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., x_tr_vals, data_p, re_wt, 1, 0.0, y_pred, 1);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    cblas_dscal(data_p, 1. / (para_num_passes * data_n), re_wt_bar, 1);
    cblas_dscal(*(re_len_auc), 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(y_pred);
    free(nega_x_mean);
    free(posi_x_mean);
    free(grad_wt);
}

void _algo_sht_am(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, bool is_sparse, bool record_aucs, int data_n, int data_p, int para_s, int para_b,
        double para_c, double para_l2_reg, int para_num_passes, int para_verbose, double *re_wt, double *re_wt_bar,
        double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(NULL));
    double *ut = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double *vt = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double *posi_ut = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double *nega_vt = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double posi_t = 0.0, nega_t = 0.0, prob_p, eta_t, t_eval;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    int min_b_ind = 0, max_b_ind = data_n / para_b;
    int total_blocks = para_num_passes * (data_n / para_b);
    if (is_sparse) {
        for (int i = 0; i < data_n; i++) {
            const int *xt_inds = x_tr_inds + x_tr_poss[i];
            const double *xt_vals = x_tr_vals + x_tr_poss[i];
            if (data_y_tr[i] > 0) {
                posi_t++;
                for (int kk = 0; kk < x_tr_lens[i]; kk++) ut[xt_inds[kk]] += xt_vals[kk];
            } else {
                nega_t++;
                for (int kk = 0; kk < x_tr_lens[i]; kk++) vt[xt_inds[kk]] += xt_vals[kk];
            }
        }
    } else {
        for (int i = 0; i < data_n; i++) {
            if (data_y_tr[i] > 0) {
                posi_t++;
                cblas_daxpy(data_p, 1., (x_tr_vals + i * data_p), 1, ut, 1);
            } else {
                nega_t++;
                cblas_daxpy(data_p, 1., (x_tr_vals + i * data_p), 1, vt, 1);
            }
        }
    }
    prob_p = posi_t / (data_n * 1.0);
    cblas_dscal(data_p, 1. / posi_t, ut, 1);
    cblas_dscal(data_p, 1. / nega_t, vt, 1);
    double *var = calloc((size_t) data_p, sizeof(double));
    double *tmp = calloc((size_t) data_p, sizeof(double));
    memcpy(var, vt, sizeof(double) * data_p);
    cblas_daxpy(data_p, -1., ut, 1, var, 1);
    cblas_dscal(data_p, 2 * prob_p * (1 - prob_p), var, 1);
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    *re_len_auc = 0;
    if (para_verbose > 0) { printf("total blocks: %d", total_blocks); }
    for (int t = 1; t <= total_blocks; t++) { // for each block
        // block bi must be in [min_b_ind,max_b_ind-1]
        int bi = rand() % (max_b_ind - min_b_ind);
        double utw = cblas_ddot(data_p, re_wt, 1, ut, 1);
        double vtw = cblas_ddot(data_p, re_wt, 1, vt, 1);
        eta_t = para_c;
        // the gradient of a block training samples
        memset(grad_wt, 0, sizeof(double) * data_p);
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data_n % para_b) : para_b);
        for (int kk = 0; kk < cur_b_size; kk++) {
            int ind = bi * para_b + kk; // initial position of block bi
            const double *cur_xt = x_tr_vals + ind * data_p;
            const int *xt_inds = x_tr_inds + x_tr_poss[ind];
            const double *xt_vals = x_tr_vals + x_tr_poss[ind];
            double xtw = 0.0;
            if (is_sparse) {
                for (int tt = 0; tt < x_tr_lens[ind]; tt++) {
                    xtw += (re_wt[xt_inds[tt]] * xt_vals[tt]);
                }
            } else {
                xtw = cblas_ddot(data_p, re_wt, 1, cur_xt, 1);
            }
            memcpy(tmp, var, sizeof(double) * data_p);
            cblas_dscal(data_p, 1 + vtw - utw, tmp, 1);
            if (data_y_tr[ind] > 0) {
                double part_wei = 2. * (1 - prob_p) * (xtw - utw);
                if (is_sparse) {
                    for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                        tmp[xt_inds[tt]] += part_wei * xt_vals[tt];
                } else {
                    cblas_daxpy(data_p, part_wei, cur_xt, 1, tmp, 1);
                }
                cblas_daxpy(data_p, -part_wei, ut, 1, tmp, 1);
            } else {
                double part_wei = 2. * prob_p * (xtw - vtw);
                if (is_sparse) {
                    for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                        tmp[xt_inds[tt]] += part_wei * xt_vals[tt];
                } else {
                    cblas_daxpy(data_p, part_wei, cur_xt, 1, tmp, 1);
                }
                cblas_daxpy(data_p, -part_wei, vt, 1, tmp, 1);
            }
            cblas_daxpy(data_p, 1., tmp, 1, grad_wt, 1); // calculate the gradient
        }
        cblas_daxpy(data_p, -eta_t / cur_b_size, grad_wt, 1, re_wt, 1); // wt = wt - eta * grad(wt)
        if (para_l2_reg != 0.0) // ell_2 reg. we do not need it in our case.
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
        _hard_thresholding(re_wt, data_p, para_s); // k-sparse projection step.
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if (record_aucs) { // to evaluate AUC score
            t_eval = clock();
            if (is_sparse) {
                for (int q = 0; q < data_n; q++) {
                    const int *xt_inds = x_tr_inds + x_tr_poss[q];
                    const double *xt_vals = x_tr_vals + x_tr_poss[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++)
                        y_pred[q] += (re_wt[xt_inds[tt]] * xt_vals[tt]);
                }
            } else {
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., x_tr_vals, data_p, re_wt, 1, 0.0, y_pred, 1);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            memset(y_pred, 0, sizeof(double) * data_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    cblas_dscal(data_p, 1. / total_blocks, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(var);
    free(tmp);
    free(y_pred);
    free(vt);
    free(ut);
    free(posi_ut);
    free(nega_vt);
    free(grad_wt);
}


void _algo_graph_am(const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
                    const double *data_y_tr, const EdgePair *edges, const double *weights, int data_m,
                    int data_n, int data_p, int para_s, int para_b, double para_c, double para_l2_reg,
                    int para_num_passes, int para_verbose, bool is_sparse, bool record_aucs,
                    double *re_wt, double *re_wt_bar, double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(NULL));
    double *ut = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double *vt = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double *posi_ut = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double *nega_vt = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double posi_t = 0.0, nega_t = 0.0, prob_p, eta_t, t_eval;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    int min_b_ind = 0, max_b_ind = data_n / para_b;
    int total_blocks = para_num_passes * (data_n / para_b);
    if (is_sparse) {
        for (int i = 0; i < data_n; i++) {
            const int *xt_inds = x_tr_inds + x_tr_poss[i];
            const double *xt_vals = x_tr_vals + x_tr_poss[i];
            if (data_y_tr[i] > 0) {
                posi_t++;
                for (int kk = 0; kk < x_tr_lens[i]; kk++) ut[xt_inds[kk]] += xt_vals[kk];
            } else {
                nega_t++;
                for (int kk = 0; kk < x_tr_lens[i]; kk++) vt[xt_inds[kk]] += xt_vals[kk];
            }
        }
    } else {
        for (int i = 0; i < data_n; i++) {
            if (data_y_tr[i] > 0) {
                posi_t++;
                cblas_daxpy(data_p, 1., (x_tr_vals + i * data_p), 1, ut, 1);
            } else {
                nega_t++;
                cblas_daxpy(data_p, 1., (x_tr_vals + i * data_p), 1, vt, 1);
            }
        }
    }
    prob_p = posi_t / (data_n * 1.0);
    cblas_dscal(data_p, 1. / posi_t, ut, 1);
    cblas_dscal(data_p, 1. / nega_t, vt, 1);
    double *var = calloc((size_t) data_p, sizeof(double));
    double *tmp = calloc((size_t) data_p, sizeof(double));
    memcpy(var, vt, sizeof(double) * data_p);
    cblas_daxpy(data_p, -1., ut, 1, var, 1);
    cblas_dscal(data_p, 2 * prob_p * (1 - prob_p), var, 1);

    double *proj_prizes = malloc(sizeof(double) * data_p);   // projected prizes.
    GraphStat *graph_stat = make_graph_stat(data_p, data_m);   // head projection paras

    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    *re_len_auc = 0;

    if (para_verbose > 0) { printf("total blocks: %d\n", total_blocks); }
    for (int t = 1; t <= total_blocks; t++) { // for each block
        // block bi must be in [min_b_ind,max_b_ind-1]
        int bi = rand() % (max_b_ind - min_b_ind);
        double utw = cblas_ddot(data_p, re_wt, 1, ut, 1);
        double vtw = cblas_ddot(data_p, re_wt, 1, vt, 1);
        eta_t = para_c;
        // the gradient of a block training samples
        memset(grad_wt, 0, sizeof(double) * data_p);
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data_n % para_b) : para_b);
        for (int kk = 0; kk < cur_b_size; kk++) {
            int ind = bi * para_b + kk; // initial position of block bi
            const double *cur_xt = x_tr_vals + ind * data_p;
            const int *xt_inds = x_tr_inds + x_tr_poss[ind];
            const double *xt_vals = x_tr_vals + x_tr_poss[ind];
            double xtw = 0.0;
            if (is_sparse) {
                for (int tt = 0; tt < x_tr_lens[ind]; tt++) {
                    xtw += (re_wt[xt_inds[tt]] * xt_vals[tt]);
                }
            } else {
                xtw = cblas_ddot(data_p, re_wt, 1, cur_xt, 1);
            }
            memcpy(tmp, var, sizeof(double) * data_p);
            cblas_dscal(data_p, 1 + vtw - utw, tmp, 1);
            if (data_y_tr[ind] > 0) {
                double part_wei = 2. * (1 - prob_p) * (xtw - utw);
                if (is_sparse) {
                    for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                        tmp[xt_inds[tt]] += part_wei * xt_vals[tt];
                } else {
                    cblas_daxpy(data_p, part_wei, cur_xt, 1, tmp, 1);
                }
                cblas_daxpy(data_p, -part_wei, ut, 1, tmp, 1);
            } else {
                double part_wei = 2. * prob_p * (xtw - vtw);
                if (is_sparse) {
                    for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                        tmp[xt_inds[tt]] += part_wei * xt_vals[tt];
                } else {
                    cblas_daxpy(data_p, part_wei, cur_xt, 1, tmp, 1);
                }
                cblas_daxpy(data_p, -part_wei, vt, 1, tmp, 1);
            }
            cblas_daxpy(data_p, 1., tmp, 1, grad_wt, 1); // calculate the gradient
        }
        cblas_daxpy(data_p, -eta_t / cur_b_size, grad_wt, 1, re_wt, 1); // wt = wt - eta * grad(wt)
        if (para_l2_reg != 0.0) // ell_2 reg. we do not need it in our case.
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
        //to do graph projection.
        for (int kk = 0; kk < data_p; kk++) { proj_prizes[kk] = re_wt[kk] * re_wt[kk]; }
        int g = 1, s_low = para_s, s_high = para_s + 2;
        int tail_max_iter = 50, verbose = 0;
        head_tail_binsearch(edges, weights, proj_prizes, data_p, data_m, g, -1, s_low,
                            s_high, tail_max_iter, GWPruning, verbose, graph_stat);
        memcpy(grad_wt, re_wt, sizeof(double) * data_p);
        memset(re_wt, 0, sizeof(double) * data_p);
        for (int kk = 0; kk < graph_stat->re_nodes->size; kk++) {
            int cur_node = graph_stat->re_nodes->array[kk];
            re_wt[cur_node] = grad_wt[cur_node];
        }
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if (record_aucs) { // to evaluate AUC score
            t_eval = clock();
            if (is_sparse) {
                for (int q = 0; q < data_n; q++) {
                    const int *xt_inds = x_tr_inds + x_tr_poss[q];
                    const double *xt_vals = x_tr_vals + x_tr_poss[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++)
                        y_pred[q] += (re_wt[xt_inds[tt]] * xt_vals[tt]);
                }
            } else {
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., x_tr_vals, data_p, re_wt, 1, 0.0, y_pred, 1);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            memset(y_pred, 0, sizeof(double) * data_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    cblas_dscal(data_p, 1. / total_blocks, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(proj_prizes);
    free_graph_stat(graph_stat);
    free(var);
    free(tmp);
    free(y_pred);
    free(vt);
    free(ut);
    free(posi_ut);
    free(nega_vt);
    free(grad_wt);
}


void _algo_sht_am_old(
        const double *data_x_tr, const double *data_y_tr, int data_n, int data_p, int para_s,
        int para_b, double para_c, double para_l2_reg, int para_num_passes, int para_step_len,
        int para_verbose, double *re_wt, double *re_wt_bar, double *re_auc, double *re_rts,
        int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(NULL));
    double a_wt, *posi_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double b_wt, *nega_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double alpha_wt, posi_t = 0.0, nega_t = 0.0, prob_p, eta_t, t_eval;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    int min_b_ind = 0, max_b_ind = data_n / para_b;
    int total_blocks = para_num_passes * (data_n / para_b);
    for (int i = 0; i < data_n; i++) {
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_daxpy(data_p, 1., (data_x_tr + i * data_p), 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_daxpy(data_p, 1., (data_x_tr + i * data_p), 1, nega_x_mean, 1);
        }
    }
    prob_p = posi_t / (data_n * 1.0);
    cblas_dscal(data_p, 1. / posi_t, posi_x_mean, 1);
    cblas_dscal(data_p, 1. / nega_t, nega_x_mean, 1);
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    *re_len_auc = 0;
    if (para_verbose == 1) {
        printf("num_posi: %.0f num_nega: %.0f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("||x_posi||: %.6f ||x_nega||: %.6f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
        printf("step_len: %d\n", para_step_len);
        printf("||re_wt||: %.6f\n", sqrt(cblas_ddot(data_p, re_wt, 1, re_wt, 1)));
    }
    for (int t = 1; t <= total_blocks; t++) { // for each block
        int bi = rand() % (max_b_ind - min_b_ind); // block bi must be in [min_b_ind,max_b_ind-1]
        a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
        b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // update b(wt)
        alpha_wt = b_wt - a_wt; // update alpha(wt)
        eta_t = para_c;
        memset(grad_wt, 0, sizeof(double) * data_p);
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data_n % para_b) : para_b);
        for (int kk = 0; kk < cur_b_size; kk++) {
            int ind = bi * para_b + kk; // initial position of block bi
            const double *cur_xt = data_x_tr + ind * data_p;
            double wt_dot = cblas_ddot(data_p, re_wt, 1, cur_xt, 1);
            double weight = data_y_tr[ind] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                                 2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                            2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            cblas_daxpy(data_p, weight, cur_xt, 1, grad_wt, 1); // calculate the gradient
        }
        cblas_daxpy(data_p, -eta_t / cur_b_size, grad_wt, 1, re_wt, 1); // wt = wt - eta * grad(wt)
        if (para_l2_reg != 0.0) // ell_2 reg. we do not need it in our case.
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
        _hard_thresholding(re_wt, data_p, para_s); // k-sparse step.
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if (para_verbose == 1) {  // to evaluate AUC score
            t_eval = clock();
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            re_rts[*re_len_auc] = clock() - start_time - (clock() - t_eval);
            *re_len_auc = *re_len_auc + 1;
        }
    }
    cblas_dscal(data_p, 1. / total_blocks, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(y_pred);
    free(nega_x_mean);
    free(posi_x_mean);
    free(grad_wt);
}


void _algo_sht_am_sparse_old(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, const double *x_te_vals, const int *x_te_inds,
        const int *x_te_poss, const int *x_te_lens, const double *data_y_te,
        int data_tr_n, int data_te_n,
        int data_p, int para_s, int para_b, double para_c,
        double para_l2_reg, int para_num_passes, int para_step_len, int para_verbose,
        double *re_wt, double *re_wt_bar, double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(NULL));
    double a_wt, *posi_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double b_wt, *nega_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double alpha_wt, posi_t = 0.0, nega_t = 0.0, prob_p, eta_t, t_eval;
    double *y_pred = calloc((size_t) data_tr_n, sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    int min_b_ind = 0, max_b_ind = data_tr_n / para_b;
    int total_blocks = para_num_passes * (data_tr_n / para_b);
    for (int i = 0; i < data_tr_n; i++) {
        const int *xt_inds = x_tr_inds + x_tr_poss[i];
        const double *xt_vals = x_tr_vals + x_tr_poss[i];
        if (data_y_tr[i] > 0) {
            posi_t++;
            for (int kk = 0; kk < x_tr_lens[i]; kk++) posi_x_mean[xt_inds[kk]] += xt_vals[kk];
        } else {
            nega_t++;
            for (int kk = 0; kk < x_tr_lens[i]; kk++) nega_x_mean[xt_inds[kk]] += xt_vals[kk];
        }
    }
    prob_p = posi_t / (data_tr_n * 1.0);
    cblas_dscal(data_p, 1. / posi_t, posi_x_mean, 1);
    cblas_dscal(data_p, 1. / nega_t, nega_x_mean, 1);
    if (para_verbose > 1) {
        printf("num_posi: %.0f num_nega: %.0f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("||x_posi||: %.6f ||x_nega||: %.6f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
        printf("step_len: %d\n", para_step_len);
        printf("||re_wt||: %.6f\n", sqrt(cblas_ddot(data_p, re_wt, 1, re_wt, 1)));
    }
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;

    for (int t = 1; t <= total_blocks; t++) { // for each block
        int bi = rand() % (max_b_ind - min_b_ind); // block bi must be in [min_b_ind,max_b_ind-1]
        a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
        b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // update b(wt)
        alpha_wt = b_wt - a_wt; // update alpha(wt)
        eta_t = para_c; // TODO: current learning rate ?
        memset(grad_wt, 0, sizeof(double) * data_p); // the gradient of a block training samples
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data_tr_n % para_b) : para_b);
        for (int kk = 0; kk < cur_b_size; kk++) {
            int ind = bi * para_b + kk; // initial position of block bi
            const int *xt_inds = x_tr_inds + x_tr_poss[ind];
            const double *xt_vals = x_tr_vals + x_tr_poss[ind];
            double wt_dot = 0.0;
            for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                wt_dot += (re_wt[xt_inds[tt]] * xt_vals[tt]);
            double weight = data_y_tr[ind] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                                 2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                            2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                grad_wt[xt_inds[tt]] += (weight * xt_vals[tt]); // calculate the gradient for xi
        }
        cblas_daxpy(data_p, -eta_t / cur_b_size, grad_wt, 1, re_wt, 1); // wt = wt - eta * grad(wt)
        if (para_l2_reg != 0.0) // ell_2 reg. we do not need it in our case.
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
        _hard_thresholding(re_wt, data_p, para_s); // k-sparse projection step.
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if (para_verbose == 1) { // to evaluate AUC score
            t_eval = clock();
            for (int q = 0; q < data_tr_n; q++) {
                const int *xt_inds = x_tr_inds + x_tr_poss[q];
                const double *xt_vals = x_tr_vals + x_tr_poss[q];
                for (int tt = 0; tt < x_tr_lens[q]; tt++)
                    y_pred[q] += (re_wt[xt_inds[tt]] * xt_vals[tt]);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_tr_n);
            memset(y_pred, 0, sizeof(double) * data_tr_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    cblas_dscal(data_p, 1. / total_blocks, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(y_pred);
    free(nega_x_mean);
    free(posi_x_mean);
    free(grad_wt);
}


void _algo_graph_am_old(const double *data_x_tr, const double *data_y_tr, const EdgePair *edges,
                        const double *weights, int data_m, int data_n, int data_p, int para_sparsity,
                        int para_b, double para_xi, double para_l2_reg, int para_num_passes,
                        int para_step_len, int para_verbose, double *re_wt, double *re_wt_bar,
                        double *re_auc, double *re_rts, int *re_len_auc) {
    double start_time = clock();
    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.

    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p);
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    double posi_t = 0.0, nega_t = 0.0, prob_p, eta_t, t = 1.0;;
    for (int i = 0; i < data_n; i++) {
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_daxpy(data_p, 1., (data_x_tr + i * data_p), 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_daxpy(data_p, 1., (data_x_tr + i * data_p), 1, nega_x_mean, 1);
        }
    }
    prob_p = posi_t / (data_n * 1.0);
    cblas_dscal(data_p, 1. / posi_t, posi_x_mean, 1);
    cblas_dscal(data_p, 1. / nega_t, nega_x_mean, 1);
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    double *aver_grad = malloc(sizeof(double) * data_p);
    memset(aver_grad, 0, sizeof(double) * data_p);
    double *proj_prizes = malloc(sizeof(double) * data_p);   // projected prizes.
    GraphStat *graph_stat = make_graph_stat(data_p, data_m);   // head projection paras
    double *y_pred = malloc(sizeof(double) * data_n);
    int min_b_index = 0, max_b_index = data_n / para_b;
    *re_len_auc = 0;
    for (int i = 0; i < para_num_passes; i++) {
        for (int jj = 0;
             jj < data_n / para_b; jj++) { // data_n/para_b is the total number of blocks.
            // rand_ind is in [min_b_index,max_b_index-1]
            int rand_ind = rand() % (max_b_index - min_b_index);
            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // update b(wt)
            alpha_wt = b_wt - a_wt; // update alpha(wt)
            eta_t = para_xi / sqrt(t); // current learning rate
            // receive a block of training samples to calculate the gradient
            memset(grad_wt, 0, sizeof(double) * data_p);
            int cur_b_size = (rand_ind == (max_b_index - 1) ? para_b + (data_n % para_b) : para_b);
            for (int kk = 0; kk < cur_b_size; kk++) {
                int ind = (rand_ind * para_b + kk);
                const double *cur_xt = data_x_tr + ind * data_p;
                double wt_dot = cblas_ddot(data_p, re_wt, 1, cur_xt, 1);
                double weight = data_y_tr[ind] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                                2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
                cblas_daxpy(data_p, weight, cur_xt, 1, grad_wt, 1); // calculate the gradient
            }
            cblas_dscal(data_p, (t - 1.) / t, aver_grad, 1); // take average of wt --> wt_bar
            cblas_daxpy(data_p, 1. / t, grad_wt, 1, aver_grad, 1);
            cblas_dscal(data_p, 1. / (cur_b_size * 1.0), grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); // gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
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
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            //to do graph projection.
            for (int kk = 0; kk < data_p; kk++) {
                proj_prizes[kk] = u[kk] * u[kk];
                if (proj_prizes[kk] >= 1e8) {
                    printf(" warning! too large value: %.4f\n", proj_prizes[kk]);
                }
            }
            int g = 1, sparsity_low = para_sparsity, sparsity_high = para_sparsity + 10;
            int tail_max_iter = 20, verbose = 0;

            head_tail_binsearch(edges, weights, proj_prizes, data_p, data_m, g, -1, sparsity_low,
                                sparsity_high, tail_max_iter, GWPruning, verbose, graph_stat);
            cblas_dscal(data_p, 0.0, re_wt, 1);
            for (int kk = 0; kk < graph_stat->re_nodes->size; kk++) {
                int cur_node = graph_stat->re_nodes->array[kk];
                re_wt[cur_node] = u[cur_node];
            }
            // take average of wt --> wt_bar
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1);
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.)) {
                double t_eval = clock();
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
                t_eval = clock() - t_eval;
                re_rts[*re_len_auc] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
                *re_len_auc = *re_len_auc + 1;
            }
            t = t + 1.0; // increase time
        }
    }
    free(y_pred);
    free(proj_prizes);
    free_graph_stat(graph_stat);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
}


void _algo_graph_am_sparse_old(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, const EdgePair *edges, const double *weights, int data_m,
        int data_n, int data_p, int para_s, int para_b, double para_c, double para_l2_reg,
        int para_num_passes, int para_verbose, bool is_sparse, bool record_aucs,
        double *re_wt, double *re_wt_bar, double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);  // only one cpu at a time.

    double a_wt, *posi_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double b_wt, *nega_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double alpha_wt, posi_t = 0.0, nega_t = 0.0, prob_p, eta_t;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *proj_prizes = malloc(sizeof(double) * data_p);   // projected prizes.
    GraphStat *graph_stat = make_graph_stat(data_p, data_m);   // head projection paras
    int min_b_ind = 0, max_b_ind = data_n / para_b;
    int total_blocks = para_num_passes * (data_n / para_b);
    for (int i = 0; i < data_n; i++) {
        const int *xt_inds = x_tr_inds + x_tr_poss[i];
        const double *xt_vals = x_tr_vals + x_tr_poss[i];
        if (data_y_tr[i] > 0) {
            posi_t++;
            for (int kk = 0; kk < x_tr_lens[i]; kk++)
                posi_x_mean[xt_inds[kk]] += xt_vals[kk];
        } else {
            nega_t++;
            for (int kk = 0; kk < x_tr_lens[i]; kk++)
                nega_x_mean[xt_inds[kk]] += xt_vals[kk];
        }
    }
    prob_p = posi_t / (data_n * 1.0);
    cblas_dscal(data_p, 1. / posi_t, posi_x_mean, 1);
    cblas_dscal(data_p, 1. / nega_t, nega_x_mean, 1);
    if (para_verbose == 1) {
        printf("num_posi: %.0f num_nega: %.0f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("||x_posi||: %.6f ||x_nega||: %.6f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
        printf("||re_wt||: %.6f\n", sqrt(cblas_ddot(data_p, re_wt, 1, re_wt, 1)));
    }
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    for (int t = 0; t < total_blocks; t++) { // for each block
        int bi = rand() % (max_b_ind - min_b_ind); // block bi must be in [min_b_ind,max_b_ind-1]
        a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
        b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // update b(wt)
        alpha_wt = b_wt - a_wt; // update alpha(wt)
        eta_t = para_c / sqrt(t + 1.);
        memset(grad_wt, 0, sizeof(double) * data_p);
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data_n % para_b) : para_b);
        for (int kk = 0; kk < cur_b_size; kk++) {
            int ind = bi * para_b + kk; // current ind:
            const int *xt_inds = x_tr_inds + x_tr_poss[ind];
            const double *xt_vals = x_tr_vals + x_tr_poss[ind];
            double wt_dot = 0.0;
            for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                wt_dot += (re_wt[xt_inds[tt]] * xt_vals[tt]);
            double weight = data_y_tr[ind] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                                 2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                            2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                grad_wt[xt_inds[tt]] += (weight * xt_vals[tt]); // calculate the gradient for xi
        }
        cblas_daxpy(data_p, -eta_t / cur_b_size, grad_wt, 1, re_wt, 1);
        if (para_l2_reg != 0.0) // ell_2 reg. we do not need it in our case.
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
        //to do graph projection.
        for (int kk = 0; kk < data_p; kk++) { proj_prizes[kk] = re_wt[kk] * re_wt[kk]; }
        int g = 1, s_low = para_s, s_high = para_s + 2;
        int tail_max_iter = 50, verbose = 0;
        head_tail_binsearch(edges, weights, proj_prizes, data_p, data_m, g, -1, s_low,
                            s_high, tail_max_iter, GWPruning, verbose, graph_stat);
        memcpy(grad_wt, re_wt, sizeof(double) * data_p);
        memset(re_wt, 0, sizeof(double) * data_p);
        for (int kk = 0; kk < graph_stat->re_nodes->size; kk++) {
            int cur_node = graph_stat->re_nodes->array[kk];
            re_wt[cur_node] = grad_wt[cur_node];
        }
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if (para_verbose == 1) { // to calculate AUC score and run time
            double t_eval = clock();
            for (int q = 0; q < data_n; q++) {
                const int *xt_inds = x_tr_inds + x_tr_poss[q];
                const double *xt_vals = x_tr_vals + x_tr_poss[q];
                for (int tt = 0; tt < x_tr_lens[q]; tt++)
                    y_pred[q] += (re_wt[xt_inds[tt]] * xt_vals[tt]);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            memset(y_pred, 0, sizeof(double) * data_n);
            t_eval = clock() - t_eval;
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    cblas_dscal(data_p, 1. / total_blocks, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(y_pred);
    free(proj_prizes);
    free_graph_stat(graph_stat);
    free(nega_x_mean);
    free(posi_x_mean);
    free(grad_wt);
}


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
                 int *re_len_auc) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    double start_time = clock();
    double num_p = 0.0, num_n = 0.0;
    double *center_p = calloc((size_t) data_p, sizeof(double));
    double *center_n = calloc((size_t) data_p, sizeof(double));
    double *cov_p = calloc((size_t) (data_p * data_p), sizeof(double));
    double *cov_n = calloc((size_t) (data_p * data_p), sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p);
    double *tmp_vec = calloc((size_t) data_p, sizeof(double));
    double *tmp_mat = calloc((size_t) (data_p * data_p), sizeof(double));
    double *y_pred = malloc(sizeof(double) * data_n);
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    for (int t = 0; t < data_n * para_num_passes; t++) {
        int cur_ind = t % data_n;
        const double *cur_x = data_x_tr + cur_ind * data_p;
        double cur_y = data_y_tr[cur_ind];
        if (cur_y > 0) {
            num_p++;
            cblas_dcopy(data_p, center_p, 1, tmp_vec, 1); // copy previous center
            cblas_dscal(data_p, (num_p - 1.) / num_p, center_p, 1); // update center_p
            cblas_daxpy(data_p, 1. / num_p, cur_x, 1, center_p, 1);
            cblas_dscal(data_p * data_p, (num_p - 1.) / num_p, cov_p, 1); // update covariance matrix
            cblas_dger(CblasRowMajor, data_p, data_p, 1. / num_p, cur_x, 1, cur_x, 1, cov_p, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, (num_p - 1.) / num_p,
                       tmp_vec, 1, tmp_vec, 1, cov_p, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, -1., center_p, 1, center_p, 1, cov_p, data_p);
            if (num_n > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, center_n, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., cur_x, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, cur_x, 1, tmp_vec, 1); // xt - c_t^-
                cblas_daxpy(data_p, -1., center_n, 1, tmp_vec, 1);
                cblas_dscal(data_p * data_p, 0.0, tmp_mat, 1); // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dger(CblasRowMajor, data_p, data_p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat, data_p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., tmp_mat, data_p, re_wt,
                            1, 1.0, grad_wt, 1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., cov_n, data_p, re_wt,
                            1, 1.0, grad_wt, 1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        } else {
            num_n++;
            cblas_dcopy(data_p, center_n, 1, tmp_vec, 1); // copy previous center
            cblas_dscal(data_p, (num_n - 1.) / num_n, center_n, 1); // update center_n
            cblas_daxpy(data_p, 1. / num_n, cur_x, 1, center_n, 1);
            cblas_dscal(data_p * data_p, (num_n - 1.) / num_n, cov_n, 1); // update covariance matrix
            cblas_dger(CblasRowMajor, data_p, data_p, 1. / num_n, cur_x, 1, cur_x, 1, cov_n, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, (num_n - 1.) / num_n,
                       tmp_vec, 1, tmp_vec, 1, cov_n, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, -1., center_n, 1, center_n, 1, cov_n, data_p);
            if (num_p > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, cur_x, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., center_p, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, cur_x, 1, tmp_vec, 1); // xt - c_t^+
                cblas_daxpy(data_p, -1., center_p, 1, tmp_vec, 1);
                cblas_dscal(data_p * data_p, 0.0, tmp_mat, 1); // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dger(CblasRowMajor, data_p, data_p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat, data_p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., tmp_mat, data_p,
                            re_wt, 1, 1.0, grad_wt, 1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., cov_p, data_p, re_wt,
                            1, 1.0, grad_wt, 1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        }
        cblas_daxpy(data_p, -para_eta, grad_wt, 1, re_wt, 1); // update the solution
        cblas_dscal(data_p, (cur_ind * 1.) / (cur_ind * 1. + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (cur_ind + 1.), re_wt, 1, re_wt_bar, 1);
        if ((fmod(cur_ind, para_step_len) == 0.) && para_is_aucs) { // to calculate AUC score
            double t_eval = clock();
            cblas_dgemv(CblasRowMajor, CblasNoTrans, data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            t_eval = clock() - t_eval;
            re_rts[*re_len_auc] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
            *re_len_auc = *re_len_auc + 1;
        }
    }
    free(y_pred);
    free(tmp_vec);
    free(tmp_mat);
    free(grad_wt);
    free(cov_n);
    free(cov_p);
    free(center_n);
    free(center_p);
}


void _algo_opauc_sparse(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, int data_n, int data_p, int para_tau, double para_eta,
        double para_lambda, int para_num_passes, int para_step_len, int para_verbose,
        double *re_wt, double *re_wt_bar, double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(0));

    double num_p = 0.0, num_n = 0.0;
    double *center_p = calloc((size_t) data_p, sizeof(double));
    double *center_n = calloc((size_t) data_p, sizeof(double));
    double *h_center_p = calloc((size_t) para_tau, sizeof(double));
    double *h_center_n = calloc((size_t) para_tau, sizeof(double));
    double *z_p = calloc((size_t) (data_p * para_tau), sizeof(double));
    double *z_n = calloc((size_t) (data_p * para_tau), sizeof(double));
    double *grad_wt = malloc(sizeof(double) * data_p);

    double *tmp_vec = malloc(sizeof(double) * data_p);
    double *y_pred = malloc(sizeof(double) * data_n);
    double *xt = malloc(sizeof(double) * data_p);
    double *gaussian = malloc(sizeof(double) * para_tau);
    if (para_verbose > 0) { printf("%d %d\n", data_n, data_p); }

    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    for (int t = 0; t < data_n * para_num_passes; t++) {
        int cur_ind = t % data_n;
        const int *xt_inds = x_tr_inds + x_tr_poss[cur_ind];
        const double *xt_vals = x_tr_vals + x_tr_poss[cur_ind];
        memset(xt, 0, sizeof(double) * data_p);
        for (int tt = 0; tt < x_tr_lens[cur_ind]; tt++) { xt[xt_inds[tt]] = xt_vals[tt]; }
        std_normal(para_tau, gaussian);
        cblas_dscal(para_tau, 1. / sqrt(para_tau * 1.), gaussian, 1);
        if (data_y_tr[cur_ind] > 0) {
            num_p++;
            cblas_dscal(data_p, (num_p - 1.) / num_p, center_p, 1); // update center_p
            for (int tt = 0; tt < x_tr_lens[cur_ind]; tt++)
                center_p[xt_inds[tt]] += xt_vals[tt] / num_p;
            cblas_dscal(para_tau, (num_p - 1.) / num_p, h_center_p, 1); // update h_center_p
            cblas_daxpy(para_tau, 1. / num_p, gaussian, 1, h_center_p, 1);
            cblas_dger(CblasRowMajor, data_p, para_tau, 1., xt, 1, gaussian, 1, z_p, para_tau);
            if (num_n > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, center_n, 1, grad_wt, 1);
                for (int tt = 0; tt < x_tr_lens[cur_ind]; tt++)
                    grad_wt[xt_inds[tt]] += -xt_vals[tt];
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, xt, 1, tmp_vec, 1); // xt - c_t^-
                cblas_daxpy(data_p, -1., center_n, 1, tmp_vec, 1);
                cblas_dscal(data_p, cblas_ddot(data_p, tmp_vec, 1, re_wt, 1), tmp_vec, 1);
                cblas_daxpy(data_p, 1., tmp_vec, 1, grad_wt, 1);
                cblas_dgemv(CblasRowMajor, CblasTrans, data_p, para_tau, 1. / num_n, z_n, para_tau,
                            re_wt, 1, 0.0, tmp_vec, 1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, para_tau, 1., z_n, para_tau,
                            tmp_vec, 1, 1.0, grad_wt, 1);
                double wei = cblas_ddot(data_p, re_wt, 1, center_n, 1);
                wei = wei * cblas_ddot(para_tau, h_center_n, 1, h_center_n, 1);
                cblas_daxpy(data_p, wei, center_n, 1, grad_wt, 1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        } else {
            num_n++;
            cblas_dscal(data_p, (num_n - 1.) / num_n, center_n, 1); // update center_n
            for (int tt = 0; tt < x_tr_lens[cur_ind]; tt++)
                center_n[xt_inds[tt]] += xt_vals[tt] / num_n;
            cblas_dscal(para_tau, (num_n - 1.) / num_n, h_center_n, 1); // update h_center_n
            cblas_daxpy(para_tau, 1. / num_n, gaussian, 1, h_center_n, 1);
            cblas_dger(CblasRowMajor, data_p, para_tau, 1., xt, 1, gaussian, 1, z_n, para_tau);
            if (num_p > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, xt, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., center_p, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, xt, 1, tmp_vec, 1); // xt - c_t^+
                cblas_daxpy(data_p, -1., center_p, 1, tmp_vec, 1);
                cblas_dscal(data_p, cblas_ddot(data_p, tmp_vec, 1, re_wt, 1), tmp_vec, 1);
                cblas_daxpy(data_p, 1., tmp_vec, 1, grad_wt, 1);
                cblas_dgemv(CblasRowMajor, CblasTrans, data_p, para_tau, 1. / num_p, z_p, para_tau,
                            re_wt, 1, 0.0, tmp_vec, 1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, para_tau, 1., z_p, para_tau,
                            tmp_vec, 1, 1.0, grad_wt, 1);
                double wei = cblas_ddot(data_p, re_wt, 1, center_p, 1);
                wei = wei * cblas_ddot(para_tau, h_center_p, 1, h_center_p, 1);
                cblas_daxpy(data_p, wei, center_p, 1, grad_wt, 1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        }
        cblas_daxpy(data_p, -para_eta, grad_wt, 1, re_wt, 1); // update the solution
        cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        if ((fmod(t + 1, para_step_len) == 1.)) { // to calculate AUC score
            double t_eval = clock();
            for (int q = 0; q < data_n; q++) {
                memset(xt, 0, sizeof(double) * data_p);
                xt_inds = x_tr_inds + x_tr_poss[q];
                xt_vals = x_tr_vals + x_tr_poss[q];
                for (int tt = 0; tt < x_tr_lens[q]; tt++) { xt[xt_inds[tt]] = xt_vals[tt]; }
                y_pred[q] = cblas_ddot(data_p, xt, 1, re_wt, 1);
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    cblas_dscal(data_p, 1. / data_n, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(gaussian);
    free(xt);
    free(z_p);
    free(z_n);
    free(y_pred);
    free(tmp_vec);
    free(grad_wt);
    free(h_center_n);
    free(h_center_p);
    free(center_n);
    free(center_p);
}


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
                   int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);
    srand((unsigned int) time(NULL));
    double eta_t, t_eval;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    int min_b_ind = 0, max_b_ind = data_n / para_b;
    int total_blocks = para_num_passes * (data_n / para_b);
    memset(re_wt, 0, sizeof(double) * (data_p + 1)); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * (data_p + 1)); // wt_bar --> 0.0
    *re_len_auc = 0;
    double *loss_grad_wt = calloc((data_p + 2), sizeof(double));
    for (int t = 1; t <= total_blocks; t++) { // for each block
        // block bi must be in [min_b_ind,max_b_ind-1]
        int bi = rand() % (max_b_ind - min_b_ind);
        eta_t = para_c;
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data_n % para_b) : para_b);
        // calculate the gradient
        logistic_loss_grad(re_wt, data_x_tr + bi * para_b, data_y_tr + bi * para_b,
                           loss_grad_wt, para_l2_reg, cur_b_size, data_p);
        // wt = wt - eta * grad(wt)
        cblas_daxpy(data_p + 1, -eta_t / cur_b_size, loss_grad_wt + 1, 1, re_wt, 1);
        _hard_thresholding(re_wt, data_p, para_s); // k-sparse step.
        cblas_daxpy(data_p + 1, 1., re_wt, 1, re_wt_bar, 1);
        if (para_verbose == 1) {  // to evaluate AUC score
            t_eval = clock();
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            printf("%.4f\n", re_auc[*re_len_auc]);
            re_rts[*re_len_auc] = clock() - start_time - (clock() - t_eval);
            *re_len_auc = *re_len_auc + 1;
        }
    }
    cblas_dscal(data_p + 1, 1. / total_blocks, re_wt_bar, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(loss_grad_wt);
    free(y_pred);
}


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
                 int *re_len_auc) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    double start_time = clock(), delta = 0.1, eta = para_g, R = para_r;
    if (para_verbose > 0) { printf("eta: %.12f R: %.12f", eta, R); }
    double n_ids = para_num_passes * data_n;
    double *v_1 = malloc(sizeof(double) * (data_p + 2)), alpha_1 = 0.0, alpha;
    memset(v_1, 0, sizeof(double) * (data_p + 2));
    double *sx_pos = malloc(sizeof(double) * (data_p));
    memset(sx_pos, 0, sizeof(double) * data_p);
    double *sx_neg = malloc(sizeof(double) * (data_p));
    memset(sx_neg, 0, sizeof(double) * data_p);
    double *m_pos = malloc(sizeof(double) * (data_p));
    memset(m_pos, 0, sizeof(double) * data_p);
    double *m_neg = malloc(sizeof(double) * (data_p));
    memset(m_neg, 0, sizeof(double) * data_p);
    int m = (int) floor(0.5 * log2(2 * n_ids / log2(n_ids))) - 1;
    int n_0 = (int) floor(n_ids / m);
    para_r = 2. * sqrt(3.) * R;
    double p_hat = 0.0, beta = 9.0, D = 2. * sqrt(2.) * para_r, sp = 0.0, t = 0.0;;
    double *gd = malloc(sizeof(double) * (data_p + 2)), gd_alpha;
    memset(gd, 0, sizeof(double) * (data_p + 2));
    double *v_ave = malloc(sizeof(double) * (data_p + 2));
    memset(v_ave, 0, sizeof(double) * (data_p + 2));
    double *v_sum = malloc(sizeof(double) * (data_p + 2));
    double *v = malloc(sizeof(double) * (data_p + 2));
    double *vd = malloc(sizeof(double) * (data_p + 2)), ad;
    double *tmp_proj = malloc(sizeof(double) * data_p), beta_new;
    double *y_pred = malloc(sizeof(double) * data_n);
    *re_len_auc = 0;
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    for (int k = 0; k < m; k++) {
        memset(v_sum, 0, sizeof(double) * (data_p + 2));
        cblas_dcopy(data_p + 2, v_1, 1, v, 1);
        alpha = alpha_1;
        for (int kk = 0; kk < n_0; kk++) {
            int ind = (k * n_0 + kk) % data_n;
            const double *xt = data_x_tr + ind * data_p;
            double yt = data_y_tr[ind];
            double wx = cblas_ddot(data_p, xt, 1, v, 1);
            double is_posi_y = is_posi(yt), is_nega_y = is_nega(yt);
            sp = sp + is_posi_y;
            p_hat = sp / (t + 1.);
            cblas_daxpy(data_p, is_posi_y, xt, 1, sx_pos, 1);
            cblas_daxpy(data_p, is_nega_y, xt, 1, sx_neg, 1);
            double weight = (1. - p_hat) * (wx - v[data_p] - 1. - alpha) * is_posi_y;
            weight += p_hat * (wx - v[data_p + 1] + 1. + alpha) * is_nega_y;
            cblas_dcopy(data_p, xt, 1, gd, 1);
            cblas_dscal(data_p, weight, gd, 1);
            gd[data_p] = (p_hat - 1.) * (wx - v[data_p]) * is_posi_y;
            gd[data_p + 1] = p_hat * (v[data_p + 1] - wx) * is_nega_y;
            gd_alpha = (p_hat - 1.) * (wx + p_hat * alpha) * is_posi_y +
                       p_hat * (wx + (p_hat - 1.) * alpha) * is_nega_y;
            cblas_daxpy(data_p + 2, -eta, gd, 1, v, 1);
            alpha = alpha + eta * gd_alpha;
            _l1ballproj_condat(v, tmp_proj, data_p, R); //projection to l1-ball
            cblas_dcopy(data_p, tmp_proj, 1, v, 1);
            if (fabs(v[data_p]) > R) { v[data_p] = v[data_p] * (R / fabs(v[data_p])); }
            if (fabs(v[data_p + 1]) > R) {
                v[data_p + 1] = v[data_p + 1] * (R / fabs(v[data_p + 1]));
            }
            if (fabs(alpha) > 2. * R) { alpha = alpha * (2. * R / fabs(alpha)); }
            cblas_dcopy(data_p + 2, v, 1, vd, 1);
            cblas_daxpy(data_p + 2, -1., v_1, 1, vd, 1);
            double norm_vd = sqrt(cblas_ddot(data_p + 2, vd, 1, vd, 1));
            if (norm_vd > para_r) { cblas_dscal(data_p + 2, para_r / norm_vd, vd, 1); }
            cblas_dcopy(data_p + 2, vd, 1, v, 1);
            cblas_daxpy(data_p + 2, 1., v_1, 1, v, 1);
            ad = alpha - alpha_1;
            if (fabs(ad) > D) { ad = ad * (D / fabs(ad)); }
            alpha = alpha_1 + ad;
            cblas_daxpy(data_p + 2, 1., v, 1, v_sum, 1);
            cblas_dcopy(data_p + 2, v_sum, 1, v_ave, 1);
            cblas_dscal(data_p + 2, 1. / (kk + 1.), v_ave, 1);
            t = t + 1.0;
            if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
                double t_eval = clock();
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
                t_eval = clock() - t_eval;
                re_rts[*re_len_auc] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
                *re_len_auc = *re_len_auc + 1;
            }
        }
        para_r = para_r / 2.;
        double tmp1 = 12. * sqrt(2.) * (2. + sqrt(2. * log(12. / delta))) * R;
        double tmp2 = fmin(p_hat, 1. - p_hat) * n_0 - sqrt(2. * n_0 * log(12. / delta));
        if (tmp2 > 0) { D = 2. * sqrt(2.) * para_r + tmp1 / sqrt(tmp2); } else { D = 1e7; }
        tmp1 = 288. * (pow(2. + sqrt(2. * log(12 / delta)), 2.));
        tmp2 = fmin(p_hat, 1. - p_hat) - sqrt(2. * log(12 / delta) / n_0);
        if (tmp2 > 0) { beta_new = 9. + tmp1 / tmp2; } else { beta_new = 1e7; }
        eta = fmin(sqrt(beta_new / beta) * eta / 2, eta);
        beta = beta_new;
        if (sp > 0.0) {
            cblas_dcopy(data_p, sx_pos, 1, m_pos, 1);
            cblas_dscal(data_p, 1. / sp, m_pos, 1);
        }
        if (sp < t) {
            cblas_dcopy(data_p, sx_neg, 1, m_neg, 1);
            cblas_dscal(data_p, 1. / (t - sp), m_neg, 1);
        }
        cblas_dcopy(data_p + 2, v_ave, 1, v_1, 1);
        cblas_dcopy(data_p, m_neg, 1, tmp_proj, 1);
        cblas_daxpy(data_p, -1., m_pos, 1, tmp_proj, 1);
        alpha_1 = cblas_ddot(data_p, v_ave, 1, tmp_proj, 1);
        // wt_bar update
        cblas_dscal(data_p, k / (k + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (k + 1.), v_ave, 1, re_wt_bar, 1);
    }
    cblas_dcopy(data_p, v_ave, 1, re_wt, 1);
    free(y_pred);
    free(tmp_proj);
    free(vd);
    free(v);
    free(v_sum);
    free(v_ave);
    free(gd);
    free(m_neg);
    free(m_pos);
    free(sx_neg);
    free(sx_pos);
    free(v_1);
}

void _algo_fsauc_sparse(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, int data_n, int data_p, double para_r, double para_g,
        int para_num_passes, int para_step_len, int para_verbose, double *re_wt, double *re_wt_bar,
        double *re_auc, double *re_rts, int *re_len_auc) {

    double start_time = clock();
    openblas_set_num_threads(1);

    double delta = 0.1, eta = para_g, R = para_r;

    double n_ids = para_num_passes * data_n, alpha_1 = 0.0, alpha;
    double *v_1 = calloc(((unsigned) data_p + 2), sizeof(double));
    double *sx_pos = calloc((size_t) data_p, sizeof(double));
    double *sx_neg = calloc((size_t) data_p, sizeof(double));
    double *m_pos = calloc((size_t) data_p, sizeof(double));
    double *m_neg = calloc((size_t) data_p, sizeof(double));
    int m = (int) floor(0.5 * log2(2 * n_ids / log2(n_ids))) - 1;
    int n_0 = (int) floor(n_ids / m);
    para_r = 2. * sqrt(3.) * R;
    double p_hat = 0.0, beta = 9.0, D = 2. * sqrt(2.) * para_r, sp = 0.0, t = 0.0;;
    double *gd = calloc((unsigned) data_p + 2, sizeof(double)), gd_alpha;
    double *v_ave = calloc((unsigned) data_p + 2, sizeof(double));
    double *v_sum = malloc(sizeof(double) * (data_p + 2));
    double *v = malloc(sizeof(double) * (data_p + 2));
    double *vd = malloc(sizeof(double) * (data_p + 2)), ad;
    double *tmp_proj = malloc(sizeof(double) * data_p), beta_new;
    double *y_pred = calloc((size_t) data_n, sizeof(double));

    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    if (para_verbose > 0) { printf("eta: %.12f R: %.12f", eta, R); }

    for (int k = 0; k < m; k++) {
        memset(v_sum, 0, sizeof(double) * (data_p + 2));
        memcpy(v, v_1, sizeof(double) * (data_p + 2));
        alpha = alpha_1;
        for (int kk = 0; kk < n_0; kk++) {
            int ind = (k * n_0 + kk) % data_n;
            const int *xt_inds = x_tr_inds + x_tr_poss[ind];
            const double *xt_vals = x_tr_vals + x_tr_poss[ind];
            double yt = data_y_tr[ind];
            double wx = 0.0;
            for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                wx += (v[xt_inds[tt]] * xt_vals[tt]);
            double is_posi_y = is_posi(yt), is_nega_y = is_nega(yt);
            sp = sp + is_posi_y;
            p_hat = sp / (t + 1.);
            if (yt > 0) {
                for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                    sx_pos[xt_inds[tt]] += xt_vals[tt];
            } else {
                for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                    sx_neg[xt_inds[tt]] += xt_vals[tt];
            }
            double weight = (1. - p_hat) * (wx - v[data_p] - 1. - alpha) * is_posi_y;
            weight += p_hat * (wx - v[data_p + 1] + 1. + alpha) * is_nega_y;
            memset(gd, 0, sizeof(double) * data_p);
            for (int tt = 0; tt < x_tr_lens[ind]; tt++)
                gd[xt_inds[tt]] = weight * xt_vals[tt];
            gd[data_p] = (p_hat - 1.) * (wx - v[data_p]) * is_posi_y;
            gd[data_p + 1] = p_hat * (v[data_p + 1] - wx) * is_nega_y;
            gd_alpha = (p_hat - 1.) * (wx + p_hat * alpha) * is_posi_y +
                       p_hat * (wx + (p_hat - 1.) * alpha) * is_nega_y;
            cblas_daxpy(data_p + 2, -eta, gd, 1, v, 1);
            alpha = alpha + eta * gd_alpha;
            _l1ballproj_condat(v, tmp_proj, data_p, R); //projection to l1-ball
            memcpy(v, tmp_proj, sizeof(double) * data_p);
            if (fabs(v[data_p]) > R) { v[data_p] = v[data_p] * (R / fabs(v[data_p])); }
            if (fabs(v[data_p + 1]) > R) {
                v[data_p + 1] = v[data_p + 1] * (R / fabs(v[data_p + 1]));
            }
            if (fabs(alpha) > 2. * R) { alpha = alpha * (2. * R / fabs(alpha)); }
            memcpy(vd, v, sizeof(double) * (data_p + 2));
            cblas_daxpy(data_p + 2, -1., v_1, 1, vd, 1);
            double norm_vd = sqrt(cblas_ddot(data_p + 2, vd, 1, vd, 1));
            if (norm_vd > para_r) { cblas_dscal(data_p + 2, para_r / norm_vd, vd, 1); }
            memcpy(v, vd, sizeof(double) * (data_p + 2));
            cblas_daxpy(data_p + 2, 1., v_1, 1, v, 1);
            ad = alpha - alpha_1;
            if (fabs(ad) > D) { ad = ad * (D / fabs(ad)); }
            alpha = alpha_1 + ad;
            cblas_daxpy(data_p + 2, 1., v, 1, v_sum, 1);
            memcpy(v_ave, v_sum, sizeof(double) * (data_p + 2));
            cblas_dscal(data_p + 2, 1. / (kk + 1.), v_ave, 1);
            t = t + 1.0;
            if ((fmod(t, para_step_len) == 1.)) { // to calculate AUC score
                double t_eval = clock();
                for (int q = 0; q < data_n; q++) {
                    xt_inds = x_tr_inds + x_tr_poss[q];
                    xt_vals = x_tr_vals + x_tr_poss[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++)
                        y_pred[q] += (v_ave[xt_inds[tt]] * xt_vals[tt]);
                }
                re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
                memset(y_pred, 0, sizeof(double) * data_n);
                re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
            }
        }
        para_r = para_r / 2.;
        double tmp1 = 12. * sqrt(2.) * (2. + sqrt(2. * log(12. / delta))) * R;
        double tmp2 = fmin(p_hat, 1. - p_hat) * n_0 - sqrt(2. * n_0 * log(12. / delta));
        if (tmp2 > 0) { D = 2. * sqrt(2.) * para_r + tmp1 / sqrt(tmp2); } else { D = 1e7; }
        tmp1 = 288. * (pow(2. + sqrt(2. * log(12 / delta)), 2.));
        tmp2 = fmin(p_hat, 1. - p_hat) - sqrt(2. * log(12 / delta) / n_0);
        if (tmp2 > 0) { beta_new = 9. + tmp1 / tmp2; } else { beta_new = 1e7; }
        eta = fmin(sqrt(beta_new / beta) * eta / 2, eta);
        beta = beta_new;
        if (sp > 0.0) {
            memcpy(m_pos, sx_pos, sizeof(double) * data_p);
            cblas_dscal(data_p, 1. / sp, m_pos, 1);
        }
        if (sp < t) {
            cblas_dcopy(data_p, sx_neg, 1, m_neg, 1);
            cblas_dscal(data_p, 1. / (t - sp), m_neg, 1);
        }
        cblas_dcopy(data_p + 2, v_ave, 1, v_1, 1);
        cblas_dcopy(data_p, m_neg, 1, tmp_proj, 1);
        cblas_daxpy(data_p, -1., m_pos, 1, tmp_proj, 1);
        alpha_1 = cblas_ddot(data_p, v_ave, 1, tmp_proj, 1);
        // wt_bar update
        cblas_dscal(data_p, k / (k + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (k + 1.), v_ave, 1, re_wt_bar, 1);
    }
    cblas_dcopy(data_p, v_ave, 1, re_wt, 1);
    cblas_dscal(*re_len_auc, 1. / CLOCKS_PER_SEC, re_rts, 1);
    free(y_pred);
    free(tmp_proj);
    free(vd);
    free(v);
    free(v_sum);
    free(v_ave);
    free(gd);
    free(m_neg);
    free(m_pos);
    free(sx_neg);
    free(sx_pos);
    free(v_1);
}