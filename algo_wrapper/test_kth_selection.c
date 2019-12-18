/**
 * This program tests different selection algorithms.
 * A selection algorithm is a method for finding the
 * k-th smallest/largest number in an array
 */
#include "kth_selection.h"


/**
 * Generate a random array where each element is from
 * uniform distribution, i.e., arr[i] ~ U[u_low, u_high]
 *
 * @param arr generated random array.
 * @param n the size of the random array.
 * @param u_low the lower bound of uniform distribution.
 * @param u_high the upper bound of uniform distribution.
 */
bool gen_array_uniform(double *arr, int n, double u_low, double u_high) {
    if ((u_low > u_high) || (n <= 0)) {
        return false;
    }
    for (int j = 0; j < n; j++) {
        arr[j] = drand48() * (u_high - u_low);
    }
    return true;
}

/**
 * Generate a random array where each element is randomly picked from [0, bound].
 *
 * @param arr generated random array.
 * @param n the size of the random array.
 * @param bound the upper bound of uniform distribution.
 */
bool gen_array_int(double *arr, int n, int bound) {
    if ((bound <= 0) || (n <= 0)) {
        return false;
    }
    for (int j = 0; j < n; j++) {
        arr[j] = mrand48() % bound;
    }
    return true;
}

/**
 * Generate a random array where proportion*100 % of elements are zeros.
 *
 * @param arr generated random array.
 * @param n the size of the random array.
 * @param proportion the proportion of elements are zeros.
 */
bool gen_array_prop_zeros(double *arr, int n, double proportion) {
    if ((proportion <= 0) || (proportion > 1) || (n <= 0)) {
        return false;
    }
    memset(arr, 0, (size_t) n);
    for (int j = 0; j < n; j++) {
        if (mrand48() < proportion) {
            arr[j] = drand48();
        }
    }
    return 0;
}

bool get_array(double *arr, int n, int arr_opt) {
    if (arr_opt == 0) {
        return gen_array_uniform(arr, n, 0.0, 10.0);
    } else if (arr_opt == 1) {
        return gen_array_int(arr, n, 100);
    } else if (arr_opt == 2) {
        return gen_array_prop_zeros(arr, n, 0.8);
    } else {
        gen_array_uniform(arr, n, 0.0, 10.0);
    }
    return false;
}


int main() {
    int arr_opt = 0;
    srand48(time(0));
    clock_t begin = clock(), begin_kth;
    int n = 551970, k = 5000;
    int num_iter = 1000;

    double *arr = malloc(sizeof(double) * n);
    int *sorted_set = malloc(sizeof(int) * n);
    double total_v1 = 0.0, total_run_time_v1 = 0.0;
    double total_v2 = 0.0, total_run_time_v2 = 0.0;
    double total_v3 = 0.0, total_run_time_v3 = 0.0;
    double total_v4 = 0.0, total_run_time_v4 = 0.0;
    double total_v5 = 0.0, total_run_time_v5 = 0.0;
    double total_v6 = 0.0, total_run_time_v6 = 0.0;
    double total_v7 = 0.0, total_run_time_v7 = 0.0;
    double total_v8 = 0.0, total_run_time_v8 = 0.0;
    double total_v9 = 0.0, total_run_time_v9 = 0.0;
    for (int i = 0; i < num_iter; i++) {
        get_array(arr, n, arr_opt);
        begin_kth = clock();
        total_v9 += fabs(kth_largest_quick_sort(arr, sorted_set, k, n));
        total_run_time_v9 += (clock() - begin_kth);

        begin_kth = clock();
        total_v1 += fabs(kth_largest_quick_select_v1(arr, n, k));
        total_run_time_v1 += clock() - begin_kth;

        begin_kth = clock();
        total_v2 += fabs(kth_largest_quick_select_v2(arr, n, k));
        total_run_time_v2 += clock() - begin_kth;

        begin_kth = clock();
        total_v3 += fabs(kth_largest_max_heap(arr, n, k));
        total_run_time_v3 += clock() - begin_kth;

        begin_kth = clock();
        total_v4 += fabs(kth_largest_floyd_rivest(arr, n, k));
        total_run_time_v4 += clock() - begin_kth;

        begin_kth = clock();
        total_v5 += fabs(kth_largest_quick_select_v3(arr, n, k));
        total_run_time_v5 += clock() - begin_kth;

        begin_kth = clock();
        total_v6 += fabs(kth_largest_wirth(arr, n, k));
        total_run_time_v6 += clock() - begin_kth;

        begin_kth = clock();
        total_v7 += fabs(kth_largest_quick_select_v4(arr, n, k));
        total_run_time_v7 += clock() - begin_kth;

        begin_kth = clock();
        total_v8 += fabs(kth_largest_floyd_rivest_v2(arr, n, k));
        total_run_time_v8 += clock() - begin_kth;

        if (total_v1 != total_v2) {
            printf("%.8f %.8f %d\n", total_v1, total_v2, i);
            break;
        }
        if (total_v2 != total_v3) {
            printf("%.8f %.8f %d\n", total_v2, total_v3, i);
            break;
        }
        if (total_v3 != total_v4) {
            printf("%.8f %.8f %d\n", total_v3, total_v4, i);
            break;
        }
        if (total_v4 != total_v5) {
            printf("%.8f %.8f %d\n", total_v4, total_v5, i);
            break;
        }
        if (total_v5 != total_v6) {
            printf("%.8f %.8f %d\n", total_v5, total_v6, i);
            break;
        }
        if (total_v6 != total_v7) {
            printf("%.8f %.8f %d\n", total_v6, total_v7, i);
            break;
        }
        if (total_v7 != total_v8) {
            printf("%.8f %.8f %d\n", total_v7, total_v8, i);
            break;
        }

        if (total_v8 != total_v9) {
            printf("%.8f %.8f %d\n", total_v7, total_v8, i);
            break;
        }
    }
    free(arr);
    free(sorted_set);
    double time_spent = (double) (clock() - begin) / CLOCKS_PER_SEC;
    total_run_time_v1 = total_run_time_v1 / CLOCKS_PER_SEC;
    total_run_time_v2 = total_run_time_v2 / CLOCKS_PER_SEC;
    total_run_time_v3 = total_run_time_v3 / CLOCKS_PER_SEC;
    total_run_time_v4 = total_run_time_v4 / CLOCKS_PER_SEC;
    total_run_time_v5 = total_run_time_v5 / CLOCKS_PER_SEC;
    total_run_time_v6 = total_run_time_v6 / CLOCKS_PER_SEC;
    total_run_time_v7 = total_run_time_v7 / CLOCKS_PER_SEC;
    total_run_time_v8 = total_run_time_v8 / CLOCKS_PER_SEC;
    total_run_time_v9 = total_run_time_v9 / CLOCKS_PER_SEC;
    printf("total running time: %.2f(s) \n", time_spent);
    printf("%20s -- total time -- aver select -- \n", "method");
    printf("%20s -- %10.4f -- %11.4f\n",
           "quick_select_v1", total_run_time_v1, total_run_time_v1 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "quick_select_v2", total_run_time_v2, total_run_time_v2 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "quick_select_v3", total_run_time_v5, total_run_time_v5 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "max heap", total_run_time_v3, total_run_time_v3 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "floyd rivest", total_run_time_v4, total_run_time_v4 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "quick select wirth", total_run_time_v6, total_run_time_v6 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "quick with 3 median", total_run_time_v7, total_run_time_v7 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "floyd with 3 median", total_run_time_v8, total_run_time_v8 / (double) num_iter);
    printf("%20s -- %10.4f -- %11.4f\n",
           "quick sort", total_run_time_v9, total_run_time_v9 / (double) num_iter);
    return 0;
}