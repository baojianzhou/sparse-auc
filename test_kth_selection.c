//
// Created by baojian on 9/2/19.
//
#include "kth_selection.h"

int array_simulate_uniform(double *arr, int n, double rand_low, double rand_high) {
    for (int j = 0; j < n; j++) {
        arr[j] = drand48() * (rand_high - rand_low);
    }
}

int array_simulate_rand_int(double *arr, int n, int mod) {
    for (int j = 0; j < n; j++) {
        arr[j] = mrand48() % mod;
    }
}

int array_simulate_most_zeros(double *arr, int n, double proportion) {
    for (int j = 0; j < n; j++) {
        if (mrand48() < proportion) {
            arr[j] = drand48();
        } else {
            arr[j] = 0.0;
        }
    }
}

int main() {
    srand48(time(0));
    clock_t begin = clock(), begin_kth;
    int n = 1000, k = 100;
    double *arr = malloc(sizeof(double) * n);
    double total_v1 = 0.0, total_run_time_v1 = 0.0;
    double total_v2 = 0.0, total_run_time_v2 = 0.0;
    double total_v3 = 0.0, total_run_time_v3 = 0.0;
    double total_v4 = 0.0, total_run_time_v4 = 0.0;
    double total_v5 = 0.0, total_run_time_v5 = 0.0;
    double total_v6 = 0.0, total_run_time_v6 = 0.0;
    double total_v7 = 0.0, total_run_time_v7 = 0.0;
    double total_v8 = 0.0, total_run_time_v8 = 0.0;
    for (int i = 0; i < 100000; i++) {
        array_simulate_most_zeros(arr, n, 0.01);
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
    }
    free(arr);
    double time_spent = (double) (clock() - begin) / CLOCKS_PER_SEC;
    printf("------ quick_select_v1 ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v1 / CLOCKS_PER_SEC, total_v1);
    printf("------ quick_select_v2 ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v2 / CLOCKS_PER_SEC, total_v2);
    printf("------ max heap ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v3 / CLOCKS_PER_SEC, total_v3);
    printf("------ floyd rivest ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v4 / CLOCKS_PER_SEC, total_v4);
    printf("------ quick select v3 ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v5 / CLOCKS_PER_SEC, total_v5);
    printf("------ quick select wirth ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v6 / CLOCKS_PER_SEC, total_v6);
    printf("------ quick select 3 median ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v7 / CLOCKS_PER_SEC, total_v7);
    printf("------ floyd rivest with 3 median ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v8 / CLOCKS_PER_SEC, total_v8);
    return 0;
}