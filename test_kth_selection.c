//
// Created by baojian on 9/2/19.
//
#include "kth_selection.h"

int main() {
    srand48(time(0));
    clock_t begin = clock(), begin_kth;
    int n = 5000, k = 100;
    double *arr = malloc(sizeof(double) * n);
    double rand_low = 0.0, rand_high = 10.0;
    double total_v1 = 0.0, total_run_time_v1 = 0.0;
    double total_v2 = 0.0, total_run_time_v2 = 0.0;
    double total_v3 = 0.0, total_run_time_v3 = 0.0;
    double total_v4 = 0.0, total_run_time_v4 = 0.0;
    for (int i = 0; i < 100000; i++) {
        for (int j = 0; j < n; j++) {
            arr[j] = drand48() * (rand_high - rand_low);
        }
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
    }
    free(arr);
    double time_spent = (double) (clock() - begin) / CLOCKS_PER_SEC;
    total_run_time_v1 /= CLOCKS_PER_SEC;
    total_run_time_v2 /= CLOCKS_PER_SEC;
    total_run_time_v3 /= CLOCKS_PER_SEC;
    total_run_time_v4 /= CLOCKS_PER_SEC;
    printf("------ quick_select_v1 ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v1, total_v1);
    printf("------ quick_select_v2 ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v2, total_v2);
    printf("------ max heap ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v3, total_v3);
    printf("------ floyd rivest ------\n");
    printf("total run time: %.4f selection time: %.4f total value: %.4f\n",
           time_spent, total_run_time_v4, total_v4);
    return 0;
}