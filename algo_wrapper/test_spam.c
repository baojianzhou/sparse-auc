//
// Created by baojian on 9/10/19.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "auc_opt_methods.h"

void read_02_usps(const char *file_name, double *x_tr, double *y_tr) {
    FILE *fp;
    long line_len;
    char *line = NULL;
    char tokens[600][50];
    size_t len = 0, num_lines = 0;
    if ((fp = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "cannot open: %s!\n", file_name);
        exit(EXIT_FAILURE);
    }
    int n = 9298, p = 256;
    while ((line_len = getline(&line, &len, fp)) != -1) {
        for (int k = 0; k < line_len; k++) {
            if (line[k] == '\t') {
                line[k] = ' ';
            }
            if (line[k] == ':') {
                line[k] = ' ';
            }
        }
        int tokens_size = 0;
        for (char *token = strtok(line, " "); token != NULL; token = strtok(NULL, " ")) {
            strcpy(tokens[tokens_size++], token);
        }
        //first line is class num
        y_tr[num_lines] = (int) strtol(tokens[0], NULL, 10);
        for (int i = 0; i < p; i++) {
            int cur_ind = (int) strtol(tokens[i * 2 + 1], NULL, 10) - 1;
            int row_ind = (int) (num_lines * p);
            x_tr[row_ind + cur_ind] = strtod(tokens[i * 2 + 2], NULL);
        }
        // normalize the data sample
        double cur_x_norm = sqrt(cblas_ddot(p, x_tr + num_lines * p, 1, x_tr + num_lines * p, 1));
        cblas_dscal(p, 1. / cur_x_norm, x_tr + num_lines * p, 1);
        num_lines++;
    }
    printf("total number of data samples: %d\n", (int) num_lines);

    int labels[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int posi_index = 0;
    while (true) {
        if (posi_index == 5) {
            break;
        } else {
            posi_index = 0;
        }
        for (int k = 0; k < 10; k++) {
            if (drand48() <= 0.5) {
                labels[k] = 1;
                posi_index++;
            } else {
                labels[k] = -1;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        y_tr[i] = labels[(int) (y_tr[i] - 1)];
    }
    fclose(fp);
}

void print_input_para(spam_para *para) {
    printf("num_tr: %d\n", para->num_tr);
    printf("p: %d\n", para->p);
}

void test_02_usps() {
    spam_para *para = malloc(sizeof(spam_para));
    spam_results *results = malloc(sizeof(spam_results));

    // input parameters
    para->num_tr = 9298;
    para->p = 256;
    para->x_tr = malloc(sizeof(double) * (para->num_tr * para->p));
    para->y_tr = malloc(sizeof(double) * para->num_tr);
    para->para_xi = 20.0;
    para->para_reg_opt = 1; // l2-regularization
    para->para_step_len = 1000;
    para->para_l1_reg = 0.0;
    para->para_l2_reg = 1e-3;
    para->para_num_passes = 40;
    para->is_sparse = false;
    print_input_para(para);

    // results
    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 5;
    results->t_eval_time = 0.0;
    results->wt = malloc(sizeof(double) * para->p);
    results->wt_bar = malloc(sizeof(double) * para->p);
    results->t_indices = malloc(sizeof(int) * total_num_eval);
    results->t_run_time = malloc(sizeof(double) * total_num_eval);
    results->t_auc = malloc(sizeof(double) * total_num_eval);
    results->t_index = 0;

    read_02_usps("/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/processed_usps.txt",
                 para->x_tr, para->y_tr);
    int len_beta = 6, len_xi = 7;
    double beta[] = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    double xi[] = {0.1, 0.5, 1.0, 5.0, 10., 11., 12., 13., 15.};
    for (int i = 0; i < len_beta; i++) {
        for (int j = 0; j < len_xi; j++) {
            para->para_l2_reg = beta[i];
            para->para_xi = xi[j];
            para->num_tr = (int) (.8 * 9298);
            algo_spam(para, results);
            double mean_auc = 0.0, max_auc = 0.0;
            for (int k = 0; k < results->t_index; k++) {
                mean_auc += results->t_auc[k];
                if (results->t_auc[k] > max_auc) {
                    max_auc = results->t_auc[k];
                }
            }
            printf("beta: %.e xi: %02.1f max-auc: %.4f mean-auc: %.4f\n",
                   para->para_l2_reg, para->para_xi, max_auc, mean_auc / results->t_index);
        }
    }
    free(para->y_tr);
    free(para->x_tr);
}

void test_matrix_vector_multi() {
    int n = 4, p = 2;
    double *x_tr = malloc(sizeof(double) * n * p);
    double *wt = malloc(sizeof(double) * p);
    double *y_tr = malloc(sizeof(double) * n);
    x_tr[0] = 1.0, x_tr[1] = 1.0, x_tr[2] = 1.0, x_tr[3] = 1.0;
    x_tr[4] = 1.0, x_tr[5] = 1.0, x_tr[6] = 1.0, x_tr[7] = 1.0;
    wt[0] = 1.0, wt[1] = 1.0;
    y_tr[0] = 1.0, y_tr[1] = 1.0, y_tr[2] = 1.0, y_tr[3] = 1.0;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, p, 1., x_tr, p, wt, 1, 1.0, y_tr, 1);
    printf("%.4f %.4f %.4f %.4f\n", y_tr[0], y_tr[1], y_tr[2], y_tr[3]);
}

int main() {
    // test_matrix_vector_multi();
    test_02_usps();
    return 0;
}