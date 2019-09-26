#include <Python.h>
#include <numpy/arrayobject.h>
#include "auc_opt_methods.h"


static PyObject *test(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    double sum = 0.0;
    PyArrayObject *x_tr_;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_tr_)) { return NULL; }
    int n = (int) (x_tr_->dimensions[0]);     // number of samples
    int p = (int) (x_tr_->dimensions[1]);     // number of features
    printf("%d %d\n", n, p);
    double *x_tr = PyArray_DATA(x_tr_);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", x_tr[i * p + j]);
            sum += x_tr[i * p + j];
        }
        printf("\n");
    }
    PyObject *results = PyFloat_FromDouble(sum);
    return results;
}

static PyObject *wrap_algo_solam(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }

    PyArrayObject *x_tr, *y_tr;
    double para_r, para_xi;
    int para_num_pass, verbose;
    if (!PyArg_ParseTuple(args, "O!O!ddii",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &para_r,
                          &para_xi,
                          &para_num_pass,
                          &verbose)) { return NULL; }
    int num_tr = (int) x_tr->dimensions[0];
    int p = (int) x_tr->dimensions[1];

    if (verbose > 0) {
        printf("num_tr: %d p: %d xi: %.4f r: %.4f num_passes: %d\n",
               num_tr, p, para_xi, para_r, para_num_pass);
    }

    solam_results *result = malloc(sizeof(solam_results));
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->a = 0.0;
    result->b = 0.0;

    __solam((double *) PyArray_DATA(x_tr), (double *) PyArray_DATA(y_tr), num_tr, p,
            para_r, para_xi, para_num_pass, verbose, result);

    PyObject *results = PyTuple_New(4);
    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *a = PyFloat_FromDouble(result->a);
    PyObject *b = PyFloat_FromDouble(result->b);
    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, a);
    PyTuple_SetItem(results, 3, b);
    free(result->wt_bar);
    free(result->wt);
    free(result);
    return results;
}


static PyObject *wrap_algo_solam_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_tr_indices, *x_tr_values, *x_tr_lens, *x_tr_posis, *y_tr;
    int p, para_num_pass, verbose;
    double para_r, para_xi;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iddii",
                          &PyArray_Type, &x_tr_values,
                          &PyArray_Type, &x_tr_indices,
                          &PyArray_Type, &x_tr_lens,
                          &PyArray_Type, &x_tr_posis,
                          &PyArray_Type, &y_tr,
                          &p,
                          &para_r,
                          &para_xi,
                          &para_num_pass,
                          &verbose)) { return NULL; }
    int num_tr = (int) y_tr->dimensions[0];
    if (verbose > 0) {
        printf("num_tr: %d p: %d\n", num_tr, p);
    }
    solam_results *result = malloc(sizeof(solam_results));
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->a = 0.0;
    result->b = 0.0;
    __solam_sparse((double *) PyArray_DATA(x_tr_values),
                   (int *) PyArray_DATA(x_tr_indices),
                   (int *) PyArray_DATA(x_tr_lens),
                   (int *) PyArray_DATA(x_tr_posis),
                   (double *) PyArray_DATA(y_tr), num_tr, p,
                   para_r, para_xi, para_num_pass, verbose, result);
    if (verbose > 0) {
        printf("norm(wt): %.4f\n", cblas_ddot(p, result->wt, 1, result->wt, 1));
        printf("norm(wt-bar): %.4f\n", cblas_ddot(p, result->wt_bar, 1, result->wt_bar, 1));
    }
    PyObject *results = PyTuple_New(4);
    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *a = PyFloat_FromDouble(result->a);
    PyObject *b = PyFloat_FromDouble(result->b);
    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, a);
    PyTuple_SetItem(results, 3, b);
    free(result->wt);
    free(result->wt_bar);
    free(result);
    return results;
}


static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    double para_xi, para_l1_reg, para_l2_reg;
    int para_reg_opt, para_num_passes, para_step_len, verbose, num_tr, p;
    PyArrayObject *x_tr, *y_tr;
    if (!PyArg_ParseTuple(args, "O!O!dddiiii",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &para_xi,
                          &para_l1_reg,
                          &para_l2_reg,
                          &para_reg_opt,
                          &para_num_passes,
                          &para_step_len,
                          &verbose)) { return NULL; }

    num_tr = (int) x_tr->dimensions[0];
    p = (int) x_tr->dimensions[1];

    spam_results *result = malloc(sizeof(spam_results));
    int total_num_eval = (num_tr * para_num_passes) / para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;

    // summary of the data
    if (verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d\n", num_tr, p);
        printf("para_xi: %04e para_l1_reg: %04e para_l2_reg: %04e\n",
               para_xi, para_l1_reg, para_l2_reg);
        printf("reg_option: %d num_passes: %d step_len: %d\n",
               para_reg_opt, para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }


    //call SOLAM algorithm
    __spam((double *) PyArray_DATA(x_tr),
           (double *) PyArray_DATA(y_tr), p, num_tr, para_xi, para_l1_reg, para_l2_reg,
           para_num_passes, para_step_len, para_reg_opt, verbose, result);

    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }

    for (int i = 0; i < result->t_index; i++) {
        PyList_SetItem(t_run_time, i, PyFloat_FromDouble(result->t_run_time[i]));
        PyList_SetItem(t_auc, i, PyFloat_FromDouble(result->t_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, t_run_time);
    PyTuple_SetItem(results, 3, t_auc);
    PyTuple_SetItem(results, 4, PyInt_FromLong(result->t_index));
    free(result->wt);
    free(result->wt_bar);
    free(result->t_indices);
    free(result->t_run_time);
    free(result->t_auc);
    free(result);
    return results;
}


static PyObject *wrap_algo_spam_sparse(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm with sparse data.
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    double para_xi, para_l1_reg, para_l2_reg;
    int para_reg_opt, para_num_passes, para_step_len, verbose, num_tr, p;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iidddiiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &p,
                          &num_tr,
                          &para_xi,
                          &para_l1_reg,
                          &para_l2_reg,
                          &para_reg_opt,
                          &para_num_passes,
                          &para_step_len,
                          &verbose)) { return NULL; }

    spam_results *result = malloc(sizeof(spam_results));
    int total_num_eval = (num_tr * para_num_passes) / para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;

    // summary of the data
    if (verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d\n", num_tr, p);
        printf("para_xi: %04e para_l1_reg: %04e para_l2_reg: %04e\n",
               para_xi, para_l1_reg, para_l2_reg);
        printf("reg_option: %d num_passes: %d step_len: %d\n",
               para_reg_opt, para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }

    //call SOLAM algorithm
    __spam_sparse((double *) PyArray_DATA(x_values),
                  (int *) PyArray_DATA(x_indices),
                  (int *) PyArray_DATA(x_positions),
                  (int *) PyArray_DATA(x_len_list),
                  (double *) PyArray_DATA(y_tr),
                  p, num_tr, para_xi, para_l1_reg, para_l2_reg, para_num_passes,
                  para_step_len, para_reg_opt, verbose, result);

    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }

    for (int i = 0; i < result->t_index; i++) {
        PyList_SetItem(t_run_time, i, PyFloat_FromDouble(result->t_run_time[i]));
        PyList_SetItem(t_auc, i, PyFloat_FromDouble(result->t_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, t_run_time);
    PyTuple_SetItem(results, 3, t_auc);
    PyTuple_SetItem(results, 4, PyInt_FromLong(result->t_index));

    free(result->wt);
    free(result->wt_bar);
    free(result->t_indices);
    free(result->t_run_time);
    free(result->t_auc);
    free(result);
    return results;
}


static PyObject *wrap_algo_sht_am(PyObject *self, PyObject *args) {
    /*
     * Wrapper of the StoIHT for AUC algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_tr, *y_tr;
    double para_xi, para_l2_reg;
    int para_sparsity, para_b, para_num_passes, para_step_len, verbose;
    if (!PyArg_ParseTuple(args, "O!O!iiddiii",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &para_sparsity,
                          &para_b,
                          &para_xi,
                          &para_l2_reg,
                          &para_num_passes,
                          &para_step_len,
                          &verbose)) { return NULL; }

    int num_tr = (int) x_tr->dimensions[0];
    int p = (int) x_tr->dimensions[1];
    sht_am_results *result = malloc(sizeof(sht_am_results));

    int total_num_eval = (num_tr * para_num_passes) / para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;
    if (verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d block_size: %d\n",
               num_tr, p, para_b);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n",
               para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    __sht_am((double *) PyArray_DATA(x_tr),
             (double *) PyArray_DATA(y_tr), p, num_tr, para_b, para_xi, para_l2_reg,
             para_sparsity, para_num_passes, para_step_len, verbose, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }

    for (int i = 0; i < result->t_index; i++) {
        PyList_SetItem(t_run_time, i, PyFloat_FromDouble(result->t_run_time[i]));
        PyList_SetItem(t_auc, i, PyFloat_FromDouble(result->t_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, t_run_time);
    PyTuple_SetItem(results, 3, t_auc);
    PyTuple_SetItem(results, 4, PyInt_FromLong(result->t_index));
    free(result->wt);
    free(result->wt_bar);
    free(result->t_indices);
    free(result->t_run_time);
    free(result->t_auc);
    free(result);
    return results;
}


static PyObject *wrap_algo_sht_am_sparse(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm with sparse data.
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    double para_xi, para_l2_reg;
    int num_tr, p, para_b, para_sparsity, para_num_passes, para_step_len, verbose;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiddiiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &p,
                          &num_tr,
                          &para_b,
                          &para_xi,
                          &para_l2_reg,
                          &para_sparsity,
                          &para_num_passes,
                          &para_step_len,
                          &verbose)) { return NULL; }
    sht_am_results *result = malloc(sizeof(sht_am_results));

    int total_num_eval = (num_tr * para_num_passes) / para_step_len + 1;
    result->t_index = 0;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);

    // summary of the data
    if (verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d\n", num_tr, p);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    __sht_am_sparse((double *) PyArray_DATA(x_values),
                    (int *) PyArray_DATA(x_indices),
                    (int *) PyArray_DATA(x_positions),
                    (int *) PyArray_DATA(x_len_list),
                    (double *) PyArray_DATA(y_tr),
                    p, num_tr, para_b, para_xi, para_l2_reg, para_sparsity,
                    para_num_passes, para_step_len, verbose, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }

    for (int i = 0; i < result->t_index; i++) {
        PyList_SetItem(t_run_time, i, PyFloat_FromDouble(result->t_run_time[i]));
        PyList_SetItem(t_auc, i, PyFloat_FromDouble(result->t_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, t_run_time);
    PyTuple_SetItem(results, 3, t_auc);
    PyTuple_SetItem(results, 4, PyInt_FromLong(result->t_index));
    free(result->wt);
    free(result->wt_bar);
    free(result->t_indices);
    free(result->t_run_time);
    free(result->t_auc);
    free(result);
    return results;
}


static PyObject *wrap_algo_graph_am(PyObject *self, PyObject *args) {
    /*
     * Wrapper of the Graph for AUC algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    int para_step_len, para_sparsity, para_b, para_num_passes, verbose;
    double para_xi, para_l2_reg;
    PyArrayObject *x_tr, *y_tr, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!iiddiiiiO!O!",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &para_sparsity,
                          &para_b,
                          &para_xi,
                          &para_l2_reg,
                          &para_num_passes,
                          &para_step_len,
                          &verbose,
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &weights_)) { return NULL; }

    int num_tr = (int) x_tr->dimensions[0];
    int p = (int) x_tr->dimensions[1];
    int m = (int) edges_->dimensions[0];
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    for (int i = 0; i < m; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    graph_am_results *result = malloc(sizeof(graph_am_results));
    int total_num_eval = (num_tr * para_num_passes) / para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;
    if (verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d block_size: %d\n", num_tr, p, para_b);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }

    //call SOLAM algorithm
    __graph_am((double *) PyArray_DATA(x_tr),
               (double *) PyArray_DATA(y_tr),
               p, num_tr, para_b, para_xi, para_l2_reg, para_sparsity, para_num_passes,
               para_step_len, verbose, edges, (double *) PyArray_DATA(weights_), m, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }

    for (int i = 0; i < result->t_index; i++) {
        PyList_SetItem(t_run_time, i, PyFloat_FromDouble(result->t_run_time[i]));
        PyList_SetItem(t_auc, i, PyFloat_FromDouble(result->t_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, t_run_time);
    PyTuple_SetItem(results, 3, t_auc);
    PyTuple_SetItem(results, 4, PyInt_FromLong(result->t_index));

    free(edges);
    free(result->wt);
    free(result->wt_bar);
    free(result->t_indices);
    free(result->t_run_time);
    free(result->t_auc);
    free(result);
    return results;
}


static PyObject *wrap_algo_graph_am_sparse(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm with sparse data.
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }

    int p, para_b, num_tr, para_sparsity, para_num_passes, para_step_len, verbose;
    double para_xi, para_l2_reg;
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiiddiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &num_tr,
                          &p,
                          &para_b,
                          &para_sparsity,
                          &para_xi,
                          &para_l2_reg,
                          &para_num_passes,
                          &para_step_len,
                          &verbose)) { return NULL; }

    graph_am_results *result = malloc(sizeof(graph_am_results));

    int total_num_eval = (num_tr * para_num_passes) / para_step_len + 1;
    result->t_index = 0;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * p);
    result->wt_bar = malloc(sizeof(double) * p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);

    __graph_am_sparse((double *) PyArray_DATA(x_values), (int *) PyArray_DATA(x_indices),
                      (int *) PyArray_DATA(x_positions), (int *) PyArray_DATA(x_len_list),
                      (double *) PyArray_DATA(y_tr), p, num_tr, para_b, para_sparsity, para_xi,
                      para_l2_reg, para_num_passes, para_step_len, verbose, result);

    // summary of the data
    printf("--------------------------------------------------------------\n");
    printf("num_tr: %d p: %d\n", num_tr, p);
    printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
    printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
    printf("num_eval: %d\n", total_num_eval);
    printf("--------------------------------------------------------------\n");

    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt_bar[i]));
    }

    for (int i = 0; i < result->t_index; i++) {
        PyList_SetItem(t_run_time, i, PyFloat_FromDouble(result->t_run_time[i]));
        PyList_SetItem(t_auc, i, PyFloat_FromDouble(result->t_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, t_run_time);
    PyTuple_SetItem(results, 3, t_auc);
    PyTuple_SetItem(results, 4, PyInt_FromLong(result->t_index));
    free(result->wt);
    free(result->wt_bar);
    free(result->t_indices);
    free(result->t_run_time);
    free(result->t_auc);
    free(result);
    return results;
}


static PyObject *wrap_algo_opauc(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm with sparse data.
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_tr, *y_tr;
    int p, n;
    double eta, lambda;

    if (!PyArg_ParseTuple(args, "O!O!iidd",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &p, &n, &eta, &lambda)) { return NULL; }
    double *wt = malloc(sizeof(double) * p);
    double *wt_bar = malloc(sizeof(double) * p);
    algo_opauc((double *) PyArray_DATA(x_tr),
               (double *) PyArray_DATA(y_tr), p, n, eta, lambda, wt, wt_bar);
    PyObject *results = PyTuple_New(2);
    PyObject *p_wt = PyList_New(p);
    PyObject *p_wt_bar = PyList_New(p);
    for (int i = 0; i < p; i++) {
        PyList_SetItem(p_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(p_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    PyTuple_SetItem(results, 0, p_wt);
    PyTuple_SetItem(results, 1, p_wt_bar);
    free(wt);
    free(wt_bar);
    return results;
}

static PyObject *wrap_algo_fsauc(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the FSAUC algorithm.
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_tr, *y_tr;
    int p, n, num_passes, verbose = 0;
    double para_r, para_g;

    if (!PyArg_ParseTuple(args, "O!O!iiidd",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &p, &n, &num_passes, &para_r, &para_g)) { return NULL; }
    if (verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d \n", n, p);
        printf("para_r: %04e para_g: %04e num_passes: %d \n", para_r, para_g, num_passes);
        printf("--------------------------------------------------------------\n");

    }
    double *wt = malloc(sizeof(double) * p);
    double *wt_bar = malloc(sizeof(double) * p);
    algo_fsauc((double *) PyArray_DATA(x_tr),
               (double *) PyArray_DATA(y_tr), p, n, para_r, para_g, num_passes, wt, wt_bar);
    PyObject *results = PyTuple_New(2);
    PyObject *p_wt = PyList_New(p);
    PyObject *p_wt_bar = PyList_New(p);
    for (int i = 0; i < p; i++) {
        PyList_SetItem(p_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(p_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    PyTuple_SetItem(results, 0, p_wt);
    PyTuple_SetItem(results, 1, p_wt_bar);
    free(wt);
    free(wt_bar);
    return results;
}


// wrap_algo_solam_sparse
static PyMethodDef sparse_methods[] = {
        {"c_test",                 (PyCFunction) test,                      METH_VARARGS, "docs"},
        {"c_algo_solam",           (PyCFunction) wrap_algo_solam,           METH_VARARGS, "docs"},
        {"c_algo_spam",            (PyCFunction) wrap_algo_spam,            METH_VARARGS, "docs"},
        {"c_algo_sht_am",          (PyCFunction) wrap_algo_sht_am,          METH_VARARGS, "docs"},
        {"c_algo_graph_am",        (PyCFunction) wrap_algo_graph_am,        METH_VARARGS, "docs"},
        {"c_algo_opauc",           (PyCFunction) wrap_algo_opauc,           METH_VARARGS, "docs"},
        {"c_algo_fsauc",           (PyCFunction) wrap_algo_fsauc,           METH_VARARGS, "docs"},

        {"c_algo_solam_sparse",    (PyCFunction) wrap_algo_solam_sparse,    METH_VARARGS, "docs"},
        {"c_algo_sht_am_sparse",   (PyCFunction) wrap_algo_sht_am_sparse,   METH_VARARGS, "docs"},
        {"c_algo_spam_sparse",     (PyCFunction) wrap_algo_spam_sparse,     METH_VARARGS, "docs"},
        {"c_algo_graph_am_sparse", (PyCFunction) wrap_algo_graph_am_sparse, METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods, "some docs for solam algorithm.");
    import_array();
}

int main() {
    printf("test of main wrapper!\n");
}