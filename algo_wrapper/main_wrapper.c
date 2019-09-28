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
    if (self != NULL) { return NULL; }
    PyArrayObject *x_tr, *y_tr;
    double para_xi, para_r;
    int para_num_passes, para_verbose;
    if (!PyArg_ParseTuple(args, "O!O!ddii",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &para_xi, &para_r, &para_num_passes, &para_verbose)) { return NULL; }
    int n = (int) x_tr->dimensions[0];
    int p = (int) x_tr->dimensions[1];
    if (para_verbose > 0) {
        printf("n: %d p: %d xi: %.4f r: %.4f para_num_passes: %d\n",
               n, p, para_xi, para_r, para_num_passes);
    }
    double *re_wt = malloc(sizeof(double) * p);
    double *re_wt_bar = malloc(sizeof(double) * p);
    _algo_solam((double *) PyArray_DATA(x_tr),
                (double *) PyArray_DATA(y_tr),
                n, p, para_xi, para_r, para_num_passes, para_verbose, re_wt, re_wt_bar);
    PyObject *results = PyTuple_New(2);
    PyObject *wt = PyList_New(p);
    PyObject *wt_bar = PyList_New(p);
    for (int i = 0; i < p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    free(re_wt_bar);
    free(re_wt);
    return results;
}


static PyObject *wrap_algo_solam_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; }
    PyArrayObject *x_tr_values, *x_tr_indices, *x_tr_posis, *x_tr_lens, *y_tr;
    int data_p, para_num_passes, para_verbose;
    double para_r, para_xi;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iddii",
                          &PyArray_Type, &x_tr_values,
                          &PyArray_Type, &x_tr_indices,
                          &PyArray_Type, &x_tr_posis,
                          &PyArray_Type, &x_tr_lens,
                          &PyArray_Type, &y_tr,
                          &data_p, &para_xi, &para_r, &para_num_passes,
                          &para_verbose)) { return NULL; }
    int data_n = (int) y_tr->dimensions[0];
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    _algo_solam_sparse((double *) PyArray_DATA(x_tr_values),
                       (int *) PyArray_DATA(x_tr_indices),
                       (int *) PyArray_DATA(x_tr_posis),
                       (int *) PyArray_DATA(x_tr_lens),
                       (double *) PyArray_DATA(y_tr),
                       data_n, data_p, para_xi, para_r, para_num_passes, para_verbose, re_wt,
                       re_wt_bar);
    PyObject *results = PyTuple_New(2);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    free(re_wt);
    free(re_wt_bar);
    return results;
}


static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; }
    double para_xi, para_l1_reg, para_l2_reg;
    int para_reg_opt, para_num_passes, para_step_len, para_verbose, data_n, data_p;
    PyArrayObject *data_x_tr, *data_y_tr;
    if (!PyArg_ParseTuple(args, "O!O!dddiiii",
                          &PyArray_Type, &data_x_tr,
                          &PyArray_Type, &data_y_tr,
                          &para_xi, &para_l1_reg, &para_l2_reg,
                          &para_reg_opt, &para_num_passes,
                          &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    int total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    double *re_auc = malloc(sizeof(double) * total_num_eval);
    if (para_verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("data_n: %d data_p: %d\n", data_n, data_p);
        printf("para_xi: %04e para_l1_reg: %04e para_l2_reg: %04e\n",
               para_xi, para_l1_reg, para_l2_reg);
        printf("reg_option: %d num_passes: %d step_len: %d\n",
               para_reg_opt, para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    _algo_spam((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
               data_n, data_p, para_xi, para_l1_reg, para_l2_reg,
               para_num_passes, para_step_len, para_reg_opt,
               para_verbose, re_wt, re_wt_bar, re_auc);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(total_num_eval);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    for (int i = 0; i < total_num_eval; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    free(re_wt);
    free(re_wt_bar);
    free(re_auc);
    return results;
}


static PyObject *wrap_algo_spam_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; }
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    double para_xi, para_l1_reg, para_l2_reg;
    int para_reg_opt, para_num_passes, para_step_len, para_verbose, data_p;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!idddiiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &data_p, &para_xi, &para_l1_reg, &para_l2_reg,
                          &para_reg_opt, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    int data_n = (int) y_tr->dimensions[0];
    int total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    double *re_auc = malloc(sizeof(double) * total_num_eval);
    if (para_verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d\n", data_n, data_p);
        printf("para_xi: %04e para_l1_reg: %04e para_l2_reg: %04e\n",
               para_xi, para_l1_reg, para_l2_reg);
        printf("reg_option: %d num_passes: %d step_len: %d\n",
               para_reg_opt, para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    _algo_spam_sparse((double *) PyArray_DATA(x_values),
                      (int *) PyArray_DATA(x_indices),
                      (int *) PyArray_DATA(x_positions),
                      (int *) PyArray_DATA(x_len_list),
                      (double *) PyArray_DATA(y_tr),
                      data_n, data_p, para_xi, para_l1_reg, para_l2_reg, para_num_passes,
                      para_step_len, para_reg_opt, para_verbose, re_wt, re_wt_bar, re_auc);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(total_num_eval);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    for (int i = 0; i < total_num_eval; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    free(re_wt);
    free(re_wt_bar);
    free(re_auc);
    return results;
}


static PyObject *wrap_algo_sht_am(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; }
    PyArrayObject *data_x_tr, *data_y_tr;
    double para_xi, para_l2_reg;
    int para_sparsity, para_b, para_num_passes, para_step_len, para_verbose;
    if (!PyArg_ParseTuple(args, "O!O!iiddiii",
                          &PyArray_Type, &data_x_tr,
                          &PyArray_Type, &data_y_tr,
                          &para_sparsity, &para_b, &para_xi, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    int data_n = (int) data_x_tr->dimensions[0];
    int data_p = (int) data_x_tr->dimensions[1];
    int total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    double *re_auc = malloc(sizeof(double) * total_num_eval);
    if (para_verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("data_n: %d data_p: %d block_size: %d\n", data_n, data_p, para_b);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    _algo_sht_am((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                 data_n, data_p, para_sparsity, para_b, para_xi, para_l2_reg, para_num_passes,
                 para_step_len, para_verbose, re_wt, re_wt_bar, re_auc);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(total_num_eval);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    for (int i = 0; i < total_num_eval; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    free(re_auc);
    free(re_wt_bar);
    free(re_wt);
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
    int data_n, data_p, para_b, para_sparsity, para_num_passes, para_step_len, para_verbose;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiddiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &data_p,
                          &para_sparsity,
                          &para_b,
                          &para_xi,
                          &para_l2_reg,
                          &para_num_passes,
                          &para_step_len,
                          &para_verbose)) { return NULL; }
    data_n = (int) y_tr->dimensions[0];
    int total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    double *re_auc = malloc(sizeof(double) * total_num_eval);
    if (para_verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d\n", data_n, data_p);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    _algo_sht_am_sparse((double *) PyArray_DATA(x_values),
                        (int *) PyArray_DATA(x_indices),
                        (int *) PyArray_DATA(x_positions),
                        (int *) PyArray_DATA(x_len_list),
                        (double *) PyArray_DATA(y_tr),
                        data_n, data_p, para_sparsity, para_b, para_xi, para_l2_reg,
                        para_num_passes, para_step_len, para_verbose, re_wt, re_wt_bar, re_auc);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(total_num_eval);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    for (int i = 0; i < total_num_eval; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    free(re_auc);
    free(re_wt_bar);
    free(re_wt);
    return results;
}


static PyObject *wrap_algo_graph_am(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; }
    PyArrayObject *data_x_tr, *data_y_tr, *graph_edges, *graph_weights;
    int para_step_len, para_sparsity, para_b, para_num_passes, para_verbose;
    double para_xi, para_l2_reg;
    if (!PyArg_ParseTuple(args, "O!O!O!O!iiddiii",
                          &PyArray_Type, &data_x_tr,
                          &PyArray_Type, &data_y_tr,
                          &PyArray_Type, &graph_edges,
                          &PyArray_Type, &graph_weights,
                          &para_sparsity, &para_b, &para_xi, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    int data_n = (int) data_x_tr->dimensions[0];
    int data_p = (int) data_x_tr->dimensions[1];
    int graph_m = (int) graph_weights->dimensions[0];
    EdgePair *edges = malloc(sizeof(EdgePair) * graph_m);
    for (int i = 0; i < graph_m; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(graph_edges, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(graph_edges, i, 1);
    }
    int total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    double *re_auc = malloc(sizeof(double) * total_num_eval);
    if (para_verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("data_n: %d data_p: %d block_size: %d", data_n, data_p, para_b);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    _algo_graph_am((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                   edges, (double *) PyArray_DATA(graph_weights), graph_m,
                   data_n, data_p, para_sparsity, para_b, para_xi, para_l2_reg, para_num_passes,
                   para_step_len, para_verbose, re_wt, re_wt_bar, re_auc);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(total_num_eval);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }

    for (int i = 0; i < total_num_eval; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    free(edges);
    free(re_auc);
    free(re_wt_bar);
    free(re_wt);
    return results;
}


static PyObject *wrap_algo_graph_am_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; }
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *data_y_tr;
    PyArrayObject *graph_edges, *graph_weights;
    int data_p, para_b, data_n, para_sparsity, para_num_passes, para_step_len, para_verbose;
    double para_xi, para_l2_reg;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iiiddiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &data_y_tr,
                          &PyArray_Type, &graph_edges,
                          &PyArray_Type, &graph_weights,
                          &data_p, &para_sparsity, &para_b, &para_xi, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_y_tr->dimensions[0];
    int graph_m = (int) graph_weights->dimensions[0];
    EdgePair *edges = malloc(sizeof(EdgePair) * graph_m);
    for (int i = 0; i < graph_m; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(graph_edges, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(graph_edges, i, 1);
    }
    int total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    double *re_wt = malloc(sizeof(double) * data_p);
    double *re_wt_bar = malloc(sizeof(double) * data_p);
    double *re_auc = malloc(sizeof(double) * total_num_eval);
    if (para_verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("data_n: %d data_p: %d block_size: %d", data_n, data_p, para_b);
        printf("para_xi: %04e para_l2_reg: %04e\n", para_xi, para_l2_reg);
        printf("num_passes: %d step_len: %d\n", para_num_passes, para_step_len);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }
    _algo_graph_am_sparse((double *) PyArray_DATA(x_values), (int *) PyArray_DATA(x_indices),
                          (int *) PyArray_DATA(x_positions), (int *) PyArray_DATA(x_len_list),
                          (double *) PyArray_DATA(data_y_tr), edges,
                          (double *) PyArray_DATA(graph_weights),
                          graph_m, data_n, data_p, para_sparsity, para_b, para_xi, para_l2_reg,
                          para_num_passes, para_step_len, para_verbose, re_wt, re_wt_bar, re_auc);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(total_num_eval);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    for (int i = 0; i < total_num_eval; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    free(edges);
    free(re_auc);
    free(re_wt_bar);
    free(re_wt);
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
    _algo_opauc((double *) PyArray_DATA(x_tr),
                (double *) PyArray_DATA(y_tr),
                p, n, eta, lambda, wt, wt_bar);
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


static PyObject *wrap_algo_opauc_sparse(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm with sparse data.
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    int p, num_tr, para_num_passes, verbose;
    double para_eta, para_lambda;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiddii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &num_tr,
                          &p,
                          &para_eta,
                          &para_lambda,
                          &para_num_passes,
                          &verbose)) { return NULL; }
    double *wt = malloc(sizeof(double) * p);
    double *wt_bar = malloc(sizeof(double) * p);
    _algo_opauc_sparse((double *) PyArray_DATA(x_values),
                       (int *) PyArray_DATA(x_indices),
                       (int *) PyArray_DATA(x_positions),
                       (int *) PyArray_DATA(x_len_list),
                       (double *) PyArray_DATA(y_tr),
                       p, num_tr, para_eta, para_lambda, wt, wt_bar);
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
    if (self != NULL) { return NULL; } // error: unknown error !!

    PyArrayObject *x_tr, *y_tr;
    int para_num_passes, para_verbose = 0;
    double para_r, para_g;

    if (!PyArg_ParseTuple(args, "O!O!idd",
                          &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr,
                          &para_num_passes, &para_r, &para_g)) { return NULL; }
    int n = (int) x_tr->dimensions[0];
    int p = (int) x_tr->dimensions[1];
    if (para_verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d \n", n, p);
        printf("para_r: %04e para_g: %04e para_num_passes: %d \n", para_r, para_g,
               para_num_passes);
        printf("--------------------------------------------------------------\n");
    }
    double *re_wt = malloc(sizeof(double) * p);
    double *re_wt_bar = malloc(sizeof(double) * p);
    _algo_fsauc((double *) PyArray_DATA(x_tr),
                (double *) PyArray_DATA(y_tr),
                p, n, para_r, para_g, para_num_passes, re_wt, re_wt_bar);
    PyObject *results = PyTuple_New(2);
    PyObject *p_wt = PyList_New(p);
    PyObject *p_wt_bar = PyList_New(p);
    for (int i = 0; i < p; i++) {
        PyList_SetItem(p_wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(p_wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    PyTuple_SetItem(results, 0, p_wt);
    PyTuple_SetItem(results, 1, p_wt_bar);
    free(re_wt);
    free(re_wt_bar);
    return results;
}


static PyObject *wrap_algo_fsauc_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    int p, num_tr, para_num_passes, para_step_len, verbose = 0;
    double para_r, para_g;
    if (!PyArg_ParseTuple(args,
                          "O!O!O!O!O!iiiddii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &p,
                          &num_tr,
                          &para_num_passes,
                          &para_r,
                          &para_g,
                          &para_step_len,
                          &verbose)) { return NULL; }
    if (verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d \n", num_tr, p);
        printf("para_r: %04e para_g: %04e num_passes: %d \n", para_r, para_g, para_num_passes);
        printf("--------------------------------------------------------------\n");
    }
    double *wt = malloc(sizeof(double) * p);
    double *wt_bar = malloc(sizeof(double) * p);
    _algo_fsauc_sparse((double *) PyArray_DATA(x_values),
                       (int *) PyArray_DATA(x_indices),
                       (int *) PyArray_DATA(x_positions),
                       (int *) PyArray_DATA(x_len_list),
                       (double *) PyArray_DATA(y_tr),
                       p,
                       num_tr,
                       para_r,
                       para_g,
                       para_num_passes,
                       wt, wt_bar);
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
        {"c_algo_fsauc_sparse",    (PyCFunction) wrap_algo_fsauc_sparse,    METH_VARARGS, "docs"},
        {"c_algo_opauc_sparse",    (PyCFunction) wrap_algo_opauc_sparse,    METH_VARARGS, "docs"},
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