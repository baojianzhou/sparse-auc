#include <Python.h>
#include <numpy/arrayobject.h>
#include "auc_opt_methods.h"
#include "algo_hsg_ht.h"


static PyObject *hello(PyObject *self, PyObject *args) {
    printf("Hello, world!\n");
    Py_RETURN_NONE;
}

static PyObject *hello_name(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    const char *name;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    printf("Hello, %s!\n", name);
    Py_RETURN_NONE;
}

static PyObject *test(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    int verbose = 0;
    double sum = 0.0;
    PyArrayObject *x_tr_;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_tr_)) { return NULL; }
    int n = (int) (x_tr_->dimensions[0]);     // number of samples
    int p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = PyArray_DATA(x_tr_);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            if (verbose > 0) {
                printf("%.2f ", x_tr[i * p + j]);
            }

            sum += x_tr[i * p + j];
        }
        if (verbose > 0) {
            printf("\n");
        }
    }
    PyObject *results = PyFloat_FromDouble(sum);
    return results;
}

PyObject *get_results(int data_p, double *re_wt, double *re_wt_bar,
                      double *re_auc, double *re_rts, const int *re_len_auc) {
    PyObject *results = PyTuple_New(4);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(*re_len_auc);
    PyObject *rts = PyList_New(*re_len_auc);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re_wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re_wt_bar[i]));
    }
    for (int i = 0; i < *re_len_auc; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re_auc[i]));
        PyList_SetItem(rts, i, PyFloat_FromDouble(re_rts[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    PyTuple_SetItem(results, 3, rts);
    return results;
}


static PyObject *wrap_algo_solam(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr, *y_tr;
    double para_xi, para_r, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_n, data_p, para_num_passes, para_verbose, para_step_len, total_num_eval, re_len_auc;
    if (!PyArg_ParseTuple(args, "O!O!ddiii",
                          &PyArray_Type, &x_tr, &PyArray_Type, &y_tr,
                          &para_xi, &para_r, &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) x_tr->dimensions[0];
    data_p = (int) x_tr->dimensions[1];
    total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_solam((double *) PyArray_DATA(x_tr), (double *) PyArray_DATA(y_tr),
                data_n, data_p, para_xi, para_r, para_num_passes, para_step_len, para_verbose,
                re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc), free(re_rts);
    return results;
}


static PyObject *wrap_algo_solam_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_values, *x_tr_indices, *x_tr_posis, *x_tr_lens, *data_y_tr;
    int data_n, data_p, para_num_passes, para_verbose, para_step_len, total_num_eval, re_len_auc;
    double para_r, para_c, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iddiii",
                          &PyArray_Type, &x_tr_values, &PyArray_Type, &x_tr_indices,
                          &PyArray_Type, &x_tr_posis, &PyArray_Type, &x_tr_lens,
                          &PyArray_Type, &data_y_tr, &data_p, &para_c, &para_r, &para_num_passes,
                          &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_y_tr->dimensions[0];
    total_num_eval = (data_n * (para_num_passes + 1)) / para_step_len;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_solam_sparse((double *) PyArray_DATA(x_tr_values), (int *) PyArray_DATA(x_tr_indices),
                       (int *) PyArray_DATA(x_tr_posis), (int *) PyArray_DATA(x_tr_lens),
                       (double *) PyArray_DATA(data_y_tr), data_n, data_p, para_c, para_r,
                       para_num_passes, para_step_len, para_verbose, re_wt, re_wt_bar, re_auc,
                       re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt);
    return results;
}


static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *data_x_tr, *data_y_tr;
    int data_n, data_p, para_reg_opt, para_num_passes, para_step_len,
            para_verbose, total_num_eval, re_len_auc;
    double para_xi, para_l1_reg, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!dddiiii",
                          &PyArray_Type, &data_x_tr, &PyArray_Type, &data_y_tr,
                          &para_xi, &para_l1_reg, &para_l2_reg, &para_reg_opt, &para_num_passes,
                          &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_spam((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
               data_n, data_p, para_xi, para_l1_reg, para_l2_reg, para_num_passes, para_step_len,
               para_reg_opt, para_verbose, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc), free(re_rts);
    return results;
}


static PyObject *wrap_algo_spam_sparse(PyObject *self, PyObject *args) {
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr;
    int data_n, data_p, para_reg_opt, para_num_passes,
            para_step_len, para_verbose, total_num_eval, re_len_auc;
    double para_xi, para_l1_reg, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!idddiiii",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds,
                          &PyArray_Type, &x_tr_poss, &PyArray_Type, &x_tr_lens,
                          &PyArray_Type, &data_y_tr, &data_p, &para_xi, &para_l1_reg, &para_l2_reg,
                          &para_reg_opt, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    printf("%d\n", data_p);
    data_n = (int) data_y_tr->dimensions[0];
    total_num_eval = (data_n * (para_num_passes + 1)) / para_step_len;
    re_wt = calloc((size_t) data_p, sizeof(double));
    re_wt_bar = calloc((size_t) data_p, sizeof(double));
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_spam_sparse((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                      (int *) PyArray_DATA(x_tr_poss), (int *) PyArray_DATA(x_tr_lens),
                      (double *) PyArray_DATA(data_y_tr), data_n, data_p, para_xi, para_l1_reg,
                      para_l2_reg, para_num_passes, para_step_len, para_reg_opt, para_verbose,
                      re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc), free(re_rts);
    return results;
}


static PyObject *wrap_algo_sht_am(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; } // error: unknown error !!
    PyArrayObject *data_x_tr, *data_y_tr;
    double para_c, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_n, data_p, para_s, para_b, para_num_passes, para_step_len, para_verbose, re_len_auc;
    int total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!iiddiii",
                          &PyArray_Type, &data_x_tr,
                          &PyArray_Type, &data_y_tr,
                          &para_s, &para_b, &para_c, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n / para_b) * para_num_passes;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_sht_am((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                 data_n, data_p, para_s, para_b, para_c, para_l2_reg, para_num_passes,
                 para_step_len, para_verbose, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc);
    return results;
}

static PyObject *wrap_algo_sht_am_old(PyObject *self, PyObject *args) {
    if (self != NULL) { return NULL; } // error: unknown error !!
    PyArrayObject *data_x_tr, *data_y_tr;
    double para_c, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_n, data_p, para_s, para_b, para_num_passes,
            para_step_len, para_verbose, re_len_auc;
    int total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!iiddiii",
                          &PyArray_Type, &data_x_tr,
                          &PyArray_Type, &data_y_tr,
                          &para_s, &para_b, &para_c, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n / para_b) * para_num_passes;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_sht_am_old((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                     data_n, data_p, para_s, para_b, para_c, para_l2_reg, para_num_passes,
                     para_step_len, para_verbose, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc), free(re_rts);
    return results;
}


static PyObject *wrap_algo_sht_am_sparse_old(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_posis, *x_tr_lens, *data_y_tr;
    PyArrayObject *x_te_vals, *x_te_inds, *x_te_posis, *x_te_lens, *data_y_te;
    double para_xi, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_tr_n, data_te_n, data_p, para_b, para_sparsity, para_num_passes,
            para_step_len, para_verbose, re_len_auc;
    int total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!iiiddiii",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type,
                          &x_tr_posis, &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr,
                          &PyArray_Type, &x_te_vals, &PyArray_Type, &x_te_inds, &PyArray_Type,
                          &x_te_posis, &PyArray_Type, &x_te_lens, &PyArray_Type, &data_y_te,
                          &data_p, &para_sparsity, &para_b, &para_xi, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_tr_n = (int) data_y_tr->dimensions[0];
    data_te_n = (int) data_y_te->dimensions[0];
    total_num_eval = (data_tr_n / para_b) * (para_num_passes + 1);
    re_wt = calloc((size_t) data_p, sizeof(double));
    re_wt_bar = calloc((size_t) data_p, sizeof(double));
    re_auc = malloc(total_num_eval * sizeof(double));
    re_rts = malloc(total_num_eval * sizeof(double));
    _algo_sht_am_sparse_old((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                            (int *) PyArray_DATA(x_tr_posis), (int *) PyArray_DATA(x_tr_lens),
                            (double *) PyArray_DATA(data_y_tr), (double *) PyArray_DATA(x_te_vals),
                            (int *) PyArray_DATA(x_te_inds), (int *) PyArray_DATA(x_te_posis),
                            (int *) PyArray_DATA(x_te_lens), (double *) PyArray_DATA(data_y_te),
                            data_tr_n, data_te_n, data_p, para_sparsity, para_b, para_xi,
                            para_l2_reg, para_num_passes, para_step_len, para_verbose, re_wt,
                            re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc), free(re_rts);
    return results;
}


static PyObject *wrap_algo_sht_am_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_posis, *x_tr_lens, *data_y_tr;
    PyArrayObject *x_te_vals, *x_te_inds, *x_te_posis, *x_te_lens, *data_y_te;
    double para_xi, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_tr_n, data_te_n, data_p, para_b, para_sparsity, para_num_passes,
            para_step_len, para_verbose, re_len_auc;
    int total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!iiiddiii",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type,
                          &x_tr_posis, &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr,
                          &PyArray_Type, &x_te_vals, &PyArray_Type, &x_te_inds, &PyArray_Type,
                          &x_te_posis, &PyArray_Type, &x_te_lens, &PyArray_Type, &data_y_te,
                          &data_p, &para_sparsity, &para_b, &para_xi, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_tr_n = (int) data_y_tr->dimensions[0];
    data_te_n = (int) data_y_te->dimensions[0];
    total_num_eval = (data_tr_n / para_b) * (para_num_passes + 1);
    re_wt = calloc((size_t) data_p, sizeof(double));
    re_wt_bar = calloc((size_t) data_p, sizeof(double));
    re_auc = malloc(total_num_eval * sizeof(double));
    re_rts = malloc(total_num_eval * sizeof(double));
    _algo_sht_am_sparse((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                        (int *) PyArray_DATA(x_tr_posis), (int *) PyArray_DATA(x_tr_lens),
                        (double *) PyArray_DATA(data_y_tr), (double *) PyArray_DATA(x_te_vals),
                        (int *) PyArray_DATA(x_te_inds), (int *) PyArray_DATA(x_te_posis),
                        (int *) PyArray_DATA(x_te_lens), (double *) PyArray_DATA(data_y_te),
                        data_tr_n, data_te_n, data_p, para_sparsity, para_b, para_xi, para_l2_reg,
                        para_num_passes, para_step_len, para_verbose, re_wt,
                        re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc), free(re_rts);
    return results;
}


static PyObject *wrap_algo_graph_am(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *data_x_tr, *data_y_tr, *graph_edges, *graph_weights;
    double para_xi, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_n, data_p, para_step_len, para_sparsity, para_b, para_num_passes, para_verbose, re_len_auc;
    int total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!O!O!iiddiii",
                          &PyArray_Type, &data_x_tr, &PyArray_Type, &data_y_tr,
                          &PyArray_Type, &graph_edges, &PyArray_Type, &graph_weights,
                          &para_sparsity, &para_b, &para_xi, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    EdgePair *edges = malloc(sizeof(EdgePair) * (int) graph_weights->dimensions[0]);
    for (int i = 0; i < (int) graph_weights->dimensions[0]; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(graph_edges, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(graph_edges, i, 1);
    }
    total_num_eval = (data_n * para_num_passes) / para_step_len + 1;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_graph_am((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                   edges, (double *) PyArray_DATA(graph_weights),
                   (int) graph_weights->dimensions[0], data_n, data_p, para_sparsity, para_b,
                   para_xi, para_l2_reg, para_num_passes, para_step_len, para_verbose, re_wt,
                   re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(edges), free(re_auc), free(re_wt_bar), free(re_wt);
    return results;
}


static PyObject *wrap_algo_graph_am_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr;
    PyArrayObject *graph_edges, *graph_weights;
    double para_xi, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_p, para_b, data_n, para_sparsity, para_num_passes,
            para_step_len, para_verbose, re_len_auc;
    int total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iiiddiii",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds,
                          &PyArray_Type, &x_tr_poss, &PyArray_Type, &x_tr_lens,
                          &PyArray_Type, &data_y_tr, &PyArray_Type, &graph_edges,
                          &PyArray_Type, &graph_weights, &data_p, &para_sparsity, &para_b,
                          &para_xi, &para_l2_reg, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    data_n = (int) data_y_tr->dimensions[0];
    EdgePair *edges = malloc(sizeof(EdgePair) * (int) graph_weights->dimensions[0]);
    for (int i = 0; i < (int) graph_weights->dimensions[0]; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(graph_edges, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(graph_edges, i, 1);
    }
    total_num_eval = (data_n * (para_num_passes + 1)) / para_step_len;
    re_wt = calloc((size_t) data_p, sizeof(double));
    re_wt_bar = calloc((size_t) data_p, sizeof(double));
    re_auc = malloc(total_num_eval * sizeof(double));
    re_rts = malloc(total_num_eval * sizeof(double));
    _algo_graph_am_sparse((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                          (int *) PyArray_DATA(x_tr_poss), (int *) PyArray_DATA(x_tr_lens),
                          (double *) PyArray_DATA(data_y_tr), edges,
                          (double *) PyArray_DATA(graph_weights),
                          (int) graph_weights->dimensions[0], data_n, data_p, para_sparsity,
                          para_b, para_xi, para_l2_reg, para_num_passes, para_step_len,
                          para_verbose, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(edges), free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}


static PyObject *wrap_algo_opauc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *data_x_tr, *data_y_tr;
    int data_p, data_n, para_num_passes, para_step_len, para_verbose, total_num_eval, re_len_auc;
    double para_eta, para_lambda, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!ddiii", &PyArray_Type, &data_x_tr, &PyArray_Type,
                          &data_y_tr, &para_eta, &para_lambda, &para_num_passes,
                          &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n * para_num_passes) / para_step_len;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    bool para_is_aucs = true;
    if (total_num_eval == 0) {
        para_is_aucs = false;
    }
    _algo_opauc((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                data_n, data_p, para_eta, para_lambda, para_num_passes, para_step_len, para_is_aucs,
                para_verbose, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}


static PyObject *wrap_algo_opauc_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_posis, *x_tr_lens, *data_y_tr;
    int data_n, data_p, para_tau, para_num_passes, para_step_len,
            para_verbose, total_num_eval, re_len_auc;
    double para_eta, para_lambda, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiddiii",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds,
                          &PyArray_Type, &x_tr_posis, &PyArray_Type, &x_tr_lens,
                          &PyArray_Type, &data_y_tr, &data_p, &para_tau, &para_eta, &para_lambda,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }

    data_n = (int) data_y_tr->dimensions[0];
    total_num_eval = data_n;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_opauc_sparse((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                       (int *) PyArray_DATA(x_tr_posis), (int *) PyArray_DATA(x_tr_lens),
                       (double *) PyArray_DATA(data_y_tr), data_n, data_p, para_tau, para_eta,
                       para_lambda, para_num_passes, para_step_len, para_verbose,
                       re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}

static PyObject *wrap_algo_fsauc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *data_x_tr, *data_y_tr;
    int data_n, data_p, para_num_passes, para_step_len, para_verbose, total_num_eval, re_len_auc;
    double para_r, para_g, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!ddiii", &PyArray_Type, &data_x_tr, &PyArray_Type, &data_y_tr,
                          &para_r, &para_g, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n * para_num_passes) / para_step_len;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_fsauc((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                data_n, data_p, para_r, para_g, para_num_passes, para_step_len, para_verbose,
                re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}

static PyObject *wrap_algo_fsauc_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_posis, *x_tr_lens, *data_y_tr;
    int data_n, data_p, para_num_passes, para_step_len, para_verbose, total_num_eval, re_len_auc;
    double para_r, para_g, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iddiii", &PyArray_Type, &x_tr_vals,
                          &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_posis,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr,
                          &data_p, &para_r, &para_g, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    data_n = (int) data_y_tr->dimensions[0];
    total_num_eval = (data_n * (para_num_passes + 1)) / para_step_len;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_fsauc_sparse((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                       (int *) PyArray_DATA(x_tr_posis), (int *) PyArray_DATA(x_tr_lens),
                       (double *) PyArray_DATA(data_y_tr), data_n, data_p, para_r, para_g,
                       para_num_passes, para_step_len, para_verbose, re_wt, re_wt_bar, re_auc,
                       re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}

static PyObject *wrap_algo_hsg_ht(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *data_x_tr, *data_y_tr;
    int data_n, data_p, para_num_passes, para_step_len, para_verbose, total_num_eval, re_len_auc;
    double para_r, para_g, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!ddiii", &PyArray_Type, &data_x_tr, &PyArray_Type, &data_y_tr,
                          &para_r, &para_g, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n * para_num_passes) / para_step_len;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_fsauc((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                data_n, data_p, para_r, para_g, para_num_passes, para_step_len, para_verbose,
                re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}


static PyObject *wrap_algo_sto_iht(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *data_x_tr, *data_y_tr;
    double para_c, para_l2_reg, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    int data_n, data_p, para_s, para_b, para_num_passes, para_step_len, para_verbose, re_len_auc, total_num_eval;
    if (!PyArg_ParseTuple(args, "O!O!iiddiii",
                          &PyArray_Type, &data_x_tr,
                          &PyArray_Type, &data_y_tr,
                          &para_s, &para_b, &para_c, &para_l2_reg,
                          &para_num_passes, &para_step_len, &para_verbose)) { return NULL; }
    data_n = (int) data_x_tr->dimensions[0];
    data_p = (int) data_x_tr->dimensions[1];
    total_num_eval = (data_n / para_b) * para_num_passes;
    re_wt = malloc(sizeof(double) * (data_p + 1)); // with one additional intercept.
    re_wt_bar = malloc(sizeof(double) * (data_p + 1)); // with one additional intercept.
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_sto_iht((double *) PyArray_DATA(data_x_tr), (double *) PyArray_DATA(data_y_tr),
                  data_n, data_p, para_s, para_b, para_c, para_l2_reg, para_num_passes,
                  para_step_len, para_verbose, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_wt), free(re_wt_bar), free(re_auc);
    return results;
}

static PyObject *wrap_algo_hsg_ht_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_posis, *x_tr_lens, *data_y_tr;
    int data_n, data_p, para_num_passes, para_step_len, para_verbose, total_num_eval, re_len_auc;
    double para_r, para_g, *re_wt, *re_wt_bar, *re_auc, *re_rts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iddiii", &PyArray_Type, &x_tr_vals,
                          &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_posis,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr,
                          &data_p, &para_r, &para_g, &para_num_passes, &para_step_len,
                          &para_verbose)) { return NULL; }
    data_n = (int) data_y_tr->dimensions[0];
    total_num_eval = (data_n * (para_num_passes + 1)) / para_step_len;
    re_wt = malloc(sizeof(double) * data_p);
    re_wt_bar = malloc(sizeof(double) * data_p);
    re_auc = malloc(sizeof(double) * total_num_eval);
    re_rts = malloc(sizeof(double) * total_num_eval);
    _algo_fsauc_sparse((double *) PyArray_DATA(x_tr_vals), (int *) PyArray_DATA(x_tr_inds),
                       (int *) PyArray_DATA(x_tr_posis), (int *) PyArray_DATA(x_tr_lens),
                       (double *) PyArray_DATA(data_y_tr), data_n, data_p, para_r, para_g,
                       para_num_passes, para_step_len, para_verbose, re_wt, re_wt_bar, re_auc,
                       re_rts, &re_len_auc);
    PyObject *results = get_results(data_p, re_wt, re_wt_bar, re_auc, re_rts, &re_len_auc);
    free(re_auc), free(re_wt_bar), free(re_wt), free(re_rts);
    return results;
}

// wrap_algo_solam_sparse
static PyMethodDef sparse_methods[] = { // hello_name
        {"c_test",                   test,                        METH_VARARGS, "docs"},
        {"c_hello",                  hello,                       METH_VARARGS, "docs"},
        {"c_hello_name",             hello_name,                  METH_VARARGS, "docs"},
        {"c_algo_spam",              wrap_algo_spam,              METH_VARARGS, "docs"},
        {"c_algo_solam",             wrap_algo_solam,             METH_VARARGS, "docs"},
        {"c_algo_sht_am",            wrap_algo_sht_am,            METH_VARARGS, "docs"},
        {"c_algo_sht_am_old",        wrap_algo_sht_am_old,        METH_VARARGS, "docs"},
        {"c_algo_graph_am",          wrap_algo_graph_am,          METH_VARARGS, "docs"},
        {"c_algo_opauc",             wrap_algo_opauc,             METH_VARARGS, "docs"},
        {"c_algo_fsauc",             wrap_algo_fsauc,             METH_VARARGS, "docs"},
        {"c_algo_hsg_ht",            wrap_algo_hsg_ht,            METH_VARARGS, "docs"},
        {"c_algo_sto_iht",           wrap_algo_sto_iht,           METH_VARARGS, "docs"},

        {"c_algo_hsg_ht_sparse",     wrap_algo_hsg_ht_sparse,     METH_VARARGS, "docs"},
        {"c_algo_solam_sparse",      wrap_algo_solam_sparse,      METH_VARARGS, "docs"},
        {"c_algo_sht_am_sparse",     wrap_algo_sht_am_sparse,     METH_VARARGS, "docs"},
        {"c_algo_sht_am_sparse_old", wrap_algo_sht_am_sparse_old, METH_VARARGS, "docs"},
        {"c_algo_spam_sparse",       wrap_algo_spam_sparse,       METH_VARARGS, "docs"},
        {"c_algo_fsauc_sparse",      wrap_algo_fsauc_sparse,      METH_VARARGS, "docs"},
        {"c_algo_opauc_sparse",      wrap_algo_opauc_sparse,      METH_VARARGS, "docs"},
        {"c_algo_graph_am_sparse",   wrap_algo_graph_am_sparse,   METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sparse_module",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        sparse_methods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif


/** Python version 2 for module initialization */
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_sparse_module(void){
     Py_Initialize();
     import_array(); // In order to use numpy, you must include this!
    return PyModule_Create(&moduledef);
}
#else
initsparse_module(void) {
    Py_InitModule3("sparse_module", sparse_methods, "some docs for solam algorithm.");
    import_array(); // In order to use numpy, you must include this!
}

#endif

int main() {
    printf("test of main wrapper!\n");
}