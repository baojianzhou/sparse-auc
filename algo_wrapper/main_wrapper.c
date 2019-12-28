#include <Python.h>
#include <numpy/arrayobject.h>
#include "auc_opt_methods.h"

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

PyObject *get_results(int data_p, AlgoResults *re) {
    PyObject *results = PyTuple_New(4);
    PyObject *wt = PyList_New(data_p);
    PyObject *wt_bar = PyList_New(data_p);
    PyObject *auc = PyList_New(re->auc_len);
    PyObject *rts = PyList_New(re->auc_len);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(re->wt_bar[i]));
    }
    for (int i = 0; i < re->auc_len; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re->aucs[i]));
        PyList_SetItem(rts, i, PyFloat_FromDouble(re->rts[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, auc);
    PyTuple_SetItem(results, 3, rts);
    return results;
}

void init_data(Data *data, PyArrayObject *x_tr_vals, PyArrayObject *x_tr_inds, PyArrayObject *x_tr_poss,
               PyArrayObject *x_tr_lens, PyArrayObject *data_y_tr) {
    data->x_tr_vals = (double *) PyArray_DATA(x_tr_vals);
    data->x_tr_inds = (int *) PyArray_DATA(x_tr_inds);
    data->x_tr_poss = (int *) PyArray_DATA(x_tr_poss);
    data->x_tr_lens = (int *) PyArray_DATA(x_tr_lens);
    data->y_tr = (double *) PyArray_DATA(data_y_tr);
    data->n = (int) data_y_tr->dimensions[0];
}

void init_global_paras(GlobalParas *paras, PyArrayObject *global_paras) {
    double *arr_paras = (double *) PyArray_DATA(global_paras);
    //order should be: num_passes, step_len, verbose, record_aucs, stop_eps
    paras->num_passes = (int) arr_paras[0];
    paras->step_len = (int) arr_paras[1];
    paras->verbose = (int) arr_paras[2];
    paras->record_aucs = (int) arr_paras[3];
    paras->stop_eps = arr_paras[4];
}

static PyObject *wrap_algo_solam(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_r; //SOLAM has two parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!dd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras,
                          &para_xi, &para_r)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_solam(data, paras, re, para_xi, para_r);
    PyObject *results = get_results(data->p, re);
    free(paras), free_algo_results(re), free(data);
    return results;
}


static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_l1_reg, para_l2_reg; //SPAM has three parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!ddd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras,
                          &para_xi, &para_l1_reg, &para_l2_reg)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_spam(data, paras, re, para_xi, para_l1_reg, para_l2_reg);
    PyObject *results = get_results(data->p, re);
    free(paras), free_algo_results(re), free(data);
    return results;
}

static PyObject *wrap_algo_sht_am(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_l2_reg;
    int version, para_s, para_b;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!iiidd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras,
                          &version, &para_s, &para_b, &para_xi, &para_l2_reg)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n / para_b) * paras->num_passes + 1);
    _algo_sht_am(data, paras, re, version, 0, para_s, para_b, para_xi, para_l2_reg);
    PyObject *results = get_results(data->p, re);
    free(paras), free_algo_results(re), free(data);
    return results;
}

static PyObject *wrap_algo_graph_am(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr,
            *graph_edges, *graph_weights, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_l2_reg;
    int version, para_s, para_b;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!O!O!iiiidd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras, &PyArray_Type, &graph_edges, &PyArray_Type, &graph_weights,
                          &data->g, &version, &para_s, &para_b, &para_xi, &para_l2_reg)) { return NULL; }
    // graph info
    data->is_graph = true;
    data->m = (int) graph_weights->dimensions[0];
    data->edges = malloc(sizeof(EdgePair) * data->m);
    for (int i = 0; i < (int) graph_weights->dimensions[0]; i++) {
        data->edges[i].first = *(int *) PyArray_GETPTR2(graph_edges, i, 0);
        data->edges[i].second = *(int *) PyArray_GETPTR2(graph_edges, i, 1);
    }
    // ---
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    data->weights = (double *) PyArray_DATA(graph_weights);
    data->proj_prizes = malloc(sizeof(double) * data->p);   // projected prizes.
    data->graph_stat = make_graph_stat(data->p, data->m);   // head projection paras
    AlgoResults *re = make_algo_results(data->p, (data->n / para_b) * paras->num_passes + 1);
    _algo_sht_am(data, paras, re, version, 1, para_s, para_b, para_xi, para_l2_reg);
    PyObject *results = get_results(data->p, re);
    free_graph_stat(data->graph_stat);
    free(data->proj_prizes);
    free(data->edges);
    free(paras), free_algo_results(re), free(data);
    return results;
}

static PyObject *wrap_algo_opauc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    int para_tau;
    double para_eta, para_lambda;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiiiiddi",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &paras->num_passes, &paras->step_len, &paras->verbose, &paras->record_aucs,
                          &para_eta, &para_lambda, &para_tau)) { return NULL; }
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_opauc(data, paras, re, para_tau, para_eta, para_lambda);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    return results;
}

static PyObject *wrap_algo_fsauc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_r, para_g;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiiiidd", &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds,
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &paras->num_passes, &paras->step_len, &paras->verbose, &paras->record_aucs,
                          &para_r, &para_g)) { return NULL; }
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_fsauc(data, paras, re, para_r, para_g);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    return results;
}


static PyObject *wrap_algo_sto_iht(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    int para_s, para_b;
    double para_xi, para_l2_reg;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiiiiiidd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &paras->num_passes, &paras->step_len, &paras->verbose, &paras->record_aucs,
                          &para_s, &para_b, &para_xi, &para_l2_reg)) { return NULL; }
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_sto_iht(data, paras, re, para_s, para_b, para_xi, para_l2_reg);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    return results;
}

static PyObject *wrap_algo_hsg_ht(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    int para_s;
    double para_tau, para_zeta, para_c, para_l2;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiiiiidddd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &paras->num_passes, &paras->step_len, &paras->verbose, &paras->record_aucs,
                          &para_s, &para_tau, &para_zeta, &para_c, &para_l2)) { return NULL; }
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_hsg_ht(data, paras, re, para_s, para_tau, para_zeta, para_c, para_l2);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    return results;
}

// wrap_algo_solam_sparse
static PyMethodDef sparse_methods[] = { // hello_name
        {"c_test",          test,               METH_VARARGS, "docs"},
        {"c_algo_solam",    wrap_algo_solam,    METH_VARARGS, "docs"},
        {"c_algo_spam",     wrap_algo_spam,     METH_VARARGS, "docs"},
        {"c_algo_sht_am",   wrap_algo_sht_am,   METH_VARARGS, "docs"},
        {"c_algo_graph_am", wrap_algo_graph_am, METH_VARARGS, "docs"},
        {"c_algo_opauc",    wrap_algo_opauc,    METH_VARARGS, "docs"},
        {"c_algo_fsauc",    wrap_algo_fsauc,    METH_VARARGS, "docs"},
        {"c_algo_sto_iht",  wrap_algo_sto_iht,  METH_VARARGS, "docs"},
        {"c_algo_hsg_ht",   wrap_algo_hsg_ht,   METH_VARARGS, "docs"},
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