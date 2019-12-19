//
// Created by baojian on 12/19/19.
//

#include <Python.h>
#include <numpy/arrayobject.h>
#include "cuda_spam.cu"


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
                          &para_verbose)) {
        printf("test");
        return NULL;
    }
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


// wrap_algo_solam_sparse
static PyMethodDef sparse_methods[] = {
        {"c_algo_spam_sparse", (PyCFunction) wrap_algo_spam_sparse, METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};

// wrap_algo_solam_sparse
static PyMethodDef sparse_methods_3[] = { // hello_name
        {"c_algo_spam_sparse", wrap_algo_spam_sparse, METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sparse_module",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        sparse_methods_3,      /* m_methods */
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
