#include <Python.h>
#include <numpy/arrayobject.h>
#include "algo_solam.h"
#include "algo_da_solam.h"
#include "algo_sparse_solam.h"


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
    /*
     * Wrapper of the SOLAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    solam_para *para = malloc(sizeof(solam_para));
    PyArrayObject *x_tr_, *y_tr_, *rand_ind_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &rand_ind_,
                          &para->para_r,
                          &para->para_xi,
                          &para->para_num_pass,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->para_rand_ind = (int *) PyArray_DATA(rand_ind_);
    solam_results *result = malloc(sizeof(solam_results));
    result->wt = malloc(sizeof(double) * para->p);
    result->a = 0.0;
    result->b = 0.0;
    //call SOLAM algorithm
    algo_solam_func(para, result);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(para->p);
    PyObject *a = PyFloat_FromDouble(result->a);
    PyObject *b = PyFloat_FromDouble(result->b);
    for (int i = 0; i < para->p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, a);
    PyTuple_SetItem(results, 2, b);
    free(para);
    free(result->wt);
    free(result);
    return results;
}


static PyObject *wrap_algo_sparse_solam(PyObject *self, PyObject *args) {
    /*
     * Wrapper of the SOLAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    sparse_solam_para *para = malloc(sizeof(sparse_solam_para));
    PyArrayObject *x_tr_, *y_tr_, *rand_ind_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &rand_ind_,
                          &para->para_r,
                          &para->para_xi,
                          &para->para_s,
                          &para->para_num_pass,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->para_rand_ind = (int *) PyArray_DATA(rand_ind_);
    sparse_solam_results *result = malloc(sizeof(sparse_solam_results));
    result->wt = malloc(sizeof(double) * para->p);
    result->a = 0.0;
    result->b = 0.0;
    //call SOLAM algorithm
    algo_sparse_solam_func(para, result);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(para->p);
    PyObject *a = PyFloat_FromDouble(result->a);
    PyObject *b = PyFloat_FromDouble(result->b);
    for (int i = 0; i < para->p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, a);
    PyTuple_SetItem(results, 2, b);
    free(para);
    free(result->wt);
    free(result);
    return results;
}

static PyObject *wrap_algo_da_solam(PyObject *self, PyObject *args) {
    /*
     * Wrapper of the SOLAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    da_solam_para *para = malloc(sizeof(da_solam_para));
    PyArrayObject *x_tr_, *y_tr_, *rand_ind_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &rand_ind_,
                          &para->para_r,
                          &para->para_xi,
                          &para->para_s,
                          &para->para_num_pass,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->para_rand_ind = (int *) PyArray_DATA(rand_ind_);
    da_solam_results *result = malloc(sizeof(da_solam_results));
    result->wt = malloc(sizeof(double) * para->p);
    result->a = 0.0;
    result->b = 0.0;
    //call SOLAM algorithm
    algo_da_solam_func(para, result);
    PyObject *results = PyTuple_New(3);
    PyObject *wt = PyList_New(para->p);
    PyObject *a = PyFloat_FromDouble(result->a);
    PyObject *b = PyFloat_FromDouble(result->b);
    for (int i = 0; i < para->p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, a);
    PyTuple_SetItem(results, 2, b);
    free(para);
    free(result->wt);
    free(result);
    return results;
}


static PyMethodDef sparse_methods[] = {
        {"c_test",          (PyCFunction) test,               METH_VARARGS, "test docs"},
        {"c_algo_solam",    (PyCFunction) wrap_algo_solam,    METH_VARARGS, "wrap_algo_solam docs"},
        {"c_algo_sparse_solam",    (PyCFunction) wrap_algo_sparse_solam,
         METH_VARARGS, "wrap_algo_sparse_solam docs"},
        {"c_algo_da_solam",    (PyCFunction) wrap_algo_da_solam,
                METH_VARARGS, "wrap_algo_da_solam docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods, "some docs for solam algorithm.");
    import_array();
}

int main() {
    printf("test of main wrapper!\n");
}

int main_2(PyObject *args){
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
    return 0;
}