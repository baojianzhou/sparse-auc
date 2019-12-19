//
// Created by baojian on 12/19/19.
//

#include <Python.h>
#include <numpy/arrayobject.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"

#define sign(x) (x > 0) - (x < 0)

typedef struct {
    double val;
    int index;
} data_pair;

static inline int __comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

void _arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

double _auc_score(const double *true_labels, const double *scores, int len) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    double *fpr = malloc(sizeof(double) * (len + 1));
    double *tpr = malloc(sizeof(double) * (len + 1));
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < len; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * len);
    _arg_sort_descend(scores, sorted_indices, len);
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < len; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    // cblas_dscal(len, 1. / num_posi, tpr, 1);
    // cblas_dscal(len, 1. / num_nega, fpr, 1);
    double alpha = 1. / num_posi;
    cudaMallocManaged(&tpr, len * sizeof(double));
    cublasDscal(handle, len, &alpha, tpr, 1);
    alpha = 1. / num_nega;
    cudaMallocManaged(&fpr, len * sizeof(double));
    cublasDscal(handle, len, &alpha, fpr, 1);
    cudaDeviceSynchronize();
    //AUC score
    double auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < len; i++) {
        auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
    }
    free(sorted_indices);
    free(fpr);
    free(tpr);
    cublasDestroy(handle);
    return auc;
}

void _algo_spam_sparse(
        const double *x_tr_vals, const int *x_tr_inds, const int *x_tr_poss, const int *x_tr_lens,
        const double *data_y_tr, int data_n, int data_p, double para_c, double para_l1_reg,
        double para_l2_reg, int para_num_passes, int para_step_len, int para_reg_opt,
        int para_verbose, double *re_wt, double *re_wt_bar, double *re_auc, double *re_rts,
        int *re_len_auc) {

    double start_time = clock();
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double a_wt, *posi_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=1]
    double b_wt, *nega_x_mean = calloc((size_t) data_p, sizeof(double)); // w^T*E[x|y=-1]
    double alpha_wt, posi_t = 0.0, nega_t = 0.0;
    double *y_pred = calloc((size_t) data_n, sizeof(double));
    for (int i = 0; i < data_n; i++) {
        const int *xt_inds = x_tr_inds + x_tr_poss[i];
        const double *xt_vals = x_tr_vals + x_tr_poss[i];
        if (data_y_tr[i] > 0) {
            posi_t++;
            for (int kk = 0; kk < x_tr_lens[i]; kk++)
                posi_x_mean[xt_inds[kk]] += xt_vals[kk];
        } else {
            nega_t++;
            for (int kk = 0; kk < x_tr_lens[i]; kk++)
                nega_x_mean[xt_inds[kk]] += xt_vals[kk];
        }
    }
    double tmp1 = 1. / posi_t, tmp2 = 1. / nega_t;
    //cblas_dscal(data_p, 1. / posi_t, posi_x_mean, 1);
    //cblas_dscal(data_p, 1. / nega_t, nega_x_mean, 1);
    cublasHandle_t handle;
    cudaMallocManaged(&posi_x_mean, data_p * sizeof(double));
    cudaMallocManaged(&posi_x_mean, data_p * sizeof(double));
    cublasCreate(&handle);
    cublasDscal(handle, data_p, &tmp1, posi_x_mean, 1);
    cublasDscal(handle, data_p, &tmp2, nega_x_mean, 1);
    cudaDeviceSynchronize();
    double prob_p = posi_t / (data_n * 1.0), eta_t, t_eval;
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    *re_len_auc = 0;
    for (int t = 1; t <= (para_num_passes * data_n); t++) {
        const int *xt_inds = x_tr_inds + x_tr_poss[(t - 1) % data_n]; // receive zt=(xt,yt)
        const double *xt_vals = x_tr_vals + x_tr_poss[(t - 1) % data_n];
        eta_t = para_c / sqrt(t); // current learning rate
        //a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
        //b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // para_b(wt)
        cudaMallocManaged(&re_wt, data_p * sizeof(double));
        cublasDdot(handle, data_p, re_wt, 1, posi_x_mean, 1, &a_wt);
        cublasDdot(handle, data_p, re_wt, 1, nega_x_mean, 1, &b_wt);
        alpha_wt = b_wt - a_wt; // alpha(wt)
        double wt_dot = 0.0;
        for (int tt = 0; tt < x_tr_lens[(t - 1) % data_n]; tt++)
            wt_dot += (re_wt[xt_inds[tt]] * xt_vals[tt]);
        double weight = data_y_tr[(t - 1) % data_n] > 0 ?
                        2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                        2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                        2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
        for (int tt = 0; tt < x_tr_lens[(t - 1) % data_n]; tt++) // gradient descent
            re_wt[xt_inds[tt]] += -eta_t * weight * xt_vals[tt];
        if (para_reg_opt == 0) { // elastic-net
            double tmp_demon = (eta_t * para_l2_reg + 1.);
            for (int k = 0; k < data_p; k++) {
                double tmp_sign = (double) sign(re_wt[k]) / tmp_demon;
                re_wt[k] = tmp_sign * fmax(0.0, fabs(re_wt[k]) - eta_t * para_l1_reg);
            }
        } else { // l2-regularization
            //cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), re_wt, 1);
            double tmp3 = 1. / (eta_t * para_l2_reg + 1.);
            cublasDscal(handle, data_p, &tmp3, re_wt, 1);
        }
        double tmp4 = 1.0;
        //cblas_daxpy(data_p, 1., re_wt, 1, re_wt_bar, 1);
        cublasDaxpy(handle, data_p, &tmp4, re_wt, 1, re_wt_bar, 1);
        if ((fmod(t, para_step_len) == 1.)) { // evaluate the AUC score
            t_eval = clock();
            for (int q = 0; q < data_n; q++) {
                xt_inds = x_tr_inds + x_tr_poss[q];
                xt_vals = x_tr_vals + x_tr_poss[q];
                y_pred[q] = 0.0;
                for (int tt = 0; tt < x_tr_lens[q]; tt++)
                    y_pred[q] += re_wt[xt_inds[tt]] * xt_vals[tt];
            }
            re_auc[*re_len_auc] = _auc_score(data_y_tr, y_pred, data_n);
            re_rts[(*re_len_auc)++] = clock() - start_time - (clock() - t_eval);
        }
    }
    double tmp5 =  1. / (para_num_passes * data_n);
    // cblas_dscal(data_p, 1. / (para_num_passes * data_n), re_wt_bar, 1);
    cublasDscal(handle, data_p, &tmp5, re_wt_bar, 1);
    double tmp6 = 1. / CLOCKS_PER_SEC;
    // cblas_dscal(*(re_len_auc), 1. / CLOCKS_PER_SEC, re_rts, 1);
    cublasDscal(handle, *(re_len_auc), &tmp6, re_rts, 1);
    free(y_pred);
    free(nega_x_mean);
    free(posi_x_mean);
    free(grad_wt);
    cublasDestroy(handle);
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
