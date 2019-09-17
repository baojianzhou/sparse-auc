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
    /*
     * Wrapper of the SOLAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    solam_para *para = malloc(sizeof(solam_para));
    PyArrayObject *x_tr_, *y_tr_;
    if (!PyArg_ParseTuple(args, "O!O!ddii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &para->para_r,
                          &para->para_xi,
                          &para->para_num_pass,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    solam_results *result = malloc(sizeof(solam_results));
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->a = 0.0;
    result->b = 0.0;
    if(para->verbose >0){
        printf("num_tr: %d p: %d xi: %.4f r: %.4f num_passes: %d\n",
               para->num_tr, para->p, para->para_xi, para->para_r, para->para_num_pass);
    }
    __solam(para, result);
    PyObject *results = PyTuple_New(4);
    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *a = PyFloat_FromDouble(result->a);
    PyObject *b = PyFloat_FromDouble(result->b);
    for (int i = 0; i < para->p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(result->wt[i]));
        PyList_SetItem(wt_bar, i, PyFloat_FromDouble(result->wt[i]));
    }
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, wt_bar);
    PyTuple_SetItem(results, 2, a);
    PyTuple_SetItem(results, 3, b);
    free(para);
    free(result->wt_bar);
    free(result->wt);
    free(result);
    return results;
}


static PyObject *wrap_algo_solam_sparse(PyObject *self, PyObject *args) {
    /*
     * Wrapper of the SOLAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    solam_para_sparse *para = malloc(sizeof(solam_para_sparse));
    PyArrayObject *x_tr_indices, *x_tr_values, *y_tr_, *rand_ind_;
    if (!PyArg_ParseTuple(args, "O!O!O!O!iddii",
                          &PyArray_Type, &x_tr_indices,
                          &PyArray_Type, &x_tr_values,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &rand_ind_,
                          &para->p,
                          &para->para_r,
                          &para->para_xi,
                          &para->para_num_pass,
                          &para->verbose)) { return NULL; }
    para->num_tr = (int) x_tr_indices->dimensions[0];
    para->max_nonzero = (int) x_tr_indices->dimensions[1];
    para->x_tr_indices = (int *) PyArray_DATA(x_tr_indices);
    para->x_tr_values = (double *) PyArray_DATA(x_tr_values);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->para_rand_ind = (int *) PyArray_DATA(rand_ind_);
    //printf("num_tr: %d max_nonzero: %d p: %d\n", para->num_tr, para->max_nonzero, para->p);
    //printf("len: %d 5th: %d\n", para->x_tr_indices[0], para->x_tr_indices[5]);
    solam_results *result = malloc(sizeof(solam_results));
    result->wt = malloc(sizeof(double) * para->p);
    result->a = 0.0;
    result->b = 0.0;
    //call SOLAM algorithm
    __solam_sparse(para, result);
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


static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    /**
     * Wrapper of the SPAM algorithm
     */
    if (self != NULL) {
        printf("error: unknown error !!\n");
        return NULL;
    }
    spam_para *para = malloc(sizeof(spam_para));
    PyArrayObject *x_tr_, *y_tr_;
    if (!PyArg_ParseTuple(args, "O!O!dddiiiii",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &para->para_xi,
                          &para->para_l1_reg,
                          &para->para_l2_reg,
                          &para->para_reg_opt,
                          &para->para_num_passes,
                          &para->para_step_len,
                          &para->is_sparse,
                          &para->verbose)) { return NULL; }

    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    spam_results *result = malloc(sizeof(spam_results));

    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;

    // summary of the data
    if (para->verbose > 0) {
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d x_tr[0]: %.4f y_tr[0]:%.4f\n",
               para->num_tr, para->p, para->x_tr[0], para->y_tr[0]);
        printf("para_xi: %04e para_l1_reg: %04e para_l2_reg: %04e\n",
               para->para_xi, para->para_l1_reg, para->para_l2_reg);
        printf("reg_option: %d num_passes: %d step_len: %d is_sparse: %d \n",
               para->para_reg_opt, para->para_num_passes, para->para_step_len, para->is_sparse);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }


    //call SOLAM algorithm
    algo_spam(para, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < para->p; i++) {
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
    free(para);
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
    spam_para *para = malloc(sizeof(spam_para));
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iidddiiiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &para->num_tr,
                          &para->p,
                          &para->para_xi,
                          &para->para_l1_reg,
                          &para->para_l2_reg,
                          &para->para_reg_opt,
                          &para->para_num_passes,
                          &para->para_step_len,
                          &para->is_sparse,
                          &para->verbose)) { return NULL; }

    para->sparse_x_values = (double *) PyArray_DATA(x_values);
    para->sparse_x_indices = (int *) PyArray_DATA(x_indices);
    para->sparse_x_positions = (int *) PyArray_DATA(x_positions);
    para->sparse_x_len_list = (int *) PyArray_DATA(x_len_list);
    para->y_tr = (double *) PyArray_DATA(y_tr);
    spam_results *result = malloc(sizeof(spam_results));

    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;

    // summary of the data
    printf("--------------------------------------------------------------\n");
    printf("num_tr: %d p: %d x_tr[0]: %.4f y_tr[0]:%.4f\n",
           para->num_tr, para->p, para->x_tr[0], para->y_tr[0]);
    printf("para_xi: %04e para_l1_reg: %04e para_l2_reg: %04e\n",
           para->para_xi, para->para_l1_reg, para->para_l2_reg);
    printf("reg_option: %d num_passes: %d step_len: %d is_sparse: %d \n",
           para->para_reg_opt, para->para_num_passes, para->para_step_len, para->is_sparse);
    printf("num_eval: %d\n", total_num_eval);
    printf("--------------------------------------------------------------\n");

    //call SOLAM algorithm
    algo_spam(para, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < para->p; i++) {
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
    free(para);
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
    sht_am_para *para = malloc(sizeof(sht_am_para));
    PyArrayObject *x_tr_, *y_tr_, *sub_nodes_;
    if (!PyArg_ParseTuple(args, "O!O!iiddiiiiO!",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &para->para_sparsity,
                          &para->para_b,
                          &para->para_xi,
                          &para->para_l2_reg,
                          &para->para_num_passes,
                          &para->para_step_len,
                          &para->is_sparse,
                          &para->verbose,
                          &PyArray_Type, &sub_nodes_)) { return NULL; }

    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->sub_nodes = (int *) PyArray_DATA(sub_nodes_);
    para->nodes_len = (int) sub_nodes_->dimensions[0];
    sht_am_results *result = malloc(sizeof(sht_am_results));

    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;
    if (para->verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d block_size: %d x_tr[0]: %.4f y_tr[0]:%.4f\n",
               para->num_tr, para->p, para->para_b, para->x_tr[0], para->y_tr[0]);
        printf("para_xi: %04e para_l2_reg: %04e\n", para->para_xi, para->para_l2_reg);
        printf("num_passes: %d step_len: %d is_sparse: %d \n",
               para->para_num_passes, para->para_step_len, para->is_sparse);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }

    //call SOLAM algorithm
    algo_sht_am(para, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < para->p; i++) {
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
    free(para);
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
    graph_am_para *para = malloc(sizeof(graph_am_para));
    PyArrayObject *x_tr_, *y_tr_, *sub_nodes_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!iiddiiiiO!O!O!",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &para->para_sparsity,
                          &para->para_b,
                          &para->para_xi,
                          &para->para_l2_reg,
                          &para->para_num_passes,
                          &para->para_step_len,
                          &para->is_sparse,
                          &para->verbose,
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &weights_,
                          &PyArray_Type, &sub_nodes_)) { return NULL; }

    para->num_tr = (int) x_tr_->dimensions[0];
    para->p = (int) x_tr_->dimensions[1];
    para->x_tr = (double *) PyArray_DATA(x_tr_);
    para->y_tr = (double *) PyArray_DATA(y_tr_);
    para->sub_nodes = (int *) PyArray_DATA(sub_nodes_);
    para->m = (int) edges_->dimensions[0];
    para->edges = malloc(sizeof(EdgePair) * para->m);
    para->weights = (double *) PyArray_DATA(weights_);
    para->nodes_len = (int) sub_nodes_->dimensions[0];
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }

    graph_am_results *result = malloc(sizeof(graph_am_results));

    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 1;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);
    result->t_index = 0;
    if (para->verbose > 0) {
        // summary of the data
        printf("--------------------------------------------------------------\n");
        printf("num_tr: %d p: %d block_size: %d x_tr[0]: %.4f y_tr[0]:%.4f\n",
               para->num_tr, para->p, para->para_b, para->x_tr[0], para->y_tr[0]);
        printf("para_xi: %04e para_l2_reg: %04e\n", para->para_xi, para->para_l2_reg);
        printf("num_passes: %d step_len: %d is_sparse: %d \n",
               para->para_num_passes, para->para_step_len, para->is_sparse);
        printf("num_eval: %d\n", total_num_eval);
        printf("--------------------------------------------------------------\n");
    }

    //call SOLAM algorithm
    algo_graph_am(para, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < para->p; i++) {
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

    free(para->edges);
    free(para);
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
    sht_am_para *para = malloc(sizeof(sht_am_para));
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiddiiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &para->num_tr,
                          &para->p,
                          &para->para_sparsity,
                          &para->para_xi,
                          &para->para_l2_reg,
                          &para->para_num_passes,
                          &para->para_step_len,
                          &para->is_sparse,
                          &para->verbose)) { return NULL; }

    para->sparse_x_values = (double *) PyArray_DATA(x_values);
    para->sparse_x_indices = (int *) PyArray_DATA(x_indices);
    para->sparse_x_positions = (int *) PyArray_DATA(x_positions);
    para->sparse_x_len_list = (int *) PyArray_DATA(x_len_list);
    para->y_tr = (double *) PyArray_DATA(y_tr);
    sht_am_results *result = malloc(sizeof(sht_am_results));

    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 1;
    result->t_index = 0;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);


    // summary of the data
    printf("--------------------------------------------------------------\n");
    printf("num_tr: %d p: %d x_tr[0]: %.4f y_tr[0]:%.4f\n",
           para->num_tr, para->p, para->x_tr[0], para->y_tr[0]);
    printf("para_xi: %04e para_l2_reg: %04e\n", para->para_xi, para->para_l2_reg);
    printf("num_passes: %d step_len: %d is_sparse: %d \n",
           para->para_num_passes, para->para_step_len, para->is_sparse);
    printf("num_eval: %d\n", total_num_eval);
    printf("--------------------------------------------------------------\n");

    //call SOLAM algorithm
    algo_sht_am(para, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < para->p; i++) {
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
    free(para);
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
    sht_am_para *para = malloc(sizeof(sht_am_para));
    PyArrayObject *x_values, *x_indices, *x_positions, *x_len_list, *y_tr;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiddiiii",
                          &PyArray_Type, &x_values,
                          &PyArray_Type, &x_indices,
                          &PyArray_Type, &x_positions,
                          &PyArray_Type, &x_len_list,
                          &PyArray_Type, &y_tr,
                          &para->num_tr,
                          &para->p,
                          &para->para_sparsity,
                          &para->para_xi,
                          &para->para_l2_reg,
                          &para->para_num_passes,
                          &para->para_step_len,
                          &para->is_sparse,
                          &para->verbose)) { return NULL; }

    para->sparse_x_values = (double *) PyArray_DATA(x_values);
    para->sparse_x_indices = (int *) PyArray_DATA(x_indices);
    para->sparse_x_positions = (int *) PyArray_DATA(x_positions);
    para->sparse_x_len_list = (int *) PyArray_DATA(x_len_list);
    para->y_tr = (double *) PyArray_DATA(y_tr);
    sht_am_results *result = malloc(sizeof(sht_am_results));

    int total_num_eval = (para->num_tr * para->para_num_passes) / para->para_step_len + 1;
    result->t_index = 0;
    result->t_eval_time = 0.0;
    result->wt = malloc(sizeof(double) * para->p);
    result->wt_bar = malloc(sizeof(double) * para->p);
    result->t_run_time = malloc(sizeof(double) * total_num_eval);
    result->t_auc = malloc(sizeof(double) * total_num_eval);
    result->t_indices = malloc(sizeof(int) * total_num_eval);


    // summary of the data
    printf("--------------------------------------------------------------\n");
    printf("num_tr: %d p: %d x_tr[0]: %.4f y_tr[0]:%.4f\n",
           para->num_tr, para->p, para->x_tr[0], para->y_tr[0]);
    printf("para_xi: %04e para_l2_reg: %04e\n", para->para_xi, para->para_l2_reg);
    printf("num_passes: %d step_len: %d is_sparse: %d \n",
           para->para_num_passes, para->para_step_len, para->is_sparse);
    printf("num_eval: %d\n", total_num_eval);
    printf("--------------------------------------------------------------\n");

    //call SOLAM algorithm
    algo_sht_am(para, result);
    PyObject *results = PyTuple_New(5);

    PyObject *wt = PyList_New(para->p);
    PyObject *wt_bar = PyList_New(para->p);
    PyObject *t_run_time = PyList_New(result->t_index);
    PyObject *t_auc = PyList_New(result->t_index);

    for (int i = 0; i < para->p; i++) {
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
    free(para);
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


// wrap_algo_solam_sparse
static PyMethodDef sparse_methods[] = {
        {"c_test",                 (PyCFunction) test,                      METH_VARARGS, "docs"},
        {"c_algo_solam",           (PyCFunction) wrap_algo_solam,           METH_VARARGS, "docs"},
        {"c_algo_solam_sparse",    (PyCFunction) wrap_algo_solam_sparse,    METH_VARARGS, "docs"},
        {"c_algo_da_solam",        (PyCFunction) wrap_algo_da_solam,        METH_VARARGS, "docs"},
        {"c_algo_spam",            (PyCFunction) wrap_algo_spam,            METH_VARARGS, "docs"},
        {"c_algo_spam_sparse",     (PyCFunction) wrap_algo_spam_sparse,     METH_VARARGS, "docs"},
        {"c_algo_sht_am",          (PyCFunction) wrap_algo_sht_am,          METH_VARARGS, "docs"},
        {"c_algo_sht_am_sparse",   (PyCFunction) wrap_algo_sht_am_sparse,   METH_VARARGS, "docs"},
        {"c_algo_graph_am",        (PyCFunction) wrap_algo_graph_am,        METH_VARARGS, "docs"},
        {"c_algo_graph_am_sparse", (PyCFunction) wrap_algo_graph_am_sparse, METH_VARARGS, "docs"},
        {"c_algo_opauc",           (PyCFunction) wrap_algo_opauc,           METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods, "some docs for solam algorithm.");
    import_array();
}

int main() {


    printf("test of main wrapper!\n");
}