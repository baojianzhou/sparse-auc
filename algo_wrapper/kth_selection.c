//
// Created by baojian on 9/2/19.
//
#include "kth_selection.h"


typedef struct {
    double val;
    int index;
} data_pair;

typedef struct {
    double first;
    double second;
    int index;
} lex_pair;


static inline int __comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

double kth_largest_quick_sort(const double *x, int *sorted_set, int k, int x_len) {
    if (k > x_len) {
        printf("Error: k should be <= x_len\n");
        exit(EXIT_FAILURE);
    }
    data_pair *w_tmp = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_tmp[i].val = fabs(x[i]);
        w_tmp[i].index = i;
    }
    qsort(w_tmp, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < k; i++) {
        sorted_set[i] = w_tmp[i].index;
    }
    free(w_tmp);
    return x[sorted_set[k - 1]];
}

double kth_largest_quick_select_v1(const double *array, int n, int k) {
    int i, j, l = 0, r = n - 1, mid;
    double a, kth_largest;
    double *arr = malloc(sizeof(double) * n);
    memcpy(arr, array, sizeof(double) * n);
    while (true) {
        if (r <= l + 1) {
            if ((r == l + 1) && (arr[r] > arr[l])) {
                swap(arr[l], arr[r]);
            }
            kth_largest = arr[k - 1];
            free(arr);
            return kth_largest;
        } else {
            mid = (l + r) >> 1;
            swap(arr[mid], arr[l + 1]);
            if (arr[l] < arr[r]) {
                swap(arr[l], arr[r]);
            }
            if (arr[l + 1] < arr[r]) {
                swap(arr[l + 1], arr[r]);
            }
            if (arr[l] < arr[l + 1]) {
                swap(arr[l], arr[l + 1]);
            }
            i = l + 1;
            j = r;
            a = arr[l + 1];
            for (;;) {
                do i++; while (arr[i] > a);
                do j--; while (arr[j] < a);
                if (j < i) break;
                swap(arr[i], arr[j]);
            }
            arr[l + 1] = arr[j];
            arr[j] = a;
            if (j >= (k - 1)) r = j - 1;
            if (j <= (k - 1)) l = i;
        }
    }
}

double kth_largest_quick_select_rand(double *arr, int l, int r, int k) {
    if (k > 0 && k <= r - l + 1) {

        // random partition
        int n = r - l + 1, pivot = (int) (lrand48() % n);
        swap(arr[l + pivot], arr[r]);
        int i = l;
        for (int j = l; j <= r - 1; j++) {
            if (arr[j] > arr[r]) {
                swap(arr[i], arr[j]);
                i++;
            }
        }
        swap(arr[i], arr[r]);
        int pos = i;

        if ((pos - l) == (k - 1)) {
            return arr[pos];
        }
        if ((pos - l) > (k - 1)) {
            return kth_largest_quick_select_rand(arr, l, pos - 1, k);
        }
        return kth_largest_quick_select_rand(arr, pos + 1, r, k - pos + l - 1);
    }
    return -1;
}

double kth_largest_quick_select_v2(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n);
    memcpy(arr, array, sizeof(double) * n);
    double kth_largest = kth_largest_quick_select_rand(arr, 0, n - 1, k);
    free(arr);
    return kth_largest;
}

double kth_largest_quick_select_v3(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n);
    memcpy(arr, array, sizeof(double) * n);
    int l = 0, r = n - 1, pos;
    for (int j = l; j < r; j++) {
        double pivot = arr[k - 1];
        swap(arr[k - 1], arr[r]);
        for (int i = pos = l; i < r; i++) {
            if (arr[i] > pivot) {
                swap(arr[i], arr[pos]);
                pos++;
            }
        }
        swap(arr[r], arr[pos]);
        if (pos == (k - 1)) {
            break;
        }
        if (pos < (k - 1)) {
            l = pos + 1;
        } else {
            r = pos - 1;
        }
    }
    double kth_largest = arr[k - 1];
    free(arr);
    return kth_largest;
}

/**
 * Quickselect with 3 median:
 * http://ndevilla.free.fr/median/median/src/quickselect.c
 * @param array
 * @param n
 * @param k
 * @return
 */
double kth_largest_quick_select_v4(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n), kth_largest;
    memcpy(arr, array, sizeof(double) * n);
    int low = 0, high = n - 1;
    int middle, ll, hh;
    for (;;) {
        if (high <= low) {
            kth_largest = arr[k - 1];
            free(arr);
            return kth_largest;
        }
        if (high == low + 1) {
            if (arr[low] < arr[high]) {
                swap(arr[low], arr[high]);
            }
            kth_largest = arr[k - 1];
            free(arr);
            return kth_largest;
        }
        middle = (low + high) / 2;
        if (arr[middle] < arr[high]) swap(arr[middle], arr[high]);
        if (arr[low] < arr[high]) swap(arr[low], arr[high]);
        if (arr[middle] < arr[low]) swap(arr[middle], arr[low]);
        swap(arr[middle], arr[low + 1]);
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] < arr[ll]);
            do hh--; while (arr[hh] < arr[low]);
            if (hh < ll)
                break;
            swap(arr[ll], arr[hh]);
        }
        /* Swap middle item (in position low) back into correct position */
        swap(arr[low], arr[hh]);
        /* Re-set active partition */
        if (hh <= (k - 1))
            low = ll;
        if (hh >= (k - 1))
            high = hh - 1;
    }
}


double kth_largest_wirth(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n), kth_largest;
    memcpy(arr, array, sizeof(double) * n);
    int i, j, l = 0, m = n - 1;
    double x;
    while (l < m) {
        x = arr[k - 1];
        i = l;
        j = m;
        do {
            while (arr[i] > x) i++;
            while (x > arr[j]) j--;
            if (i <= j) {
                swap(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < (k - 1)) {
            l = i;
        }
        if ((k - 1) < i) {
            m = j;
        }
    }
    kth_largest = arr[k - 1];
    free(arr);
    return kth_largest;
}


void heapify(double *arr, int n, int i) {
    int largest = i; // Initialize largest as root
    int l = 2 * i + 1; // left = 2*i + 1
    int r = 2 * i + 2; // right = 2*i + 2
    double temp;
    // If left child is larger than root
    if (l < n && arr[l] > arr[largest]) largest = l;
    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest]) largest = r;
    // If largest is not root
    if (largest != i) {
        temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}

double kth_largest_max_heap(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n);
    memcpy(arr, array, sizeof(double) * n);
    // build heap (rearrange array)
    double kth_largest;
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
    for (int i = n - 1; i >= n - k; i--) {
        // move current root to end
        swap(arr[0], arr[i]);
        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
    kth_largest = arr[n - k];
    free(arr);
    return kth_largest;
}


/**
 * Please find the algorithm in the following paper:
 * ---
 * @article{floyd1975algorithm,
 * title={Algorithm 489: the algorithm SELECT—for finding the ith
 *        smallest of n elements [M1]},
 * author={Floyd, Robert W and Rivest, Ronald L},
 * journal={Communications of the ACM},
 * volume={18}, number={3}, pages={173},
 * year={1975},
 * publisher={ACM}}
 * @param array
 * @param l
 * @param r
 * @param k
 */
void _floyd_rivest_select(double *array, int l, int r, int k) {
    register int n, i, j, s, sd, ll, rr;
    register double z, t;
    while (r > l) {
        if (r - l > 600) {
            /**
             * use select() recursively on a sample of size s to get an
             * estimate for the (k-l+1)-th smallest element into array[k],
             * biased slightly so that the (k-l+1)-th element is expected to
             * lie in the smaller set after partitioning.
             */
            n = r - l + 1;
            i = k - l + 1;
            z = log(n);
            s = (int) (0.5 * exp(2 * z / 3));
            sd = (int) (0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2));
            ll = max(l, k - i * s / n + sd);
            rr = min(r, k + (n - i) * s / n + sd);
            _floyd_rivest_select(array, ll, rr, k);
        }
        t = array[k];
        /**
         * the following code partitions x[l:r] about t, it is similar to partition
         * but will run faster on most machines since subscript range checking on i
         * and j has been eliminated.
         */
        i = l;
        j = r;
        swap(array[l], array[k]);
        if (array[r] < t) {
            swap(array[r], array[l]);
        }
        while (i < j) {
            swap(array[i], array[j]);
            do i++; while (array[i] > t);
            do j--; while (array[j] < t);
        }
        if (array[l] == t) {
            swap(array[l], array[j]);
        } else {
            j++;
            swap(array[j], array[r]);
        }
        /**
         * New adjust l, r so they surround the subset containing the
         * (k-l+1)-th smallest element.
         */
        if (j <= k) {
            l = j + 1;
        }
        if (k <= j) {
            r = j - 1;
        }
    }
}

double kth_largest_floyd_rivest(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n), kth_largest;
    memcpy(arr, array, sizeof(double) * n);
    _floyd_rivest_select(arr, 0, n - 1, k - 1);
    kth_largest = arr[k - 1];
    free(arr);
    return kth_largest;
}

double kth_largest_floyd_rivest_v2(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n), kth_largest;
    memcpy(arr, array, sizeof(double) * n);
    int left = 0, right = n - 1;
    int left2 = 0, right2 = n - 1;
    while (left < right) {
        if (arr[right2] > arr[left2]) {
            swap(arr[left2], arr[right2]);
        }
        if (arr[right2] > arr[(k - 1)]) {
            swap(arr[(k - 1)], arr[right2]);
        }
        if (arr[(k - 1)] > arr[left2]) {
            swap(arr[left2], arr[(k - 1)]);
        }
        if ((right - left) < (k - 1)) {
            int n = right - left + 1;
            int ii = (k - 1) - left + 1;
            int s = (n + n) / 3;
            int sd = (n * s * (n - s) / n) * sign(ii - n / 2);
            int left2 = max(left, (k - 1) - ii * s / n + sd);
            int right2 = min(right, (k - 1) + (n - ii) * s / n + sd);
        }
        double x = arr[(k - 1)];
        while ((right2 > (k - 1)) && (left2 < (k - 1))) {
            do left2++; while (arr[left2] > x);
            do right2--; while (arr[right2] < x);
            swap(arr[left2], arr[right2]);
        }
        left2++;
        right2--;
        if (right2 < (k - 1)) {
            while (arr[left2] > x) {
                left2++;
            }
            left = left2;
            right2 = right;
        }
        if ((k - 1) < left2) {
            while (x > arr[right2]) {
                right2--;
            }
            right = right2;
            left2 = left;
        }
        if (arr[left] > arr[right]) {
            swap(arr[right], arr[left]);
        }
    }
    kth_largest = arr[(k - 1)];
    free(arr);
    return kth_largest;
}

double kth_largest_floyd_rivest_v3(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n), kth_largest;
    memcpy(arr, array, sizeof(double) * n);
    int left = 0, right = n - 1;
    int left2 = 0, right2 = n - 1;
    while (left < right) {
        if (arr[right2] > arr[left2]) {
            swap(arr[left2], arr[right2]);
        }
        if (arr[right2] > arr[(k - 1)]) {
            swap(arr[(k - 1)], arr[right2]);
        }
        if (arr[(k - 1)] > arr[left2]) {
            swap(arr[left2], arr[(k - 1)]);
        }
        if ((right - left) < (k - 1)) {
            int n = right - left + 1;
            int ii = (k - 1) - left + 1;
            int s = (n + n) / 3;
            int sd = (n * s * (n - s) / n) * sign(ii - n / 2);
            int left2 = max(left, (k - 1) - ii * s / n + sd);
            int right2 = min(right, (k - 1) + (n - ii) * s / n + sd);
        }
        double x = arr[(k - 1)];
        while ((right2 > (k - 1)) && (left2 < (k - 1))) {
            do left2++; while (arr[left2] > x);
            do right2--; while (arr[right2] < x);
            swap(arr[left2], arr[right2]);
        }
        left2++;
        right2--;
        if (right2 < (k - 1)) {
            while (arr[left2] > x) {
                left2++;
            }
            left = left2;
            right2 = right;
        }
        if ((k - 1) < left2) {
            while (x > arr[right2]) {
                right2--;
            }
            right = right2;
            left2 = left;
        }
        if (arr[left] > arr[right]) {
            swap(arr[right], arr[left]);
        }
    }
    kth_largest = arr[(k - 1)];
    free(arr);
    return kth_largest;
}

