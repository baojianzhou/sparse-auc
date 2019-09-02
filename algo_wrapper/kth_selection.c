//
// Created by baojian on 9/2/19.
//
#include "kth_selection.h"

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


void floyd_rivest(double *array, int left, int right, int k) {
    int n, i, j, ll, rr;
    int s, sd;
    double z, t;
    while (right > left) {
        // use select recursively to sample a smaller set of size s
        // the arbitrary constants 600 and 0.5 are used in the original
        // version to minimize execution time
        if (right - left > 600) {
            n = right - left + 1;
            i = k - left + 1;
            z = log(n);
            s = 0.5 * exp(2 * z / 3);
            sd = 0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2);
            ll = max(left, k - i * s / n + sd);
            rr = min(right, k + (n - i) * s / n + sd);
            floyd_rivest(array, ll, rr, k);
        }
        // partition the elements between left and right around t
        t = array[k];
        i = left;
        j = right;
        swap(array[left], array[k]);
        if (array[right] < t) {
            swap(array[right], array[left]);
        }
        while (i < j) {
            swap(array[i], array[j]);
            do i++; while (array[i] > t);
            do j--; while (array[j] < t);
        }
        if (array[left] == t) {
            swap(array[left], array[j])
        } else {
            j++;
            swap(array[j], array[right])
        }
        // adjust left and right towards the boundaries of the subset
        // containing the (k - left + 1)th smallest element
        if (j <= k) {
            left = j + 1;
        }
        if (k <= j) {
            right = j - 1;
        }
    }
}

double kth_largest_floyd_rivest(const double *array, int n, int k) {
    double *arr = malloc(sizeof(double) * n), kth_largest;
    memcpy(arr, array, sizeof(double) * n);
    floyd_rivest(arr, 0, n - 1, k - 1);
    kth_largest = arr[k - 1];
    free(arr);
    return kth_largest;
}

