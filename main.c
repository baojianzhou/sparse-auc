#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "sort.h"

#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { float temp=(a);(a)=(b);(b)=temp; }

double kth_largest_floyd_rivest(double *arr, int left, int right, int k) {
    int n, i, j;
    double t, z, s, sd, temp;
    double new_left, new_right;
    while (right > left) {
        if ((right - left) > 600) {
            n = right - left + 1;
            i = k - left + 1;
            z = log((double) n);
            s = 0.5 * exp(2 * z / 3);
            sd = 0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2);
            new_left = fmax(left, k - i * s / n + sd);
            new_right = fmin(right, k + (n - i) * s / n + sd);
            kth_largest_floyd_rivest(arr, (int) new_left, (int) new_right, k);
        }
        t = arr[k];
        i = left;
        j = right;
        temp = arr[left];
        arr[left] = arr[k];
        arr[k] = temp;
        if (arr[right] > t) {
            temp = arr[right];
            arr[right] = arr[left];
            arr[left] = temp;
        }
        while (i < j) {
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
            while (arr[i] < t) {
                i++;
            }
            while (arr[j] > t) {
                j++;
            }
        }
        if (arr[left] == t) {
            temp = arr[left];
            arr[left] = arr[j];
            arr[j] = temp;
        } else {
            j++;
            temp = arr[j];
            arr[j] = arr[right];
            arr[right] = temp;
        }
        if (j <= k) {
            left = j + 1;
        }
        if (k <= j) {
            right = j - 1;
        }
    }
    return arr[k];
}

void k_largest_select(double array[], int left, int right, int k) {
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
            k_largest_select(array, ll, rr, k);
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
            i++;
            j--;
            while (array[i] > t) {
                i++;
            }
            while (array[j] < t) {
                j--;
            }
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

int test_quick_sort() {
    clock_t begin = clock();
    int n = 5000, k = 200;
    double *arr = malloc(sizeof(double) * n);
    int *sorted_ind = malloc(sizeof(int) * n);
    double rand_low = 0.0, rand_high = 10.0;
    double total = 0.0;
    for (int i = 0; i < 20000; i++) {
        for (int j = 0; j < n; j++) {
            arr[j] = drand48() * (rand_high - rand_low);
            arr[j] /= (double) RAND_MAX + rand_low;
        }
        arg_magnitude_sort_top_k(arr, sorted_ind, k, n);
    }
    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("quick_sort: %.8f %.8f\n", time_spent, total);
    return 0;
}


int test() {
    double arr[] = {1, 23, 12, 9, 30, 2, 50};
    int k = 3;
    // double x = kth_largest_floyd_rivest(arr,0,6,k);
    k_largest_select(arr, 0, 6, k - 1);
    printf("%.2f ", arr[k - 1]);
}

int main() {
    test();
    //test_k_largest_max_heap();
    //test_quick_sort();
    //test_rand_select();
}