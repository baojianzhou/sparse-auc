#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "sort.h"

#define sign(x) (x > 0) - (x < 0)
// Standard partition process of QuickSort().  It considers the last
// element as pivot and moves all smaller element to left of it and
// greater elements to right. This function is used by random_partition()
int partition(double *arr, int l, int r) {
    double x = arr[r], temp;
    int i = l;
    for (int j = l; j <= r - 1; j++) {
        if (arr[j] >= x) {
            //---swap
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            //---swap
            i++;
        }
    }
    //---swap
    temp = arr[i];
    arr[i] = arr[r];
    arr[r] = temp;
    //---swap
    return i;
}

int random_partition(double *arr, int l, int r) {
    int n = r-l+1, pivot = (int)(lrand48() % n);
    double temp;
    temp = arr[l + pivot];
    arr[l + pivot] = arr[r];
    arr[r] = temp;
    return partition(arr, l, r);
}

double kth_largest_quick_select(double *arr, int l, int r, int k) {
    if (k > 0 && k <= r - l + 1) {
        int pos = random_partition(arr, l, r);
        if (pos-l == k-1){
            return arr[pos];
        }
        if (pos-l > k-1) {
            return kth_largest_quick_select(arr, l, pos - 1, k);
        }
        return kth_largest_quick_select(arr, pos + 1, r, k - pos + l - 1);
    }
    return - 1;
}

double kth_largest_floyd_rivest(double *arr,int left,int right,int k){
    int n,i,j;
    double t,z,s,sd,temp;
    double new_left,new_right;
    while(right > left){
        if ((right - left) > 600){
            n = right - left + 1;
            i = k - left + 1;
            z = log((double)n);
            s = 0.5 * exp(2 * z/3);
            sd = 0.5 * sqrt(z * s * (n - s)/n) * sign(i - n/2);
            new_left = fmax(left, k - i * s/n + sd);
            new_right = fmin(right, k + (n - i) * s/n + sd);
            kth_largest_floyd_rivest(arr, (int)new_left, (int)new_right, k);
        }
        t = arr[k];
        i = left;
        j = right;
        temp =  arr[left];
        arr[left]= arr[k];
        arr[k] = temp;
        if (arr[right] > t){
            temp = arr[right];
            arr[right]= arr[left];
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
        if (arr[left] == t){
            temp = arr[left];
            arr[left] = arr[j];
            arr[j] = temp;
        }else{
            j++;
            temp = arr[j];
            arr[j] = arr[right];
            arr[right] = temp;
        }
        if (j <= k){
            left = j + 1;
        }
        if(k<=j){
            right = j - 1;
        }
    }
    return arr[k];
}

void heapify(double *arr, int n, int i) {
    int largest = i; // Initialize largest as root
    int l = 2*i + 1; // left = 2*i + 1
    int r = 2*i + 2; // right = 2*i + 2
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

double kth_largest_max_heap(double *arr, int l, int r, int k){
    // Build heap (rearrange array)
    int n = r - l +1;
    double temp;
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    for (int i=n-1; i>=n-k; i--){
        // Move current root to end
        temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
    return arr[n-k];
}

int test_rand_select(){
    clock_t begin = clock();
    int n = 5000, k = 200;
    double *arr = malloc(sizeof(double)*n);
    double rand_low = 0.0, rand_high = 10.0;
    double total = 0.0;
    for(int i = 0 ; i < 20000;i++){
        for(int j=0;j<n;j++){
            arr[j] =  drand48() * ( rand_high - rand_low );
            arr[j]/= (double)RAND_MAX + rand_low;
        }
        total += kth_largest_quick_select(arr, 0, n - 1, k);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("rand_select: %.8f %.8f\n", time_spent, total);
    return 0;
}

int test_quick_sort(){
    clock_t begin = clock();
    int n = 5000, k = 200;
    double *arr = malloc(sizeof(double)*n);
    int *sorted_ind = malloc(sizeof(int)*n);
    double rand_low = 0.0, rand_high = 10.0;
    double total = 0.0;
    for(int i = 0 ; i < 20000;i++){
        for(int j=0;j<n;j++){
            arr[j] =  drand48() * ( rand_high - rand_low );
            arr[j]/= (double)RAND_MAX + rand_low;
        }
        arg_magnitude_sort_top_k(arr,sorted_ind, k, n);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("quick_sort: %.8f %.8f\n", time_spent, total);
    return 0;
}

int test_k_largest_max_heap(){
    clock_t begin = clock();
    int n = 5000, k = 200;
    double *arr = malloc(sizeof(double)*n);
    double rand_low = 0.0, rand_high = 10.0;
    double total = 0.0;
    for(int i = 0 ; i < 20000;i++){
        for(int j=0;j<n;j++){
            arr[j] =  drand48() * ( rand_high - rand_low );
            arr[j]/= (double)RAND_MAX + rand_low;
        }
        // double *arr, int l, int r, int k
        total += kth_largest_max_heap(arr,0, n-1, k);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("max_heap: %.8f %.8f\n", time_spent, total);
    return 0;
}

int test(){

}

int main() {
    test_k_largest_max_heap();
    test_quick_sort();
    test_rand_select();
}