/**
 * Created by Baojian(bzhou6@albany.edu) on 9/2/19.
 *
 * Kth selection algorithms:
 * Given an unsorted array A[1:n], kth selection algorithms find the
 * kth smallest/largest number in A. Such a number is also called
 * kth order statistic.
 *
 */

#ifndef SPARSE_AUC_KTH_SELECTION_H
#define SPARSE_AUC_KTH_SELECTION_H

#include "algo_base.h"

#define sign(x) (x > 0) - (x < 0)
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define swap(a,b) { double temp=(a);(a)=(b);(b)=temp; }

/**
 * Quickselect algorithm is a kth selection algorithm based on quick sort.
 * This algorithm was developed by Sir Charles Antony Richard Hoare, the
 * inventor of the quick sort algorithm and the recipient of the Turing
 * Award in 1980.
 *
 * Worst-case time complexity: O(n^2)
 * Best-case time complexity: O(n)
 * Average time complexity: O(n)
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned. k should be in [1,n]
 * @return the kth LARGEST element in this array.
 */
double kth_largest_quick_select_v1(const double *array, int n, int k);

/**
 * Different from v1, it uses a randomized pivot.
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned index. k should be in [1,n]
 * @return the kth LARGEST element in this array.
 */
double kth_largest_quick_select_v2(const double *array, int n, int k);

/**
 * The kth largest element by using maximum heap.
 *
 * Time complexity: O(n) + O(k*log(n)).
 *
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned index. k should be in [1,n]
 * @return the kth LARGEST element in this array.
 */
double kth_largest_max_heap(const double *array, int n, int k);

/**
 *
 * The kth largest element by using maximum heap.
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned index. k should be in [1,n]
 * @return the kth LARGEST element in this array.
 */
double kth_largest_floyd_rivest(const double *array, int n, int k);

#endif //SPARSE_AUC_KTH_SELECTION_H
