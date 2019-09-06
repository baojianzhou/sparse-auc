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
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }

double kth_largest_quick_sort(const double *x, int *sorted_set, int k, int x_len);

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
 * Different from v1, it is non-recursive.
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned index. k should be in [1,n]
 * @return the kth LARGEST element in this array.
 */
double kth_largest_quick_select_v3(const double *array, int n, int k);


/**
 * Different from v1, it is with 3 median.
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned index. k should be in [1,n]
 * @return the kth LARGEST element in this array.
 */
double kth_largest_quick_select_v4(const double *array, int n, int k);

/**
 * Algorithm from N. Wirth's book, implementation by N. Devillard.
 * This code in public domain.
 * http://ndevilla.free.fr/median/median/src/wirth.c
 * @param array
 * @param n
 * @param k
 * @return
 */
double kth_largest_wirth(const double *array, int n, int k);

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
 * The kth largest element by using Floyd-Rivest algorithm.
 * This algorithm was proposed in ,
 * Ronald Linn Rivest won the Turing Award in 2002 while his advisor,
 * Robert W. Floyd won the Turing Award in 1978.
 *
 * Expected time complexity: O(n)
 * Expected number of comparisons: O(n+min(k,n-k)) + O(sqrt{n})
 *
 * ---
 * @article{blum1973time,
 * title={Time bounds for selection},
 * author={Blum, Manuel and Floyd, Robert W. and
 *         Pratt, Vaughan R. and Rivest, Ronald L.
 *         and Tarjan, Robert Endre},
 * year={1973}}
 * ---
 * @article{floyd1975algorithm,
 * title={Algorithm 489: the algorithm SELECT—for finding the ith
 *        smallest of n elements [M1]},
 * author={Floyd, Robert W and Rivest, Ronald L},
 * journal={Communications of the ACM},
 * volume={18}, number={3}, pages={173},
 * year={1975},
 * publisher={ACM}}
 *
 * ---
 * @param array, the unsorted array
 * @param n, the number of elements in this array.
 * @param k, the kth-largest returned index. k should be in [1,n]
 * @return the kth LARGEST element in this array.

 */
double kth_largest_floyd_rivest(const double *array, int n, int k);

double kth_largest_floyd_rivest_v2(const double *array, int n, int k);

#endif //SPARSE_AUC_KTH_SELECTION_H
