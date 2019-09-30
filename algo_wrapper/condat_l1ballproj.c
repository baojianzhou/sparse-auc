/*
 #  File            : condat_l1ballproj.c 
 #
 #  Version History : 1.0, Aug. 15, 2014 
 #
 #  Author          : Laurent Condat, PhD, CNRS research fellow in France.
 #
 #  Description     : This file contains an implementation in the C language
 #                    of algorithms described in the research paper:
 #	
 #                    L. Condat, "Fast Projection onto the Simplex and the
 #                    l1 Ball", preprint Hal-01056171, 2014.
 #
 #                    This implementation comes with no warranty: due to the
 #                    limited number of tests performed, there may remain
 #                    bugs. In case the functions would not do what they are
 #                    supposed to do, please email the author (contact info
 #                    to be found on the web).
 #
 #                    If you use this code or parts of it for any purpose,
 #                    the author asks you to cite the paper above or, in 
 #                    that event, its published version. Please email him if 
 #                    the proposed algorithms were useful for one of your 
 #                    projects, or for any comment or suggestion.
 #
 #  Usage rights    : Copyright Laurent Condat.
 #                    This file is distributed under the terms of the CeCILL
 #                    licence (compatible with the GNU GPL), which can be
 #                    found at the URL "http://www.cecill.info".
 #
 #  This software is governed by the CeCILL license under French law and
 #  abiding by the rules of distribution of free software. You can  use,
 #  modify and or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL :
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
*/


/* This code was compiled using
gcc -march=native -O2 condat_l1ballproj.c -o main  -I/usr/local/include/ 
	-lm -lgsl  -L/usr/local/lib/
On my machine, gcc is actually a link to the compiler Apple LLVM version 5.1 
(clang-503.0.40) */


/* The following functions are implemented:
l1ballproj_IBIS
l1ballproj_Condat (proposed algorithm)
These functions take the same parameters. They project the vector y onto
the closest vector x of same length (parameter N in the paper) with 
sum_{n=0}^{N-1}|x[n]|<=a. 
We must have length>=1.
For l1ballproj_condat only:
we can have x==y (projection done in place). If x!=y, the arrays x and y must
not overlap, as x is used for temporary calculations before y is accessed.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>


/* Proposed algorithm */
static void l1ballproj_condat(double *y, double *x, int length, const double a) {
    if (a <= 0.0) {
        if (a == 0.0) memset(x, 0, length * sizeof(double));
        return;
    }
    double *aux = (x == y ? (double *) malloc(length * sizeof(double)) : x);
    int aux_len = 1;
    int aux_len_hold = -1;
    double tau = (*aux = (*y >= 0.0 ? *y : -*y)) - a;
    int i = 1;
    for (; i < length; i++) {
        if (y[i] > 0.0) {
            if (y[i] > tau) {
                if ((tau += ((aux[aux_len] = y[i]) - tau) / (aux_len - aux_len_hold)) <=
                    y[i] - a) {
                    tau = y[i] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        } else if (y[i] != 0.0) {
            if (-y[i] > tau) {
                if ((tau += ((aux[aux_len] = -y[i]) - tau) / (aux_len - aux_len_hold))
                    <= aux[aux_len] - a) {
                    tau = aux[aux_len] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        }
    }
    if (tau <= 0) {    /* y is in the l1 ball => x=y */
        if (x != y) memcpy(x, y, length * sizeof(double));
        else free(aux);
    } else {
        double *aux0 = aux;
        if (aux_len_hold >= 0) {
            aux_len -= ++aux_len_hold;
            aux += aux_len_hold;
            while (--aux_len_hold >= 0)
                if (aux0[aux_len_hold] > tau)
                    tau += ((*(--aux) = aux0[aux_len_hold]) - tau) / (++aux_len);
        }
        do {
            aux_len_hold = aux_len - 1;
            for (i = aux_len = 0; i <= aux_len_hold; i++)
                if (aux[i] > tau)
                    aux[aux_len++] = aux[i];
                else
                    tau += (tau - aux[i]) / (aux_len_hold - i + aux_len);
        } while (aux_len <= aux_len_hold);
        for (i = 0; i < length; i++)
            x[i] = (y[i] - tau > 0.0 ? y[i] - tau : (y[i] + tau < 0.0 ? y[i] + tau : 0.0));
        if (x == y) free(aux0);
    }
}

int main() {
    double *y;
    double *x;
    int i, j;
    int Nbrea;
    unsigned int length;
    clock_t start, end;
    double timemean = 0.0, timevar = 0.0, timedelta;
    const double a = 1.0;
    srand((unsigned int) pow(time(NULL) % 100, 3));
    double *timetable;

    length = 1400000;
    Nbrea = 1000;
    y = (double *) malloc(length * sizeof(double));
    x = (double *) malloc(length * sizeof(double));
    timetable = (double *) malloc((Nbrea + 1) * sizeof(double));
    for (j = 0; j <= Nbrea; j++) {
        for (i = 0; i < length; i++) {
            y[i] = drand48() * 0.01;
        }
        y[(int) (rand() / (((double) RAND_MAX + 1.0) / length))] += a;
        start = clock();
        l1ballproj_condat(y, x, length, a);
        end = clock();
        timetable[j] = (double) (end - start) / CLOCKS_PER_SEC;
    }
    /* we discard the first value, because the computation time is higher,
    probably because of cache operations */
    for (j = 1; j <= Nbrea; j++) {
        timedelta = timetable[j] - timemean;
        timemean += timedelta / j;
        timevar += timedelta * (timetable[j] - timemean);
    }
    timevar = sqrt(timevar / Nbrea);
    printf("av. time: %e std dev: %e\n", timemean, timevar);
    free(x);
    free(y);
    return 0;
}





