//
// Created by baojian on 10/3/19.
//

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define PI 3.14159265358979323846

int main() {
    srand48(time(0));
    int i = 0, n = 10;
    double samples[10];
    while (i < n) {
        double x = drand48();
        double y = drand48();
        if (x != 0.0) { // to avoid zero
            samples[i++] = sqrt(-2.0 * log(x)) * cos(2.0 * PI * y);
        }
        printf("%.6f ", samples[i - 1]);
    }
}