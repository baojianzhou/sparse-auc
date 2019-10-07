//
// Created by baojian on 10/3/19.
//

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define PI 3.14159265358979323846

void test_random() {
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

void test_mod() {
    int data_n = 6721;
    int data_b = 52;
    int n_max = data_n / data_b;
    int n_min = 0;
    srand((unsigned int) time(NULL));
    for (int i = 0; i < 1000; i++) {
        int n_rand_num = rand() % ((n_max) - n_min) + n_min;
        if (n_rand_num == 128) {
            printf("%03d ", n_rand_num);
        }
    }
}

int main() {
    test_mod();
}