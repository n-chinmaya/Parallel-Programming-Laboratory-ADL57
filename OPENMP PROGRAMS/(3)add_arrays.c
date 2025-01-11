#include <stdio.h>
#include <omp.h>

#define N 1000 // Size of the arrays
#define CHUNK_SIZE 100 // Chunk size for dynamic scheduling

int main() {
    int a[N], b[N], c[N]; // Arrays to hold the data

    // Initialize arrays a and b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = N - i;
    }

    // Parallel addition of arrays a and b, storing the result in c
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    // Print some of the results for verification
    printf("First 10 elements of the result array:\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}