#include <stdio.h>
#include <omp.h>

#define N 1000 // Size of the arrays

int main() {
    int a[N], b[N], sum[N], diff[N]; // Arrays for input and results

    // Initialize arrays a and b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = N - i;
    }

    // Parallel region using sections for addition and subtraction
    #pragma omp parallel sections
    {
        // Section for addition
        #pragma omp section
        {
            for (int i = 0; i < N; i++) {
                sum[i] = a[i] + b[i];
            }
            printf("Addition completed by thread %d\n", omp_get_thread_num());
        }

        // Section for subtraction
        #pragma omp section
        {
            for (int i = 0; i < N; i++) {
                diff[i] = a[i] - b[i];
            }
            printf("Subtraction completed by thread %d\n", omp_get_thread_num());
        }
    }

    // Print some results for verification
    printf("First 10 elements of the sum array:\n");
    for (int i = 0; i < 10; i++) {
        printf("sum[%d] = %d\n", i, sum[i]);
    }

    printf("First 10 elements of the difference array:\n");
    for (int i = 0; i < 10; i++) {
        printf("diff[%d] = %d\n", i, diff[i]);
    }

    return 0;
}