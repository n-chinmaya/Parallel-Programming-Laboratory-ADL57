#include <stdio.h>
#include <omp.h>

#define N 1000 // Size of the array

int main() {
    int a[N]; // Array to hold the data
    int sum = 0; // Variable to hold the sum

    // Initialize the array with values
    for (int i = 0; i < N; i++) {
        a[i] = i + 1; // Fill the array with values 1, 2, ..., N
    }

    // Parallel computation of sum using reduction clause
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }

    // Print the final result
    printf("Sum of array elements: %d\\n", sum);

    return 0;
}
