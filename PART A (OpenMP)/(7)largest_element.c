#include <stdio.h>
#include <omp.h>

#define N 1000 // Size of the array

int main() {
    int a[N]; // Array to hold the data
    int largest_parallel = -1; // Variable to hold the largest element (parallel)
    int largest_serial = -1;   // Variable to hold the largest element (serial)

    // Initialize the array with some values
    for (int i = 0; i < N; i++) {
        a[i] = i * 2; // Array elements are 0, 2, 4, ..., 2*(N-1)
    }

    // Parallel computation to find the largest element
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp critical
        {
            if (a[i] > largest_parallel) {
                largest_parallel = a[i];
            }
        }
    }

    // Serial computation to find the largest element
    for (int i = 0; i < N; i++) {
        if (a[i] > largest_serial) {
            largest_serial = a[i];
        }
    }

    // Print the results
    printf("Largest element (parallel): %d\n", largest_parallel);
    printf("Largest element (serial): %d\n", largest_serial);

    // Verify if the results match
    if (largest_parallel == largest_serial) {
        printf("The results match. Verification successful!\n");
    } else {
        printf("The results do not match. Verification failed!\n");
    }

    return 0;
}
