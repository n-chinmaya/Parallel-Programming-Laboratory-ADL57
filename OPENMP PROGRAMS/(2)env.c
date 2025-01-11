#include <stdio.h>
#include <omp.h>

int main() {
    // Start a parallel region
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();       // Get the thread ID
        int num_threads = omp_get_num_threads();   // Total number of threads
        int num_procs = omp_get_num_procs();       // Number of processors available
        int in_parallel = omp_in_parallel();       // Check if in parallel region

        // Print information about the environment
        #pragma omp critical
        {
            printf("Thread %d out of %d threads:\n", thread_id, num_threads);
            printf("  Number of processors available: %d\n", num_procs);
            printf("  In parallel region: %s\n", in_parallel ? "Yes" : "No");
        }
    }

    return 0;
}