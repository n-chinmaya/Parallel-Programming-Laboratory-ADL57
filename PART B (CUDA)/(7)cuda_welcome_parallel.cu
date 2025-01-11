#include <stdio.h>
#include <cuda.h>

__global__ void printWelcomeMessage(int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index

    if (idx < N) {
        printf("Welcome to Parallel Programming from thread %d\n", idx);
    }
}

int main() {
    int N; // Number of times to print the message
    int threadsPerBlock, blocksPerGrid;

    // Input from user
    printf("Enter the number of messages (N): ");
    scanf("%d", &N);

    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);

    // Calculate the number of blocks needed
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    printWelcomeMessage<<<blocksPerGrid, threadsPerBlock>>>(N);

    // Synchronize and finish
    cudaDeviceSynchronize();

    return 0;
}
