#include <stdio.h>
#include <cuda.h>

#define N 512 // Size of the vectors

// CUDA kernel for dot product
__global__ void dotProduct(int *a, int *b, int *result, int n) {
    __shared__ int partialSum[256]; // Shared memory for partial sums

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int threadID = threadIdx.x;
    partialSum[threadID] = 0;

    // Compute partial dot product
    if (tid < n) {
        partialSum[threadID] = a[tid] * b[tid];
    }

    // Synchronize threads within the block
    __syncthreads();

    // Reduce the partial sums within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadID < stride) {
            partialSum[threadID] += partialSum[threadID + stride];
        }
        __syncthreads();
    }

    // Add the block's result to the global result array
    if (threadID == 0) {
        atomicAdd(result, partialSum[0]);
    }
}

int main() {
    int host_a[N], host_b[N], host_result = 0; // Host data
    int *dev_a, *dev_b, *dev_result; // Device pointers

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        host_a[i] = i + 1;
        host_b[i] = i + 2;
    }

    // Allocate memory on the device
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_result, sizeof(int));

    // Copy vectors to device
    cudaMemcpy(dev_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, &host_result, sizeof(int), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_result, N);

    // Copy the result back to the host
    cudaMemcpy(&host_result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Dot product: %d\n", host_result);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    return 0;
}
