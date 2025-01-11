#include <stdio.h>
#include <cuda.h>

#define N 512 // Size of the vectors

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Calculate global thread index
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int host_a[N], host_b[N], host_c[N]; // Host arrays
    int *dev_a, *dev_b, *dev_c;          // Device pointers

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        host_a[i] = i;
        host_b[i] = N - i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with enough blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate the number of blocks
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

    // Copy the result from device to host
    cudaMemcpy(host_c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a portion of the result for verification
    printf("Result (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d\n", i, host_c[i]);
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
