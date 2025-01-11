#include <stdio.h>
#include <cuda.h>

#define N 512 // Size of the matrix (N x N)

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int host_A[N][N], host_B[N][N], host_C[N][N]; // Host matrices
    int *dev_A, *dev_B, *dev_C; // Device pointers

    // Initialize host matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_A[i][j] = i + j;
            host_B[i][j] = i - j;
        }
    }

    // Allocate memory on the device
    size_t size = N * N * sizeof(int);
    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    // Copy input data from host to device
    cudaMemcpy(dev_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, size, cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);

    // Copy the result from device to host
    cudaMemcpy(host_C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print a portion of the result for verification
    printf("Result matrix (first 5x5 block):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", host_C[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}
