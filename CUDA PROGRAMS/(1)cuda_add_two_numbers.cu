#include <stdio.h>

// CUDA kernel to add two numbers
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}

int main() {
    int host_a = 5, host_b = 10, host_result; // Host variables
    int *dev_a, *dev_b, *dev_result;         // Device pointers

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, sizeof(int));
    cudaMalloc((void**)&dev_b, sizeof(int));
    cudaMalloc((void**)&dev_result, sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(dev_a, &host_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &host_b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and 1 thread
    addNumbers<<<1, 1>>>(dev_a, dev_b, dev_result);

    // Copy the result from device to host
    cudaMemcpy(&host_result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("The sum of %d and %d is %d\n", host_a, host_b, host_result);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    return 0;
}
