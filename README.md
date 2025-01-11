# CUDA and OpenMP Programs
## Below are the instructions on how to run these programs. 

---

## Prerequisites

1. **CUDA Toolkit**: Ensure you have the CUDA Toolkit installed on your system. You can download it from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads).
2. **OpenMP Support**: Ensure your compiler supports OpenMP (e.g., `gcc` for Linux).
3. **Compiler**:
   - Use `nvcc` for CUDA programs.
   - Use `gcc` for OpenMP programs.

---

## How to Run the Programs

### CUDA Programs

1. Open a terminal.
2. Navigate to the directory containing the `.cu` files.
3. Compile the desired program using `nvcc` (e.g., `nvcc -o output_file source_file.cu`).
4. Run the compiled program (e.g., `./output_file`).

Example:
```bash
nvcc -o <file_name.cu> <file_name.out>

./<file_name.out>
```

### OpenMP Programs

1. Open a terminal.
2. Navigate to the directory containing the `.c` files.
3. Compile the desired program using `gcc` with the `-fopenmp` flag (e.g., `gcc -o output_file source_file.c -fopenmp`).
4. Run the compiled program (e.g., `./output_file`).

Example:
```bash
gcc -fopenmp <file_name.c> -o <file_name.out>

./<file_name.out>
```

---

## Files Included

- CUDA Programs:
  - `(1)cuda_add_two_numbers.cu`
  - `(2)cuda_vector_add.cu`
  - `(3)cuda_matrix_add.cu`
  - `(4)cuda_devive_info.cu`
  - `(5)cuda_matrix_multiplication.cu`
  - `(6)cuda_dot_product.cu`
  - `(7)cuda_welcome_parallel.cu`
    
- OpenMP Programs:
  - `(1)pi.c`
  - `(2)env.c`
  - `(3)add_arrays.c`
  - `(4)sub_arrays.c`
  - `(5)sum_arrays_reduction_clause.c`
  - `(6)matrix_multiplication.c`
  - `(7)largest_element.c`

---

## Follow Me
If you found these programs helpful, please consider following me on GitHub for more help from me:

[GitHub Profile](https://github.com/Srinidhi-070)

---

## Disclaimer
This repository is for educational purposes only. If you use this code to cheat in your internals, **I am not responsible for any consequences**. Please do not blame me for any issues arising from copying this repository.

Thank you for your understanding and support! 😊
