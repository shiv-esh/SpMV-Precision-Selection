#include <cstdio>    // For FILE, fopen, fclose, etc.
#include <cstdlib>   // For EXIT_FAILURE, EXIT_SUCCESS, etc.
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/fp24.cuh"
// #include "../include/fp40.h"
// #include "../include/fp48.h"
// #include "../include/fp56.h"
#include "mmio.c"  // Include this for Matrix Market I/O

 // For EXIT_FAILURE, EXIT_SUCCESS, etc.


__global__ void adaptivePrecisionSpMV(int m, int* row_ptr, int* col_idx, float* values, float* x, float* y, int* bucket_start, int precision_level) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        float sum = 0.0f;

        // Iterate over each bucket
        for (int b = 0; b < bucket_start[i + 1] - bucket_start[i]; b++) {
            float y_partial = 0.0f;

            int start_idx = bucket_start[i] + b;
            int end_idx = bucket_start[i] + b + 1;

            // Iterate over non-zero elements in this bucket
            for (int jj = row_ptr[start_idx]; jj < row_ptr[end_idx]; jj++) {
                if (precision_level == 0) {
                    // Use half precision (__half)
                    __half val_half = __float2half(values[jj]);
                    y_partial += __half2float(val_half) * x[col_idx[jj]];
                } 
            }

            // Accumulate in the lowest precision needed
            sum += y_partial;
        }

        // Store result in output vector y
        y[i] = sum;
    }
}
__global__ void normalSpMV(int m, int* row_ptr, int* col_idx, float* values, float* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        double sum = 0.0;
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        for (int jj = row_start; jj < row_end; jj++) {
            sum += (double)values[jj] * (double)x[col_idx[jj]];
        }

        y[i] = sum;
    }
}
int main(int argc, char** argv) {
   int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n, nz;   
    int i, *I, *J;
    double *val;

    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0)
        exit(1);


    std::vector<int> h_row_idx(nz), h_col_idx(nz);
    std::vector<float> h_values(nz);

    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %f\n", &h_row_idx[i], &h_col_idx[i], &h_values[i]);
        h_row_idx[i]--;  // Convert 1-based to 0-based index
        h_col_idx[i]--;
    }

    fclose(f);

    // Convert the COO format to CSR format
    std::vector<int> h_row_ptr(m + 1, 0);
    for (int i = 0; i < nz; i++) {
        h_row_ptr[h_row_idx[i] + 1]++;
    }
    for (int i = 0; i < m; i++) {
        h_row_ptr[i + 1] += h_row_ptr[i];
    }

    std::vector<int> h_col_idx_csr(nz);
    std::vector<float> h_values_csr(nz);

    for (int i = 0; i < nz; i++) {
        int row = h_row_idx[i];
        int dest = h_row_ptr[row];

        h_col_idx_csr[dest] = h_col_idx[i];
        h_values_csr[dest] = h_values[i];

        h_row_ptr[row]++;
    }

    // Fix the shifted row pointers
    for (int i = m; i > 0; i--) {
        h_row_ptr[i] = h_row_ptr[i - 1];
    }
    h_row_ptr[0] = 0;

    // Create a vector of all ones
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(m);

    // Example bucket partitioning (for simplicity, let's assume one bucket per row)
    std::vector<int> h_bucket_start(m + 1);
    for (int i = 0; i <= m; i++) {
        h_bucket_start[i] = i;
    }

    // Choose the precision level
    int precision_level = 1;  // 0 for half precision, 1 for fp24, etc.

    // Device memory pointers
    int *d_row_ptr, *d_col_idx, *d_bucket_start;
    float *d_values, *d_x, *d_y;
    double *d_y_normal;
    std::vector<double> h_y_normal(m);
    // Allocate device memory
    cudaMalloc((void**)&d_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, nz * sizeof(int));
    cudaMalloc((void**)&d_values, nz * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));
    cudaMalloc((void**)&d_bucket_start, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_y_normal, m * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx_csr.data(), nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values_csr.data(), nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucket_start, h_bucket_start.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;

    normalSpMV<<<gridSize, blockSize>>>(m, d_row_ptr, d_col_idx, d_values, d_x, d_y_normal);
    // Launch the CUDA kernel
    adaptivePrecisionSpMV<<<gridSize, blockSize>>>(m, d_row_ptr, d_col_idx, d_values, d_x, d_y, d_bucket_start, precision_level);

    // Copy the result back to the host
    cudaMemcpy(h_y.data(), d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_normal.data(), d_y_normal, m * sizeof(double), cudaMemcpyDeviceToHost);
    // Display the result
  std::cout << "Comparing results between normal SpMV (fp64) and adaptive precision SpMV:" << std::endl;
    for (int i = 0; i < m; i++) {
        double diff = std::abs(h_y_normal[i] - h_y[i]);
        std::cout << "Row " << i << ": normal = " << h_y_normal[i] << ", adaptive = " << h_y[i] << ", difference = " << diff << std::endl;
    }

    // Free device memory
    cudaFree(d_y_normal);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_bucket_start);

    return 0;
}
