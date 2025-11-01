#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "rpfpCSR.cuh"
#include "loadCSR.h"
#include<iostream>
void createBuckets(const csr_matrix_t* A, int q, double epsilon, rpfpCSR* Ap) {
    int m = A->nrows;

    // Initialize bucket counters
    cudaMallocManaged(&Ap->ia8, (m + 1) * sizeof(int));
    cudaMallocManaged(&Ap->ia16, (m + 1) * sizeof(int));
    cudaMallocManaged(&Ap->ia32, (m + 1) * sizeof(int));
    cudaMallocManaged(&Ap->ia64, (m + 1) * sizeof(int));

    for (int i = 0; i < m + 1; i++) {
        Ap->ia8[i] = Ap->ia16[i] = Ap->ia32[i] = Ap->ia64[i] = 0;
    }
    double thresholds[q];
    // Iterate through each row of the CSR matrix
    for (int i = 0; i < m; i++) {
        // Calculate the sum of absolute values in row i
        double row_sum = 0.0;
        for (int jj = A->row_ptr[i]; jj < A->row_ptr[i + 1]; jj++) {
            row_sum += fabs(A->values[jj]);
        }

        // Determine precision intervals for each bucket
        
        for (int k = 0; k < q; k++) {
            thresholds[k] = epsilon * row_sum / pow(2, k + 1);
        }

        // Assign elements to the appropriate bucket
        for (int jj = A->row_ptr[i]; jj < A->row_ptr[i + 1]; jj++) {
            double abs_val = fabs(A->values[jj]);

            int assigned_bucket = q - 1;  // Default to the lowest precision
            for (int k = 0; k < q - 1; k++) {
                if (abs_val > thresholds[k]) {
                    assigned_bucket = k;
                    break;
                }
            }

            // Assign to the appropriate CSR structure based on the bucket
            if (assigned_bucket == 0) {
                Ap->ia64[i + 1]++;
            } else if (assigned_bucket == 1) {
                Ap->ia32[i + 1]++;
            } else if (assigned_bucket == 2) {
                Ap->ia16[i + 1]++;
            } else {
                Ap->ia8[i + 1]++;
            }
        }
    }

    // Convert row counters to cumulative sum
    for (int i = 0; i < m; i++) {
        Ap->ia8[i + 1] += Ap->ia8[i];
        Ap->ia16[i + 1] += Ap->ia16[i];
        Ap->ia32[i + 1] += Ap->ia32[i];
        Ap->ia64[i + 1] += Ap->ia64[i];
    }

    // Allocate memory for data
    cudaMallocManaged(&Ap->ja8, Ap->ia8[m] * sizeof(int));
    cudaMallocManaged(&Ap->a8, Ap->ia8[m] * sizeof(uint8_t));
    cudaMallocManaged(&Ap->ja16f, Ap->ia16[m] * sizeof(int));
    cudaMallocManaged(&Ap->a16f, Ap->ia16[m] * sizeof(__half));
    cudaMallocManaged(&Ap->ja32, Ap->ia32[m] * sizeof(int));
    cudaMallocManaged(&Ap->a32, Ap->ia32[m] * sizeof(float));
    cudaMallocManaged(&Ap->ja64, Ap->ia64[m] * sizeof(int));
    cudaMallocManaged(&Ap->a64, Ap->ia64[m] * sizeof(double));

    // Fill the buckets with values and column indices
    int* counter8 = (int*)malloc((m + 1) * sizeof(int));
    int* counter16 = (int*)malloc((m + 1) * sizeof(int));
    int* counter32 = (int*)malloc((m + 1) * sizeof(int));
    int* counter64 = (int*)malloc((m + 1) * sizeof(int));

    memcpy(counter8, Ap->ia8, (m + 1) * sizeof(int));
    memcpy(counter16, Ap->ia16, (m + 1) * sizeof(int));
    memcpy(counter32, Ap->ia32, (m + 1) * sizeof(int));
    memcpy(counter64, Ap->ia64, (m + 1) * sizeof(int));

    for (int i = 0; i < m; i++) {
        for (int jj = A->row_ptr[i]; jj < A->row_ptr[i + 1]; jj++) {
            double abs_val = fabs(A->values[jj]);
            int col_idx = A->col_idx[jj];

            int assigned_bucket = q - 1;
            for (int k = 0; k < q - 1; k++) {
                if (abs_val > thresholds[k]) {
                    assigned_bucket = k;
                    break;
                }
            }

            if (assigned_bucket == 0) {
                Ap->a64[counter64[i]] = (double)A->values[jj];
                Ap->ja64[counter64[i]++] = col_idx;
            } else if (assigned_bucket == 1) {
                Ap->a32[counter32[i]] = A->values[jj];
                Ap->ja32[counter32[i]++] = col_idx;
            } else if (assigned_bucket == 2) {
                Ap->a16f[counter16[i]] = __float2half(A->values[jj]);
                Ap->ja16f[counter16[i]++] = col_idx;
            } else {
                Ap->a8[counter8[i]] = (uint8_t)(A->values[jj] * 255.0f);
                Ap->ja8[counter8[i]++] = col_idx;
            }
        }
    }

    free(counter8);
    free(counter16);
    free(counter32);
    free(counter64);
}
__global__ void adaptiveSpMV(int n, rpfpCSR Ap, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0;

        // FP8
        for (int k = Ap.ia8[i]; k < Ap.ia8[i + 1]; ++k) {
            sum += ((double)Ap.a8[k] / 255.0) * x[Ap.ja8[k]];
        }
        // FP16
        for (int k = Ap.ia16[i]; k < Ap.ia16[i + 1]; ++k) {
            sum += __half2float(Ap.a16f[k]) * x[Ap.ja16f[k]];
        }
        // FP32
        for (int k = Ap.ia32[i]; k < Ap.ia32[i + 1]; ++k) {
            sum += Ap.a32[k] * x[Ap.ja32[k]];
        }
        // FP64
        for (int k = Ap.ia64[i]; k < Ap.ia64[i + 1]; ++k) {
            sum += Ap.a64[k] * x[Ap.ja64[k]];
        }

        y[i] = sum;
    }
}
__global__ void standardSpMV(int m, int* row_ptr, int* col_idx, float* values, float* x, double* y) {
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

double computeL2Norm(std::vector<double> vec, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += vec[i] * vec[i];
    }
    return sqrt(norm);
}
template <typename T> 
double L2Norm(std::vector<T>pred, std::vector<double>truth, int n) {
  double sum_diff = 0, diff, sum_truth = 0;
  for (int i = 0; i < n; ++i) {
    // || x - x' ||_2
    diff = double(pred[i]) - truth[i];
    sum_diff += diff * diff;
    // || x ||_2
    sum_truth += truth[i] * truth[i];
  }
  if (sum_truth == sum_diff && sum_truth == 0) return 0;
  return sqrt(sum_diff) / sqrt(sum_truth);
}

// Function to calculate the norm-wise relative error between two vectors
double calculateNormwiseRelativeError(std::vector<float> approximate, std::vector<double> exact, int size) {
    std::vector<double>diff(size);
    for (int i = 0; i < size; ++i) {
        diff[i] = approximate[i] - exact[i];
    }

    float norm_diff = computeL2Norm(diff, size);
    double norm_exact = computeL2Norm(exact, size);


    return norm_diff / norm_exact;
}

int main(int argc, char** argv) {
    const char* filename;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    } else { 
        filename = argv[1];
    }

    // Load the CSR matrix
    csr_matrix_t* csr_matrix = load_csr_matrix(filename);

    // Initialize the adaptive CSR structure
    rpfpCSR Ap;
    double epsilon = 2e-53;
    int q = 4;  // Number of precision levels

    // Allocate device memory
    float* d_x,*d_y;
    double *d_y64;
    int* d_row_ptr, *d_col_idx;
    float* d_values;
    
    cudaMalloc((void**)&d_row_ptr, (csr_matrix->nrows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, csr_matrix->nnz * sizeof(int));
    cudaMalloc((void**)&d_values, csr_matrix->nnz * sizeof(float));
    cudaMallocManaged(&d_x, csr_matrix->nrows * sizeof(float));
    cudaMallocManaged(&d_y, csr_matrix->nrows * sizeof(float));
    cudaMallocManaged(&d_y64, csr_matrix->nrows * sizeof(double));

    // Initialize vector x to all ones
    for (int i = 0; i < csr_matrix->nrows; i++) {
        d_x[i] = 1.0f;
    }

    // Copy CSR matrix data to the device
    cudaMemcpy(d_row_ptr, csr_matrix->row_ptr, (csr_matrix->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, csr_matrix->col_idx, csr_matrix->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, csr_matrix->values, csr_matrix->nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (csr_matrix->nrows + blockSize - 1) / blockSize;
    int iters = 500;
    createBuckets(csr_matrix, q, epsilon, &Ap);
    for(int i = 1; i <= iters; i++) {
        adaptiveSpMV<<<numBlocks, blockSize>>>(csr_matrix->nrows, Ap, d_x, d_y);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Synchronize and copy results back to the host
    cudaDeviceSynchronize();

     standardSpMV<<<numBlocks, blockSize>>>(csr_matrix->nrows, d_row_ptr, d_col_idx, d_values, d_x, d_y64);
    
    std::vector<float> h_y(csr_matrix->nrows);
    std::vector<double> h_y64(csr_matrix->nrows);
    cudaMemcpy(h_y.data(), d_y, csr_matrix->nrows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y64.data(), d_y64, csr_matrix->nrows * sizeof(double), cudaMemcpyDeviceToHost);


   
    double relative_error = L2Norm(h_y,h_y64,csr_matrix->nrows);
    std::cout << "CSV_OUTPUT: " << elapsedTime <<","<<relative_error<<std::endl;
   

    // Free resources
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y64);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);

    cudaFree(Ap.ia8);
    cudaFree(Ap.ja8);
    cudaFree(Ap.a8);
    cudaFree(Ap.ia16);
    cudaFree(Ap.ja16f);
    cudaFree(Ap.a16f);
    cudaFree(Ap.ia32);
    cudaFree(Ap.ja32);
    cudaFree(Ap.a32);
    cudaFree(Ap.ia64);
    cudaFree(Ap.ja64);
    cudaFree(Ap.a64);

    free_csr_matrix(csr_matrix);  // Assume a function exists
    return 0;
}
