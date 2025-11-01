#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <thrust/pair.h>
#include<assert.h>
#include <sstream>
#include <random>
#include <assert.h>
#include <cstdlib>
using namespace std;
template <typename T>
struct CSR_Matrix {
  int M, N, nz, *rowptr, *cols;
  T *vals;
};

typedef enum { EMPTY = -1, DOUBLE = 0, SINGLE = 1 } precision_e;
void split_entrywise(CSR_Matrix<double> *mat, precision_e *p, CSR_Matrix<float> *matS, CSR_Matrix<double> *matD) {
  int nzS = 0, nzD = 0, i, j;

  // First, count the non-zeros that so you can allocate accordingly
  for (i = 0; i < mat->M; ++i)
    for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j)
      if (p[j] == SINGLE)
        nzS++;
      else if (p[j] == DOUBLE)
        nzD++;
  assert(nzS + nzD == mat->nz);

  // Allocations
  matS->M = mat->M;
  matS->N = mat->N;
  matS->nz = nzS;
  matS->rowptr = (int *)malloc((mat->M + 1) * sizeof(int));
  matS->cols = (int *)malloc(nzS * sizeof(int));
  matS->vals = (float *)malloc(nzS * sizeof(float));

  matD->M = mat->M;
  matD->N = mat->N;
  matD->nz = nzD;
  matD->rowptr = (int *)malloc((mat->M + 1) * sizeof(int));
  matD->cols = (int *)malloc(nzD * sizeof(int));
  matD->vals = (double *)malloc(nzD * sizeof(double));

  // Assigning the non-zeros
  int nzS_i = 0, nzD_i = 0;
  matS->rowptr[0] = 0;
  matD->rowptr[0] = 0;
  for (i = 0; i < mat->M; ++i) {
    matS->rowptr[i + 1] = matS->rowptr[i];
    matD->rowptr[i + 1] = matD->rowptr[i];
    for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
      if (p[j] == SINGLE) {
        matS->cols[nzS_i] = mat->cols[j];
        matS->vals[nzS_i] = (float)(mat->vals[j]);
        matS->rowptr[i + 1]++;
        nzS_i++;
      } else if (p[j] == DOUBLE) {
        matD->cols[nzD_i] = mat->cols[j];
        matD->vals[nzD_i] = mat->vals[j];
        matD->rowptr[i + 1]++;
        nzD_i++;
      }
    }
  }
  assert((nzS_i == nzS) && (nzD_i == nzD));
}

typedef thrust::pair<int, int> CoordinateT;
__device__ CoordinateT MergePathSearch(int diagonal, int *a, int a_len, int *b, int b_len) {
    int x_min = max(diagonal - b_len, 0);
    int x_max = min(diagonal, a_len);

    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;
        if (a[pivot] <= b[diagonal - pivot - 1]) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    return CoordinateT(min(x_min, a_len), diagonal - x_min);
}
// Template kernel for combining partial sums
template <typename T>
__global__ void combinePartialSum(int nrow, T *d_partial, T *d_res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrow) {
        printf("%d: %lf %lf\n", tid, static_cast<double>(d_res[tid]), static_cast<double>(d_partial[tid]));
        d_res[tid] += d_partial[tid];
    }
}

// Template kernel for CUDA merge-based CSR matrix-vector multiplication
template <typename T>
__global__ void CudaMergeCsrmv(int n_threads, int *d_row, int *d_col, T *d_data, double *d_v, T *d_res, T *d_partial, int nrow, int ncol, int ndata, int *nz_indices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < (ndata + nrow)) {
        int n_items = nrow + ndata;
        int items_per_thread = (n_items + n_threads - 1) / n_threads;

        int diag_start = min(items_per_thread * tid, n_items);
        int diag_end = min(diag_start + items_per_thread, n_items);
        CoordinateT thread_coord = MergePathSearch(diag_start, d_row, nrow, nz_indices, ndata);
        CoordinateT thread_coord_end = MergePathSearch(diag_end, d_row, nrow, nz_indices, ndata);

        T running_total = static_cast<T>(0);

        for (int i = 0; i < items_per_thread; ++i) {
            if (nz_indices[thread_coord.second] < d_row[thread_coord.first]) {
                running_total += d_data[thread_coord.second] * d_v[d_col[thread_coord.second]];
                ++thread_coord.second;

                d_partial[thread_coord.first] = running_total;

            } else {
                d_res[thread_coord.first] += running_total;
                d_partial[thread_coord.first] = static_cast<T>(0);

                running_total = static_cast<T>(0);
                ++thread_coord.first;
            }
        }
    }
}

// MergePathSearch function remains unchanged


__global__ void combineResults(int nrow, float *d_resS, double *d_resD, double *d_result, precision_e *precisionMap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nrow) {
        // Use the precision map to determine which result to include
        if (precisionMap[tid] == SINGLE) {
            d_result[tid] = d_resS[tid];
        } else if (precisionMap[tid] == DOUBLE) {
            d_result[tid] = d_resD[tid];
        }
    }
}

void processAndComputeUnified(CSR_Matrix<double> *mat, precision_e *p, double *vec, double *result) {
    // Step 1: Split the matrix based on precision
    CSR_Matrix<float> matS;
    CSR_Matrix<double> matD;
    split_entrywise(mat, p, &matS, &matD);

    // Allocate GPU memory for single-precision matrix
    int *d_rowS, *d_colS, *nz_indS;
    float *d_dataS;
    cudaMalloc(&d_rowS, (matS.M + 1) * sizeof(int));
    cudaMalloc(&d_colS, matS.nz * sizeof(int));
    cudaMalloc(&d_dataS, matS.nz * sizeof(float));
    cudaMemcpy(d_rowS, matS.rowptr, (matS.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colS, matS.cols, matS.nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataS, matS.vals, matS.nz * sizeof(float), cudaMemcpyHostToDevice);

    vector<int> x(matS.nz, 0);
    for (int i = 0; i < matS.nz; i++) x[i] = i;
    cudaMalloc(&nz_indS, matS.nz * sizeof(int));
    cudaMemcpy(nz_indS, x.data(), matS.nz * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate GPU memory for double-precision matrix
    int *d_rowD, *d_colD, *nz_indD;
    double *d_dataD;
    cudaMalloc(&d_rowD, (matD.M + 1) * sizeof(int));
    cudaMalloc(&d_colD, matD.nz * sizeof(int));
    cudaMalloc(&d_dataD, matD.nz * sizeof(double));
    cudaMemcpy(d_rowD, matD.rowptr, (matD.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colD, matD.cols, matD.nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataD, matD.vals, matD.nz * sizeof(double), cudaMemcpyHostToDevice);

    vector<int> y(matD.nz, 0);
    for (int i = 0; i < matD.nz; i++) y[i] = i;
    cudaMalloc(&nz_indD, matD.nz * sizeof(int));
    cudaMemcpy(nz_indD, y.data(), matD.nz * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for results
    // Ensure consistency of `d_vec` type
double *d_vec; 
cudaMalloc(&d_vec, mat->N * sizeof(double)); // Allocate as double
cudaMemcpy(d_vec, vec, mat->N * sizeof(double), cudaMemcpyHostToDevice);

// Allocate memory for single-precision results
float *d_resS, *d_partialS;
cudaMalloc(&d_resS, matS.M * sizeof(float));
cudaMalloc(&d_partialS, matS.M * sizeof(float));

// Allocate memory for double-precision results
double *d_resD, *d_partialD;
cudaMalloc(&d_resD, matD.M * sizeof(double));
cudaMalloc(&d_partialD, matD.M * sizeof(double));

// Launch kernel for single precision
int blockSize = 128;
int gridSizeS = (matS.nz + blockSize - 1) / blockSize;

CudaMergeCsrmv<float><<<gridSizeS, blockSize>>>(
    blockSize, d_rowS, d_colS, d_dataS, d_vec, d_resS, d_partialS, matS.M, matS.N, matS.nz, nz_indS);
combinePartialSum<<<(matS.M + blockSize - 1) / blockSize, blockSize>>>(matS.M, d_partialS, d_resS);

// Launch kernel for double precision
int gridSizeD = (matD.nz + blockSize - 1) / blockSize;

CudaMergeCsrmv<double><<<gridSizeD, blockSize>>>(
    blockSize, d_rowD, d_colD, d_dataD, d_vec, d_resD, d_partialD, matD.M, matD.N, matD.nz, nz_indD);
combinePartialSum<<<(matD.M + blockSize - 1) / blockSize, blockSize>>>(matD.M, d_partialD, d_resD);

double *d_result;
precision_e *d_precisionMap;
cudaMalloc(&d_result, mat->M * sizeof(double));
    cudaMalloc(&d_precisionMap, mat->nz * sizeof(precision_e));
    cudaMemcpy(d_precisionMap, p, mat->nz * sizeof(precision_e), cudaMemcpyHostToDevice);
int gridSizeResult = (mat->M + blockSize - 1) / blockSize;
    combineResults<<<gridSizeResult, blockSize>>>(mat->M, d_resS, d_resD, d_result, d_precisionMap);
    
 // Step 7: Copy results back to CPU and clean up
    cudaMemcpy(result, d_result, mat->M * sizeof(double), cudaMemcpyDeviceToHost);
    // Free GPU memory
    cudaFree(d_rowS); cudaFree(d_colS); cudaFree(d_dataS); cudaFree(nz_indS);
    cudaFree(d_rowD); cudaFree(d_colD); cudaFree(d_dataD); cudaFree(nz_indD);
    cudaFree(d_vec); cudaFree(d_resS); cudaFree(d_resD); cudaFree(d_partialS); cudaFree(d_partialD);
}



// Function to read an MTX file and convert it to CSR format
void readMatrixFromMTX(const std::string &filename, CSR_Matrix<double> &mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Skip header lines starting with '%'
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix dimensions and number of non-zeros
    std::istringstream iss(line);
    int rows, cols, nonZeros;
    iss >> rows >> cols >> nonZeros;

    mat.M = rows;
    mat.N = cols;
    mat.nz = nonZeros;

    // Temporary storage for COO format
    std::vector<int> row_indices(nonZeros);
    std::vector<int> col_indices(nonZeros);
    std::vector<double> values(nonZeros);

    for (int i = 0; i < nonZeros; ++i) {
        int r, c;
        double v;
        file >> r >> c >> v;
        row_indices[i] = r - 1; // Convert 1-based to 0-based indexing
        col_indices[i] = c - 1;
        values[i] = v;
    }
    file.close();

    // Convert COO to CSR
    mat.rowptr = (int *)malloc((rows + 1) * sizeof(int));
    mat.cols = (int *)malloc(nonZeros * sizeof(int));
    mat.vals = (double *)malloc(nonZeros * sizeof(double));

    std::fill(mat.rowptr, mat.rowptr + rows + 1, 0);

    for (int i = 0; i < nonZeros; ++i) {
        mat.rowptr[row_indices[i] + 1]++;
    }

    for (int i = 1; i <= rows; ++i) {
        mat.rowptr[i] += mat.rowptr[i - 1];
    }

    for (int i = 0; i < nonZeros; ++i) {
        int row = row_indices[i];
        int dest = mat.rowptr[row];

        mat.cols[dest] = col_indices[i];
        mat.vals[dest] = values[i];
        mat.rowptr[row]++;
    }

    for (int i = rows; i > 0; --i) {
        mat.rowptr[i] = mat.rowptr[i - 1];
    }
    mat.rowptr[0] = 0;
}


// Function to generate a random vector
void generateRandomVector(double *vec, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
}

// Perform normal CSR multiplication on CPU
void csrSpMV(const CSR_Matrix<double> &mat, const double *vec, double *result) {
    for (int i = 0; i < mat.M; ++i) {
        result[i] = 0.0;
        for (int j = mat.rowptr[i]; j < mat.rowptr[i + 1]; ++j) {
            result[i] += mat.vals[j] * vec[mat.cols[j]];
        }
    }
}

// Function to compare GPU and CPU results
void compareResults(const double *resultGPU, const double *resultCPU, int size) {
    bool match = true;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(resultGPU[i] - resultCPU[i]) > 1e-6) { // Tolerance
            match = false;
            std::cout << "Mismatch at index " << i
                      << ": GPU=" << resultGPU[i]
                      << ", CPU=" << resultCPU[i] << std::endl;
        }
    }

    if (match) {
        std::cout << "GPU and CPU results match!" << std::endl;
    } else {
        std::cout << "GPU and CPU results do not match." << std::endl;
    }
}

// Main Function
int main() {
    std::string matrixFile = "494_bus.mtx"; // Specify the MTX file name

    // Read matrix from MTX file
    CSR_Matrix<double> mat;
    readMatrixFromMTX(matrixFile, mat);

    // Initialize the precision array
    precision_e *precision = (precision_e *)malloc(mat.nz * sizeof(precision_e));
    for (int i = 0; i < mat.nz; ++i) {
        precision[i] = (mat.vals[i] >= -1 && mat.vals[i] <= 1) ? SINGLE : DOUBLE; // Alternate precision for testing
    }

    // Generate a random input vector
    double *vec = (double *)malloc(mat.N * sizeof(double));
    generateRandomVector(vec, mat.N);

    // Allocate memory for the GPU and CPU result vectors
    double *resultGPU = (double *)malloc(mat.M * sizeof(double));
    double *resultCPU = (double *)malloc(mat.M * sizeof(double));

    // Perform GPU-based matrix-vector multiplication
    processAndComputeUnified(&mat, precision, vec, resultGPU);

    // Perform CPU-based matrix-vector multiplication
    csrSpMV(mat, vec, resultCPU);

    // Compare GPU and CPU results
    compareResults(resultGPU, resultCPU, mat.M);

    // Free allocated memory
    free(mat.rowptr);
    free(mat.cols);
    free(mat.vals);
    free(precision);
    free(vec);
    free(resultGPU);
    free(resultCPU);

    return 0;
}
