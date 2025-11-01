#ifndef LOAD_CSR_H
#define LOAD_CSR_H
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "mmio.c" // Assuming you have mmio.h and mmio.c for reading Matrix Market files

// Define the CSR matrix structure
struct csr_matrix_t {
    int nrows;      // Number of rows
    int ncols;      // Number of columns
    int nnz;        // Number of non-zero elements
    int* row_ptr;   // Row pointers (size nrows + 1)
    int* col_idx;   // Column indices (size nnz)
    float* values;  // Non-zero values (size nnz)
};

csr_matrix_t* load_csr_matrix(const char* filename) {
    // Open the Matrix Market file
    FILE* f;
    if ((f = fopen(filename, "r")) == NULL) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read the Matrix Market banner
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        std::cerr << "Could not process Matrix Market banner." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Check if the matrix is in the expected format
    if (mm_is_complex(matcode) || !mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        std::cerr << "Unsupported Matrix Market format: only real, sparse matrices are supported." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read the matrix size and number of non-zero elements
    int nrows, ncols, nnz;
    if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0) {
        std::cerr << "Could not read matrix size." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate space for the COO format data
    std::vector<int> row_indices(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<float> values(nnz);

    // Read the matrix data
    for (int i = 0; i < nnz; i++) {
        int row, col;
        double value;  // Use double for reading, then cast to float

        if (fscanf(f, "%d %d %lf\n", &row, &col, &value) != 3) {
            std::cerr << "Error reading matrix data." << std::endl;
            exit(EXIT_FAILURE);
        }

        row_indices[i] = row - 1; // Convert to zero-based indexing
        col_indices[i] = col - 1; // Convert to zero-based indexing
        values[i] = (float)value; // Store the value as a float
    }

    fclose(f); // Close the file

    // Allocate the CSR data structures
    csr_matrix_t* csr_matrix = new csr_matrix_t;
    csr_matrix->nrows = nrows;
    csr_matrix->ncols = ncols;
    csr_matrix->nnz = nnz;
    csr_matrix->row_ptr = new int[nrows + 1];
    csr_matrix->col_idx = new int[nnz];
    csr_matrix->values = new float[nnz];

    // Initialize row_ptr with zeros
    for (int i = 0; i <= nrows; i++) {
        csr_matrix->row_ptr[i] = 0;
    }

    // Count the number of elements in each row
    for (int i = 0; i < nnz; i++) {
        csr_matrix->row_ptr[row_indices[i] + 1]++;
    }

    // Cumulative sum to get row_ptr
    for (int i = 0; i < nrows; i++) {
        csr_matrix->row_ptr[i + 1] += csr_matrix->row_ptr[i];
    }

    // Fill col_idx and values
    std::vector<int> current_row(nrows, 0);
    for (int i = 0; i < nnz; i++) {
        int row = row_indices[i];
        int dest = csr_matrix->row_ptr[row] + current_row[row];
        csr_matrix->col_idx[dest] = col_indices[i];
        csr_matrix->values[dest] = values[i];
        current_row[row]++;
    }

    return csr_matrix;
}

void free_csr_matrix(csr_matrix_t* csr_matrix) {
    delete[] csr_matrix->row_ptr;
    delete[] csr_matrix->col_idx;
    delete[] csr_matrix->values;
    delete csr_matrix;
}
#endif