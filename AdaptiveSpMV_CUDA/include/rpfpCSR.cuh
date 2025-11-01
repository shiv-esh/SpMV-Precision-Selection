#ifndef RPFP_CSR_H
#define RPFP_CSR_H
#include <cuda.h>

#include <cuda_fp16.h>

struct rpfpCSR {
    int n;
    int nnz;

    int* ia8;     uint8_t* a8;      int* ja8;
    int* ia16;    __half* a16f;     int* ja16f;
    int* ia32;    float* a32;       int* ja32;
    int* ia64;    double* a64;      int* ja64;
};


#endif