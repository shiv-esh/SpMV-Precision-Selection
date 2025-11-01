#include <cuda_runtime.h>
#include <iostream>

// Define a pair of integers to represent a 2D coordinate
typedef thrust::pair<int, int> CoordinateT;

__device__ CoordinateT MergePathSearch(int diagonal, int* a, int a_len, int* b, int b_len) {
    // Diagonal search range (in x coordinate space)
    int x_min = max(diagonal - b_len, 0);
    int x_max = min(diagonal, a_len);

    // 2D binary-search along the diagonal search range
    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;
        if (a[pivot] <= b[diagonal - pivot - 1]) {
            // Keep top-right half of diagonal range
            x_min = pivot + 1;
        } else {
            // Keep bottom-left half of diagonal range
            x_max = pivot;
        }
    }

    return CoordinateT(min(x_min, a_len), diagonal - x_min);
}

__global__ void CudaMergeCsrmv(int num_threads, const CsrMatrix A, double* x, double* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_threads) {
        int* row_end_offsets = A.row_offsets + 1;
        int num_merge_items = A.num_rows + A.num_nonzeros;
        int items_per_thread = (num_merge_items + num_threads - 1) / num_threads;

        int diagonal = min(items_per_thread * tid, num_merge_items);
        int diagonal_end = min(diagonal + items_per_thread, num_merge_items);
        CoordinateT thread_coord = MergePathSearch(diagonal, row_end_offsets, nz_indices, A.num_rows, A.num_nonzeros);
        CoordinateT thread_coord_end = MergePathSearch(diagonal_end, row_end_offsets, nz_indices, A.num_rows, A.num_nonzeros);

        double running_total = 0.0;
        // for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x) {
        //     for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y) {
        //         running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]];
        //     }
        //     y[thread_coord.x] = running_total;
        //     running_total = 0.0;
        // }
        for (int i = 0; i < items_per_thread; ++i) {
            if (thread_coord.y < row_end_offsets[thread_coord.x]) {
                running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]];
                ++thread_coord.y;
            } else {
                y[thread_coord.x] = running_total;
                running_total = 0.0;
                ++thread_coord.x;
            }
        }

        for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y) {
            running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]];
        }

        if (tid < num_threads - 1 && thread_coord_end.x < A.num_rows) {
            atomicAdd(&y[thread_coord_end.x], running_total);
        }
    }
}

int main() {
   ifstream vin("vector.txt"); 
    vector<int> vec;
    string line;

    while(!vin.eof())
    {
        string nstr;
        while(getline(vin,nstr,',')) {
            stringstream ss;
            int num;
            ss << nstr;
            ss >> num;
            vec.push_back(num);
        }
    }
    vin.close();
    
    int vsize = vec.size();

    ifstream fin("inputfile.mtx"); 
    //string line;
    int nrow,ncol,ndata;

    
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    
    fin >> nrow >> ncol >> ndata;
    
    
    vector<int> row(ndata),col(ndata);
    vector<double> data(ndata);
    
    
    for(int i = 0; i < ndata; i++) {
        int r,c;
        double val;
        fin >> r >> c >> val;
        row[i] = r-1;
        col[i] = c-1;
        data[i] = val;
    }

    fin.close();
    vector<int> csr_row(nrow+1,0);
    vector<int> csr_col(ndata);
    vector<double> csr_data(ndata);

    for(int i = 0; i < ndata; i++) {
        csr_row[row[i]+1]++;
    }

    for (int i = 1; i <= nrow; i++) {
        csr_row[i] += csr_row[i - 1];
    }

    for(int i = 0; i < ndata; i++) {
        int r = row[i];
        int x = csr_row[r];
        csr_col[x] = col[i];
        csr_data[x] = data[i];
        csr_row[r]++;
    }
    for(int i = nrow; i > 0; i--) {
        csr_row[i] = csr_row[i-1];
    }
    csr_row[0] = 0;
    
    int *d_row, *d_col,*d_v;
    double *d_data, *d_res;

    cudaMalloc(&d_row,(nrow+1)*sizeof(int));
    cudaMalloc(&d_col,ndata*sizeof(int));
    cudaMalloc(&d_data,ndata*sizeof(double));
    cudaMalloc(&d_v,ncol*sizeof(double));
    cudaMalloc(&d_res,nrow*sizeof(double));

    cudaMemcpy(d_row,csr_row.data(),(nrow+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,csr_col.data(),(ndata)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,csr_data.data(),(ndata)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,vec.data(),(ncol)*sizeof(int),cudaMemcpyHostToDevice);

    vector<double>res(nrow);

    return 0;
}
