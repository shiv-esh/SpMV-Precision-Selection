//%%writefile csrmv.cu
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <thrust/pair.h>

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

__global__ void combinePartialSum(int nrow,double *d_partial,double *d_res) {
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < nrow){
    printf("%d: %lf %lf\n",tid,d_res[tid],d_partial[tid]);
    d_res[tid] +=d_partial[tid];
  }
}

__global__ void CudaMergeCsrmv(int n_threads,int *d_row, int *d_col,double *d_data,int *d_v,double *d_res,double *d_partial,int nrow,int ncol,int ndata,int *nz_indices) {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < (ndata+nrow)) {
       
        int n_items = nrow + ndata;
        int items_per_thread = (n_items+n_threads-1)/n_threads;

        int diag_start = min(items_per_thread*tid,n_items);
        int diag_end = min(diag_start+items_per_thread,n_items);
        CoordinateT thread_coord = MergePathSearch(diag_start,d_row,nrow,nz_indices,ndata);
        CoordinateT thread_coord_end = MergePathSearch(diag_end,d_row,nrow,nz_indices,ndata);

        double running_total = 0.0;
        running_total = 0.0;


       for (int i = 0; i < items_per_thread; ++i) {
            if (nz_indices[thread_coord.second] < d_row[thread_coord.first]) {

                running_total += d_data[thread_coord.second] * d_v[d_col[thread_coord.second]];
                ++thread_coord.second;

                  d_partial[thread_coord.first] = running_total;


            } else {


                d_res[thread_coord.first] += running_total;
                d_partial[thread_coord.first]=0.0;


                running_total = 0.0;
                ++thread_coord.first;
              }

            }





        /*for (; thread_coord.second < thread_coord_end.second; ++thread_coord.second) {
            running_total += d_data[thread_coord.second] * d_v[d_col[thread_coord.second]];
        }
        d_partial[thread_coord.first] = running_total;*/
        /*printf("%lf ",d_partial[thread_coord.first]);
        if(d_partial[thread_coord_end.first] != 0) {
          d_res[thread_coord_end.first] = d_partial[thread_coord_end.first];
        }*/


    }
}
using namespace std;
void multiplication_seq(vector<int>&csr_row,vector<int>&csr_col,vector<double>&csr_data,vector<int>&v,vector<double>&res)
{
    int n = csr_row.size();
    for(int i = 0; i < n-1; i++) {
        res[i] = 0;
        for(int k = csr_row[i]; k < csr_row[i+1];k++) {
            res[i] += csr_data[k]*v[csr_col[k]];
        }


    }
}
int main() {
   ifstream vin("vec.txt");
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

    ifstream fin("494_bus.mtx");
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

    int *d_row, *d_col,*d_v,*nz_indices;
    double *d_data, *d_res,*d_partial;
    vector<int> x(ndata,0);
    for(int i = 0; i < ndata;i++) {
      x[i] = i;
    }

    cudaMalloc(&d_row,(nrow+1)*sizeof(int));
    cudaMalloc(&d_col,ndata*sizeof(int));
    cudaMalloc(&d_data,ndata*sizeof(double));
    cudaMalloc(&d_v,ncol*sizeof(int));
    cudaMalloc(&d_res,nrow*sizeof(double));
    cudaMalloc(&nz_indices,ndata*sizeof(int));



    cudaMemcpy(d_row,csr_row.data(),(nrow+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,csr_col.data(),(ndata)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,csr_data.data(),(ndata)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,vec.data(),(ncol)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(nz_indices,x.data(),(ndata)*sizeof(int),cudaMemcpyHostToDevice);

    int n_threads = 128;
    int n_blocks = (nrow + ndata+n_threads-1)/n_threads;
    cudaMalloc(&d_partial,nrow*sizeof(double));
    //CudaMergeCsrmv<<<1,n_threads>>>(n_threads,d_row+1,d_col,d_data,d_v,d_res,d_partial,nrow,ncol,ndata);
    CudaMergeCsrmv<<<n_blocks,n_threads>>>(n_threads,d_row+1,d_col,d_data,d_v,d_res,d_partial,nrow,ncol,ndata,nz_indices);
    combinePartialSum<<<1,nrow>>>(nrow,d_partial,d_res);
    vector<double>res_seq(nrow);
    multiplication_seq(csr_row,csr_col,csr_data,vec,res_seq);


    vector<double>res(nrow);
    cudaMemcpy(res.data(),d_res,nrow*sizeof(double),cudaMemcpyDeviceToHost);
    cout<<endl;
    for(int i = 0; i < nrow; i++){
      cout<<res[i]<<" "<<res_seq[i]<<" "<<endl;


    }

    return 0;
}