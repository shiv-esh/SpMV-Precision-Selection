%%writefile SpMV.cu
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <chrono>
using namespace std;


__global__ void multiplication_kernel(int n, int *d_row, int *d_col, double *d_data, int *d_v, double *d_res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        double val = 0;
        for (int i = d_row[tid]; i < d_row[tid + 1]; i++) {
            val += d_data[i] * d_v[d_col[i]];
        }
        d_res[tid] = val;
    }
}

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
int main()
{

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
    int m,n,l;

    
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    
    fin >> m >> n >> l;
    
    
    vector<int> row(l),col(l);
    vector<double> data(l);
    
    
    for(int i = 0; i < l; i++) {
        int r,c;
        double val;
        fin >> r >> c >> val;
        row[i] = r-1;
        col[i] = c-1;
        data[i] = val;
    }

    fin.close();
    vector<int> csr_row(m+1,0);
    vector<int> csr_col(l);
    vector<double> csr_data(l);

    for(int i = 0; i < l; i++) {
        csr_row[row[i]+1]++;
    }

    for (int i = 1; i <= m; i++) {
        csr_row[i] += csr_row[i - 1];
    }

    for(int i = 0; i < l; i++) {
        int r = row[i];
        int x = csr_row[r];
        csr_col[x] = col[i];
        csr_data[x] = data[i];
        csr_row[r]++;
    }
    for(int i = m; i > 0; i--) {
        csr_row[i] = csr_row[i-1];
    }
    csr_row[0] = 0;
    
    int *d_row, *d_col,*d_v;
    double *d_data, *d_res;

    cudaMalloc(&d_row,(m+1)*sizeof(int));
    cudaMalloc(&d_col,l*sizeof(int));
    cudaMalloc(&d_data,l*sizeof(double));
    cudaMalloc(&d_v,n*sizeof(double));
    cudaMalloc(&d_res,m*sizeof(double));

    cudaMemcpy(d_row,csr_row.data(),(m+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,csr_col.data(),(l)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,csr_data.data(),(l)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,vec.data(),(n)*sizeof(int),cudaMemcpyHostToDevice);

    vector<double>res(m);
    auto start1 = chrono::high_resolution_clock::now();
    multiplication_seq(csr_row,csr_col,csr_data,vec,res);
    auto stop1 = chrono::high_resolution_clock::now();
    auto dur1 = chrono::duration_cast<chrono::microseconds>(stop1-start1);

    cout<<"Sequential Code: "<<dur1.count()<<"ms"<<endl;

    vector<double> res_k(m);
    for(int i = 4; i < m; i=i*2){
        int block_size = i;
        int grid_size = (m + block_size-1)/block_size;

        

        auto start = chrono::high_resolution_clock::now();
        multiplication_kernel<<<grid_size,block_size>>>(m,d_row,d_col,d_data,d_v,d_res);
        auto stop = chrono::high_resolution_clock::now();
        auto dur = chrono::duration_cast<chrono::microseconds>(stop-start);

        
        cout<<"Kernel with block size "<<block_size<<": "<<dur.count()<<"ms";
        
        cudaMemcpy(res_k.data(),d_res,m*sizeof(double),cudaMemcpyDeviceToHost);
        
        int flag = 0;

        for(int i = 0; i < m;i++) {
            //cout<<res[i]<<" "<<res_k[i]<<" "<<i<<endl;
            if(res[i] != res_k[i]) {
                flag = 1;
                break;
        
            }
        }
        if(flag)
            cout<<" Error "<<endl;
        else
            cout<<" No error "<<endl;
    }
    return 0;
}
