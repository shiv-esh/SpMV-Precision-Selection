#include <fstream>
#include <vector>
#include <random>
#include<iostream>
#include<sstream>
using namespace std;

typedef pair<int, int> IndexPair;

int main() {
    
    vector<pair<IndexPair, double>> matrix;

    int nrow,ncol;
    cin >>nrow>>ncol;
    // Fill the matrix with some values
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            if (distribution(generator) < 0.2) {  // For example, let's fill the diagonal with ones
                matrix.push_back({{i, j}, distribution(generator)});
            }
        }
    }

   
    ofstream file("sparse_matrix.mtx");
    file<<nrow<<" "<<ncol<<" "<<matrix.size()<<'\n';
    // Write the matrix to the file in MTX format
    for (const auto& entry : matrix) {
       
        file << entry.first.first + 1 << " " << entry.first.second + 1 << " " << entry.second << "\n";
    }

    file.close();
    cout << "Sparse matrix has been written to 'sparse_matrix.mtx'." << endl;
    ofstream vec("vec.txt");

    uniform_int_distribution<int> int_distribution(1,100);
    vector<int> v(ncol);
    for (int& i : v) {
        i = int_distribution(generator);
    }

    for(const int& i:v){
        vec<<i<<", ";
    }
    vec.close();
    cout << "Vector has been written to 'vector.txt'." << endl;
    ifstream vin("vec.txt"); 
    
    string line;
    vector<int>v1;
    while(!vin.eof())
    {
        string nstr;
        while(getline(vin,nstr,',')) {
            stringstream ss;
            int num;
            ss << nstr;
            ss >> num;
            v1.push_back(num);
            //cout<<num<<endl;
        }
    }
    vin.close();
    
    int vsize = v1.size();
    //cout<<"\n"<<vsize<<endl;

    
    return 0;
}
