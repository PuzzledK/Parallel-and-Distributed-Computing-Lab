#include<iostream>
#include<mpi.h>
#include<timer.hpp>
#include<vector>

#define dims 70

using namespace std;

void matMul(vector<double> &a,vector<double> &b,vector<double> &c,int rows_per_process){
    for(int i = 0;i<rows_per_process;i++){
        for(int j = 0;j<dims;j++){
            c[i*dims + j] = 0;
            for(int k = 0;k<dims;k++){
                c[i*dims + j] += a[i*dims + k]*b[k*dims + j];
            }
        }
    }
}

int main(){

    MPI_Init(NULL,NULL);

    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    vector<double> a,b,c;
    vector<double> local_a,local_c;
    int rows_per_process = dims/size;

    if(rank == 0){
        a.resize(dims*dims);
        b.resize(dims*dims);
        c.resize(dims*dims);

        for(int i = 0;i<dims;i++){
            for(int j = 0;j<dims;j++){
                a[i*dims + j] = (double) rand();
                b[i*dims + j] = (double) rand();
            }
        }
    }

    local_a.resize(rows_per_process * dims);
    local_c.resize(rows_per_process * dims);
    MPI_Scatter((void *)a.data(),rows_per_process * dims,MPI_BYTE,(void*)local_a.data(),rows_per_process*dims,MPI_BYTE,0,MPI_COMM_WORLD);

    MPI_Bcast((void*)b.data(),b.size(),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Recv((void*)b.data(),b.size(),MPI_BYTE,0,MPI_ANY_TAG,MPI_COMM_WORLD,NULL);  

    matMul(local_a,b,local_c,rows_per_process);

    MPI_Gather((void*)local_c.data(),rows_per_process*dims,MPI_BYTE,(void*)c.data(),rows_per_process*dims,MPI_BYTE,0,MPI_COMM_WORLD);

    if(rank == 0){
        cout<<"OUTPUT MATRIX\n";
        for(int i = 0;i<dims;i++){
            for(int j = 0;j<dims;j++){
                cout<<c[i*dims + j]<<" ";
            }
            cout<<endl;
        }
    }

}