#include<iostream>
#include<mpi.h>
#include<timer.hpp>

#define dims 70

using namespace std;

void matMul(vector<double> a,vector<double> b,vector<double> &c,int rank,int size){
    for(int i = rank;i<dims;i+=(size-1)){
        for(int j = 0;j<dims;j++){
            c[i*dims + j] = 0;
            for(int k = 0;k<dims;k++){
                c[i*dims + j] += a[i*dims + k]*b[k*dims + j];
            }
        }
    }
}

int main(){
    vector<double> a(dims*dims);
    vector<double> b(dims*dims);

    MPI_Init(NULL,NULL);

    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank == 0){
        MPI_Bcast((void *) a.data(),a.size(),MPI_BYTE,0,MPI_COMM_WORLD);
        MPI_Bcast((void *) b.data(),b.size(),MPI_BYTE,0,MPI_COMM_WORLD);
    }
    

}