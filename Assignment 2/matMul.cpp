#include<iostream>
#include<mpi.h>
#include<timer.hpp>
#include<vector>

#define dims 70

using namespace std;

void matMul(vector<double> &a,vector<double> &b,vector<double> &c,int rows_per_process,int rank,int size){

    int n = rows_per_process;
    if(rank == size - 1){
        n += (size%rank);
    }
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
    Timer t;

    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    vector<double> a,b(dims*dims),c;
    vector<double> local_a,local_c;
    int rows_per_process = dims/size;

    if(rank == 0){
        srand(time(NULL));
        a.resize(dims*dims);
        c.resize(dims*dims);

        for(int i = 0;i<dims;i++){
            for(int j = 0;j<dims;j++){
                a[i*dims + j] = (double)rand();
                b[i*dims + j] = (double)rand();
            }
        }
    }

    local_a.resize(rows_per_process * dims);
    local_c.resize(rows_per_process * dims);
    MPI_Scatter(a.data(),rows_per_process * dims,MPI_DOUBLE,local_a.data(),rows_per_process*dims,MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Bcast(b.data(),b.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) t.tick();

    matMul(local_a,b,local_c,rows_per_process,rank,size);

    MPI_Barrier(MPI_COMM_WORLD);    
    if(rank == 0) t.tock();
    MPI_Gather(local_c.data(),rows_per_process*dims,MPI_DOUBLE,c.data(),rows_per_process*dims,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    if(rank == 0){
        cout<<"PARTIAL COMPUTATION TIME -> "<<t.time()<<endl;
        vector<double> c_local(dims*dims);
        t.tick();
        matMul(a,b,c_local,dims,rank,size);
        t.tock();
        cout<<"SERIAL COMPUTATION TIME -> "<<t.time()<<endl;

        for(int i = 0;i<dims;i++){
            int bruh = false;
            for(int j = 0;j<dims;j++){
                if(c_local[i*dims + j] != c[i*dims + j]){
                    cout<<"IDHAR GALAT HAI BAWA -> "<<i<<" "<<j<<endl;
                    bruh = true;
                    break;
                }
            }
            if(bruh) break;
        }

    }

    MPI_Finalize();

}