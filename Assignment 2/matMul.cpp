#include<iostream>
#include<mpi.h>
#include<timer.hpp>
#include<vector>

using namespace std;

void matMul(vector<double> &a,vector<double> &b,vector<double> &c,int rows_per_process,int dims){
    for(int i = 0;i<rows_per_process;i++){
        for(int j = 0;j<dims;j++){
            c[i*dims + j] = 0;
            for(int k = 0;k<dims;k++){
                c[i*dims + j] += a[i*dims + k]*b[k*dims + j];
            }
        }
    }
}

int main(int argc, char** argv){

    MPI_Init(NULL,NULL);
    Timer t;


    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(argc < 2){
        cout<<(rank==0 ? "INPUT NUMBER OF MATRIX ROWS IN EXECUTION ARGUMENT\n" : "");
        MPI_Finalize();
        return 1;
    }

    int dims = atoi(argv[1]);

    vector<double> a,b(dims*dims),c;
    vector<double> local_a,local_c;
    int rows_per_process = dims/size;
    int leftout_rows = dims % size;

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

    vector<int> send_count(size);
    vector<int> displacement(size);
    for(int i = 0;i<size;i++){
        send_count[i] = rows_per_process * dims;
        if(i < leftout_rows){
            send_count[i] += dims;
        }

        displacement[i] = i == 0 ? 0 : (displacement[i-1] + send_count[i-1]);
    }

    int local_rows = send_count[rank] / dims;
    local_a.resize(send_count[rank]);
    local_c.resize(send_count[rank]);

    MPI_Scatterv(a.data(),send_count.data(),displacement.data(),MPI_DOUBLE,local_a.data(),send_count[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Bcast(b.data(),b.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) t.tick();

    matMul(local_a,b,local_c,local_rows,dims);

    MPI_Barrier(MPI_COMM_WORLD);    
    if(rank == 0) t.tock();
    MPI_Gatherv(local_c.data(),send_count[rank],MPI_DOUBLE,c.data(),send_count.data(),displacement.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    if(rank == 0){
        cout<<"PARTIAL COMPUTATION TIME -> "<<t.time()<<endl;
        vector<double> c_local(dims*dims);
        t.tick();
        matMul(a,b,c_local,dims,dims);
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