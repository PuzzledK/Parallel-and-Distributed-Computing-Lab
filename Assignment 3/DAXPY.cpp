#include<mpi.h>
#include<vector>
#include<iostream>
#include<random>
#include<iomanip>

using namespace std;

void DAXPY(vector<double> &X,const vector<double> &Y,const double A,int size){
    for(int i = 0;i<size;i++){
        X[i] = A*X[i] + Y[i];
    }
}

int main(){
    cout << fixed <<setprecision(7);
    MPI_Init(NULL,NULL);

    int rank,size;
    MPI_Comm_rank((MPI_COMM_WORLD), &rank);
    MPI_Comm_size((MPI_COMM_WORLD), &size);

    int n = 1 << 16;

    double A = 25.5;
    vector<double> X;
    vector<double> Y;
    double tick,tock;

    if(rank == 0){
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<double> dist(0.0, 1.0);

        for(int i = 0;i<(1<<16);i++){
            X.push_back(dist(gen));
            Y.push_back(dist(gen));
        }
    }

    int num_per_proc = (1 << 16) / size;
    int left_out = (1 << 16) % size;

    vector<int> send_count(size),disp(size);
    for(int i = 0;i<size;i++){
        send_count[i] = num_per_proc;
        if(i < left_out) send_count[i] += 1;

        disp[i] = i == 0 ? 0 : send_count[i-1] + disp[i-1];
    }

    vector<double> local_X(send_count[rank]),local_Y(send_count[rank]);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(X.data(),send_count.data(),disp.data(),MPI_INT,local_X.data(),send_count[rank],MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatterv(Y.data(),send_count.data(),disp.data(),MPI_INT,local_Y.data(),send_count[rank],MPI_INT,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) tick = MPI_Wtime();
    DAXPY(local_X,local_Y,A,send_count[rank]);
    if(rank == 0) tock = MPI_Wtime();

    if(rank == 0){
        cout<<"Parrallel execution time -> "<<tock-tick<<endl;

        tick = MPI_Wtime();
        DAXPY(X,Y,A,(1 << 16));
        tock = MPI_Wtime();

        cout<<"Serial execution time -> "<<tock-tick<<endl;
    }

    MPI_Finalize();
    return 0;
}
