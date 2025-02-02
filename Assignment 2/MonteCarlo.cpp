#include<iostream>
#include<mpi.h>
#include<timer.hpp>

void monte_carlo(long num_points,long &insideCircle,int rank,int world_size){
    srand(time(NULL) * rank);
    for(int i = rank;i<num_points;i+=world_size){
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if(x*x + y*y <= 1){
            insideCircle++;
        }
    }
}

int main(int argc,char** argv){
    Timer t;

    MPI_Init(NULL,NULL);

    if(argc < 2){
        std::cerr<<"NOT ENOUGH ARGUMENTS\n";
        return -1;
    }

    long num_points = atol(argv[1]);
    long insideCircleLocal = 0;
    long insideCircle;

    int rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    MPI_Barrier(MPI_COMM_WORLD); 
    if(rank == 0){
        t.tick();
    }

    monte_carlo(num_points,insideCircleLocal,rank,world_size);

    MPI_Reduce(&insideCircleLocal, &insideCircle, 1, MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);

    if(rank == 0){
        t.tock();
    }

    if(rank == 0){
        std::cout << "Estimated Pi: " << 4.0 * (double)insideCircle / num_points << std::endl;
        std::cout << "Time Taken: " << t.time() << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;

}