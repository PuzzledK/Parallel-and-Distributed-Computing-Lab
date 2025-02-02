#include<iostream>
#include<vector>
#include<mpi.h>

void divide_domains(int domain_size,int rank,int world_size,int &start,int &size){
    if(world_size > domain_size){
        MPI_Abort(MPI_COMM_WORLD,0);
        return;
    } 

    start = (domain_size / world_size) * rank;
    size = (domain_size / world_size);

    if(rank == world_size - 1){
        size += (domain_size % world_size);
    }

}

typedef struct Walker
{
    int location;
    int steps_left;
};

void initialize_walkers(int num_walkers,int max_walk,int start,int size,std::vector<Walker> &incoming_walkers){
    Walker walker;
    for(int i = 0;i<num_walkers;i++){
        walker.location = start;
        walker.steps_left = (rand()) % max_walk;

        incoming_walkers.push_back(walker);
    }
}

void walk(Walker &walker,int start,int size,int domain_size,std::vector<Walker> &outgoing_walkers){
    while(walker.steps_left > 0){
        if(walker.location == start + size){
            if(walker.location == domain_size){
                walker.location = 0;
            }

            outgoing_walkers.push_back(walker);
            break;
        }

        else{
            walker.location ++;
            walker.steps_left --;
        }

    }
}

void send_outgoing_walker(std::vector<Walker> &outgoing_walkers,int rank,int world_size){
    MPI_Send((void*) outgoing_walkers.data(),outgoing_walkers.size(),MPI_BYTE,(rank+1)%world_size,0,MPI_COMM_WORLD);

    outgoing_walkers.clear();
}

void receive_incoming_walkers(std::vector<Walker> &incoming_walkers,int rank,int world_size){
    MPI_Status status;

    int prev_rank = (rank == 0) ? world_size - 1 : rank - 1;

    MPI_Probe(prev_rank,0,MPI_COMM_WORLD,&status);

    int size_recv;
    MPI_Get_count(&status,MPI_BYTE,&size_recv);

    incoming_walkers.resize((size_recv)/sizeof(Walker));

    MPI_Recv((void*) incoming_walkers.data(),size_recv,MPI_BYTE,prev_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}

int main(int argc,char ** argv){
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int qno = atoi(argv[1]);
    switch(qno){
        case 1:
            //BASIC SENDING RECEIVING WITH MPI_Send and MPI_Recv
            int num;
            if(rank == 0){
                num = 56;
                MPI_Send(&num,1,MPI_INT,6,0,MPI_COMM_WORLD);
            }
            if(rank == 6){
                MPI_Recv(&num,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                std::cout<<"Received "<<num<<" FROM "<<0<<std::endl;
            }

            break;
        case 2:
            //RING PASSING 
            int token;
            if(rank == 0){
                token = -10;
            }
            else{
                MPI_Recv(&token,1,MPI_INT,(rank - 1),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                std::cout<<"PROCESS "<<rank<<" RECEIVED" <<token<<" FROM "<<rank-1<<std::endl;
            }

            MPI_Send(&token,1,MPI_INT,(rank+1)%size,0,MPI_COMM_WORLD);

            if(rank == 0){
                MPI_Recv(&token,1,MPI_INT,size -1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                std::cout<<"PROCESS "<<rank<<" RECEIVED" <<token<<" FROM "<<size-1<<std::endl;
            }

            break;
        case 3:
            //MPI_Probe and MPI_Status
            if(rank == 0){
                int max_nums = 100;
                int numbers[max_nums];

                srand(time(NULL));
                int to_send = int(rand())%max_nums;

                MPI_Send(numbers,to_send,MPI_INT,1,0,MPI_COMM_WORLD);

                std::cout<<"SENT "<<to_send<<" NUMBERS TO 1\n";
            }

            if(rank == 1){
                MPI_Status status;
                MPI_Probe(0,0,MPI_COMM_WORLD,&status);

                int to_recv;
                MPI_Get_count(&status,MPI_INT,&to_recv);

                int* numbers = new int[to_recv];

                MPI_Recv(numbers,to_recv,MPI_INT,0,0,MPI_COMM_WORLD,&status);

                std::cout<<"RECEIVED "<<to_recv<<" NUMBERS FROM 0\n"; 

            }
            break;
        default:
            if(rank==0) std::cout<<"DOES NOT EXIST\n";
            break;
    }

    MPI_Finalize();

    return 0;
}