#include<cuda.h>
#include<iostream>
using namespace std;

#define N 1024
#define block_dim 32

__global__ void add_iter(int *a,int *sum){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%d\n",idx);
    if(idx < N){
        atomicAdd(sum,idx+1);
    }
}

__global__ void add_direct(int* a,int *b){
    
}

int main(){
    int a[N];
    for(int i = 0;i<N;i++){
        a[i] = i + 1;
    }

    int *a_d,*b_d;
    cudaMalloc(&a_d,N*sizeof(int));
    cudaMalloc(&b_d,N*sizeof(int));

    cudaMemcpy(a_d,a,N*sizeof(int),cudaMemcpyHostToDevice);
    dim3 block(block_dim,1,1);
    dim3 grid((N + block_dim) / (block_dim+1),1,1);

    int* sum;
    cudaMallocManaged(&sum,sizeof(int));

    add_iter<<<grid,block>>>(a_d,sum);
    cudaDeviceSynchronize();

    cout<<*sum<<endl;
    cout<<(*sum == ((N*(N+1)/2)) ? "Correct" : "Wrong")<<endl;

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(sum);


    return 0;
}