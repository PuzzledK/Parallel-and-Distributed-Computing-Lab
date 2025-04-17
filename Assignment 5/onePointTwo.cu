#include<iostream>
#include<cuda.h>
using namespace std;

#define threads_per_block 2
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
const size_t N = 1000000;

__device__ float a_d[N],b_d[N],c_d[N];

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void Q1(size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){
        c_d[i] = a_d[i] + b_d[i];
    }
}

int main(){

    cudaEvent_t start,stop;
    float ms = 0;

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    dim3 threads(threads_per_block,1,1);
    dim3 blocks((N + threads_per_block - 1) / threads_per_block,1,1);

    cudaCheck(cudaMemcpyToSymbol(a_d, a, N * sizeof(float)));
    cudaCheck(cudaMemcpyToSymbol(b_d, b, N * sizeof(float)));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));

    Q1<<<blocks,threads>>>(N);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&ms,start,stop);
    cudaCheck(cudaMemcpyFromSymbol(c, c_d, N * sizeof(float)));


    cout<<"KERNEL EXECUTION TIME -> "<<ms<<" milliseconds"<<endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] a;
    delete[] b;
    delete[] c;
}