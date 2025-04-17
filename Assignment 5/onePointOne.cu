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

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    dim3 threads(threads_per_block,1,1);
    dim3 blocks((N + threads_per_block - 1) / threads_per_block,1,1);

    cudaCheck(cudaMemcpyToSymbol(a_d, a, N * sizeof(float)));
    cudaCheck(cudaMemcpyToSymbol(b_d, b, N * sizeof(float)));


    Q1<<<blocks,threads>>>(N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpyFromSymbol(c, c_d, N * sizeof(float)));

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    delete[] a;
    delete[] b;
    delete[] c;
}