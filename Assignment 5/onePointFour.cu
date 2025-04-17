#include<iostream>
#include<cuda.h>
using namespace std;

#define threads_per_block 256
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
const size_t N = 100000000;


inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void Q1(float* a,float* b,float* c,size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){
        c[i] = a[i] + b[i];
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

    float *a_d,*b_d,*c_d;
    cudaCheck(cudaMalloc(&a_d, N * sizeof(float)));
    cudaCheck(cudaMalloc(&b_d, N * sizeof(float)));
    cudaCheck(cudaMalloc(&c_d, N * sizeof(float)));

    cudaCheck(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));

    Q1<<<blocks,threads>>>(a_d,b_d,c_d,N);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    cudaCheck(cudaEventElapsedTime(&ms,start,stop));
    cudaCheck(cudaMemcpy(c, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));


    float bytes = (N * (2 * sizeof(float) + sizeof(float))) / (1e9f);

    cout<<"KERNEL EXECUTION TIME -> "<<ms<<" milliseconds"<<endl;

    cout<<"BANDWIDTH -> "<< bytes / (ms/1000.0f)<<" GB/s"<<endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    delete[] a;
    delete[] b;
    delete[] c;
}