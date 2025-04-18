#include<cuda.h>
#include<iostream>
#include<random>
#include<timer.hpp>

using namespace std;

#define N 1000
#define chunk_size 256
#define threads_per_block 128

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void generate_random_array(float *arr) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1000.0f);
    
    for (int i = 0; i < N; ++i) {
        arr[i] = dis(gen);
    }

}

__device__ void merge(float *arr,int low,int mid,int high){
    float* temp = new float[high - low + 1];
    int l = low;
    int r = mid + 1;
    int i = 0;

    while(l <= mid && r <= high){
        if(arr[l] <= arr[r]){
            temp[i] = arr[l++];
        }
        else{
            temp[i] = arr[r++];
        }

        i++;
    }

    while(l <= mid){
        temp[i++] = arr[l++];
    }

    while(r <= high){
        temp[i++] = arr[r++];
    }

    for(int i = low;i<=high;i++){
        arr[i] = temp[i - low]; 
    }

    delete[] temp;

}

__global__ void final_merge_kernel(float* arr, int n, int chunkSize) {
    for (int curr_size = chunkSize; curr_size < n; curr_size *= 2) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int left = tid * 2 * curr_size;
        
        if (left >= n) continue;
        
        int mid = min(left + curr_size - 1, n - 1);
        int right = min(left + 2 * curr_size - 1, n - 1);
        
        merge(arr, left, mid, right);
        
        __syncthreads();
    }
}

__global__ void divideCuda(float* arr,int n,int chunkSize){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int start = id * chunkSize;

    if(start > n) return;

    int end = min(start + chunkSize - 1, n - 1);

    for(int curr_size = 1;curr_size <= end - start;curr_size *= 2){
        for(int left = start;left < end;left += 2*curr_size){
            int mid = min(left + curr_size - 1, end);
            int right = min(left + 2*curr_size -1,end);

            merge(arr,left,mid,right);
        }
    }
}

int main(){
    Timer t;
    float arr[N];
    generate_random_array(arr);

    float* arr_d;
    cudaCheck(cudaMalloc(&arr_d,N*sizeof(float)));


    int num_chunks = (N + chunk_size - 1) / chunk_size;

    dim3 threads(threads_per_block,1,1);
    dim3 blocks((num_chunks + threads_per_block - 1 ) / threads_per_block,1,1);

    t.tick();

    cudaCheck(cudaMemcpy(arr_d,arr,N*sizeof(float),cudaMemcpyHostToDevice));

    divideCuda<<<blocks,threads>>>(arr_d,N,chunk_size);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    final_merge_kernel<<<1,256>>>(arr_d,N,chunk_size);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();
    
    cudaCheck(cudaMemcpy(arr,arr_d,N*sizeof(float),cudaMemcpyDeviceToHost));

    t.tock();

    cout<<"TIME TAKEN -> "<<t.time()<<endl;

    for (int i = 0; i < N - 1; ++i) {
        if (arr[i] > arr[i+1]) {
            cout << "Sorting failed at index " << i << endl;
            break;
        }
    }
    
    cudaFree(arr_d);

}