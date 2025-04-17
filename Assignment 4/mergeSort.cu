#include<cuda.h>
#include<iostream>

using namespace std;

#define N 10

void merge(float *arr,int low,int mid,int high){
    int temp[high + low -1];
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

}

void divide(float *nums,int l,int r){
    if(l>=r) return;

    int mid = l + (r - l) / 2;

    divide(nums,l,mid);
    divide(nums,mid+1,r);

    merge(nums,l,mid,r);
}

__global__ void hello(){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("HELLO FROM %d\n",id);
}

int main(){
    float nums[] = {23,124,2353,4324,875,347,984732,9875,365,34895};

    // for(int i = 0;i<N;i++){
    //     cin>>nums[i];
    // }

    divide(nums,0,N-1);
    cudaDeviceSynchronize();
}