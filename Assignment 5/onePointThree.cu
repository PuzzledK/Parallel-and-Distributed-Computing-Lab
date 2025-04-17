#include<cuda.h>
#include<iostream>
using namespace std;

int main(){

    cudaDeviceProp prop;
    int device = 0;

    cudaGetDeviceProperties(&prop,device);

    float memClockRate = prop.memoryClockRate;
    int memBW = prop.memoryBusWidth;

    cout<<"DEVICE NAME -> "<<prop.name<<endl;
    cout<<"MEMORY CLOCK RATE -> "<<memClockRate<<" kHz"<<endl;
    cout<<"MEMORY BANDWIDTH -> "<<memBW<<" Bits"<<endl;

    float bandWidth = 2.0f * memClockRate * memBW / 8.0f / 1e6f;

    cout<<"THEORETICAL BANDWIDTH -> "<<bandWidth<<"GB/s"<<endl;

}