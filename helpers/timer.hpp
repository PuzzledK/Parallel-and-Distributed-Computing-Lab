#include<chrono>

class Timer{
    std::chrono::time_point<std::chrono::high_resolution_clock> st;
    std::chrono::time_point<std::chrono::high_resolution_clock> en;

    public:
    void tick(){
        st = std::chrono::high_resolution_clock::now();
    }
    void tock(){
        en = std::chrono::high_resolution_clock::now();
    }
    double time(){
        std::chrono::duration<double> duration = en - st;
        return duration.count();
    }
};