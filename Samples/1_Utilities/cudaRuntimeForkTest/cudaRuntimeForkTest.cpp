/*
Verification that the CUDA Runtime API use of `cudaGetDeviceCount` in turn initializes the CUDA Driver API (via
`cuInit`) leading to an initialization error for a subsequently forked process.


$ ./cudaRuntimeForkTest
Bypassing pre-fork CUDA initialization. 
printed from child process 2217816: Device count: 2
printed from parent process 2217815: Device count: 2
$ export EXPLICIT_INIT=1
$ ./cudaRuntimeForkTest
successfully init cuda from: 2217829
printed from parent process 2217829: Device count: 2
printed from child process 2217835: Device count: CUDA Runtime Error at: cudaRuntimeForkTest.cpp:48
initialization error cudaGetDeviceCount(&deviceCount)
0

*/

#include <cuda.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sys/wait.h>
#include <cstring>


using std::cout; using std::endl;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

int deviceCount() {
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

void explicit_init_cuda(){
  cuInit(0);
  cout << "successfully init cuda from: " << getpid() << endl;
}

int main() {
    const char* exp_init = std::getenv("EXPLICIT_INIT");
    const bool expl_init = (exp_init == nullptr) ? false : std::strcmp(exp_init, "1") == 0;
    if (expl_init) {
      explicit_init_cuda();
    } else {
      cout << "Bypassing pre-fork CUDA initialization. \n";
    }
    pid_t c_pid = fork();

    if (c_pid == -1) {
        perror("fork");
        return 1;
    } else if (c_pid > 0) {
        cout << "printed from parent process " << getpid() << ": Device count: " << deviceCount() << endl;
        wait(nullptr);
    } else {
        cout << "printed from child process " << getpid() << ": Device count: " << deviceCount() << endl;
        return 0;
    }

    return 0;
}
