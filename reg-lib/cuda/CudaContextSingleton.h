#pragma once

#include "_reg_maths.h"
#include <cuda.h>

class CudaContextSingleton {
public:
    static CudaContextSingleton& Instance() {
        static CudaContextSingleton instance; // Guaranteed to be destroyed.
        // Instantiated on first use.
        return instance;
    }
    void SetCudaIdx(unsigned int cudaIdxIn);
    void PickCard(unsigned deviceId);

    CUcontext GetContext();

    bool GetIsCardDoubleCapable();

private:

    static CudaContextSingleton* _instance;

    CudaContextSingleton();
    ~CudaContextSingleton();

    CudaContextSingleton(CudaContextSingleton const&);// Don't Implement
    void operator=(CudaContextSingleton const&); // Don't implement

    bool isCardDoubleCapable;
    CUcontext cudaContext;
    unsigned numDevices;
    unsigned cudaIdx;
};
