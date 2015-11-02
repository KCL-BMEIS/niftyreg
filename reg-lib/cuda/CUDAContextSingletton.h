#ifndef CUDACONTEXTSINGLETTON_H
#define CUDACONTEXTSINGLETTON_H

#include "cuda.h"

class CUDAContextSingletton
{
    public:
        static CUDAContextSingletton& Instance() {
            static CUDAContextSingletton instance; // Guaranteed to be destroyed.
            // Instantiated on first use.
            return instance;
        }
        void setCudaIdx(int cudaIdxIn);
        void pickCard(unsigned deviceId);

        CUcontext getContext();

     private:

        static CUDAContextSingletton* _instance;

        CUDAContextSingletton();
        ~CUDAContextSingletton() {}

        CUDAContextSingletton(CUDAContextSingletton const&);// Don't Implement
        void operator=(CUDAContextSingletton const&); // Don't implement

        CUcontext cudaContext;
        unsigned numDevices;
        int cudaIdx;
};

#endif // CUDACONTEXTSINGLETTON_H
