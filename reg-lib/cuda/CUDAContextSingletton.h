#ifndef CUDACONTEXTSINGLETTON_H
#define CUDACONTEXTSINGLETTON_H

#include "_reg_maths.h"
#include "cuda.h"

class CUDAContextSingletton
{
    public:
        static CUDAContextSingletton& Instance() {
            static CUDAContextSingletton instance; // Guaranteed to be destroyed.
            // Instantiated on first use.
            return instance;
        }
        void setCudaIdx(unsigned int cudaIdxIn);
        void pickCard(unsigned deviceId);

        CUcontext getContext();

        bool getIsCardDoubleCapable();

     private:

        static CUDAContextSingletton* _instance;

        CUDAContextSingletton();
        ~CUDAContextSingletton();

        CUDAContextSingletton(CUDAContextSingletton const&);// Don't Implement
        void operator=(CUDAContextSingletton const&); // Don't implement

        bool isCardDoubleCapable;
        CUcontext cudaContext;
        unsigned numDevices;
        unsigned cudaIdx;
};

#endif // CUDACONTEXTSINGLETTON_H
