#ifndef CUDAOPTIMISEKERNEL_H
#define CUDAOPTIMISEKERNEL_H

#include "OptimiseKernel.h"
#include "CUDAContent.h"

//kernel functions for numerical optimisation
class CUDAOptimiseKernel: public OptimiseKernel
{
public:
    CUDAOptimiseKernel(Content *conIn, std::string name);
    void calculate(bool affine);
private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
    //CUDAContextSingletton *cudaSContext;
    CudaContent *con;

    float* transformationMatrix_d;
//    float* AR_d; // Removed until CUDA SVD is added back
//    float* U_d; // Removed until CUDA SVD is added back
//    float* Sigma_d; // Removed until CUDA SVD is added back
//    float* VT_d; // Removed until CUDA SVD is added back
//    float* lengths_d; // Removed until CUDA SVD is added back
    float* referencePos_d;
    float* warpedPos_d;
//    float* newWarpedPos_d; // Removed until CUDA SVD is added back

};

#endif // CUDAOPTIMISEKERNEL_H
