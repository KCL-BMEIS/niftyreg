#pragma once

#include "OptimiseKernel.h"
#include "CudaAladinContent.h"

// Kernel functions for numerical optimisation
class CudaOptimiseKernel: public OptimiseKernel {
public:
    CudaOptimiseKernel(Content *conIn);
    void Calculate(bool affine);

private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
    CudaAladinContent *con;

//    float *AR_d; // Removed until CUDA SVD is added back
//    float *U_d; // Removed until CUDA SVD is added back
//    float *Sigma_d; // Removed until CUDA SVD is added back
//    float *VT_d; // Removed until CUDA SVD is added back
//    float *lengths_d; // Removed until CUDA SVD is added back
//    float *newWarpedPos_d; // Removed until CUDA SVD is added back
};
