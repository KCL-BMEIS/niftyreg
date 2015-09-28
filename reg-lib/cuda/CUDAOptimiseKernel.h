#ifndef CUDAOPTIMISEKERNEL_H
#define CUDAOPTIMISEKERNEL_H

#include "OptimiseKernel.h"
#include "CUDAContent.h"

//kernel functions for numerical optimisation
class CudaOptimiseKernel: public OptimiseKernel
{
public:
    CudaOptimiseKernel(Content *conIn, std::string name);
    void calculate(bool affine, bool ils, bool cusvd);
private:
    _reg_blockMatchingParam *blockMatchingParams;
    mat44 *transformationMatrix;
    CudaContent *con;

    float* transformationMatrix_d;
    float* AR_d;
    float* U_d;
    float* Sigma_d;
    float* VT_d;
    float* lengths_d;
    float* referencePos_d;
    float* warpedPos_d;
    float* newWarpedPos_d;

};

#endif // CUDAOPTIMISEKERNEL_H
