#include "cuda_runtime.h"
#include "cuda.h"
#include "CUDAOptimiseKernel.h"
#include "optimizeKernel.h"

/* *************************************************************** */
CudaOptimiseKernel::CudaOptimiseKernel(Content *conIn, std::string name) :
OptimiseKernel(name)
{
    //get CudaContent ptr
    con = static_cast<CudaContent*>(conIn);

    //get cpu ptrs
    transformationMatrix = con->Content::getTransformationMatrix();
    blockMatchingParams = con->Content::getBlockMatchingParams();

    transformationMatrix_d = con->getTransformationMatrix_d();
    AR_d = con->getAR_d();
    U_d = con->getU_d();
    Sigma_d = con->getSigma_d();
    VT_d = con->getVT_d();
    lengths_d = con->getLengths_d();
    referencePos_d = con->getReferencePosition_d();
    warpedPos_d = con->getWarpedPosition_d();
    newWarpedPos_d = con->getNewResultPos_d();

}
/* *************************************************************** */
void CudaOptimiseKernel::calculate(bool affine, bool ils, bool cusvd)
{
#ifdef __i386__
    this->blockMatchingParams = con->getBlockMatchingParams();
    optimize(this->blockMatchingParams, transformationMatrix, affine);
#else
    //for now. Soon we will have a GPU version of it
    int* cudaRunTimeVersion = (int*)malloc(sizeof(int));
    int* cudaDriverVersion = (int*)malloc(sizeof(int));
    cudaRuntimeGetVersion(cudaRunTimeVersion);
    cudaDriverGetVersion(cudaDriverVersion);
#ifndef DEBUG
    printf("CUDA RUNTIME VERSION=%i", *cudaRunTimeVersion);
    printf("CUDA DRIVER VERSION=%i", *cudaDriverVersion);
#endif

    if (*cudaRunTimeVersion < 7050) {
        this->blockMatchingParams = con->getBlockMatchingParams();
        optimize(this->blockMatchingParams, transformationMatrix, affine);
    }
    else {
        const unsigned long num_to_keep = (unsigned long)(blockMatchingParams->definedActiveBlockNumber *(blockMatchingParams->percent_to_keep / 100.0f));
        optimize_affine3D_cuda(transformationMatrix,
            transformationMatrix_d,
            AR_d,
            U_d,
            Sigma_d,
            VT_d,
            lengths_d,
            referencePos_d,
            warpedPos_d,
            newWarpedPos_d,
            blockMatchingParams->definedActiveBlockNumber * 3,
            12,
            num_to_keep,
            ils, affine);
    }
#endif
}
/* *************************************************************** */
