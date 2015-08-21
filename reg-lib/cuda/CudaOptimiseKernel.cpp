#include "CudaOptimiseKernel.h"

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
    referencePos_d = con->getTargetPosition_d();
    warpedPos_d = con->getResultPosition_d();
    newWarpedPos_d = con->getNewResultPos_d();

}
/* *************************************************************** */
void CudaOptimiseKernel::calculate(bool affine, bool ils, bool cusvd)
{
    //for now. Soon we will have a GPU version of it
#ifndef CUDA7
    this->blockMatchingParams = con->getBlockMatchingParams();
    optimize(this->blockMatchingParams, transformationMatrix, affine);
#else
    const unsigned long num_to_keep = (unsigned long) (blockMatchingParams->definedActiveBlock *
                                                                        (blockMatchingParams->percent_to_keep / 100.0f));
    if (affine) {
        if (cusvd)
            optimize_affine3D_cuda(transformationMatrix,
                                          transformationMatrix_d,
                                          AR_d,
                                          U_d,
                                          Sigma_d,
                                          VT_d,
                                          lengths_d,
                                          targetPos_d,
                                          resultPos_d,
                                          newResultPos_d,
                                          blockMatchingParams->definedActiveBlock * 3,
                                          12,
                                          num_to_keep,
                                          ils);
        else {
            this->blockMatchingParams = con->getBlockMatchingParams();
            optimize(this->blockMatchingParams, transformationMatrix, affine);
        }
    }
    else {
        this->blockMatchingParams = con->getBlockMatchingParams();
        optimize(this->blockMatchingParams, transformationMatrix, affine);
    }
#endif
}
/* *************************************************************** */
