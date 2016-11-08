#include "cuda_runtime.h"
#include "cuda.h"
#include "CUDAOptimiseKernel.h"
#include "optimizeKernel.h"

/* *************************************************************** */
CUDAOptimiseKernel::CUDAOptimiseKernel(AladinContent *conIn, std::string name) :
   OptimiseKernel(name)
{
   //get CudaAladinContent ptr
   con = static_cast<CudaAladinContent*>(conIn);

   //cudaSContext = &CUDAContextSingletton::Instance();

   //get cpu ptrs
   transformationMatrix = con->AladinContent::getTransformationMatrix();
   blockMatchingParams = con->AladinContent::getBlockMatchingParams();

//   transformationMatrix_d = con->getTransformationMatrix_d();
//   AR_d = con->getAR_d(); // Removed until CUDA SVD is added back
//   U_d = con->getU_d(); // Removed until CUDA SVD is added back
//   Sigma_d = con->getSigma_d(); // Removed until CUDA SVD is added back
//   VT_d = con->getVT_d(); // Removed until CUDA SVD is added back
//   lengths_d = con->getLengths_d(); // Removed until CUDA SVD is added back
//   referencePos_d = con->getReferencePosition_d();
//   warpedPos_d = con->getWarpedPosition_d();
//   newWarpedPos_d = con->getNewWarpedPos_d(); // Removed until CUDA SVD is added back

}
/* *************************************************************** */
void CUDAOptimiseKernel::calculate(bool affine) {
   /* // Removed until CUDA SVD is added back
#if _WIN64 || __x86_64__ || __ppc64__

    //for now. Soon we will have a GPU version of it
    int* cudaRunTimeVersion = (int*)malloc(sizeof(int));
    int* cudaDriverVersion = (int*)malloc(sizeof(int));
    cudaRuntimeGetVersion(cudaRunTimeVersion);
    cudaDriverGetVersion(cudaDriverVersion);

#ifndef DEBUG
    printf("CUDA RUNTIME VERSION=%i\n", *cudaRunTimeVersion);
    printf("CUDA DRIVER VERSION=%i\n", *cudaDriverVersion);
#endif

    if (*cudaRunTimeVersion < 7050) {
        this->blockMatchingParams = con->getBlockMatchingParams();
        optimize(this->blockMatchingParams, transformationMatrix, affine);
    }
    else {
        //HAVE TO DO THE RIGID AND 2D VERSION
        if(affine && this->blockMatchingParams->dim == 3) {
            const unsigned long num_to_keep = (unsigned long)(blockMatchingParams->activeBlockNumber *(blockMatchingParams->percent_to_keep / 100.0f));
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
                                   blockMatchingParams->activeBlockNumber * 3,
                                   12,
                                   num_to_keep,
                                   ils,
                                   affine);
        } else {
            this->blockMatchingParams = con->getBlockMatchingParams();
            optimize(this->blockMatchingParams, transformationMatrix, affine);
        }
    }
#else
    this->blockMatchingParams = con->getBlockMatchingParams();
    optimize(this->blockMatchingParams, transformationMatrix, affine);
#endif
*/
   this->blockMatchingParams = con->getBlockMatchingParams();
   optimize(this->blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
