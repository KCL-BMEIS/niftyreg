#include <cuda_runtime.h>
#include <cuda.h>
#include "CudaLtsKernel.h"
#include "optimizeKernel.h"

/* *************************************************************** */
CudaLtsKernel::CudaLtsKernel(Content *conIn) : LtsKernel() {
    //get CudaAladinContent ptr
    con = static_cast<CudaAladinContent*>(conIn);

    //get cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();

    //   transformationMatrix_d = con->GetTransformationMatrix_d();
    //   AR_d = con->GetAR_d(); // Removed until CUDA SVD is added back
    //   U_d = con->GetU_d(); // Removed until CUDA SVD is added back
    //   Sigma_d = con->GetSigma_d(); // Removed until CUDA SVD is added back
    //   VT_d = con->GetVT_d(); // Removed until CUDA SVD is added back
    //   lengths_d = con->GetLengths_d(); // Removed until CUDA SVD is added back
    //   referencePos_d = con->GetReferencePosition_d();
    //   warpedPos_d = con->GetWarpedPosition_d();
    //   newWarpedPos_d = con->GetNewWarpedPos_d(); // Removed until CUDA SVD is added back

}
/* *************************************************************** */
void CudaLtsKernel::Calculate(bool affine) {
    /* // Removed until CUDA SVD is added back
 #if _WIN64 || __x86_64__ || __ppc64__

     //for now. Soon we will have a GPU version of it
     int* cudaRunTimeVersion = (int*)malloc(sizeof(int));
     int* cudaDriverVersion = (int*)malloc(sizeof(int));
     cudaRuntimeGetVersion(cudaRunTimeVersion);
     cudaDriverGetVersion(cudaDriverVersion);

     NR_DEBUG("CUDA runtime version=" << *cudaRunTimeVersion);
     NR_DEBUG("CUDA driver version=" << *cudaDriverVersion);

     if (*cudaRunTimeVersion < 7050) {
         blockMatchingParams = con->GetBlockMatchingParams();
         optimize(blockMatchingParams, transformationMatrix, affine);
     }
     else {
         //HAVE TO DO THE RIGID AND 2D VERSION
         if(affine && blockMatchingParams->dim == 3) {
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
             blockMatchingParams = con->GetBlockMatchingParams();
             optimize(blockMatchingParams, transformationMatrix, affine);
         }
     }
 #else
     blockMatchingParams = con->GetBlockMatchingParams();
     optimize(blockMatchingParams, transformationMatrix, affine);
 #endif
 */
    blockMatchingParams = con->GetBlockMatchingParams();
    optimize(blockMatchingParams, transformationMatrix, affine);
}
/* *************************************************************** */
