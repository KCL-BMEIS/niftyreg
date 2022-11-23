#include "ClOptimiseKernel.h"

/* *************************************************************** */
ClOptimiseKernel::ClOptimiseKernel(AladinContent *conIn, std::string name) : OptimiseKernel(name) {
    //populate the CLAladinContent object ptr
    con = static_cast<ClAladinContent*>(conIn);

    //get opencl context params
    sContext = &ClContextSingleton::Instance();
    /*clContext = sContext->GetContext();*/
    /*commandQueue = sContext->GetCommandQueue();*/

    //get necessary cpu ptrs
    transformationMatrix = con->AladinContent::GetTransformationMatrix();
    blockMatchingParams = con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
void ClOptimiseKernel::Calculate(bool affine) {
    //cpu atm
    this->blockMatchingParams = con->GetBlockMatchingParams();
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
/* *************************************************************** */
ClOptimiseKernel::~ClOptimiseKernel() {}
/* *************************************************************** */
