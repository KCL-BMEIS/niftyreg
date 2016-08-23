#include "CLOptimiseKernel.h"

/* *************************************************************** */
CLOptimiseKernel::CLOptimiseKernel(GlobalContent *conIn, std::string name) : OptimiseKernel(name) {
    //populate the CLAladinContent object ptr
    this->con = dynamic_cast<ClAladinContent*>(conIn);

    //get opencl context params
    sContext = &CLContextSingletton::Instance();
    /*clContext = sContext->getContext();*/
    /*commandQueue = sContext->getCommandQueue();*/

    //get necessary cpu ptrs
    transformationMatrix = con->AladinContent::getTransformationMatrix();
    blockMatchingParams = con->AladinContent::getBlockMatchingParams();
}
/* *************************************************************** */
void CLOptimiseKernel::calculate(bool affine) {
    //cpu atm
    this->blockMatchingParams = con->getBlockMatchingParams();
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
/* *************************************************************** */
CLOptimiseKernel::~CLOptimiseKernel() {}
/* *************************************************************** */
