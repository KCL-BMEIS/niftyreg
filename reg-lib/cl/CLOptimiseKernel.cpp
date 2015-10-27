#include "CLOptimiseKernel.h"

/* *************************************************************** */
CLOptimiseKernel::CLOptimiseKernel(Content *conIn, std::string name) : OptimiseKernel(name) {
    //populate the CLContent object ptr
    con = static_cast<ClContent*>(conIn);

    //get opencl context params
    sContext = &CLContextSingletton::Instance();
    /*clContext = sContext->getContext();*/
    /*commandQueue = sContext->getCommandQueue();*/

    //get necessary cpu ptrs
    transformationMatrix = con->Content::getTransformationMatrix();
    blockMatchingParams = con->Content::getBlockMatchingParams();
}
/* *************************************************************** */
void CLOptimiseKernel::calculate(bool affine, bool ils) {
    //cpu atm
    this->blockMatchingParams = con->getBlockMatchingParams();
    optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
/* *************************************************************** */
CLOptimiseKernel::~CLOptimiseKernel() {}
/* *************************************************************** */
