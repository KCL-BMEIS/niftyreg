#include "CudaKernels.h"
#include "CudaKernelFuncs.h"
#include "_reg_tools.h"

#include"_reg_resampling.h"
#include"_reg_globalTransformation.h"

//------------------------------------------------------------------------------------------------------------------------
//..................CudaConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CudaConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	//cpu cheat
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}

void CudaAffineDeformationFieldKernel::execute(  bool compose ) {

	launchAffine(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, compose);

}

void CudaResampleImageKernel::execute( int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {
	launchResample(floatingImage, warpedImage,  mask, interp, paddingValue, dti_timepoint, jacMat, &floatingImageArray_d, &warpedImageArray_d, &deformationFieldImageArray_d, &mask_d);

}
void CudaBlockMatchingKernel::execute(){

	launchBlockMatching(target, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d);
}
void CudaOptimiseKernel::execute( bool affine) {

	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}

