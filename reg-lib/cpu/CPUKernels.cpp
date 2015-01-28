#include "CPUKernels.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"


//------------------------------------------------------------------------------------------------------------------------
//..................CPUAffineDeformationFieldKernel---------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUAffineDeformationFieldKernel::calculate(bool compose) {
	reg_affine_getDeformationField(this->affineTransformation, this->deformationFieldImage, compose, this->mask);
}
//------------------------------------------------------------------------------------------------------------------------
//..................END CPUAffineDeformationFieldKernel-----------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------
//..................CPUConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
//------------------------------------------------------------------------------------------------------------------------
//..................END CPUConvolutionKernel------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUBlockMatchingKernel::calculate(int range) {
	block_matching_method(this->target, this->result, this->params, this->mask, range);
}

void CPUOptimiseKernel::calculate(bool affine, bool ils, bool svd) {
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}

void CPUResampleImageKernel::calculate(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {
	reg_resampleImage(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dti_timepoint, jacMat);
}
