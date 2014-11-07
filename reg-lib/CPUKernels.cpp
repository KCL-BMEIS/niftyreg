#include "CPUKernels.h"
#include"_reg_resampling.h"
#include"_reg_globalTransformation.h"


//------------------------------------------------------------------------------------------------------------------------
//..................CPUAffineDeformationFieldKernel---------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUAffineDeformationFieldKernel::execute(bool compose) {
	reg_affine_getDeformationField(this->affineTransformation, this->deformationFieldImage, compose, this->mask);
}
//------------------------------------------------------------------------------------------------------------------------
//..................END CPUAffineDeformationFieldKernel-----------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------
//..................CPUConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
//------------------------------------------------------------------------------------------------------------------------
//..................END CPUConvolutionKernel------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUBlockMatchingKernel::execute() {
	block_matching_method(this->target, this->result, this->params, this->mask);


}

void CPUOptimiseKernel::execute(bool affine) {
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}

void CPUResampleImageKernel::execute(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {
	reg_resampleImage(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dti_timepoint, jacMat);
}
