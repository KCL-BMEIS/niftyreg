#include "CPUKernels.h"
#include"_reg_resampling.h"
#include"_reg_globalTransformation.h"


//------------------------------------------------------------------------------------------------------------------------
//..................CPUAffineDeformationFieldKernel---------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------



//------------------------------------------------------------------------------------------------------------------------
void CPUAffineDeformationFieldKernel::execute(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask) {
	reg_affine_getDeformationField(affineTransformation, deformationField);
}
//------------------------------------------------------------------------------------------------------------------------


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
void CPUBlockMatchingKernel::execute(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) {
block_matching_method(target,result,params,mask);

}

void CPUBlockMatchingKernel::initialize(nifti_image * target, _reg_blockMatchingParam *params, int percentToKeep_block, int percentToKeep_opt, int *mask, bool runningOnGPU) {
	initialise_block_matching_method(target, params, percentToKeep_block, percentToKeep_opt, mask, false);
}

void CPUOptimiseKernel::execute(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine) {
	optimize(params, transformation_matrix, affine);
}

void CPUResampleImageKernel::execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {
	reg_resampleImage(floatingImage, warpedImage, deformationField, mask,interp, paddingValue, dti_timepoint, jacMat);
}
