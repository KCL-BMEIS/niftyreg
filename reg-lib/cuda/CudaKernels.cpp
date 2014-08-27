#include "CudaKernels.h"
#include "cudaKernelFuncs.h"

//------------------------------------------------------------------------------------------------------------------------
//..................CudaConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CudaConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	std::cout << "Launch cuda kernel!"<< std::endl;
	launch(image, sigma, kernelType, mask, timePoint, axis);
}

void CudaAffineDeformationFieldKernel::execute( mat44 *affineTransformation, nifti_image *deformationField, bool compose , int *mask ) {
	std::cout << "Launching cuda  affine kernel!" << std::endl;
	launchAffine(affineTransformation, deformationField, compose, mask);
}
void CudaOptimiseKernel::execute(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine) {
	std::cout << "Launching cuda  optimise kernel!" << std::endl;
	launchOptimize(params, transformation_matrix, affine);
}
void CudaResampleImageKernel::execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {

	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  resample kernel!" << std::endl;
	
	launchResample(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dti_timepoint,  jacMat);
	std::cout << "===================================================" << std::endl;
}
