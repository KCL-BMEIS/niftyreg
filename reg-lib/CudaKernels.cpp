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
