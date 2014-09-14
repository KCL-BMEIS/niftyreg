#include "CudaKernels.h"
#include "cudaKernelFuncs.h"
#include "_reg_tools.h"

//------------------------------------------------------------------------------------------------------------------------
//..................CudaConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CudaConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	std::cout << "Launch cuda kernel! (CPU Cheat)"<< std::endl;
	/*launch(image, sigma, kernelType, mask, timePoint, axis);*/
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);

}

void CudaAffineDeformationFieldKernel::execute(  bool compose ) {
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  affine kernel!" << std::endl;
	launchAffine(this->affineTransformation, this->deformationFieldImage, compose, this->mask);
	std::cout << "===================================================" << std::endl;
}

void CudaResampleImageKernel::execute( int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {

	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  resample kernel!" << std::endl;
	
	launchResample(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dti_timepoint,  jacMat);
	std::cout << "===================================================" << std::endl;
}
void CudaBlockMatchingKernel::execute(){
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  block matching kernel!" << std::endl;

	launchBlockMatching(target, result, params, mask);
	std::cout << "===================================================" << std::endl;
}
void CudaOptimiseKernel::execute( bool affine) {
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  optimize kernel! (CPU cheating)" << std::endl;
	//launchOptimize(params, transformation_matrix, affine);
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
	std::cout << "===================================================" << std::endl;
}

