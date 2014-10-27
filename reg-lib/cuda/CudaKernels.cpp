#include "CudaKernels.h"
#include "cudaKernelFuncs.h"
#include "_reg_tools.h"

//------------------------------------------------------------------------------------------------------------------------
//..................CudaConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CudaConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	//std::cout << "Launch cuda kernel! (CPU Cheat)"<< std::endl;
	/*launch(image, sigma, kernelType, mask, timePoint, axis);*/
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);

}

void CudaAffineDeformationFieldKernel::execute(  bool compose ) {
	/*std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  affine kernel!" << std::endl;*/
	//launchAffine(this->affineTransformation, this->deformationFieldImage, compose, this->mask);
	launchAffine2(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, compose);
	//std::cout << "===================================================" << std::endl;
}

void CudaResampleImageKernel::execute( int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {

	/*std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  resample kernel!" << std::endl;*/

	launchResample2(floatingImage, warpedImage,  mask, interp, paddingValue, dti_timepoint, jacMat, &floatingImageArray_d, &warpedImageArray_d, &deformationFieldImageArray_d, &mask_d);

	//launchResample(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dti_timepoint,  jacMat);
	//std::cout << "===================================================" << std::endl;
}
void CudaBlockMatchingKernel::execute(){
	/*std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  block matching kernel!" << std::endl;*/
	//tomorrow test each block matching step between cpu and gpu here!
	/*this->result = con->getCurrentWarped();
	block_matching_method(this->target, this->result, this->params, this->mask); 
	printf("definedActiveBlock: %d\n", params->definedActiveBlock);
	printf("activeBlockNumber: %d\n", params->activeBlockNumber);*/
	
	launchBlockMatching2(target, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d);
	//printf("definedActiveBlock: %d\n", params->definedActiveBlock);
	//std::cout << "===================================================" << std::endl;
}
void CudaOptimiseKernel::execute( bool affine) {
	/*std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  optimize kernel! (CPU cheating)" << std::endl;*/

	this->blockMatchingParams = con->getBlockMatchingParams(); 
	//printf("definedActiveBlock: %d\n", blockMatchingParams->definedActiveBlock);
	//blockMatchingParams->definedActiveBlock = blockMatchingParams->activeBlockNumber;//small hack as we do not get the definedActiveBlockNumber on GPUs
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
	
	//std::cout << "===================================================" << std::endl;
}

