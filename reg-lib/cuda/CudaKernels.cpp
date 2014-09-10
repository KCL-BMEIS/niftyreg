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

void CudaAffineDeformationFieldKernel::execute( mat44 *affineTransformation, nifti_image *deformationField, bool compose , int *mask ) {
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  affine kernel!" << std::endl;
	launchAffine(affineTransformation, deformationField, compose, mask);
	std::cout << "===================================================" << std::endl;
}
void CudaOptimiseKernel::execute(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine) {
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  optimize kernel! (CPU cheating)" << std::endl;
	//launchOptimize(params, transformation_matrix, affine);
	optimize(params, transformation_matrix, affine);
	std::cout << "===================================================" << std::endl;
}
void CudaResampleImageKernel::execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {

	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  resample kernel!" << std::endl;
	
	launchResample(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dti_timepoint,  jacMat);
	std::cout << "===================================================" << std::endl;
}

void CudaBlockMatchingKernel::execute(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask){
	std::cout << "===================================================" << std::endl;
	std::cout << "Launching cuda  block matching kernel!" << std::endl;

	launchBlockMatching(target, result, params, mask);
	std::cout << "===================================================" << std::endl;
}

////temporary for tests: cpu code
void CudaBlockMatchingKernel::initialize(nifti_image * target, _reg_blockMatchingParam *params, int percentToKeep_block, int percentToKeep_opt, int *mask, bool runningOnGPU) {
	if (params->activeBlock != NULL) {
		free(params->activeBlock);
		params->activeBlock = NULL;
	}
	if (params->targetPosition != NULL) {
		free(params->targetPosition);
		params->targetPosition = NULL;
	}
	if (params->resultPosition != NULL) {
		free(params->resultPosition);
		params->resultPosition = NULL;
	}

	params->blockNumber[0] = (int)reg_ceil((float)target->nx / (float)BLOCK_WIDTH);
	params->blockNumber[1] = (int)reg_ceil((float)target->ny / (float)BLOCK_WIDTH);
	if (target->nz>1)
		params->blockNumber[2] = (int)reg_ceil((float)target->nz / (float)BLOCK_WIDTH);
	else params->blockNumber[2] = 1;

	params->percent_to_keep = percentToKeep_opt;
	params->activeBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * percentToKeep_block / 100;

	params->activeBlock = (int *)malloc(params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * sizeof(int));
	switch (target->datatype) {
	case NIFTI_TYPE_FLOAT32:
		_reg_set_active_blocks<float>(target, params, mask, true);
		break;
	case NIFTI_TYPE_FLOAT64:
		_reg_set_active_blocks<double>(target, params, mask, true);
		break;
	default:
		fprintf(stderr, "[NiftyReg ERROR] initialise_block_matching_method\tThe target image data type is not supported\n");
		reg_exit(1);
	}
	if (params->activeBlockNumber<2) {
		fprintf(stderr, "[NiftyReg ERROR] There are no active blocks\n");
		fprintf(stderr, "[NiftyReg ERROR] ... Exit ...\n");
		reg_exit(1);
	}
#ifndef NDEBUG
	printf("[NiftyReg DEBUG]: There are %i active block(s) out of %i.\n", params->activeBlockNumber, params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2]);
#endif
	if (target->nz>1) {
		std::cout << "allocating: " << params->activeBlockNumber << std::endl;
		params->targetPosition = (float *)malloc(params->activeBlockNumber * 3 * sizeof(float));
		params->resultPosition = (float *)malloc(params->activeBlockNumber * 3 * sizeof(float));
	}
	else {
		params->targetPosition = (float *)malloc(params->activeBlockNumber * 2 * sizeof(float));
		params->resultPosition = (float *)malloc(params->activeBlockNumber * 2 * sizeof(float));
	}
#ifndef NDEBUG
	printf("[NiftyReg DEBUG] block matching initialisation done.\n");
#endif
}


