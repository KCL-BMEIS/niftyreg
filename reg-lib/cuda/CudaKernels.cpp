#include "CudaKernels.h"
#include "cudaKernelFuncs.h"
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


	/*std::cout << "===================================================" << std::endl;
		std::cout << "Launching cuda  affine kernel!" << std::endl;*/
		//launchAffine(this->affineTransformation, this->deformationFieldImage, compose, this->mask);
		//launchAffine2(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, compose);


		/*float* a = static_cast<float*>(deformationFieldImage->data);
		float* aa = (float*)malloc(deformationFieldImage->nvox*sizeof(float));
		float* bb = (float*)malloc(deformationFieldImage->nvox*sizeof(float));*/



		launchAffine(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, compose);

		/*float* b = static_cast<float*>(con->getCurrentDeformationField()->data);
		for (size_t i = 0; i < deformationFieldImage->nvox; i++)
		{
			bb[i] = b[i];
		}


		reg_affine_getDeformationField(this->affineTransformation, this->deformationFieldImage, compose, this->mask);
		for (size_t i = 0; i < deformationFieldImage->nvox; i++)
		{
			aa[i] = a[i];
		}

		double maxDiff = reg_test_compare_arrays<float>(aa, bb, deformationFieldImage->nvox);
		std::cout << "dif: " << maxDiff << std::endl;*/


}

void CudaResampleImageKernel::execute( int interp, float paddingValue, bool *dti_timepoint , mat33 * jacMat ) {

//	con->setCurrentDeformationField(this->deformationField);
	launchResample(floatingImage, warpedImage,  mask, interp, paddingValue, dti_timepoint, jacMat, &floatingImageArray_d, &warpedImageArray_d, &deformationFieldImageArray_d, &mask_d);

}
void CudaBlockMatchingKernel::execute(){

	launchBlockMatching(target, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d);
}
void CudaOptimiseKernel::execute( bool affine) {

	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}

