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


//==============================Cuda Affine Kernel================================================================
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Context* conIn, std::string nameIn) :
		AffineDeformationFieldKernel(nameIn) {

	con = ((CudaContext*) conIn);

	this->deformationFieldImage = con->CurrentDeformationField;
	this->affineTransformation = con->transformationMatrix;

	mask_d = con->getMask_d();
	deformationFieldArray_d = con->getDeformationFieldArray_d();

}

void CudaAffineDeformationFieldKernel::execute(bool compose) {

	launchAffine(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, compose);

}
//------------------------------------------------------------------------------------

//==============================Cuda Resamlple Kernel================================================================

CudaResampleImageKernel::CudaResampleImageKernel(Context* conIn, std::string name) :
		ResampleImageKernel(name) {

	con = static_cast<CudaContext*>(conIn);

	floatingImage = conIn->CurrentFloating;
	warpedImage = conIn->CurrentWarped;

	//cuda ptrs
	floatingImageArray_d = con->getFloatingImageArray_d();
	warpedImageArray_d = con->getWarpedImageArray_d();
	deformationFieldImageArray_d = con->getDeformationFieldArray_d();
	mask_d = con->getMask_d();
	floIJKMat_d = con->getFloIJKMat_d();

	if (floatingImage->datatype != warpedImage->datatype) {
		printf("[NiftyReg ERROR] reg_resampleImage\tSource and result image should have the same data type\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

	if (floatingImage->nt != warpedImage->nt) {
		printf("[NiftyReg ERROR] reg_resampleImage\tThe source and result images have different dimension along the time axis\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

}

void CudaResampleImageKernel::execute(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {
	launchResample(floatingImage, warpedImage, interp, paddingValue, dti_timepoint, jacMat, &floatingImageArray_d, &warpedImageArray_d, &deformationFieldImageArray_d, &mask_d, &floIJKMat_d);

}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//==============================Cuda Block Matching Kernel================================================================
CudaBlockMatchingKernel::CudaBlockMatchingKernel(Context* conIn, std::string name) :
		BlockMatchingKernel(name) {

	con = ((CudaContext*) conIn);

	target = conIn->CurrentReference;
	params = conIn->blockMatchingParams;

	targetImageArray_d = con->getReferenceImageArray_d();
	resultImageArray_d = con->getWarpedImageArray_d();
	targetPosition_d = con->getTargetPosition_d();
	resultPosition_d = con->getResultPosition_d();
	activeBlock_d = con->getActiveBlock_d();
	mask_d = con->getMask_d();
	targetMat_d = con->getTargetMat_d();

}

void CudaBlockMatchingKernel::execute() {

	launchBlockMatching(target, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d, &targetMat_d);
}
//===================================================================================================================================================================
CudaOptimiseKernel::CudaOptimiseKernel(Context* conIn, std::string name) :
			OptimiseKernel(name) {
		con = static_cast<CudaContext*>(conIn);
		transformationMatrix = con->transformationMatrix;
		blockMatchingParams = con->blockMatchingParams;

	}

void CudaOptimiseKernel::execute(bool affine) {

	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}

