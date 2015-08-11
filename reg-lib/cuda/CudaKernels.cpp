#include "CudaKernels.h"
#include "CudaKernelFuncs.h"
#include "_reg_tools.h"
#include "_reg_blockMatching_cuda.h"
#include "_reg_blockMatching.h"
#include"_reg_resampling.h"
#include"_reg_globalTrans.h"

/* *************************************************************** */
void CudaConvolutionKernel::calculate(nifti_image *image,
												  float *sigma,
												  int kernelType,
												  int *mask,
												  bool *timePoint,
												  bool *axis)
{
	//cpu cheat
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
/* *************************************************************** */
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Content *conIn, std::string nameIn) :
		AffineDeformationFieldKernel(nameIn)
{
	con = static_cast<CudaContent*>(conIn);

	//get necessary cpu ptrs
	this->deformationFieldImage = con->Content::getCurrentDeformationField();
	this->affineTransformation = con->Content::getTransformationMatrix();

	//get necessary cuda ptrs
	mask_d = con->getMask_d();
	deformationFieldArray_d = con->getDeformationFieldArray_d();
	transformationMatrix_d = con->getTransformationMatrix_d();
}
/* *************************************************************** */
void CudaAffineDeformationFieldKernel::calculate(bool compose)
{
	launchAffine(this->affineTransformation,
					 this->deformationFieldImage,
					 &deformationFieldArray_d,
					 &mask_d,
					 &transformationMatrix_d,
					 compose);
}
/* *************************************************************** */
CudaResampleImageKernel::CudaResampleImageKernel(Content *conIn, std::string name) :
		ResampleImageKernel(name)
{
	con = static_cast<CudaContent*>(conIn);

	floatingImage = con->Content::getCurrentFloating();
	warpedImage = con->Content::getCurrentWarped();

	//cuda ptrs
	floatingImageArray_d = con->getFloatingImageArray_d();
	warpedImageArray_d = con->getWarpedImageArray_d();
	deformationFieldImageArray_d = con->getDeformationFieldArray_d();
	mask_d = con->getMask_d();
	floIJKMat_d = con->getFloIJKMat_d();

	if (floatingImage->datatype != warpedImage->datatype) {
		reg_print_fct_error("CudaResampleImageKernel::CudaResampleImageKernel");
		reg_print_msg_error("Floating and warped images should have the same data type. Exit.");
		reg_exit(1);
	}

	if (floatingImage->nt != warpedImage->nt) {
		reg_print_fct_error("CudaResampleImageKernel::CudaResampleImageKernel");
		reg_print_msg_error("Floating and warped images have different dimension along the time axis. Exit.");
		reg_exit(1);
	}
}
/* *************************************************************** */
void CudaResampleImageKernel::calculate(int interp,
													 float paddingValue,
													 bool *dti_timepoint,
													 mat33 * jacMat)
{
	launchResample(this->floatingImage,
						this->warpedImage,
						interp,
						paddingValue,
						dti_timepoint,
						jacMat,
						&this->floatingImageArray_d,
						&this->warpedImageArray_d,
						&this->deformationFieldImageArray_d,
						&this->mask_d,
						&this->floIJKMat_d);
}
/* *************************************************************** */
CudaBlockMatchingKernel::CudaBlockMatchingKernel(Content *conIn, std::string name) :
		BlockMatchingKernel(name)
{
	//get CudaContent ptr
	con = static_cast<CudaContent*>(conIn);

	//get cpu ptrs
	reference = con->Content::getCurrentReference();
	params = con->Content::getBlockMatchingParams();

	//get cuda ptrs
	referenceImageArray_d = con->getReferenceImageArray_d();
	warpedImageArray_d = con->getWarpedImageArray_d();
	referencePosition_d = con->getTargetPosition_d();
	warpedPosition_d = con->getResultPosition_d();
	activeBlock_d = con->getActiveBlock_d();
	mask_d = con->getMask_d();
	referenceMat_d = con->getTargetMat_d();
}
/* *************************************************************** */
void CudaBlockMatchingKernel::calculate()
{
	block_matching_method_gpu(reference,
									  params,
									  &referenceImageArray_d,
									  &warpedImageArray_d,
									  &referencePosition_d,
									  &warpedPosition_d,
									  &activeBlock_d,
									  &mask_d,
									  &referenceMat_d);
}
/* *************************************************************** */
CudaOptimiseKernel::CudaOptimiseKernel(Content *conIn, std::string name) :
		OptimiseKernel(name)
{
	//get CudaContent ptr
	con = static_cast<CudaContent*>(conIn);

	//get cpu ptrs
	transformationMatrix = con->Content::getTransformationMatrix();
	blockMatchingParams = con->Content::getBlockMatchingParams();

	transformationMatrix_d = con->getTransformationMatrix_d();
	AR_d = con->getAR_d();
	U_d = con->getU_d();
	Sigma_d = con->getSigma_d();
	VT_d = con->getVT_d();
	lengths_d = con->getLengths_d();
	referencePos_d = con->getTargetPosition_d();
	warpedPos_d = con->getResultPosition_d();
	newWarpedPos_d = con->getNewResultPos_d();

}
/* *************************************************************** */
void CudaOptimiseKernel::calculate(bool affine, bool ils, bool cusvd)
{
	//for now. Soon we will have a GPU version of it
#ifndef CUDA7
	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, transformationMatrix, affine);
#else
	const unsigned long num_to_keep = (unsigned long) (blockMatchingParams->definedActiveBlock *
																		(blockMatchingParams->percent_to_keep / 100.0f));
	if (affine) {
		if (cusvd)
			optimize_affine3D_cuda(transformationMatrix,
										  transformationMatrix_d,
										  AR_d,
										  U_d,
										  Sigma_d,
										  VT_d,
										  lengths_d,
										  targetPos_d,
										  resultPos_d,
										  newResultPos_d,
										  blockMatchingParams->definedActiveBlock * 3,
										  12,
										  num_to_keep,
										  ils);
		else {
			this->blockMatchingParams = con->getBlockMatchingParams();
			optimize(this->blockMatchingParams, transformationMatrix, affine);
		}
	}
	else {
		this->blockMatchingParams = con->getBlockMatchingParams();
		optimize(this->blockMatchingParams, transformationMatrix, affine);
	}
#endif
}
/* *************************************************************** */
