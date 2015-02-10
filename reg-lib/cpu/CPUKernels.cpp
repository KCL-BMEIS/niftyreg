#include "CPUKernels.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"


/* *************************************************************** */
void CPUAffineDeformationFieldKernel::calculate(bool compose)
{
	reg_affine_getDeformationField(this->affineTransformation,
											 this->deformationFieldImage,
											 compose,
											 this->mask);
}
/* *************************************************************** */
void CPUConvolutionKernel::calculate(nifti_image *image,
												 float *sigma,
												 int kernelType,
												 int *mask,
												 bool *timePoint,
												 bool *axis)
{
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}
/* *************************************************************** */
void CPUBlockMatchingKernel::calculate()
{
	block_matching_method(this->reference, this->warped, this->params, this->mask);
}
/* *************************************************************** */
void CPUOptimiseKernel::calculate(bool affine, bool ils, bool svd)
{
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
/* *************************************************************** */
void CPUResampleImageKernel::calculate(int interp,
													float paddingValue,
													bool *dti_timepoint,
													mat33 * jacMat)
{
	reg_resampleImage(this->floatingImage,
							this->warpedImage,
							this->deformationField,
							this->mask,
							interp,
							paddingValue,
							dti_timepoint,
							jacMat);
}
/* *************************************************************** */
