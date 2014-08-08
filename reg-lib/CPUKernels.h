#pragma once
#include "kernels.h"
#include "Context.h"

#include "_reg_tools.h"

template<class FieldType> class CPUAffineDeformationField3DKernel;
template<class DTYPE> class CPUConvolutionKernel;

template<class FieldType>
class CPUAffineDeformationField3DKernel : public AffineDeformationField3DKernel<FieldType> {
public:
	CPUAffineDeformationField3DKernel(std::string name, const Platform& platform) : AffineDeformationField3DKernel<FieldType>(name, platform) {
	}

	void initialize(nifti_image *CurrentReference, nifti_image **deformationFieldImage);
	void beginComputation(Context& contexts);
	void execute(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool composition, int *mask);
	double finishComputation(Context& context);
	void ClearDeformationField(nifti_image *deformationFieldImage);

	
};

template <class DTYPE>
class CPUConvolutionKernel : public ConvolutionKernel<DTYPE> {
public:

	CPUConvolutionKernel(std::string name, const Platform& platform) : ConvolutionKernel<DTYPE>(name, platform) {
	}

	 void beginComputation(Context& context);
	 double finishComputation(Context& context);
	 void execute(nifti_image *image,float *sigma, int kernelType,int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
	 
};