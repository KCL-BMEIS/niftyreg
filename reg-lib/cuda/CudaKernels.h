#pragma once
#include "kernels.h"



class CudaAffineDeformationFieldKernel;
class CudaBlockMatchingKernel;
class CudaConvolutionKernel;
class CudaResampleImageKernel;


//Kernel functions for affine deformation field 
class CudaAffineDeformationFieldKernel : public AffineDeformationFieldKernel {
public:
	CudaAffineDeformationFieldKernel(std::string name, const Platform& platform) : AffineDeformationFieldKernel(name, platform) {
	}

	void initialize(nifti_image *CurrentReference, nifti_image **deformationFieldImage, const size_t dataSize) {}
	void clear(nifti_image *deformationFieldImage) {}
	void execute(mat44 *affineTransformation, nifti_image *deformationField, bool compose = false, int *mask = NULL);

	/*template<class FieldTYPE> void runKernel3D(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);
	template <class FieldTYPE> void runKernel2D(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool compose, int *mask);*/


};
//Kernel functions for block matching
class CudaBlockMatchingKernel : public BlockMatchingKernel {
public:

	CudaBlockMatchingKernel(std::string name, const Platform& platform) : BlockMatchingKernel(name, platform) {
	}

	void initialize(nifti_image * target, _reg_blockMatchingParam *params, int percentToKeep_block, int percentToKeep_opt, int *mask, bool runningOnGPU);
	/*template <class DTYPE>
	void setActiveBlocks(nifti_image *targetImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU);*/

	void execute(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask);
	/*template<class T> void runKernel(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask);*/

};
//a kernel function for convolution (gaussian smoothing?)
class CudaConvolutionKernel : public ConvolutionKernel {
public:

	CudaConvolutionKernel(std::string name, const Platform& platform) : ConvolutionKernel(name, platform) {
	}

	void execute(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);
	template<class T> void runKernel(nifti_image *image, float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL);

};

//kernel functions for numerical optimisation
class CudaOptimiseKernel : public OptimiseKernel {
public:

	CudaOptimiseKernel(std::string name, const Platform& platform) : OptimiseKernel(name, platform) {
	}

	void execute(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine);
};

//kernel functions for image resampling with three interpolation variations
class CudaResampleImageKernel : public ResampleImageKernel {
public:
	CudaResampleImageKernel(std::string name, const Platform& platform) : ResampleImageKernel(name, platform) {
	}


	void execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL);
	/*template <class FieldTYPE, class SourceTYPE> void runKernel(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint = NULL, mat33 * jacMat = NULL);*/
};

