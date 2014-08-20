#ifndef KERNELS_H_
#define KERNELS_H_

#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "KernelImpl.h"
#include "nifti1_io.h"
#include "_reg_blockMatching.h"//temporarily

class AffineDeformationFieldKernel : public KernelImpl {
public:
	static std::string Name() {
		return "AffineDeformationFieldKernel";
	}
	AffineDeformationFieldKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}

	virtual void initialize(nifti_image *CurrentReference, nifti_image **deformationFieldImage, const size_t sp) = 0;
	virtual void clear(nifti_image *deformationFieldImage) = 0;
	virtual void execute(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool compose = false, int *mask = NULL) = 0;
};

	class BlockMatchingKernel : public KernelImpl {
public:
	static std::string Name() {
		return "blockMatchingKernel";
	}
	BlockMatchingKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}

	virtual void  initialize(nifti_image * target,_reg_blockMatchingParam *params, int percentToKeep_block,int percentToKeep_opt, int *mask, bool runningOnGPU) = 0;
	virtual void execute(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) = 0;
};


	class ConvolutionKernel : public KernelImpl {
	public:
		static std::string Name() {
			return "ConvolutionKernel";
		}
		ConvolutionKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
		}

		virtual void execute(nifti_image *image,
							 float *sigma, int kernelType, int *mask = NULL, bool *timePoints = NULL, bool *axis = NULL) = 0;
	};

	class OptimiseKernel : public KernelImpl{
	public: 
		static std::string Name() {
			return "OptimiseKernel";
		}
		OptimiseKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
		}
		virtual void execute(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine) = 0;
	};
	class ResampleImageKernel : public KernelImpl {
	public:
		static std::string Name() {
			return "ResampleImageKernel";
		}
		ResampleImageKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
		}

		
		virtual void execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL) = 0;
	};
#endif /*KERNELS_H_*/
