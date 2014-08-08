#ifndef KERNELS_H_
#define KERNELS_H_

#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "KernelImpl.h"
#include "nifti1_io.h"

	class blockMatching3DKernel : public KernelImpl {
public:
	static std::string Name() {
		return "blockMatching3DKernel";
	}
	blockMatching3DKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
	}

	virtual void initialize() = 0;
	virtual void beginComputation(Context& context) = 0;
	virtual double finishComputation(Context& context) = 0;
};
	template <class FieldType>
	class AffineDeformationField3DKernel : public KernelImpl {
	public:
		static std::string Name() {
			return "AffineDeformationField3DKernel";
		}
		AffineDeformationField3DKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
		}

		virtual void initialize(nifti_image *CurrentReference, nifti_image **deformationFieldImage) = 0;
		virtual void beginComputation(Context& context) = 0;
		virtual double finishComputation(Context& context) = 0;
		virtual void execute(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool composition, int *mask) = 0;
	};
	template <class DTYPE>
	class ConvolutionKernel : public KernelImpl {
	public:
		static std::string Name() {
			return "ConvolutionKernel";
		}
		ConvolutionKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
		}

		virtual void beginComputation(Context& context) = 0;
		virtual double finishComputation(Context& context) = 0;
		virtual void execute(nifti_image *image,
							 float *sigma,
							 int kernelType,
							 int *mask = NULL,
							 bool *timePoints = NULL,
							 bool *axis = NULL) = 0;
	};
#endif /*KERNELS_H_*/
