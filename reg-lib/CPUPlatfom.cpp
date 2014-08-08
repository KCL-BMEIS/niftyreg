#include "CPUPlatform.h"
#include "CPUKernelFactory.h"
#include "Kernels.h"
CPUPlatform::CPUPlatform() {

	//register the necessary kernels for the platform 
	CPUKernelFactory* factory = new CPUKernelFactory();
	registerKernelFactory(AffineDeformationField3DKernel<void>::Name(), factory);
	registerKernelFactory(ConvolutionKernel<void>::Name(), factory);

}