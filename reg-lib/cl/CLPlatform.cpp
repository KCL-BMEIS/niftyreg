#include "CLPlatform.h"
#include "CLKernelFactory.h"
#include "kernels.h"
CLPlatform::CLPlatform() {


	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_device_id device = 0;
	//register the necessary kernels for the platform 
	CLKernelFactory* factory = new CLKernelFactory();
	registerKernelFactory(AffineDeformationFieldKernel::Name(), factory);
	registerKernelFactory(BlockMatchingKernel::Name(), factory);
	registerKernelFactory(ConvolutionKernel::Name(), factory);
	registerKernelFactory(OptimiseKernel::Name(), factory);
	registerKernelFactory(ResampleImageKernel::Name(), factory);
}
