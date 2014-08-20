

#include "Context.h"
#include "Kernels.h"
#include "Platform.h"

#include <iostream>


using namespace std;

Context::Context(Platform* platformIn):platform(platformIn){

	// Find the list of kernels required.
	//Maybe we could be passing a polymorphic reg object, which will be storing a list of data including a kernel map for the registration method
	vector<string> kernelNames;

	// Select a platform to use.
	//get available platforms somehow. Have a speed estimate. Sort them from fastest to slowest. Pick fastest

	//derive this from images
	const unsigned int dType = 16;
	// Create and initialize kernels and other objects.
	affineTransformation3DKernel = platform->createKernel(AffineDeformationFieldKernel::Name(), dType);
	convolutionKernel = platform->createKernel(ConvolutionKernel::Name(), dType);
	blockMatchingKernel = platform->createKernel(BlockMatchingKernel::Name(), dType);
	resamplingKernel = platform->createKernel(ResampleImageKernel::Name(), dType);
	optimiseKernel = platform->createKernel(OptimiseKernel::Name(), dType);


}
void Context::shout() {
	std::cout << "context listens" << std::endl;
	Platform *platform = new Platform();
	platform->shout();
}



