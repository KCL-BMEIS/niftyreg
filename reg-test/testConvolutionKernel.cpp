#include"Kernel.h"
#include"Kernels.h"
#include "Context.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "_reg_ReadWriteImage.h"


void test(Platform* platform) {

	Context *con = new Context();//temp
	Kernel* convolutionKernel = platform->createKernel(ConvolutionKernel::Name(), con);
	//init ref params
	nifti_image* input = reg_io_ReadImageFile("mock_convolution_input.nii");
	nifti_image* output = reg_io_ReadImageFile("mock_convolution_output.nii");

	bool axisToSmooth[1] = { true };
	float sigma[1] = { 1.1f };

	//run kernel
	convolutionKernel->castTo<ConvolutionKernel>()->execute(input, sigma, 0, NULL, axisToSmooth);

	//measure performance (elapsed time)

	//compare results
	double maxDifferenceBEG = reg_test_compare_images(input, output);


	//output
	std::cout << "diff:" << maxDifferenceBEG << std::endl;
	nifti_image_free(input);
	nifti_image_free(output);
}


int main(int argc, char **argv) {

	//init platform params
	Platform *cpuPlatform = new CPUPlatform();
	Platform *cudaPlatform = new CudaPlatform();

	std::cout << "testing CPU:" << std::endl;
	test(cpuPlatform);

	std::cout << "testing Cuda:" << std::endl;
	test(cudaPlatform);

	return 0;

}