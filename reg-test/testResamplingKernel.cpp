#include"Kernel.h"
#include"Kernels.h"
#include "CPUPlatform.h"
#include "_reg_ReadWriteImage.h"

#define LINEAR_CODE 1
#define CUBIC_CODE  3
#define NN_CODE     0

#define LINEAR_FILENAME "mock_resample_linear_warped.nii"
#define CUBIC_FILENAME "mock_resample_cubic_warped.nii"
#define NN_FILENAME "mock_resample_nn_warped.nii"


float test(Platform* platform, const unsigned int interp) {
	Kernel resamplingKernel = platform->createKernel(ResampleImageKernel::Name(), 16);

	//init ref params
	nifti_image* CurrentFloating = reg_io_ReadImageFile("mock_resample_input_float.nii");
	nifti_image* CurrentWarped = reg_io_ReadImageFile("mock_resample_input_warped.nii");
	nifti_image* deformationFieldImage = reg_io_ReadImageFile("mock_affine_output.nii");

	int* CurrentReferenceMask = NULL;

	nifti_image* output;
	if(interp == LINEAR_CODE )
		output = reg_io_ReadImageFile(LINEAR_FILENAME);
	else if( interp == CUBIC_CODE )
		output = reg_io_ReadImageFile(CUBIC_FILENAME);
	else
		output = reg_io_ReadImageFile(NN_FILENAME);

	//run kernel
	resamplingKernel.getAs<ResampleImageKernel>().execute(CurrentFloating, CurrentWarped, deformationFieldImage, CurrentReferenceMask, interp, 0);

	//measure performance (elapsed time)

	//compare results
	double maxDiff = reg_test_compare_images(CurrentWarped, output);




	nifti_image_free(CurrentFloating);
	nifti_image_free(CurrentWarped);
	free(CurrentReferenceMask);

	return maxDiff;
}

int main(int argc, char **argv) {

	//init platform params
	Platform *platform = new CPUPlatform();
	
	const float nnDiff = test(platform, 0);
	const float linearDiff = test(platform, 1);
	const float cubicDiff = test(platform, 3);


	//output
	std::cout << "nnDiff:" << nnDiff << std::endl;
	std::cout << "linearDiff:" << linearDiff << std::endl;
	std::cout << "cubicDiff:" << cubicDiff << std::endl;

	return 0;

}