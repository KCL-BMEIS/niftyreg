#include"Kernel.h"
#include"Kernels.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "CLPlatform.h"
#include "_reg_ReadWriteImage.h"
#include "Context.h"
#include <string>

#define LINEAR_CODE 1
#define CUBIC_CODE  3
#define NN_CODE     0

#define LINEAR_FILENAME "mock_resample_linear_warped.nii"
#define CUBIC_FILENAME "mock_resample_cubic_warped.nii"
#define NN_FILENAME "mock_resample_nn_warped.nii"


float test(Platform* platform, const unsigned int interp, std::string message) {
	std::cout << "================================" << std::endl;
	

	//init ref params
	nifti_image* CurrentFloating = reg_io_ReadImageFile("mock_resample_input_float.nii");
	nifti_image* CurrentWarped = reg_io_ReadImageFile("mock_resample_input_warped.nii");
	nifti_image* deformationFieldImage = reg_io_ReadImageFile("mock_affine_output.nii");
	int* CurrentReferenceMask = NULL;
	nifti_image* mockRef = reg_io_ReadImageFile(CUBIC_FILENAME);//any image doesn't matter


	std::cout << "conb" << std::endl;
	Context *con = new Context(mockRef, CurrentFloating, CurrentReferenceMask, sizeof(float), 50, 50);//temp
	con->setCurrentWarped(CurrentWarped);
	con->setCurrentDeformationField(deformationFieldImage);

	std::cout << "kernel" << std::endl;
	Kernel resamplingKernel = platform->createKernel(ResampleImageKernel::Name(), con);

	

	nifti_image* output;
	if( interp == LINEAR_CODE )
		output = reg_io_ReadImageFile(LINEAR_FILENAME);
	else if( interp == CUBIC_CODE )
		output = reg_io_ReadImageFile(CUBIC_FILENAME);
	else
		output = reg_io_ReadImageFile(NN_FILENAME);
	std::cout << "exe" << std::endl;
	//run kernel
	resamplingKernel.getAs<ResampleImageKernel>().execute( interp, 0);
	std::cout << "dne" << std::endl;

	//measure performance (elapsed time)
	
	//compare results
	double maxDiff = reg_test_compare_images(CurrentWarped, output);

	//output
	std::cout << message << maxDiff << std::endl;


	nifti_image_free(CurrentFloating);
	nifti_image_free(CurrentWarped);
	nifti_image_free(deformationFieldImage);
	nifti_image_free(output);
	free(CurrentReferenceMask);



	return maxDiff;
}

int main(int argc, char **argv) {

	//init platform params
	Platform *platform = new CPUPlatform();
	Platform *cudaPlatform = new CudaPlatform();
	Platform *clPlatform = new CLPlatform();

	const float nnDiff = test(platform, 0, "nnDiff:");
	const float linearDiff = test(platform, 1, "linear diff: ");
	const float cubicDiff  = test(platform, 3, "cubic diff: ");

	const float linearDiffCuda = test(cudaPlatform, 1, "cuda linear: ");
	const float nnDiffCuda = test(cudaPlatform, 0, "cuda nn: ");
	const float cubicDiffCuda = test(cudaPlatform, 3, "cuda cubic: ");

	const float linearDiffCl = test(clPlatform, 1, "cl linear: ");
	const float nnDiffCl = test(clPlatform, 0, "cl nn: ");
	const float cubicDiffCl = test(clPlatform, 3, "cl cubic: ");


	return 0;

}