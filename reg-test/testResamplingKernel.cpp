#include"Kernel.h"
#include"Kernels.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "CLPlatform.h"
#include "_reg_ReadWriteImage.h"
#include "Context.h"
#include "CudaContext.h"
#include <string>

#define LINEAR_CODE 1
#define CUBIC_CODE  3
#define NN_CODE     0

#define LINEAR_FILENAME "mock_resample_linear_warped.nii"
#define CUBIC_FILENAME "mock_resample_cubic_warped.nii"
#define NN_FILENAME "mock_resample_nn_warped.nii"

#define CPUX 0
#define CUDA 1
#define OCLX 2


float test(const unsigned int platformCode, const unsigned int interp, std::string message) {
	std::cout << "================================" << std::endl;
	Platform *platform;
	if (platformCode == CPUX)
		platform = new CPUPlatform();
	else if (platformCode == CUDA)
		platform = new CudaPlatform();
	else
		platform = new CLPlatform();

	//init ref params
	nifti_image* CurrentFloating = reg_io_ReadImageFile("mock_resample_input_float.nii");
	nifti_image* CurrentWarped = reg_io_ReadImageFile("mock_resample_input_warped.nii");
	nifti_image* deformationFieldImage = reg_io_ReadImageFile("mock_affine_output.nii");
	nifti_image* mockRef = reg_io_ReadImageFile(CUBIC_FILENAME);//any image doesn't matter
	int* CurrentReferenceMask = (int *)calloc(mockRef->nx*mockRef->ny*mockRef->nz, sizeof(int));

	std::cout << platform->getName() << std::endl;

	Context *con;

	if (platform->getName() == "cpu_platform")
		con = new Context(mockRef, CurrentFloating, CurrentReferenceMask, sizeof(float), 50, 50);
	else if (platform->getName() == "cuda_platform")
		con = new CudaContext(mockRef, CurrentFloating, CurrentReferenceMask, sizeof(float), 50, 50);
	else 
		con = new Context(mockRef, CurrentFloating, CurrentReferenceMask, sizeof(float), 50, 50);

	con->setCurrentWarped(CurrentWarped);
	con->setCurrentDeformationField(deformationFieldImage);


	Kernel resamplingKernel = platform->createKernel(ResampleImageKernel::Name(), con);

	nifti_image* output;
	if( interp == LINEAR_CODE )
		output = reg_io_ReadImageFile(LINEAR_FILENAME);
	else if( interp == CUBIC_CODE )
		output = reg_io_ReadImageFile(CUBIC_FILENAME);
	else
		output = reg_io_ReadImageFile(NN_FILENAME);

	//run kernel
	resamplingKernel.getAs<ResampleImageKernel>().execute( interp, 0);

	//measure performance (elapsed time)
	
	//compare results
	double maxDiff = reg_test_compare_images(con->getCurrentWarped(), output);

	//output
	std::cout << message << maxDiff << std::endl;

	nifti_image_free(mockRef);
	nifti_image_free(CurrentFloating);
	nifti_image_free(output);
	free(CurrentReferenceMask);

	delete con;
	delete platform;

	return maxDiff;
}

int main(int argc, char **argv) {

	//init platform params


	const float nnDiff = test(CPUX, NN_CODE, "nnDiff:");
	const float linearDiff = test(CPUX, LINEAR_CODE, "linear diff: ");
	const float cubicDiff = test(CPUX, CUBIC_CODE, "cubic diff: ");


	const float nnDiffCuda = test(CUDA, NN_CODE, "cuda nn: ");
	const float linearDiffCuda = test(CUDA, LINEAR_CODE, "cuda linear: ");
	const float cubicDiffCuda = test(CUDA, CUBIC_CODE, "cuda cubic: ");

	const float linearDiffCl = test(OCLX, LINEAR_CODE, "cl linear: ");
	const float nnDiffCl = test(OCLX, NN_CODE, "cl nn: ");
	const float cubicDiffCl = test(OCLX, CUBIC_CODE, "cl cubic: ");



	return 0;

}