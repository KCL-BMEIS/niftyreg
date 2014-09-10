#include"Kernel.h"
#include"Kernels.h"
//#include"CudaKernels.h"
//#include"CLKernels.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "CLPlatform.h"
#include "_reg_ReadWriteImage.h"
#include"cuda_runtime.h"
#include <ctime>



void mockParams(Platform* platform) {
	_reg_blockMatchingParam refParams;
	Kernel bmKernel = platform->createKernel(BlockMatchingKernel::Name(), 16);

	//init ref params
	nifti_image* reference = reg_io_ReadImageFile("mock_bm_reference.nii");
	nifti_image* warped = reg_io_ReadImageFile("mock_bm_reference.nii");
	
	int* mask = (int *)calloc(reference->nx*reference->ny*reference->nz, sizeof(int));
	bmKernel.getAs<BlockMatchingKernel>().initialize(reference, &refParams, 50, 50, mask, false);

	//run kernel
	bmKernel.getAs<BlockMatchingKernel>().execute(reference, warped, &refParams, mask);

	//not the ideal copy, but should do the job!
	int dim[8] = { 1, refParams.activeBlockNumber * 3, 1, 1, 1, 1, 1, 1 };
	nifti_image* result = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	nifti_image* target = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);

	float* tempTarget = static_cast<float*>(target->data);
	float* tempResult = static_cast<float*>(result->data);
	for (size_t i = 0; i < refParams.activeBlockNumber * 3; i++)
	{
		tempTarget[i] = refParams.targetPosition[i];
		tempResult[i] = refParams.resultPosition[i];
	}

	char* outputResultName = (char *)"mock_bm_result.nii";
	char* outputTargetName = (char *)"mock_bm_target.nii";
	reg_io_WriteImageFile(result, outputResultName);
	reg_io_WriteImageFile(target, outputTargetName);

	nifti_image_free(result);
	nifti_image_free(target);
	nifti_image_free(reference);
	nifti_image_free(warped);
	free(mask);



}

void test(Platform* platform, const char* msg) {
	_reg_blockMatchingParam refParams;
	Kernel bmKernel = platform->createKernel(BlockMatchingKernel::Name(), 16);

	//load images
	nifti_image* reference = reg_io_ReadImageFile("mock_bm_reference.nii");
	nifti_image* warped = reg_io_ReadImageFile("mock_bm_reference.nii");


	nifti_image* result = reg_io_ReadImageFile("mock_bm_result.nii");
	nifti_image* target = reg_io_ReadImageFile("mock_bm_target.nii");

	int* mask = (int *)calloc(reference->nx*reference->ny*reference->nz, sizeof(int));
	bmKernel.getAs<BlockMatchingKernel>().initialize(reference, &refParams, 50, 50, mask, false);

	//run kernel
	bmKernel.getAs<BlockMatchingKernel>().execute(reference, warped, &refParams, mask);

	//std::cout << "bn: " << refParams.activeBlockNumber << std::endl;
	//compare results
	double maxTargetDiff = reg_test_compare_arrays<float>(refParams.targetPosition, static_cast<float*>(target->data), refParams.activeBlockNumber * 3);
	double maxResultDiff = reg_test_compare_arrays<float>(refParams.resultPosition, static_cast<float*>(result->data), refParams.activeBlockNumber * 3);
	std::cout << "===================================" << msg << "===================================" << std::endl;
	std::cout << std::endl;
	std::cout << "maxTargetDiff: " << maxTargetDiff << std::endl;
	std::cout << "maxResultDiff: " << maxResultDiff << std::endl;
	std::cout << "===================================" << msg << " END ===============================" << std::endl;

	nifti_image_free(reference);
	nifti_image_free(warped);
	nifti_image_free(result);
	nifti_image_free(target);
	free(mask);
}

int main(int argc, char **argv) {

	//init platform params
	Platform *cpuPlatform = new CPUPlatform();
	Platform *cudaPlatform = new CudaPlatform();
	Platform *clPlatform = new CLPlatform();

	//mockParams(cpuPlatform);


	test(cpuPlatform, "CPU Platform");
	test(cudaPlatform, "Cuda Platform");
	cudaDeviceReset();


	return 0;

}