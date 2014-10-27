#include"Kernel.h"
#include"Kernels.h"
//#include"CudaKernels.h"
//#include"CLKernels.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "CLPlatform.h"
#include "_reg_ReadWriteImage.h"
#include"cuda_runtime.h"
//#include "Context.h"
#include "CpuContext.h"
#include "CudaContext.h"
#include <ctime>



struct _reg_sorted_point3D
{
	float target[3];
	float result[3];

	double distance;

	_reg_sorted_point3D(float * t, float * r, double d)
		:distance(d)
	{
		target[0] = t[0];
		target[1] = t[1];
		target[2] = t[2];

		result[0] = r[0];
		result[1] = r[1];
		result[2] = r[2];
	}

	bool operator <(const _reg_sorted_point3D &sp) const
	{
		return (sp.distance < distance);
	}
};

void mockParams(Platform* platform) {




	//init ref params
	nifti_image* reference = reg_io_ReadImageFile("mock_bm_reference.nii");
	nifti_image* warped = reg_io_ReadImageFile("mock_bm_warped.nii");
	int* mask = (int *)calloc(reference->nx*reference->ny*reference->nz, sizeof(int));

	Context *con = new CpuContext(reference, reference, mask, sizeof(float), 50, 50);//temp
	con->setCurrentWarped(warped);

	Kernel* bmKernel = platform->createKernel(BlockMatchingKernel::Name(), con);

	//run kernel
	bmKernel->castTo<BlockMatchingKernel>()->execute();

	_reg_blockMatchingParam* refParams = con->getBlockMatchingParams();

	//not the ideal copy, but should do the job!
	int dim[8] = { 1, refParams->activeBlockNumber * 3, 1, 1, 1, 1, 1, 1 };
	nifti_image* result = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	nifti_image* target = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);

	float* tempTarget = static_cast<float*>(target->data);
	float* tempResult = static_cast<float*>(result->data);
	for (size_t i = 0; i < refParams->activeBlockNumber * 3; i++)
	{
		tempTarget[i] = refParams->targetPosition[i];
		tempResult[i] = refParams->resultPosition[i];
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
	free(con);



}

void test(Platform* platform, const char* msg, const unsigned int arch) {

	//load images
	nifti_image* reference = reg_io_ReadImageFile("mock_bm_reference.nii");
	nifti_image* warped = reg_io_ReadImageFile("mock_bm_warped.nii");

	nifti_image* result = reg_io_ReadImageFile("mock_bm_result.nii");
	nifti_image* target = reg_io_ReadImageFile("mock_bm_target.nii");

	int* mask = (int *)calloc(reference->nx*reference->ny*reference->nz, sizeof(int));

	Context *con;

	if (platform->getName() == "cpu_platform")
		con = new Context(reference, reference, mask, sizeof(float), 50, 50);//temp
	else if (platform->getName() == "cuda_platform")
		con = new CudaContext(reference, reference, mask, sizeof(float), 50, 50);//temp
	else con = new Context();
	con->setCurrentWarped(warped);

	Kernel* bmKernel = platform->createKernel(BlockMatchingKernel::Name(), con);

	clock_t begin = clock();
	//run kernel
	bmKernel->castTo<BlockMatchingKernel>()->execute();
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	_reg_blockMatchingParam* refParams = con->getBlockMatchingParams();

	//std::cout << "bn: " << refParams.activeBlockNumber << std::endl;
	//compare results


	std::vector<_reg_sorted_point3D> refSP;
	std::vector<_reg_sorted_point3D> outSP;

	float* targetData = static_cast<float*>(target->data);
	float* resultData = static_cast<float*>(result->data);

	for (unsigned long j = 0; j < refParams->definedActiveBlock * 3; j += 3)
	{
		refSP.push_back(_reg_sorted_point3D(&(refParams->targetPosition[j]),
			&(refParams->resultPosition[j]), 0.0f));

		outSP.push_back(_reg_sorted_point3D(&(targetData[j]),
			&(resultData[j]), 0.0f));
	}


	double maxTargetDiff = /*reg_test_compare_arrays<float>(refParams->targetPosition, static_cast<float*>(target->data), refParams->definedActiveBlock * 3)*/0.0;
	double maxResultDiff = /*reg_test_compare_arrays<float>(refParams->resultPosition, static_cast<float*>(result->data), refParams->definedActiveBlock * 3)*/0.0;

	double targetSum[3] = /*reg_test_compare_arrays<float>(refParams->targetPosition, static_cast<float*>(target->data), refParams->definedActiveBlock * 3)*/{ 0.0, 0.0, 0.0 };
	double resultSum[3] = /*reg_test_compare_arrays<float>(refParams->resultPosition, static_cast<float*>(result->data), refParams->definedActiveBlock * 3)*/{ 0.0, 0.0, 0.0 };
	
	
	//a better test will be to sort the 3d points and test the diff of each one!
	for (unsigned long i = 0; i < refParams->definedActiveBlock; i++)
	{
		_reg_sorted_point3D ref = refSP.at(i);
		_reg_sorted_point3D out = outSP.at(i);

		float* refTargetPt = ref.target;
		float* refResultPt = ref.result;

		float* outTargetPt = out.target;
		float* outResultPt = out.result;

		targetSum[0] += outTargetPt[0] - refTargetPt[0];
		targetSum[1] += outTargetPt[1] - refTargetPt[1];
		targetSum[2] += outTargetPt[2] - refTargetPt[2];


		resultSum[0] += outResultPt[0] - refResultPt[0];
		resultSum[1] += outResultPt[1] - refResultPt[1];
		resultSum[2] += outResultPt[2] - refResultPt[2];


		/*double targetDiff = abs(refTargetPt[0] - outTargetPt[0]) + abs(refTargetPt[1] - outTargetPt[1]) + abs(refTargetPt[2] - outTargetPt[2]);
		double resultDiff = abs(refResultPt[0] - outResultPt[0]) + abs(refResultPt[1] - outResultPt[1]) + abs(refResultPt[2] - outResultPt[2]);

		maxTargetDiff = (targetDiff > maxTargetDiff) ? targetDiff : maxTargetDiff;
		maxResultDiff = (resultDiff > maxResultDiff) ? resultDiff : maxResultDiff;*/
	}
	std::cout << "===================================" << msg << "===================================" << std::endl;
	std::cout << std::endl;
	/*std::cout << "maxTargetDiff: " << maxTargetDiff << std::endl;
	std::cout << "maxResultDiff: " << maxResultDiff << std::endl;*/

	printf("res: %f-%f-%f\n", resultSum[0], resultSum[1], resultSum[2]);
	printf("tar: %f-%f-%f\n", targetSum[0], targetSum[1], targetSum[2]);
	std::cout << "elapsed: " << elapsed_secs << std::endl;
	std::cout << "===================================" << msg << " END ===============================" << std::endl;

	nifti_image_free(reference);
	//nifti_image_free(warped);
	nifti_image_free(result);
	nifti_image_free(target);
	free(mask);
	delete con;
}

int main(int argc, char **argv) {

	//init platform params
	Platform *cpuPlatform = new CPUPlatform();
	Platform *cudaPlatform = new CudaPlatform();
	Platform *clPlatform = new CLPlatform();

	//mockParams(cpuPlatform);

	test(cudaPlatform, "Cuda Platform", 1);
	test(cpuPlatform, "CPU Platform", 0);

	//cudaDeviceReset();


	return 0;

}