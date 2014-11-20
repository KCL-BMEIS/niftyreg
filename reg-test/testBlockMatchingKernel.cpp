#include"Kernel.h"
#include"Kernels.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "CLPlatform.h"
#include "_reg_ReadWriteImage.h"
#include "Context.h"
#include "CudaContext.h"


#define REF "/home/thanasis/Documents/mockRef.nii"
#define FLO "/home/thanasis/Documents/mockFlo.nii"
#define WRP "/home/thanasis/Documents/mockWrpd.nii"

#define RES "/home/thanasis/Documents/mockRes.nii"
#define TAR "/home/thanasis/Documents/mockTar.nii"


#define BMV_PNT 50
#define INLIERS 50



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

void mockParams(Platform* platform, const unsigned int blocksPercentage, const unsigned int inliers) {




	//init ref params
	nifti_image* reference = reg_io_ReadImageFile(REF);
	nifti_image* floating = reg_io_ReadImageFile(FLO);
	nifti_image* warped = reg_io_ReadImageFile(WRP);
	int* mask = (int *)calloc(reference->nx*reference->ny*reference->nz, sizeof(int));

	reg_tools_changeDatatype<float>(reference);
	reg_tools_changeDatatype<float>(floating);

	Context *con = new Context(reference, floating, mask, sizeof(float), blocksPercentage, inliers, 1);//temp
	con->setCurrentWarped(warped);
	Kernel* bmKernel = platform->createKernel(BlockMatchingKernel::Name(), con);
	//run kernel
	bmKernel->castTo<BlockMatchingKernel>()->execute();
	_reg_blockMatchingParam* refParams = con->getBlockMatchingParams();

	//not the ideal copy, but should do the job!
	const int dim[8] = { 3, refParams->blockNumber[0]*2, refParams->blockNumber[1]*2, refParams->blockNumber[2]*2, 1, 1, 1, 1 };
	nifti_image* result = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	nifti_image* target = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
	std::cout << "4 " << refParams->blockNumber[0]* refParams->blockNumber[1]* refParams->blockNumber[2]<< std::endl;
	float* tempTarget = static_cast<float*>(target->data);
	float* tempResult = static_cast<float*>(result->data);

	std::cout<<refParams->activeBlockNumber * 3<<" - "<<refParams->blockNumber[0] * refParams->blockNumber[1] * refParams->blockNumber[2]<<std::endl;
	for (size_t i = 0; i < refParams->activeBlockNumber * 3; i++)
	{
		tempTarget[i] = refParams->targetPosition[i];
		tempResult[i] = refParams->resultPosition[i];
	}
	std::cout << "5 " << refParams->blockNumber[0] * refParams->blockNumber[1] * refParams->blockNumber[2] << std::endl;
	char* outputResultName = (char *)RES;
	char* outputTargetName = (char *)TAR;
	reg_io_WriteImageFile(result, outputResultName);
	std::cout << "4 " << refParams->blockNumber[0] * refParams->blockNumber[1] * refParams->blockNumber[2] << std::endl;
	reg_io_WriteImageFile(target, outputTargetName);
	std::cout << "6 " << refParams->blockNumber[0] * refParams->blockNumber[1] * refParams->blockNumber[2] << std::endl;
	nifti_image_free(result);
	nifti_image_free(target);
	nifti_image_free(reference);
	nifti_image_free(floating);
	free(mask);
	delete con;
	std::cout << "done" << std::endl;



}

void test(Platform* platform, const char* msg,  const unsigned int blocksPercentage, const unsigned int inliers) {

	//load images
	nifti_image* reference = reg_io_ReadImageFile(REF);
	nifti_image* floating = reg_io_ReadImageFile(FLO);
	nifti_image* warped = reg_io_ReadImageFile(WRP);

	reg_tools_changeDatatype<float>(reference);
	reg_tools_changeDatatype<float>(floating);

	nifti_image* result = reg_io_ReadImageFile(RES);
	nifti_image* target = reg_io_ReadImageFile(TAR);

	int* mask = (int *)calloc(reference->nx*reference->ny*reference->nz, sizeof(int));

	Context *con;

	if (platform->getName() == "cpu_platform")
		con = new Context(reference, reference, mask, sizeof(float), blocksPercentage, inliers, 1);//temp
	else if (platform->getName() == "cuda_platform")
		con = new CudaContext(reference, reference, mask, sizeof(float), blocksPercentage, inliers,1);//temp
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


		double targetDiff = abs(refTargetPt[0] - outTargetPt[0]) + abs(refTargetPt[1] - outTargetPt[1]) + abs(refTargetPt[2] - outTargetPt[2]);
		double resultDiff = abs(refResultPt[0] - outResultPt[0]) + abs(refResultPt[1] - outResultPt[1]) + abs(refResultPt[2] - outResultPt[2]);

		maxTargetDiff = (targetDiff > maxTargetDiff) ? targetDiff : maxTargetDiff;
		maxResultDiff = (resultDiff > maxResultDiff) ? resultDiff : maxResultDiff;
	}

	std::cout << "===================================" << msg << "===================================" << std::endl;
	std::cout << std::endl;
	/*std::cout << "maxTargetDiff: " << maxTargetDiff << std::endl;
	std::cout << "maxResultDiff: " << maxResultDiff << std::endl;*/

	std::cout << "total: " << refParams->blockNumber[0] * refParams->blockNumber[1] * refParams->blockNumber[2] << std::endl;
	std::cout << "defined: " << refParams->definedActiveBlock << std::endl;
	std::cout << "active: " << refParams->activeBlockNumber << std::endl;
	std::cout << "bn0: " << refParams->blockNumber[0] << "| bn1: " << refParams->blockNumber[1] << "| bn2: " << refParams->blockNumber[2] << std::endl;

	printf("res: %f-%f-%f\n", resultSum[0], resultSum[1], resultSum[2]);
	printf("tar: %f-%f-%f\n", targetSum[0], targetSum[1], targetSum[2]);
	std::cout << "elapsed: " << elapsed_secs << std::endl;
	std::cout << "===================================" << msg << " END ===============================" << std::endl;

	for (unsigned long i = 0; i < refParams->definedActiveBlock; i++)
	{
		
		_reg_sorted_point3D out = outSP.at(i);

		float* outTargetPt = out.target;
		float* outResultPt = out.result;

		for (unsigned long j = 0; j < refParams->definedActiveBlock; j++)
		{
			_reg_sorted_point3D ref = refSP.at(j);

			float* refTargetPt = ref.target;
			float* refResultPt = ref.result;

			targetSum[0] = outTargetPt[0] - refTargetPt[0];
			targetSum[1] = outTargetPt[1] - refTargetPt[1];
			targetSum[2] = outTargetPt[2] - refTargetPt[2];

			if (targetSum[0] == 0 && targetSum[1] == 0 && targetSum[2] == 0){
				resultSum[0] = abs(outResultPt[0] - refResultPt[0]);
				resultSum[1] = abs(outResultPt[1] - refResultPt[1]);
				resultSum[2] = abs(outResultPt[2] - refResultPt[2]);
				if (resultSum[0] >0 || resultSum[1] > 0 || resultSum[2]>0)
					printf("i: %d | j: %d | (dif: %f-%f-%f) | (out: %f, %f, %f) | (ref: %f, %f, %f)\n", i, j, resultSum[0], resultSum[1], resultSum[2], outResultPt[0], outResultPt[1], outResultPt[2], refResultPt[0], refResultPt[1], refResultPt[2]);
			}
		}

		

		

		


		


		/*double targetDiff = abs(refTargetPt[0] - outTargetPt[0]) + abs(refTargetPt[1] - outTargetPt[1]) + abs(refTargetPt[2] - outTargetPt[2]);
		double resultDiff = abs(refResultPt[0] - outResultPt[0]) + abs(refResultPt[1] - outResultPt[1]) + abs(refResultPt[2] - outResultPt[2]);

		maxTargetDiff = (targetDiff > maxTargetDiff) ? targetDiff : maxTargetDiff;
		maxResultDiff = (resultDiff > maxResultDiff) ? resultDiff : maxResultDiff;*/
	}

	

	nifti_image_free(reference);
	nifti_image_free(floating);
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

//	mockParams(cpuPlatform, BMV_PNT, INLIERS);




	test(cpuPlatform, "CPU Platform", BMV_PNT, INLIERS);

	test(cudaPlatform, "Cuda Platform", BMV_PNT, INLIERS);

	test(clPlatform, "Cl Platform", BMV_PNT, INLIERS);

	//cudaDeviceReset();


	return 0;

}
