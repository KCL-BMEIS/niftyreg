#include "CudaKernels.h"
#include "CudaKernelFuncs.h"
#include "_reg_tools.h"
#include "_reg_blockMatching_gpu.h"
#include "_reg_blockMatching.h"
#include"_reg_resampling.h"
#include"_reg_globalTransformation.h"

//------------------------------------------------------------------------------------------------------------------------
//..................CudaConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CudaConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	//cpu cheat
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}

//==============================Cuda Affine Kernel================================================================
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Context* conIn, std::string nameIn) :
		AffineDeformationFieldKernel(nameIn) {

	con = static_cast<CudaContext*>(conIn);

	//get necessary cpu ptrs
	this->deformationFieldImage = con->Context::getCurrentDeformationField();
	this->affineTransformation = con->Context::getTransformationMatrix();

	//get necessary cuda ptrs
	mask_d = con->getMask_d();
	deformationFieldArray_d = con->getDeformationFieldArray_d();
	transformationMatrix_d = con->getTransformationMatrix_d();

}

void CudaAffineDeformationFieldKernel::compare(bool compose) {


	nifti_image* gpuField = con->getCurrentDeformationField();
	float* gpuData = static_cast<float*>(gpuField->data);

	nifti_image *cpuField = nifti_copy_nim_info(gpuField);
	cpuField->data = (void *) malloc(gpuField->nvox * gpuField->nbyper);

	reg_affine_getDeformationField(con->Context::getTransformationMatrix(), cpuField, compose, con->Context::getCurrentReferenceMask());
	float* cpuData = static_cast<float*>(cpuField->data);

	int count = 0;
	float threshold = 0.000015f;

	for (unsigned long i = 0; i < gpuField->nvox; i++) {
		float base = fabs(cpuData[i]) > 1 ? fabs(cpuData[i]) : fabs(cpuData[i]) + 1;
		if (fabs(cpuData[i] - gpuData[i]) / base > threshold) {
			printf("i: %d | cpu: %f | gpu: %f\n", i, cpuData[i], gpuData[i]);
			count++;
		}
	}

	std::cout << count << "[DEFCHECK]: pixels above threshold: " << threshold << std::endl;
	if (count > 0)
		std::cin.get();
}

void CudaAffineDeformationFieldKernel::calculate(bool compose) {
	launchAffine(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, &transformationMatrix_d, compose);
#ifndef NDEBUG
	compare(compose);
#endif

}
//------------------------------------------------------------------------------------

//==============================Cuda Resamlple Kernel================================================================

CudaResampleImageKernel::CudaResampleImageKernel(Context* conIn, std::string name) :
		ResampleImageKernel(name) {

	con = static_cast<CudaContext*>(conIn);

	floatingImage = con->Context::getCurrentFloating();
	warpedImage = con->Context::getCurrentWarped();

	//cuda ptrs
	floatingImageArray_d = con->getFloatingImageArray_d();
	warpedImageArray_d = con->getWarpedImageArray_d();
	deformationFieldImageArray_d = con->getDeformationFieldArray_d();
	mask_d = con->getMask_d();
	floIJKMat_d = con->getFloIJKMat_d();

	if (floatingImage->datatype != warpedImage->datatype) {
		printf("[NiftyReg ERROR] reg_resampleImage\tSource and result image should have the same data type\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

	if (floatingImage->nt != warpedImage->nt) {
		printf("[NiftyReg ERROR] reg_resampleImage\tThe source and result images have different dimension along the time axis\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

}

void CudaResampleImageKernel::calculate(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {
	launchResample(floatingImage, warpedImage, interp, paddingValue, dti_timepoint, jacMat, &floatingImageArray_d, &warpedImageArray_d, &deformationFieldImageArray_d, &mask_d, &floIJKMat_d);
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//==============================Cuda Block Matching Kernel================================================================
CudaBlockMatchingKernel::CudaBlockMatchingKernel(Context* conIn, std::string name) :
		BlockMatchingKernel(name) {

	//get CudaContext ptr
	con = static_cast<CudaContext*>(conIn);

	//get cpu ptrs
	target = con->Context::getCurrentReference();
	params = con->Context::getBlockMatchingParams();

	//get cuda ptrs
	targetImageArray_d = con->getReferenceImageArray_d();
	resultImageArray_d = con->getWarpedImageArray_d();
	targetPosition_d = con->getTargetPosition_d();
	resultPosition_d = con->getResultPosition_d();
	activeBlock_d = con->getActiveBlock_d();
	mask_d = con->getMask_d();
	targetMat_d = con->getTargetMat_d();

}
void CudaBlockMatchingKernel::compare() {
	nifti_image* referenceImage = con->Context::getCurrentReference();
	nifti_image* warpedImage = con->getCurrentWarped(16);
	int* mask = con->getCurrentReferenceMask();
	_reg_blockMatchingParam *refParams = con->getBlockMatchingParams();

	_reg_blockMatchingParam *cpu = new _reg_blockMatchingParam();
	initialise_block_matching_method(referenceImage, cpu, 50, 50, 1, mask, false);

	block_matching_method(referenceImage, warpedImage, cpu, mask);

	int count = 0, count2 = 0;
	float* cpuTargetData = static_cast<float*>(cpu->targetPosition);
	float* cpuResultData = static_cast<float*>(cpu->resultPosition);

	float* cudaTargetData = static_cast<float*>(refParams->targetPosition);
	float* cudaResultData = static_cast<float*>(refParams->resultPosition);

	double maxTargetDiff =0.0;
	double maxResultDiff = 0.0;

	double targetSum[3] ={ 0.0, 0.0, 0.0 };
	double resultSum[3] ={ 0.0, 0.0, 0.0 };


	for (unsigned long i = 0; i < refParams->definedActiveBlock; i++) {

		float cpuTargetPt[3] = { cpuTargetData[3 * i + 0], cpuTargetData[3 * i + 1], cpuTargetData[3 * i + 2] };
		float cpuResultPt[3] = { cpuResultData[3 * i + 0], cpuResultData[3 * i + 1], cpuResultData[3 * i + 2] };

		bool found = false;
		for (unsigned long j = 0; j < refParams->definedActiveBlock; j++) {
			float cudaTargetPt[3] = { cudaTargetData[3 * j + 0], cudaTargetData[3 * j + 1], cudaTargetData[3 * j + 2] };
			float cudaResultPt[3] = { cudaResultData[3 * j + 0], cudaResultData[3 * j + 1], cudaResultData[3 * j + 2] };

			targetSum[0] = fabs(cpuTargetPt[0] - cudaTargetPt[0]);
			targetSum[1] = fabs(cpuTargetPt[1] - cudaTargetPt[1]);
			targetSum[2] = fabs(cpuTargetPt[2] - cudaTargetPt[2]);

			const float threshold = 0.00001f;
			if (targetSum[0] <= threshold && targetSum[1] <= threshold && targetSum[2] <= threshold) {

				resultSum[0] = fabs(cpuResultPt[0] - cudaResultPt[0]);
				resultSum[1] = fabs(cpuResultPt[1] - cudaResultPt[1]);
				resultSum[2] = fabs(cpuResultPt[2] - cudaResultPt[2]);
				found = true;
				if (resultSum[0] > 0.000001f || resultSum[1] > 0.000001f || resultSum[2] > 0.000001f) {
					mat44 mat = referenceImage->qto_ijk;
					float out[3], res[3];
					reg_mat44_mul(&mat, cudaTargetPt, out);
					reg_mat44_mul(&mat, cudaResultPt, res);
					printf("i: %lu | j: %lu | target: (%f-%f-%f) | (dif: %f-%f-%f) | (cpu: %f, %f, %f) | (ref: %f, %f, %f) | (%f-%F-%f)\n", i, j, out[0], out[1], out[2],resultSum[0] , resultSum[1] , resultSum[2], cpuResultPt[0], cpuResultPt[1], cpuResultPt[2], cudaResultPt[0], cudaResultPt[1], cudaResultPt[2],  res[0], res[1], res[2]);
					count2++;
				}
			}
		}
		if (!found) {
			mat44 mat = referenceImage->qto_ijk;
			float out[3];
			reg_mat44_mul(&mat, cpuTargetPt, out);
			printf("i: %lu has no match | target: %f-%f-%f\n", i, out[0] / 4, out[1] / 4, out[2] / 4);
			count++;
		}
	}

	std::cout << count << "BM targets have no match" << std::endl;
	std::cout << count2 << "BM results have no match" << std::endl;
	if (count > 0)
		exit(0);
	if (count2 > 0) {
		std::cout << "Press a key to continue..!!!!!!!!" << std::endl;
		std::cin.get();
	}
}

void CudaBlockMatchingKernel::calculate(int range) {

	block_matching_method_gpu(target, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d, &targetMat_d);
#ifndef NDEBUG
	compare();
#endif
}
//===================================================================================================================================================================
CudaOptimiseKernel::CudaOptimiseKernel(Context* conIn, std::string name) :
		OptimiseKernel(name) {

	//get CudaContext ptr
	con = static_cast<CudaContext*>(conIn);

	//get cpu ptrs
	transformationMatrix = con->Context::getTransformationMatrix();
	blockMatchingParams = con->Context::getBlockMatchingParams();

}

void CudaOptimiseKernel::calculate(bool affine, bool ils) {

	//for now. Soon we will have a GPU version of it
	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, con->Context::getTransformationMatrix(), affine, ils);
}

