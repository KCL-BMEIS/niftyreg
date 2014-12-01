#include "CudaKernels.h"
#include "CudaKernelFuncs.h"
#include "_reg_tools.h"

//debugging
#include"_reg_resampling.h"
#include"_reg_globalTransformation.h"
//----

//------------------------------------------------------------------------------------------------------------------------
//..................CudaConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CudaConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	//cpu cheat
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoint, axis);
}


//==============================Cuda Affine Kernel================================================================
CudaAffineDeformationFieldKernel::CudaAffineDeformationFieldKernel(Context* conIn, std::string nameIn) :
		AffineDeformationFieldKernel(nameIn) {

	con = ((CudaContext*) conIn);

	this->deformationFieldImage = con->CurrentDeformationField;
	this->affineTransformation = con->transformationMatrix;

	mask_d = con->getMask_d();
	deformationFieldArray_d = con->getDeformationFieldArray_d();

}


void CudaAffineDeformationFieldKernel::execute(bool compose) {
	reg_affine_getDeformationField(this->affineTransformation, con->CurrentDeformationField, compose, con->CurrentReferenceMask);
//	launchAffine(this->affineTransformation, this->deformationFieldImage, &deformationFieldArray_d, &mask_d, compose);

}
//------------------------------------------------------------------------------------

//==============================Cuda Resamlple Kernel================================================================

CudaResampleImageKernel::CudaResampleImageKernel(Context* conIn, std::string name) :
		ResampleImageKernel(name) {

	con = static_cast<CudaContext*>(conIn);

	floatingImage = con->CurrentFloating;
	warpedImage = con->CurrentWarped;

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

void CudaResampleImageKernel::execute(int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {
//	launchResample(floatingImage, warpedImage, interp, paddingValue, dti_timepoint, jacMat, &floatingImageArray_d, &warpedImageArray_d, &deformationFieldImageArray_d, &mask_d, &floIJKMat_d);
	reg_resampleImage(con->CurrentFloating, con->CurrentWarped, con->CurrentDeformationField, con->CurrentReferenceMask, interp, paddingValue, dti_timepoint, jacMat);
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//==============================Cuda Block Matching Kernel================================================================
CudaBlockMatchingKernel::CudaBlockMatchingKernel(Context* conIn, std::string name) :
		BlockMatchingKernel(name) {

	con = ((CudaContext*) conIn);

	target = con->CurrentReference;
	params = con->blockMatchingParams;

	targetImageArray_d = con->getReferenceImageArray_d();
	resultImageArray_d = con->getWarpedImageArray_d();
	targetPosition_d = con->getTargetPosition_d();
	resultPosition_d = con->getResultPosition_d();
	activeBlock_d = con->getActiveBlock_d();
	mask_d = con->getMask_d();
	targetMat_d = con->getTargetMat_d();

}
void CudaBlockMatchingKernel::compare(nifti_image *referenceImage,nifti_image *warpedImage,int* mask, _reg_blockMatchingParam *refParams) {

	_reg_blockMatchingParam *cpu = new _reg_blockMatchingParam();
	initialise_block_matching_method(referenceImage, cpu, 50, 50, 1, mask, false);
	block_matching_method(referenceImage, warpedImage, cpu, mask);

	float* cpuTargetData = static_cast<float*>(cpu->targetPosition);
	float* cpuResultData = static_cast<float*>(cpu->resultPosition);

	float* cudaTargetData = static_cast<float*>(refParams->targetPosition);
	float* cudaResultData = static_cast<float*>(refParams->resultPosition);

	double maxTargetDiff = /*reg_test_compare_arrays<float>(refParams->targetPosition, static_cast<float*>(target->data), refParams->definedActiveBlock * 3)*/0.0;
	double maxResultDiff = /*reg_test_compare_arrays<float>(refParams->resultPosition, static_cast<float*>(result->data), refParams->definedActiveBlock * 3)*/0.0;

	double targetSum[3] = /*reg_test_compare_arrays<float>(refParams->targetPosition, static_cast<float*>(target->data), refParams->definedActiveBlock * 3)*/{ 0.0, 0.0, 0.0 };
	double resultSum[3] = /*reg_test_compare_arrays<float>(refParams->resultPosition, static_cast<float*>(result->data), refParams->definedActiveBlock * 3)*/{ 0.0, 0.0, 0.0 };

	//a better test will be to sort the 3d points and test the diff of each one!
	/*for (unsigned int i = 0; i < refParams->definedActiveBlock*3; i++) {

	 printf("i: %d target|%f-%f| result|%f-%f|\n", i, cpuTargetData[i], cudaTargetData[i], cpuResultData[i], cudaResultData[i]);
	 }*/
	std::cout<<"cpu definedActive: "<<cpu->definedActiveBlock<<" | cuda definedActive: "<<refParams->definedActiveBlock<<std::endl;
	std::cout<<"cpu active: "<<cpu->activeBlockNumber<<" | cuda active: "<<refParams->activeBlockNumber<<std::endl;
	std::cout<<"cpu active: "<<cpu->blockNumber[0]*cpu->blockNumber[1]*cpu->blockNumber[2]<<" | cuda active: "<<refParams->blockNumber[0]*refParams->blockNumber[1]*refParams->blockNumber[2]<<std::endl;
	for (unsigned long i = 0; i < refParams->definedActiveBlock; i++) {

		float cpuTargetPt[3] = { cpuTargetData[3 * i + 0], cpuTargetData[3 * i + 1], cpuTargetData[3 * i + 2] };
		float cpuResultPt[3] = { cpuResultData[3 * i + 0], cpuResultData[3 * i + 1], cpuResultData[3 * i + 2] };

		bool found = false;
		for (unsigned long j = 0; j < refParams->definedActiveBlock; j++) {
			float cudaTargetPt[3] = { cudaTargetData[3 * j + 0], cudaTargetData[3 * j + 1], cudaTargetData[3 * j + 2] };
			float cudaResultPt[3] = { cudaResultData[3 * j + 0], cudaResultData[3 * j + 1], cudaResultData[3 * j + 2] };

			targetSum[0] = cpuTargetPt[0] - cudaTargetPt[0];
			targetSum[1] = cpuTargetPt[1] - cudaTargetPt[1];
			targetSum[2] = cpuTargetPt[2] - cudaTargetPt[2];

			if (targetSum[0] == 0 && targetSum[1] == 0 && targetSum[2] == 0) {

				resultSum[0] = abs(cpuResultPt[0] - cudaResultPt[0]);
				resultSum[1] = abs(cpuResultPt[1] - cudaResultPt[1]);
				resultSum[2] = abs(cpuResultPt[2] - cudaResultPt[2]);
				found = true;
				if (resultSum[0] > 0.000001f || resultSum[1] > 0.000001f || resultSum[2] > 0.000001f)
					printf("i: %lu | j: %lu | (dif: %f-%f-%f) | (out: %f, %f, %f) | (ref: %f, %f, %f)\n", i, j, resultSum[0], resultSum[1], resultSum[2], cpuResultPt[0], cpuResultPt[1], cpuResultPt[2], cudaResultPt[0], cudaResultPt[1], cudaResultPt[2]);

			}
		}
		if (!found)
			printf("i: %lu has no match\n", i);
		/*double targetDiff = abs(refTargetPt[0] - outTargetPt[0]) + abs(refTargetPt[1] - outTargetPt[1]) + abs(refTargetPt[2] - outTargetPt[2]);
		 double resultDiff = abs(refResultPt[0] - outResultPt[0]) + abs(refResultPt[1] - outResultPt[1]) + abs(refResultPt[2] - outResultPt[2]);

		 maxTargetDiff = (targetDiff > maxTargetDiff) ? targetDiff : maxTargetDiff;
		 maxResultDiff = (resultDiff > maxResultDiff) ? resultDiff : maxResultDiff;*/
	}
}

void CudaBlockMatchingKernel::execute() {
//	block_matching_method(con->CurrentReference, con->CurrentWarped, con->blockMatchingParams, con->getCurrentReferenceMask());
	con->setCurrentWarped(con->CurrentWarped);
	resultImageArray_d = con->getWarpedImageArray_d();
	launchBlockMatching(target, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d, &targetMat_d);

	this->params = con->getBlockMatchingParams();
	compare(con->CurrentReference, con->CurrentWarped,con->getCurrentReferenceMask(), this->params);
	exit(0);
}
//===================================================================================================================================================================
CudaOptimiseKernel::CudaOptimiseKernel(Context* conIn, std::string name) :
			OptimiseKernel(name) {
		con = static_cast<CudaContext*>(conIn);
		transformationMatrix = con->transformationMatrix;
		blockMatchingParams = con->blockMatchingParams;

	}

void CudaOptimiseKernel::execute(bool affine) {

	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}

