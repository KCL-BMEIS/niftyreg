#include"Kernel.h"
#include"Kernels.h"
#include "CPUPlatform.h"
#include "CudaPlatform.h"
#include "CLPlatform.h"
#include "_reg_ReadWriteImage.h"
#include <math.h>
#include <algorithm>
#include <stdlib.h>

#define AFFINE  1
#define RIGID   0

float compareMats(mat44* mat1, mat44* mat2) {
	float maxDiff = -1;
	for( unsigned int i = 0; i < 4; i++ ) {
		for( unsigned int j = 0; j < 4; j++ ) {
			float tempDiff = fabsf(mat1->m[i][j] - mat2->m[i][j]);
			maxDiff = std::max<float>(maxDiff, tempDiff);

		}
	}
	return maxDiff;
}

void mockInitialMatrix(mat44* TransformationMatrix) {

	TransformationMatrix->m[0][0] = 1;
	TransformationMatrix->m[1][0] = 0;
	TransformationMatrix->m[2][0] = 0;
	TransformationMatrix->m[3][0] = 0.0000;

	TransformationMatrix->m[0][1] = 0;
	TransformationMatrix->m[1][1] = 1;
	TransformationMatrix->m[2][1] = 0;
	TransformationMatrix->m[3][1] = 0.0000;

	TransformationMatrix->m[0][2] = 0;
	TransformationMatrix->m[1][2] = 0;
	TransformationMatrix->m[2][2] = 1;
	TransformationMatrix->m[3][2] = 0.0000;

	TransformationMatrix->m[0][3] = 0;
	TransformationMatrix->m[1][3] = 0;
	TransformationMatrix->m[2][3] = 0;
	TransformationMatrix->m[3][3] = 1.0000;
}

void mockRigidOutput(mat44* rigidOutput) {
	rigidOutput->m[0][0] = 0.999959409;
	rigidOutput->m[1][0] = 0.00179867819;
	rigidOutput->m[2][0] = 0.00884950161;
	rigidOutput->m[3][0] = 0.0000;

	rigidOutput->m[0][1] = -0.00192992762;
	rigidOutput->m[1][1] = 0.999887764;
	rigidOutput->m[2][1] = 0.0148488125;
	rigidOutput->m[3][1] = 0.0000;

	rigidOutput->m[0][2] = -0.00882171094;
	rigidOutput->m[1][2] = -0.0148652513;
	rigidOutput->m[2][2] = 0.999850631;
	rigidOutput->m[3][2] = 0.0000;

	rigidOutput->m[0][3] = -0.396646500;
	rigidOutput->m[1][3] = -1.08658767;
	rigidOutput->m[2][3] = -0.711698532;
	rigidOutput->m[3][3] = 1.0000;
}

void mockAffineOutput(mat44* affineOutput) {
	affineOutput->m[0][0] = 0.990324914;
	affineOutput->m[1][0] = -0.0233036540;
	affineOutput->m[2][0] = -0.00630508503;
	affineOutput->m[3][0] = 0.0000;

	affineOutput->m[0][1] = -0.0134074651;
	affineOutput->m[1][1] = 0.995832980;
	affineOutput->m[2][1] = 0.0147247789;
	affineOutput->m[3][1] = 0.0000;

	affineOutput->m[0][2] = -0.0159819070;
	affineOutput->m[1][2] = -0.0231159087;
	affineOutput->m[2][2] = 1.00234115;
	affineOutput->m[3][2] = 0.0000;

	affineOutput->m[0][3] = -0.225879997;
	affineOutput->m[1][3] = -0.910197318;
	affineOutput->m[2][3] = -0.655013084;
	affineOutput->m[3][3] = 1.0000;
}

void initParams(Platform* platform, _reg_blockMatchingParam* blockMatchingParams) {
	Kernel blockMatchingKernel = platform->createKernel(BlockMatchingKernel::Name(), 16);
	nifti_image* CurrentReference = reg_io_ReadImageFile("mock_bm_reference.nii");
	nifti_image* CurrentWarped = reg_io_ReadImageFile("mock_bm_warped.nii");

	const unsigned int CurrentPercentageOfBlockToUse = 50;
	const unsigned int InlierLts = 50;
	const unsigned int activeVoxelNumber = CurrentReference->nx*CurrentReference->ny*CurrentReference->nz;


	int* CurrentReferenceMask = (int *)calloc(activeVoxelNumber, sizeof(int));

	//prep kernels
	blockMatchingKernel.getAs<BlockMatchingKernel>().initialize(CurrentReference, blockMatchingParams, CurrentPercentageOfBlockToUse, InlierLts, CurrentReferenceMask, false);
	blockMatchingKernel.getAs<BlockMatchingKernel>().execute(CurrentReference, CurrentWarped, blockMatchingParams, CurrentReferenceMask);
}


float test(Platform* platform, _reg_blockMatchingParam* blockMatchingParams, const unsigned int type, char* msg) {
	Kernel optimizeKernel = platform->createKernel(OptimiseKernel::Name(), 16);


	mat44* TransformationMatrix = new mat44;
	mockInitialMatrix(TransformationMatrix);

	mat44* output = new mat44;
	if( type == RIGID )
		mockRigidOutput(output);
	else if( type == AFFINE )
		mockAffineOutput(output);

	//test kernels
	optimizeKernel.getAs<OptimiseKernel>().execute(blockMatchingParams, TransformationMatrix, type == AFFINE);
	//measure performance (elapsed time)

	//compare results
	float diff = compareMats(TransformationMatrix, output);

	//output
	std::cout << "===================================" << msg << "===================================" << std::endl;
	std::cout << std::endl;
	std::cout << msg<<": " << diff << std::endl;
	std::cout << "===================================" << msg << " END ===============================" << std::endl;

	
	return diff;

}

int main(int argc, char **argv) {

	//init platform params
	Platform *cpuPlatform = new CPUPlatform();
	Platform *cudaPlatform = new CudaPlatform();
	Platform *clPlatform = new CLPlatform();

	//init ref params
	_reg_blockMatchingParam blockMatchingParams;
	initParams(cpuPlatform, &blockMatchingParams);

	//run tests for rigid and affine
	float maxDiff1 = test(cpuPlatform, &blockMatchingParams, RIGID, "CPU RIGID ");
	float maxDiff2 = test(cpuPlatform, &blockMatchingParams, AFFINE,"CPU AFFINE");


	return 0;

}