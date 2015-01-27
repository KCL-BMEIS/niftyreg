/*
 *  _reg_blockMatching_gpu.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright 2009 UCL - CMIC. All rights reserved.
 *
 */

#ifndef _REG_BLOCKMATCHING_GPU_CU
#define _REG_BLOCKMATCHING_GPU_CU

#include "_reg_blockMatching_gpu.h"
#include "_reg_blockMatching_kernels.cu"

//#include "_reg_blocksize_gpu.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
#include "cublas_v2.h"
#include "cusolverDn.h"

#include <vector>
#include "_reg_maths.h"

#include "CudaKernelFuncs.h"

#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"

/* *************************************************************** */

void block_matching_method_gpu(nifti_image *targetImage, _reg_blockMatchingParam *params, float **targetImageArray_d, float **resultImageArray_d, float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d, float** targetMat_d) {

	// Copy some required parameters over to the device
	uint3 imageSize = make_uint3(targetImage->nx, targetImage->ny, targetImage->nz); // Image size

	// Texture binding
	const unsigned int numBlocks = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, targetImageArray_texture, *targetImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, resultImageArray_texture, *resultImageArray_d, targetImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, activeBlock_texture, *activeBlock_d, numBlocks * sizeof(int)));

	unsigned int* definedBlock_d;
	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	NR_CUDA_SAFE_CALL(cudaMalloc((void** )(&definedBlock_d), sizeof(unsigned int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(definedBlock_d, definedBlock_h, sizeof(unsigned int), cudaMemcpyHostToDevice));

	dim3 BlockDims1D(64, 1, 1);
	dim3 BlocksGrid3D(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	const int blockRange = params->voxelCaptureRange % 4 ? params->voxelCaptureRange / 4 + 1 : params->voxelCaptureRange / 4;
	const unsigned int sMem = (blockRange * 2 + 1) * (blockRange * 2 + 1) * (blockRange * 2 + 1) * 64 * sizeof(float);
	blockMatchingKernel<< <BlocksGrid3D, BlockDims1D, sMem >> >(*resultPosition_d, *targetPosition_d, *mask_d, *targetMat_d, definedBlock_d, imageSize, blockRange, params->stepSize);

#ifndef NDEBUG
	NR_CUDA_CHECK_KERNEL(BlocksGrid3D, BlockDims1D)
#endif
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

	NR_CUDA_SAFE_CALL(cudaMemcpy((void * )definedBlock_h, (void * )definedBlock_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	params->definedActiveBlock = *definedBlock_h;
//	printf("kernel definedActiveBlock: %d\n", params->definedActiveBlock);
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(activeBlock_texture));

	free(definedBlock_h);
	cudaFree(definedBlock_d);

}

//------------Optimizer------------------------------------

void checkCublasStatus(cublasStatus_t status) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS  error\n");
		exit(0);
	}
}
void checkCUSOLVERStatus(cusolverStatus_t status, char* msg) {
	if (status != CUSOLVER_STATUS_SUCCESS) {
		if (status == CUSOLVER_STATUS_SUCCESS)
			printf("%s: PASS\n", msg);
		else if (status == CUSOLVER_STATUS_NOT_INITIALIZED)
			printf("%s: the library was not initialized.\n", msg);
		else if (status == CUSOLVER_STATUS_INVALID_VALUE)
			printf("%s: invalid parameters were passed (m,n<0 or lda<max(1,m) or ldu<max(1,m) or ldvt<max(1,n) ).\n", msg);
		else if (status == CUSOLVER_STATUS_ARCH_MISMATCH)
			printf("%s: the device only supports compute capability 2.0 and above.\n", msg);
		else if (status == CUSOLVER_STATUS_INTERNAL_ERROR)
			printf("%s: an internal operation failed.\n", msg);
		else if (status == CUSOLVER_STATUS_EXECUTION_FAILED)
			printf("%s: a kernel failed to launch on the GPU.\n", msg);
		else
			printf("%s: %d\n", msg, status);
		exit(0);
	}
}
void checkDevInfo(int *devInfo) {
	int * hostDevInfo = (int*) malloc(sizeof(int));
	cudaMemcpy(hostDevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (hostDevInfo < 0)
		printf("parameter: %d is wrong\n", hostDevInfo);
	if (hostDevInfo > 0)
		printf("%d superdiagonals of an intermediate bidiagonal form B did not converge to zero.\n", hostDevInfo);
	else
		printf(" %d: operation successful\n", hostDevInfo);
	free(hostDevInfo);
}
void downloadMat44(mat44 *lastTransformation, float* transform_d) {
	float* tempMat = (float*) malloc(16 * sizeof(float));
	cudaMemcpy(tempMat, transform_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);
	cPtrToMat44(lastTransformation, tempMat);
	free(tempMat);
}
void uploadMat44(mat44 lastTransformation, float* transform_d) {
	float* tempMat = (float*) malloc(16 * sizeof(float));
	mat44ToCptr(lastTransformation, tempMat);
	cudaMemcpy(transform_d, tempMat, 16 * sizeof(float), cudaMemcpyHostToDevice);
	free(tempMat);
}
/*
 * the function computes the SVD of a matrix A
 * A = V* x S x U, where V* is a (conjugate) transpose of V
 * */
void cusolverSVD(float* A_d, unsigned int m, unsigned int n, float* S_d, float* VT_d, float* U_d) {

	const int lda = m;
	const int ldu = m;
	const int ldvt = n;

	/*
	 * 'A': all m columns of U are returned in array
	 * 'S': the first min(m,n) columns of U (the left singular vectors) are returned in the array
	 * 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on the array
	 * 'N': no columns of U (no left singular vectors) are computed
	 */
	const char jobu = 'A';

	/*
	 * 'A': all N rows of V**T are returned in the array
	 * 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the array
	 * 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array
	 * 'N': no rows of V**T (no right singular vectors) are computed
	 */
	const char jobvt = 'A';

	cusolverDnHandle_t gH = NULL;
	int Lwork;
	//device ptrs
	float *Work;
	float *rwork;
	int *devInfo;
	nvtxNameOsThread(1, "MAIN");
	nvtxRangePush(__FUNCTION__);

	//init cusolver compute SVD and shut down
	nvtxMark("Init...");
	checkCUSOLVERStatus(cusolverDnCreate(&gH), "cusolverDnCreate");
	nvtxMark("Lwork...");
	checkCUSOLVERStatus(cusolverDnSgesvd_bufferSize(gH, m, n, &Lwork), "cusolverDnSgesvd_bufferSize");

	cudaMalloc(&Work, Lwork * sizeof(float));
	cudaMalloc(&rwork, Lwork * sizeof(float));
	cudaMalloc(&devInfo, sizeof(int));
	nvtxMark("SVD...");
	checkCUSOLVERStatus(cusolverDnSgesvd(gH, jobu, jobvt, m, n, A_d, lda, S_d, U_d, ldu, VT_d, ldvt, Work, Lwork, NULL, devInfo), "cusolverDnSgesvd");
	nvtxMark("Destroy!!!");
	checkCUSOLVERStatus(cusolverDnDestroy(gH), "cusolverDnDestroy");
	nvtxRangePop();
	//free vars
	cudaFree(devInfo);
	cudaFree(rwork);
	cudaFree(Work);

}
/*
 * the function computes the Pseudoinverse from the products of the SVD factorisation of A
 * R = V x inv(S) x U*
 * */
void cublasPseudoInverse(float* transformation, float *R_d, float* result_d, float *VT_d, float* Sigma_d, float *U_d, const unsigned int m, const unsigned int n) {
	// First we make sure that the really small singular values
	// are set to 0. and compute the inverse by taking the reciprocal of the entries
	nvtxNameOsThread(1, "MAIN");
	nvtxRangePush(__FUNCTION__);
	trimAndInvertSingularValuesKernel<<<1, n>>>(Sigma_d);	//test 3

	cublasHandle_t handle;

	const float alpha = 1.f;
	const float beta = 0.f;

	const int ldvt = n;	//VT's lead dimension
	const int ldu = m;	//U's lead dimension
	const int ldr = n;	//Pseudoinverse's r lead dimension

	const int rowsVTandR = n;	//VT and r's num rows
	const int colsUandR = m;	//U and r's num cols
	const int colsVtRowsU = n;	//VT's cols and U's rows

	// V x inv(S) in place | We scale eaach row with the corresponding singular value as V is transpose
	scaleV<<<n,n>>>(VT_d, n, n, Sigma_d);

	//Initialize CUBLAS perform ops and shut down
	checkCublasStatus(cublasCreate(&handle));

	//now R = V x inv(S) x U*
	checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, rowsVTandR, colsUandR, colsVtRowsU, &alpha, VT_d, ldvt, U_d, ldu, &beta, R_d, ldr));

	//finally M=Rxb, where M is our affine matrix and b a vector containg the result points
	checkCublasStatus(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, R_d, ldr, result_d, 1, &beta, transformation, 1));
	checkCublasStatus(cublasDestroy(handle));
	permuteAffineMatrix<<<1,16>>>(transformation);
	cudaThreadSynchronize();
	nvtxRangePop();

}

//OPTIMIZER-----------------------------------------------

// estimate an affine transformation using least square
void getAffineMat3D(float* AR_d, float* Sigma_d, float* VT_d, float* U_d, float* target_d, float* result_d, float *transformation, const unsigned int numBlocks, unsigned int m, unsigned int n) {

//populate A
	populateMatrixA<<<numBlocks, 512>>>(AR_d,target_d, m/3); //test 2

//calculate SVD on the GPU

	nvtxNameOsThread(1,"MAIN");
	nvtxRangePush(__FUNCTION__);
	cusolverSVD(AR_d, m, n, Sigma_d, VT_d, U_d);

	cublasPseudoInverse(transformation, AR_d,result_d, VT_d,Sigma_d, U_d, m, n);
	nvtxRangePop();

}

void optimize_affine3D_cuda(mat44* cpuMat, float* final_d, float* AR_d, float* U_d, float* Sigma_d, float* VT_d, float* lengths_d, float* target_d, float* result_d, float* newResult_d, unsigned int m, unsigned int n, const unsigned int numToKeep, bool ilsIn) {

	//m | blockMatchingParams->definedActiveBlock * 3
	//n | 12
	const unsigned int numEquations = m / 3;
	const unsigned int numBlocks = (numEquations % 512) ? (numEquations / 512) + 1 : numEquations / 512;

	uploadMat44(*cpuMat, final_d);
	transformResultPointsKernel<<<numBlocks, 512>>>(final_d, result_d,newResult_d, m/3); //test 1
	cudaMemcpy(result_d, newResult_d, m * sizeof(float), cudaMemcpyDeviceToDevice);

	// run the local search optimization routine
	affineLocalSearch3DCuda(cpuMat, final_d, AR_d, Sigma_d, U_d, VT_d, newResult_d, target_d, result_d, lengths_d, numBlocks, numToKeep, m, n);

	downloadMat44(cpuMat, final_d);
}
void affineLocalSearch3DCuda(mat44 *cpuMat, float* final_d, float *AR_d, float* Sigma_d, float* U_d, float* VT_d, float * newResultPos_d, float* targetPos_d, float* resultPos_d, float* lengths_d, const unsigned int numBlocks, const unsigned int num_to_keep, const unsigned int m, const unsigned int n) {

	double lastDistance = std::numeric_limits<double>::max();

	float* lastTransformation_d;
	cudaMalloc(&lastTransformation_d, 16 * sizeof(float));

	//get initial affine matrix
	getAffineMat3D(AR_d, Sigma_d, VT_d, U_d, targetPos_d, resultPos_d, final_d, numBlocks, m, n);

	for (unsigned int count = 0; count < MAX_ITERATIONS; ++count) {

		// Transform the points in the target
		transformResultPointsKernel<<<numBlocks, 512>>>(final_d, targetPos_d,newResultPos_d, m/3); //test 1
		double distance = sortAndReduce( lengths_d, targetPos_d, resultPos_d, newResultPos_d, numBlocks,num_to_keep, m);

		// If the change is not substantial or we are getting worst, we return
		if ((distance >= lastDistance) || (lastDistance - distance) < TOLERANCE) break;

		lastDistance = distance;

		cudaMemcpy( lastTransformation_d,final_d, 16*sizeof(float), cudaMemcpyDeviceToDevice);
		getAffineMat3D(AR_d, Sigma_d, VT_d, U_d, targetPos_d, resultPos_d, final_d, numBlocks, m, n);
	}

	//async cudamemcpy here
	cudaMemcpy(final_d, lastTransformation_d, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(lastTransformation_d);
}

#endif
