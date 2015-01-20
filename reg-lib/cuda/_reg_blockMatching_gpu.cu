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
/*
 #include <thrust/sort.h>
 #include <thrust/device_vector.h>
 */
/*
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
*/
#include "CudaKernelFuncs.h"

void checkStatus(cusolverStatus_t status, char* msg){
	if (status == CUSOLVER_STATUS_SUCCESS) printf("%s: PASS\n", msg);
	else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) printf("%s: the library was not initialized.\n", msg);
	else if (status == CUSOLVER_STATUS_INVALID_VALUE) printf("%s: invalid parameters were passed (m,n<0 or lda<max(1,m) or ldu<max(1,m) or ldvt<max(1,n) ).\n", msg);
	else if (status == CUSOLVER_STATUS_ARCH_MISMATCH) printf("%s: the device only supports compute capability 2.0 and above.\n", msg);
	else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) printf("%s: an internal operation failed.\n", msg);
	else if (status == CUSOLVER_STATUS_EXECUTION_FAILED) printf("%s: a kernel failed to launch on the GPU.\n", msg);
	else printf("%s: %d\n", msg, status);
}
void checkDevInfo(int *devInfo){
	int * hostDevInfo = (int*)malloc(sizeof(int));
	cudaMemcpy(hostDevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (hostDevInfo<0) printf("parameter: %d is wrong\n", hostDevInfo);
	if (hostDevInfo>0) printf("%d superdiagonals of an intermediate bidiagonal form B did not converge to zero.\n", hostDevInfo);
	else printf(" %d: operation successful\n", hostDevInfo);
	free(hostDevInfo);
}
void cusolverSVD(float* A_d, unsigned int m, unsigned int n, float* S_d, float* VT_d, float* U_d) {

	/* cuda */
	cudaError_t cudaStatus; /* cusolver */
	cusolverStatus_t status;
	cusolverDnHandle_t gH = NULL;
	int Lwork;

	//device ptrs
	float *Work;
	float *rwork;
	int *devInfo;

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

	status = cusolverDnCreate(&gH);
	checkStatus(status, "cusolverDnCreate");


	status = cusolverDnSgesvd_bufferSize(gH, m, n, &Lwork);
	checkStatus(status, "cusolverDnSgesvd_bufferSize");

	printf("LWork: %d:%d:%d | m: %d n: %d \n", Lwork, Lwork/4,  Lwork%4, m, n);

	cudaMalloc(&Work, Lwork * sizeof(float));
	cudaMalloc(&rwork, Lwork * sizeof(float));
	cudaMalloc(&devInfo, sizeof(int));

	status = cusolverDnSgesvd(gH, jobu, jobvt, m, n, A_d, m, S_d, U_d, m, VT_d, n, Work, Lwork, NULL, devInfo);
	cudaDeviceSynchronize();
	checkStatus(status, "cusolverDnSgesvd");
	checkDevInfo(devInfo);


	status = cusolverDnDestroy(gH);
	checkStatus(status, "cusolverDnDestroy");

	cudaFree(devInfo);
	cudaFree(rwork);
	cudaFree(Work);

}
/* *************************************************************** */
//is it square distance or just distance?
// Helper function: Get the square of the Euclidean distance
double get_square_distance3D1(float * first_point3D, float * second_point3D) {
	return sqrt((first_point3D[0] - second_point3D[0]) * (first_point3D[0] - second_point3D[0]) + (first_point3D[1] - second_point3D[1]) * (first_point3D[1] - second_point3D[1]) + (first_point3D[2] - second_point3D[2]) * (first_point3D[2] - second_point3D[2]));
}
/* *************************************************************** */
// Multiply matrices A and B together and store the result in r.
// We assume that the input pointers are valid and can store the result.
// A = ar * ac
// B = ac * bc
// r = ar * bc
// We can specify if we want to multiply A with the transpose of B
void mul_matrices1(float ** a, float ** b, int ar, int ac, int bc, float ** r, bool transposeB) {
	if (transposeB) {
		for (int i = 0; i < ar; ++i) {
			for (int j = 0; j < bc; ++j) {
				r[i][j] = 0.0f;
				for (int k = 0; k < ac; ++k) {
					r[i][j] += static_cast<float>(static_cast<double>(a[i][k]) * static_cast<double>(b[j][k]));
				}
			}
		}
	} else {
		for (int i = 0; i < ar; ++i) {
			for (int j = 0; j < bc; ++j) {
				r[i][j] = 0.0f;
				for (int k = 0; k < ac; ++k) {
					r[i][j] += static_cast<float>(static_cast<double>(a[i][k]) * static_cast<double>(b[k][j]));
				}
			}
		}
	}
}
/* *************************************************************** */

// Multiply a matrix with a vctor
void mul_matvec1(float ** a, int ar, int ac, float * b, float * r) {
	for (int i = 0; i < ar; ++i) {
		r[i] = 0;
		for (int k = 0; k < ac; ++k) {
			r[i] += static_cast<float>(static_cast<double>(a[i][k]) * static_cast<double>(b[k]));
		}
	}
}
/* *************************************************************** */
// Compute determinant of a 3x3 matrix
float compute_determinant3x3_1(float ** mat) {
	return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])) - (mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])) + (mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]));
}
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

void perturbate1(mat44 *affine, double ratio) {
//	std::cout << "perturbating: " << ratio << std::endl;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			affine->m[i][j] += ratio * affine->m[i][j];
		}
	}
}
void downloadMat44(mat44 *lastTransformation, float* transform_d) {
	float* tempMat = (float*) malloc(16 * sizeof(float));
	cudaMemcpy(tempMat, transform_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);
	cPtrToMat44(lastTransformation, tempMat);
	free(tempMat);
}

void affineLocalSearch3DCuda(mat44 *cpuMat, float* final_d, float *A_d, float* Sigma_d, float* U_d, float* VT_d, float* r_d, float * newResultPos_d, float* targetPos_d, float* resultPos_d, float* lengths_d, const unsigned int numBlocks, const unsigned long num_to_keep, const unsigned int m, const unsigned int n) {

	double lastDistance = std::numeric_limits<double>::max();

	float* lastTransformation_d;
	cudaMalloc(&lastTransformation_d, 16 * sizeof(float));
	//transform result points
	transformResultPointsKernel<<<numBlocks, 512>>>(final_d, resultPos_d,newResultPos_d, m/3); //test 0
	cudaMemcpy(resultPos_d, newResultPos_d, m * sizeof(float), cudaMemcpyDeviceToDevice); //test 0

	//get initial affine matrix
	getAffineMat3D(A_d, Sigma_d, VT_d, U_d, targetPos_d, resultPos_d, r_d, final_d, numBlocks, m, n);

	for (unsigned count = 0; count < MAX_ITERATIONS; ++count) {

		// Transform the points in the target
		transformResultPointsKernel<<<numBlocks, 512>>>(final_d, targetPos_d,newResultPos_d, m/3); //test 1

		double distance = sortAndReduce( lengths_d, targetPos_d, resultPos_d, newResultPos_d, numBlocks, m);

		// If the change is not substantial or we are getting worst, we return
		if ((distance >= lastDistance) || (lastDistance - distance) < TOLERANCE) break;

		lastDistance = distance;

		cudaMemcpy( lastTransformation_d,final_d, 16*sizeof(float), cudaMemcpyDeviceToDevice);
		getAffineMat3D(A_d, Sigma_d, VT_d, U_d, targetPos_d, resultPos_d, r_d, final_d, numBlocks, m, n);
	}

	//async cudamemcpy here
	cudaMemcpy(final_d, lastTransformation_d, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(lastTransformation_d);

	downloadMat44(cpuMat, final_d);

}
void affineLocalSearch3D1(_reg_blockMatchingParam *params, std::vector<_reg_sorted_point3D> top_points, mat44 * final, float * w, float ** v) {
	// The LS in the iterations is done on subsample of the input data

	unsigned long num_points = params->definedActiveBlock;
	const unsigned long num_to_keep = (unsigned long) (num_points * (params->percent_to_keep / 100.0f));
	const unsigned long num_equations = num_to_keep * 3;
	float * newResultPosition = new float[num_points * 3];
	std::multimap<double, _reg_sorted_point3D> queue;
	mat44 lastTransformation;
	memset(&lastTransformation, 0, sizeof(mat44));
	double lastLowest = 1000000000.0;

	double pert = 0.8;
	//optimization routine
	unsigned int count = 0;

	double distance = 0.0;
	double lastDistance = std::numeric_limits<double>::max();
	unsigned long i;

	// The LHS matrix
	float** a = new float *[num_equations];
	for (unsigned k = 0; k < num_equations; ++k) {
		a[k] = new float[12]; // full affine
	}

	// Allocate memory for pseudoinverse
	float** r = new float *[12];
	for (unsigned k = 0; k < 12; ++k) {
		r[k] = new float[num_equations];
	}

	// Allocate memory for RHS vector
	float* b = new float[num_equations];

	for (unsigned count = 0; count < MAX_ITERATIONS; ++count)
			{
		// Transform the points in the target
		for (unsigned j = 0; j < num_points * 3; j += 3)
				{
			reg_mat44_mul(final, &(params->targetPosition[j]), &newResultPosition[j]);
		}

		queue = std::multimap<double, _reg_sorted_point3D>();
		for (unsigned j = 0; j < num_points * 3; j += 3) {
			distance = get_square_distance3D1(&newResultPosition[j], &(params->resultPosition[j]));
			queue.insert(std::pair<double, _reg_sorted_point3D>(distance, _reg_sorted_point3D(&(params->targetPosition[j]),
					&(params->resultPosition[j]), distance)));
		}

		distance = 0.0;
		i = 0;
		top_points.clear();

		for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin(); it != queue.end(); ++it, ++i) {
			if (i >= num_to_keep)
				break;
			top_points.push_back((*it).second);
			distance += (*it).first;
		}

		// If the change is not substantial or we are getting worst, we return
		if ((distance >= lastDistance) || (lastDistance - distance) < TOLERANCE)
		{
			// restore the last transformation
			memcpy(final, &lastTransformation, sizeof(mat44));
			break;
		}
		lastDistance = distance;
		memcpy(&lastTransformation, final, sizeof(mat44));
		estimate_affine_transformation3D1(top_points, final, a, w, v, r, b);
	}

	memcpy(final, &lastTransformation, sizeof(mat44));
	delete[] newResultPosition;

	delete[] b;
	for (unsigned k = 0; k < 12; ++k) {
		delete[] r[k];
	}
	delete[] r;

	// free the memory
	for (unsigned int k = 0; k < num_equations; ++k) {
		delete[] a[k];
	}
	delete[] a;
}

void affineIterativeLocalSearch3D1(_reg_blockMatchingParam *params, std::vector<_reg_sorted_point3D> top_points, mat44 * final, float * w, float ** v) {
	// The LS in the iterations is done on subsample of the input data

	const unsigned long num_points = params->definedActiveBlock;
	const unsigned long num_to_keep = (unsigned long) (num_points * (params->percent_to_keep / 100.0f));
	const unsigned long num_equations = num_to_keep * 3;
	float * newResultPosition = new float[num_points * 3];
	mat44 lastTransformation;
	memset(&lastTransformation, 0, sizeof(mat44));

	double lastLowest = std::numeric_limits<double>::max();
	double lastDistance = std::numeric_limits<double>::max();
	const double pert = 0.1;
	unsigned int count = 0, iter = 0;

	// The LHS matrix
	float** a = new float *[num_equations];
	for (unsigned k = 0; k < num_equations; ++k) {
		a[k] = new float[12]; // full affine
	}

	// Allocate memory for pseudoinverse
	float** r = new float *[12];
	for (unsigned k = 0; k < 12; ++k) {
		r[k] = new float[num_equations];
	}

	// Allocate memory for RHS vector
	float* b = new float[num_equations];

	while (count < 10 && iter < MAX_ITERATIONS) {

		bool foundLower = true;

		// Transform the points in the target
		for (unsigned j = 0; j < num_points * 3; j += 3) {
			reg_mat44_mul(final, &(params->targetPosition[j]), &newResultPosition[j]);
		}

		std::multimap<double, _reg_sorted_point3D> queue = std::multimap<double, _reg_sorted_point3D>();
		for (unsigned j = 0; j < num_points * 3; j += 3) {
			const double distanceIn = get_square_distance3D1(&newResultPosition[j], &(params->resultPosition[j]));
			queue.insert(std::pair<double, _reg_sorted_point3D>(distanceIn, _reg_sorted_point3D(&(params->targetPosition[j]), &(params->resultPosition[j]), distanceIn)));
		}

		top_points.clear();

		double distance = 0.0;
		unsigned long i = 0;

		for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin(); it != queue.end(); ++it, ++i) {
			if (i >= num_to_keep)
				break;
			top_points.push_back((*it).second);
			distance += (*it).first;
		}

//		local search converged

		if (std::abs(distance - lastDistance) / distance < 0.0000001) {
			perturbate1(final, pert * (count % 10));

//			pert -= 0.1;
			count++;
			iter = 0;
		} else {

			//		std::cout << count << "/" << MAX_ITERATIONS << ": " << distance << " - " << lastDistance << " - " << lastLowest << std::endl;
			lastDistance = distance;
			if (distance <= lastLowest) {
				lastLowest = distance;
				memcpy(&lastTransformation, final, sizeof(mat44));
			}
			estimate_affine_transformation3D1(top_points, final, a, w, v, r, b);
			iter++;
		}

	}

	memcpy(final, &lastTransformation, sizeof(mat44));
	delete[] newResultPosition;

	delete[] b;
	for (unsigned k = 0; k < 12; ++k) {
		delete[] r[k];
	}
	delete[] r;

	// free the memory
	for (unsigned int k = 0; k < num_equations; ++k) {
		delete[] a[k];
	}
	delete[] a;
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation3D1(std::vector<_reg_sorted_point3D> &points, mat44 * transformation, float ** A, float * w, float ** v, float ** r, float * b) {
	// Create our A matrix
	// we need at least 4 points. Assuming we have that here.
	int num_equations = points.size() * 3;
	unsigned c = 0;
	for (unsigned k = 0; k < points.size(); ++k) {
		c = k * 3;
		A[c][0] = points[k].target[0];
		A[c][1] = points[k].target[1];
		A[c][2] = points[k].target[2];
		A[c][3] = A[c][4] = A[c][5] = A[c][6] = A[c][7] = A[c][8] = A[c][10] = A[c][11] = 0.0f;
		A[c][9] = 1.0f;

		A[c + 1][3] = points[k].target[0];
		A[c + 1][4] = points[k].target[1];
		A[c + 1][5] = points[k].target[2];
		A[c + 1][0] = A[c + 1][1] = A[c + 1][2] = A[c + 1][6] = A[c + 1][7] = A[c + 1][8] = A[c + 1][9] = A[c + 1][11] = 0.0f;
		A[c + 1][10] = 1.0f;

		A[c + 2][6] = points[k].target[0];
		A[c + 2][7] = points[k].target[1];
		A[c + 2][8] = points[k].target[2];
		A[c + 2][0] = A[c + 2][1] = A[c + 2][2] = A[c + 2][3] = A[c + 2][4] = A[c + 2][5] = A[c + 2][9] = A[c + 2][10] = 0.0f;
		A[c + 2][11] = 1.0f;
	}

	//	A:U | w:Sigma | v:V
	// Now we can compute our svd
	svd(A, num_equations, 12, w, v);

	// First we make sure that the really small singular values
	// are set to 0. and compute the inverse by taking the reciprocal
	// of the entries
	for (unsigned k = 0; k < 12; ++k)
		w[k] = (w[k] < 0.0001) ? 0.0f : (1.0 / w[k]);

	// Now we can compute the pseudoinverse which is given by
	// V*inv(W)*U'
	// First compute the V * inv(w) in place.
	// Simply scale each column by the corresponding singular value
	for (unsigned k = 0; k < 12; ++k) {
		for (unsigned j = 0; j < 12; ++j) {
			v[j][k] = static_cast<float>(static_cast<double>(v[j][k]) * static_cast<double>(w[k]));
		}
	}

	// Now multiply the matrices together
	// Pseudoinverse = v * e * A(transpose)
	mul_matrices1(v, A, 12, 12, num_equations, r, true);
	// Now r contains the pseudoinverse
	// Create vector b and then multiple rb to get the affine paramsA
	for (unsigned k = 0; k < points.size(); ++k) {
		c = k * 3;
		b[c] = points[k].result[0];
		b[c + 1] = points[k].result[1];
		b[c + 2] = points[k].result[2];
	}

	float * transform = new float[12];
	mul_matvec1(r, 12, num_equations, b, transform);

	transformation->m[0][0] = transform[0];
	transformation->m[0][1] = transform[1];
	transformation->m[0][2] = transform[2];
	transformation->m[0][3] = transform[9];

	transformation->m[1][0] = transform[3];
	transformation->m[1][1] = transform[4];
	transformation->m[1][2] = transform[5];
	transformation->m[1][3] = transform[10];

	transformation->m[2][0] = transform[6];
	transformation->m[2][1] = transform[7];
	transformation->m[2][2] = transform[8];
	transformation->m[2][3] = transform[11];

	transformation->m[3][0] = 0.0f;
	transformation->m[3][1] = 0.0f;
	transformation->m[3][2] = 0.0f;
	transformation->m[3][3] = 1.0f;

	delete[] transform;
}
//OPTIMIZER-----------------------------------------------
void estimate_rigid_transformation3D1(std::vector<_reg_sorted_point3D> &points, mat44 * transformation) {
	float centroid_target[3] = { 0.0f };
	float centroid_result[3] = { 0.0f };

	for (unsigned j = 0; j < points.size(); ++j) {
		centroid_target[0] += points[j].target[0];
		centroid_target[1] += points[j].target[1];
		centroid_target[2] += points[j].target[2];

		centroid_result[0] += points[j].result[0];
		centroid_result[1] += points[j].result[1];
		centroid_result[2] += points[j].result[2];
	}

	centroid_target[0] /= (float) (points.size());
	centroid_target[1] /= (float) (points.size());
	centroid_target[2] /= (float) (points.size());

	centroid_result[0] /= (float) (points.size());
	centroid_result[1] /= (float) (points.size());
	centroid_result[2] /= (float) (points.size());

	float ** u = new float*[3];
	float * w = new float[3];
	float ** v = new float*[3];
	float ** ut = new float*[3];
	float ** r = new float*[3];

	for (unsigned i = 0; i < 3; ++i) {
		u[i] = new float[3];
		v[i] = new float[3];
		ut[i] = new float[3];
		r[i] = new float[3];

		w[i] = 0.0f;

		for (unsigned j = 0; j < 3; ++j) {
			u[i][j] = v[i][j] = ut[i][j] = r[i][j] = 0.0f;
		}
	}

	// Demean the input points
	for (unsigned j = 0; j < points.size(); ++j) {
		points[j].target[0] -= centroid_target[0];
		points[j].target[1] -= centroid_target[1];
		points[j].target[2] -= centroid_target[2];

		points[j].result[0] -= centroid_result[0];
		points[j].result[1] -= centroid_result[1];
		points[j].result[2] -= centroid_result[2];

		u[0][0] += points[j].target[0] * points[j].result[0];
		u[0][1] += points[j].target[0] * points[j].result[1];
		u[0][2] += points[j].target[0] * points[j].result[2];

		u[1][0] += points[j].target[1] * points[j].result[0];
		u[1][1] += points[j].target[1] * points[j].result[1];
		u[1][2] += points[j].target[1] * points[j].result[2];

		u[2][0] += points[j].target[2] * points[j].result[0];
		u[2][1] += points[j].target[2] * points[j].result[1];
		u[2][2] += points[j].target[2] * points[j].result[2];

	}

	svd(u, 3, 3, w, v);

	// Calculate transpose
	ut[0][0] = u[0][0];
	ut[1][0] = u[0][1];
	ut[2][0] = u[0][2];

	ut[0][1] = u[1][0];
	ut[1][1] = u[1][1];
	ut[2][1] = u[1][2];

	ut[0][2] = u[2][0];
	ut[1][2] = u[2][1];
	ut[2][2] = u[2][2];

	// Calculate the rotation matrix
	mul_matrices1(v, ut, 3, 3, 3, r, false);

	float det = compute_determinant3x3_1(r);

	// Take care of possible reflection
	if (det < 0.0f) {
		v[0][2] = -v[0][2];
		v[1][2] = -v[1][2];
		v[2][2] = -v[2][2];

	}
	// Calculate the rotation matrix
	mul_matrices1(v, ut, 3, 3, 3, r, false);

	// Calculate the translation
	float t[3];
	t[0] = centroid_result[0] - (r[0][0] * centroid_target[0] + r[0][1] * centroid_target[1] + r[0][2] * centroid_target[2]);

	t[1] = centroid_result[1] - (r[1][0] * centroid_target[0] + r[1][1] * centroid_target[1] + r[1][2] * centroid_target[2]);

	t[2] = centroid_result[2] - (r[2][0] * centroid_target[0] + r[2][1] * centroid_target[1] + r[2][2] * centroid_target[2]);

	transformation->m[0][0] = r[0][0];
	transformation->m[0][1] = r[0][1];
	transformation->m[0][2] = r[0][2];
	transformation->m[0][3] = t[0];

	transformation->m[1][0] = r[1][0];
	transformation->m[1][1] = r[1][1];
	transformation->m[1][2] = r[1][2];
	transformation->m[1][3] = t[1];

	transformation->m[2][0] = r[2][0];
	transformation->m[2][1] = r[2][1];
	transformation->m[2][2] = r[2][2];
	transformation->m[2][3] = t[2];

	transformation->m[3][0] = 0.0f;
	transformation->m[3][1] = 0.0f;
	transformation->m[3][2] = 0.0f;
	transformation->m[3][3] = 1.0f;

	// Do the deletion here
	for (int i = 0; i < 3; ++i) {
		delete[] u[i];
		delete[] v[i];
		delete[] ut[i];
		delete[] r[i];
	}
	delete[] u;
	delete[] v;
	delete[] ut;
	delete[] r;
	delete[] w;
}
// estimate an affine transformation using least square
void getAffineMat3D(float* A_d, float* Sigma_d, float* VT_d, float* U_d, float* target_d, float* result_d, float* r_d, float *transformation, const unsigned int numBlocks, unsigned int m, unsigned int n) {

	//populate A
	populateMatrixA<<<numBlocks, 512>>>(A_d,target_d, m/3);//test 1
	printf("pre\n");
	//calculate SVD on the GPU
	cusolverSVD(A_d, m, n, Sigma_d, VT_d, U_d);
	printf("done\n");
	// First we make sure that the really small singular values
	// are set to 0. and compute the inverse by taking the reciprocal of the entries
	trimAndInvertSingularValuesKernel<<<1, n>>>(Sigma_d);

	cublasStatus_t status;
	cublasHandle_t handle;
	const float alpha = 1.f;
	const float beta = 0.f;

	/* Initialize CUBLAS */

	printf("CUBLAS\n");
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	fprintf(stderr, "!!!! CUBLAS initialization error\n");

	// Now we can compute the pseudoinverse which is given by V*inv(W)*U'

	// First compute the V * inv(w) in place.
	status = cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, VT_d, n, Sigma_d, 1, &beta, VT_d, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		fprintf(stderr, "!!!! CUBLAS cublasSgemv error\n");
	// Now multiply the matrices together
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, VT_d, n, U_d, n, &beta, r_d, n);
	if (status != CUBLAS_STATUS_SUCCESS)
		fprintf(stderr, "!!!! CUBLAS cublasSgemm 1 error\n");
	//r*b -> trans
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, r_d, n, result_d, n, &beta, transformation, n);
	if (status != CUBLAS_STATUS_SUCCESS)
		fprintf(stderr, "!!!! CUBLAS cublasSgemm 2 error\n");
	/* Shutdown */
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		fprintf(stderr, "!!!! CUBLAS cublasDestroy error\n");
}

void optimize_affine3D_cuda(mat44* cpuMat, float* final_d, float* A_d, float* U_d, float* Sigma_d, float* VT_d, float* r_d, float* lengths_d, float* target_d, float* result_d, float* newResult_d, unsigned int m, unsigned int n, const unsigned int numToKeep, bool ilsIn) {

	const unsigned int numEquations = m / 3;
	const unsigned int numBlocks = (numEquations % 512) ? (numEquations / 512) + 1 : numEquations / 512;

	// run the local search optimization routine
	affineLocalSearch3DCuda(cpuMat, final_d, newResult_d, A_d, Sigma_d, U_d, VT_d, r_d, target_d, result_d, lengths_d, numBlocks, numToKeep, m, n);

}
void optimize_affine3D1(_reg_blockMatchingParam *params, mat44 * final, bool ilsIn) {
// Set the current transformation to identity
	reg_mat44_eye(final);

	const unsigned num_points = params->definedActiveBlock;
	unsigned long num_equations = num_points * 3;
	std::multimap<double, _reg_sorted_point3D> queue;
	std::vector<_reg_sorted_point3D> top_points;
	double distance = 0.0;
	double lastDistance = std::numeric_limits<double>::max();
	unsigned long i;

// massive left hand side matrix
	float ** a = new float *[num_equations];
	for (unsigned k = 0; k < num_equations; ++k)
		a[k] = new float[12]; // full affine

// The array of singular values returned by svd
	float *w = new float[12];

// v will be n x n
	float **v = new float *[12];
	for (unsigned k = 0; k < 12; ++k)
		v[k] = new float[12];

// Allocate memory for pseudoinverse
	float **r = new float *[12];
	for (unsigned k = 0; k < 12; ++k)
		r[k] = new float[num_equations];

// Allocate memory for RHS vector
	float *b = new float[num_equations];

// The initial vector with all the input points
	for (unsigned j = 0; j < num_points * 3; j += 3)
		top_points.push_back(_reg_sorted_point3D(&(params->targetPosition[j]), &(params->resultPosition[j]), 0.0f));

// estimate the optimal transformation while considering all the points
	estimate_affine_transformation3D1(top_points, final, a, w, v, r, b);

// Delete a, b and r. w and v will not change size in subsequent svd operations.
	for (unsigned int k = 0; k < num_equations; ++k)
		delete[] a[k];

	delete[] a;
	delete[] b;

	for (unsigned k = 0; k < 12; ++k)
		delete[] r[k];

	delete[] r;

	if (ilsIn)
		affineIterativeLocalSearch3D1(params, top_points, final, w, v);
	else
		affineLocalSearch3D1(params, top_points, final, w, v);

	delete[] w;

	for (int k = 0; k < 12; ++k)
		delete[] v[k];
	delete[] v;
}
void optimize_rigid3D1(_reg_blockMatchingParam *params, mat44 *final, bool ilsIn) {
	const unsigned num_points = params->definedActiveBlock;
// Keep a sorted list of the distance measure
	std::multimap<double, _reg_sorted_point3D> queue;
	std::vector<_reg_sorted_point3D> top_points;
	double distance = 0.0;
	double lastDistance = std::numeric_limits<double>::max();
	unsigned long i;

// Set the current transformation to identity
	reg_mat44_eye(final);

	for (unsigned j = 0; j < num_points * 3; j += 3) {
		top_points.push_back(_reg_sorted_point3D(&(params->targetPosition[j]), &(params->resultPosition[j]), 0.0f));
	}

	estimate_rigid_transformation3D1(top_points, final);
	unsigned long num_to_keep = (unsigned long) (num_points * (params->percent_to_keep / 100.0f));
	float * newResultPosition = new float[num_points * 3];

	mat44 lastTransformation;
	memset(&lastTransformation, 0, sizeof(mat44));

	for (unsigned count = 0; count < MAX_ITERATIONS; ++count) {
		// Transform the points in the target
		for (unsigned j = 0; j < num_points * 3; j += 3) {
			reg_mat44_mul(final, &(params->targetPosition[j]), &newResultPosition[j]);
		}
		queue = std::multimap<double, _reg_sorted_point3D>();
		for (unsigned j = 0; j < num_points * 3; j += 3) {
			distance = get_square_distance3D1(&newResultPosition[j], &(params->resultPosition[j]));
			queue.insert(std::pair<double, _reg_sorted_point3D>(distance, _reg_sorted_point3D(&(params->targetPosition[j]), &(params->resultPosition[j]), distance)));
		}

		distance = 0.0;
		i = 0;
		top_points.clear();
		for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin(); it != queue.end(); ++it, ++i) {
			if (i >= num_to_keep)
				break;
			top_points.push_back((*it).second);
			distance += (*it).first;
		}

		// If the change is not substantial, we return
		if ((distance > lastDistance) || (lastDistance - distance) < TOLERANCE) {
			memcpy(final, &lastTransformation, sizeof(mat44));
			break;
		}
		lastDistance = distance;
		memcpy(&lastTransformation, final, sizeof(mat44));
		estimate_rigid_transformation3D1(top_points, final);
	}
	delete[] newResultPosition;
}

#endif
