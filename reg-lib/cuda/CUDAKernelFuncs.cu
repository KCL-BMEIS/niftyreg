#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include"_reg_resampling.h"
#include"_reg_maths.h"
#include "CUDAKernelFuncs.h"
#include "_reg_common_cuda.h"
#include"_reg_tools.h"
#include"_reg_ReadWriteImage.h"
#include <thrust/sort.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>

#define SINC_KERNEL_RADIUS 3
#define SINC_KERNEL_SIZE SINC_KERNEL_RADIUS*2

/* *************************************************************** */
unsigned int min1(unsigned int a, unsigned int b)
{
	return (a < b) ? a : b;
}
/* *************************************************************** */
__device__ __constant__ float cIdentity[16];
__device__ __inline__ void reg_mat44_expm_cuda(float* mat)
{
	//todo
}
__device__ __inline__
void reg_mat44_logm_cuda(float* mat)
{
	//todo
}
/* *************************************************************** */
template<class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(DTYPE const* mat, DTYPE const* in, DTYPE *out)
{
    out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
    out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
    out[2] = (DTYPE)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
	return;
}
/* *************************************************************** */
template<class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
    out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
    out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
    out[2] = (DTYPE)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
	return;
}
/* *************************************************************** */
__device__ __inline__ int cuda_reg_floor(float a)
{
	return (int) (floor(a));
}
/* *************************************************************** */
template<class FieldTYPE>
__device__ __inline__ void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis)
{
	if (ratio < 0.0f)
		ratio = 0.0f; //reg_rounding error
	FieldTYPE FF = ratio * ratio;
	basis[0] = (FieldTYPE) ((ratio * ((2.0f - ratio) * ratio - 1.0f)) / 2.0f);
	basis[1] = (FieldTYPE) ((FF * (3.0f * ratio - 5.0) + 2.0f) / 2.0f);
	basis[2] = (FieldTYPE) ((ratio * ((4.0f - 3.0f * ratio) * ratio + 1.0f)) / 2.0f);
	basis[3] = (FieldTYPE) ((ratio - 1.0f) * FF / 2.0f);
}
/* *************************************************************** */
__device__ __inline__
void reg_mat44_eye(float *mat) {
	mat[0 * 4 + 0] = 1.f;
	mat[0 * 4 + 1] = mat[0 * 4 + 2] = mat[0 * 4 + 3] = 0.f;
	mat[1 * 4 + 1] = 1.f;
	mat[1 * 4 + 0] = mat[1 * 4 + 2] = mat[1 * 4 + 3] = 0.f;
	mat[2 * 4 + 2] = 1.f;
	mat[2 * 4 + 0] = mat[2 * 4 + 1] = mat[2 * 4 + 3] = 0.f;
	mat[3 * 4 + 3] = 1.f;
	mat[3 * 4 + 0] = mat[3 * 4 + 1] = mat[3 * 4 + 2] = 0.f;
}
/* *************************************************************** */
__device__ __inline__ void getPosition(float* position, float* matrix, float* voxel, const unsigned int idx)
{
	position[idx] = matrix[idx * 4 + 0] * voxel[0] +
			matrix[idx * 4 + 1] * voxel[1] +
			matrix[idx * 4 + 2] * voxel[2] +
			matrix[idx * 4 + 3];
}
/* *************************************************************** */
__device__ __inline__ double getPosition(float* matrix, double* voxel, const unsigned int idx)
{
	unsigned long index = idx*4;
	return (double) matrix[index++] * voxel[0] +
			 (double) matrix[index++] * voxel[1] +
			 (double) matrix[index++] * voxel[2] +
			 (double) matrix[index];
}
/* *************************************************************** */
__inline__ __device__ void interpWindowedSincKernel(double relative, double *basis)
{
	if (relative < 0.0)
		relative = 0.0; //reg_rounding error
	int j = 0;
	double sum = 0.;
	for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
		double x = relative - (double) (i);
		if (x == 0.0)
			basis[j] = 1.0;
		else if (abs(x) >= (double) (SINC_KERNEL_RADIUS))
			basis[j] = 0;
		else {
			double pi_x = M_PI * x;
			basis[j] = (SINC_KERNEL_RADIUS) * sin(pi_x) * sin(pi_x / SINC_KERNEL_RADIUS) / (pi_x * pi_x);
		}
		sum += basis[j];
		j++;
	}
	for (int i = 0; i < SINC_KERNEL_SIZE; ++i)
		basis[i] /= sum;
}
/* *************************************************************** */
__inline__ __device__ void interpCubicSplineKernel(double relative, double *basis)
{
	if (relative < 0.0)
		relative = 0.0; //reg_rounding error
	double FF = relative * relative;
	basis[0] = (relative * ((2.0 - relative) * relative - 1.0)) / 2.0;
	basis[1] = (FF * (3.0 * relative - 5.0) + 2.0) / 2.0;
	basis[2] = (relative * ((4.0 - 3.0 * relative) * relative + 1.0)) / 2.0;
	basis[3] = (relative - 1.0) * FF / 2.0;
}
/* *************************************************************** */
__inline__ __device__ void interpLinearKernel(double relative, double *basis)
{
	if (relative < 0.0)
		relative = 0.0; //reg_rounding error
	basis[1] = relative;
	basis[0] = 1.0 - relative;
}
/* *************************************************************** */
__inline__ __device__ void interpNearestNeighKernel(double relative, double *basis)
{
	if (relative < 0.0)
		relative = 0.0; //reg_rounding error
	basis[0] = basis[1] = 0.0;
	if (relative > 0.5)
		basis[1] = 1;
	else
		basis[0] = 1;
}
/* *************************************************************** */
__inline__ __device__ double interpLoop(float* floatingIntensity,
													 double* xBasis,
													 double* yBasis,
													 double* zBasis,
													 int *previous,
													 uint3 fi_xyz,
													 double paddingValue,
													 unsigned int kernel_size)
{
	double intensity = static_cast<double>(paddingValue);
	for (int c = 0; c < kernel_size; c++) {
		int Z = previous[2] + c;
		bool zInBounds = -1 < Z && Z < fi_xyz.z;
		double yTempNewValue = 0.0;
		for (int b = 0; b < kernel_size; b++) {
			int Y = previous[1] + b;
			bool yInBounds = -1 < Y && Y < fi_xyz.y;
			double xTempNewValue = 0.0;
			for (int a = 0; a < kernel_size; a++) {
				int X = previous[0] + a;
				bool xInBounds = -1 < X && X < fi_xyz.x;
				const unsigned int idx = Z * fi_xyz.x * fi_xyz.y + Y * fi_xyz.x + X;
				xTempNewValue += (xInBounds && yInBounds && zInBounds) ? floatingIntensity[idx] * xBasis[a] : paddingValue * xBasis[a];
			}
			yTempNewValue += xTempNewValue * yBasis[b];
		}
		intensity += yTempNewValue * zBasis[c];
	}
	return intensity;
}
/* *************************************************************** */
__global__ void ResampleImage3D(float* floatingImage,
										  float* deformationField,
										  float* warpedImage,
										  int *mask,
										  float* sourceIJKMatrix,
										  ulong2 voxelNumber,
										  uint3 fi_xyz,
										  uint2 wi_tu,
										  float paddingValue,
										  int kernelType)
{
	float *sourceIntensityPtr = (floatingImage);
	float *resultIntensityPtr = (warpedImage);
	float *deformationFieldPtrX = (deformationField);
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	int *maskPtr = &mask[0];

	long index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < voxelNumber.x) {

		for (unsigned int t = 0; t < wi_tu.x * wi_tu.y; t++) {

			float *resultIntensity = &resultIntensityPtr[t * voxelNumber.x];
			float *floatingIntensity = &sourceIntensityPtr[t * voxelNumber.y];
			double intensity = paddingValue;

			if (maskPtr[index] > -1) {

				int previous[3];
                float world[3], position[3];
                double relative[3];

				world[0] = static_cast<float>(deformationFieldPtrX[index]);
				world[1] = static_cast<float>(deformationFieldPtrY[index]);
				world[2] = static_cast<float>(deformationFieldPtrZ[index]);

				// real -> voxel; floating space
				reg_mat44_mul_cuda<float>(sourceIJKMatrix, world, position);

				previous[0] = static_cast<int>(cuda_reg_floor(position[0]));
				previous[1] = static_cast<int>(cuda_reg_floor(position[1]));
				previous[2] = static_cast<int>(cuda_reg_floor(position[2]));

                relative[0] = static_cast<double>(position[0]) - static_cast<double>(previous[0]);
                relative[1] = static_cast<double>(position[1]) - static_cast<double>(previous[1]);
                relative[2] = static_cast<double>(position[2]) - static_cast<double>(previous[2]);

				if (kernelType == 0) {

					double xBasisIn[2], yBasisIn[2], zBasisIn[2];
					interpNearestNeighKernel(relative[0], xBasisIn);
					interpNearestNeighKernel(relative[1], yBasisIn);
					interpNearestNeighKernel(relative[2], zBasisIn);
					intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
				} else if (kernelType == 1) {

					double xBasisIn[2], yBasisIn[2], zBasisIn[2];
					interpLinearKernel(relative[0], xBasisIn);
					interpLinearKernel(relative[1], yBasisIn);
					interpLinearKernel(relative[2], zBasisIn);
					intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
				} else if (kernelType == 4) {

					double xBasisIn[6], yBasisIn[6], zBasisIn[6];

					previous[0] -= SINC_KERNEL_RADIUS;
					previous[1] -= SINC_KERNEL_RADIUS;
					previous[2] -= SINC_KERNEL_RADIUS;

					interpWindowedSincKernel(relative[0], xBasisIn);
					interpWindowedSincKernel(relative[1], yBasisIn);
					interpWindowedSincKernel(relative[2], zBasisIn);
					intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
				} else {

					double xBasisIn[4], yBasisIn[4], zBasisIn[4];

					previous[0]--;
					previous[1]--;
					previous[2]--;

					interpCubicSplineKernel(relative[0], xBasisIn);
					interpCubicSplineKernel(relative[1], yBasisIn);
					interpCubicSplineKernel(relative[2], zBasisIn);
					intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
				}
			}
			resultIntensity[index] = intensity;
		}
	}
	index += blockDim.x * gridDim.x;
}
/* *************************************************************** */
__global__ void affineKernel(float* transformationMatrix,
									  float* defField,
									  int *mask,
									  const uint3 dims,
									  const unsigned long voxelNumber,
									  const bool composition)
{
	// Get the current coordinate
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
	const unsigned long index = x + dims.x * ( y + z * dims.y);

	if( z<dims.z && y<dims.y && x<dims.x &&  mask[index] >= 0 )
	{
		double voxel[3];
		float *deformationFieldPtrX = &defField[index] ;
		float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
		float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

		voxel[0] = composition ? *deformationFieldPtrX : x;
		voxel[1] = composition ? *deformationFieldPtrY : y;
		voxel[2] = composition ? *deformationFieldPtrZ : z;

		/* the deformation field (real coordinates) is stored */
		*deformationFieldPtrX = (float)getPosition(transformationMatrix, voxel, 0);
		*deformationFieldPtrY = (float)getPosition(transformationMatrix, voxel, 1);
		*deformationFieldPtrZ = (float)getPosition(transformationMatrix, voxel, 2);
	}
}
/* *************************************************************** */
__inline__ __device__ double getSquareDistance3Dcu(float * first_point3D, float * second_point3D)
{
	return sqrt((first_point3D[0] - second_point3D[0]) * (first_point3D[0] - second_point3D[0]) + (first_point3D[1] - second_point3D[1]) * (first_point3D[1] - second_point3D[1]) + (first_point3D[2] - second_point3D[2]) * (first_point3D[2] - second_point3D[2]));
}
/* *************************************************************** */
//threads: 512 | blocks:numEquations/512
__global__ void populateLengthsKernel(float* lengths, float* result_d, float* newResult_d, unsigned int numEquations)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int c = tid * 3;

	if (tid < numEquations) {
		newResult_d += c;
		result_d += c;
		lengths[tid] = getSquareDistance3Dcu(result_d, newResult_d);
	}

}
/* *************************************************************** */
void launchAffine(mat44 *affineTransformation,
						nifti_image *deformationField,
						float **def_d,
						int **mask_d,
						float **trans_d,
						bool compose) {

	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((deformationField->nx % xThreads) == 0) ? (deformationField->nx / xThreads) : (deformationField->nx / xThreads) + 1;
	const unsigned int yBlocks = ((deformationField->ny % yThreads) == 0) ? (deformationField->ny / yThreads) : (deformationField->ny / yThreads) + 1;
	const unsigned int zBlocks = ((deformationField->nz % zThreads) == 0) ? (deformationField->nz / zThreads) : (deformationField->nz / zThreads) + 1;

	dim3 G1_b(xBlocks, yBlocks, zBlocks);
	dim3 B1_b(xThreads, yThreads, zThreads);

	float* trans = (float *) malloc(16 * sizeof(float));
	const mat44 *targetMatrix = (deformationField->sform_code > 0) ? &(deformationField->sto_xyz) : &(deformationField->qto_xyz);
	mat44 transformationMatrix = (compose == true) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);
	mat44ToCptr(transformationMatrix, trans);
	NR_CUDA_SAFE_CALL(cudaMemcpy(*trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));
	free(trans);

	uint3 dims_d = make_uint3(deformationField->nx, deformationField->ny, deformationField->nz);
	affineKernel<<<G1_b, B1_b >>>(*trans_d, *def_d, *mask_d, dims_d, deformationField->nx* deformationField->ny* deformationField->nz, compose);
	NR_CUDA_CHECK_KERNEL(G1_b, B1_b)
			NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

}
/* *************************************************************** */
void launchResample(nifti_image *floatingImage,
						  nifti_image *warpedImage,
						  int interp,
						  float paddingValue,
						  bool *dti_timepoint,
						  mat33 *jacMat,
						  float **floatingImage_d,
						  float **warpedImage_d,
						  float **deformationFieldImage_d,
						  int **mask_d,
						  float **sourceIJKMatrix_d) {

	// Define the DTI indices if required
	if(dti_timepoint!=NULL || jacMat!=NULL){
		reg_print_fct_error("launchResample");
		reg_print_msg_error("The DTI resampling has not yet been implemented with the CUDA platform. Exit.");
		reg_exit(1);
	}

	long targetVoxelNumber = (long) warpedImage->nx * warpedImage->ny * warpedImage->nz;

	//the below lines need to be moved to cu common
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	unsigned int maxThreads = 512;
	unsigned int maxBlocks = 65365;
	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min1(blocks, maxBlocks);

	dim3 mygrid(blocks, 1, 1);
	dim3 myblocks(maxThreads, 1, 1);

	ulong2 voxelNumber = make_ulong2(warpedImage->nx * warpedImage->ny * warpedImage->nz, floatingImage->nx * floatingImage->ny * floatingImage->nz);
	uint3 fi_xyz = make_uint3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
	uint2 wi_tu = make_uint2(warpedImage->nt, warpedImage->nu);

	ResampleImage3D<<<mygrid, myblocks>>>(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, *sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue, interp);

    NR_CUDA_CHECK_KERNEL(mygrid, myblocks);
    NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
}
/* *************************************************************** */
void identityConst()
{
	float* mat_h = (float*) malloc(16 * sizeof(float));
	mat44 *final = new mat44();
	// Set the current transformation to identity
	final->m[0][0] = final->m[1][1] = final->m[2][2] = final->m[3][3] = 1.0f;
	final->m[0][1] = final->m[0][2] = final->m[0][3] = 0.0f;
	final->m[1][0] = final->m[1][2] = final->m[1][3] = 0.0f;
	final->m[2][0] = final->m[2][1] = final->m[2][3] = 0.0f;
	final->m[3][0] = final->m[3][1] = final->m[3][2] = 0.0f;
	mat44ToCptr(*final, mat_h);
	cudaMemcpyToSymbol(cIdentity, &mat_h, 16 * sizeof(float));
}
/* *************************************************************** */
double sortAndReduce(float* lengths_d,
							float* target_d,
							float* result_d,
							float* newResult_d,
							const unsigned int numBlocks,
							const unsigned int numToKeep,
							const unsigned int m)
{
	//populateLengthsKernel
	populateLengthsKernel<<<numBlocks, 512>>>(lengths_d, result_d, newResult_d, m/3);

	// The initial vector with all the input points
	thrust::device_ptr<float> target_d_ptr(target_d);
	thrust::device_vector<float> vecTarget_d(target_d_ptr, target_d_ptr + m);

	thrust::device_ptr<float> result_d_ptr(result_d);
	thrust::device_vector<float> vecResult_d(result_d_ptr, result_d_ptr + m);

	thrust::device_ptr<float> lengths_d_ptr(lengths_d);
	thrust::device_vector<float> vec_lengths_d(lengths_d_ptr, lengths_d_ptr + m/3);

	// initialize indices vector to [0,1,2,..m]
	thrust::counting_iterator<int> iter(0);
	thrust::device_vector<int> indices(m);
	thrust::copy(iter, iter + indices.size(), indices.begin());

	//sort an indices array by lengths as key. Then use it to sort target and result arrays
	thrust::sort_by_key(vec_lengths_d.begin(), vec_lengths_d.end(), indices.begin());
	thrust::gather(indices.begin(), indices.end(), vecTarget_d.begin(), vecTarget_d.begin());//end()?
	thrust::gather(indices.begin(), indices.end(), vecResult_d.begin(), vecResult_d.begin());//end()?

	return thrust::reduce(lengths_d_ptr, lengths_d_ptr + numToKeep,0, thrust::plus<double>());
}
/* *************************************************************** */
