#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include"_reg_resampling.h"
#include"_reg_maths.h"
#include "resampleKernel.h"
#include "_reg_common_cuda.h"
#include"_reg_tools.h"
#include"_reg_ReadWriteImage.h"

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
__device__ __inline__ int cuda_reg_floor(double a)
{
   return (int) (floor(a));
}
/* *************************************************************** */
template<class FieldTYPE>
__device__ __inline__ void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis)
{
    if (ratio < 0.0)
        ratio = 0.0; //reg_rounding error
    double FF = (double) ratio * ratio;
    basis[0] = (FieldTYPE) ((ratio * (((double)2.0 - ratio) * ratio - (double)1.0)) / (double)2.0);
    basis[1] = (FieldTYPE) ((FF * ((double)3.0 * ratio - 5.0) + 2.0) / (double)2.0);
    basis[2] = (FieldTYPE) ((ratio * (((double)4.0 - (double)3.0 * ratio) * ratio + (double)1.0)) / (double)2.0);
    basis[3] = (FieldTYPE) ((ratio - (double)1.0) * FF / (double)2.0);
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
    if (relative >= 0.5)
		basis[1] = 1;
	else
		basis[0] = 1;
}
/* *************************************************************** */
__inline__ __device__ double interpLoop2D(float* floatingIntensity,
    double* xBasis,
    double* yBasis,
    double* zBasis,
    int *previous,
    uint3 fi_xyz,
    float paddingValue,
    unsigned int kernel_size)
{
    double intensity = (double)(0.0);

        for (int b = 0; b < kernel_size; b++) {
            int Y = previous[1] + b;
            bool yInBounds = -1 < Y && Y < fi_xyz.y;
            double xTempNewValue = 0.0;

            for (int a = 0; a < kernel_size; a++) {
                int X = previous[0] + a;
                bool xInBounds = -1 < X && X < fi_xyz.x;

                const unsigned int idx = Y * fi_xyz.x + X;

                xTempNewValue += (xInBounds && yInBounds) ? floatingIntensity[idx] * xBasis[a] : paddingValue * xBasis[a];
            }
            intensity += xTempNewValue * yBasis[b];
        }
    return intensity;
}
/* *************************************************************** */
__inline__ __device__ double interpLoop3D(float* floatingIntensity,
                                          double* xBasis,
                                          double* yBasis,
                                          double* zBasis,
                                          int *previous,
                                          uint3 fi_xyz,
                                          float paddingValue,
                                          unsigned int kernel_size)
{
	double intensity = (double)(0.0);
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
__global__ void ResampleImage2D(float* floatingImage,
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

                world[0] = (float)(deformationFieldPtrX[index]);
                world[1] = (float)(deformationFieldPtrY[index]);
                world[2] = 0.0f;

                // real -> voxel; floating space
                reg_mat44_mul_cuda<float>(sourceIJKMatrix, world, position);

                previous[0] = cuda_reg_floor(position[0]);
                previous[1] = cuda_reg_floor(position[1]);

                relative[0] = (double)(position[0]) - (double)(previous[0]);
                relative[1] = (double)(position[1]) - (double)(previous[1]);

                if (kernelType == 0) {

                    double xBasisIn[2], yBasisIn[2], zBasisIn[2];
                    interpNearestNeighKernel(relative[0], xBasisIn);
                    interpNearestNeighKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
                }
                else if (kernelType == 1) {

                    double xBasisIn[2], yBasisIn[2], zBasisIn[2];
                    interpLinearKernel(relative[0], xBasisIn);
                    interpLinearKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
                }
                else if (kernelType == 4) {

                    double xBasisIn[6], yBasisIn[6], zBasisIn[6];

                    previous[0] -= SINC_KERNEL_RADIUS;
                    previous[1] -= SINC_KERNEL_RADIUS;
                    previous[2] -= SINC_KERNEL_RADIUS;

                    interpWindowedSincKernel(relative[0], xBasisIn);
                    interpWindowedSincKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
                }
                else {

                    double xBasisIn[4], yBasisIn[4], zBasisIn[4];

                    previous[0]--;
                    previous[1]--;
                    previous[2]--;

                    interpCubicSplineKernel(relative[0], xBasisIn);
                    interpCubicSplineKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
                }
            }
            resultIntensity[index] = (float)intensity;
        }
        index += blockDim.x * gridDim.x;
    }
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

            world[0] = (float) (deformationFieldPtrX[index]);
                world[1] = (float) (deformationFieldPtrY[index]);
                world[2] = (float) (deformationFieldPtrZ[index]);

				// real -> voxel; floating space
				reg_mat44_mul_cuda<float>(sourceIJKMatrix, world, position);

				previous[0] = cuda_reg_floor(position[0]);
				previous[1] = cuda_reg_floor(position[1]);
				previous[2] = cuda_reg_floor(position[2]);

                relative[0] = (double)(position[0]) - (double)(previous[0]);
                relative[1] = (double)(position[1]) - (double)(previous[1]);
                relative[2] = (double)(position[2]) - (double)(previous[2]);

            if (kernelType == 0) {

					double xBasisIn[2], yBasisIn[2], zBasisIn[2];
					interpNearestNeighKernel(relative[0], xBasisIn);
					interpNearestNeighKernel(relative[1], yBasisIn);
					interpNearestNeighKernel(relative[2], zBasisIn);
					intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
				} else if (kernelType == 1) {

					double xBasisIn[2], yBasisIn[2], zBasisIn[2];
					interpLinearKernel(relative[0], xBasisIn);
					interpLinearKernel(relative[1], yBasisIn);
					interpLinearKernel(relative[2], zBasisIn);
					intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
				} else if (kernelType == 4) {

					double xBasisIn[6], yBasisIn[6], zBasisIn[6];

					previous[0] -= SINC_KERNEL_RADIUS;
					previous[1] -= SINC_KERNEL_RADIUS;
					previous[2] -= SINC_KERNEL_RADIUS;

					interpWindowedSincKernel(relative[0], xBasisIn);
					interpWindowedSincKernel(relative[1], yBasisIn);
					interpWindowedSincKernel(relative[2], zBasisIn);
					intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
				} else {

					double xBasisIn[4], yBasisIn[4], zBasisIn[4];

					previous[0]--;
					previous[1]--;
					previous[2]--;

					interpCubicSplineKernel(relative[0], xBasisIn);
					interpCubicSplineKernel(relative[1], yBasisIn);
					interpCubicSplineKernel(relative[2], zBasisIn);
					intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
				}
			}
            resultIntensity[index] = (float)intensity;
		}
		  index += blockDim.x * gridDim.x;
	}
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
		reg_exit();
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
	 if (floatingImage->nz > 1) {
		  ResampleImage3D <<<mygrid, myblocks >>>(*floatingImage_d,
																*deformationFieldImage_d,
																*warpedImage_d,
																*mask_d,
																*sourceIJKMatrix_d,
																voxelNumber,
																fi_xyz,
																wi_tu,
																paddingValue,
																interp);
	 }
	 else{
		  ResampleImage2D <<<mygrid, myblocks >>>(*floatingImage_d,
																*deformationFieldImage_d,
																*warpedImage_d,
																*mask_d,
																*sourceIJKMatrix_d,
																voxelNumber,
																fi_xyz,
																wi_tu,
																paddingValue,
																interp);
	 }
#ifndef NDEBUG
	NR_CUDA_CHECK_KERNEL(mygrid, myblocks)
#else
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif
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
