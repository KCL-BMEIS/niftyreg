
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include"_reg_blocksize_gpu.h"
#include"_reg_resampling.h"
#include"_reg_maths.h"
#include "cudaKernelFuncs.h"
#include "_reg_common_gpu.h"

#include"_reg_tools.h"
#include"_reg_ReadWriteImage.h"
#include "cuda_profiler_api.h"


#include "_reg_resampling.h"
#include "_reg_maths.h"
#include "_reg_blockMatching_gpu.h"
#include "_reg_blockMatching.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

unsigned int min1(unsigned int a, unsigned int b) {
	return (a < b) ? a : b;
}

texture<float, 3, cudaReadModeElementType> floatingTexture;

__device__ __constant__ float cIdentity[16];
void runKernel(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationFieldImage, int *mask, int interp, float paddingValue, int *dtiIndeces, mat33 * jacMat);

__device__ __inline__ void reg_mat44_expm_cuda(float* mat) {
	//todo 
}

__device__ __inline__
void reg_mat44_logm_cuda(float* mat) {
	//todo
}


template <class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(DTYPE const* mat, DTYPE const* in, DTYPE *out) {
	out[0] = mat[0 * 4 + 0] * in[0] +
		mat[0 * 4 + 1] * in[1] +
		mat[0 * 4 + 2] * in[2] +
		mat[0 * 4 + 3];
	out[1] = mat[1 * 4 + 0] * in[0] +
		mat[1 * 4 + 1] * in[1] +
		mat[1 * 4 + 2] * in[2] +
		mat[1 * 4 + 3];
	out[2] = mat[2 * 4 + 0] * in[0] +
		mat[2 * 4 + 1] * in[1] +
		mat[2 * 4 + 2] * in[2] +
		mat[2 * 4 + 3];
	return;
}


__device__ __inline__ int cuda_reg_floor(float a) {
	return a > 0 ? (int)a : (int)(a - 1);
}

template <class FieldTYPE>
__device__ __inline__ void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis) {
	if (ratio < 0.0f) ratio = 0.0f; //reg_rounding error
	FieldTYPE FF = ratio*ratio;
	basis[0] = (FieldTYPE)((ratio * ((2.0f - ratio)*ratio - 1.0f)) / 2.0f);
	basis[1] = (FieldTYPE)((FF * (3.0f*ratio - 5.0) + 2.0f) / 2.0f);
	basis[2] = (FieldTYPE)((ratio * ((4.0f - 3.0f*ratio)*ratio + 1.0f)) / 2.0f);
	basis[3] = (FieldTYPE)((ratio - 1.0f) * FF / 2.0f);
}
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

template <class DTYPE>
__global__ void reg_dti_resampling_postprocessing(DTYPE *inputImage, DTYPE *warpedImage, int *mask, float *jacMat, int *dtiIndeces, uint3 fi_xyz, uint2 ii_tu) {
	// If we have some valid diffusion tensor indicies, we need to exponentiate the previously logged tensor components
	// we also need to reorient the tensors based on the local transformation Jacobians

	//if (dtiIndeces[0] != -1) {

	//	long warpedIndex = blockIdx.x*blockDim.x + threadIdx.x;
	//	long voxelNumber = fi_xyz.x*fi_xyz.y*fi_xyz.z;

	//	DTYPE *warpVox, *warpedXX, *warpedXY, *warpedXZ, *warpedYY, *warpedYZ, *warpedZZ;
	//	if (warpedImage != NULL) {
	//		warpVox = static_cast<DTYPE *>(warpedImage);
	//		warpedXX = &warpVox[voxelNumber*dtiIndeces[0]];
	//		warpedXY = &warpVox[voxelNumber*dtiIndeces[1]];
	//		warpedYY = &warpVox[voxelNumber*dtiIndeces[2]];
	//		warpedXZ = &warpVox[voxelNumber*dtiIndeces[3]];
	//		warpedYZ = &warpVox[voxelNumber*dtiIndeces[4]];
	//		warpedZZ = &warpVox[voxelNumber*dtiIndeces[5]];
	//	}
	//	for (int u = 0; u < ii_tu.y; ++u) {
	//		// Now, we need to exponentiate the warped intensities back to give us a regular tensor
	//		// let's reorient each tensor based on the rigid component of the local warping
	//		/* As the tensor has 6 unique components that we need to worry about, read them out
	//		for the warped image. */
	//		const unsigned int txu = ii_tu.x*ii_tu.y;

	//		DTYPE *firstWarpVox = static_cast<DTYPE *>(inputImage);
	//		DTYPE *inputIntensityXX = &firstWarpVox[voxelNumber*(dtiIndeces[0] + txu)];
	//		DTYPE *inputIntensityXY = &firstWarpVox[voxelNumber*(dtiIndeces[1] + txu)];
	//		DTYPE *inputIntensityYY = &firstWarpVox[voxelNumber*(dtiIndeces[2] + txu)];
	//		DTYPE *inputIntensityXZ = &firstWarpVox[voxelNumber*(dtiIndeces[3] + txu)];
	//		DTYPE *inputIntensityYZ = &firstWarpVox[voxelNumber*(dtiIndeces[4] + txu)];
	//		DTYPE *inputIntensityZZ = &firstWarpVox[voxelNumber*(dtiIndeces[5] + txu)];

	//		// Step through each voxel in the warped image
	//		double testSum = 0;
	//		float jacobianMatrix[9], R[9];
	//		float inputTensor[16], warpedTensor[16], RotMat[16], RotMatT[16], preMult[16];
	//		int col, row;

	//		if (mask[warpedIndex] > -1) {
	//			reg_mat44_eye(inputTensor);
	//			// Fill the rest of the mat44 with the tensor components
	//			inputTensor[0 * 4 + 0] = static_cast<double>(inputIntensityXX[warpedIndex]);
	//			inputTensor[0 * 4 + 1] = static_cast<double>(inputIntensityXY[warpedIndex]);
	//			inputTensor[1 * 4 + 0] = inputTensor[0 * 4 + 1];
	//			inputTensor[1 * 4 + 1] = static_cast<double>(inputIntensityYY[warpedIndex]);
	//			inputTensor[0 * 4 + 2] = static_cast<double>(inputIntensityXZ[warpedIndex]);
	//			inputTensor[2 * 4 + 0] = inputTensor[0 * 4 + 2];
	//			inputTensor[1 * 4 + 2] = static_cast<double>(inputIntensityYZ[warpedIndex]);
	//			inputTensor[2 * 4 + 1] = inputTensor[1 * 4 + 2];
	//			inputTensor[2 * 4 + 2] = static_cast<double>(inputIntensityZZ[warpedIndex]);
	//			// Exponentiate the warped tensor
	//			if (warpedImage == NULL) {
	//				inputTensor[3 * 4 + 3] = static_cast<double>(0.0);
	//				reg_mat44_expm_cuda(inputTensor);
	//				testSum = 0.;
	//			}
	//			else {
	//				inputTensor[3 * 4 + 3] = 1.0;
	//				reg_mat44_eye(warpedTensor);
	//				warpedTensor[0 * 4 + 0] = static_cast<double>(warpedXX[warpedIndex]);
	//				warpedTensor[0 * 4 + 1] = static_cast<double>(warpedXY[warpedIndex]);
	//				warpedTensor[1 * 4 + 0] = warpedTensor[0 * 4 + 1];
	//				warpedTensor[1 * 4 + 1] = static_cast<double>(warpedYY[warpedIndex]);
	//				warpedTensor[0 * 4 + 2] = static_cast<double>(warpedXZ[warpedIndex]);
	//				warpedTensor[2 * 4 + 0] = warpedTensor[0 * 4 + 2];
	//				warpedTensor[1 * 4 + 2] = static_cast<double>(warpedYZ[warpedIndex]);
	//				warpedTensor[2 * 4 + 1] = warpedTensor[1 * 4 + 2];
	//				warpedTensor[2 * 4 + 2] = static_cast<double>(warpedZZ[warpedIndex]);
	//				reg_mat44_mul_cuda<DTYPE>(warpedTensor, inputTensor, inputTensor);
	//				testSum = static_cast<double>(warpedTensor[0 * 4 + 0] + warpedTensor[0 * 4 + 1] + warpedTensor[0 * 4 + 2] +
	//					warpedTensor[1 * 4 + 0] + warpedTensor[1 * 4 + 1] + warpedTensor[1 * 4 + 2] +
	//					warpedTensor[2 * 4 + 0] + warpedTensor[2 * 4 + 1] + warpedTensor[2 * 4 + 2]);
	//			}

	//			if (testSum == testSum) {
	//				// Find the rotation matrix from the local warp Jacobian
	//				jacobianMatrix = jacMat[warpedIndex];
	//				// Calculate the polar decomposition of the local Jacobian matrix, which
	//				// tells us how to rotate the local tensor information
	//				R = nifti_mat33_polar(jacobianMatrix);
	//				// We need both the rotation matrix, and it's transpose as a mat44
	//				reg_mat44_eye(&RotMat);
	//				reg_mat44_eye(&RotMatT);
	//				for (col = 0; col < 3; col++) {
	//					for (row = 0; row < 3; row++) {
	//						RotMat[col * 4 + row] = static_cast<double>(R[col * 4 + row]);
	//						RotMatT[col * 4 + row] = static_cast<double>(R[row * 4 + col]);
	//					}
	//				}
	//				// As the mat44 multiplication uses pointers, do the multiplications separately
	//				reg_mat44_mul_cuda(RotMatT, inputTensor, preMult);
	//				reg_mat44_mul_cuda(preMult, RotMat, inputTensor);

	//				// Finally, read the tensor back out as a warped image
	//				inputIntensityXX[warpedIndex] = static_cast<DTYPE>(inputTensor[0 * 4 + 0]);
	//				inputIntensityYY[warpedIndex] = static_cast<DTYPE>(inputTensor[1 * 4 + 1]);
	//				inputIntensityZZ[warpedIndex] = static_cast<DTYPE>(inputTensor[2 * 4 + 2]);
	//				inputIntensityXY[warpedIndex] = static_cast<DTYPE>(inputTensor[0 * 4 + 1]);
	//				inputIntensityXZ[warpedIndex] = static_cast<DTYPE>(inputTensor[0 * 4 + 2]);
	//				inputIntensityYZ[warpedIndex] = static_cast<DTYPE>(inputTensor[1 * 4 + 2]);
	//			}
	//			else {
	//				inputIntensityXX[warpedIndex] = 0;
	//				inputIntensityYY[warpedIndex] = 0;
	//				inputIntensityZZ[warpedIndex] = 0;
	//				inputIntensityXY[warpedIndex] = 0;
	//				inputIntensityXZ[warpedIndex] = 0;
	//				inputIntensityYZ[warpedIndex] = 0;
	//			}
	//		}

	//	}
	//}
}

template <class DTYPE>
__global__
void reg_dti_resampling_preprocessing(DTYPE *floatingImage, int *dtiIndeces, uint3 fi_xyz) {
	// If we have some valid diffusion tensor indicies, we need to replace the tensor components
	// by the the log tensor components

	//if (dtiIndeces[0] != -1) {

	//	long floatingIndex;
	//	long floatingVoxelNumber = (long)fi_xyz.x*fi_xyz.y*fi_xyz.z;


	//	/* As the tensor has 6 unique components that we need to worry about, read them out
	//	for the floating image. */
	//	DTYPE *firstVox = static_cast<DTYPE *>(floatingImage);
	//	DTYPE *floatingIntensityXX = &firstVox[floatingVoxelNumber*dtiIndeces[0]];
	//	DTYPE *floatingIntensityXY = &firstVox[floatingVoxelNumber*dtiIndeces[1]];
	//	DTYPE *floatingIntensityYY = &firstVox[floatingVoxelNumber*dtiIndeces[2]];
	//	DTYPE *floatingIntensityXZ = &firstVox[floatingVoxelNumber*dtiIndeces[3]];
	//	DTYPE *floatingIntensityYZ = &firstVox[floatingVoxelNumber*dtiIndeces[4]];
	//	DTYPE *floatingIntensityZZ = &firstVox[floatingVoxelNumber*dtiIndeces[5]];

	//	// We need a mat44 to store the diffusion tensor at each voxel for our calculating. Although the DT is 3x3 really,
	//	// it is convenient to store it as a 4x4 to work with existing code for the matrix logarithm/exponential
	//	float diffTensor[16];

	//	// Should log the tensor up front
	//	// We need to take the logarithm of the tensor for each voxel in the floating intensity image, and replace the warped

	//	long index = blockIdx.x*blockDim.x + threadIdx.x;
	//	// Check that the tensor component is not extremely small or extremely large
	//	if ((floatingIntensityXX[floatingIndex] > 1e-10) && (floatingIntensityXX[floatingIndex] < 1e10)) {
	//		// Fill a mat44 with the tensor components
	//		//reg_mat44_eye(&diffTensor);
	//		diffTensor[0 * 4 + 0] = floatingIntensityXX[floatingIndex];
	//		diffTensor[0 * 4 + 1] = floatingIntensityXY[floatingIndex];
	//		diffTensor[1 * 4 + 0] = diffTensor[0 * 4 + 1];
	//		diffTensor[1 * 4 + 1] = floatingIntensityYY[floatingIndex];
	//		diffTensor[0 * 4 + 2] = floatingIntensityXZ[floatingIndex];
	//		diffTensor[2 * 4 + 0] = diffTensor[0 * 4 + 2];
	//		diffTensor[1 * 4 + 2] = floatingIntensityYZ[floatingIndex];
	//		diffTensor[2 * 4 + 1] = diffTensor[1 * 4 + 2];
	//		diffTensor[2 * 4 + 2] = floatingIntensityZZ[floatingIndex];
	//		// Decompose the mat33 into a rotation and a diagonal matrix of eigen values
	//		// Recompose as a log tensor Rt log(E) R, where E is a diagonal matrix
	//		// containing the eigen values and R is a rotation matrix. This is the same as
	//		// taking the logarithm of the tensor
	//		reg_mat44_logm_cuda(diffTensor);
	//		// Write this out as a new image
	//		floatingIntensityXX[floatingIndex] = static_cast<DTYPE>(diffTensor[0 * 4 + 0]);
	//		floatingIntensityXY[floatingIndex] = static_cast<DTYPE>(diffTensor[0 * 4 + 1]);
	//		floatingIntensityYY[floatingIndex] = static_cast<DTYPE>(diffTensor[1 * 4 + 1]);
	//		floatingIntensityXZ[floatingIndex] = static_cast<DTYPE>(diffTensor[0 * 4 + 2]);
	//		floatingIntensityYZ[floatingIndex] = static_cast<DTYPE>(diffTensor[1 * 4 + 2]);
	//		floatingIntensityZZ[floatingIndex] = static_cast<DTYPE>(diffTensor[2 * 4 + 2]);
	//	}
	//	else  // if junk diffusion data, set the diagonal to log the minimum value
	//	{
	//		floatingIntensityXX[floatingIndex] = static_cast<DTYPE>(-23.02585f);
	//		floatingIntensityYY[floatingIndex] = static_cast<DTYPE>(-23.02585f);
	//		floatingIntensityZZ[floatingIndex] = static_cast<DTYPE>(-23.02585f);
	//		floatingIntensityXY[floatingIndex] = static_cast<DTYPE>(0.0f);
	//		floatingIntensityXZ[floatingIndex] = static_cast<DTYPE>(0.0f);
	//		floatingIntensityYZ[floatingIndex] = static_cast<DTYPE>(0.0f);
	//	}
	//}
}

__global__ void CubicSplineResampleImage3D(float *floatingImage, float *deformationField, float *warpedImage, int *mask, /*mat44*/float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {
	//long resultVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;vn.x
	//long sourceVoxelNumber = (long)floatingImage->nx*floatingImage->ny*floatingImage->nz;vn.y

	float *sourceIntensityPtr = (floatingImage);
	float *resultIntensityPtr = (warpedImage);
	float *deformationFieldPtrX = (deformationField);
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	int *maskPtr = &mask[0];
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	while (index < voxelNumber.x) {

		// Iteration over the different volume along the 4th axis
		for (unsigned int t = 0; t < wi_tu.x*wi_tu.y; t++) {


			float *resultIntensity = &resultIntensityPtr[t*voxelNumber.x];
			float *sourceIntensity = &sourceIntensityPtr[t*voxelNumber.y];

			float xBasis[4], yBasis[4], zBasis[4], relative;
			int a, b, c, Y, Z, previous[3];

			float *zPointer, *yzPointer, *xyzPointer;
			float xTempNewValue, yTempNewValue, intensity, world[3], position[3];



			intensity = (0.0f);

			if ((maskPtr[index]) > -1) {
				world[0] = deformationFieldPtrX[index];
				world[1] = deformationFieldPtrY[index];
				world[2] = deformationFieldPtrZ[index];

				/* real -> voxel; source space */
				reg_mat44_mul_cuda(sourceIJKMatrix, world, position);

				previous[0] = (cuda_reg_floor(position[0]));
				previous[1] = (cuda_reg_floor(position[1]));
				previous[2] = (cuda_reg_floor(position[2]));

				// basis values along the x axis
				relative = position[0] - previous[0];
				relative = relative > 0 ? relative : 0;
				interpolantCubicSpline<float>(relative, xBasis);
				// basis values along the y axis
				relative = position[1] - previous[1];
				relative = relative > 0 ? relative : 0;
				interpolantCubicSpline<float>(relative, yBasis);
				// basis values along the z axis
				relative = position[2] - previous[2];
				relative = relative > 0 ? relative : 0;
				interpolantCubicSpline<float>(relative, zBasis);

				--previous[0];
				--previous[1];
				--previous[2];

				for (c = 0; c < 4; c++) {
					Z = previous[2] + c;
					zPointer = &sourceIntensity[Z*fi_xyz.x*fi_xyz.y];
					yTempNewValue = 0.0;
					for (b = 0; b < 4; b++) {
						Y = previous[1] + b;
						yzPointer = &zPointer[Y*fi_xyz.x];
						xyzPointer = &yzPointer[previous[0]];
						xTempNewValue = 0.0;
						for (a = 0; a < 4; a++) {
							if (-1 < (previous[0] + a) && (previous[0] + a) < fi_xyz.x &&
								-1 < Z && Z < fi_xyz.z &&
								-1 < Y && Y < fi_xyz.y) {
								xTempNewValue += *xyzPointer * xBasis[a];
							}
							else {
								// paddingValue
								xTempNewValue += paddingValue * xBasis[a];
							}
							xyzPointer++;
						}
						yTempNewValue += xTempNewValue * yBasis[b];
					}
					intensity += yTempNewValue * zBasis[c];
				}
			}

			resultIntensity[index] = intensity;
		}
		index += blockDim.x*gridDim.x;
	}
}

/* *************************************************************** */
__global__ void NearestNeighborResampleImage(float *floatingImage, float *deformationField, float *warpedImage, int *mask, /*mat44*/float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {

	// The resampling scheme is applied along each time

	float *sourceIntensityPtr = (floatingImage);
	float *resultIntensityPtr = (warpedImage);
	float *deformationFieldPtrX = (deformationField);
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	int *maskPtr = &mask[0];


	long index = blockIdx.x*blockDim.x + threadIdx.x;
	while (index < voxelNumber.x) {
		for (int t = 0; t<wi_tu.x*wi_tu.x; t++) {

			float *resultIntensity = &resultIntensityPtr[t*voxelNumber.x];
			float *sourceIntensity = &sourceIntensityPtr[t*voxelNumber.y];

			float intensity;
			float world[3];
			float position[3];
			int previous[3];

			if (maskPtr[index]>-1) {
				world[0] = (float)deformationFieldPtrX[index];
				world[1] = (float)deformationFieldPtrY[index];
				world[2] = (float)deformationFieldPtrZ[index];

				/* real -> voxel; source space */
				reg_mat44_mul_cuda(sourceIJKMatrix, world, position);

				previous[0] = (int)reg_round(position[0]);
				previous[1] = (int)reg_round(position[1]);
				previous[2] = (int)reg_round(position[2]);

				if (-1 < previous[2] && previous[2] < fi_xyz.z &&
					-1 < previous[1] && previous[1] < fi_xyz.y &&
					-1 < previous[0] && previous[0] < fi_xyz.x) {
					intensity = sourceIntensity[(previous[2] * fi_xyz.y + previous[1]) * fi_xyz.x + previous[0]];
					resultIntensity[index] = intensity;
				}
				else resultIntensity[index] = paddingValue;
			}
			else resultIntensity[index] = paddingValue;


		}
		index += blockDim.x*gridDim.x;
	}

}

__global__ void TrilinearResampleImage(float *floatingImage, float *deformationField, float *warpedImage, int *mask, /*mat44*/float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {

	//if( threadIdx.x == 0 ) printf("block: %d \n", blockIdx.x);

	//targetVoxelNumber voxelNumber.x
	// sourceVoxelNumber voxelNumber.y

	//intensity images
	float *sourceIntensityPtr = (floatingImage);//best to be a texture
	float *resultIntensityPtr = (warpedImage);

	//deformation field image
	float *deformationFieldPtrX = (deformationField);
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	int *maskPtr = &mask[0];

	// The resampling scheme is applied along each time

	long index = blockIdx.x*blockDim.x + threadIdx.x;
	while (index < voxelNumber.x) {
		for (unsigned int t = 0; t<wi_tu.x*wi_tu.y; t++) {


			float *resultIntensity = &resultIntensityPtr[t*voxelNumber.x];
			float *sourceIntensity = &sourceIntensityPtr[t*voxelNumber.y];

			float xBasis[2], yBasis[2], zBasis[2], relative;
			int a, b, c, X, Y, Z, previous[3];

			float *zPointer, *xyzPointer;
			float xTempNewValue, yTempNewValue, intensity, world[3], position[3];

			//for( index = 0; index<targetVoxelNumber; index++ ) {

			intensity = paddingValue;

			if (maskPtr[index]>-1) {

				intensity = 0;

				world[0] = deformationFieldPtrX[index];
				world[1] = deformationFieldPtrY[index];
				world[2] = deformationFieldPtrZ[index];

				/* real -> voxel; source space */
				reg_mat44_mul_cuda<float>(sourceIJKMatrix, world, position);

				previous[0] = cuda_reg_floor(position[0]);
				previous[1] = cuda_reg_floor(position[1]);
				previous[2] = cuda_reg_floor(position[2]);

				// basis values along the x axis
				relative = position[0] - previous[0];
				xBasis[0] = (1.0 - relative);
				xBasis[1] = relative;
				// basis values along the y axis
				relative = position[1] - previous[1];
				yBasis[0] = (1.0 - relative);
				yBasis[1] = relative;
				// basis values along the z axis
				relative = position[2] - previous[2];
				zBasis[0] = (1.0 - relative);
				zBasis[1] = relative;

				// For efficiency reason two interpolation are here, with and without using a padding value
				if (paddingValue==paddingValue) {
					// Interpolation using the padding value
					for (c = 0; c<2; c++) {
						Z = previous[2] + c;
						if (Z>-1 && Z < fi_xyz.z) {
							zPointer = &sourceIntensity[Z*fi_xyz.x*fi_xyz.y];
							yTempNewValue = 0.0;
							for (b = 0; b<2; b++) {
								Y = previous[1] + b;
								if (Y>-1 && Y < fi_xyz.y) {
									xyzPointer = &zPointer[Y*fi_xyz.x + previous[0]];
									xTempNewValue = 0.0;
									for (a = 0; a<2; a++) {
										X = previous[0] + a;
										if (X>-1 && X < fi_xyz.x) {
											xTempNewValue += *xyzPointer * xBasis[a];
										} // X
										else xTempNewValue += paddingValue * xBasis[a];
										xyzPointer++;
									} // a
									yTempNewValue += xTempNewValue * yBasis[b];
								} // Y
								else yTempNewValue += paddingValue * yBasis[b];
							} // b
							intensity += yTempNewValue * zBasis[c];
						} // Z
						else intensity += paddingValue * zBasis[c];
					} // c
				} // padding value is defined
				else if (previous[0] >= 0.f && previous[0] < (fi_xyz.x - 1) &&
					previous[1] >= 0.f && previous[1] < (fi_xyz.y - 1) &&
					previous[2] >= 0.f && previous[2] < (fi_xyz.z - 1)) {
					for (c = 0; c < 2; c++) {
						Z = previous[2] + c;
						zPointer = &sourceIntensity[Z*fi_xyz.x*fi_xyz.y];
						yTempNewValue = 0.0;
						for (b = 0; b < 2; b++) {
							Y = previous[1] + b;
							xyzPointer = &zPointer[Y*fi_xyz.x + previous[0]];
							xTempNewValue = 0.0;
							for (a = 0; a < 2; a++) {
								X = previous[0] + a;
								xTempNewValue += *xyzPointer * xBasis[a];
								xyzPointer++;
							} // a
							yTempNewValue += xTempNewValue * yBasis[b];
						} // b
						intensity += yTempNewValue * zBasis[c];
					} // c
				} // padding value is not defined
				// The voxel is outside of the source space and thus set to NaN here
				else intensity = paddingValue;
			} // voxel is in the mask

			resultIntensity[index] = intensity;

			//}
		}
		index += blockDim.x*gridDim.x;
	}

}


__device__ __inline__ void getPosition(float* position, float* matrix, float* voxel, const unsigned int idx) {
	position[idx] =
		matrix[idx * 4 + 0] * voxel[0] +
		matrix[idx * 4 + 1] * voxel[1] +
		matrix[idx * 4 + 2] * voxel[2] +
		matrix[idx * 4 + 3];
}

__global__ void affineKernel(float* transformationMatrix, float* defField, int* mask, const uint3 params, const unsigned long voxelNumber, const bool composition) {

	float *deformationFieldPtrX = defField;
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

	float voxel[3], position[3];


	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned long index = x + y*params.x + z * params.x * params.y;
	if (z < params.z && y < params.y && x < params.x &&  mask[index] >= 0) {

		voxel[0] = composition ? deformationFieldPtrX[index] : x;
		voxel[1] = composition ? deformationFieldPtrY[index] : y;
		voxel[2] = composition ? deformationFieldPtrZ[index] : z;

		getPosition(position, transformationMatrix, voxel, 0);
		getPosition(position, transformationMatrix, voxel, 1);
		getPosition(position, transformationMatrix, voxel, 2);

		/* the deformation field (real coordinates) is stored */
		deformationFieldPtrX[index] = position[0];
		deformationFieldPtrY[index] = position[1];
		deformationFieldPtrZ[index] = position[2];

	}
}

template<class DTYPE>
__global__ void convolutionKernel(nifti_image *image, float*densityPtr, bool* nanImagePtr, float *size, int kernelType, int *mask, bool *timePoint, bool *axis) {
	if (threadIdx.x == 0) {
		//printf("hi from %d-%d \n", blockIdx.x, threadIdx.x);
		const unsigned long voxelNumber = image->dim[1] * image->dim[2] * image->dim[3];
		DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
		int imageDim[3] = { image->dim[1], image->dim[2], image->dim[3] };


		// Loop over the dimension higher than 3
		for (int t = 0; t < image->dim[4] * image->dim[5]; t++) {
			if (timePoint[t]) {
				DTYPE *intensityPtr = &imagePtr[t * voxelNumber];

				for (unsigned long index = 0; index < voxelNumber; index++) {
					densityPtr[index] = (intensityPtr[index] == intensityPtr[index]) ? 1 : 0;
					densityPtr[index] *= (mask[index] >= 0) ? 1 : 0;
					nanImagePtr[index] = static_cast<bool>(densityPtr[index]);
					if (nanImagePtr[index] == 0)
						intensityPtr[index] = static_cast<DTYPE>(0);
				}
				// Loop over the x, y and z dimensions
				for (int n = 0; n < 3; n++) {
					if (axis[n] && image->dim[n] > 1) {
						double temp;
						if (size[t]>0) temp = size[t] / image->pixdim[n + 1]; // mm to voxel
						else temp = fabs(size[t]); // voxel based if negative value
						int radius;
						// Define the kernel size
						if (kernelType == 2) {
							// Mean filtering
							radius = static_cast<int>(temp);
						}
						else if (kernelType == 1) {
							// Cubic Spline kernel
							radius = static_cast<int>(temp*2.0f);
						}
						else {
							// Gaussian kernel
							radius = static_cast<int>(temp*3.0f);
						}
						if (radius > 0) {
							// Allocate the kernel
							float kernel[2048];
							double kernelSum = 0;
							// Fill the kernel
							if (kernelType == 1) {
								// Compute the Cubic Spline kernel
								for (int i = -radius; i <= radius; i++) {
									// temp contains the kernel node spacing
									double relative = (double)(fabs((double)(double)i / (double)temp));
									if (relative < 1.0) kernel[i + radius] = (float)(2.0 / 3.0 - relative*relative + 0.5*relative*relative*relative);
									else if (relative < 2.0) kernel[i + radius] = (float)(-(relative - 2.0)*(relative - 2.0)*(relative - 2.0) / 6.0);
									else kernel[i + radius] = 0;
									kernelSum += kernel[i + radius];
								}
							}
							// No kernel is required for the mean filtering
							else if (kernelType != 2) {
								// Compute the Gaussian kernel
								for (int i = -radius; i <= radius; i++) {
									// 2.506... = sqrt(2*pi)
									// temp contains the sigma in voxel
									kernel[radius + i] = static_cast<float>(exp(-(double)(i*i) / (2.0*reg_pow2(temp))) /
										(temp*2.506628274631));
									kernelSum += kernel[radius + i];
								}
							}
							// No need for kernel normalisation as this is handle by the density function
							int planeNumber, planeIndex, lineOffset;
							int lineIndex, shiftPre, shiftPst, k;
							switch (n) {
							case 0:
								planeNumber = imageDim[1] * imageDim[2];
								lineOffset = 1;
								break;
							case 1:
								planeNumber = imageDim[0] * imageDim[2];
								lineOffset = imageDim[0];
								break;
							case 2:
								planeNumber = imageDim[0] * imageDim[1];
								lineOffset = planeNumber;
								break;
							}

							size_t realIndex;
							float *kernelPtr, kernelValue;
							double densitySum, intensitySum;
							DTYPE *currentIntensityPtr = NULL;
							float *currentDensityPtr = NULL;
							DTYPE bufferIntensity[2048];;
							float bufferDensity[2048];
							DTYPE bufferIntensitycur = 0;
							float bufferDensitycur = 0;

							// Loop over the different voxel
							for (planeIndex = 0; planeIndex < planeNumber; ++planeIndex) {

								switch (n) {
								case 0:
									realIndex = planeIndex * imageDim[0];
									break;
								case 1:
									realIndex = (planeIndex / imageDim[0]) *
										imageDim[0] * imageDim[1] +
										planeIndex%imageDim[0];
									break;
								case 2:
									realIndex = planeIndex;
									break;
								default:
									realIndex = 0;
								}
								// Fetch the current line into a stack buffer
								currentIntensityPtr = &intensityPtr[realIndex];
								currentDensityPtr = &densityPtr[realIndex];
								for (lineIndex = 0; lineIndex < imageDim[n]; ++lineIndex) {
									bufferIntensity[lineIndex] = *currentIntensityPtr;
									bufferDensity[lineIndex] = *currentDensityPtr;
									currentIntensityPtr += lineOffset;
									currentDensityPtr += lineOffset;
								}
								if (kernelSum > 0) {
									// Perform the kernel convolution along 1 line
									for (lineIndex = 0; lineIndex < imageDim[n]; ++lineIndex) {
										// Define the kernel boundaries
										shiftPre = lineIndex - radius;
										shiftPst = lineIndex + radius + 1;
										if (shiftPre < 0) {
											kernelPtr = &kernel[-shiftPre];
											shiftPre = 0;
										}
										else kernelPtr = &kernel[0];
										if (shiftPst > imageDim[n]) shiftPst = imageDim[n];
										// Set the current values to zero
										intensitySum = 0;
										densitySum = 0;
										// Increment the current value by performing the weighted sum
										for (k = shiftPre; k < shiftPst; ++k) {
											kernelValue = *kernelPtr++;
											intensitySum += kernelValue * bufferIntensity[k];
											densitySum += kernelValue * bufferDensity[k];
										}
										// Store the computed value inplace
										intensityPtr[realIndex] = static_cast<DTYPE>(intensitySum);
										densityPtr[realIndex] = static_cast<float>(densitySum);
										realIndex += lineOffset;
									} // line convolution
								} // kernel type
								else {
									for (lineIndex = 1; lineIndex < imageDim[n]; ++lineIndex) {
										bufferIntensity[lineIndex] += bufferIntensity[lineIndex - 1];
										bufferDensity[lineIndex] += bufferDensity[lineIndex - 1];
									}
									shiftPre = -radius - 1;
									shiftPst = radius;
									for (lineIndex = 0; lineIndex < imageDim[n]; ++lineIndex, ++shiftPre, ++shiftPst) {
										if (shiftPre > -1) {
											if (shiftPst < imageDim[n]) {
												bufferIntensitycur = (DTYPE)(bufferIntensity[shiftPre] - bufferIntensity[shiftPst]);
												bufferDensitycur = (DTYPE)(bufferDensity[shiftPre] - bufferDensity[shiftPst]);
											}
											else {
												bufferIntensitycur = (DTYPE)(bufferIntensity[shiftPre] - bufferIntensity[imageDim[n] - 1]);
												bufferDensitycur = (DTYPE)(bufferDensity[shiftPre] - bufferDensity[imageDim[n] - 1]);
											}
										}
										else {
											if (shiftPst < imageDim[n]) {
												bufferIntensitycur = (DTYPE)(-bufferIntensity[shiftPst]);
												bufferDensitycur = (DTYPE)(-bufferDensity[shiftPst]);
											}
											else {
												bufferIntensitycur = (DTYPE)(0);
												bufferDensitycur = (DTYPE)(0);
											}
										}
										intensityPtr[realIndex] = bufferIntensitycur;
										densityPtr[realIndex] = bufferDensitycur;

										realIndex += lineOffset;
									} // line convolution of mean filter
								} // No kernel computation
							} // pixel in starting plane
						} // radius > 0
					} // active axis
				} // axes
				// Normalise per timepoint
				for (unsigned long index = 0; index < voxelNumber; ++index) {
					if (nanImagePtr[index] != 0)
						intensityPtr[index] = static_cast<DTYPE>((float)intensityPtr[index] / densityPtr[index]);
					else intensityPtr[index] = 0;
				}
			} // check if the time point is active
		} // loop over the time points
	}
}

void launch(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
	bool *axisToSmooth = new bool[3];
	bool *activeTimePoint = new bool[image->nt*image->nu];
	unsigned long voxelNumber = (long)image->nx*image->ny*image->nz;

	bool *nanImagePtr;
	float *densityPtr;
	float *sigma_d;
	int *mask_d;
	bool* timePoint_d;
	bool* axis_d;


	int dim[3] = { image->nx, image->ny, image->nz };
	std::cout << image->nx << ": " << image->ny << ": " << image->nz << std::endl;
	nifti_image* image_d;


	if (image->nx > 2048 || image->ny > 2048 || image->nz > 2048) {
		reg_print_fct_error("reg_tools_kernelConvolution_core");
		reg_print_msg_error("This function does not support images with dimension > 2048");
		reg_exit(1);
	}

	if (image->nt <= 0) image->nt = image->dim[4] = 1;
	if (image->nu <= 0) image->nu = image->dim[5] = 1;




	/*densityPtr[4] = 8.8f;
	std::cout << "test float: " << densityPtr[4] << std::endl;*/


	if (axis == NULL) {
		// All axis are smoothed by default
		for (int i = 0; i < 3; i++) axisToSmooth[i] = true;
	}
	else for (int i = 0; i < 3; i++) axisToSmooth[i] = axis[i];

	if (timePoint == NULL) {
		// All time points are considered as active
		for (int i = 0; i < image->nt*image->nu; i++) activeTimePoint[i] = true;
	}
	else for (int i = 0; i < image->nt*image->nu; i++) activeTimePoint[i] = timePoint[i];

	int *currentMask = NULL;
	if (mask == NULL) {
		currentMask = (int *)calloc(image->nx*image->ny*image->nz, sizeof(int));
	}
	else currentMask = mask;

	/*cudaCommon_allocateNiftiToDevice<float>(&image_d, dim);
	cudaCommon_transferNiftiToNiftiOnDevice1<float>(&image_d, image);*/

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(sigma_d), image->dim[4] * image->dim[5] * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(sigma_d, sigma, image->dim[4] * image->dim[5] * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(mask_d), voxelNumber * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(mask_d, currentMask, voxelNumber * sizeof(int), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(timePoint_d), image->dim[4] * image->dim[5] * sizeof(bool)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(timePoint_d, timePoint, image->dim[4] * image->dim[5] * sizeof(bool), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(axis_d), 3 * sizeof(bool)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(axis_d, axis, 3 * sizeof(bool), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc(&nanImagePtr, voxelNumber*sizeof(bool)));
	NR_CUDA_SAFE_CALL(cudaMalloc(&densityPtr, voxelNumber*sizeof(float)));

	switch (image->datatype) {
	case NIFTI_TYPE_UINT8:
		//convolutionKernel<unsigned char> <<<1, 1 >>>( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	case NIFTI_TYPE_INT8:
		//convolutionKernel <char> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	case NIFTI_TYPE_UINT16:
		//convolutionKernel <unsigned short> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	case NIFTI_TYPE_INT16:
		//convolutionKernel <short> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	case NIFTI_TYPE_UINT32:
		//convolutionKernel<unsigned int> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	case NIFTI_TYPE_INT32:
		//convolutionKernel <int> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	case NIFTI_TYPE_FLOAT32:
		std::cout << "called instead of kernel!" << std::endl;
		convolutionKernel <float> << <1, 1 >> >(image_d, densityPtr, nanImagePtr, sigma_d, kernelType, mask_d, timePoint_d, axis_d);
		//NR_CUDA_CHECK_KERNEL(1, 1)
		break;
	case NIFTI_TYPE_FLOAT64:
		//convolutionKernel <double> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	default:
		fprintf(stderr, "[NiftyReg ERROR] reg_gaussianSmoothing\tThe image data type is not supported\n");
		reg_exit(1);
	}

	if (mask == NULL) free(currentMask);
	delete[]axisToSmooth;
	delete[]activeTimePoint;
	cudaFree(nanImagePtr);
	cudaFree(densityPtr);
}




nifti_params_t getParams(nifti_image image) {
	nifti_params_t params = {
		image.ndim,                    /*!< last dimension greater than 1 (1..7) */
		image.nx,                      /*!< dimensions of grid array             */
		image.ny,                      /*!< dimensions of grid array             */
		image.nz,                      /*!< dimensions of grid array             */
		image.nt,                      /*!< dimensions of grid array             */
		image.nu,                      /*!< dimensions of grid array             */
		image.nv,                      /*!< dimensions of grid array             */
		image.nw,                      /*!< dimensions of grid array             */
		image.nvox,					   /*!< number of voxels = nx*ny*nz*...*nw   */
		image.nbyper,                  /*!< bytes per voxel, matches datatype    */
		image.datatype,                /*!< type of data in voxels: DT_* code    */

		image.dx,					/*!< grid spacings      */
		image.dy,                   /*!< grid spacings      */
		image.dz,                   /*!< grid spacings      */
		image.dt,                   /*!< grid spacings      */
		image.du,                   /*!< grid spacings      */
		image.dv,                   /*!< grid spacings      */
		image.dw,                    /*!< grid spacings      */
		image.nx*image.ny*image.nz   //xyz image size
	};

	return params;
}
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask) {

	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((deformationField->nx % xThreads) == 0) ? (deformationField->nx / xThreads) : (deformationField->nx / xThreads) + 1;
	const unsigned int yBlocks = ((deformationField->ny % yThreads) == 0) ? (deformationField->ny / yThreads) : (deformationField->ny / yThreads) + 1;
	const unsigned int zBlocks = ((deformationField->nz % zThreads) == 0) ? (deformationField->nz / zThreads) : (deformationField->nz / zThreads) + 1;


	dim3 G1_b(xBlocks, yBlocks, zBlocks);
	dim3 B1_b(xThreads, yThreads, zThreads);


	int *tempMask = mask;
	if (mask == NULL) {
		tempMask = (int *)calloc(deformationField->nx*
			deformationField->ny*
			deformationField->nz,
			sizeof(int));
	}

	const mat44 *targetMatrix = (deformationField->sform_code > 0) ? &(deformationField->sto_xyz) : &(deformationField->qto_xyz);
	mat44 transformationMatrix = (compose == true) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);

	float* trans = (float *)malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	nifti_params params_d = getParams(*deformationField);
	float *trans_d, *def_d;
	int* mask_d;



	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&trans_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&def_d), params_d.nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(def_d, deformationField->data, params_d.nvox * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&mask_d), params_d.nxyz * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(mask_d, tempMask, params_d.nxyz * sizeof(int), cudaMemcpyHostToDevice));



	uint3 pms_d = make_uint3(params_d.nx, params_d.ny, params_d.nz);
	affineKernel << <G1_b, B1_b >> >(trans_d, def_d, mask_d, pms_d, params_d.nxyz, compose);
	NR_CUDA_CHECK_KERNEL(G1_b, B1_b)

		NR_CUDA_SAFE_CALL(cudaMemcpy(deformationField->data, def_d, params_d.nvox * sizeof(float), cudaMemcpyDeviceToHost));

	if (mask == NULL)
		free(tempMask);

	cudaFree(trans_d);
	cudaFree(def_d);
	cudaFree(mask_d);

}
void launchAffine2(mat44 *affineTransformation, nifti_image *deformationField, float** def_d, int** mask_d, bool compose) {

	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((deformationField->nx % xThreads) == 0) ? (deformationField->nx / xThreads) : (deformationField->nx / xThreads) + 1;
	const unsigned int yBlocks = ((deformationField->ny % yThreads) == 0) ? (deformationField->ny / yThreads) : (deformationField->ny / yThreads) + 1;
	const unsigned int zBlocks = ((deformationField->nz % zThreads) == 0) ? (deformationField->nz / zThreads) : (deformationField->nz / zThreads) + 1;


	dim3 G1_b(xBlocks, yBlocks, zBlocks);
	dim3 B1_b(xThreads, yThreads, zThreads);



	const mat44 *targetMatrix = (deformationField->sform_code > 0) ? &(deformationField->sto_xyz) : &(deformationField->qto_xyz);
	mat44 transformationMatrix = (compose == true) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);

	float* trans = (float *)malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	nifti_params params_d = getParams(*deformationField);
	float *trans_d;

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&trans_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));

	uint3 pms_d = make_uint3(params_d.nx, params_d.ny, params_d.nz);
	affineKernel << <G1_b, B1_b >> >(trans_d, *def_d, *mask_d, pms_d, params_d.nxyz, compose);
	//NR_CUDA_CHECK_KERNEL(G1_b, B1_b)
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
	cudaFree(trans_d);
	free(trans);

}
void launchOptimize(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine) {
	float in[3];
	float out[3];
	for (size_t i = 0; i<static_cast<size_t>(params->activeBlockNumber); ++i)
	{
		size_t index = 3 * i;
		in[0] = params->resultPosition[index];
		in[1] = params->resultPosition[index + 1];
		in[2] = params->resultPosition[index + 2];
		reg_mat44_mul(transformation_matrix, in, out);
		params->resultPosition[index++] = out[0];
		params->resultPosition[index++] = out[1];
		params->resultPosition[index] = out[2];
	}
	if (affine)
		launchOptimizeAffine(params, transformation_matrix, true);
	else launchOptimizeRigid(params, transformation_matrix, false);
}

void launchResample(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat) {

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

	// Define the DTI indices if required
	int dtiIndeces[6];
	for (int i = 0; i < 6; ++i) dtiIndeces[i] = -1;
	if (dti_timepoint != NULL) {

		if (jacMat == NULL) {
			printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
			reg_exit(1);
		}
		int j = 0;
		for (int i = 0; i < floatingImage->nt; ++i) {
			if (dti_timepoint[i] == true)
				dtiIndeces[j++] = i;
		}
		if ((floatingImage->nz>1 && j != 6) && (floatingImage->nz == 1 && j != 3)) {
			printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
			reg_exit(1);
		}
	}

	// a mask array is created if no mask is specified
	bool MrPropreRules = false;
	if (mask == NULL) {
		// voxels in the backgreg_round are set to -1 so 0 will do the job here
		mask = (int *)calloc(warpedImage->nx*warpedImage->ny*warpedImage->nz, sizeof(int));
		MrPropreRules = true;
	}


	runKernel(floatingImage, warpedImage, deformationField, mask, interp, paddingValue, dtiIndeces, jacMat);

	if (MrPropreRules == true) {
		free(mask);
		mask = NULL;
	}
}
void launchResample2(nifti_image *floatingImage, nifti_image *warpedImage, int *mask, int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat, float** floatingImage_d,  float** warpedImage_d, float** deformationFieldImage_d, int** mask_d) {

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

	// Define the DTI indices if required
	int dtiIndeces[6];
	for (int i = 0; i < 6; ++i) dtiIndeces[i] = -1;
	if (dti_timepoint != NULL) {

		if (jacMat == NULL) {
			printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
			reg_exit(1);
		}
		int j = 0;
		for (int i = 0; i < floatingImage->nt; ++i) {
			if (dti_timepoint[i] == true)
				dtiIndeces[j++] = i;
		}
		if ((floatingImage->nz>1 && j != 6) && (floatingImage->nz == 1 && j != 3)) {
			printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
			reg_exit(1);
		}
	}

	// a mask array is created if no mask is specified
	bool MrPropreRules = false;
	if (mask == NULL) {
		// voxels in the backgreg_round are set to -1 so 0 will do the job here
		mask = (int *)calloc(warpedImage->nx*warpedImage->ny*warpedImage->nz, sizeof(int));
		MrPropreRules = true;
	}

	//printf("kernel2run");
	runKernel2(floatingImage, warpedImage, mask, interp, paddingValue, dtiIndeces, jacMat,  floatingImage_d, warpedImage_d, deformationFieldImage_d,  mask_d);

	if (MrPropreRules == true) {
		free(mask);
		mask = NULL;
	}
}

void initTextures() {
	cudaArray **floatingImageArray_d;
	//cudaCommon_transferNiftiToArrayOnDevice1(floatingImageArray_d, floatingImage)
	////Bind floating image array to a 3D texture
	//floatingTexture.normalized = false;
	//floatingTexture.filterMode = cudaFilterModeLinear;
	//floatingTexture.addressMode[0] = cudaAddressModeWrap;
	//floatingTexture.addressMode[1] = cudaAddressModeWrap;
	//floatingTexture.addressMode[2] = cudaAddressModeWrap;

	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//NR_CUDA_SAFE_CALL(cudaBindTextureToArray(floatingTexture, *floatingImageArray_d, channelDesc))
}

void runKernel(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationFieldImage, int *mask, int interp, float paddingValue, int *dtiIndeces, mat33 * jacMat) {


	long targetVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;
	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
	unsigned int maxThreads = prop.maxThreadsDim[0];
	unsigned int maxBlocks = prop.maxThreadsDim[0];
	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min1(blocks, maxBlocks);



	dim3 mygrid(blocks, 1, 1);
	dim3 myblocks(maxThreads, 1, 1);
	printf("maxBlocks b: %d | maxThreads: %d\n", maxBlocks, maxThreads);
	printf("blocks: %d | threads: %d\n", blocks, maxThreads);

	// The floating image data is copied in case one deal with DTI
	void *originalFloatingData = NULL;
	originalFloatingData = (void *)malloc(floatingImage->nvox*sizeof(float));
	memcpy(originalFloatingData, floatingImage->data, floatingImage->nvox*sizeof(float));


	int numMats = 0;
	mat44 *sourceIJKMatrix;
	float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
	float* jacMat_h = (float*)malloc(9 * numMats*sizeof(float));

	if (floatingImage->sform_code > 0)
		sourceIJKMatrix = &(floatingImage->sto_ijk);
	else sourceIJKMatrix = &(floatingImage->qto_ijk);

	float *floatingImage_d, *deformationFieldImage_d, *warpedImage_d, paddingValue_d;
	float* sourceIJKMatrix_d, *jacMat_d;
	int* mask_d, *dtiIndeces_d;
	long2 voxelNumber = make_long2(warpedImage->nx*warpedImage->ny*warpedImage->nz, floatingImage->nx*floatingImage->ny*floatingImage->nz);
	uint3 fi_xyz = make_uint3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
	uint2 wi_tu = make_uint2(warpedImage->nt, warpedImage->nu);


	mat44ToCptr(*sourceIJKMatrix, sourceIJKMatrix_h);
	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);

	char* floating = "floating";
	char* floating1 = "deformationFieldImage_d";
	char* floating2 = "warpedImage_d";
	char* floating3 = "mask_d";
	char* floating4 = "matrix";

	//printf("uploading %s\n", floating);

	//floatingImage_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&floatingImage_d), floatingImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(floatingImage_d, floatingImage->data, floatingImage->nvox * sizeof(float), cudaMemcpyHostToDevice));

	//printf("uploading %s\n", floating1);
	//deformationFieldImage_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&deformationFieldImage_d), deformationFieldImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(deformationFieldImage_d, deformationFieldImage->data, deformationFieldImage->nvox * sizeof(float), cudaMemcpyHostToDevice));

	//printf("uploading %s\n", floating2);
	//warpedImage_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&warpedImage_d), warpedImage->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(warpedImage_d, warpedImage->data, warpedImage->nvox * sizeof(float), cudaMemcpyHostToDevice));

	//printf("uploading %s\n", floating3);
	//mask_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&mask_d), targetVoxelNumber * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(mask_d, mask, targetVoxelNumber * sizeof(int), cudaMemcpyHostToDevice));

	//mask_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&dtiIndeces_d), 6 * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(dtiIndeces_d, dtiIndeces, 6 * sizeof(int), cudaMemcpyHostToDevice));

	//printf("uploading %s\n", floating4);
	//sourceIJKMatrix_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&sourceIJKMatrix_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(sourceIJKMatrix_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));

	//sourceIJKMatrix_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&jacMat_d), numMats * 9 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(jacMat_d, jacMat_h, numMats * 9 * sizeof(float), cudaMemcpyHostToDevice));

	// The DTI are logged
	reg_dti_resampling_preprocessing<float>(floatingImage, &originalFloatingData, dtiIndeces);
	//reg_dti_resampling_preprocessing<float> << <mygrid, myblocks >> >(floatingImage_d, dtiIndeces, fi_xyz);

	//printf("kernel %s\n", floating);
	if (interp == 1)
		TrilinearResampleImage << <mygrid, myblocks >> >(floatingImage_d, deformationFieldImage_d, warpedImage_d, mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	else if (interp == 3)
		CubicSplineResampleImage3D << <mygrid, myblocks >> >(floatingImage_d, deformationFieldImage_d, warpedImage_d, mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	else
		NearestNeighborResampleImage << <mygrid, myblocks >> >(floatingImage_d, deformationFieldImage_d, warpedImage_d, mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	NR_CUDA_CHECK_KERNEL(mygrid, myblocks)

		//printf("copy %s\n", floating);
		NR_CUDA_SAFE_CALL(cudaMemcpy(warpedImage->data, warpedImage_d, warpedImage->nvox * sizeof(float), cudaMemcpyDeviceToHost));
	//printf("done %s\n", floating);
	// The temporary logged floating array is deleted
	if (originalFloatingData != NULL) {
		free(floatingImage->data);
		floatingImage->data = originalFloatingData;
		originalFloatingData = NULL;
	}
	// The interpolated tensors are reoriented and exponentiated
	//reg_dti_resampling_postprocessing<float> << <mygrid, myblocks >> >(warpedImage_d, NULL, mask_d, jacMat_d, dtiIndeces_d, fi_xyz, wi_tu);
	reg_dti_resampling_postprocessing<float>(warpedImage, mask, jacMat, dtiIndeces);

	cudaFree(floatingImage_d);
	cudaFree(deformationFieldImage_d);
	cudaFree(warpedImage_d);
	cudaFree(mask_d);
	cudaFree(sourceIJKMatrix_d);
	cudaFree(jacMat_d);



}

void runKernel2(nifti_image *floatingImage, nifti_image *warpedImage, int *mask, int interp, float paddingValue, int *dtiIndeces, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d,  int** mask_d) {


	long targetVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;
	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
	unsigned int maxThreads = prop.maxThreadsDim[0];
	unsigned int maxBlocks = prop.maxThreadsDim[0];
	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min1(blocks, maxBlocks);

	dim3 mygrid(blocks, 1, 1);
	dim3 myblocks(maxThreads, 1, 1);
	//printf("maxBlocks b: %d | maxThreads: %d\n", maxBlocks, maxThreads);
	//printf("blocks: %d | threads: %d\n", blocks, maxThreads);

	// The floating image data is copied in case one deal with DTI
	void *originalFloatingData = NULL;

	//number of jacobian matrices
	int numMats = 0;//needs to be transfered to a param 
	
	float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
	float* jacMat_h = (float*)malloc(9 * numMats*sizeof(float));
	
	mat44 *sourceIJKMatrix;
	if (floatingImage->sform_code > 0)
		sourceIJKMatrix = &(floatingImage->sto_ijk);
	else sourceIJKMatrix = &(floatingImage->qto_ijk);

	float* sourceIJKMatrix_d, *jacMat_d;
	int* dtiIndeces_d;
	long2 voxelNumber = make_long2(warpedImage->nx*warpedImage->ny*warpedImage->nz, floatingImage->nx*floatingImage->ny*floatingImage->nz);
	uint3 fi_xyz = make_uint3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
	uint2 wi_tu = make_uint2(warpedImage->nt, warpedImage->nu);


	mat44ToCptr(*sourceIJKMatrix, sourceIJKMatrix_h);
	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);

	char* floating = "floating";
	char* floating1 = "deformationFieldImage_d";
	char* floating2 = "warpedImage_d";
	char* floating3 = "mask_d";
	char* floating4 = "matrix";

	//mask_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&dtiIndeces_d), 6 * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(dtiIndeces_d, dtiIndeces, 6 * sizeof(int), cudaMemcpyHostToDevice));

	//sourceIJKMatrix_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&sourceIJKMatrix_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(sourceIJKMatrix_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));

	//sourceIJKMatrix_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&jacMat_d), numMats * 9 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(jacMat_d, jacMat_h, numMats * 9 * sizeof(float), cudaMemcpyHostToDevice));

	// The DTI are logged
	reg_dti_resampling_preprocessing<float>(floatingImage, &originalFloatingData, dtiIndeces);
	//reg_dti_resampling_preprocessing<float> << <mygrid, myblocks >> >(floatingImage_d, dtiIndeces, fi_xyz);

	if (interp == 1)
		TrilinearResampleImage << <mygrid, myblocks >> >(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	else if (interp == 3)
		CubicSplineResampleImage3D << <mygrid, myblocks >> >(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	else
		NearestNeighborResampleImage << <mygrid, myblocks >> >(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	//NR_CUDA_CHECK_KERNEL(mygrid, myblocks)
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

	//NR_CUDA_SAFE_CALL(cudaMemcpy(warpedImage->data, *warpedImage_d, warpedImage->nvox * sizeof(float), cudaMemcpyDeviceToHost));
	// The temporary logged floating array is deleted
	if (originalFloatingData != NULL) {
		free(floatingImage->data);
		floatingImage->data = originalFloatingData;
		originalFloatingData = NULL;
	}
	// The interpolated tensors are reoriented and exponentiated
	//reg_dti_resampling_postprocessing<float> << <mygrid, myblocks >> >(warpedImage_d, NULL, mask_d, jacMat_d, dtiIndeces_d, fi_xyz, wi_tu);
	reg_dti_resampling_postprocessing<float>(warpedImage, mask, jacMat, dtiIndeces);

	cudaFree(sourceIJKMatrix_d);
	cudaFree(jacMat_d);
	cudaFree(dtiIndeces_d);
	
	//free(originalFloatingData);
	free(sourceIJKMatrix_h);
	free(jacMat_h);


}

void launchBlockMatching(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask){

	float *targetImageArray_d;
	float *resultImageArray_d;
	float *targetPosition_d;
	float *resultPosition_d;
	int *activeBlock_d, *mask_d;

	//targetImageArray_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&targetImageArray_d), target->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(targetImageArray_d, target->data, target->nvox * sizeof(float), cudaMemcpyHostToDevice));

	//resultImageArray_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&resultImageArray_d), result->nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(resultImageArray_d, result->data, result->nvox * sizeof(float), cudaMemcpyHostToDevice));

	//targetPosition_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&targetPosition_d), params->activeBlockNumber * 3 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(targetPosition_d, params->targetPosition, params->activeBlockNumber * 3 * sizeof(float), cudaMemcpyHostToDevice));

	//resultPosition_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&resultPosition_d), params->activeBlockNumber * 3 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(resultPosition_d, params->resultPosition, params->activeBlockNumber * 3 * sizeof(float), cudaMemcpyHostToDevice));

	//activeBlock_d

	int3 bDim = make_int3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
	const int numBlocks = bDim.x*bDim.y*bDim.z;
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&activeBlock_d), numBlocks  * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(activeBlock_d, params->activeBlock, numBlocks  * sizeof(int), cudaMemcpyHostToDevice));
	
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&mask_d), target->nvox * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(mask_d, mask, target->nvox * sizeof(int), cudaMemcpyHostToDevice));

	block_matching_method_gpu(target, result, params, &targetImageArray_d, &resultImageArray_d, &targetPosition_d, &resultPosition_d, &activeBlock_d, &mask_d);

	
	//cudaDeviceReset();
	/*cudaFree(targetImageArray_d);
	cudaFree(resultImageArray_d);
	cudaFree(targetPosition_d);
	cudaFree(resultPosition_d);
	cudaFree(activeBlock_d);*/
}

void identityConst(){
	float* mat_h = (float*)malloc(16*sizeof(float));
	mat44* final;
	// Set the current transformation to identity
	final->m[0][0] = final->m[1][1] = final->m[2][2] = final->m[3][3] = 1.0f;
	final->m[0][1] = final->m[0][2] = final->m[0][3] = 0.0f;
	final->m[1][0] = final->m[1][2] = final->m[1][3] = 0.0f;
	final->m[2][0] = final->m[2][1] = final->m[2][3] = 0.0f;
	final->m[3][0] = final->m[3][1] = final->m[3][2] = 0.0f;
	mat44ToCptr(*final, mat_h);
	cudaMemcpyToSymbol(cIdentity, &mat_h, 16*sizeof(float));
}

void launchBlockMatching2(nifti_image * target,  _reg_blockMatchingParam *params, float **targetImageArray_d,
	float **resultImageArray_d,
	float **targetPosition_d,
	float **resultPosition_d,
	int **activeBlock_d, int **mask_d){



	block_matching_method_gpu3(target, params, targetImageArray_d, resultImageArray_d, targetPosition_d, resultPosition_d, activeBlock_d, mask_d);
}



void launchOptimizeAffine(_reg_blockMatchingParam* params, mat44* final, bool affine){

	//

	////    const unsigned num_points = params->activeBlockNumber;
	//const unsigned num_points = params->definedActiveBlock;
	//unsigned long num_equations = num_points * 3;
	//std::multimap<double, _reg_sorted_point3D> queue;
	//std::vector<_reg_sorted_point3D> top_points;
	//double distance = 0.0;
	//double lastDistance = std::numeric_limits<double>::max();
	//unsigned long i;

	//float* a_h, *w_h, *v_h, *r_h = (float*)malloc(num_equations*12*sizeof(float));
	//float* b_h = (float*)malloc(num_equations * sizeof(float));


	//// massive left hand side matrix
	//float ** a = new float *[num_equations];
	//for (unsigned k = 0; k < num_equations; ++k)
	//{
	//	a[k] = new float[12]; // full affine
	//}

	//// The array of singular values returned by svd
	//float *w = new float[12];

	//// v will be n x n
	//float **v = new float *[12];
	//for (unsigned k = 0; k < 12; ++k)
	//{
	//	v[k] = new float[12];
	//}

	//// Allocate memory for pseudoinverse
	//float **r = new float *[12];
	//for (unsigned k = 0; k < 12; ++k)
	//{
	//	r[k] = new float[num_equations];
	//}

	//// Allocate memory for RHS vector
	//float *b = new float[num_equations];

	//// The initial vector with all the input points
	//for (unsigned j = 0; j < num_points * 3; j += 3)
	//{
	//	top_points.push_back(_reg_sorted_point3D(&(params->targetPosition[j]), &(params->resultPosition[j]), 0.0f));
	//}

	//// estimate the optimal transformation while considering all the points
	//estimate_affine_transformation3D(top_points, final, a, w, v, r, b);

	//// Delete a, b and r. w and v will not change size in subsequent svd operations.
	//for (unsigned int k = 0; k < num_equations; ++k)
	//{
	//	delete[] a[k];
	//}
	//delete[] a;
	//delete[] b;

	//for (unsigned k = 0; k < 12; ++k)
	//{
	//	delete[] r[k];
	//}
	//delete[] r;


	//// The LS in the iterations is done on subsample of the input data
	//float * newResultPosition = new float[num_points * 3];
	//const unsigned long num_to_keep = (unsigned long)(num_points * (params->percent_to_keep / 100.0f));
	//num_equations = num_to_keep * 3;

	//// The LHS matrix
	//a = new float *[num_equations];
	//for (unsigned k = 0; k < num_equations; ++k)
	//{
	//	a[k] = new float[12]; // full affine
	//}

	//// Allocate memory for pseudoinverse
	//r = new float *[12];
	//for (unsigned k = 0; k < 12; ++k)
	//{
	//	r[k] = new float[num_equations];
	//}

	//// Allocate memory for RHS vector
	//b = new float[num_equations];
	//mat44 lastTransformation;
	//memset(&lastTransformation, 0, sizeof(mat44));

	//for (unsigned count = 0; count < MAX_ITERATIONS; ++count)
	//{
	//	// Transform the points in the target
	//	for (unsigned j = 0; j < num_points * 3; j += 3)
	//	{
	//		reg_mat44_mul(final, &(params->targetPosition[j]), &newResultPosition[j]);
	//	}

	//	queue = std::multimap<double, _reg_sorted_point3D>();
	//	for (unsigned j = 0; j < num_points * 3; j += 3)
	//	{
	//		distance = get_square_distance(&newResultPosition[j], &(params->resultPosition[j]));
	//		queue.insert(std::pair<double, _reg_sorted_point3D>(distance, _reg_sorted_point3D(&(params->targetPosition[j]),
	//			&(params->resultPosition[j]), distance)));
	//	}

	//	distance = 0.0;
	//	i = 0;
	//	top_points.clear();

	//	for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin();
	//		it != queue.end(); ++it, ++i)
	//	{
	//		if (i >= num_to_keep) break;
	//		top_points.push_back((*it).second);
	//		distance += (*it).first;
	//	}

	//	// If the change is not substantial or we are getting worst, we return
	//	if ((distance >= lastDistance) || (lastDistance - distance) < TOLERANCE)
	//	{
	//		// restore the last transformation
	//		copy_transformation_4x4(lastTransformation, *(final));
	//		break;
	//	}
	//	lastDistance = distance;
	//	copy_transformation_4x4(*(final), lastTransformation);
	//	estimate_affine_transformation3D(top_points, final, a, w, v, r, b);
	//}
	//delete[] newResultPosition;
	//delete[] b;
	//for (unsigned k = 0; k < 12; ++k)
	//{
	//	delete[] r[k];
	//}
	//delete[] r;

	//// free the memory
	//for (unsigned int k = 0; k < num_equations; ++k)
	//{
	//	delete[] a[k];
	//}
	//delete[] a;

	//delete[] w;
	//for (int k = 0; k < 12; ++k)
	//{
	//	delete[] v[k];
	//}
	//delete[] v;

}
void launchOptimizeRigid(_reg_blockMatchingParam* params, mat44* transformation_matrix, bool affine){}
