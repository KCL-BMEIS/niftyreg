
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
#include "_reg_blockMatching_gpu.h"
#include "_reg_blockMatching.h"


unsigned int min1(unsigned int a, unsigned int b) {
	return (a < b) ? a : b;
}


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
	return (int)(floor(a));
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


__device__ __inline__ void getPosition(float* position, float* matrix, float* voxel, const unsigned int idx) {
	position[idx] =
		matrix[idx * 4 + 0] * voxel[0] +
		matrix[idx * 4 + 1] * voxel[1] +
		matrix[idx * 4 + 2] * voxel[2] +
		matrix[idx * 4 + 3];
}

__device__ __inline__ double getPosition( float* matrix, double* voxel, const unsigned int idx) {
//	if ( voxel[0] == 74.0f && voxel[1] == 0.0f && voxel[2]==0.0f && idx ==0) printf("(%f-%f-%f) (%f-%f-%f-%f)\n",voxel[0],voxel[1], voxel[2], matrix[idx * 4 + 0], matrix[idx * 4 + 1], matrix[idx * 4 + 2], matrix[idx * 4 + 3]);
	return
		(double)matrix[idx * 4 + 0] * voxel[0] +
		(double)matrix[idx * 4 + 1] * voxel[1] +
		(double)matrix[idx * 4 + 2] * voxel[2] +
		(double)matrix[idx * 4 + 3];
}


__global__ void CubicSplineResampleImage3D(float *floatingImage, float *deformationField, float *warpedImage, int *mask, /*mat44*/float* sourceIJKMatrix, ulong2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {
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
__global__ void NearestNeighborResampleImage(float *floatingImage, float *deformationField, float *warpedImage, int *mask, /*mat44*/float* sourceIJKMatrix, ulong2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {

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


__global__ void TrilinearResampleImage(float *floatingImage, float *deformationField, float *warpedImage, int *mask, /*mat44*/float* sourceIJKMatrix, ulong2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {

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

	bool flag = (threadIdx.x==839 || threadIdx.x==903)&& blockIdx.x==6;//temp code

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

									xTempNewValue = 0.0f;
									for (a = 0; a<2; a++) {
										X = previous[0] + a;
										if (X>-1 && X < fi_xyz.x) {
											xyzPointer = &zPointer[Y*fi_xyz.x + X];
											xTempNewValue += *xyzPointer * xBasis[a];
										} // X
										else xTempNewValue += paddingValue * xBasis[a];
//										xyzPointer++;
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
						yTempNewValue = 0.0f;
						for (b = 0; b < 2; b++) {
							Y = previous[1] + b;
							xyzPointer = &zPointer[Y*fi_xyz.x + previous[0]];
							xTempNewValue = 0.0f;
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



__global__ void affineKernel(float* transformationMatrix, float* defField, int* mask, const uint3 dims, const unsigned long voxelNumber, const bool composition) {

	float *deformationFieldPtrX = defField;
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

	double voxel[3];


	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	const unsigned long index = x + y*dims.x + z * dims.x * dims.y;


	if (z < dims.z && y < dims.y && x < dims.x &&  mask[index] >= 0) {

		voxel[0] = composition ? deformationFieldPtrX[index] : (double)x;
		voxel[1] = composition ? deformationFieldPtrY[index] : (double)y;
		voxel[2] = composition ? deformationFieldPtrZ[index] : (double)z;

		/* the deformation field (real coordinates) is stored */
//		if (index == 165 ) printf("x: %f | val: %f\n",voxel[0], getPosition( transformationMatrix, voxel, 0));
		deformationFieldPtrX[index] = getPosition( transformationMatrix, voxel, 0);
		deformationFieldPtrY[index] = getPosition( transformationMatrix, voxel, 1);
		deformationFieldPtrZ[index] = getPosition( transformationMatrix, voxel, 2);

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

//++++++++++++++++++++++++++++++++++++++++++ wraper funcs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void launchConvolution(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis) {
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


void launchAffine(mat44 *affineTransformation, nifti_image *deformationField, float** def_d, int** mask_d, bool compose) {

	float* trans = (float *)malloc(16 * sizeof(float));
	float *trans_d;

	const mat44 *targetMatrix = (deformationField->sform_code > 0) ? &(deformationField->sto_xyz) : &(deformationField->qto_xyz);
	mat44 transformationMatrix = (compose == true) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);
	mat44ToCptr(transformationMatrix, trans);

	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			if(transformationMatrix.m[i][j] != trans[4*i + j]) printf("Probs\n");


	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((deformationField->nx % xThreads) == 0) ? (deformationField->nx / xThreads) : (deformationField->nx / xThreads) + 1;
	const unsigned int yBlocks = ((deformationField->ny % yThreads) == 0) ? (deformationField->ny / yThreads) : (deformationField->ny / yThreads) + 1;
	const unsigned int zBlocks = ((deformationField->nz % zThreads) == 0) ? (deformationField->nz / zThreads) : (deformationField->nz / zThreads) + 1;

	dim3 G1_b(xBlocks, yBlocks, zBlocks);
	dim3 B1_b(xThreads, yThreads, zThreads);

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&trans_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));

	uint3 dims_d = make_uint3(deformationField->nx, deformationField->ny, deformationField->nz);
	affineKernel << <G1_b, B1_b >> >(trans_d, *def_d, *mask_d, dims_d, deformationField->nx* deformationField->ny* deformationField->nz, compose);
	//NR_CUDA_CHECK_KERNEL(G1_b, B1_b)
	NR_CUDA_SAFE_CALL(cudaThreadSynchronize());

	cudaFree(trans_d);
	free(trans);

}



void launchResample(nifti_image *floatingImage, nifti_image *warpedImage, int *mask, int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat, float** floatingImage_d,  float** warpedImage_d, float** deformationFieldImage_d, int** mask_d) {

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



void runKernel2(nifti_image *floatingImage, nifti_image *warpedImage, int *mask, int interp, float paddingValue, int *dtiIndeces, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d,  int** mask_d) {


	long targetVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;

	//the below lines need to be moved to cu common
	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
	unsigned int maxThreads = prop.maxThreadsDim[0];
	unsigned int maxBlocks = prop.maxThreadsDim[0];
	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min1(blocks, maxBlocks);

	dim3 mygrid(blocks, 1, 1);
	dim3 myblocks(maxThreads, 1, 1);



	//number of jacobian matrices
	int numMats = 0;//needs to be transfered to a param

	float* sourceIJKMatrix_d, *jacMat_d;
	float *sourceIJKMatrix_h = (float*)malloc(16 * sizeof(float));
	float* jacMat_h = (float*)malloc(9 * numMats*sizeof(float));

	int* dtiIndeces_d;

	mat44 *sourceIJKMatrix = (floatingImage->sform_code > 0)? &(floatingImage->sto_ijk):sourceIJKMatrix = &(floatingImage->qto_ijk);
	mat44ToCptr(*sourceIJKMatrix, sourceIJKMatrix_h);

	ulong2 voxelNumber = make_ulong2(warpedImage->nx*warpedImage->ny*warpedImage->nz, floatingImage->nx*floatingImage->ny*floatingImage->nz);
	uint3 fi_xyz = make_uint3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
	uint2 wi_tu = make_uint2(warpedImage->nt, warpedImage->nu);

	if (numMats) mat33ToCptr(jacMat, jacMat_h, numMats);

	//dti indeces
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&dtiIndeces_d), 6 * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(dtiIndeces_d, dtiIndeces, 6 * sizeof(int), cudaMemcpyHostToDevice));

	//sourceIJKMatrix_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&sourceIJKMatrix_d), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(sourceIJKMatrix_d, sourceIJKMatrix_h, 16 * sizeof(float), cudaMemcpyHostToDevice));

	//sourceIJKMatrix_d
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)(&jacMat_d), numMats * 9 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(jacMat_d, jacMat_h, numMats * 9 * sizeof(float), cudaMemcpyHostToDevice));


	// The floating image data is copied in case one deal with DTI
	void *originalFloatingData = NULL;
	// The DTI are logged
	reg_dti_resampling_preprocessing<float>(floatingImage, &originalFloatingData, dtiIndeces);//need to either write it in cuda or do the transfers
	//reg_dti_resampling_preprocessing<float> << <mygrid, myblocks >> >(floatingImage_d, dtiIndeces, fi_xyz);

	if (interp == 1)
		TrilinearResampleImage << <mygrid, myblocks >> >(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	else if (interp == 3)
		CubicSplineResampleImage3D << <mygrid, myblocks >> >(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
	else
		NearestNeighborResampleImage << <mygrid, myblocks >> >(*floatingImage_d, *deformationFieldImage_d, *warpedImage_d, *mask_d, sourceIJKMatrix_d, voxelNumber, fi_xyz, wi_tu, paddingValue);
//	NR_CUDA_CHECK_KERNEL(mygrid, myblocks)
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
	reg_dti_resampling_postprocessing<float>(warpedImage, mask, jacMat, dtiIndeces);//need to either write it in cuda or do the transfers

	cudaFree(sourceIJKMatrix_d);
	cudaFree(jacMat_d);
	cudaFree(dtiIndeces_d);

	//free(originalFloatingData);
	free(sourceIJKMatrix_h);
	free(jacMat_h);


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

void launchBlockMatching(nifti_image * target,  _reg_blockMatchingParam *params, float **targetImageArray_d,float **resultImageArray_d,float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d){

	block_matching_method_gpu3(target, params, targetImageArray_d, resultImageArray_d, targetPosition_d, resultPosition_d, activeBlock_d, mask_d);
}


