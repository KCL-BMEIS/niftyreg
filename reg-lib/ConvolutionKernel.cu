// System includes
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include"_reg_blocksize_gpu.h"
#include"_reg_maths.h"
#include "cudaKernelFuncs.h"
#include "_reg_common_gpu.h"

#include"_reg_tools.h"
#include"_reg_ReadWriteImage.h"
#include "cuda_profiler_api.h"



__device__ __inline__ void getPosition(float* position, float* matrix, float* voxel, const unsigned int idx) {
	position[idx] =
		matrix[idx * 4 + 0] * voxel[0] +
		matrix[idx * 4 + 1] * voxel[1] +
		matrix[idx * 4 + 2] * voxel[2] +
		matrix[idx * 4 + 3];
}

__global__ void affineKernel(float* transformationMatrix,  float* defField, int* mask, const uint3 params, const unsigned long voxelNumber, const bool composition) {

	float *deformationFieldPtrX =  defField ;
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber]; 

	float voxel[3], position[3];

	
	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned long index = x + y*params.x + z * params.x * params.y;
	if( z<params.z && y<params.y && x<params.x &&  mask[index] >= 0 ) {

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
	if( threadIdx.x == 0 ) {
		printf("hi from %d-%d \n", blockIdx.x, threadIdx.x);
		const unsigned long voxelNumber = image->dim[1]* image->dim[2]* image->dim[3];
		DTYPE *imagePtr = static_cast<DTYPE *>( image->data );
		int imageDim[3] = { image->dim[1], image->dim[2], image->dim[3] };


		// Loop over the dimension higher than 3
		for( int t = 0; t<image->dim[4]*image->dim[5]; t++ ) {
			if( timePoint[t] ) {
				DTYPE *intensityPtr = &imagePtr[t * voxelNumber];

				for( unsigned long index = 0; index<voxelNumber; index++ ) {
					densityPtr[index] = ( intensityPtr[index] == intensityPtr[index] ) ? 1 : 0;
					densityPtr[index] *= ( mask[index] >= 0 ) ? 1 : 0;
					nanImagePtr[index] = static_cast<bool>( densityPtr[index] );
					if( nanImagePtr[index] == 0 )
						intensityPtr[index] = static_cast<DTYPE>( 0 );
				}
				// Loop over the x, y and z dimensions
				for( int n = 0; n<3; n++ ) {
					if( axis[n] && image->dim[n]>1 ) {
						double temp;
						if( size[t]>0 ) temp = size[t] / image->pixdim[n + 1]; // mm to voxel
						else temp = fabs(size[t]); // voxel based if negative value
						int radius;
						// Define the kernel size
						if( kernelType == 2 ) {
							// Mean filtering
							radius = static_cast<int>( temp );
						}
						else if( kernelType == 1 ) {
							// Cubic Spline kernel
							radius = static_cast<int>( temp*2.0f );
						}
						else {
							// Gaussian kernel
							radius = static_cast<int>( temp*3.0f );
						}
						if( radius>0 ) {
							// Allocate the kernel
							float kernel[2048];
							double kernelSum = 0;
							// Fill the kernel
							if( kernelType == 1 ) {
								// Compute the Cubic Spline kernel
								for( int i = -radius; i <= radius; i++ ) {
									// temp contains the kernel node spacing
									double relative = (double)( fabs((double)(double)i / (double)temp) );
									if( relative<1.0 ) kernel[i + radius] = (float)( 2.0 / 3.0 - relative*relative + 0.5*relative*relative*relative );
									else if( relative<2.0 ) kernel[i + radius] = (float)( -( relative - 2.0 )*( relative - 2.0 )*( relative - 2.0 ) / 6.0 );
									else kernel[i + radius] = 0;
									kernelSum += kernel[i + radius];
								}
							}
							// No kernel is required for the mean filtering
							else if( kernelType != 2 ) {
								// Compute the Gaussian kernel
								for( int i = -radius; i <= radius; i++ ) {
									// 2.506... = sqrt(2*pi)
									// temp contains the sigma in voxel
									kernel[radius + i] = static_cast<float>( exp(-(double)( i*i ) / ( 2.0*reg_pow2(temp) )) /
																			( temp*2.506628274631 ) );
									kernelSum += kernel[radius + i];
								}
							}
							// No need for kernel normalisation as this is handle by the density function
							int planeNumber, planeIndex, lineOffset;
							int lineIndex, shiftPre, shiftPst, k;
							switch( n ) {
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
							for( planeIndex = 0; planeIndex<planeNumber; ++planeIndex ) {

								switch( n ) {
								case 0:
									realIndex = planeIndex * imageDim[0];
									break;
								case 1:
									realIndex = ( planeIndex / imageDim[0] ) *
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
								for( lineIndex = 0; lineIndex<imageDim[n]; ++lineIndex ) {
									bufferIntensity[lineIndex] = *currentIntensityPtr;
									bufferDensity[lineIndex] = *currentDensityPtr;
									currentIntensityPtr += lineOffset;
									currentDensityPtr += lineOffset;
								}
								if( kernelSum>0 ) {
									// Perform the kernel convolution along 1 line
									for( lineIndex = 0; lineIndex<imageDim[n]; ++lineIndex ) {
										// Define the kernel boundaries
										shiftPre = lineIndex - radius;
										shiftPst = lineIndex + radius + 1;
										if( shiftPre<0 ) {
											kernelPtr = &kernel[-shiftPre];
											shiftPre = 0;
										}
										else kernelPtr = &kernel[0];
										if( shiftPst>imageDim[n] ) shiftPst = imageDim[n];
										// Set the current values to zero
										intensitySum = 0;
										densitySum = 0;
										// Increment the current value by performing the weighted sum
										for( k = shiftPre; k<shiftPst; ++k ) {
											kernelValue = *kernelPtr++;
											intensitySum += kernelValue * bufferIntensity[k];
											densitySum += kernelValue * bufferDensity[k];
										}
										// Store the computed value inplace
										intensityPtr[realIndex] = static_cast<DTYPE>( intensitySum );
										densityPtr[realIndex] = static_cast<float>( densitySum );
										realIndex += lineOffset;
									} // line convolution
								} // kernel type
								else {
									for( lineIndex = 1; lineIndex<imageDim[n]; ++lineIndex ) {
										bufferIntensity[lineIndex] += bufferIntensity[lineIndex - 1];
										bufferDensity[lineIndex] += bufferDensity[lineIndex - 1];
									}
									shiftPre = -radius - 1;
									shiftPst = radius;
									for( lineIndex = 0; lineIndex<imageDim[n]; ++lineIndex, ++shiftPre, ++shiftPst ) {
										if( shiftPre>-1 ) {
											if( shiftPst<imageDim[n] ) {
												bufferIntensitycur = (DTYPE)( bufferIntensity[shiftPre] - bufferIntensity[shiftPst] );
												bufferDensitycur = (DTYPE)( bufferDensity[shiftPre] - bufferDensity[shiftPst] );
											}
											else {
												bufferIntensitycur = (DTYPE)( bufferIntensity[shiftPre] - bufferIntensity[imageDim[n] - 1] );
												bufferDensitycur = (DTYPE)( bufferDensity[shiftPre] - bufferDensity[imageDim[n] - 1] );
											}
										}
										else {
											if( shiftPst<imageDim[n] ) {
												bufferIntensitycur = (DTYPE)( -bufferIntensity[shiftPst] );
												bufferDensitycur = (DTYPE)( -bufferDensity[shiftPst] );
											}
											else {
												bufferIntensitycur = (DTYPE)( 0 );
												bufferDensitycur = (DTYPE)( 0 );
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
				for( unsigned long index = 0; index<voxelNumber; ++index ) {
					if( nanImagePtr[index] != 0 )
						intensityPtr[index] = static_cast<DTYPE>( (float)intensityPtr[index] / densityPtr[index] );
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
	

	if( image->nx>2048 || image->ny>2048 || image->nz>2048 ) {
		reg_print_fct_error("reg_tools_kernelConvolution_core");
		reg_print_msg_error("This function does not support images with dimension > 2048");
		reg_exit(1);
	}

	if( image->nt <= 0 ) image->nt = image->dim[4] = 1;
	if( image->nu <= 0 ) image->nu = image->dim[5] = 1;

	

	
	/*densityPtr[4] = 8.8f;
	std::cout << "test float: " << densityPtr[4] << std::endl;*/


	if( axis == NULL ) {
		// All axis are smoothed by default
		for( int i = 0; i<3; i++ ) axisToSmooth[i] = true;
	}
	else for( int i = 0; i<3; i++ ) axisToSmooth[i] = axis[i];

	if( timePoint == NULL ) {
		// All time points are considered as active
		for( int i = 0; i<image->nt*image->nu; i++ ) activeTimePoint[i] = true;
	}
	else for( int i = 0; i<image->nt*image->nu; i++ ) activeTimePoint[i] = timePoint[i];

	int *currentMask = NULL;
	if( mask == NULL ) {
		currentMask = (int *)calloc(image->nx*image->ny*image->nz, sizeof(int));
	}
	else currentMask = mask;

	/*cudaCommon_allocateNiftiToDevice<float>(&image_d, dim);
	cudaCommon_transferNiftiToNiftiOnDevice1<float>(&image_d, image);*/

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( sigma_d ), image->dim[4] * image->dim[5] * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(sigma_d, sigma, image->dim[4] * image->dim[5] * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( mask_d ), voxelNumber * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(mask_d, currentMask, voxelNumber * sizeof(int), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( timePoint_d ), image->dim[4] * image->dim[5] * sizeof(bool)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(timePoint_d, timePoint, image->dim[4] * image->dim[5] * sizeof(bool), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( axis_d ), 3 * sizeof(bool)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(axis_d, axis, 3 * sizeof(bool), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc(&nanImagePtr, voxelNumber*sizeof(bool)));
	NR_CUDA_SAFE_CALL(cudaMalloc(&densityPtr, voxelNumber*sizeof(float)));

	switch( image->datatype ) {
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
		convolutionKernel <float> <<<1, 1 >>>( image_d, densityPtr, nanImagePtr, sigma_d, kernelType, mask_d, timePoint_d, axis_d );
		//NR_CUDA_CHECK_KERNEL(1, 1)
		break;
	case NIFTI_TYPE_FLOAT64:
		//convolutionKernel <double> << <1, 1 >> >( image, densityPtr, nanImagePtr, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth );
		break;
	default:
		fprintf(stderr, "[NiftyReg ERROR] reg_gaussianSmoothing\tThe image data type is not supported\n");
		reg_exit(1);
	}

	if( mask == NULL ) free(currentMask);
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
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField, bool compose , int *mask ) {

	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ( ( deformationField->nx % xThreads ) == 0 ) ? ( deformationField->nx / xThreads ) : ( deformationField->nx / xThreads ) + 1;
	const unsigned int yBlocks = ( ( deformationField->ny % yThreads ) == 0 ) ? ( deformationField->ny / yThreads ) : ( deformationField->ny / yThreads ) + 1;
	const unsigned int zBlocks = ( ( deformationField->nz % zThreads ) == 0 ) ? ( deformationField->nz / zThreads ) : ( deformationField->nz / zThreads ) + 1;


	dim3 G1_b(xBlocks, yBlocks, zBlocks);
	dim3 B1_b(xThreads, yThreads, zThreads);
	

	int *tempMask = mask;
	if( mask == NULL ) {
		tempMask = (int *)calloc(deformationField->nx*
								 deformationField->ny*
								 deformationField->nz,
								 sizeof(int));
	}

	const mat44 *targetMatrix = ( deformationField->sform_code>0 ) ? &( deformationField->sto_xyz ) : &( deformationField->qto_xyz );
	mat44 transformationMatrix = ( compose == true ) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);

	float* trans = (float *)malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	nifti_params params_d = getParams(*deformationField);
	float *trans_d, *def_d;
	int* mask_d;

	

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( &trans_d ), 16 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( &def_d ), params_d.nvox * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(def_d, deformationField->data, params_d.nvox * sizeof(float), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)( &mask_d ), params_d.nxyz * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpy(mask_d, tempMask, params_d.nxyz * sizeof(int), cudaMemcpyHostToDevice));



	uint3 pms_d = make_uint3(params_d.nx, params_d.ny, params_d.nz);
	affineKernel << <G1_b, B1_b >> >( trans_d, def_d, mask_d, pms_d, params_d.nxyz, compose );
	NR_CUDA_CHECK_KERNEL(G1_b, B1_b)
	
	NR_CUDA_SAFE_CALL(cudaMemcpy(deformationField->data, def_d, params_d.nvox * sizeof(float), cudaMemcpyDeviceToHost));

	if( mask == NULL )
		free(tempMask);

	cudaFree(trans_d);
	cudaFree(def_d);
	cudaFree(mask_d);

}




