#include "CPUKernels.h"
#include"_reg_resampling.h"

//------------------------------------------------------------------------------------------------------------------------
//..................CPUAffineDeformationFieldKernel---------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUAffineDeformationFieldKernel::initialize(nifti_image *CurrentReference, nifti_image **deformationFieldImage, const size_t dataSize) {

	if( CurrentReference == NULL ) {
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference image is not defined. Exit.\n");
		reg_exit(1);
	}
	clear(*deformationFieldImage);
	*deformationFieldImage = nifti_copy_nim_info(CurrentReference);
	( *deformationFieldImage )->dim[0] = ( *deformationFieldImage )->ndim = 5;
	( *deformationFieldImage )->dim[4] = ( *deformationFieldImage )->nt = 1;
	( *deformationFieldImage )->pixdim[4] = ( *deformationFieldImage )->dt = 1.0;
	if( CurrentReference->nz == 1 )
		( *deformationFieldImage )->dim[5] = ( *deformationFieldImage )->nu = 2;
	else ( *deformationFieldImage )->dim[5] = ( *deformationFieldImage )->nu = 3;
	( *deformationFieldImage )->pixdim[5] = ( *deformationFieldImage )->du = 1.0;
	( *deformationFieldImage )->dim[6] = ( *deformationFieldImage )->nv = 1;
	( *deformationFieldImage )->pixdim[6] = ( *deformationFieldImage )->dv = 1.0;
	( *deformationFieldImage )->dim[7] = ( *deformationFieldImage )->nw = 1;
	( *deformationFieldImage )->pixdim[7] = ( *deformationFieldImage )->dw = 1.0;
	( *deformationFieldImage )->nvox = (size_t)( *deformationFieldImage )->nx *
		(size_t)( *deformationFieldImage )->ny *
		(size_t)( *deformationFieldImage )->nz *
		(size_t)( *deformationFieldImage )->nt *
		(size_t)( *deformationFieldImage )->nu;
	( *deformationFieldImage )->nbyper = dataSize;
	if( dataSize == 4 )
		( *deformationFieldImage )->datatype = NIFTI_TYPE_FLOAT32;
	else if( dataSize == 8 )
		( *deformationFieldImage )->datatype = NIFTI_TYPE_FLOAT64;
	else {
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Only float or double are expected for the deformation field. Exit.\n");
		reg_exit(1);
	}
	( *deformationFieldImage )->scl_slope = 1.f;
	( *deformationFieldImage )->scl_inter = 0.f;
	( *deformationFieldImage )->data = (void *)calloc(( *deformationFieldImage )->nvox, ( *deformationFieldImage )->nbyper);
	return;
}
//------------------------------------------------------------------------------------------------------------------------

template<class FieldTYPE>
void CPUAffineDeformationFieldKernel::runKernel3D(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool composition, int *mask) {

	size_t voxelNumber = deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz;
	FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>( deformationFieldImage->data );
	FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
	FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

	const mat44 *targetMatrix = ( deformationFieldImage->sform_code>0 ) ? &( deformationFieldImage->sto_xyz ) : &( deformationFieldImage->qto_xyz );
	mat44 transformationMatrix = ( composition == true ) ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);

	float voxel[3], position[3];
	int x, y, z;
	//size_t index;
#if defined (NDEBUG) && defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(deformationFieldImage, transformationMatrix, deformationFieldPtrX, \
		  deformationFieldPtrY, deformationFieldPtrZ, mask, composition) \
   private(voxel, position, x, y, z, index)
#endif
	for( z = 0; z<deformationFieldImage->nz; z++ ) {
		//index = z*deformationFieldImage->nx*deformationFieldImage->ny;
		voxel[2] = (float)z;
		for( y = 0; y<deformationFieldImage->ny; y++ ) {
			voxel[1] = (float)y;
			for( x = 0; x<deformationFieldImage->nx; x++ ) {
				const unsigned long index = x + y*deformationFieldImage->nx + z * deformationFieldImage->nx * deformationFieldImage->ny;
				voxel[0] = (float)x;
				if( mask[index] >= 0 ) {
					if( composition == true ) {
						voxel[0] = deformationFieldPtrX[index];
						voxel[1] = deformationFieldPtrY[index];
						voxel[2] = deformationFieldPtrZ[index];
					}
					position[0] =
						transformationMatrix.m[0][0] * voxel[0] +
						transformationMatrix.m[0][1] * voxel[1] +
						transformationMatrix.m[0][2] * voxel[2] +
						transformationMatrix.m[0][3];
					position[1] =
						transformationMatrix.m[1][0] * voxel[0] +
						transformationMatrix.m[1][1] * voxel[1] +
						transformationMatrix.m[1][2] * voxel[2] +
						transformationMatrix.m[1][3];
					position[2] =
						transformationMatrix.m[2][0] * voxel[0] +
						transformationMatrix.m[2][1] * voxel[1] +
						transformationMatrix.m[2][2] * voxel[2] +
						transformationMatrix.m[2][3];

					/* the deformation field (real coordinates) is stored */
					deformationFieldPtrX[index] = position[0];
					deformationFieldPtrY[index] = position[1];
					deformationFieldPtrZ[index] = position[2];
				}
				//index++;
			}
		}
	}

}

template <class FieldTYPE>
void CPUAffineDeformationFieldKernel::runKernel2D(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool compose, int *mask) {
	FieldTYPE *deformationFieldPtr = static_cast<FieldTYPE *>( deformationFieldImage->data );

	size_t deformationFieldIndX = 0;
	size_t deformationFieldIndY = deformationFieldImage->nx*deformationFieldImage->ny;

	mat44 *targetMatrix;
	if( deformationFieldImage->sform_code>0 ) {
		targetMatrix = &( deformationFieldImage->sto_xyz );
	}
	else targetMatrix = &( deformationFieldImage->qto_xyz );

	mat44 transformationMatrix;
	if( compose == true )
		transformationMatrix = *affineTransformation;
	else transformationMatrix = reg_mat44_mul(affineTransformation, targetMatrix);

	float index[3];
	float position[3];
	index[2] = 0;
	for( int y = 0; y<deformationFieldImage->ny; y++ ) {
		index[1] = (float)y;
		for( int x = 0; x<deformationFieldImage->nx; x++ ) {
			index[0] = (float)x;

			if( compose == true ) {
				index[0] = deformationFieldPtr[deformationFieldIndX];
				index[1] = deformationFieldPtr[deformationFieldIndY];
			}
			reg_mat44_mul(&transformationMatrix, index, position);

			/* the deformation field (real coordinates) is stored */
			deformationFieldPtr[deformationFieldIndX++] = position[0];
			deformationFieldPtr[deformationFieldIndY++] = position[1];
		}
	}
}
//------------------------------------------------------------------------------------------------------------------------
void CPUAffineDeformationFieldKernel::execute(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask) {

	int *tempMask = mask;
	if( mask == NULL ) {
		tempMask = (int *)calloc(deformationField->nx*
								 deformationField->ny*
								 deformationField->nz,
								 sizeof(int));
	}

	if( deformationField->nz == 1 ) {
		switch( deformationField->datatype ) {
		case NIFTI_TYPE_FLOAT32:
			runKernel2D<float>(affineTransformation, deformationField, compose, tempMask);
			break;
		case NIFTI_TYPE_FLOAT64:
			runKernel2D<double>(affineTransformation, deformationField, compose, tempMask);
			break;
		default:
			fprintf(stderr, "[NiftyReg ERROR] reg_affine_deformationField\tThe deformation field data type is not supported\n");
			return;
		}
	}
	else {

		switch( deformationField->datatype ) {
		case NIFTI_TYPE_FLOAT32:
			runKernel3D<float>(affineTransformation, deformationField, compose, tempMask);
			break;
		case NIFTI_TYPE_FLOAT64:
			runKernel3D<double>(affineTransformation, deformationField, compose, tempMask);
			break;
		default:
			fprintf(stderr, "[NiftyReg ERROR] reg_affine_deformationField: The deformation field data type is not supported\n");
			return;
		}
	}
	if( mask == NULL )
		free(tempMask);
}
//------------------------------------------------------------------------------------------------------------------------
void::CPUAffineDeformationFieldKernel::clear(nifti_image *deformationFieldImage) {
	if( deformationFieldImage != NULL )
		nifti_image_free(deformationFieldImage);
	deformationFieldImage = NULL;
}

//------------------------------------------------------------------------------------------------------------------------
//..................END CPUAffineDeformationFieldKernel-----------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------
//..................CPUConvolutionKernel----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUConvolutionKernel::execute(nifti_image *image, float *sigma, int kernelType,  int *mask,  bool *timePoint, bool *axis) {

	if( image->nx>2048 || image->ny>2048 || image->nz>2048 ) {
		reg_print_fct_error("reg_tools_kernelConvolution_core");
		reg_print_msg_error("This function does not support images with dimension > 2048");
		reg_exit(1);
	}

	if( image->nt <= 0 ) image->nt = image->dim[4] = 1;
	if( image->nu <= 0 ) image->nu = image->dim[5] = 1;

	bool *axisToSmooth = new bool[3];
	bool *activeTimePoint = new bool[image->nt*image->nu];
	unsigned long voxelNumber = (long)image->nx*image->ny*image->nz;

	nanImagePtr = (bool *)calloc(voxelNumber, sizeof(bool));
	densityPtr = (float *)calloc(voxelNumber, sizeof(float));

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



	switch( image->datatype ) {
	case NIFTI_TYPE_UINT8:
		runKernel<unsigned char>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_INT8:
		runKernel<char>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_UINT16:
		runKernel<unsigned short>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_INT16:
		runKernel<short>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_UINT32:
		runKernel<unsigned int>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_INT32:
		runKernel<int>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_FLOAT32:
		runKernel<float>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	case NIFTI_TYPE_FLOAT64:
		runKernel<double>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
		break;
	default:
		fprintf(stderr, "[NiftyReg ERROR] reg_gaussianSmoothing\tThe image data type is not supported\n");
		reg_exit(1);
	}

	if( mask == NULL ) free(currentMask);
	delete[]axisToSmooth;
	delete[]activeTimePoint;
	free(nanImagePtr);
	free(densityPtr);
}
//------------------------------------------------------------------------------------------------------------------------
template<class DTYPE>
void CPUConvolutionKernel::runKernel(nifti_image *image, float *size, int kernelType, int *mask, bool *timePoint, bool *axis) {
	

	unsigned long voxelNumber = (long)image->nx*image->ny*image->nz;

	DTYPE *imagePtr = static_cast<DTYPE *>( image->data );
	int imageDim[3] = { image->nx, image->ny, image->nz };

	

	// Loop over the dimension higher than 3
	for( int t = 0; t<image->nt*image->nu; t++ ) {
		if( timePoint[t] ) {
			DTYPE *intensityPtr = &imagePtr[t * voxelNumber];
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
		 shared(densityPtr, intensityPtr, mask, nanImagePtr, voxelNumber) \
		 private(index)
#endif
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
#ifndef NDEBUG
						printf("[NiftyReg DEBUG] Convolution type[%i] dim[%i] tp[%i] radius[%i] kernelSum[%g]\n", kernelType, n, t, radius, kernelSum);
#endif
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

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
				  shared(imageDim, intensityPtr, densityPtr, radius, kernel, lineOffset, n, \
						 planeNumber,kernelSum) \
				  private(realIndex,currentIntensityPtr,currentDensityPtr,lineIndex,bufferIntensity, \
						  bufferDensity,shiftPre,shiftPst,kernelPtr,kernelValue,densitySum,intensitySum, \
						  k, bufferIntensitycur,bufferDensitycur, planeIndex)
#endif // _OPENMP
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
#if defined (NDEBUG) && defined (_OPENMP)
#pragma omp parallel for default(none) \
		 shared(voxelNumber, intensityPtr, densityPtr, nanImagePtr) \
		 private(index)
#endif
			for( unsigned long index = 0; index<voxelNumber; ++index ) {
				if( nanImagePtr[index] != 0 )
					intensityPtr[index] = static_cast<DTYPE>( (float)intensityPtr[index] / densityPtr[index] );
				else intensityPtr[index] = std::numeric_limits<DTYPE>::quiet_NaN();
			}
		} // check if the time point is active
	} // loop over the time points
	
}
//------------------------------------------------------------------------------------------------------------------------
//..................END CPUConvolutionKernel------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
void CPUBlockMatchingKernel::execute(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) {

	if( target->datatype != result->datatype ) {
		reg_print_fct_error("block_matching_method");
		reg_print_msg_error("Both input images are expected to be of the same type");
	}
	if( target->nz == 1 ) {
		//2D to do
		switch( target->datatype ) {
		case NIFTI_TYPE_FLOAT64:
			block_matching_method2D<double, double, double>
				(target, result, params, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			block_matching_method2D<float, float, float>
				(target, result, params, mask);
			break;
		default:
			reg_print_fct_error("block_matching_method");
			reg_print_msg_error("The target image data type is not supported");
			reg_exit(1);
		}
	}
	else {
		switch( target->datatype ) {
		case NIFTI_TYPE_FLOAT64:
			runKernel<double>
				(target, result, params, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			runKernel<float>
				(target, result, params, mask);
			break;
		default:
			reg_print_fct_error("block_matching_method");
			reg_print_msg_error("The target image data type is not supported");
			reg_exit(1);
		}
	}

}
template<class DTYPE> void CPUBlockMatchingKernel::runKernel(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) {

	DTYPE *targetPtr = static_cast<DTYPE *>( target->data );
	DTYPE *resultPtr = static_cast<DTYPE *>( result->data );


	mat44 *targetMatrix_xyz;
	if( target->sform_code >0 )
		targetMatrix_xyz = &( target->sto_xyz );
	else targetMatrix_xyz = &( target->qto_xyz );

	int targetIndex_start_x;
	int targetIndex_start_y;
	int targetIndex_start_z;
	int targetIndex_end_x;
	int targetIndex_end_y;
	int targetIndex_end_z;
	int resultIndex_start_x;
	int resultIndex_start_y;
	int resultIndex_start_z;
	int resultIndex_end_x;
	int resultIndex_end_y;
	int resultIndex_end_z;

	int index, i, j, k, l, m, n, x, y, z;
	int *maskPtr_Z, *maskPtr_XYZ;
	DTYPE *targetPtr_Z, *targetPtr_XYZ, *resultPtr_Z, *resultPtr_XYZ;
	DTYPE value, bestCC, targetMean, resultMean, targetVar, resultVar;
	DTYPE voxelNumber, localCC, targetTemp, resultTemp;
	float bestDisplacement[3], targetPosition_temp[3], tempPosition[3];
	size_t targetIndex, resultIndex, blockIndex, tid = 0;
	params->definedActiveBlock = 0;
#if defined (_OPENMP)
	int threadNumber = omp_get_max_threads();
	if( threadNumber>16 )
		omp_set_num_threads(16);
	DTYPE targetValues[16][BLOCK_SIZE];
	DTYPE resultValues[16][BLOCK_SIZE];
	bool targetOverlap[16][BLOCK_SIZE];
	bool resultOverlap[16][BLOCK_SIZE];
#else
	DTYPE targetValues[1][BLOCK_SIZE];
	DTYPE resultValues[1][BLOCK_SIZE];
	bool targetOverlap[1][BLOCK_SIZE];
	bool resultOverlap[1][BLOCK_SIZE];
#endif

#if defined (_OPENMP)
	float *currentTargetPosition = (float *)
		malloc(3 * params->activeBlockNumber*sizeof(float));
	float *currentResultPosition = (float *)
		malloc(3 * params->activeBlockNumber*sizeof(float));
	for( i = 0; i<3 * params->activeBlockNumber; i += 3 ) {
		currentTargetPosition[i] = std::numeric_limits<float>::quiet_NaN();
		currentResultPosition[i] = std::numeric_limits<float>::quiet_NaN();
	}
#pragma omp parallel for default(none) \
   shared(params, target, result, targetPtr, resultPtr, mask, targetMatrix_xyz, \
		  targetOverlap, resultOverlap, targetValues, resultValues, \
		  currentTargetPosition, currentResultPosition) \
   private(i, j, k, l, m, n, x, y, z, blockIndex, targetIndex, \
		   index, tid, targetPtr_Z, targetPtr_XYZ, resultPtr_Z, resultPtr_XYZ, \
		   maskPtr_Z, maskPtr_XYZ, value, bestCC, bestDisplacement, \
		   targetIndex_start_x, targetIndex_start_y, targetIndex_start_z, \
		   targetIndex_end_x, targetIndex_end_y, targetIndex_end_z, \
		   resultIndex_start_x, resultIndex_start_y, resultIndex_start_z, \
		   resultIndex_end_x, resultIndex_end_y, resultIndex_end_z, \
		   resultIndex, targetPosition_temp, tempPosition, targetTemp, resultTemp, \
		   targetMean, targetVar, resultMean, resultVar, voxelNumber,localCC)
#endif
	for( k = 0; k<params->blockNumber[2]; k++ ) {
#if defined (_OPENMP)
		tid = omp_get_thread_num();
#endif
		blockIndex = k * params->blockNumber[0] * params->blockNumber[1];
		targetIndex_start_z = k*BLOCK_WIDTH;
		targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;

		for( j = 0; j<params->blockNumber[1]; j++ ) {
			targetIndex_start_y = j*BLOCK_WIDTH;
			targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;

			for( i = 0; i<params->blockNumber[0]; i++ ) {
				targetIndex_start_x = i*BLOCK_WIDTH;
				targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;

				if( params->activeBlock[blockIndex] > -1 ) {
					targetIndex = 0;
					memset(targetOverlap[tid], 0, BLOCK_SIZE*sizeof(bool));
					for( z = targetIndex_start_z; z<targetIndex_end_z; z++ ) {
						if( -1<z && z<target->nz ) {
							index = z*target->nx*target->ny;
							targetPtr_Z = &targetPtr[index];
							maskPtr_Z = &mask[index];
							for( y = targetIndex_start_y; y<targetIndex_end_y; y++ ) {
								if( -1<y && y<target->ny ) {
									index = y*target->nx + targetIndex_start_x;
									targetPtr_XYZ = &targetPtr_Z[index];
									maskPtr_XYZ = &maskPtr_Z[index];
									for( x = targetIndex_start_x; x<targetIndex_end_x; x++ ) {
										if( -1<x && x<target->nx ) {
											value = *targetPtr_XYZ;
											if( value == value && *maskPtr_XYZ>-1 ) {
												targetValues[tid][targetIndex] = value;
												targetOverlap[tid][targetIndex] = 1;
											}
										}
										targetPtr_XYZ++;
										maskPtr_XYZ++;
										targetIndex++;
									}
								}
								else targetIndex += BLOCK_WIDTH;
							}
						}
						else targetIndex += BLOCK_WIDTH*BLOCK_WIDTH;
					}
					bestCC = 0.0;
					bestDisplacement[0] = std::numeric_limits<float>::quiet_NaN();
					bestDisplacement[1] = 0.f;
					bestDisplacement[2] = 0.f;

					// iteration over the result blocks
					for( n = -OVERLAP_SIZE; n <= OVERLAP_SIZE; n += STEP_SIZE ) {
						resultIndex_start_z = targetIndex_start_z + n;
						resultIndex_end_z = resultIndex_start_z + BLOCK_WIDTH;
						for( m = -OVERLAP_SIZE; m <= OVERLAP_SIZE; m += STEP_SIZE ) {
							resultIndex_start_y = targetIndex_start_y + m;
							resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;
							for( l = -OVERLAP_SIZE; l <= OVERLAP_SIZE; l += STEP_SIZE ) {
								resultIndex_start_x = targetIndex_start_x + l;
								resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;
								resultIndex = 0;
								memset(resultOverlap[tid], 0, BLOCK_SIZE*sizeof(bool));
								for( z = resultIndex_start_z; z<resultIndex_end_z; z++ ) {
									if( -1<z && z<result->nz ) {
										index = z*result->nx*result->ny;
										resultPtr_Z = &resultPtr[index];
										int *maskPtr_Z = &mask[index];
										for( y = resultIndex_start_y; y<resultIndex_end_y; y++ ) {
											if( -1<y && y<result->ny ) {
												index = y*result->nx + resultIndex_start_x;
												resultPtr_XYZ = &resultPtr_Z[index];
												int *maskPtr_XYZ = &maskPtr_Z[index];
												for( x = resultIndex_start_x; x<resultIndex_end_x; x++ ) {
													if( -1<x && x<result->nx ) {
														value = *resultPtr_XYZ;
														if( value == value && *maskPtr_XYZ>-1 ) {
															resultValues[tid][resultIndex] = value;
															resultOverlap[tid][resultIndex] = 1;
														}
													}
													resultPtr_XYZ++;
													resultIndex++;
													maskPtr_XYZ++;
												}
											}
											else resultIndex += BLOCK_WIDTH;
										}
									}
									else resultIndex += BLOCK_WIDTH*BLOCK_WIDTH;
								}
								targetMean = 0.0;
								resultMean = 0.0;
								voxelNumber = 0.0;
								for( int a = 0; a<BLOCK_SIZE; a++ ) {
									if( targetOverlap[tid][a] && resultOverlap[tid][a] ) {
										targetMean += targetValues[tid][a];
										resultMean += resultValues[tid][a];
										voxelNumber++;
									}
								}

								if( voxelNumber>BLOCK_SIZE / 2 ) {
									targetMean /= voxelNumber;
									resultMean /= voxelNumber;

									targetVar = 0.0;
									resultVar = 0.0;
									localCC = 0.0;

									for( int a = 0; a<BLOCK_SIZE; a++ ) {
										if( targetOverlap[tid][a] && resultOverlap[tid][a] ) {
											targetTemp = ( targetValues[tid][a] - targetMean );
											resultTemp = ( resultValues[tid][a] - resultMean );
											targetVar += (targetTemp)*( targetTemp );
											resultVar += (resultTemp)*( resultTemp );
											localCC += (targetTemp)*( resultTemp );
										}
									}

									localCC = fabs(localCC / sqrt(targetVar*resultVar));

									if( localCC>bestCC ) {
										bestCC = localCC;
										bestDisplacement[0] = (float)l;
										bestDisplacement[1] = (float)m;
										bestDisplacement[2] = (float)n;
									}
								}
							}
						}
					}
					if( bestDisplacement[0] == bestDisplacement[0] ) {
						targetPosition_temp[0] = (float)( i*BLOCK_WIDTH );
						targetPosition_temp[1] = (float)( j*BLOCK_WIDTH );
						targetPosition_temp[2] = (float)( k*BLOCK_WIDTH );

						bestDisplacement[0] += targetPosition_temp[0];
						bestDisplacement[1] += targetPosition_temp[1];
						bestDisplacement[2] += targetPosition_temp[2];

						reg_mat44_mul(targetMatrix_xyz, targetPosition_temp, tempPosition);
#if defined (_OPENMP)
						z = 3 * params->activeBlock[blockIndex];
						currentTargetPosition[z] = tempPosition[0];
						currentTargetPosition[z + 1] = tempPosition[1];
						currentTargetPosition[z + 2] = tempPosition[2];
#else
						z = 3 * params->definedActiveBlock;
						params->targetPosition[z] = tempPosition[0];
						params->targetPosition[z + 1] = tempPosition[1];
						params->targetPosition[z + 2] = tempPosition[2];
#endif
						reg_mat44_mul(targetMatrix_xyz, bestDisplacement, tempPosition);
#if defined (_OPENMP)
						currentResultPosition[z] = tempPosition[0];
						currentResultPosition[z + 1] = tempPosition[1];
						currentResultPosition[z + 2] = tempPosition[2];
#else
						params->resultPosition[z] = tempPosition[0];
						params->resultPosition[z + 1] = tempPosition[1];
						params->resultPosition[z + 2] = tempPosition[2];
						params->definedActiveBlock++;
#endif
					}
				}
				blockIndex++;
			}
		}
	}

#if defined (_OPENMP)
	j = 0;
	for( i = 0; i<3 * params->activeBlockNumber; i += 3 ) {
		if( currentTargetPosition[i] == currentTargetPosition[i] ) {
			params->targetPosition[j] = currentTargetPosition[i];
			params->targetPosition[j + 1] = currentTargetPosition[i + 1];
			params->targetPosition[j + 2] = currentTargetPosition[i + 2];
			params->resultPosition[j] = currentResultPosition[i];
			params->resultPosition[j + 1] = currentResultPosition[i + 1];
			params->resultPosition[j + 2] = currentResultPosition[i + 2];
			params->definedActiveBlock++;
			j += 3;
		}
	}
	free(currentTargetPosition);
	free(currentResultPosition);
	omp_set_num_threads(threadNumber);
#endif

}

void CPUBlockMatchingKernel::initialize(nifti_image * target, _reg_blockMatchingParam *params, int percentToKeep_block,  int percentToKeep_opt, int *mask, bool runningOnGPU) {
	if( params->activeBlock != NULL ) {
		free(params->activeBlock);
		params->activeBlock = NULL;
	}
	if( params->targetPosition != NULL ) {
		free(params->targetPosition);
		params->targetPosition = NULL;
	}
	if( params->resultPosition != NULL ) {
		free(params->resultPosition);
		params->resultPosition = NULL;
	}

	params->blockNumber[0] = (int)reg_ceil((float)target->nx / (float)BLOCK_WIDTH);
	params->blockNumber[1] = (int)reg_ceil((float)target->ny / (float)BLOCK_WIDTH);
	if( target->nz>1 )
		params->blockNumber[2] = (int)reg_ceil((float)target->nz / (float)BLOCK_WIDTH);
	else params->blockNumber[2] = 1;

	params->percent_to_keep = percentToKeep_opt;
	params->activeBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * percentToKeep_block / 100;

	params->activeBlock = (int *)malloc(params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * sizeof(int));
	switch( target->datatype ) {
	case NIFTI_TYPE_FLOAT32:
		setActiveBlocks<float>(target, params, mask, runningOnGPU);
		break;
	case NIFTI_TYPE_FLOAT64:
		setActiveBlocks<double>(target, params, mask, runningOnGPU);
		break;
	default:
		fprintf(stderr, "[NiftyReg ERROR] initialise_block_matching_method\tThe target image data type is not supported\n");
		reg_exit(1);
	}
	if( params->activeBlockNumber<2 ) {
		fprintf(stderr, "[NiftyReg ERROR] There are no active blocks\n");
		fprintf(stderr, "[NiftyReg ERROR] ... Exit ...\n");
		reg_exit(1);
	}
#ifndef NDEBUG
	printf("[NiftyReg DEBUG]: There are %i active block(s) out of %i.\n", params->activeBlockNumber, params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2]);
#endif
	if( target->nz>1 ) {
		std::cout << "allocating: " << params->activeBlockNumber << std::endl;
		params->targetPosition = (float *)malloc(params->activeBlockNumber * 3 * sizeof(float));
		params->resultPosition = (float *)malloc(params->activeBlockNumber * 3 * sizeof(float));
	}
	else {
		params->targetPosition = (float *)malloc(params->activeBlockNumber * 2 * sizeof(float));
		params->resultPosition = (float *)malloc(params->activeBlockNumber * 2 * sizeof(float));
	}
#ifndef NDEBUG
	printf("[NiftyReg DEBUG] block matching initialisation done.\n");
#endif
}

template <class DTYPE>
void CPUBlockMatchingKernel::setActiveBlocks(nifti_image *targetImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU) {
	const size_t totalBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
	float *varianceArray = (float *)malloc(totalBlockNumber*sizeof(float));
	int *indexArray = (int *)malloc(totalBlockNumber*sizeof(int));

	int *maskPtr = &mask[0];

	int unusableBlock = 0;
	size_t index;

	DTYPE *targetValues = (DTYPE *)malloc(BLOCK_SIZE * sizeof(DTYPE));
	DTYPE *targetPtr = static_cast<DTYPE *>( targetImage->data );
	int blockIndex = 0;

	if( targetImage->nz>1 ) {
		// Version using 3D blocks
		for( int k = 0; k<params->blockNumber[2]; k++ ) {
			for( int j = 0; j<params->blockNumber[1]; j++ ) {
				for( int i = 0; i<params->blockNumber[0]; i++ ) {
					for( unsigned int n = 0; n<BLOCK_SIZE; n++ )
						targetValues[n] = (DTYPE)std::numeric_limits<float>::quiet_NaN();
					float mean = 0.0f;
					float voxelNumber = 0.0f;
					int coord = 0;
					for( int z = k*BLOCK_WIDTH; z<( k + 1 )*BLOCK_WIDTH; z++ ) {
						if( z<targetImage->nz ) {
							index = z*targetImage->nx*targetImage->ny;
							DTYPE *targetPtrZ = &targetPtr[index];
							int *maskPtrZ = &maskPtr[index];
							for( int y = j*BLOCK_WIDTH; y<( j + 1 )*BLOCK_WIDTH; y++ ) {
								if( y<targetImage->ny ) {
									index = y*targetImage->nx + i*BLOCK_WIDTH;
									DTYPE *targetPtrXYZ = &targetPtrZ[index];
									int *maskPtrXYZ = &maskPtrZ[index];
									for( int x = i*BLOCK_WIDTH; x<( i + 1 )*BLOCK_WIDTH; x++ ) {
										if( x<targetImage->nx ) {
											targetValues[coord] = *targetPtrXYZ;
											if( targetValues[coord] == targetValues[coord] && targetValues[coord] != 0. && *maskPtrXYZ>-1 ) {
												mean += (float)targetValues[coord];
												voxelNumber++;
											}
										}
										targetPtrXYZ++;
										maskPtrXYZ++;
										coord++;
									}
								}
							}
						}
					}
					if( voxelNumber>BLOCK_SIZE / 2 ) {
						float variance = 0.0f;
						for( int i = 0; i<BLOCK_SIZE; i++ ) {
							if( targetValues[i] == targetValues[i] )
								variance += ( mean - (float)targetValues[i] )
								* ( mean - (float)targetValues[i] );
						}

						variance /= voxelNumber;
						varianceArray[blockIndex] = variance;
					}
					else {
						varianceArray[blockIndex] = -1;
						unusableBlock++;
					}
					indexArray[blockIndex] = blockIndex;
					blockIndex++;
				}
			}
		}
	}
	else {
		// Version using 2D blocks
		for( int j = 0; j<params->blockNumber[1]; j++ ) {
			for( int i = 0; i<params->blockNumber[0]; i++ ) {

				for( unsigned int n = 0; n<BLOCK_2D_SIZE; n++ )
					targetValues[n] = (DTYPE)std::numeric_limits<float>::quiet_NaN();
				float mean = 0.0f;
				float voxelNumber = 0.0f;
				int coord = 0;

				for( int y = j*BLOCK_WIDTH; y<( j + 1 )*BLOCK_WIDTH; y++ ) {
					if( y<targetImage->ny ) {
						index = y*targetImage->nx + i*BLOCK_WIDTH;
						DTYPE *targetPtrXY = &targetPtr[index];
						int *maskPtrXY = &maskPtr[index];
						for( int x = i*BLOCK_WIDTH; x<( i + 1 )*BLOCK_WIDTH; x++ ) {
							if( x<targetImage->nx ) {
								targetValues[coord] = *targetPtrXY;
								if( targetValues[coord] == targetValues[coord] && targetValues[coord] != 0. && *maskPtrXY>-1 ) {
									mean += (float)targetValues[coord];
									voxelNumber++;
								}
							}
							targetPtrXY++;
							maskPtrXY++;
							coord++;
						}
					}
				}
				if( voxelNumber>BLOCK_2D_SIZE / 2 ) {
					float variance = 0.0f;
					for( int i = 0; i<BLOCK_2D_SIZE; i++ ) {
						if( targetValues[i] == targetValues[i] )
							variance += ( mean - (float)targetValues[i] )
							* ( mean - (float)targetValues[i] );
					}

					variance /= voxelNumber;
					varianceArray[blockIndex] = variance;
				}
				else {
					varianceArray[blockIndex] = -1;
					unusableBlock++;
				}
				indexArray[blockIndex] = blockIndex;
				blockIndex++;
			}
		}
	}
	free(targetValues);

	params->activeBlockNumber = params->activeBlockNumber<( (int)totalBlockNumber - unusableBlock ) ? params->activeBlockNumber : ( totalBlockNumber - unusableBlock );

	reg_heapSort(varianceArray, indexArray, totalBlockNumber);

	memset(params->activeBlock, 0, totalBlockNumber * sizeof(int));
	int *indexArrayPtr = &indexArray[totalBlockNumber - 1];
	int count = 0;
	for( int i = 0; i<params->activeBlockNumber; i++ ) {
		params->activeBlock[*indexArrayPtr--] = count++;
	}
	for( size_t i = params->activeBlockNumber; i < totalBlockNumber; ++i ) {
		params->activeBlock[*indexArrayPtr--] = -1;
	}

	count = 0;
	if( runningOnGPU ) {
		for( size_t i = 0; i < totalBlockNumber; ++i ) {
			if( params->activeBlock[i] != -1 ) {
				params->activeBlock[i] = -1;
				params->activeBlock[count] = i;
				++count;
			}
		}
	}

	free(varianceArray);
	free(indexArray);
}

void CPUOptimiseKernel::execute(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine) {
	optimize(params, transformation_matrix, affine);
}

void CPUResampleImageKernel::execute(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp,float paddingValue, bool *dti_timepoint,mat33 * jacMat) {

	if( floatingImage->datatype != warpedImage->datatype ) {
		printf("[NiftyReg ERROR] reg_resampleImage\tSource and result image should have the same data type\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

	if( floatingImage->nt != warpedImage->nt ) {
		printf("[NiftyReg ERROR] reg_resampleImage\tThe source and result images have different dimension along the time axis\n");
		printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
		reg_exit(1);
	}

	// Define the DTI indices if required
	int dtIndicies[6];
	for( int i = 0; i<6; ++i ) dtIndicies[i] = -1;
	if( dti_timepoint != NULL ) {

		if( jacMat == NULL ) {
			printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
			reg_exit(1);
		}
		int j = 0;
		for( int i = 0; i<floatingImage->nt; ++i ) {
			if( dti_timepoint[i] == true )
				dtIndicies[j++] = i;
		}
		if( ( floatingImage->nz>1 && j != 6 ) && ( floatingImage->nz == 1 && j != 3 ) ) {
			printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
			printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
			reg_exit(1);
		}
	}

	// a mask array is created if no mask is specified
	bool MrPropreRules = false;
	if( mask == NULL ) {
		// voxels in the backgreg_round are set to -1 so 0 will do the job here
		mask = (int *)calloc(warpedImage->nx*warpedImage->ny*warpedImage->nz, sizeof(int));
		MrPropreRules = true;
	}

	switch( deformationField->datatype ) {
	case NIFTI_TYPE_FLOAT32:
		switch( floatingImage->datatype ) {
		case NIFTI_TYPE_UINT8:
			runKernel<float, unsigned char>(floatingImage,
													 warpedImage,
													 deformationField,
													 mask,
													 interp,
													 paddingValue,
													 dtIndicies,
													 jacMat);
			break;
		case NIFTI_TYPE_INT8:
			runKernel<float, char>(floatingImage,
											warpedImage,
											deformationField,
											mask,
											interp,
											paddingValue,
											dtIndicies,
											jacMat);
			break;
		case NIFTI_TYPE_UINT16:
			runKernel<float, unsigned short>(floatingImage,
													  warpedImage,
													  deformationField,
													  mask,
													  interp,
													  paddingValue,
													  dtIndicies,
													  jacMat);
			break;
		case NIFTI_TYPE_INT16:
			runKernel<float, short>(floatingImage,
											 warpedImage,
											 deformationField,
											 mask,
											 interp,
											 paddingValue,
											 dtIndicies,
											 jacMat);
			break;
		case NIFTI_TYPE_UINT32:
			runKernel<float, unsigned int>(floatingImage,
													warpedImage,
													deformationField,
													mask,
													interp,
													paddingValue,
													dtIndicies,
													jacMat);
			break;
		case NIFTI_TYPE_INT32:
			runKernel<float, int>(floatingImage,
										   warpedImage,
										   deformationField,
										   mask,
										   interp,
										   paddingValue,
										   dtIndicies,
										   jacMat);
			break;
		case NIFTI_TYPE_FLOAT32:
			runKernel<float, float>(floatingImage,
											 warpedImage,
											 deformationField,
											 mask,
											 interp,
											 paddingValue,
											 dtIndicies,
											 jacMat);
			break;
		case NIFTI_TYPE_FLOAT64:
			runKernel<float, double>(floatingImage,
											  warpedImage,
											  deformationField,
											  mask,
											  interp,
											  paddingValue,
											  dtIndicies,
											  jacMat);
			break;
		default:
			printf("Source pixel type unsupported.");
			break;
		}
		break;
	case NIFTI_TYPE_FLOAT64:
		switch( floatingImage->datatype ) {
		case NIFTI_TYPE_UINT8:
			runKernel<double, unsigned char>(floatingImage,
													  warpedImage,
													  deformationField,
													  mask,
													  interp,
													  paddingValue,
													  dtIndicies,
													  jacMat);
			break;
		case NIFTI_TYPE_INT8:
			runKernel<double, char>(floatingImage,
											 warpedImage,
											 deformationField,
											 mask,
											 interp,
											 paddingValue,
											 dtIndicies,
											 jacMat);
			break;
		case NIFTI_TYPE_UINT16:
			runKernel<double, unsigned short>(floatingImage,
													   warpedImage,
													   deformationField,
													   mask,
													   interp,
													   paddingValue,
													   dtIndicies,
													   jacMat);
			break;
		case NIFTI_TYPE_INT16:
			runKernel<double, short>(floatingImage,
											  warpedImage,
											  deformationField,
											  mask,
											  interp,
											  paddingValue,
											  dtIndicies,
											  jacMat);
			break;
		case NIFTI_TYPE_UINT32:
			runKernel<double, unsigned int>(floatingImage,
													 warpedImage,
													 deformationField,
													 mask,
													 interp,
													 paddingValue,
													 dtIndicies,
													 jacMat);
			break;
		case NIFTI_TYPE_INT32:
			runKernel<double, int>(floatingImage,
											warpedImage,
											deformationField,
											mask,
											interp,
											paddingValue,
											dtIndicies,
											jacMat);
			break;
		case NIFTI_TYPE_FLOAT32:
			runKernel<double, float>(floatingImage,
											  warpedImage,
											  deformationField,
											  mask,
											  interp,
											  paddingValue,
											  dtIndicies,
											  jacMat);
			break;
		case NIFTI_TYPE_FLOAT64:
			runKernel<double, double>(floatingImage,
											   warpedImage,
											   deformationField,
											   mask,
											   interp,
											   paddingValue,
											   dtIndicies,
											   jacMat);
			break;
		default:
			printf("Source pixel type unsupported.");
			break;
		}
		break;
	default:
		printf("Deformation field pixel type unsupported.");
		break;
	}
	if( MrPropreRules == true ) {
		free(mask);
		mask = NULL;
	}

}

template <class FieldTYPE, class SourceTYPE>
void CPUResampleImageKernel::runKernel(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationFieldImage, int *mask, int interp, float paddingValue, int *dtIndicies, mat33 * jacMat) {


	// The floating image data is copied in case one deal with DTI
	void *originalFloatingData = NULL;
	// The DTI are logged
	reg_dti_resampling_preprocessing<SourceTYPE>(floatingImage, &originalFloatingData, dtIndicies);

	/* The deformation field contains the position in the real world */
	if( interp == 3 && dtIndicies[0] == -1 ) {
		if( deformationFieldImage->nz>1 ) {
			CubicSplineResampleImage3D<SourceTYPE, FieldTYPE>(floatingImage,
															  deformationFieldImage,
															  warpedImage,
															  mask,
															  paddingValue);
		}
		else {
			CubicSplineResampleImage2D<SourceTYPE, FieldTYPE>(floatingImage,
															  deformationFieldImage,
															  warpedImage,
															  mask,
															  paddingValue);
		}
	}
	else if( interp == 0 && dtIndicies[0] == -1 )  // Nearest neighbor interpolation
	{
		if( deformationFieldImage->nz>1 ) {
			NearestNeighborResampleImage<SourceTYPE, FieldTYPE>(floatingImage,
																deformationFieldImage,
																warpedImage,
																mask,
																paddingValue);
		}
		else {
			NearestNeighborResampleImage2D<SourceTYPE, FieldTYPE>(floatingImage,
																  deformationFieldImage,
																  warpedImage,
																  mask,
																  paddingValue);
		}

	}
	else  // trilinear interpolation [ by default ]
	{
		if( deformationFieldImage->nz>1 ) {
			TrilinearResampleImage<SourceTYPE, FieldTYPE>(floatingImage,
														  deformationFieldImage,
														  warpedImage,
														  mask,
														  paddingValue);
		}
		else {
			BilinearResampleImage<SourceTYPE, FieldTYPE>(floatingImage,
														 deformationFieldImage,
														 warpedImage,
														 mask,
														 paddingValue);
		}
	}
	// The temporary logged floating array is deleted
	if( originalFloatingData != NULL ) {
		free(floatingImage->data);
		floatingImage->data = originalFloatingData;
		originalFloatingData = NULL;
	}
	// The interpolated tensors are reoriented and exponentiated
	reg_dti_resampling_postprocessing<SourceTYPE>(warpedImage, mask, jacMat, dtIndicies);

}





//template definitions to keep compiler happy
template void CPUAffineDeformationFieldKernel::runKernel3D<float>(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);
template void CPUAffineDeformationFieldKernel::runKernel3D<double>(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);
template void CPUAffineDeformationFieldKernel::runKernel2D<float>(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);
template void CPUAffineDeformationFieldKernel::runKernel2D<double>(mat44 *affineTransformation, nifti_image *deformationField, bool compose, int *mask);



template void CPUConvolutionKernel::runKernel<char>(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis);
template void CPUConvolutionKernel::runKernel<short>(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis);
template void CPUConvolutionKernel::runKernel<int>(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis);
template void CPUConvolutionKernel::runKernel<unsigned char>(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis);
template void CPUConvolutionKernel::runKernel<unsigned short>(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis);
template void CPUConvolutionKernel::runKernel<unsigned int>(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis);


template void CPUBlockMatchingKernel::setActiveBlocks<float>(nifti_image *targetImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU);
template void CPUBlockMatchingKernel::setActiveBlocks<double>(nifti_image *targetImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU);



template void CPUResampleImageKernel::runKernel<float, unsigned char>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float, unsigned short>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float, unsigned int>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float,  char>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float,  short>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float, int>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float, float>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<float, double>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);


template void CPUResampleImageKernel::runKernel<double, unsigned char>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, unsigned short>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, unsigned int>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, char>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, short>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, int>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, float>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
template void CPUResampleImageKernel::runKernel<double, double>(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, int *dti_timepoint, mat33 * jacMat);
