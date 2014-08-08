#include "CPUKernels.h"
#define T float


template <class FieldType> void CPUAffineDeformationField3DKernel<FieldType>::beginComputation(Context& context) {}
template <class FieldType> double CPUAffineDeformationField3DKernel<FieldType>::finishComputation(Context& context) { return 0.0; }

template <class FieldType>
void CPUAffineDeformationField3DKernel<FieldType>::initialize(nifti_image *CurrentReference, nifti_image **deformationFieldImage) {

	if( CurrentReference == NULL ) {
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::AllocateDeformationField()\n");
		fprintf(stderr, "[NiftyReg ERROR] Reference image is not defined. Exit.\n");
		reg_exit(1);
	}
	ClearDeformationField(*deformationFieldImage);
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
	( *deformationFieldImage )->nbyper = sizeof(T);
	if( sizeof(T) == 4 )
		( *deformationFieldImage )->datatype = NIFTI_TYPE_FLOAT32;
	else if( sizeof(T) == 8 )
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


template <class FieldType>
void CPUAffineDeformationField3DKernel<FieldType>::execute(mat44 *affineTransformation, nifti_image *deformationFieldImage, bool composition, int *mask) {

	std::cout << "running from kernel!" << std::endl;
		size_t voxelNumber = deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz;
		FieldType *deformationFieldPtrX = static_cast<FieldType *>( deformationFieldImage->data );
		FieldType *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
		FieldType *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

		mat44 *targetMatrix;
		if( deformationFieldImage->sform_code>0 ) {
			targetMatrix = &( deformationFieldImage->sto_xyz );
		}
		else targetMatrix = &( deformationFieldImage->qto_xyz );

		mat44 transformationMatrix;
		if( composition == true )
			transformationMatrix = *affineTransformation;
		else transformationMatrix = reg_mat44_mul(affineTransformation, targetMatrix);

		float voxel[3], position[3];
		int x, y, z;
		size_t index;
#if defined (NDEBUG) && defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(deformationFieldImage, transformationMatrix, deformationFieldPtrX, \
		  deformationFieldPtrY, deformationFieldPtrZ, mask, composition) \
   private(voxel, position, x, y, z, index)
#endif
		for( z = 0; z<deformationFieldImage->nz; z++ ) {
			index = z*deformationFieldImage->nx*deformationFieldImage->ny;
			voxel[2] = (float)z;
			for( y = 0; y<deformationFieldImage->ny; y++ ) {
				voxel[1] = (float)y;
				for( x = 0; x<deformationFieldImage->nx; x++ ) {
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
						//                    reg_mat44_mul(&transformationMatrix, voxel, position);

						/* the deformation field (real coordinates) is stored */
						deformationFieldPtrX[index] = position[0];
						deformationFieldPtrY[index] = position[1];
						deformationFieldPtrZ[index] = position[2];
					}
					index++;
				}
			}
		}
	}


template <class FieldType>
void::CPUAffineDeformationField3DKernel<FieldType>::ClearDeformationField(nifti_image *deformationFieldImage) {
	if( deformationFieldImage != NULL )
		nifti_image_free(deformationFieldImage);
	deformationFieldImage = NULL;
}

template <class DTYPE>
void CPUConvolutionKernel<DTYPE>::execute(nifti_image *image,
										  float *size,
										  int kernelType,
										  int *mask,
										  bool *timePoint,
										  bool *axis) {

	if( image->nx>2048 || image->ny>2048 || image->nz>2048 ) {
		reg_print_fct_error("convolutionKernelCore");
		reg_print_msg_error("This function does not support images with dimension > 2048");
		reg_exit(1);
	}
#ifdef WIN32
	long index;
	long voxelNumber = (long)image->nx*image->ny*image->nz;
#else
	size_t index;
	size_t voxelNumber = (size_t)image->nx*image->ny*image->nz;
#endif
	DTYPE *imagePtr = static_cast<DTYPE *>( image->data );
	int imageDim[3] = { image->nx, image->ny, image->nz };

	bool *nanImagePtr = (bool *)calloc(voxelNumber, sizeof(bool));
	float *densityPtr = (float *)calloc(voxelNumber, sizeof(float));

	// Loop over the dimension higher than 3
	for( int t = 0; t<image->nt*image->nu; t++ ) {
		if( timePoint[t] ) {
			DTYPE *intensityPtr = &imagePtr[t * voxelNumber];
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
		 shared(densityPtr, intensityPtr, mask, nanImagePtr, voxelNumber) \
		 private(index)
#endif
			for( index = 0; index<voxelNumber; index++ ) {
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
			for( index = 0; index<voxelNumber; ++index ) {
				if( nanImagePtr[index] != 0 )
					intensityPtr[index] = static_cast<DTYPE>( (float)intensityPtr[index] / densityPtr[index] );
				else intensityPtr[index] = std::numeric_limits<DTYPE>::quiet_NaN();
			}
		} // check if the time point is active
	} // loop over the time points
	free(nanImagePtr);
	free(densityPtr);

	
}

template <class DTYPE> void CPUConvolutionKernel<DTYPE>::beginComputation(Context& context) {}
template <class DTYPE> double CPUConvolutionKernel<DTYPE>::finishComputation(Context& context) { return 0.0; }


void TempFunction() {
	std::string name = "test";
	const Platform *platform;
	//just to keep the compiler happy with templated functions
	CPUConvolutionKernel<unsigned int> tempUintObj(name, *platform);
	CPUConvolutionKernel<unsigned short> tempUShortObj(name, *platform);
	CPUConvolutionKernel<unsigned char> tempUcharObj(name, *platform);

	CPUConvolutionKernel< int> tempintObj(name, *platform);
	CPUConvolutionKernel< short> tempShortObj(name, *platform);
	CPUConvolutionKernel< char> tempcharObj(name, *platform);

	CPUConvolutionKernel< float > tempFloatObj(name, *platform);
	CPUConvolutionKernel< double> tempDoubleObj(name, *platform);

	CPUAffineDeformationField3DKernel<unsigned int> tempUcharObj1(name, *platform);
	CPUAffineDeformationField3DKernel<unsigned short> tempUcharObj2(name, *platform);
	CPUAffineDeformationField3DKernel<unsigned char> tempUcharObj3(name, *platform);

	CPUAffineDeformationField3DKernel<int> tempUcharObj4(name, *platform);
	CPUAffineDeformationField3DKernel<short> tempUcharObj5(name, *platform);
	CPUAffineDeformationField3DKernel<char> tempUcharObj6(name, *platform);

	CPUAffineDeformationField3DKernel<float> tempUcharObj7(name, *platform);
	CPUAffineDeformationField3DKernel<double> tempUcharObj8(name, *platform);

}

