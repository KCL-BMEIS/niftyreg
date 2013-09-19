/*
 *  _reg_resampling.cpp
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_CPP
#define _REG_RESAMPLING_CPP

#include "_reg_resampling.h"
#include "_reg_maths.h"

/* *************************************************************** */
template <class FieldTYPE>
void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis)
{
    if(ratio<0.0) ratio=0.0; //reg_rounding error
    FieldTYPE FF= ratio*ratio;
    basis[0] = (FieldTYPE)((ratio * ((2.0-ratio)*ratio - 1.0))/2.0);
    basis[1] = (FieldTYPE)((FF * (3.0*ratio-5.0) + 2.0)/2.0);
    basis[2] = (FieldTYPE)((ratio * ((4.0-3.0*ratio)*ratio + 1.0))/2.0);
    basis[3] = (FieldTYPE)((ratio-1.0) * FF/2.0);
}
/* *************************************************************** */
template <class FieldTYPE>
void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis, FieldTYPE *derivative)
{
    interpolantCubicSpline<FieldTYPE>(ratio,basis);
    if(ratio<0.0) ratio=0.0; //reg_rounding error
    FieldTYPE FF= ratio*ratio;
    derivative[0] = (FieldTYPE)((4.0*ratio - 3.0*FF - 1.0)/2.0);
    derivative[1] = (FieldTYPE)((9.0*ratio - 10.0) * ratio/2.0);
    derivative[2] = (FieldTYPE)((8.0*ratio - 9.0*FF + 1)/2.0);
    derivative[3] = (FieldTYPE)((3.0*ratio - 2.0) * ratio/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_dti_resampling_preprocessing(nifti_image *floatingImage,
									  void **originalFloatingData,
                                      int *dtIndicies)
{
    // If we have some valid diffusion tensor indicies, we need to replace the tensor components
    // by the the log tensor components
    if( dtIndicies[0] != -1 ){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] DTI indices:");
        for(unsigned int i = 0; i < 6; i++ )
            printf(" %i", dtIndicies[i]);
        printf("\n");
#endif
        size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;

		*originalFloatingData=(void *)malloc(floatingImage->nvox*sizeof(DTYPE));
		memcpy(*originalFloatingData,
               floatingImage->data,
			   floatingImage->nvox*sizeof(DTYPE));
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] The floating image data has been copied\n");
#endif

        /* As the tensor has 6 unique components that we need to worry about, read them out
        for the floating image. */
        DTYPE *firstVox = static_cast<DTYPE *>(floatingImage->data);
        DTYPE *floatingIntensityXX = &firstVox[floatingVoxelNumber*dtIndicies[0]];
		DTYPE *floatingIntensityXY = &firstVox[floatingVoxelNumber*dtIndicies[1]];
		DTYPE *floatingIntensityYY = &firstVox[floatingVoxelNumber*dtIndicies[2]];
        DTYPE *floatingIntensityXZ = &firstVox[floatingVoxelNumber*dtIndicies[3]];
        DTYPE *floatingIntensityYZ = &firstVox[floatingVoxelNumber*dtIndicies[4]];
		DTYPE *floatingIntensityZZ = &firstVox[floatingVoxelNumber*dtIndicies[5]];

        // We need a mat44 to store the diffusion tensor at each voxel for our calculating. Although the DT is 3x3 really,
        // it is convenient to store it as a 4x4 to work with existing code for the matrix logarithm/exponential
        mat44 diffTensor;

        // Should log the tensor up front
        // We need to take the logarithm of the tensor for each voxel in the floating intensity image, and replace the warped
        size_t floatingIndex;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(floatingIndex,diffTensor) \
    shared(floatingVoxelNumber,floatingIntensityXX,floatingIntensityYY, \
        floatingIntensityZZ,floatingIntensityXY,floatingIntensityXZ, \
        floatingIntensityYZ)
#endif
        for(floatingIndex=0; floatingIndex<floatingVoxelNumber; ++floatingIndex){
            if(floatingIntensityXX[floatingIndex] > 1e-2){
                // Fill a mat44 with the tensor components
                reg_mat44_eye(&diffTensor);
                diffTensor.m[0][0] = floatingIntensityXX[floatingIndex];
                diffTensor.m[0][1] = floatingIntensityXY[floatingIndex];
                diffTensor.m[1][0] = diffTensor.m[0][1];
                diffTensor.m[1][1] = floatingIntensityYY[floatingIndex];
                diffTensor.m[0][2] = floatingIntensityXZ[floatingIndex];
                diffTensor.m[2][0] = diffTensor.m[0][2];
                diffTensor.m[1][2] = floatingIntensityYZ[floatingIndex];
                diffTensor.m[2][1] = diffTensor.m[1][2];
                diffTensor.m[2][2] = floatingIntensityZZ[floatingIndex];
                // Decompose the mat33 into a rotation and a diagonal matrix of eigen values
                // Recompose as a log tensor Rt log(E) R, where E is a diagonal matrix
                // containing the eigen values and R is a rotation matrix. This is the same as
				// taking the logarithm of the tensor
				diffTensor = reg_mat44_logm(&diffTensor);
                // Write this out as a new image
                floatingIntensityXX[floatingIndex] = diffTensor.m[0][0];
                floatingIntensityXY[floatingIndex] = diffTensor.m[0][1];
                floatingIntensityYY[floatingIndex] = diffTensor.m[1][1];
                floatingIntensityXZ[floatingIndex] = diffTensor.m[0][2];
                floatingIntensityYZ[floatingIndex] = diffTensor.m[1][2];
                floatingIntensityZZ[floatingIndex] = diffTensor.m[2][2];
            }
            else{ // if junk diffusion data, log it
                floatingIntensityXX[floatingIndex] = static_cast<DTYPE>(-4.606f);
                floatingIntensityYY[floatingIndex] = static_cast<DTYPE>(-4.606f);
                floatingIntensityZZ[floatingIndex] = static_cast<DTYPE>(-4.606f);
            }
        }
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Tensors have been logged\n");
#endif
    }
}

/* *************************************************************** */
template <class DTYPE>
void reg_dti_resampling_postprocessing(nifti_image *inputImage,
                                       int *mask,
                                       mat33 *jacMat,
									   int *dtIndicies,
									   nifti_image *warpedImage = NULL)
{
    // If we have some valid diffusion tensor indicies, we need to exponentiate the previously logged tensor components
    // we also need to reorient the tensors based on the local transformation Jacobians
    if(dtIndicies[0] != -1 ){

		size_t voxelNumber = (size_t)inputImage->nx*inputImage->ny*inputImage->nz;
		DTYPE *warpVox,*warpedXX,*warpedXY,*warpedXZ,*warpedYY,*warpedYZ,*warpedZZ;
		if(warpedImage!=NULL){
			warpVox = static_cast<DTYPE *>(warpedImage->data);
			warpedXX = &warpVox[voxelNumber*dtIndicies[0]];
			warpedXY = &warpVox[voxelNumber*dtIndicies[1]];
			warpedYY = &warpVox[voxelNumber*dtIndicies[2]];
			warpedXZ = &warpVox[voxelNumber*dtIndicies[3]];
			warpedYZ = &warpVox[voxelNumber*dtIndicies[4]];
			warpedZZ = &warpVox[voxelNumber*dtIndicies[5]];
		}
		for(int u=0;u<inputImage->nu;++u){
            // Now, we need to exponentiate the warped intensities back to give us a regular tensor
			// let's reorient each tensor based on the rigid component of the local warping
            /* As the tensor has 6 unique components that we need to worry about, read them out
			for the warped image. */

			DTYPE *firstWarpVox = static_cast<DTYPE *>(inputImage->data);
			DTYPE *inputIntensityXX = &firstWarpVox[voxelNumber*(dtIndicies[0]+inputImage->nt*u)];
			DTYPE *inputIntensityXY = &firstWarpVox[voxelNumber*(dtIndicies[1]+inputImage->nt*u)];
			DTYPE *inputIntensityYY = &firstWarpVox[voxelNumber*(dtIndicies[2]+inputImage->nt*u)];
			DTYPE *inputIntensityXZ = &firstWarpVox[voxelNumber*(dtIndicies[3]+inputImage->nt*u)];
			DTYPE *inputIntensityYZ = &firstWarpVox[voxelNumber*(dtIndicies[4]+inputImage->nt*u)];
			DTYPE *inputIntensityZZ = &firstWarpVox[voxelNumber*(dtIndicies[5]+inputImage->nt*u)];

            // Step through each voxel in the warped image
			size_t warpedIndex; double testSum=0;
			reg_mat44d inputTensor, warpedTensor, RotMat, RotMatT, preMult;
			mat33 jacobianMatrix, R;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
	private(warpedIndex,inputTensor,jacobianMatrix,R,RotMat,RotMatT,preMult, warpedTensor) \
	shared(voxelNumber,inputIntensityXX,inputIntensityYY,inputIntensityZZ, \
	warpedXX, warpedXY, warpedXZ, warpedYY, warpedYZ, warpedZZ, warpedImage, \
	inputIntensityXY,inputIntensityXZ,inputIntensityYZ, jacMat, mask, testSum)
#endif
			for(warpedIndex=0; warpedIndex<voxelNumber; ++warpedIndex){
				if(mask[warpedIndex]>-1){
					// The fourth row/colum are all 0s, include the diagonal as this is a log tensor
					inputTensor.m[0][3] = 0.0; inputTensor.m[3][0] = 0.0;
					inputTensor.m[1][3] = 0.0; inputTensor.m[3][1] = 0.0;
					inputTensor.m[2][3] = 0.0; inputTensor.m[3][2] = 0.0;
					// Fill the rest of the mat44 with the tensor components
					inputTensor.m[0][0] = (double)inputIntensityXX[warpedIndex];
					inputTensor.m[0][1] = (double)inputIntensityXY[warpedIndex];
					inputTensor.m[1][0] = inputTensor.m[0][1];
					inputTensor.m[1][1] = (double)inputIntensityYY[warpedIndex];
					inputTensor.m[0][2] = (double)inputIntensityXZ[warpedIndex];
					inputTensor.m[2][0] = inputTensor.m[0][2];
					inputTensor.m[1][2] = (double)inputIntensityYZ[warpedIndex];
					inputTensor.m[2][1] = inputTensor.m[1][2];
					inputTensor.m[2][2] = (double)inputIntensityZZ[warpedIndex];
					// Exponentiate the warped tensor
					if(warpedImage==NULL){
						inputTensor.m[3][3] = 0.0;
						inputTensor = reg_mat44_expm(&inputTensor);
						testSum=0;
					}
					else{
						inputTensor.m[3][3]  = 1.0;
						reg_mat44_eye(&warpedTensor);
						warpedTensor.m[0][0] = (double)warpedXX[warpedIndex];
						warpedTensor.m[0][1] = (double)warpedXY[warpedIndex];
						warpedTensor.m[1][0] = warpedTensor.m[0][1];
						warpedTensor.m[1][1] = (double)warpedYY[warpedIndex];
						warpedTensor.m[0][2] = (double)warpedXZ[warpedIndex];
						warpedTensor.m[2][0] = warpedTensor.m[0][2];
						warpedTensor.m[1][2] = (double)warpedYZ[warpedIndex];
						warpedTensor.m[2][1] = warpedTensor.m[1][2];
						warpedTensor.m[2][2] = (double)warpedZZ[warpedIndex];
						inputTensor = reg_mat44_mul(&warpedTensor,&inputTensor);
						testSum=(double)warpedTensor.m[0][0]+warpedTensor.m[0][1]+warpedTensor.m[0][2]+
								warpedTensor.m[1][0]+warpedTensor.m[1][1]+warpedTensor.m[1][2]+
								warpedTensor.m[2][0]+warpedTensor.m[2][1]+warpedTensor.m[2][2];
					}

					if(testSum==testSum){
						// Find the rotation matrix from the local warp Jacobian
						jacobianMatrix = jacMat[warpedIndex];
						// Calculate the polar decomposition of the local Jacobian matrix, which
						// tells us how to rotate the local tensor information
						R = nifti_mat33_polar(jacobianMatrix);
						// We need both the rotation matrix, and it's transpose as a mat44
						reg_mat44_eye(&RotMat);
						reg_mat44_eye(&RotMatT);
						for(unsigned char col=0; col<3; col++){
							for(unsigned char row=0; row<3; row++){
								RotMat.m[col][row] = R.m[col][row];
								RotMatT.m[col][row] = R.m[row][col];
							}
						}
						// As the mat44 multiplication uses pointers, do the multiplications separately
						preMult = reg_mat44_mul(&RotMatT, &inputTensor);
						inputTensor = reg_mat44_mul(&preMult,&RotMat);

						// Finally, read the tensor back out as a warped image
						inputIntensityXX[warpedIndex] = inputTensor.m[0][0];
						inputIntensityYY[warpedIndex] = inputTensor.m[1][1];
						inputIntensityZZ[warpedIndex] = inputTensor.m[2][2];
						inputIntensityXY[warpedIndex] = inputTensor.m[0][1];
						inputIntensityXZ[warpedIndex] = inputTensor.m[0][2];
						inputIntensityYZ[warpedIndex] = inputTensor.m[1][2];
					}
					else{
						inputIntensityXX[warpedIndex] = 0;
						inputIntensityYY[warpedIndex] = 0;
						inputIntensityZZ[warpedIndex] = 0;
						inputIntensityXY[warpedIndex] = 0;
						inputIntensityXZ[warpedIndex] = 0;
						inputIntensityYZ[warpedIndex] = 0;
					}
                }
            }
        }
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Exponentiated and rotated all voxels\n");
#endif
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void CubicSplineResampleImage3D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *warpedImage,
                                int *mask,
                                FieldTYPE paddingValue)
{
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    size_t resultVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[resultVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[resultVoxelNumber];

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedImage->nt * warpedImage->nu * resultVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Iteration over the different volume along the 4th axis
    for(size_t t=0; t<(size_t)warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D Cubic spline resampling of volume number %lu\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*resultVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[4], yBasis[4], zBasis[4], relative;
        int a, b, c, Y, Z, previous[3];
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        SourceTYPE *zPointer, *yzPointer, *xyzPointer;
        FieldTYPE xTempNewValue, yTempNewValue, intensity, world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, resultIntensity, resultVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<resultVoxelNumber; index++){

            intensity=(FieldTYPE)(0.0);

            if((maskPtr[index])>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                previous[2] = static_cast<int>(reg_floor(position[2]));

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, xBasis);
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, yBasis);
                // basis values along the z axis
                relative=position[2]-(FieldTYPE)previous[2];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, zBasis);

                --previous[0];--previous[1];--previous[2];

                for(c=0; c<4; c++){
                    Z= previous[2]+c;
                    zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                    yTempNewValue=0.0;
                    for(b=0; b<4; b++){
                        Y= previous[1]+b;
                        yzPointer = &zPointer[Y*floatingImage->nx];
                        xyzPointer = &yzPointer[previous[0]];
                        xTempNewValue=0.0;
                        for(a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<floatingImage->nx &&
                                    -1<Z && Z<floatingImage->nz &&
                                    -1<Y && Y<floatingImage->ny){
                                xTempNewValue +=  (FieldTYPE)*xyzPointer * xBasis[a];
                            }
                            else{
                                // paddingValue
                                xTempNewValue +=  paddingValue * xBasis[a];
                            }
                            xyzPointer++;
                        }
                        yTempNewValue += xTempNewValue * yBasis[b];
                    }
                    intensity += yTempNewValue * zBasis[c];
                }
            }

            switch(floatingImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)reg_round(intensity);
                break;
            }

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing 3D Cubic Spline Resampling...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void CubicSplineResampleImage2D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *warpedImage,
                                int *mask,
                                FieldTYPE paddingValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    size_t targetVoxelNumber = warpedImage->nx*warpedImage->ny;
    size_t sourceVoxelNumber = floatingImage->nx*floatingImage->ny;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=floatingImage->sto_ijk;
    else sourceIJKMatrix=floatingImage->qto_ijk;

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedImage->nt * warpedImage->nu * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D Cubic spline resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[4], yBasis[4], relative;
        int a, b, Y, previous[2];
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        SourceTYPE *yPointer, *xyPointer;
        FieldTYPE xTempNewValue, intensity, world[2], position[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, relative, \
    a, b, Y, yPointer, xyPointer, xTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            intensity=0.0;

            if((maskPtr[index])>-1){

                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix.m[0][0] + world[1]*sourceIJKMatrix.m[0][1] +
                        sourceIJKMatrix.m[0][3];
                position[1] = world[0]*sourceIJKMatrix.m[1][0] + world[1]*sourceIJKMatrix.m[1][1] +
                        sourceIJKMatrix.m[1][3];

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, xBasis);
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, yBasis);

                previous[0]--;previous[1]--;

                for(b=0; b<4; b++){
                    Y= previous[1]+b;
                    yPointer = &sourceIntensity[Y*floatingImage->nx];
                    xyPointer = &yPointer[previous[0]];
                    xTempNewValue=0.0;
                    for(a=0; a<4; a++){
                        if(-1<(previous[0]+a) && (previous[0]+a)<floatingImage->nx &&
                                -1<Y && Y<floatingImage->ny){
                            xTempNewValue +=  (FieldTYPE)*xyPointer * xBasis[a];
                        }
                        else{
                            // paddingValue x
                            xTempNewValue +=  paddingValue * xBasis[a];
                        }
                        xyPointer++;
                    }
                    intensity += xTempNewValue * yBasis[b];
                }
            }

            switch(floatingImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)reg_round(intensity);
                break;
            }

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing 2D Cubic Spline Resampling...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void TrilinearResampleImage(nifti_image *floatingImage,
                            nifti_image *deformationField,
                            nifti_image *warpedImage,
                            int *mask,
                            FieldTYPE paddingValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    size_t targetVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];
    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedImage->nt * warpedImage->nu * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[2], yBasis[2], zBasis[2], relative;
        int a, b, c, X, Y, Z, previous[3];
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        SourceTYPE *zPointer, *xyzPointer;
        FieldTYPE xTempNewValue, yTempNewValue, intensity, world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, X, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            intensity=paddingValue;

            if(maskPtr[index]>-1){

                intensity=0;

                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                previous[2] = static_cast<int>(reg_floor(position[2]));

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                xBasis[0]= (FieldTYPE)(1.0-relative);
                xBasis[1]= relative;
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                yBasis[0]= (FieldTYPE)(1.0-relative);
                yBasis[1]= relative;
                // basis values along the z axis
                relative=position[2]-(FieldTYPE)previous[2];
                zBasis[0]= (FieldTYPE)(1.0-relative);
                zBasis[1]= relative;

                // For efficiency reason two interpolation are here, with and without using a padding value
                if(paddingValue==paddingValue){
                    // Interpolation using the padding value
                    for(c=0; c<2; c++){
                        Z= previous[2]+c;
                        if(Z>-1 && Z<floatingImage->nz){
                            zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                            yTempNewValue=0.0;
                            for(b=0; b<2; b++){
                                Y= previous[1]+b;
                                if(Y>-1 && Y<floatingImage->ny){
                                    xyzPointer = &zPointer[Y*floatingImage->nx+previous[0]];
                                    xTempNewValue=0.0;
                                    for(a=0; a<2; a++){
                                        X= previous[0]+a;
                                        if(X>-1 && X<floatingImage->nx){
                                            xTempNewValue +=  *xyzPointer * xBasis[a];
                                        } // X
                                        else xTempNewValue +=  paddingValue * xBasis[a];
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
                else if(previous[0]>=0.f && previous[0]<(floatingImage->nx-1) &&
                        previous[1]>=0.f && previous[1]<(floatingImage->ny-1) &&
                        previous[2]>=0.f && previous[2]<(floatingImage->nz-1) ){
                    for(c=0; c<2; c++){
                        Z= previous[2]+c;
                        zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                        yTempNewValue=0.0;
                        for(b=0; b<2; b++){
                            Y= previous[1]+b;
                            xyzPointer = &zPointer[Y*floatingImage->nx+previous[0]];
                            xTempNewValue=0.0;
                            for(a=0; a<2; a++){
                                X= previous[0]+a;
                                xTempNewValue +=  *xyzPointer * xBasis[a];
                                xyzPointer++;
                            } // a
                            yTempNewValue += xTempNewValue * yBasis[b];
                        } // b
                        intensity += yTempNewValue * zBasis[c];
                    } // c
                } // padding value is not defined
                // The voxel is outside of the source space and thus set to NaN here
                else intensity=paddingValue;
            } // voxel is in the mask

            switch(floatingImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)reg_round(intensity);
                break;
            }

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing Trilinear Resampling...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void BilinearResampleImage(nifti_image *floatingImage,
                           nifti_image *deformationField,
                           nifti_image *warpedImage,
                           int *mask,
                           FieldTYPE paddingValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    size_t  targetVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny;
    size_t  sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = (unsigned long)targetVoxelNumber*warpedImage->nt * warpedImage->nu;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){

#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D linear resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[2], yBasis[2], relative;
        int a, b, X, Y, previous[3];
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        SourceTYPE *xyPointer;
        FieldTYPE xTempNewValue, intensity, world[2], position[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, relative, \
    a, b, X, Y, xyPointer, xTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0; index<targetVoxelNumber; ++index){

            intensity=paddingValue;

            if(maskPtr[index]>-1){

                intensity=0;

                world[0] = deformationFieldPtrX[index];
                world[1] = deformationFieldPtrY[index];

                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix->m[0][0] +
                        world[1]*sourceIJKMatrix->m[0][1] +
                        sourceIJKMatrix->m[0][3];
                position[1] = world[0]*sourceIJKMatrix->m[1][0] +
                        world[1]*sourceIJKMatrix->m[1][1] +
                        sourceIJKMatrix->m[1][3];

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                relative=relative>0?relative:0;
                xBasis[0]= (FieldTYPE)(1.0-relative);
                xBasis[1]= relative;
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                relative=relative>0?relative:0;
                yBasis[0]= (FieldTYPE)(1.0-relative);
                yBasis[1]= relative;

                for(b=0; b<2; b++){
                    Y = previous[1]+b;
                    if(Y>-1 && Y<floatingImage->ny){
                        xyPointer = &sourceIntensity[Y*floatingImage->nx+previous[0]];
                        xTempNewValue=0.0;
                        for(a=0; a<2; a++){
                            X = previous[0]+a;
                            if(X>-1 && X<floatingImage->nx){
                                xTempNewValue +=  *xyPointer * xBasis[a];
                            }
                            else xTempNewValue +=  paddingValue * xBasis[a];
                            xyPointer++;
                        } // a
                        intensity += xTempNewValue * yBasis[b];
                    } // Y outside
                    else intensity += paddingValue * yBasis[b];
                } // b
            } // mask

            switch(floatingImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?reg_round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)reg_round(intensity);
                break;
            }

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing Bilinear Resampling...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleImage(nifti_image *floatingImage,
                                  nifti_image *deformationField,
                                  nifti_image *warpedImage,
                                  int *mask,
                                  FieldTYPE paddingValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    size_t targetVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedImage->nt * warpedImage->nu * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D nearest neighbor resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        SourceTYPE intensity;
        FieldTYPE world[3];
        FieldTYPE position[3];
        int previous[3];
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0; index<targetVoxelNumber; index++){

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = (int)reg_round(position[0]);
                previous[1] = (int)reg_round(position[1]);
                previous[2] = (int)reg_round(position[2]);

                if( -1<previous[2] && previous[2]<floatingImage->nz &&
                        -1<previous[1] && previous[1]<floatingImage->ny &&
                        -1<previous[0] && previous[0]<floatingImage->nx){
                    intensity = sourceIntensity[(previous[2]*floatingImage->ny+previous[1]) *
                            floatingImage->nx+previous[0]];
                    resultIntensity[index]=intensity;
                }
                else resultIntensity[index]=(SourceTYPE)paddingValue;
            }
            else resultIntensity[index]=(SourceTYPE)paddingValue;

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing 3D Nearest Neighbour Resampling...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleImage2D(nifti_image *floatingImage,
                                    nifti_image *deformationField,
                                    nifti_image *warpedImage,
                                    int *mask,
                                    FieldTYPE paddingValue)
{
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    size_t targetVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedImage->nt * warpedImage->nu * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D nearest neighbor resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        SourceTYPE intensity;
        FieldTYPE world[2];
        FieldTYPE position[2];
        int previous[2];
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            if((*maskPtr++)>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix->m[0][0] + world[1]*sourceIJKMatrix->m[0][1] +
                        sourceIJKMatrix->m[0][3];
                position[1] = world[0]*sourceIJKMatrix->m[1][0] + world[1]*sourceIJKMatrix->m[1][1] +
                        sourceIJKMatrix->m[1][3];

                previous[0] = static_cast<int>(reg_round(position[0]));
                previous[1] = static_cast<int>(reg_round(position[1]));

                if( -1<previous[1] && previous[1]<floatingImage->ny &&
                        -1<previous[0] && previous[0]<floatingImage->nx){
                    intensity = sourceIntensity[previous[1]*floatingImage->nx+previous[0]];
                    resultIntensity[index]=intensity;
                }
                else resultIntensity[index]=(SourceTYPE)paddingValue;
            }
            else resultIntensity[index]=(SourceTYPE)paddingValue;

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing 2D Nearest Neighbour Resampling...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */

/** This function resample a source image into the referential
 * of a reference image by applying an affine transformation and
 * a deformation field. The affine transformation has to be in
 * real coordinate and the deformation field is in mm in the space
 * of the target image.
 * interp can be either 0, 1 or 3 meaning nearest neighbor, linear
 * or cubic spline interpolation.
 * every voxel which is not fully in the source image takes the
 * backgreg_round value. The dtIndicies are an array of size 6
 * that provides the position of the DT components (if there are any)
 * these values are set to -1 if there are not
 */
template <class FieldTYPE, class SourceTYPE>
void reg_resampleImage2(nifti_image *floatingImage,
                        nifti_image *warpedImage,
                        nifti_image *deformationFieldImage,
                        int *mask,
                        int interp,
                        FieldTYPE paddingValue,
                        int* dtIndicies,
                        mat33 * jacMat)
{
    // The floating image data is copied in case one deal with DTI
	void *originalFloatingData=NULL;
    // The DTI are logged
	reg_dti_resampling_preprocessing<SourceTYPE>(floatingImage,
												 &originalFloatingData,
												 dtIndicies);

    /* The deformation field contains the position in the real world */
    if(interp==3 && dtIndicies[0] == -1){
        if(deformationFieldImage->nz>1){
            CubicSplineResampleImage3D<SourceTYPE,FieldTYPE>(floatingImage,
                                                             deformationFieldImage,
                                                             warpedImage,
                                                             mask,
                                                             paddingValue);
        }
        else
        {
            CubicSplineResampleImage2D<SourceTYPE,FieldTYPE>(floatingImage,
                                                             deformationFieldImage,
                                                             warpedImage,
                                                             mask,
                                                             paddingValue);
        }
    }
    else if(interp==0 && dtIndicies[0] == -1){ // Nearest neighbor interpolation
        if(deformationFieldImage->nz>1){
            NearestNeighborResampleImage<SourceTYPE, FieldTYPE>(floatingImage,
                                                                deformationFieldImage,
                                                                warpedImage,
                                                                mask,
                                                                paddingValue);
        }
        else
        {
            NearestNeighborResampleImage2D<SourceTYPE, FieldTYPE>(floatingImage,
                                                                  deformationFieldImage,
                                                                  warpedImage,
                                                                  mask,
                                                                  paddingValue);
        }

    }
    else{ // trilinear interpolation [ by default ]
        if(deformationFieldImage->nz>1){
            TrilinearResampleImage<SourceTYPE, FieldTYPE>(floatingImage,
                                                          deformationFieldImage,
                                                          warpedImage,
                                                          mask,
                                                          paddingValue);
        }
        else{
            BilinearResampleImage<SourceTYPE, FieldTYPE>(floatingImage,
                                                         deformationFieldImage,
                                                         warpedImage,
                                                         mask,
                                                         paddingValue);
        }
    }
    // The temporary logged floating array is deleted
	if(originalFloatingData!=NULL){
		free(floatingImage->data);
		floatingImage->data=originalFloatingData;
		originalFloatingData=NULL;
    }
    // The interpolated tensors are reoriented and exponentiated
	reg_dti_resampling_postprocessing<SourceTYPE>(warpedImage,
												  mask,
												  jacMat,
												  dtIndicies);
}
/* *************************************************************** */
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       nifti_image *deformationField,
                       int *mask,
                       int interp,
                       float paddingValue,
                       bool *dti_timepoint,
                       mat33 * jacMat)
{
    if(floatingImage->datatype != warpedImage->datatype){
        printf("[NiftyReg ERROR] reg_resampleImage\tSource and result image should have the same data type\n");
        printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
        reg_exit(1);
    }

    if(floatingImage->nt != warpedImage->nt){
        printf("[NiftyReg ERROR] reg_resampleImage\tThe source and result images have different dimension along the time axis\n");
        printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
        reg_exit(1);
    }

    // Define the DTI indices if required
    int dtIndicies[6];
    for(int i=0;i<6;++i) dtIndicies[i]=-1;
    if(dti_timepoint!=NULL){

        if(jacMat==NULL){
            printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
            printf("[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
            reg_exit(1);
        }
        int j=0;
        for(int i=0;i<floatingImage->nt;++i){
            if(dti_timepoint[i]==true)
                dtIndicies[j++]=i;
        }
        if((floatingImage->nz>1 && j!=6) && (floatingImage->nz==1 && j!=3)){
            printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
            printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
            reg_exit(1);
        }
    }

    // a mask array is created if no mask is specified
    bool MrPropreRules = false;
    if(mask==NULL){
        // voxels in the backgreg_round are set to -1 so 0 will do the job here
        mask=(int *)calloc(warpedImage->nx*warpedImage->ny*warpedImage->nz,sizeof(int));
        MrPropreRules = true;
    }

    switch ( deformationField->datatype ){
    case NIFTI_TYPE_FLOAT32:
        switch ( floatingImage->datatype ){
        case NIFTI_TYPE_UINT8:
            reg_resampleImage2<float,unsigned char>(floatingImage,
                                                    warpedImage,
                                                    deformationField,
                                                    mask,
                                                    interp,
                                                    paddingValue,
                                                    dtIndicies,
                                                    jacMat);
            break;
        case NIFTI_TYPE_INT8:
            reg_resampleImage2<float,char>(floatingImage,
                                           warpedImage,
                                           deformationField,
                                           mask,
                                           interp,
                                           paddingValue,
                                           dtIndicies,
                                           jacMat);
            break;
        case NIFTI_TYPE_UINT16:
            reg_resampleImage2<float,unsigned short>(floatingImage,
                                                     warpedImage,
                                                     deformationField,
                                                     mask,
                                                     interp,
                                                     paddingValue,
                                                     dtIndicies,
                                                     jacMat);
            break;
        case NIFTI_TYPE_INT16:
            reg_resampleImage2<float,short>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            mask,
                                            interp,
                                            paddingValue,
                                            dtIndicies,
                                            jacMat);
            break;
        case NIFTI_TYPE_UINT32:
            reg_resampleImage2<float,unsigned int>(floatingImage,
                                                   warpedImage,
                                                   deformationField,
                                                   mask,
                                                   interp,
                                                   paddingValue,
                                                   dtIndicies,
                                                   jacMat);
            break;
        case NIFTI_TYPE_INT32:
            reg_resampleImage2<float,int>(floatingImage,
                                          warpedImage,
                                          deformationField,
                                          mask,
                                          interp,
                                          paddingValue,
                                          dtIndicies,
                                          jacMat);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_resampleImage2<float,float>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            mask,
                                            interp,
                                            paddingValue,
                                            dtIndicies,
                                            jacMat);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleImage2<float,double>(floatingImage,
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
        switch ( floatingImage->datatype ){
        case NIFTI_TYPE_UINT8:
            reg_resampleImage2<double,unsigned char>(floatingImage,
                                                     warpedImage,
                                                     deformationField,
                                                     mask,
                                                     interp,
                                                     paddingValue,
                                                     dtIndicies,
                                                     jacMat);
            break;
        case NIFTI_TYPE_INT8:
            reg_resampleImage2<double,char>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            mask,
                                            interp,
                                            paddingValue,
                                            dtIndicies,
                                            jacMat);
            break;
        case NIFTI_TYPE_UINT16:
            reg_resampleImage2<double,unsigned short>(floatingImage,
                                                      warpedImage,
                                                      deformationField,
                                                      mask,
                                                      interp,
                                                      paddingValue,
                                                      dtIndicies,
                                                      jacMat);
            break;
        case NIFTI_TYPE_INT16:
            reg_resampleImage2<double,short>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             mask,
                                             interp,
                                             paddingValue,
                                             dtIndicies,
                                             jacMat);
            break;
        case NIFTI_TYPE_UINT32:
            reg_resampleImage2<double,unsigned int>(floatingImage,
                                                    warpedImage,
                                                    deformationField,
                                                    mask,
                                                    interp,
                                                    paddingValue,
                                                    dtIndicies,
                                                    jacMat );
            break;
        case NIFTI_TYPE_INT32:
            reg_resampleImage2<double,int>(floatingImage,
                                           warpedImage,
                                           deformationField,
                                           mask,
                                           interp,
                                           paddingValue,
                                           dtIndicies,
                                           jacMat);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_resampleImage2<double,float>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             mask,
                                             interp,
                                             paddingValue,
                                             dtIndicies,
                                             jacMat);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleImage2<double,double>(floatingImage,
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
    if(MrPropreRules==true){ free(mask);mask=NULL;}
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bilinearResampleGradient(nifti_image *floatingImage,
                                  nifti_image *warpedImage,
                                  nifti_image *deformationField,
                                  float paddingValue)
{
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
    size_t warpedVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    DTYPE *floatingIntensityX = static_cast<DTYPE *>(floatingImage->data);
    DTYPE *floatingIntensityY = &floatingIntensityX[floatingVoxelNumber];
    DTYPE *warpedIntensityX = static_cast<DTYPE *>(warpedImage->data);
    DTYPE *warpedIntensityY = &warpedIntensityX[warpedVoxelNumber];
    DTYPE *deformationFieldPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *deformationFieldPtrY = &deformationFieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];

    // Extract the relevant affine matrix
    mat44 *floating_mm_to_voxel = &floatingImage->qto_ijk;
    if(floatingImage->sform_code!=0)
        floating_mm_to_voxel = &floatingImage->sto_ijk;

    // The spacing is computed in case the sform if defined
    DTYPE realSpacing[2];
    if(warpedImage->sform_code>0){
        reg_getRealImageSpacing(warpedImage,realSpacing);
    }
    else{
        realSpacing[0]=warpedImage->dx;
        realSpacing[1]=warpedImage->dy;
    }

    // Reorientation matrix is assessed in order to remove the rigid component
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));

    // Some useful variables
    mat33 jacMat;
    DTYPE defX,defY;
    DTYPE basisX[2], basisY[2], deriv[2], basis[2];
    DTYPE xFloCoord,yFloCoord;
    int anteIntX[2],anteIntY[2];
    int x,y,a,b,defIndex,floIndex,warpedIndex;
    DTYPE val_x,val_y,weight[2];

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedImage->nx * warpedImage->ny;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    // Loop over all voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(x,y,a,b,val_x,val_y,defIndex,floIndex,warpedIndex, \
    anteIntX,anteIntY,xFloCoord,yFloCoord, \
    basisX,basisY,deriv,basis,defX,defY,jacMat,weight) \
    shared(warpedImage,warpedIntensityX,warpedIntensityY, \
    deformationField,deformationFieldPtrX,deformationFieldPtrY, \
    floatingImage,floatingIntensityX,floatingIntensityY,floating_mm_to_voxel, \
    paddingValue, reorient,realSpacing, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
    for(y=0; y<warpedImage->ny; ++y){
        warpedIndex=y*warpedImage->nx;
        deriv[0]=-1;
        deriv[1]=1;
        basis[0]=1;
        basis[1]=0;
        for(x=0; x<warpedImage->nx; ++x){
            warpedIntensityX[warpedIndex]=paddingValue;
            warpedIntensityY[warpedIndex]=paddingValue;

            // Compute the index in the floating image
            defX=deformationFieldPtrX[warpedIndex];
            defY=deformationFieldPtrY[warpedIndex];
            xFloCoord =
                    floating_mm_to_voxel->m[0][0] * defX +
                    floating_mm_to_voxel->m[0][1] * defY +
                    floating_mm_to_voxel->m[0][3];
            yFloCoord =
                    floating_mm_to_voxel->m[1][0] * defX +
                    floating_mm_to_voxel->m[1][1] * defY +
                    floating_mm_to_voxel->m[1][3];

            // Extract the floating value using bilinear interpolation
            anteIntX[0]=static_cast<int>(reg_floor(xFloCoord));
            anteIntX[1]=static_cast<int>(reg_ceil(xFloCoord));
            anteIntY[0]=static_cast<int>(reg_floor(yFloCoord));
            anteIntY[1]=static_cast<int>(reg_ceil(yFloCoord));
            val_x=0;val_y=0;
            basisX[1]=fabs(xFloCoord-(DTYPE)anteIntX[0]);
            basisY[1]=fabs(yFloCoord-(DTYPE)anteIntY[0]);
            basisX[0]=1.0-basisX[1];
            basisY[0]=1.0-basisY[1];
            for(b=0;b<2;++b){
                if(anteIntY[b]>-1 && anteIntY[b]<floatingImage->ny){
                    for(a=0;a<2;++a){
                        weight[0]=basisX[a] * basisY[b];
                        if(anteIntX[a]>-1 && anteIntX[a]<floatingImage->nx){
                            floIndex = anteIntY[b]*floatingImage->nx+anteIntX[a];
                            val_x += floatingIntensityX[floIndex] * weight[0];
                            val_y += floatingIntensityY[floIndex] * weight[0];
                        } // anteIntX not in the floating image space
                        else{
                            val_x += paddingValue * weight[0];
                            val_y += paddingValue * weight[0];
                        }
                    } // a
                } // anteIntY not in the floating image space
                else{
                    val_x += paddingValue * basisY[b];
                    val_y += paddingValue * basisY[b];
                }
            } // b

            // Compute the Jacobian matrix
            memset(&jacMat,0,sizeof(mat33));
            jacMat.m[2][2]=1.;
            for(b=0;b<2;++b){
                anteIntY[0]=y+b;
                basisY[0]=basis[b];
                basisY[1]=deriv[b];
                // Boundary conditions along y - slidding
                if(y==deformationField->ny-1){
                    if(b==1)
                        anteIntY[0]-=2;
                    basisY[0]=fabs(basisY[0]-1.);
                    basisY[1]*=-1.;
                }
                for(a=0;a<2;++a){
                    anteIntX[0]=x+a;
                    basisX[0]=basis[a];
                    basisX[1]=deriv[a];
                    // Boundary conditions along x - slidding
                    if(x==deformationField->nx-1){
                        if(a==1)
                            anteIntX[0]-=2;
                        basisX[0]=fabs(basisX[0]-1.);
                        basisX[1]*=-1.;
                    }

                    // Compute the basis function values
                    weight[0] = basisX[1]*basisY[0];
                    weight[1] = basisX[0]*basisY[1];

                    // Get the deformation field index
                    defIndex=anteIntY[0]*deformationField->nx+anteIntX[0];

                    // Get the deformation field values
                    defX=deformationFieldPtrX[defIndex];
                    defY=deformationFieldPtrY[defIndex];

                    // Symmetric difference to compute the derivatives
                    jacMat.m[0][0] += weight[0]*defX;
                    jacMat.m[0][1] += weight[1]*defX;
                    jacMat.m[1][0] += weight[0]*defY;
                    jacMat.m[1][1] += weight[1]*defY;
                }
            }
            // reorient and scale the Jacobian matrix
            jacMat=nifti_mat33_mul(reorient,jacMat);
            jacMat.m[0][0] /= realSpacing[0];
            jacMat.m[0][1] /= realSpacing[1];
            jacMat.m[1][0] /= realSpacing[0];
            jacMat.m[1][1] /= realSpacing[1];

            // Modulate the gradient scalar values
            warpedIntensityX[warpedIndex]=jacMat.m[0][0]*val_x + jacMat.m[0][1]*val_y;
            warpedIntensityY[warpedIndex]=jacMat.m[1][0]*val_x + jacMat.m[1][1]*val_y;

            ++warpedIndex;

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing Bilinear Gradient Resampling...");

            // Increment the progress counter
            iProgressStep++;

        } // x
    } // y
}
/* *************************************************************** */
template <class DTYPE>
void reg_trilinearResampleGradient(nifti_image *floatingImage,
                                   nifti_image *warpedImage,
                                   nifti_image *deformationField,
                                   float paddingValue)
{
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
    size_t warpedVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    DTYPE *floatingIntensityX = static_cast<DTYPE *>(floatingImage->data);
    DTYPE *floatingIntensityY = &floatingIntensityX[floatingVoxelNumber];
    DTYPE *floatingIntensityZ = &floatingIntensityY[floatingVoxelNumber];
    DTYPE *warpedIntensityX = static_cast<DTYPE *>(warpedImage->data);
    DTYPE *warpedIntensityY = &warpedIntensityX[warpedVoxelNumber];
    DTYPE *warpedIntensityZ = &warpedIntensityY[warpedVoxelNumber];
    DTYPE *deformationFieldPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *deformationFieldPtrY = &deformationFieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];
    DTYPE *deformationFieldPtrZ = &deformationFieldPtrY[deformationField->nx*deformationField->ny*deformationField->nz];

    // Extract the relevant affine matrix
    mat44 *floating_mm_to_voxel = &floatingImage->qto_ijk;
    if(floatingImage->sform_code!=0)
        floating_mm_to_voxel = &floatingImage->sto_ijk;

    // The spacing is computed in case the sform if defined
    DTYPE realSpacing[3];
    if(warpedImage->sform_code>0){
        reg_getRealImageSpacing(warpedImage,realSpacing);
    }
    else{
        realSpacing[0]=warpedImage->dx;
        realSpacing[1]=warpedImage->dy;
        realSpacing[2]=warpedImage->dz;
    }

    // Reorientation matrix is assessed in order to remove the rigid component
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));

    // Some useful variables
    mat33 jacMat;
    DTYPE defX,defY,defZ;
    DTYPE basisX[2], basisY[2], basisZ[2], deriv[2], basis[2];
    DTYPE xFloCoord,yFloCoord,zFloCoord;
    int anteIntX[2],anteIntY[2],anteIntZ[2];
    int x,y,z,a,b,c,defIndex,floIndex,warpedIndex;
    DTYPE val_x,val_y,val_z,weight[3];

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = warpedVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    // Loop over all voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(x,y,z,a,b,c,val_x,val_y,val_z,defIndex,floIndex,warpedIndex, \
    anteIntX,anteIntY,anteIntZ,xFloCoord,yFloCoord,zFloCoord, \
    basisX,basisY,basisZ,deriv,basis,defX,defY,defZ,jacMat,weight) \
    shared(warpedImage,warpedIntensityX,warpedIntensityY,warpedIntensityZ, \
    deformationField,deformationFieldPtrX,deformationFieldPtrY,deformationFieldPtrZ, \
    floatingImage,floatingIntensityX,floatingIntensityY,floatingIntensityZ,floating_mm_to_voxel, \
    paddingValue, reorient, realSpacing, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
    for(z=0; z<warpedImage->nz; ++z){
        warpedIndex=z*warpedImage->nx*warpedImage->ny;
        deriv[0]=-1;
        deriv[1]=1;
        basis[0]=1;
        basis[1]=0;
        for(y=0; y<warpedImage->ny; ++y){
            for(x=0; x<warpedImage->nx; ++x){
                warpedIntensityX[warpedIndex]=paddingValue;
                warpedIntensityY[warpedIndex]=paddingValue;
                warpedIntensityZ[warpedIndex]=paddingValue;

                // Compute the index in the floating image
                defX=deformationFieldPtrX[warpedIndex];
                defY=deformationFieldPtrY[warpedIndex];
                defZ=deformationFieldPtrZ[warpedIndex];
                xFloCoord =
                        floating_mm_to_voxel->m[0][0] * defX +
                        floating_mm_to_voxel->m[0][1] * defY +
                        floating_mm_to_voxel->m[0][2] * defZ +
                        floating_mm_to_voxel->m[0][3];
                yFloCoord =
                        floating_mm_to_voxel->m[1][0] * defX +
                        floating_mm_to_voxel->m[1][1] * defY +
                        floating_mm_to_voxel->m[1][2] * defZ +
                        floating_mm_to_voxel->m[1][3];
                zFloCoord =
                        floating_mm_to_voxel->m[2][0] * defX +
                        floating_mm_to_voxel->m[2][1] * defY +
                        floating_mm_to_voxel->m[2][2] * defZ +
                        floating_mm_to_voxel->m[2][3];

                // Extract the floating value using bilinear interpolation
                anteIntX[0]=static_cast<int>(reg_floor(xFloCoord));
                anteIntX[1]=static_cast<int>(reg_ceil(xFloCoord));
                anteIntY[0]=static_cast<int>(reg_floor(yFloCoord));
                anteIntY[1]=static_cast<int>(reg_ceil(yFloCoord));
                anteIntZ[0]=static_cast<int>(reg_floor(zFloCoord));
                anteIntZ[1]=static_cast<int>(reg_ceil(zFloCoord));
                val_x=0;val_y=0;val_z=0;
                basisX[1]=fabs(xFloCoord-(DTYPE)anteIntX[0]);
                basisY[1]=fabs(yFloCoord-(DTYPE)anteIntY[0]);
                basisZ[1]=fabs(zFloCoord-(DTYPE)anteIntZ[0]);
                basisX[0]=1.0-basisX[1];
                basisY[0]=1.0-basisY[1];
                basisZ[0]=1.0-basisZ[1];
                for(c=0;c<2;++c){
                    if(anteIntZ[c]>-1 && anteIntZ[c]<floatingImage->nz){
                        for(b=0;b<2;++b){
                            if(anteIntY[b]>-1 && anteIntY[b]<floatingImage->ny){
                                for(a=0;a<2;++a){
                                    weight[0]=basisX[a] * basisY[b] * basisZ[c];
                                    if(anteIntX[a]>-1 && anteIntX[a]<floatingImage->nx){
                                        floIndex = (anteIntZ[c]*floatingImage->ny+anteIntY[b])*floatingImage->nx+anteIntX[a];
                                        val_x += floatingIntensityX[floIndex] * weight[0];
                                        val_y += floatingIntensityY[floIndex] * weight[0];
                                        val_z += floatingIntensityZ[floIndex] * weight[0];
                                    } // anteIntX not in the floating image space
                                    else{
                                        val_x += paddingValue * weight[0];
                                        val_y += paddingValue * weight[0];
                                        val_z += paddingValue * weight[0];
                                    }
                                } // a
                            } // anteIntY not in the floating image space
                            else{
                                val_x += paddingValue * basisY[b] * basisZ[c];
                                val_y += paddingValue * basisY[b] * basisZ[c];
                                val_z += paddingValue * basisY[b] * basisZ[c];
                            }
                        } // b
                    } // anteIntZ not in the floating image space
                    else{
                        val_x += paddingValue * basisZ[c];
                        val_y += paddingValue * basisZ[c];
                        val_z += paddingValue * basisZ[c];
                    }
                } // c

                // Compute the Jacobian matrix
                memset(&jacMat,0,sizeof(mat33));
                for(c=0;c<2;++c){
                    anteIntZ[0]=z+c;
                    basisZ[0]=basis[c];
                    basisZ[1]=deriv[c];
                    // Boundary conditions along z - slidding
                    if(z==deformationField->nz-1){
                        if(c==1)
                            anteIntZ[0]-=2;
                        basisZ[0]=fabs(basisZ[0]-1.);
                        basisZ[1]*=-1.;
                    }
                    for(b=0;b<2;++b){
                        anteIntY[0]=y+b;
                        basisY[0]=basis[b];
                        basisY[1]=deriv[b];
                        // Boundary conditions along y - slidding
                        if(y==deformationField->ny-1){
                            if(b==1)
                                anteIntY[0]-=2;
                            basisY[0]=fabs(basisY[0]-1.);
                            basisY[1]*=-1.;
                        }
                        for(a=0;a<2;++a){
                            anteIntX[0]=x+a;
                            basisX[0]=basis[a];
                            basisX[1]=deriv[a];
                            // Boundary conditions along x - slidding
                            if(x==deformationField->nx-1){
                                if(a==1)
                                    anteIntX[0]-=2;
                                basisX[0]=fabs(basisX[0]-1.);
                                basisX[1]*=-1.;
                            }

                            // Compute the basis function values
                            weight[0] = basisX[1]*basisY[0]*basisZ[0];
                            weight[1] = basisX[0]*basisY[1]*basisZ[0];
                            weight[2] = basisX[0]*basisY[0]*basisZ[1];

                            // Get the deformation field index
                            defIndex=(anteIntZ[0]*deformationField->ny+anteIntY[0]) *
                                    deformationField->nx+anteIntX[0];

                            // Get the deformation field values
                            defX=deformationFieldPtrX[defIndex];
                            defY=deformationFieldPtrY[defIndex];
                            defZ=deformationFieldPtrZ[defIndex];

                            // Symmetric difference to compute the derivatives
                            jacMat.m[0][0] += weight[0]*defX;
                            jacMat.m[0][1] += weight[1]*defX;
                            jacMat.m[0][2] += weight[2]*defX;
                            jacMat.m[1][0] += weight[0]*defY;
                            jacMat.m[1][1] += weight[1]*defY;
                            jacMat.m[1][2] += weight[2]*defY;
                            jacMat.m[2][0] += weight[0]*defZ;
                            jacMat.m[2][1] += weight[1]*defZ;
                            jacMat.m[2][2] += weight[2]*defZ;
                        }
                    }
                }
                // reorient and scale the Jacobian matrix
                jacMat=nifti_mat33_mul(reorient,jacMat);
                jacMat.m[0][0] /= realSpacing[0];
                jacMat.m[0][1] /= realSpacing[1];
                jacMat.m[0][2] /= realSpacing[2];
                jacMat.m[1][0] /= realSpacing[0];
                jacMat.m[1][1] /= realSpacing[1];
                jacMat.m[1][2] /= realSpacing[2];
                jacMat.m[2][0] /= realSpacing[0];
                jacMat.m[2][1] /= realSpacing[1];
                jacMat.m[2][2] /= realSpacing[2];

                // Modulate the gradient scalar values
                warpedIntensityX[warpedIndex]=jacMat.m[0][0]*val_x+jacMat.m[0][1]*val_y+jacMat.m[0][2]*val_z;
                warpedIntensityY[warpedIndex]=jacMat.m[1][0]*val_x+jacMat.m[1][1]*val_y+jacMat.m[1][2]*val_z;
                warpedIntensityZ[warpedIndex]=jacMat.m[2][0]*val_x+jacMat.m[2][1]*val_y+jacMat.m[2][2]*val_z;

                ++warpedIndex;

                // Announce the progress via CLI
                if (iProgressStep % progressUnit == 0)
                    progressXML(100 * iProgressStep / nProgressSteps, "Performing Trilinear Gradient Resampling...");

                // Increment the progress counter
                iProgressStep++;

            } // x
        } // y
    } // z
}
/* *************************************************************** */
void reg_resampleGradient(nifti_image *floatingImage,
                          nifti_image *warpedImage,
                          nifti_image *deformationField,
                          int interp,
                          float paddingValue)
{
    interp=interp; // to avoid a warning - need to add the spline interpolation
    if(floatingImage->datatype!=warpedImage->datatype ||
            floatingImage->datatype!=deformationField->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_resampleGradient - Input images are expected to have the same type\n");
        reg_exit(1);
    }
    switch(floatingImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        if(warpedImage->nz>1){
            reg_trilinearResampleGradient<float>(floatingImage,
                                                 warpedImage,
                                                 deformationField,
                                                 paddingValue);
        }
        else
        {
            reg_bilinearResampleGradient<float>(floatingImage,
                                                warpedImage,
                                                deformationField,
                                                paddingValue);
        }
        break;
    case NIFTI_TYPE_FLOAT64:
        if(warpedImage->nz>1){
            reg_trilinearResampleGradient<double>(floatingImage,
                                                  warpedImage,
                                                  deformationField,
                                                  paddingValue);
        }
        else
        {
            reg_bilinearResampleGradient<double>(floatingImage,
                                                 warpedImage,
                                                 deformationField,
                                                 paddingValue);
        }
        break;
    default:
        fprintf(stderr, "[NiftyReg ERROR] reg_resampleGradient - Only single and double floating precision are supported\n");
        reg_exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearImageGradient(nifti_image *floatingImage,
                            nifti_image *deformationField,
                            nifti_image *resultGradientImage,
                            int *mask,
                            float paddingValue)
{
    size_t targetVoxelNumber = (size_t)resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = resultGradientImage->nt * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear gradient computation of volume number %i\n",t);
#endif
        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[t*3*targetVoxelNumber];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[targetVoxelNumber];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[targetVoxelNumber];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        int previous[3], a, b, c, X, Y, Z;
        FieldTYPE position[3], xBasis[2], yBasis[2], zBasis[2];
        FieldTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        FieldTYPE relative, world[3], grad[3], coeff;
        FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
        SourceTYPE *zPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, zBasis, relative, grad, coeff, \
    a, b, c, X, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, deriv, paddingValue, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY, resultGradientPtrZ, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;
            grad[2]=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                previous[2] = static_cast<int>(reg_floor(position[2]));
                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                xBasis[0]= (FieldTYPE)(1.0-relative);
                xBasis[1]= relative;
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                yBasis[0]= (FieldTYPE)(1.0-relative);
                yBasis[1]= relative;
                // basis values along the z axis
                relative=position[2]-(FieldTYPE)previous[2];
                zBasis[0]= (FieldTYPE)(1.0-relative);
                zBasis[1]= relative;

                // The padding value is used for interpolation if it is different from NaN
                if(paddingValue==paddingValue){
                    for(c=0; c<2; c++){
                        Z=previous[2]+c;
                        if(Z>-1 && Z<floatingImage->nz){
                            zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                            xxTempNewValue=0.0;
                            yyTempNewValue=0.0;
                            zzTempNewValue=0.0;
                            for(b=0; b<2; b++){
                                Y=previous[1]+b;
                                if(Y>-1 && Y<floatingImage->ny){
                                    xyzPointer = &zPointer[Y*floatingImage->nx+previous[0]];
                                    xTempNewValue=0.0;
                                    yTempNewValue=0.0;
                                    for(a=0; a<2; a++){
                                        X=previous[0]+a;
                                        if(X>-1 && X<floatingImage->nx){
                                            coeff = *xyzPointer;
                                            xTempNewValue +=  coeff * deriv[a];
                                            yTempNewValue +=  coeff * xBasis[a];
                                        } // end X in range
                                        else{
                                            xTempNewValue +=  paddingValue * deriv[a];
                                            yTempNewValue +=  paddingValue * xBasis[a];
                                        }
                                        xyzPointer++;
                                    } // end a
                                    xxTempNewValue += xTempNewValue * yBasis[b];
                                    yyTempNewValue += yTempNewValue * deriv[b];
                                    zzTempNewValue += yTempNewValue * yBasis[b];
                                } // end Y in range
                                else{
                                    xxTempNewValue += paddingValue * yBasis[b];
                                    yyTempNewValue += paddingValue * deriv[b];
                                    zzTempNewValue += paddingValue * yBasis[b];
                                }
                            } // end b
                            grad[0] += xxTempNewValue * zBasis[c];
                            grad[1] += yyTempNewValue * zBasis[c];
                            grad[2] += zzTempNewValue * deriv[c];
                        } // end Z in range
                        else{
                            grad[0] += paddingValue * zBasis[c];
                            grad[1] += paddingValue * zBasis[c];
                            grad[2] += paddingValue * deriv[c];
                        }
                    } // end c
                } // end padding value is different from NaN
                else if(previous[0]>=0.f && previous[0]<(floatingImage->nx-1) &&
                        previous[1]>=0.f && previous[1]<(floatingImage->ny-1) &&
                        previous[2]>=0.f && previous[2]<(floatingImage->nz-1) ){
                    for(c=0; c<2; c++){
                        Z=previous[2]+c;
                        zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                        xxTempNewValue=0.0;
                        yyTempNewValue=0.0;
                        zzTempNewValue=0.0;
                        for(b=0; b<2; b++){
                            Y=previous[1]+b;
                            xyzPointer = &zPointer[Y*floatingImage->nx+previous[0]];
                            xTempNewValue=0.0;
                            yTempNewValue=0.0;
                            for(a=0; a<2; a++){
                                X=previous[0]+a;
                                coeff = *xyzPointer;
                                xTempNewValue +=  coeff * deriv[a];
                                yTempNewValue +=  coeff * xBasis[a];
                                xyzPointer++;
                            } // end a
                            xxTempNewValue += xTempNewValue * yBasis[b];
                            yyTempNewValue += yTempNewValue * deriv[b];
                            zzTempNewValue += yTempNewValue * yBasis[b];
                        } // end b
                        grad[0] += xxTempNewValue * zBasis[c];
                        grad[1] += yyTempNewValue * zBasis[c];
                        grad[2] += zzTempNewValue * deriv[c];
                    } // end c
                } // end padding value is NaN
                else grad[0]=grad[1]=grad[2]=0;
            } // end mask

            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];
            resultGradientPtrZ[index] = (GradientTYPE)grad[2];

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing Trilinear Gradient Computation...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void BilinearImageGradient(nifti_image *floatingImage,
                           nifti_image *deformationField,
                           nifti_image *resultGradientImage,
                           int *mask,
                           float paddingValue)
{
    size_t targetVoxelNumber = (size_t)resultGradientImage->nx*resultGradientImage->ny;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=floatingImage->sto_ijk;
    else sourceIJKMatrix=floatingImage->qto_ijk;

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = resultGradientImage->nt * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D linear gradient computation of volume number %i\n",t);
#endif
        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[2*t*targetVoxelNumber];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[targetVoxelNumber];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE position[3], xBasis[2], yBasis[2], relative, world[2], grad[2];
        FieldTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        FieldTYPE coeff, xTempNewValue, yTempNewValue;
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        int previous[3], a, b, X, Y;
        SourceTYPE *xyPointer;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, relative, grad, coeff, \
    a, b, X, Y, xyPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, paddingValue, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];

                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix.m[0][0] + world[1]*sourceIJKMatrix.m[0][1] +
                        sourceIJKMatrix.m[0][3];
                position[1] = world[0]*sourceIJKMatrix.m[1][0] + world[1]*sourceIJKMatrix.m[1][1] +
                        sourceIJKMatrix.m[1][3];

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                relative=relative>0?relative:0;
                xBasis[0]= (FieldTYPE)(1.0-relative);
                xBasis[1]= relative;
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                relative=relative>0?relative:0;
                yBasis[0]= (FieldTYPE)(1.0-relative);
                yBasis[1]= relative;

                for(b=0; b<2; b++){
                    Y= previous[1]+b;
                    if(Y>-1 && Y<floatingImage->ny){
                        xyPointer = &sourceIntensity[Y*floatingImage->nx+previous[0]];
                        xTempNewValue=0.0;
                        yTempNewValue=0.0;
                        for(a=0; a<2; a++){
                            X= previous[0]+a;
                            if(X>-1 && X<floatingImage->nx){
                                coeff = *xyPointer;
                                xTempNewValue +=  coeff * deriv[a];
                                yTempNewValue +=  coeff * xBasis[a];
                            }
                            else{
                                xTempNewValue +=  paddingValue * deriv[a];
                                yTempNewValue +=  paddingValue * xBasis[a];
                            }
                            xyPointer++;
                        }
                        grad[0] += xTempNewValue * yBasis[b];
                        grad[1] += yTempNewValue * deriv[b];
                    }
                    else{
                        grad[0] += paddingValue * yBasis[b];
                        grad[1] += paddingValue * deriv[b];
                    }
                }
                if(grad[0]!=grad[0]) grad[0]=0;
                if(grad[1]!=grad[1]) grad[1]=0;
            }// mask

            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing Bilinear Gradient Computation...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineImageGradient3D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *resultGradientImage,
                                int *mask,
                                float paddingValue)
{
    size_t targetVoxelNumber = (size_t)resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = resultGradientImage->nt * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D cubic spline gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[3*t*targetVoxelNumber];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[targetVoxelNumber];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[targetVoxelNumber];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], c, Z, b, Y, a;
#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        FieldTYPE xBasis[4], yBasis[4], zBasis[4], xDeriv[4], yDeriv[4], zDeriv[4];
        FieldTYPE coeff, position[3], relative, world[3], grad[3];
        FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
        SourceTYPE *zPointer, *yzPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, zBasis, xDeriv, yDeriv, zDeriv, relative, grad, coeff, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, paddingValue, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY, resultGradientPtrZ, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;
            grad[2]=0.0;

            if((*maskPtr++)>-1){

                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                previous[2] = static_cast<int>(reg_floor(position[2]));

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                interpolantCubicSpline<FieldTYPE>(relative, xBasis, xDeriv);

                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                interpolantCubicSpline<FieldTYPE>(relative, yBasis, yDeriv);

                // basis values along the z axis
                relative=position[2]-(FieldTYPE)previous[2];
                interpolantCubicSpline<FieldTYPE>(relative, zBasis, zDeriv);

                previous[0]--;previous[1]--;previous[2]--;

                for(c=0; c<4; c++){
                    Z = previous[2]+c;
                    if(-1<Z && Z<floatingImage->nz){
                        zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                        xxTempNewValue=0.0;
                        yyTempNewValue=0.0;
                        zzTempNewValue=0.0;
                        for(b=0; b<4; b++){
                            Y= previous[1]+b;
                            yzPointer = &zPointer[Y*floatingImage->nx];
                            if(-1<Y && Y<floatingImage->ny){
                                xyzPointer = &yzPointer[previous[0]];
                                xTempNewValue=0.0;
                                yTempNewValue=0.0;
                                for(a=0; a<4; a++){
                                    if(-1<(previous[0]+a) && (previous[0]+a)<floatingImage->nx){
                                        coeff = *xyzPointer;
                                        xTempNewValue +=  coeff * xDeriv[a];
                                        yTempNewValue +=  coeff * xBasis[a];
                                    } // previous[0]+a in range
                                    else{
                                        xTempNewValue +=  paddingValue * xDeriv[a];
                                        yTempNewValue +=  paddingValue * xBasis[a];
                                    }
                                    xyzPointer++;
                                } // a
                                xxTempNewValue += xTempNewValue * yBasis[b];
                                yyTempNewValue += yTempNewValue * yDeriv[b];
                                zzTempNewValue += yTempNewValue * yBasis[b];
                            } // Y in range
                            else{
                                xxTempNewValue += paddingValue * yBasis[b];
                                yyTempNewValue += paddingValue * yDeriv[b];
                                zzTempNewValue += paddingValue * yBasis[b];
                            }
                        } // b
                        grad[0] += xxTempNewValue * zBasis[c];
                        grad[1] += yyTempNewValue * zBasis[c];
                        grad[2] += zzTempNewValue * zDeriv[c];
                    } // Z in range
                    else{
                        grad[0] += paddingValue * zBasis[c];
                        grad[1] += paddingValue * zBasis[c];
                        grad[2] += paddingValue * zDeriv[c];
                    }
                } // c

                grad[0]=grad[0]==grad[0]?grad[0]:0.0;
                grad[1]=grad[1]==grad[1]?grad[1]:0.0;
                grad[2]=grad[2]==grad[2]?grad[2]:0.0;
            } // outside of the mask

            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];
            resultGradientPtrZ[index] = (GradientTYPE)grad[2];

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing 3D Cubic Spline Gradient Computation...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineImageGradient2D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *resultGradientImage,
                                int *mask)
{
    size_t targetVoxelNumber = (size_t)resultGradientImage->nx*resultGradientImage->ny;
    size_t sourceVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Compute the resolution of the progress bar
    unsigned long iProgressStep  = 1;
    unsigned long nProgressSteps = resultGradientImage->nt * targetVoxelNumber;
    unsigned long progressUnit   = (unsigned long)ceil((float)nProgressSteps / 100.0f);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D cubic spline gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[t*2*targetVoxelNumber];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

#ifdef _WIN32
        int  index;
#else
        size_t  index;
#endif
        int previous[2], b, Y, a; bool bg;
        FieldTYPE xBasis[4], yBasis[4], xDeriv[4], yDeriv[4];
        FieldTYPE coeff, position[3], relative, world[3], grad[2];
        FieldTYPE xTempNewValue, yTempNewValue;
        SourceTYPE *yPointer, *xyPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, xDeriv, yDeriv, relative, grad, coeff, bg, \
    a, b, Y, yPointer, xyPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY, \
    iProgressStep, nProgressSteps, progressUnit)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];

                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix->m[0][0] + world[1]*sourceIJKMatrix->m[0][1] +
                        sourceIJKMatrix->m[0][3];
                position[1] = world[0]*sourceIJKMatrix->m[1][0] + world[1]*sourceIJKMatrix->m[1][1] +
                        sourceIJKMatrix->m[1][3];

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));
                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, xBasis, xDeriv);
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                relative=relative>0?relative:0;
                interpolantCubicSpline<FieldTYPE>(relative, yBasis, yDeriv);

                previous[0]--;previous[1]--;

                bg=false;
                for(b=0; b<4; b++){
                    Y= previous[1]+b;
                    yPointer = &sourceIntensity[Y*floatingImage->nx];
                    if(-1<Y && Y<floatingImage->ny){
                        xyPointer = &yPointer[previous[0]];
                        xTempNewValue=0.0;
                        yTempNewValue=0.0;
                        for(a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<floatingImage->nx){
                                coeff = (FieldTYPE)*xyPointer;
                                xTempNewValue +=  coeff * xDeriv[a];
                                yTempNewValue +=  coeff * xBasis[a];
                            }
                            else bg=true;
                            xyPointer++;
                        }
                        grad[0] += (xTempNewValue * yBasis[b]);
                        grad[1] += (yTempNewValue * yDeriv[b]);
                    }
                    else bg=true;
                }

                if(bg==true){
                    grad[0]=0.0;
                    grad[1]=0.0;
                }
            }
            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];

            // Announce the progress via CLI
            if (iProgressStep % progressUnit == 0)
                progressXML(100 * iProgressStep / nProgressSteps, "Performing 2D Cubic Spline Gradient Computation...");

            // Increment the progress counter
            iProgressStep++;
        }
    }
}
/* *************************************************************** */
template <class FieldTYPE, class SourceTYPE, class GradientTYPE>
void reg_getImageGradient3(nifti_image *floatingImage,
                           nifti_image *resultGradientImage,
                           nifti_image *deformationField,
                           int *mask,
                           int interp,
                           float paddingValue,
                           int *dtIndicies,
						   mat33 *jacMat,
						   nifti_image *warpedImage = NULL
                           )
{
	// The floating image data is copied in case one deal with DTI
	void *originalFloatingData=NULL;
    // The DTI are logged
    reg_dti_resampling_preprocessing<SourceTYPE>(floatingImage,
												 &originalFloatingData,
                                                 dtIndicies);
    /* The deformation field contains the position in the real world */
    if(interp==3){
        if(deformationField->nz>1){
            CubicSplineImageGradient3D
                    <SourceTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask,
                                                        paddingValue);
        }
        else{
            CubicSplineImageGradient2D
                    <SourceTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask);
        }
    }
    else{ // trilinear interpolation [ by default ]
        if(deformationField->nz>1){
            TrilinearImageGradient
                    <SourceTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask,
                                                        paddingValue);
        }
        else{
            BilinearImageGradient
                    <SourceTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask,
                                                        paddingValue);
        }
    }
    // The temporary logged floating array is deleted
	if(originalFloatingData!=NULL){
		free(floatingImage->data);
		floatingImage->data=originalFloatingData;
		originalFloatingData=NULL;
    }
    // The interpolated tensors are reoriented and exponentiated
    reg_dti_resampling_postprocessing<SourceTYPE>(resultGradientImage,
                                                  mask,
                                                  jacMat,
												  dtIndicies,
												  warpedImage
												  );
}
/* *************************************************************** */
template <class FieldTYPE, class SourceTYPE>
void reg_getImageGradient2(nifti_image *floatingImage,
                           nifti_image *resultGradientImage,
                           nifti_image *deformationField,
                           int *mask,
                           int interp,
                           float paddingValue,
                           int *dtIndicies,
						   mat33 *jacMat,
						   nifti_image *warpedImage
                           )
{
    switch(resultGradientImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient3<FieldTYPE,SourceTYPE,float>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient3<FieldTYPE,SourceTYPE,double>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
        return;
    }
}
/* *************************************************************** */
template <class FieldTYPE>
void reg_getImageGradient1(nifti_image *floatingImage,
                           nifti_image *resultGradientImage,
                           nifti_image *deformationField,
                           int *mask,
                           int interp,
                           float paddingValue,
                           int *dtIndicies,
						   mat33 *jacMat,
						   nifti_image *warpedImage
                           )
{
    switch(floatingImage->datatype){
    case NIFTI_TYPE_UINT8:
        reg_getImageGradient2<FieldTYPE,unsigned char>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_INT8:
        reg_getImageGradient2<FieldTYPE,char>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_UINT16:
        reg_getImageGradient2<FieldTYPE,unsigned short>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_INT16:
        reg_getImageGradient2<FieldTYPE,short>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_UINT32:
        reg_getImageGradient2<FieldTYPE,unsigned int>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_INT32:
        reg_getImageGradient2<FieldTYPE,int>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient2<FieldTYPE,float>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient2<FieldTYPE,double>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
        return;
    }
}
/* *************************************************************** */
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *resultGradientImage,
                          nifti_image *deformationField,
                          int *mask,
                          int interp,
                          float paddingValue,
                          bool *dti_timepoint,
						  mat33 *jacMat,
						  nifti_image *warpedImage
                          )
{
    // a mask array is created if no mask is specified
    bool MrPropreRule=false;
    if(mask==NULL){
        // voxels in the backgreg_round are set to -1 so 0 will do the job here
        mask=(int *)calloc(deformationField->nx*deformationField->ny*deformationField->nz,sizeof(int));
        MrPropreRule=true;
    }

    // Check if the dimension are correct
    if(floatingImage->nt != resultGradientImage->nt){
        printf("[NiftyReg ERROR] reg_getImageGradient\tThe source and result images have different dimension along the time axis\n");
        printf("[NiftyReg ERROR] reg_getImageGradient\tNothing has been done\n");
        return;
    }

    // Define the DTI indices if required
    int dtIndicies[6];
    for(int i=0;i<6;++i) dtIndicies[i]=-1;
    if(dti_timepoint!=NULL){

        if(jacMat==NULL){
            printf("[NiftyReg ERROR] reg_resampleImage\tDTI resampling\n");
            printf("[NiftyReg ERROR] reg_resampleImage\tNo Jacobian matrix array has been provided\n");
            reg_exit(1);
        }
        int j=0;
        for(int i=0;i<floatingImage->nt;++i){
            if(dti_timepoint[i]==true)
                dtIndicies[j++]=i;
        }
        if((floatingImage->nz>1 && j!=6) && (floatingImage->nz==1 && j!=3)){
            printf("[NiftyReg ERROR] reg_resampleImage\tUnexpected number of DTI components\n");
            printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
            reg_exit(1);
        }
    }

    switch(deformationField->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient1<float>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient1<double>
				(floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue,dtIndicies,jacMat, warpedImage);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getImageGradient\tDeformation field pixel type unsupported.\n");
        break;
    }
    if(MrPropreRule==true) free(mask);
}
/* *************************************************************** */
/* *************************************************************** */
nifti_image *reg_makeIsotropic(nifti_image *img,
                               int inter)
{
    // Get the smallest voxel size
    float smallestPixDim=img->pixdim[1];
    for(size_t i=2;i<4;++i)
        if(i<static_cast<size_t>(img->dim[0]+2))
            smallestPixDim=img->pixdim[i]<smallestPixDim?img->pixdim[i]:smallestPixDim;
    // Define the size of the new image
    int newDim[8];
    for(size_t i=0;i<8;++i) newDim[i]=img->dim[i];
    for(size_t i=1;i<4;++i){
        if(i<static_cast<size_t>(img->dim[0]+1))
            newDim[i]=(int)ceilf(img->dim[i]*img->pixdim[i]/smallestPixDim);
    }
    // Create the new image
    nifti_image *newImg=nifti_make_new_nim(newDim,img->datatype,true);
    newImg->pixdim[1]=newImg->dx=smallestPixDim;
    newImg->pixdim[2]=newImg->dy=smallestPixDim;
    newImg->pixdim[3]=newImg->dz=smallestPixDim;
    newImg->qform_code=img->qform_code;
    newImg->sform_code=img->sform_code;
    // Update the qform matrix
    newImg->qfac=img->qfac;
    newImg->quatern_b=img->quatern_b;
    newImg->quatern_c=img->quatern_c;
    newImg->quatern_d=img->quatern_d;
    newImg->qoffset_x=img->qoffset_x+smallestPixDim/2.f-img->dx/2.f;
    newImg->qoffset_y=img->qoffset_y+smallestPixDim/2.f-img->dy/2.f;
    newImg->qoffset_z=img->qoffset_z+smallestPixDim/2.f-img->dz/2.f;
    newImg->qto_xyz=nifti_quatern_to_mat44(newImg->quatern_b,
                                           newImg->quatern_c,
                                           newImg->quatern_d,
                                           newImg->qoffset_x,
                                           newImg->qoffset_y,
                                           newImg->qoffset_z,
                                           smallestPixDim,
                                           smallestPixDim,
                                           smallestPixDim,
                                           newImg->qfac);
    newImg->qto_ijk=nifti_mat44_inverse(newImg->qto_xyz);
    if(newImg->sform_code>0){
        // Compute the new sform
        float scalingRatio[3];
        scalingRatio[0]= newImg->dx / img->dx;
        scalingRatio[1]= newImg->dy / img->dy;
        scalingRatio[2]= newImg->dz / img->dz;
        newImg->sto_xyz.m[0][0]=img->sto_xyz.m[0][0] * scalingRatio[0];
        newImg->sto_xyz.m[1][0]=img->sto_xyz.m[1][0] * scalingRatio[0];
        newImg->sto_xyz.m[2][0]=img->sto_xyz.m[2][0] * scalingRatio[0];
        newImg->sto_xyz.m[3][0]=img->sto_xyz.m[3][0];
        newImg->sto_xyz.m[0][1]=img->sto_xyz.m[0][1] * scalingRatio[1];
        newImg->sto_xyz.m[1][1]=img->sto_xyz.m[1][1] * scalingRatio[1];
        newImg->sto_xyz.m[2][1]=img->sto_xyz.m[2][1] * scalingRatio[1];
        newImg->sto_xyz.m[3][1]=img->sto_xyz.m[3][1];
        newImg->sto_xyz.m[0][2]=img->sto_xyz.m[0][2] * scalingRatio[2];
        newImg->sto_xyz.m[1][2]=img->sto_xyz.m[1][2] * scalingRatio[2];
        newImg->sto_xyz.m[2][2]=img->sto_xyz.m[2][2] * scalingRatio[2];
        newImg->sto_xyz.m[3][2]=img->sto_xyz.m[3][2];
        newImg->sto_xyz.m[0][3]=img->sto_xyz.m[0][3]+smallestPixDim/2.f-img->dx/2.f;
        newImg->sto_xyz.m[1][3]=img->sto_xyz.m[1][3]+smallestPixDim/2.f-img->dy/2.f;
        newImg->sto_xyz.m[2][3]=img->sto_xyz.m[2][3]+smallestPixDim/2.f-img->dz/2.f;
        newImg->sto_xyz.m[3][3]=img->sto_xyz.m[3][3];
        newImg->sto_ijk=nifti_mat44_inverse(newImg->sto_xyz);
    }
    reg_checkAndCorrectDimension(newImg);
    // Create a deformation field
    nifti_image *def=nifti_copy_nim_info(newImg);
    def->dim[0]=def->ndim=5;
    def->dim[4]=def->nt=1;
    def->pixdim[4]=def->dt=1.0;
    if(newImg->nz==1)
        def->dim[5]=def->nu=2;
    else def->dim[5]=def->nu=3;
    def->pixdim[5]=def->du=1.0;
    def->dim[6]=def->nv=1;
    def->pixdim[6]=def->dv=1.0;
    def->dim[7]=def->nw=1;
    def->pixdim[7]=def->dw=1.0;
    def->nvox =
            (size_t)def->nx *
            (size_t)def->ny *
            (size_t)def->nz *
            (size_t)def->nt *
            (size_t)def->nu;
    def->nbyper = sizeof(float);
    def->datatype = NIFTI_TYPE_FLOAT32;
    def->data = (void *)calloc(def->nvox,def->nbyper);
    // Fill the deformation field with an identity transformation
    reg_getDeformationFromDisplacement(def);
    // resample the original image into the space of the new image
    reg_resampleImage(img,newImg,def,NULL,inter,0.f);
    nifti_set_filenames(newImg,"tempIsotropicImage",0,0);
    nifti_image_free(def);
    return newImg;
}
/* *************************************************************** */
/* *************************************************************** */

#endif
