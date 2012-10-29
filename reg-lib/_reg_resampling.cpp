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
template<class SourceTYPE, class FieldTYPE>
void CubicSplineResampleImage3D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *warpedImage,
                                int *mask)
{
    // I here assume a zero padding by default

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int resultVoxelNumber = warpedImage->nx*warpedImage->ny*warpedImage->nz;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny*floatingImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[resultVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[resultVoxelNumber];


    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D Cubic spline resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*resultVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[4], yBasis[4], zBasis[4], relative;
        int a, b, c, Y, Z, previous[3], index;
        SourceTYPE *zPointer, *yzPointer, *xyzPointer;
        FieldTYPE xTempNewValue, yTempNewValue, intensity, world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, resultIntensity, resultVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage)
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
                    if(-1<Z && Z<floatingImage->nz){
                        zPointer = &sourceIntensity[Z*floatingImage->nx*floatingImage->ny];
                        yTempNewValue=0.0;
                        for(b=0; b<4; b++){
                            Y= previous[1]+b;
                            yzPointer = &zPointer[Y*floatingImage->nx];
                            if(-1<Y && Y<floatingImage->ny){
                                xyzPointer = &yzPointer[previous[0]];
                                xTempNewValue=0.0;
                                for(a=0; a<4; a++){
                                    if(-1<(previous[0]+a) && (previous[0]+a)<floatingImage->nx){
                                        xTempNewValue +=  (FieldTYPE)*xyzPointer * xBasis[a];
                                    }
                                    xyzPointer++;
                                }
                                yTempNewValue += (xTempNewValue * yBasis[b]);
                            }
                        }
                        intensity += yTempNewValue * zBasis[c];
                    }
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
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void CubicSplineResampleImage2D(  nifti_image *floatingImage,
                                  nifti_image *deformationField,
                                  nifti_image *warpedImage,
                                  int *mask)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int targetVoxelNumber = warpedImage->nx*warpedImage->ny;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=floatingImage->sto_ijk;
    else sourceIJKMatrix=floatingImage->qto_ijk;

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D Cubic spline resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[4], yBasis[4], relative;
        int a, b, Y, previous[2], index;
        SourceTYPE *yPointer, *xyPointer;
        FieldTYPE xTempNewValue, intensity, world[2], position[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, relative, \
    a, b, Y, yPointer, xyPointer, xTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage)
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
                    if(-1<Y && Y<floatingImage->ny){
                        xyPointer = &yPointer[previous[0]];
                        xTempNewValue=0.0;
                        for(a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<floatingImage->nx){
                                xTempNewValue +=  (FieldTYPE)*xyPointer * xBasis[a];
                            }
                            xyPointer++;
                        }
                        intensity += (xTempNewValue * yBasis[b]);
                    }
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
    int targetVoxelNumber = warpedImage->nx*warpedImage->ny*warpedImage->nz;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny*floatingImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];
    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[2], yBasis[2], zBasis[2], relative;
        int a, b, c, X, Y, Z, previous[3], index;
        SourceTYPE *zPointer, *xyzPointer;
        FieldTYPE xTempNewValue, yTempNewValue, intensity, world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, X, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue)
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
                relative=relative>0?relative:0;
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
    int targetVoxelNumber = warpedImage->nx*warpedImage->ny;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny;
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    for(int t=0; t<warpedImage->nt*warpedImage->nu;t++){

#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D linear resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[2], yBasis[2], relative;
        int a, b, X, Y, previous[3], index;
        SourceTYPE *xyPointer;
        FieldTYPE xTempNewValue, intensity, world[2], position[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, relative, \
    a, b, X, Y, xyPointer, xTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue)
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
    int targetVoxelNumber = warpedImage->nx*warpedImage->ny*warpedImage->nz;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny*floatingImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

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
        int index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue)
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
    int targetVoxelNumber = warpedImage->nx*warpedImage->ny;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

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
        int index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, paddingValue)
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

                previous[0] = static_cast<int>(reg_floor(position[0]));
                previous[1] = static_cast<int>(reg_floor(position[1]));

                if( -1<previous[1] && previous[1]<floatingImage->ny &&
                        -1<previous[0] && previous[0]<floatingImage->nx){
                    intensity = sourceIntensity[previous[1]*floatingImage->nx+previous[0]];
                    resultIntensity[index]=intensity;
                }
                else resultIntensity[index]=(SourceTYPE)paddingValue;
            }
            else resultIntensity[index]=(SourceTYPE)paddingValue;
        }
    }
}
/* *************************************************************** */

/** This function resample a source image into the referential
 * of a target image by applying an affine transformation and
 * a deformation field. The affine transformation has to be in
 * real coordinate and the deformation field is in mm in the space
 * of the target image.
 * interp can be either 0, 1 or 3 meaning nearest neighbor, linear
 * or cubic spline interpolation.
 * every voxel which is not fully in the source image takes the
 * backgreg_round value.
 */
template <class FieldTYPE, class SourceTYPE>
void reg_resampleImage2(nifti_image *floatingImage,
                        nifti_image *warpedImage,
                        nifti_image *deformationFieldImage,
                        int *mask,
                        int interp,
                        FieldTYPE paddingValue
                        )
{
    /* The deformation field contains the position in the real world */
    if(interp==3){
        if(deformationFieldImage->nz>1){
            CubicSplineResampleImage3D<SourceTYPE,FieldTYPE>(floatingImage,
                                                             deformationFieldImage,
                                                             warpedImage,
                                                             mask);
        }
        else
        {
            CubicSplineResampleImage2D<SourceTYPE,FieldTYPE>(floatingImage,
                                                             deformationFieldImage,
                                                             warpedImage,
                                                             mask);
        }
    }
    else if(interp==0){ // Nearest neighbor interpolation
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
}

/* *************************************************************** */
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       nifti_image *deformationField,
                       int *mask,
                       int interp,
                       float paddingValue)
{
    if(floatingImage->datatype != warpedImage->datatype){
        printf("[NiftyReg ERROR] reg_resampleImage\tSource and result image should have the same data type\n");
        printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
        exit(1);
    }

    if(floatingImage->nt != warpedImage->nt){
        printf("[NiftyReg ERROR] reg_resampleImage\tThe source and result images have different dimension along the time axis\n");
        printf("[NiftyReg ERROR] reg_resampleImage\tNothing has been done\n");
        exit(1);
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
                                                    paddingValue);
            break;
        case NIFTI_TYPE_INT8:
            reg_resampleImage2<float,char>(floatingImage,
                                           warpedImage,
                                           deformationField,
                                           mask,
                                           interp,
                                           paddingValue);
            break;
        case NIFTI_TYPE_UINT16:
            reg_resampleImage2<float,unsigned short>(floatingImage,
                                                     warpedImage,
                                                     deformationField,
                                                     mask,
                                                     interp,
                                                     paddingValue);
            break;
        case NIFTI_TYPE_INT16:
            reg_resampleImage2<float,short>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            mask,
                                            interp,
                                            paddingValue);
            break;
        case NIFTI_TYPE_UINT32:
            reg_resampleImage2<float,unsigned int>(floatingImage,
                                                   warpedImage,
                                                   deformationField,
                                                   mask,
                                                   interp,
                                                   paddingValue);
            break;
        case NIFTI_TYPE_INT32:
            reg_resampleImage2<float,int>(floatingImage,
                                          warpedImage,
                                          deformationField,
                                          mask,
                                          interp,
                                          paddingValue);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_resampleImage2<float,float>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            mask,
                                            interp,
                                            paddingValue);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleImage2<float,double>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             mask,
                                             interp,
                                             paddingValue);
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
                                                     paddingValue);
            break;
        case NIFTI_TYPE_INT8:
            reg_resampleImage2<double,char>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            mask,
                                            interp,
                                            paddingValue);
            break;
        case NIFTI_TYPE_UINT16:
            reg_resampleImage2<double,unsigned short>(floatingImage,
                                                      warpedImage,
                                                      deformationField,
                                                      mask,
                                                      interp,
                                                      paddingValue);
            break;
        case NIFTI_TYPE_INT16:
            reg_resampleImage2<double,short>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             mask,
                                             interp,
                                             paddingValue);
            break;
        case NIFTI_TYPE_UINT32:
            reg_resampleImage2<double,unsigned int>(floatingImage,
                                                    warpedImage,
                                                    deformationField,
                                                    mask,
                                                    interp,
                                                    paddingValue);
            break;
        case NIFTI_TYPE_INT32:
            reg_resampleImage2<double,int>(floatingImage,
                                           warpedImage,
                                           deformationField,
                                           mask,
                                           interp,
                                           paddingValue);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_resampleImage2<double,float>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             mask,
                                             interp,
                                             paddingValue);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleImage2<double,double>(floatingImage,
                                              warpedImage,
                                              deformationField,
                                              mask,
                                              interp,
                                              paddingValue);
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
    unsigned int floatingPixelNumber = floatingImage->nx*floatingImage->ny;
    unsigned int warpedPixelNumber = warpedImage->nx*warpedImage->ny;
    DTYPE *floatingIntensityX = static_cast<DTYPE *>(floatingImage->data);
    DTYPE *floatingIntensityY = &floatingIntensityX[floatingPixelNumber];
    DTYPE *warpedIntensityX = static_cast<DTYPE *>(warpedImage->data);
    DTYPE *warpedIntensityY = &warpedIntensityX[warpedPixelNumber];
    DTYPE *deformationFieldPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *deformationFieldPtrY = &deformationFieldPtrX[deformationField->nx*deformationField->ny];

    // Extract the relevant affine matrices
    mat44 *warped_voxel_to_mm = &warpedImage->qto_xyz;
    if(warpedImage->sform_code>1)
        warped_voxel_to_mm = &warpedImage->sto_xyz;
    mat44 *floating_mm_to_voxel = &floatingImage->qto_ijk;
    if(floatingImage->sform_code>1)
        floating_mm_to_voxel = &floatingImage->sto_ijk;
    mat44 *def_mm_to_voxel = &deformationField->qto_ijk;
    if(deformationField->sform_code>1)
        def_mm_to_voxel = &deformationField->sto_ijk;
    mat44 warped_to_def_voxel = reg_mat44_mul(def_mm_to_voxel,warped_voxel_to_mm);

    // Some usefull variable
    mat33 jacMat;
    jacMat.m[0][0]=1;jacMat.m[0][1]=0;jacMat.m[0][2]=0;
    jacMat.m[1][0]=0;jacMat.m[1][1]=1;jacMat.m[1][2]=0;
    jacMat.m[2][0]=0;jacMat.m[2][1]=0;jacMat.m[2][2]=1;

    // Loop over all pixel
    int warpedIndex=0;
    for(int y=0; y<warpedImage->ny; ++y){
        for(int x=0; x<warpedImage->nx; ++x){
            warpedIntensityX[warpedIndex]=paddingValue;
            warpedIntensityY[warpedIndex]=paddingValue;
            // Compte the index in the deformation field
            DTYPE xDef = x*warped_to_def_voxel.m[0][0] +
                    y*warped_to_def_voxel.m[0][1] +
                    warped_to_def_voxel.m[0][3];
            DTYPE yDef = x*warped_to_def_voxel.m[1][0] +
                    y*warped_to_def_voxel.m[1][1] +
                    warped_to_def_voxel.m[1][3];
            // Extract the corresponding position using linear interpolation
            int anteY = static_cast<int>(reg_floor(yDef));
            if(anteY>-1 && anteY<deformationField->ny-1){
                int anteX = static_cast<int>(reg_floor(xDef));
                if(anteX>-1 && anteX<deformationField->nx-1){
                    DTYPE xPos=0, yPos=0;
                    jacMat.m[0][0]=0;jacMat.m[0][1]=0;
                    jacMat.m[1][0]=0;jacMat.m[1][1]=0;
                    DTYPE basisX[2], basisY[2], deriv[2]={-1,1};
                    basisX[1]=xDef-anteX;
                    basisY[1]=yDef-anteY;
                    basisX[0]=1.f-basisX[1];
                    basisY[0]=1.f-basisY[1];
                    for(unsigned int b=0;b<2;++b){
                        for(unsigned int a=0;a<2;++a){
                            unsigned int defIndex = (anteY+b)*deformationField->nx+anteX+a;
                            xPos += deformationFieldPtrX[defIndex] * basisX[a] * basisY[b];
                            yPos += deformationFieldPtrY[defIndex] * basisX[a] * basisY[b];
                            jacMat.m[0][0] += deformationFieldPtrX[defIndex] * deriv[a] * basisY[b];
                            jacMat.m[0][1] += deformationFieldPtrY[defIndex] * deriv[a] * basisY[b];
                            jacMat.m[1][0] += deformationFieldPtrX[defIndex] * basisX[a] * deriv[b];
                            jacMat.m[1][1] += deformationFieldPtrY[defIndex] * basisX[a] * deriv[b];
                        }
                    }
                    // Extract the corresponding coordinates in the floating image
                    DTYPE xFloCoord = xPos * floating_mm_to_voxel->m[0][0] +
                            yPos * floating_mm_to_voxel->m[0][1] +
                            floating_mm_to_voxel->m[0][3];
                    DTYPE yFloCoord = xPos * floating_mm_to_voxel->m[1][0] +
                            yPos * floating_mm_to_voxel->m[1][1] +
                            floating_mm_to_voxel->m[1][3];
                    // Extract the floating value using bilinear interpolation
                    int anteIntX = static_cast<int>(reg_floor(xFloCoord));
                    int anteIntY = static_cast<int>(reg_floor(yFloCoord));
                    DTYPE val_x=0, val_y=0;
                    basisX[1]=fabs(xFloCoord-(DTYPE)anteIntX);
                    basisY[1]=fabs(yFloCoord-(DTYPE)anteIntY);
                    basisX[0]=1.f-basisX[1];
                    basisY[0]=1.f-basisY[1];
                    for(int b=0;b<2;++b){
                        int B=anteIntY+b;
                        if(B>-1 && B<floatingImage->ny){
                            for(int a=0;a<2;++a){
                                int A=anteIntX+a;
                                if(A>-1 && A<floatingImage->nx){
                                    unsigned int floIndex = (B)*floatingImage->nx+A;
                                    val_x += floatingIntensityX[floIndex] * basisX[a] * basisY[b];
                                    val_y += floatingIntensityY[floIndex] * basisX[a] * basisY[b];
                                } // anteX not in the floating image space
                                else{
                                    val_x += paddingValue * basisX[a] * basisY[b];
                                    val_y += paddingValue * basisX[a] * basisY[b];
                                }
                            } // a
                        } // anteY not in the floating image space
                        else{
                            val_x += paddingValue * basisY[b];
                            val_y += paddingValue * basisY[b];
                        }
                    } // b
                    warpedIntensityX[warpedIndex]=jacMat.m[0][0]*val_x+jacMat.m[0][1]*val_y;
                    warpedIntensityY[warpedIndex]=jacMat.m[1][0]*val_x+jacMat.m[1][1]*val_y;
                } // anteX not in deformation field space
            } // anteY not in deformation field space
            ++warpedIndex;
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
    size_t floatingVoxelNumber = floatingImage->nx*floatingImage->ny*floatingImage->nz;
    size_t warpedVoxelNumber = warpedImage->nx*warpedImage->ny*warpedImage->nz;
    DTYPE *floatingIntensityX = static_cast<DTYPE *>(floatingImage->data);
    DTYPE *floatingIntensityY = &floatingIntensityX[floatingVoxelNumber];
    DTYPE *floatingIntensityZ = &floatingIntensityY[floatingVoxelNumber];
    DTYPE *warpedIntensityX = static_cast<DTYPE *>(warpedImage->data);
    DTYPE *warpedIntensityY = &warpedIntensityX[warpedVoxelNumber];
    DTYPE *warpedIntensityZ = &warpedIntensityY[warpedVoxelNumber];
    DTYPE *deformationFieldPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *deformationFieldPtrY = &deformationFieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];
    DTYPE *deformationFieldPtrZ = &deformationFieldPtrY[deformationField->nx*deformationField->ny*deformationField->nz];

    // Extract the relevant affine matrices
    mat44 *warped_voxel_to_mm = &warpedImage->qto_xyz;
    if(warpedImage->sform_code>1)
        warped_voxel_to_mm = &warpedImage->sto_xyz;
    mat44 *floating_mm_to_voxel = &floatingImage->qto_ijk;
    if(floatingImage->sform_code>1)
        floating_mm_to_voxel = &floatingImage->sto_ijk;
    mat44 *def_mm_to_voxel = &deformationField->qto_ijk;
    if(deformationField->sform_code>1)
        def_mm_to_voxel = &deformationField->sto_ijk;
    mat44 warped_to_def_voxel = reg_mat44_mul(def_mm_to_voxel,warped_voxel_to_mm);

    // Some usefull variable
    mat33 jacMat;
    reg_mat33_eye(&jacMat);

    // Loop over all voxel
    int warpedIndex=0;
    for(int z=0; z<warpedImage->nz; ++z){
        for(int y=0; y<warpedImage->ny; ++y){
            for(int x=0; x<warpedImage->nx; ++x){
                warpedIntensityX[warpedIndex]=0;
                warpedIntensityY[warpedIndex]=0;
                warpedIntensityZ[warpedIndex]=0;
                // Compte the index in the deformation field
                DTYPE xDef = x*warped_to_def_voxel.m[0][0] +
                        y*warped_to_def_voxel.m[0][1] +
                        z*warped_to_def_voxel.m[0][2] +
                        warped_to_def_voxel.m[0][3];
                DTYPE yDef = x*warped_to_def_voxel.m[1][0] +
                        y*warped_to_def_voxel.m[1][1] +
                        z*warped_to_def_voxel.m[1][2] +
                        warped_to_def_voxel.m[1][3];
                DTYPE zDef = x*warped_to_def_voxel.m[2][0] +
                        y*warped_to_def_voxel.m[2][1] +
                        z*warped_to_def_voxel.m[2][2] +
                        warped_to_def_voxel.m[2][3];
                // Extract the corresponding position using linear interpolation
                int anteZ = static_cast<int>(reg_floor(zDef));
                if(anteZ>-1 && anteZ<deformationField->nz-1){
                    int anteY = static_cast<int>(reg_floor(yDef));
                    if(anteY>-1 && anteY<deformationField->ny-1){
                        int anteX = static_cast<int>(reg_floor(xDef));
                        if(anteX>-1 && anteX<deformationField->nx-1){
                            DTYPE xPos=0, yPos=0, zPos=0;
                            jacMat.m[0][0]=jacMat.m[0][1]=jacMat.m[0][2]=0;
                            jacMat.m[1][0]=jacMat.m[1][1]=jacMat.m[1][2]=0;
                            jacMat.m[2][0]=jacMat.m[2][1]=jacMat.m[2][2]=0;
                            DTYPE basisX[2], basisY[2], basisZ[2], deriv[2]={-1,1};
                            basisX[1]=xDef-anteX;
                            basisY[1]=yDef-anteY;
                            basisZ[1]=zDef-anteZ;
                            basisX[0]=1.f-basisX[1];
                            basisY[0]=1.f-basisY[1];
                            basisZ[0]=1.f-basisZ[1];
                            for(unsigned int c=0;c<2;++c){
                                for(unsigned int b=0;b<2;++b){
                                    for(unsigned int a=0;a<2;++a){
                                        unsigned int defIndex = ((anteZ+c)*deformationField->ny+anteY+b)*deformationField->nx+anteX+a;
                                        xPos += deformationFieldPtrX[defIndex] * basisX[a] * basisY[b] * basisZ[c];
                                        yPos += deformationFieldPtrY[defIndex] * basisX[a] * basisY[b] * basisZ[c];
                                        zPos += deformationFieldPtrZ[defIndex] * basisX[a] * basisY[b] * basisZ[c];
                                        jacMat.m[0][0] += deformationFieldPtrX[defIndex] * deriv[a] * basisY[b] * basisZ[c];
                                        jacMat.m[0][1] += deformationFieldPtrY[defIndex] * deriv[a] * basisY[b] * basisZ[c];
                                        jacMat.m[0][2] += deformationFieldPtrZ[defIndex] * deriv[a] * basisY[b] * basisZ[c];
                                        jacMat.m[1][0] += deformationFieldPtrX[defIndex] * basisX[a] * deriv[b] * basisZ[c];
                                        jacMat.m[1][1] += deformationFieldPtrY[defIndex] * basisX[a] * deriv[b] * basisZ[c];
                                        jacMat.m[1][2] += deformationFieldPtrZ[defIndex] * basisX[a] * deriv[b] * basisZ[c];
                                        jacMat.m[2][0] += deformationFieldPtrX[defIndex] * basisX[a] * basisY[b] * deriv[c];
                                        jacMat.m[2][1] += deformationFieldPtrY[defIndex] * basisX[a] * basisY[b] * deriv[c];
                                        jacMat.m[2][2] += deformationFieldPtrZ[defIndex] * basisX[a] * basisY[b] * deriv[c];
                                    }
                                }
                            }
                            // Extract the corresponding coordinates in the floating image
                            DTYPE xFloCoord =
                                    xPos * floating_mm_to_voxel->m[0][0] +
                                    yPos * floating_mm_to_voxel->m[0][1] +
                                    zPos * floating_mm_to_voxel->m[0][2] +
                                    floating_mm_to_voxel->m[0][3];
                            DTYPE yFloCoord =
                                    xPos * floating_mm_to_voxel->m[1][0] +
                                    yPos * floating_mm_to_voxel->m[1][1] +
                                    zPos * floating_mm_to_voxel->m[1][2] +
                                    floating_mm_to_voxel->m[1][3];
                            DTYPE zFloCoord =
                                    xPos * floating_mm_to_voxel->m[2][0] +
                                    yPos * floating_mm_to_voxel->m[2][1] +
                                    zPos * floating_mm_to_voxel->m[2][2] +
                                    floating_mm_to_voxel->m[2][3];
                            // Extract the floating value using bilinear interpolation
                            int anteIntX = static_cast<int>(reg_floor(xFloCoord));
                            int anteIntY = static_cast<int>(reg_floor(yFloCoord));
                            int anteIntZ = static_cast<int>(reg_floor(zFloCoord));
                            DTYPE val_x=0, val_y=0, val_z=0;
                            basisX[1]=fabs(xFloCoord-(DTYPE)anteIntX);
                            basisY[1]=fabs(yFloCoord-(DTYPE)anteIntY);
                            basisZ[1]=fabs(zFloCoord-(DTYPE)anteIntZ);
                            basisX[0]=1.0-basisX[1];
                            basisY[0]=1.0-basisY[1];
                            basisZ[0]=1.0-basisZ[1];
                            for(int c=0;c<2;++c){
                                int C=anteIntZ+c;
                                if(C>-1 && C<floatingImage->nz){
                                    for(int b=0;b<2;++b){
                                        int B=anteIntY+b;
                                        if(B>-1 && B<floatingImage->ny){
                                            for(int a=0;a<2;++a){
                                                int A=anteIntX+a;
                                                if(A>-1 && A<floatingImage->nx){
                                                    unsigned int floIndex = (C*floatingImage->ny+B)*floatingImage->nx+A;
                                                    val_x += floatingIntensityX[floIndex] * basisX[a] * basisY[b] * basisZ[C];
                                                    val_y += floatingIntensityY[floIndex] * basisX[a] * basisY[b] * basisZ[C];
                                                    val_z += floatingIntensityZ[floIndex] * basisX[a] * basisY[b] * basisZ[C];
                                                } // anteIntX not in the floating image space
                                                else{
                                                    val_x += paddingValue * basisX[a] * basisY[b] * basisZ[C];
                                                    val_y += paddingValue * basisX[a] * basisY[b] * basisZ[C];
                                                    val_z += paddingValue * basisX[a] * basisY[b] * basisZ[C];
                                                }
                                            } // a
                                        } // anteIntY not in the floating image space
                                        else{
                                            val_x += paddingValue * basisY[b] * basisZ[C];
                                            val_y += paddingValue * basisY[b] * basisZ[C];
                                            val_z += paddingValue * basisY[b] * basisZ[C];
                                        }
                                    } // b
                                } // anteIntZ not in the floating image space
                                else{
                                    val_x += paddingValue * basisZ[C];
                                    val_y += paddingValue * basisZ[C];
                                    val_z += paddingValue * basisZ[C];
                                }
                            } // c
                            warpedIntensityX[warpedIndex]=jacMat.m[0][0]*val_x+jacMat.m[0][1]*val_y+jacMat.m[0][2]*val_z;
                            warpedIntensityY[warpedIndex]=jacMat.m[1][0]*val_x+jacMat.m[1][1]*val_y+jacMat.m[1][2]*val_z;
                            warpedIntensityZ[warpedIndex]=jacMat.m[2][0]*val_x+jacMat.m[2][1]*val_y+jacMat.m[2][2]*val_z;
                        } // anteX not in deformation field space
                    } // anteY not in deformation field space
                } // anteZ not in deformation field space
                ++warpedIndex;
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
    if(floatingImage->datatype!=warpedImage->datatype ||
            floatingImage->datatype!=deformationField->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_resampleGradient - Input images are expected to have the same type\n");
        exit(1);
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
        exit(1);
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
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny*floatingImage->nz;
    int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;
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

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[gradientOffSet];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], index, a, b, c, X, Y, Z;
        FieldTYPE position[3], xBasis[2], yBasis[2], zBasis[2];
        FieldTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        FieldTYPE relative, world[3], grad[3], coeff;
        FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
        SourceTYPE *zPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, zBasis, relative, grad, coeff, \
    a, b, c, X, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY, resultGradientPtrZ)
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
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=floatingImage->sto_ijk;
    else sourceIJKMatrix=floatingImage->qto_ijk;

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D linear gradient computation of volume number %i\n",t);
#endif
        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE position[3], xBasis[2], yBasis[2], relative, world[2], grad[2];
        FieldTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        FieldTYPE coeff, xTempNewValue, yTempNewValue;
        int previous[3], index, a, b, X, Y;
        SourceTYPE *xyPointer;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, relative, grad, coeff, \
    a, b, X, Y, xyPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY)
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
    size_t targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    size_t sourceVoxelNumber = floatingImage->nx*floatingImage->ny*floatingImage->nz;
    size_t gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

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


    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D cubic spline gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[gradientOffSet];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], index, c, Z, b, Y, a;
        FieldTYPE xBasis[4], yBasis[4], zBasis[4], xDeriv[4], yDeriv[4], zDeriv[4];
        FieldTYPE coeff, position[3], relative, world[3], grad[3];
        FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
        SourceTYPE *zPointer, *yzPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, zBasis, xDeriv, yDeriv, zDeriv, relative, grad, coeff, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY, resultGradientPtrZ)
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
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineImageGradient2D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *resultGradientImage,
                                int *mask,
                                float paddingValue)
{
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny;
    int sourceVoxelNumber = floatingImage->nx*floatingImage->ny;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(floatingImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(floatingImage->sform_code>0)
        sourceIJKMatrix=&(floatingImage->sto_ijk);
    else sourceIJKMatrix=&(floatingImage->qto_ijk);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D cubic spline gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], index, b, Y, a; bool bg;
        FieldTYPE xBasis[4], yBasis[4], xDeriv[4], yDeriv[4];
        FieldTYPE coeff, position[3], relative, world[3], grad[3];
        FieldTYPE xTempNewValue, yTempNewValue;
        SourceTYPE *yPointer, *xyPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, xDeriv, yDeriv, relative, grad, coeff, bg, \
    a, b, Y, yPointer, xyPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, floatingImage, resultGradientPtrX, resultGradientPtrY)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[index]=0.0;
            grad[index]=0.0;

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
                           float paddingValue)
{
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
                                                        mask,
                                                        paddingValue);
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
}
/* *************************************************************** */
template <class FieldTYPE, class SourceTYPE>
void reg_getImageGradient2(nifti_image *floatingImage,
                           nifti_image *resultGradientImage,
                           nifti_image *deformationField,
                           int *mask,
                           int interp,
                           float paddingValue
                           )
{
    switch(resultGradientImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient3<FieldTYPE,SourceTYPE,float>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient3<FieldTYPE,SourceTYPE,double>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
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
                           float paddingValue
                           )
{
    switch(floatingImage->datatype){
    case NIFTI_TYPE_UINT8:
        reg_getImageGradient2<FieldTYPE,unsigned char>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_INT8:
        reg_getImageGradient2<FieldTYPE,char>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_UINT16:
        reg_getImageGradient2<FieldTYPE,unsigned short>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_INT16:
        reg_getImageGradient2<FieldTYPE,short>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_UINT32:
        reg_getImageGradient2<FieldTYPE,unsigned int>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_INT32:
        reg_getImageGradient2<FieldTYPE,int>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient2<FieldTYPE,float>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient2<FieldTYPE,double>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
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
                          float paddingValue
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

    switch(deformationField->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient1<float>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient1<double>
                (floatingImage,resultGradientImage,deformationField,mask,interp,paddingValue);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getImageGradient\tDeformation field pixel type unsupported.\n");
        break;
    }
    if(MrPropreRule==true) free(mask);
}
/* *************************************************************** */
/* *************************************************************** */

#endif
