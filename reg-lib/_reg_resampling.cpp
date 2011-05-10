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

// No round() function available in windows.
#ifdef _WINDOWS
template<class DTYPE>
int round(DTYPE x)
{
    return static_cast<int>(x > 0.0 ? x + 0.5 : x - 0.5);
}
#endif

/* *************************************************************** */
template <class PrecisionTYPE>
void interpolantCubicSpline(PrecisionTYPE ratio, PrecisionTYPE *basis)
{
    if(ratio<0.0) ratio=0.0; //rounding error
    PrecisionTYPE FF= ratio*ratio;
    basis[0] = (PrecisionTYPE)((ratio * ((2.0-ratio)*ratio - 1.0))/2.0);
    basis[1] = (PrecisionTYPE)((FF * (3.0*ratio-5.0) + 2.0)/2.0);
    basis[2] = (PrecisionTYPE)((ratio * ((4.0-3.0*ratio)*ratio + 1.0))/2.0);
    basis[3] = (PrecisionTYPE)((ratio-1.0) * FF/2.0);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void interpolantCubicSpline(PrecisionTYPE ratio, PrecisionTYPE *basis, PrecisionTYPE *derivative)
{
    interpolantCubicSpline<PrecisionTYPE>(ratio,basis);
    if(ratio<0.0) ratio=0.0; //rounding error
    PrecisionTYPE FF= ratio*ratio;
    derivative[0] = (PrecisionTYPE)((4.0*ratio - 3.0*FF - 1.0)/2.0);
    derivative[1] = (PrecisionTYPE)((9.0*ratio - 10.0) * ratio/2.0);
    derivative[2] = (PrecisionTYPE)((8.0*ratio - 9.0*FF + 1)/2.0);
    derivative[3] = (PrecisionTYPE)((3.0*ratio - 2.0) * ratio/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void CubicSplineResampleSourceImage(nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultImage,
                                    int *mask,
                                    PrecisionTYPE bgValue)
{
    // The spline decomposition assumes a background set to 0 the bgValue variable is thus not use here

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);

    unsigned int targetVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D Cubic spline resampling of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
        FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceCoefficients = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[3];
        int previous[3];
        PrecisionTYPE xBasis[4];
        PrecisionTYPE yBasis[4];
        PrecisionTYPE zBasis[4];
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE intensity=(PrecisionTYPE)(0.0);

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;
                PrecisionTYPE worldZ=(PrecisionTYPE) *deformationFieldPtrZ;
                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                    worldZ*sourceIJKMatrix.m[0][2] +  sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                    worldZ*sourceIJKMatrix.m[1][2] +  sourceIJKMatrix.m[1][3];
                position[2] = worldX*sourceIJKMatrix.m[2][0] + worldY*sourceIJKMatrix.m[2][1] +
                    worldZ*sourceIJKMatrix.m[2][2] +  sourceIJKMatrix.m[2][3];

                previous[0] = static_cast<int>(floor(position[0]));
                previous[1] = static_cast<int>(floor(position[1]));
                previous[2] = static_cast<int>(floor(position[2]));

                // basis values along the x axis
                relative=position[0]-(PrecisionTYPE)previous[0];
                interpolantCubicSpline<PrecisionTYPE>(relative, xBasis);
                // basis values along the y axis
                relative=position[1]-(PrecisionTYPE)previous[1];
                interpolantCubicSpline<PrecisionTYPE>(relative, yBasis);
                // basis values along the z axis
                relative=position[2]-(PrecisionTYPE)previous[2];
                interpolantCubicSpline<PrecisionTYPE>(relative, zBasis);

                --previous[0];--previous[1];--previous[2];

                for(short c=0; c<4; c++){
                    short Z= previous[2]+c;
                    if(-1<Z && Z<sourceImage->nz){
                        SourceTYPE *zPointer = &sourceCoefficients[Z*sourceImage->nx*sourceImage->ny];
                        PrecisionTYPE yTempNewValue=0.0;
                        for(short b=0; b<4; b++){
                            short Y= previous[1]+b;
                            SourceTYPE *yzPointer = &zPointer[Y*sourceImage->nx];
                            if(-1<Y && Y<sourceImage->ny){
                                SourceTYPE *xyzPointer = &yzPointer[previous[0]];
                                PrecisionTYPE xTempNewValue=0.0;
                                for(short a=0; a<4; a++){
                                    if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                        const PrecisionTYPE coeff = *xyzPointer;
                                        xTempNewValue +=  coeff * xBasis[a];
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

            switch(sourceImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_UINT8:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT16:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT32:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                default:
                    (*resultIntensity)=(SourceTYPE)round(intensity);
                    break;
            }
            resultIntensity++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
            deformationFieldPtrZ++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void CubicSplineResampleSourceImage2D(  nifti_image *sourceImage,
                                        nifti_image *deformationField,
                                        nifti_image *resultImage,
                                        int *mask,
                                        PrecisionTYPE bgValue)
{
    // The spline decomposition assumes a background set to 0 the bgValue variable is thus not use here

    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    unsigned int targetVoxelNumber = resultImage->nx*resultImage->ny;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;

    for(int t=0; t<resultImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D Cubic spline resampling of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceCoefficients = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[2];
        int previous[2];
        PrecisionTYPE xBasis[4];
        PrecisionTYPE yBasis[4];
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE intensity=0.0;

            if((*maskPtr++)>-1){

                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;
                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                sourceIJKMatrix.m[1][3];

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);

                // basis values along the x axis
                relative=position[0]-(PrecisionTYPE)previous[0];
                interpolantCubicSpline<PrecisionTYPE>(relative, xBasis);
                // basis values along the y axis
                relative=position[1]-(PrecisionTYPE)previous[1];
                interpolantCubicSpline<PrecisionTYPE>(relative, yBasis);

                previous[0]--;previous[1]--;

                for(short b=0; b<4; b++){
                    short Y= previous[1]+b;
                    SourceTYPE *yPointer = &sourceCoefficients[Y*sourceImage->nx];
                    if(-1<Y && Y<sourceImage->ny){
                        SourceTYPE *xyPointer = &yPointer[previous[0]];
                        PrecisionTYPE xTempNewValue=0.0;
                        for(short a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                const PrecisionTYPE coeff = *xyPointer;
                                xTempNewValue +=  coeff * xBasis[a];
                            }
                            xyPointer++;
                        }
                        intensity += (xTempNewValue * yBasis[b]);
                    }
                }
            }

            switch(sourceImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_UINT8:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT16:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT32:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                default:
                    (*resultIntensity)=(SourceTYPE)round(intensity);
                    break;
            }
            resultIntensity++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void TrilinearResampleSourceImage(  nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultImage,
                                    int *mask,
                                    PrecisionTYPE bgValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    unsigned int targetVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;

    for(int t=0; t<resultImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear resampling of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
        FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *intensityPtr = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        float voxelIndex[3];
        float position[3];
        int previous[3];
        PrecisionTYPE xBasis[2];
        PrecisionTYPE yBasis[2];
        PrecisionTYPE zBasis[2];
        PrecisionTYPE relative;

        mat44 *sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=&(sourceImage->sto_ijk);
        else sourceIJKMatrix=&(sourceImage->qto_ijk);


        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE intensity=0.0;

            if((*maskPtr++)>-1){

                voxelIndex[0]=(float) *deformationFieldPtrX;
                voxelIndex[1]=(float) *deformationFieldPtrY;
                voxelIndex[2]=(float) *deformationFieldPtrZ;

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, voxelIndex, position);

                if( position[0]>=0.f && position[0]<(PrecisionTYPE)(sourceImage->nx-1) &&
                    position[1]>=0.f && position[1]<(PrecisionTYPE)(sourceImage->ny-1) &&
                    position[2]>=0.f && position[2]<(PrecisionTYPE)(sourceImage->nz-1) ){

                    previous[0] = (int)position[0];
                    previous[1] = (int)position[1];
                    previous[2] = (int)position[2];
                    // basis values along the x axis
                    relative=position[0]-(PrecisionTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (PrecisionTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(PrecisionTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (PrecisionTYPE)(1.0-relative);
                    yBasis[1]= relative;
                    // basis values along the z axis
                    relative=position[2]-(PrecisionTYPE)previous[2];
                    if(relative<0) relative=0.0; // rounding error correction
                    zBasis[0]= (PrecisionTYPE)(1.0-relative);
                    zBasis[1]= relative;

                    for(short c=0; c<2; c++){
                        short Z= previous[2]+c;
                        SourceTYPE *zPointer = &intensityPtr[Z*sourceImage->nx*sourceImage->ny];
                        PrecisionTYPE yTempNewValue=0.0;
                        for(short b=0; b<2; b++){
                            short Y= previous[1]+b;
                            SourceTYPE *xyzPointer = &zPointer[Y*sourceImage->nx+previous[0]];
                            PrecisionTYPE xTempNewValue=0.0;
                            for(short a=0; a<2; a++){
                                const SourceTYPE coeff = *xyzPointer;
                                xTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
                                xyzPointer++;
                            }
                            yTempNewValue += (xTempNewValue * yBasis[b]);
                        }
                        intensity += yTempNewValue * zBasis[c];
                    }
                }
                else intensity = bgValue;
            }

            switch(sourceImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_UINT8:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT16:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT32:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                default:
                    (*resultIntensity)=(SourceTYPE)round(intensity);
                    break;
            }
            resultIntensity++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
            deformationFieldPtrZ++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void TrilinearResampleSourceImage2D(nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultImage,
                                    int *mask,
                                    PrecisionTYPE bgValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    unsigned int targetVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;

    for(int t=0; t<resultImage->nt;t++){

#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear resampling of volume number %i\n",t);
#endif
        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *intensityPtr = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[2];
        int previous[2];
        PrecisionTYPE xBasis[2];
        PrecisionTYPE yBasis[2];
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE intensity=0.0;

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;

                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                sourceIJKMatrix.m[1][3];

                if( position[0]>=0.0f && position[0]<(PrecisionTYPE)(sourceImage->nx-1) &&
                    position[1]>=0.0f && position[1]<(PrecisionTYPE)(sourceImage->ny-1)) {

                    previous[0] = (int)position[0];
                    previous[1] = (int)position[1];
                    // basis values along the x axis
                    relative=position[0]-(PrecisionTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (PrecisionTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(PrecisionTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (PrecisionTYPE)(1.0-relative);
                    yBasis[1]= relative;

                    for(short b=0; b<2; b++){
                        short Y= previous[1]+b;
                        SourceTYPE *xyPointer = &intensityPtr[Y*sourceImage->nx+previous[0]];
                        PrecisionTYPE xTempNewValue=0.0;
                        for(short a=0; a<2; a++){
                            const SourceTYPE coeff = *xyPointer;
                            xTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
                            xyPointer++;
                        }
                        intensity += (xTempNewValue * yBasis[b]);
                    }
                }
                else intensity = bgValue;
            }

            switch(sourceImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    (*resultIntensity)=(SourceTYPE)intensity;
                    break;
                case NIFTI_TYPE_UINT8:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT16:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT32:
                    (*resultIntensity)=(SourceTYPE)(intensity>0?round(intensity):0);
                    break;
                default:
                    (*resultIntensity)=(SourceTYPE)round(intensity);
                    break;
            }
            resultIntensity++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleSourceImage(nifti_image *sourceImage,
                                        nifti_image *deformationField,
                                        nifti_image *resultImage,
                                        int *mask,
                                        PrecisionTYPE bgValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    unsigned int targetVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;

    for(int t=0; t<resultImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D nearest neighbor resampling of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
        FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *intensityPtr = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[3];
        int previous[3];

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0; index<targetVoxelNumber; index++){

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;
                PrecisionTYPE worldZ=(PrecisionTYPE) *deformationFieldPtrZ;
                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                worldZ*sourceIJKMatrix.m[0][2] +  sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                worldZ*sourceIJKMatrix.m[1][2] +  sourceIJKMatrix.m[1][3];
                position[2] = worldX*sourceIJKMatrix.m[2][0] + worldY*sourceIJKMatrix.m[2][1] +
                worldZ*sourceIJKMatrix.m[2][2] +  sourceIJKMatrix.m[2][3];

                previous[0] = (int)round(position[0]);
                previous[1] = (int)round(position[1]);
                previous[2] = (int)round(position[2]);

                if( -1<previous[2] && previous[2]<sourceImage->nz &&
                -1<previous[1] && previous[1]<sourceImage->ny &&
                -1<previous[0] && previous[0]<sourceImage->nx){
                    SourceTYPE intensity = intensityPtr[(previous[2]*sourceImage->ny+previous[1])*sourceImage->nx+previous[0]];
                    (*resultIntensity)=intensity;
                }
                else (*resultIntensity)=(SourceTYPE)bgValue;
            }
            else (*resultIntensity)=(SourceTYPE)bgValue;
            resultIntensity++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
            deformationFieldPtrZ++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleSourceImage2D(nifti_image *sourceImage,
                                          nifti_image *deformationField,
                                          nifti_image *resultImage,
                                          int *mask,
                                          PrecisionTYPE bgValue)
{
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    unsigned int targetVoxelNumber = resultImage->nx*resultImage->ny;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;

    for(int t=0; t<resultImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D nearest neighbor resampling of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *intensityPtr = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[2];
        int previous[2];

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;
                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                sourceIJKMatrix.m[1][3];

                previous[0] = (int)round(position[0]);
                previous[1] = (int)round(position[1]);

                if( -1<previous[1] && previous[1]<sourceImage->ny &&
                -1<previous[0] && previous[0]<sourceImage->nx){
                    SourceTYPE intensity = intensityPtr[previous[1]*sourceImage->nx+previous[0]];
                    (*resultIntensity)=intensity;
                }
                else (*resultIntensity)=(SourceTYPE)bgValue;
            }
            else (*resultIntensity)=(SourceTYPE)bgValue;
            resultIntensity++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
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
 * background value.
 */
template <class PrecisionTYPE, class FieldTYPE, class SourceTYPE>
void reg_resampleSourceImage2(	nifti_image *targetImage,
                                nifti_image *sourceImage,
                                nifti_image *resultImage,
                                nifti_image *deformationFieldImage,
                                int *mask,
                                int interp,
                                PrecisionTYPE bgValue
                                )
{
    /* The deformation field contains the position in the real world */
    if(interp==3){
        if(targetImage->nz>1){
                CubicSplineResampleSourceImage<PrecisionTYPE,SourceTYPE,FieldTYPE>( sourceImage,
                                                                                    deformationFieldImage,
                                                                                    resultImage,
                                                                                    mask,
                                                                                    bgValue);
        }
        else
        {
            CubicSplineResampleSourceImage2D<PrecisionTYPE,SourceTYPE,FieldTYPE>(  sourceImage,
                                                                                    deformationFieldImage,
                                                                                    resultImage,
                                                                                    mask,
                                                                                    bgValue);
        }
    }
    else if(interp==0){ // Nearest neighbor interpolation
        if(targetImage->nz>1){
                NearestNeighborResampleSourceImage<PrecisionTYPE,SourceTYPE, FieldTYPE>( sourceImage,
                                                                                         deformationFieldImage,
                                                                                         resultImage,
                                                                                         mask,
                                                                                         bgValue);
        }
        else
        {
                NearestNeighborResampleSourceImage2D<PrecisionTYPE,SourceTYPE, FieldTYPE>( sourceImage,
                                                                                           deformationFieldImage,
                                                                                           resultImage,
                                                                                           mask,
                                                                                           bgValue);
        }

    }
    else{ // trilinear interpolation [ by default ]
        if(targetImage->nz>1){
                TrilinearResampleSourceImage<PrecisionTYPE,SourceTYPE, FieldTYPE>( sourceImage,
                                                                                   deformationFieldImage,
                                                                                   resultImage,
                                                                                   mask,
                                                                                   bgValue);
        }
        else{
                TrilinearResampleSourceImage2D<PrecisionTYPE,SourceTYPE, FieldTYPE>( sourceImage,
                                                                                     deformationFieldImage,
                                                                                     resultImage,
                                                                                     mask,
                                                                                     bgValue);
        }
    }
}

/* *************************************************************** */
template <class PrecisionTYPE>
void reg_resampleSourceImage(	nifti_image *targetImage,
                                nifti_image *sourceImage,
                                nifti_image *resultImage,
                                nifti_image *deformationField,
                                int *mask,
                                int interp,
                                PrecisionTYPE bgValue)
{
	if(sourceImage->datatype != resultImage->datatype){
        printf("NiftyReg ERROR] reg_resampleSourceImage\tSource and result image should have the same data type\n");
        printf("NiftyReg ERROR] reg_resampleSourceImage\tNothing has been done\n");
        exit(1);
	}

    if(sourceImage->nt != resultImage->nt){
        printf("[NiftyReg ERROR] reg_resampleSourceImage\tThe source and result images have different dimension along the time axis\n");
        printf("NiftyReg ERROR] reg_resampleSourceImage\tNothing has been done\n");
        exit(1);
    }

    // a mask array is created if no mask is specified
    bool MrPropreRules = false;
    if(mask==NULL){
        mask=(int *)calloc(targetImage->nx*targetImage->ny*targetImage->nz,sizeof(int)); // voxels in the background are set to -1 so 0 will do the job here
        MrPropreRules = true;
    }

        switch ( deformationField->datatype ){
		case NIFTI_TYPE_FLOAT32:
			switch ( sourceImage->datatype ){
				case NIFTI_TYPE_UINT8:
					reg_resampleSourceImage2<PrecisionTYPE,float,unsigned char>(	targetImage,
													sourceImage,
													resultImage,
                                                                                                        deformationField,
                                                    mask,
													interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT8:
					reg_resampleSourceImage2<PrecisionTYPE,float,char>(	targetImage,
												sourceImage,
												resultImage,
                                                                                                deformationField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT16:
					reg_resampleSourceImage2<PrecisionTYPE,float,unsigned short>(	targetImage,
													sourceImage,
													resultImage,
                                                                                                        deformationField,
                                                    mask,
													interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT16:
					reg_resampleSourceImage2<PrecisionTYPE,float,short>(	targetImage,
												sourceImage,
												resultImage,
                                                                                                deformationField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT32:
					reg_resampleSourceImage2<PrecisionTYPE,float,unsigned int>(	targetImage,
													sourceImage,
													resultImage,
                                                                                                        deformationField,
                                                    mask,
													interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT32:
					reg_resampleSourceImage2<PrecisionTYPE,float,int>(	targetImage,
												sourceImage,
												resultImage,
                                                                                                deformationField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT32:
					reg_resampleSourceImage2<PrecisionTYPE,float,float>(	targetImage,
												sourceImage,
												resultImage,
                                                                                                deformationField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_resampleSourceImage2<PrecisionTYPE,float,double>(	targetImage,
												sourceImage,
												resultImage,
                                                                                                deformationField,
                                                    mask,
												interp,
										bgValue);
					break;
				default:
					printf("Source pixel type unsupported.");
					break;
			}
			break;
#ifdef _NR_DEV
		case NIFTI_TYPE_FLOAT64:
			switch ( sourceImage->datatype ){
				case NIFTI_TYPE_UINT8:
					reg_resampleSourceImage2<PrecisionTYPE,double,unsigned char>(	targetImage,
																	sourceImage,
																	resultImage,
                                                                                                                                        deformationField,
                                                    mask,
																	interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT8:
					reg_resampleSourceImage2<PrecisionTYPE,double,char>(	targetImage,
															sourceImage,
															resultImage,
                                                                                                                        deformationField,
                                                    mask,
															interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT16:
					reg_resampleSourceImage2<PrecisionTYPE,double,unsigned short>(	targetImage,
																	sourceImage,
																	resultImage,
                                                                                                                                        deformationField,
                                                    mask,
																	interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT16:
					reg_resampleSourceImage2<PrecisionTYPE,double,short>(	targetImage,
															sourceImage,
															resultImage,
                                                                                                                        deformationField,
                                                    mask,
															interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT32:
					reg_resampleSourceImage2<PrecisionTYPE,double,unsigned int>(	targetImage,
																	sourceImage,
																	resultImage,
                                                                                                                                        deformationField,
                                                    mask,
																	interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT32:
					reg_resampleSourceImage2<PrecisionTYPE,double,int>(	targetImage,
														sourceImage,
														resultImage,
                                                                                                                deformationField,
                                                    mask,
														interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT32:
					reg_resampleSourceImage2<PrecisionTYPE,double,float>(	targetImage,
															sourceImage,
															resultImage,
                                                                                                                        deformationField,
                                                    mask,
															interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_resampleSourceImage2<PrecisionTYPE,double,double>(	targetImage,
															sourceImage,
															resultImage,
                                                                                                                        deformationField,
                                                    mask,
															interp,
										bgValue);
					break;
				default:
					printf("Source pixel type unsupported.");
					break;
			}
			break;
#endif
		default:
			printf("Deformation field pixel type unsupported.");
			break;
	}
    if(MrPropreRules==true) free(mask);
}
template void reg_resampleSourceImage<float>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *,int, float);
template void reg_resampleSourceImage<double>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *,int, double);
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearGradientResultImage(  nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultGradientImage,
                                    int *mask)
{
    unsigned int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear gradient computation of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
        FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[gradientOffSet];

        SourceTYPE *sourceCoefficients = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[3];
        int previous[3];
        PrecisionTYPE xBasis[2];
        PrecisionTYPE yBasis[2];
        PrecisionTYPE zBasis[2];
        PrecisionTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE gradX=0.0;
            PrecisionTYPE gradY=0.0;
            PrecisionTYPE gradZ=0.0;

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;
                PrecisionTYPE worldZ=(PrecisionTYPE) *deformationFieldPtrZ;

                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                worldZ*sourceIJKMatrix.m[0][2] +  sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                worldZ*sourceIJKMatrix.m[1][2] +  sourceIJKMatrix.m[1][3];
                position[2] = worldX*sourceIJKMatrix.m[2][0] + worldY*sourceIJKMatrix.m[2][1] +
                worldZ*sourceIJKMatrix.m[2][2] +  sourceIJKMatrix.m[2][3];


                if( position[0]>=0.0f && position[0]<(PrecisionTYPE)(sourceImage->nx-1) &&
                    position[1]>=0.0f && position[1]<(PrecisionTYPE)(sourceImage->ny-1) &&
                    position[2]>=0.0f && position[2]<(PrecisionTYPE)(sourceImage->nz-1) ){

                    previous[0] = (int)position[0];
                    previous[1] = (int)position[1];
                    previous[2] = (int)position[2];
                    // basis values along the x axis
                    relative=position[0]-(PrecisionTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (PrecisionTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(PrecisionTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (PrecisionTYPE)(1.0-relative);
                    yBasis[1]= relative;
                    // basis values along the z axis
                    relative=position[2]-(PrecisionTYPE)previous[2];
                    if(relative<0) relative=0.0; // rounding error correction
                    zBasis[0]= (PrecisionTYPE)(1.0-relative);
                    zBasis[1]= relative;

                    for(short c=0; c<2; c++){
                        short Z= previous[2]+c;
                        SourceTYPE *zPointer = &sourceCoefficients[Z*sourceImage->nx*sourceImage->ny];
                        PrecisionTYPE xxTempNewValue=0.0;
                        PrecisionTYPE yyTempNewValue=0.0;
                        PrecisionTYPE zzTempNewValue=0.0;
                        for(short b=0; b<2; b++){
                            short Y= previous[1]+b;
                            SourceTYPE *yzPointer = &zPointer[Y*sourceImage->nx];
                            SourceTYPE *xyzPointer = &yzPointer[previous[0]];
                            PrecisionTYPE xTempNewValue=0.0;
                            PrecisionTYPE yTempNewValue=0.0;
                            for(short a=0; a<2; a++){
                                const SourceTYPE coeff = *xyzPointer;
                                xTempNewValue +=  (PrecisionTYPE)(coeff * deriv[a]);
                                yTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
                                xyzPointer++;
                            }
                            xxTempNewValue += xTempNewValue * yBasis[b];
                            yyTempNewValue += yTempNewValue * deriv[b];
                            zzTempNewValue += yTempNewValue * yBasis[b];
                        }
                        gradX += xxTempNewValue * zBasis[c];
                        gradY += yyTempNewValue * zBasis[c];
                        gradZ += zzTempNewValue * deriv[c];
                    }
                }
            }

            *resultGradientPtrX = (GradientTYPE)gradX;
            *resultGradientPtrY = (GradientTYPE)gradY;
            *resultGradientPtrZ = (GradientTYPE)gradZ;

            resultGradientPtrX++;
            resultGradientPtrY++;
            resultGradientPtrZ++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
            deformationFieldPtrZ++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearGradientResultImage2D(	nifti_image *sourceImage,
                                        nifti_image *deformationField,
                                        nifti_image *resultGradientImage,
                                        int *mask)
{
    unsigned int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D linear gradient computation of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];

        SourceTYPE *sourceCoefficients = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[2];
        int previous[2];
        PrecisionTYPE xBasis[2];
        PrecisionTYPE yBasis[2];
        PrecisionTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE gradX=0.0;
            PrecisionTYPE gradY=0.0;

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;

                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                sourceIJKMatrix.m[1][3];

                if( position[0]>=0.0f && position[0]<(PrecisionTYPE)(sourceImage->nx-1) &&
                    position[1]>=0.0f && position[1]<(PrecisionTYPE)(sourceImage->ny-1) ){

                    previous[0] = (int)position[0];
                    previous[1] = (int)position[1];
                    // basis values along the x axis
                    relative=position[0]-(PrecisionTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (PrecisionTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(PrecisionTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (PrecisionTYPE)(1.0-relative);
                    yBasis[1]= relative;

                    for(short b=0; b<2; b++){
                        short Y= previous[1]+b;
                        SourceTYPE *yPointer = &sourceCoefficients[Y*sourceImage->nx];
                        SourceTYPE *xyPointer = &yPointer[previous[0]];
                        PrecisionTYPE xTempNewValue=0.0;
                        PrecisionTYPE yTempNewValue=0.0;
                        for(short a=0; a<2; a++){
                            const SourceTYPE coeff = *xyPointer;
                            xTempNewValue +=  (PrecisionTYPE)(coeff * deriv[a]);
                            yTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
                            xyPointer++;
                        }
                        gradX += xTempNewValue * yBasis[b];
                        gradY += yTempNewValue * deriv[b];
                    }
                }
            }

            switch(resultGradientImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    *resultGradientPtrX = (GradientTYPE)gradX;
                    *resultGradientPtrY = (GradientTYPE)gradY;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    *resultGradientPtrX = (GradientTYPE)gradX;
                    *resultGradientPtrY = (GradientTYPE)gradY;
                    break;
                default:
                    *resultGradientPtrX=(GradientTYPE)round(gradX);
                    *resultGradientPtrY=(GradientTYPE)round(gradY);
                    break;
            }
            resultGradientPtrX++;
            resultGradientPtrY++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineGradientResultImage(nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultGradientImage,
                                    int *mask)
{
    unsigned int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D cubic spline gradient computation of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
        FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[gradientOffSet];

        SourceTYPE *sourceCoefficients = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[3];
        int previous[3];
        PrecisionTYPE xBasis[4], xDeriv[4];
        PrecisionTYPE yBasis[4], yDeriv[4];
        PrecisionTYPE zBasis[4], zDeriv[4];
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE gradX=0.0;
            PrecisionTYPE gradY=0.0;
            PrecisionTYPE gradZ=0.0;

            if((*maskPtr++)>-1){

                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;
                PrecisionTYPE worldZ=(PrecisionTYPE) *deformationFieldPtrZ;

                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                worldZ*sourceIJKMatrix.m[0][2] +  sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                worldZ*sourceIJKMatrix.m[1][2] +  sourceIJKMatrix.m[1][3];
                position[2] = worldX*sourceIJKMatrix.m[2][0] + worldY*sourceIJKMatrix.m[2][1] +
                worldZ*sourceIJKMatrix.m[2][2] +  sourceIJKMatrix.m[2][3];

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);
                previous[2] = (int)floor(position[2]);

                // basis values along the x axis
                relative=position[0]-(PrecisionTYPE)previous[0];
                interpolantCubicSpline<PrecisionTYPE>(relative, xBasis, xDeriv);

                // basis values along the y axis
                relative=position[1]-(PrecisionTYPE)previous[1];
                interpolantCubicSpline<PrecisionTYPE>(relative, yBasis, yDeriv);

                // basis values along the z axis
                relative=position[2]-(PrecisionTYPE)previous[2];
                interpolantCubicSpline<PrecisionTYPE>(relative, zBasis, zDeriv);

                previous[0]--;previous[1]--;previous[2]--;

                bool bg=false;
                for(short c=0; c<4; c++){
                    short Z= previous[2]+c;
                    if(-1<Z && Z<sourceImage->nz){
                        SourceTYPE *zPointer = &sourceCoefficients[Z*sourceImage->nx*sourceImage->ny];
                        PrecisionTYPE xxTempNewValue=0.0;
                        PrecisionTYPE yyTempNewValue=0.0;
                        PrecisionTYPE zzTempNewValue=0.0;
                        for(short b=0; b<4; b++){
                            short Y= previous[1]+b;
                            SourceTYPE *yzPointer = &zPointer[Y*sourceImage->nx];
                            if(-1<Y && Y<sourceImage->ny){
                                SourceTYPE *xyzPointer = &yzPointer[previous[0]];
                                PrecisionTYPE xTempNewValue=0.0;
                                PrecisionTYPE yTempNewValue=0.0;
                                PrecisionTYPE zTempNewValue=0.0;
                                for(short a=0; a<4; a++){
                                    if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                        const PrecisionTYPE coeff = *xyzPointer;
                                        xTempNewValue +=  coeff * xDeriv[a];
                                        yTempNewValue +=  coeff * xBasis[a];
                                        zTempNewValue +=  coeff * xBasis[a];
                                    }
                                    else bg=true;
                                    xyzPointer++;
                                }
                                xxTempNewValue += (xTempNewValue * yBasis[b]);
                                yyTempNewValue += (yTempNewValue * yDeriv[b]);
                                zzTempNewValue += (zTempNewValue * yBasis[b]);
                            }
                            else bg=true;
                        }
                        gradX += xxTempNewValue * zBasis[c];
                        gradY += yyTempNewValue * zBasis[c];
                        gradZ += zzTempNewValue * zDeriv[c];
                    }
                    else bg=true;
                }

                if(bg==true){
                    gradX=0.0;
                    gradY=0.0;
                    gradZ=0.0;
                }
            }
            switch(resultGradientImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    *resultGradientPtrX = (GradientTYPE)gradX;
                    *resultGradientPtrY = (GradientTYPE)gradY;
                    *resultGradientPtrZ = (GradientTYPE)gradZ;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    *resultGradientPtrX = (GradientTYPE)gradX;
                    *resultGradientPtrY = (GradientTYPE)gradY;
                    *resultGradientPtrZ = (GradientTYPE)gradZ;
                    break;
                default:
                    *resultGradientPtrX=(GradientTYPE)round(gradX);
                    *resultGradientPtrY=(GradientTYPE)round(gradY);
                    *resultGradientPtrZ=(GradientTYPE)round(gradZ);
                    break;
            }
            resultGradientPtrX++;
            resultGradientPtrY++;
            resultGradientPtrZ++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
            deformationFieldPtrZ++;
        }
    }
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineGradientResultImage2D(nifti_image *sourceImage,
                                      nifti_image *deformationField,
                                      nifti_image *resultGradientImage,
                                      int *mask)
{
    unsigned int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny;
    unsigned int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);

    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D cubic spline gradient computation of volume number %i\n",t);
#endif

        FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
        FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];

        SourceTYPE *sourceCoefficients = &sourceIntensityPtr[t*sourceVoxelNumber];

        int *maskPtr = &mask[0];

        PrecisionTYPE position[2];
        int previous[2];
        PrecisionTYPE xBasis[4], xDeriv[4];
        PrecisionTYPE yBasis[4], yDeriv[4];
        PrecisionTYPE relative;

        mat44 sourceIJKMatrix;
        if(sourceImage->sform_code>0)
            sourceIJKMatrix=sourceImage->sto_ijk;
        else sourceIJKMatrix=sourceImage->qto_ijk;

        for(unsigned int index=0;index<targetVoxelNumber; index++){

            PrecisionTYPE gradX=0.0;
            PrecisionTYPE gradY=0.0;

            if((*maskPtr++)>-1){
                PrecisionTYPE worldX=(PrecisionTYPE) *deformationFieldPtrX;
                PrecisionTYPE worldY=(PrecisionTYPE) *deformationFieldPtrY;

                /* real -> voxel; source space */
                position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
                sourceIJKMatrix.m[0][3];
                position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
                sourceIJKMatrix.m[1][3];

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);
                // basis values along the x axis
                relative=position[0]-(PrecisionTYPE)previous[0];
                interpolantCubicSpline<PrecisionTYPE>(relative, xBasis, xDeriv);
                // basis values along the y axis
                relative=position[1]-(PrecisionTYPE)previous[1];
                interpolantCubicSpline<PrecisionTYPE>(relative, yBasis, yDeriv);

                previous[0]--;previous[1]--;

                bool bg=false;
                for(short b=0; b<4; b++){
                    short Y= previous[1]+b;
                    SourceTYPE *yPointer = &sourceCoefficients[Y*sourceImage->nx];
                    if(-1<Y && Y<sourceImage->ny){
                        SourceTYPE *xyPointer = &yPointer[previous[0]];
                        PrecisionTYPE xTempNewValue=0.0;
                        PrecisionTYPE yTempNewValue=0.0;
                        for(short a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                const SourceTYPE coeff = *xyPointer;
                                xTempNewValue +=  coeff * xDeriv[a];
                                yTempNewValue +=  coeff * xBasis[a];
                            }
                            else bg=true;
                            xyPointer++;
                        }
                        gradX += (xTempNewValue * yBasis[b]);
                        gradY += (yTempNewValue * yDeriv[b]);
                    }
                    else bg=true;
                }

                if(bg==true){
                    gradX=0.0;
                    gradY=0.0;
                }
            }
            switch(resultGradientImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    *resultGradientPtrX = (GradientTYPE)gradX;
                    *resultGradientPtrY = (GradientTYPE)gradY;
                    break;
                case NIFTI_TYPE_FLOAT64:
                    *resultGradientPtrX = (GradientTYPE)gradX;
                    *resultGradientPtrY = (GradientTYPE)gradY;
                    break;
                default:
                    *resultGradientPtrX=(GradientTYPE)round(gradX);
                    *resultGradientPtrY=(GradientTYPE)round(gradY);
                    break;
            }
            resultGradientPtrX++;
            resultGradientPtrY++;
            deformationFieldPtrX++;
            deformationFieldPtrY++;
        }
    }
}
/* *************************************************************** */
template <class PrecisionTYPE, class FieldTYPE, class SourceTYPE, class GradientTYPE>
void reg_getSourceImageGradient3(   nifti_image *targetImage,
                                    nifti_image *sourceImage,
                                    nifti_image *resultGradientImage,
                                    nifti_image *deformationField,
                                    int *mask,
                                    int interp)
{
	/* The deformation field contains the position in the real world */

    if(interp==3){
        if(targetImage->nz>1){
            CubicSplineGradientResultImage
                    <PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(  sourceImage,
                                                                        deformationField,
                                                                        resultGradientImage,
                                                                        mask);
        }
        else{
            CubicSplineGradientResultImage2D
                    <PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(sourceImage,
                                                                      deformationField,
                                                                      resultGradientImage,
                                                                      mask);
        }
    }
    else{ // trilinear interpolation [ by default ]
        if(targetImage->nz>1){
            TrilinearGradientResultImage
                    <PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(   sourceImage,
                                                                         deformationField,
                                                                         resultGradientImage,
                                                                         mask);
        }
        else{
            TrilinearGradientResultImage2D
                    <PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>( sourceImage,
                                                                       deformationField,
                                                                       resultGradientImage,
                                                                       mask);
        }
    }
}
/* *************************************************************** */
template <class PrecisionTYPE, class FieldTYPE, class SourceTYPE>
void reg_getSourceImageGradient2(nifti_image *targetImage,
								nifti_image *sourceImage,
								nifti_image *resultGradientImage,
                                                                nifti_image *deformationField,
                                int *mask,
								int interp
							)
{
	switch(resultGradientImage->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getSourceImageGradient3<PrecisionTYPE,FieldTYPE,SourceTYPE,float>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getSourceImageGradient3<PrecisionTYPE,FieldTYPE,SourceTYPE,double>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		default:
                        printf("[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template <class PrecisionTYPE, class FieldTYPE>
void reg_getSourceImageGradient1(nifti_image *targetImage,
								nifti_image *sourceImage,
								nifti_image *resultGradientImage,
                                                                nifti_image *deformationField,
                                int *mask,
								int interp
							)
{
	switch(sourceImage->datatype){
		case NIFTI_TYPE_UINT8:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,unsigned char>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_INT8:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,char>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,unsigned short>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_INT16:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,short>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,unsigned int>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_INT32:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,int>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,float>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,double>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
		default:
                        printf("[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
extern "C++" template <class PrecisionTYPE>
void reg_getSourceImageGradient(	nifti_image *targetImage,
                                    nifti_image *sourceImage,
                                    nifti_image *resultGradientImage,
                                    nifti_image *deformationField,
                                    int *mask,
                                    int interp
							)
{
    // a mask array is created if no mask is specified
    bool MrPropreRule=false;
    if(mask==NULL){
        mask=(int *)calloc(targetImage->nx*targetImage->ny*targetImage->nz,sizeof(int)); // voxels in the background are set to -1 so 0 will do the job here
        MrPropreRule=true;
    }

    // Check if the dimension are correct
    if(sourceImage->nt != resultGradientImage->nt){
        printf("[NiftyReg ERROR] reg_getSourceImageGradient\tThe source and result images have different dimension along the time axis\n");
        printf("[NiftyReg ERROR] reg_getSourceImageGradient\tNothing has been done\n");
        return;
    }

        switch(deformationField->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getSourceImageGradient1<PrecisionTYPE,float>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
#ifdef _NR_DEV
		case NIFTI_TYPE_FLOAT64:
			reg_getSourceImageGradient1<PrecisionTYPE,double>
                                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
			break;
#endif
		default:
                        printf("[NiftyReg ERROR] reg_getSourceImageGradient\tDeformation field pixel type unsupported.");
			break;
	}
    if(MrPropreRule==true) free(mask);
}
/* *************************************************************** */
template void reg_getSourceImageGradient<float>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *, int);
template void reg_getSourceImageGradient<double>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *, int);
/* *************************************************************** */
/* *************************************************************** */

#endif
