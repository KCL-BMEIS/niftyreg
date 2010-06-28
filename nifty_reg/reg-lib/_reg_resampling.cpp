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
template<class PrecisionType>
int round(PrecisionType x)
{
	return int(x > 0.0 ? x + 0.5 : x - 0.5);
}
#endif

/* *************************************************************** */
template <class PrecisionTYPE>
void extractLine(int start, int end, int increment,const PrecisionTYPE *image, PrecisionTYPE *values)
{
	unsigned int index = 0;
	for(int i=start; i<end; i+=increment) values[index++] = image[i];
}
/* *************************************************************** */
template <class PrecisionTYPE>
void restoreLine(int start, int end, int increment, PrecisionTYPE *image, const PrecisionTYPE *values)
{
	unsigned int index = 0;
	for(int i=start; i<end; i+=increment) image[i] = values[index++];
}
/* *************************************************************** */
template <class PrecisionTYPE>
void intensitiesToSplineCoefficients(PrecisionTYPE *values, int number, PrecisionTYPE pole)
{
	// Border are set to zero
	PrecisionTYPE currentPole = pole;
	PrecisionTYPE currentOpposite = pow(pole,(PrecisionTYPE)(2.0*(PrecisionTYPE)number-1.0));
	PrecisionTYPE sum=0.0;
	for(short i=1; i<number; i++){
		sum += (currentPole - currentOpposite) * values[i];
		currentPole *= pole;
		currentOpposite /= pole;
	}
	values[0] = (PrecisionTYPE)((values[0] - pole*pole*(values[0] + sum)) / (1.0 - pow(pole,(PrecisionTYPE)(2.0*(double)number+2.0))));
	
	//other values forward
	for(int i=1; i<number; i++){
		values[i] += pole * values[i-1];
	}
	
	PrecisionTYPE ipp=(PrecisionTYPE)(1.0-pole); ipp*=ipp;
	
	//last value
	values[number-1] = ipp * values[number-1];
	
	//other values backward
	for(int i=number-2; 0<=i; i--){
		values[i] = pole * values[i+1] + ipp*values[i];
	}
	return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void CubicSplineResampleSourceImage(PrecisionTYPE *sourceCoefficients,
									nifti_image *sourceImage,
									nifti_image *positionField,
									nifti_image *resultImage,
                                    int *mask,
									PrecisionTYPE bgValue)
{
	// The spline decomposition assumes a background set to 0 the bgValue variable is thus not use here

	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultImage->nvox];
	FieldTYPE *positionFieldPtrZ = &positionFieldPtrY[resultImage->nvox];

	SourceTYPE *resultIntensity = static_cast<SourceTYPE *>(resultImage->data);

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
	
	for(unsigned int index=0;index<resultImage->nvox; index++){
		
        PrecisionTYPE intensity=(PrecisionTYPE)(0.0);

        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
		    PrecisionTYPE worldZ=(PrecisionTYPE) *positionFieldPtrZ;
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
		    if(relative<0) relative=0.0; // rounding error correction
		    xBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    xBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - xBasis[3]);
		    xBasis[2]= (PrecisionTYPE)(relative + xBasis[0] - 2.0*xBasis[3]);
		    xBasis[1]= (PrecisionTYPE)(1.0 - xBasis[0] - xBasis[2] - xBasis[3]);
		    // basis values along the y axis
		    relative=position[1]-(PrecisionTYPE)previous[1];
		    if(relative<0) relative=0.0; // rounding error correction
		    yBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    yBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - yBasis[3]);
		    yBasis[2]= (PrecisionTYPE)(relative + yBasis[0] - 2.0*yBasis[3]);
		    yBasis[1]= (PrecisionTYPE)(1.0 - yBasis[0] - yBasis[2] - yBasis[3]);
		    // basis values along the z axis
		    relative=position[2]-(PrecisionTYPE)previous[2];
		    if(relative<0) relative=(PrecisionTYPE)(0.0); // rounding error correction
		    zBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    zBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - zBasis[3]);
		    zBasis[2]= (PrecisionTYPE)(relative + zBasis[0] - 2.0*zBasis[3]);
		    zBasis[1]= (PrecisionTYPE)(1.0 - zBasis[0] - zBasis[2] - zBasis[3]);

		    previous[0]--;previous[1]--;previous[2]--;

		    for(short c=0; c<4; c++){
			    short Z= previous[2]+c;
			    if(-1<Z && Z<sourceImage->nz){
				    PrecisionTYPE *zPointer = &sourceCoefficients[Z*sourceImage->nx*sourceImage->ny];
				    PrecisionTYPE yTempNewValue=0.0;
				    for(short b=0; b<4; b++){
					    short Y= previous[1]+b;
					    PrecisionTYPE *yzPointer = &zPointer[Y*sourceImage->nx];
					    if(-1<Y && Y<sourceImage->ny){
						    PrecisionTYPE *xyzPointer = &yzPointer[previous[0]];
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
        positionFieldPtrX++;
        positionFieldPtrY++;
        positionFieldPtrZ++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void CubicSplineResampleSourceImage2D(PrecisionTYPE *sourceCoefficients,
									  nifti_image *sourceImage,
									  nifti_image *positionField,
                                      nifti_image *resultImage,
                                      int *mask,
									  PrecisionTYPE bgValue)
{
	// The spline decomposition assumes a background set to 0 the bgValue variable is thus not use here
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultImage->nvox];
	
	SourceTYPE *resultIntensity = static_cast<SourceTYPE *>(resultImage->data);

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
	
	for(unsigned int index=0;index<resultImage->nvox; index++){
		
        PrecisionTYPE intensity=0.0;

        if((*maskPtr++)>-1){

		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
		    /* real -> voxel; source space */
		    position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
		    sourceIJKMatrix.m[0][3];
		    position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
		    sourceIJKMatrix.m[1][3];

		    previous[0] = (int)floor(position[0]);
		    previous[1] = (int)floor(position[1]);
		    // basis values along the x axis
		    relative=position[0]-(PrecisionTYPE)previous[0];
		    if(relative<0) relative=0.0; // rounding error correction
		    xBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    xBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - xBasis[3]);
		    xBasis[2]= (PrecisionTYPE)(relative + xBasis[0] - 2.0*xBasis[3]);
		    xBasis[1]= (PrecisionTYPE)(1.0 - xBasis[0] - xBasis[2] - xBasis[3]);
		    // basis values along the y axis
		    relative=position[1]-(PrecisionTYPE)previous[1];
		    if(relative<0) relative=(PrecisionTYPE)(0.0); // rounding error correction
		    yBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    yBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - yBasis[3]);
		    yBasis[2]= (PrecisionTYPE)(relative + yBasis[0] - 2.0*yBasis[3]);
		    yBasis[1]= (PrecisionTYPE)(1.0 - yBasis[0] - yBasis[2] - yBasis[3]);

		    previous[0]--;previous[1]--;

		    for(short b=0; b<4; b++){
			    short Y= previous[1]+b;
			    PrecisionTYPE *yPointer = &sourceCoefficients[Y*sourceImage->nx];
			    if(-1<Y && Y<sourceImage->ny){
				    PrecisionTYPE *xyPointer = &yPointer[previous[0]];
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
        positionFieldPtrX++;
        positionFieldPtrY++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void TrilinearResampleSourceImage(	SourceTYPE *intensityPtr,
                                    nifti_image *sourceImage,
                                    nifti_image *positionField,
                                    nifti_image *resultImage,
                                    int *mask,
                                    PrecisionTYPE bgValue)
{
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultImage->nvox];
	FieldTYPE *positionFieldPtrZ = &positionFieldPtrY[resultImage->nvox];

	SourceTYPE *resultIntensity = static_cast<SourceTYPE *>(resultImage->data);

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


	for(unsigned int index=0;index<resultImage->nvox; index++){

        PrecisionTYPE intensity=0.0;

        if((*maskPtr++)>-1){

		    voxelIndex[0]=(float) *positionFieldPtrX;
		    voxelIndex[1]=(float) *positionFieldPtrY;
		    voxelIndex[2]=(float) *positionFieldPtrZ;

		    /* real -> voxel; source space */
            reg_mat44_mul(sourceIJKMatrix, voxelIndex, position);


            if( position[0]>=0.0f && position[0]<sourceImage->nx-1 &&
                position[1]>=0.0f && position[1]<sourceImage->ny-1 &&
                position[2]>=0.0f && position[2]<sourceImage->nz-1 ){

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);
                previous[2] = (int)floor(position[2]);
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
        positionFieldPtrX++;
        positionFieldPtrY++;
        positionFieldPtrZ++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void TrilinearResampleSourceImage2D(SourceTYPE *intensityPtr,
									nifti_image *sourceImage,
									nifti_image *positionField,
									nifti_image *resultImage,
                                    int *mask,
									PrecisionTYPE bgValue)
{
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultImage->nvox];
	
	SourceTYPE *resultIntensity = static_cast<SourceTYPE *>(resultImage->data);

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
	
	for(unsigned int index=0;index<resultImage->nvox; index++){

        PrecisionTYPE intensity=0.0;

        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;

		    /* real -> voxel; source space */
		    position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
		    sourceIJKMatrix.m[0][3];
		    position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
		    sourceIJKMatrix.m[1][3];

            if( position[0]>=0.0f && position[0]<sourceImage->nx-1 &&
                position[1]>=0.0f && position[1]<sourceImage->ny-1 ){

		        previous[0] = (int)floor(position[0]);
		        previous[1] = (int)floor(position[1]);
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
        positionFieldPtrX++;
        positionFieldPtrY++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleSourceImage(SourceTYPE *intensityPtr,
										nifti_image *sourceImage,
										nifti_image *positionField,
										nifti_image *resultImage,
                                        int *mask,
										PrecisionTYPE bgValue)
{
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultImage->nvox];
	FieldTYPE *positionFieldPtrZ = &positionFieldPtrY[resultImage->nvox];
	
	SourceTYPE *resultIntensity = static_cast<SourceTYPE *>(resultImage->data);

    int *maskPtr = &mask[0];
	
	PrecisionTYPE position[3];
	int previous[3];
	
	mat44 sourceIJKMatrix;
	if(sourceImage->sform_code>0)
		sourceIJKMatrix=sourceImage->sto_ijk;
	else sourceIJKMatrix=sourceImage->qto_ijk;
	
	for(unsigned int index=0; index<resultImage->nvox; index++){
		
        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
		    PrecisionTYPE worldZ=(PrecisionTYPE) *positionFieldPtrZ;
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
        positionFieldPtrX++;
        positionFieldPtrY++;
        positionFieldPtrZ++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleSourceImage2D(SourceTYPE *intensityPtr,
										  nifti_image *sourceImage,
										  nifti_image *positionField,
										  nifti_image *resultImage,
                                          int *mask,
										  PrecisionTYPE bgValue)
{
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultImage->nvox];
	
	SourceTYPE *resultIntensity = static_cast<SourceTYPE *>(resultImage->data);

    int *maskPtr = &mask[0];

	PrecisionTYPE position[2];
	int previous[2];
	
	mat44 sourceIJKMatrix;
	if(sourceImage->sform_code>0)
		sourceIJKMatrix=sourceImage->sto_ijk;
	else sourceIJKMatrix=sourceImage->qto_ijk;
	
	for(unsigned int index=0;index<resultImage->nvox; index++){
		
        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
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
        positionFieldPtrX++;
        positionFieldPtrY++;
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
								nifti_image *positionFieldImage,
                                int *mask,
								int interp,
								PrecisionTYPE bgValue
							)
{
	/* The deformation field contains the position in the real world */
	
	if(interp==3){
		/* in order to apply a cubic Spline resampling, the source image
		 intensities have to be decomposed */
		SourceTYPE *intensityPtr = (SourceTYPE *)sourceImage->data;
	 	PrecisionTYPE *sourceCoefficients = (PrecisionTYPE *)malloc(sourceImage->nvox*sizeof(PrecisionTYPE));
		for(unsigned int i=0; i<sourceImage->nvox;i++)
			sourceCoefficients[i]=(PrecisionTYPE)intensityPtr[i];

		PrecisionTYPE pole = (PrecisionTYPE)(sqrt(3.0) - 2.0);

			// X axis first
		int number = sourceImage->nx;
		PrecisionTYPE *values=new PrecisionTYPE[number];
		int increment = 1;
		for(int i=0;i<sourceImage->ny*sourceImage->nz;i++){
			int start = i*sourceImage->nx;
			int end =  start + sourceImage->nx;
			extractLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
			intensitiesToSplineCoefficients<PrecisionTYPE>(values, number, pole);
			restoreLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
		}
		delete[] values;
		// Y axis then
		number = sourceImage->ny;
		values=new PrecisionTYPE[number];
		increment = sourceImage->nx;
		for(int i=0;i<sourceImage->nx*sourceImage->nz;i++){
			int start = i + i/sourceImage->nx * sourceImage->nx * (sourceImage->ny - 1);
			int end =  start + sourceImage->nx*sourceImage->ny;
			extractLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
			intensitiesToSplineCoefficients<PrecisionTYPE>(values, number, pole);
			restoreLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
		}
		delete[] values;
		if(targetImage->nz>1){
			// Z axis at last
			number = sourceImage->nz;
			values=new PrecisionTYPE[number];
			increment = sourceImage->nx*sourceImage->ny;
			for(int i=0;i<sourceImage->nx*sourceImage->ny;i++){
				int start = i;
				int end =  start + sourceImage->nx*sourceImage->ny*sourceImage->nz;
				extractLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
				intensitiesToSplineCoefficients<PrecisionTYPE>(values, number, pole);
				restoreLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
			}
			delete[] values;
		}
		
		if(targetImage->nz>1){
			CubicSplineResampleSourceImage<PrecisionTYPE,SourceTYPE,FieldTYPE>(	sourceCoefficients,
																				sourceImage,
																				positionFieldImage,
																				resultImage,
                                                                                mask,
																				bgValue);
		}
		else
		{
			CubicSplineResampleSourceImage2D<PrecisionTYPE, SourceTYPE, FieldTYPE>(	sourceCoefficients,
																	   sourceImage,
																	   positionFieldImage,
																	   resultImage,
                                                                       mask,
																	   bgValue);
		}
		free(sourceCoefficients);
	}
	else if(interp==0){ // Nearest neighbor interpolation
		SourceTYPE *intensityPtr = (SourceTYPE *)sourceImage->data;
		if(targetImage->nz>1){
			NearestNeighborResampleSourceImage<PrecisionTYPE,SourceTYPE, FieldTYPE>(	intensityPtr,
																		 sourceImage,
																		 positionFieldImage,
																		 resultImage,
                                                                         mask,
																		 bgValue);
		}
		else
		{
			NearestNeighborResampleSourceImage2D<PrecisionTYPE,SourceTYPE, FieldTYPE>(	intensityPtr,
																		   sourceImage,
																		   positionFieldImage,
																		   resultImage,
                                                                           mask,
																		   bgValue);
		}

	}
	else{ // trilinear interpolation [ by default ]
		SourceTYPE *intensityPtr = (SourceTYPE *)sourceImage->data;
		if(targetImage->nz>1){
			TrilinearResampleSourceImage<PrecisionTYPE,SourceTYPE, FieldTYPE>(	intensityPtr,
																   sourceImage,
																   positionFieldImage,
																   resultImage,
                                                                   mask,
																   bgValue);
		}
		else{
			TrilinearResampleSourceImage2D<PrecisionTYPE,SourceTYPE, FieldTYPE>(	intensityPtr,
																	 sourceImage,
																	 positionFieldImage,
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
								nifti_image *positionField,
                                int *mask,
								int interp,
								PrecisionTYPE bgValue)
{
	if(sourceImage->datatype != resultImage->datatype){
		printf("err:\treg_resampleSourceImage\tSource and result image should have the same data type\n");
		printf("err:\treg_resampleSourceImage\tNothing has been done\n");
		return;
	}

    // a mask array is created if no mask is specified
    bool MrPropreRules = false;
    if(mask==NULL){
        mask=(int *)calloc(targetImage->nvox,sizeof(int)); // voxels in the background are set to -1 so 0 will do the job here
        MrPropreRules = true;
    }

	switch ( positionField->datatype ){
		case NIFTI_TYPE_FLOAT32:
			switch ( sourceImage->datatype ){
				case NIFTI_TYPE_UINT8:
					reg_resampleSourceImage2<PrecisionTYPE,float,unsigned char>(	targetImage,
													sourceImage,
													resultImage,
													positionField,
                                                    mask,
													interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT8:
					reg_resampleSourceImage2<PrecisionTYPE,float,char>(	targetImage,
												sourceImage,
												resultImage,
												positionField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT16:
					reg_resampleSourceImage2<PrecisionTYPE,float,unsigned short>(	targetImage,
													sourceImage,
													resultImage,
													positionField,
                                                    mask,
													interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT16:
					reg_resampleSourceImage2<PrecisionTYPE,float,short>(	targetImage,
												sourceImage,
												resultImage,
												positionField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT32:
					reg_resampleSourceImage2<PrecisionTYPE,float,unsigned int>(	targetImage,
													sourceImage,
													resultImage,
													positionField,
                                                    mask,
													interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT32:
					reg_resampleSourceImage2<PrecisionTYPE,float,int>(	targetImage,
												sourceImage,
												resultImage,
												positionField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT32:
					reg_resampleSourceImage2<PrecisionTYPE,float,float>(	targetImage,
												sourceImage,
												resultImage,
												positionField,
                                                    mask,
												interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_resampleSourceImage2<PrecisionTYPE,float,double>(	targetImage,
												sourceImage,
												resultImage,
												positionField,
                                                    mask,
												interp,
										bgValue);
					break;
				default:
					printf("Source pixel type unsupported.");
					break;
			}
			break;
		case NIFTI_TYPE_FLOAT64:
			switch ( sourceImage->datatype ){
				case NIFTI_TYPE_UINT8:
					reg_resampleSourceImage2<PrecisionTYPE,double,unsigned char>(	targetImage,
																	sourceImage,
																	resultImage,
																	positionField,
                                                    mask,
																	interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT8:
					reg_resampleSourceImage2<PrecisionTYPE,double,char>(	targetImage,
															sourceImage,
															resultImage,
															positionField,
                                                    mask,
															interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT16:
					reg_resampleSourceImage2<PrecisionTYPE,double,unsigned short>(	targetImage,
																	sourceImage,
																	resultImage,
																	positionField,
                                                    mask,
																	interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT16:
					reg_resampleSourceImage2<PrecisionTYPE,double,short>(	targetImage,
															sourceImage,
															resultImage,
															positionField,
                                                    mask,
															interp,
										bgValue);
					break;
				case NIFTI_TYPE_UINT32:
					reg_resampleSourceImage2<PrecisionTYPE,double,unsigned int>(	targetImage,
																	sourceImage,
																	resultImage,
																	positionField,
                                                    mask,
																	interp,
										bgValue);
					break;
				case NIFTI_TYPE_INT32:
					reg_resampleSourceImage2<PrecisionTYPE,double,int>(	targetImage,
														sourceImage,
														resultImage,
														positionField,
                                                    mask,
														interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT32:
					reg_resampleSourceImage2<PrecisionTYPE,double,float>(	targetImage,
															sourceImage,
															resultImage,
															positionField,
                                                    mask,
															interp,
										bgValue);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_resampleSourceImage2<PrecisionTYPE,double,double>(	targetImage,
															sourceImage,
															resultImage,
															positionField,
                                                    mask,
															interp,
										bgValue);
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
    if(MrPropreRules==true) free(mask);
}
template void reg_resampleSourceImage<float>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *,int, float);
template void reg_resampleSourceImage<double>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *,int, double);
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearGradientResultImage(	SourceTYPE *sourceCoefficients,
								  nifti_image *sourceImage,
								  nifti_image *positionField,
								  nifti_image *resultGradientImage,
                                    int *mask)
{
	GradientTYPE *resultGradientPtrX = static_cast<GradientTYPE *>(resultGradientImage->data);
	GradientTYPE *resultGradientPtrY = &resultGradientPtrX[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];
	GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];
	
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];
	FieldTYPE *positionFieldPtrZ = &positionFieldPtrY[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];

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

	for(int index=0;index<resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz; index++){

        PrecisionTYPE gradX=0.0;
        PrecisionTYPE gradY=0.0;
        PrecisionTYPE gradZ=0.0;

        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
		    PrecisionTYPE worldZ=(PrecisionTYPE) *positionFieldPtrZ;

		    /* real -> voxel; source space */
		    position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
		    worldZ*sourceIJKMatrix.m[0][2] +  sourceIJKMatrix.m[0][3];
		    position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
		    worldZ*sourceIJKMatrix.m[1][2] +  sourceIJKMatrix.m[1][3];
		    position[2] = worldX*sourceIJKMatrix.m[2][0] + worldY*sourceIJKMatrix.m[2][1] +
		    worldZ*sourceIJKMatrix.m[2][2] +  sourceIJKMatrix.m[2][3];

            if( position[0]>=0.0f && position[0]<sourceImage->nx-1 &&
                position[1]>=0.0f && position[1]<sourceImage->ny-1 &&
                position[2]>=0.0f && position[2]<sourceImage->nz-1 ){

		        previous[0] = (int)floor(position[0]);
		        previous[1] = (int)floor(position[1]);
		        previous[2] = (int)floor(position[2]);
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
						PrecisionTYPE zTempNewValue=0.0;
						for(short a=0; a<2; a++){
							const SourceTYPE coeff = *xyzPointer;
							xTempNewValue +=  (PrecisionTYPE)(coeff * deriv[a]);
							yTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
							zTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
							xyzPointer++;
						}
						xxTempNewValue += xTempNewValue * yBasis[b];
						yyTempNewValue += yTempNewValue * deriv[b];
						zzTempNewValue += zTempNewValue * yBasis[b];
				    }
				    gradX += xxTempNewValue * zBasis[c];
				    gradY += yyTempNewValue * zBasis[c];
				    gradZ += zzTempNewValue * deriv[c];
		        }
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
        positionFieldPtrX++;
        positionFieldPtrY++;
        positionFieldPtrZ++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearGradientResultImage2D(	SourceTYPE *sourceCoefficients,
                                        nifti_image *sourceImage,
                                        nifti_image *positionField,
                                        nifti_image *resultGradientImage,
                                        int *mask)
{
	GradientTYPE *resultGradientPtrX = static_cast<GradientTYPE *>(resultGradientImage->data);
	GradientTYPE *resultGradientPtrY = &resultGradientPtrX[resultGradientImage->nx*resultGradientImage->ny];
	
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultGradientImage->nx*resultGradientImage->ny];

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
	
	for(int index=0;index<resultGradientImage->nx*resultGradientImage->ny; index++){

        PrecisionTYPE gradX=0.0;
        PrecisionTYPE gradY=0.0;

        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;

		    /* real -> voxel; source space */
		    position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
		    sourceIJKMatrix.m[0][3];
		    position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
		    sourceIJKMatrix.m[1][3];

            if( position[0]>=0.0f && position[0]<sourceImage->nx-1 &&
                position[1]>=0.0f && position[1]<sourceImage->ny-1 ){

		        previous[0] = (int)floor(position[0]);
		        previous[1] = (int)floor(position[1]);
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
        positionFieldPtrX++;
        positionFieldPtrY++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineGradientResultImage(PrecisionTYPE *sourceCoefficients,
									nifti_image *sourceImage,
									nifti_image *positionField,
									nifti_image *resultGradientImage,
                                    int *mask)
{
	GradientTYPE *resultGradientPtrX = static_cast<GradientTYPE *>(resultGradientImage->data);
	GradientTYPE *resultGradientPtrY = &resultGradientPtrX[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];
	GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];

	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];
	FieldTYPE *positionFieldPtrZ = &positionFieldPtrY[resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz];

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
	else sourceIJKMatrix=sourceImage->qto_ijk;nifti_set_filenames(resultGradientImage, "gra.nii", 0, 0);
nifti_image_write(resultGradientImage);
exit(0);

	for(int index=0;index<resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz; index++){

        PrecisionTYPE gradX=0.0;
        PrecisionTYPE gradY=0.0;
        PrecisionTYPE gradZ=0.0;

        if((*maskPtr++)>-1){

		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
		    PrecisionTYPE worldZ=(PrecisionTYPE) *positionFieldPtrZ;

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
		    if(relative<0) relative=0.0; // rounding error correction
		    xBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    xBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - xBasis[3]);
		    xBasis[2]= (PrecisionTYPE)(relative + xBasis[0] - 2.0*xBasis[3]);
		    xBasis[1]= (PrecisionTYPE)(1.0 - xBasis[0] - xBasis[2] - xBasis[3]);
		    xDeriv[3]= (PrecisionTYPE)(relative * relative / 2.0);
		    xDeriv[0]= (PrecisionTYPE)(relative - 1.0/2.0 - xDeriv[3]);
		    xDeriv[2]= (PrecisionTYPE)(1.0 + xDeriv[0] - 2.0*xDeriv[3]);
		    xDeriv[1]= - xDeriv[0] - xDeriv[2] - xDeriv[3];
		    // basis values along the y axis
		    relative=position[1]-(PrecisionTYPE)previous[1];
		    if(relative<0) relative=0.0; // rounding error correction
		    yBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    yBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - yBasis[3]);
		    yBasis[2]= (PrecisionTYPE)(relative + yBasis[0] - 2.0*yBasis[3]);
		    yBasis[1]= (PrecisionTYPE)(1.0 - yBasis[0] - yBasis[2] - yBasis[3]);
		    yDeriv[3]= (PrecisionTYPE)(relative * relative / 2.0);
		    yDeriv[0]= (PrecisionTYPE)(relative - 1.0/2.0 - yDeriv[3]);
		    yDeriv[2]= (PrecisionTYPE)(1.0 + yDeriv[0] - 2.0*yDeriv[3]);
		    yDeriv[1]= - yDeriv[0] - yDeriv[2] - yDeriv[3];
		    // basis values along the z axis
		    relative=position[2]-(PrecisionTYPE)previous[2];
		    if(relative<0) relative=0.0; // rounding error correction
		    zBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    zBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - zBasis[3]);
		    zBasis[2]= (PrecisionTYPE)(relative + zBasis[0] - 2.0*zBasis[3]);
		    zBasis[1]= (PrecisionTYPE)(1.0 - zBasis[0] - zBasis[2] - zBasis[3]);
		    zDeriv[3]= (PrecisionTYPE)(relative * relative / 2.0);
		    zDeriv[0]= (PrecisionTYPE)(relative - 1.0/2.0 - zDeriv[3]);
		    zDeriv[2]= (PrecisionTYPE)(1.0 + zDeriv[0] - 2.0*zDeriv[3]);
		    zDeriv[1]= - zDeriv[0] - zDeriv[2] - zDeriv[3];

		    previous[0]--;previous[1]--;previous[2]--;

		    bool bg=false;
		    for(short c=0; c<4; c++){
			    short Z= previous[2]+c;
			    if(-1<Z && Z<sourceImage->nz){
				    PrecisionTYPE *zPointer = &sourceCoefficients[Z*sourceImage->nx*sourceImage->ny];
				    PrecisionTYPE xxTempNewValue=0.0;
				    PrecisionTYPE yyTempNewValue=0.0;
				    PrecisionTYPE zzTempNewValue=0.0;
				    for(short b=0; b<4; b++){
					    short Y= previous[1]+b;
					    PrecisionTYPE *yzPointer = &zPointer[Y*sourceImage->nx];
					    if(-1<Y && Y<sourceImage->ny){
						    PrecisionTYPE *xyzPointer = &yzPointer[previous[0]];
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
        positionFieldPtrX++;
        positionFieldPtrY++;
        positionFieldPtrZ++;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE, class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineGradientResultImage2D(PrecisionTYPE *sourceCoefficients,
									nifti_image *sourceImage,
									nifti_image *positionField,
									nifti_image *resultGradientImage,
                                    int *mask)
{
	GradientTYPE *resultGradientPtrX = static_cast<GradientTYPE *>(resultGradientImage->data);
	GradientTYPE *resultGradientPtrY = &resultGradientPtrX[resultGradientImage->nx*resultGradientImage->ny];
	
	FieldTYPE *positionFieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *positionFieldPtrY = &positionFieldPtrX[resultGradientImage->nx*resultGradientImage->ny];

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
	
	for(int index=0;index<resultGradientImage->nx*resultGradientImage->ny; index++){
		
        PrecisionTYPE gradX=0.0;
        PrecisionTYPE gradY=0.0;

        if((*maskPtr++)>-1){
		    PrecisionTYPE worldX=(PrecisionTYPE) *positionFieldPtrX;
		    PrecisionTYPE worldY=(PrecisionTYPE) *positionFieldPtrY;
		    
		    /* real -> voxel; source space */
		    position[0] = worldX*sourceIJKMatrix.m[0][0] + worldY*sourceIJKMatrix.m[0][1] +
		    sourceIJKMatrix.m[0][3];
		    position[1] = worldX*sourceIJKMatrix.m[1][0] + worldY*sourceIJKMatrix.m[1][1] +
		    sourceIJKMatrix.m[1][3];
		    
		    previous[0] = (int)floor(position[0]);
		    previous[1] = (int)floor(position[1]);
		    // basis values along the x axis
		    relative=position[0]-(PrecisionTYPE)previous[0];
		    if(relative<0) relative=0.0; // rounding error correction
		    xBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    xBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - xBasis[3]);
		    xBasis[2]= (PrecisionTYPE)(relative + xBasis[0] - 2.0*xBasis[3]);
		    xBasis[1]= (PrecisionTYPE)(1.0 - xBasis[0] - xBasis[2] - xBasis[3]);
		    xDeriv[3]= (PrecisionTYPE)(relative * relative / 2.0);
		    xDeriv[0]= (PrecisionTYPE)(relative - 1.0/2.0 - xDeriv[3]);
		    xDeriv[2]= (PrecisionTYPE)(1.0 + xDeriv[0] - 2.0*xDeriv[3]);
		    xDeriv[1]= - xDeriv[0] - xDeriv[2] - xDeriv[3];
		    // basis values along the y axis
		    relative=position[1]-(PrecisionTYPE)previous[1];
		    if(relative<0) relative=0.0; // rounding error correction
		    yBasis[3]= (PrecisionTYPE)(relative * relative * relative / 6.0);
		    yBasis[0]= (PrecisionTYPE)(1.0/6.0 + relative*(relative-1.0)/2.0 - yBasis[3]);
		    yBasis[2]= (PrecisionTYPE)(relative + yBasis[0] - 2.0*yBasis[3]);
		    yBasis[1]= (PrecisionTYPE)(1.0 - yBasis[0] - yBasis[2] - yBasis[3]);
		    yDeriv[3]= (PrecisionTYPE)(relative * relative / 2.0);
		    yDeriv[0]= (PrecisionTYPE)(relative - 1.0/2.0 - yDeriv[3]);
		    yDeriv[2]= (PrecisionTYPE)(1.0 + yDeriv[0] - 2.0*yDeriv[3]);
		    yDeriv[1]= - yDeriv[0] - yDeriv[2] - yDeriv[3];

		    previous[0]--;previous[1]--;

		    bool bg=false;
		    for(short b=0; b<4; b++){
			    short Y= previous[1]+b;
			    PrecisionTYPE *yPointer = &sourceCoefficients[Y*sourceImage->nx];
			    if(-1<Y && Y<sourceImage->ny){
				    PrecisionTYPE *xyPointer = &yPointer[previous[0]];
				    PrecisionTYPE xTempNewValue=0.0;
				    PrecisionTYPE yTempNewValue=0.0;
				    for(short a=0; a<4; a++){
					    if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
						    const PrecisionTYPE coeff = *xyPointer;
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
        positionFieldPtrX++;
        positionFieldPtrY++;
	}
}
/* *************************************************************** */
template <class PrecisionTYPE, class FieldTYPE, class SourceTYPE, class GradientTYPE>
void reg_getSourceImageGradient3(   nifti_image *targetImage,
                                    nifti_image *sourceImage,
                                    nifti_image *resultGradientImage,
                                    nifti_image *positionField,
                                    int *mask,
                                    int interp)
{
	/* The deformation field contains the position in the real world */

	if(interp==3){
		/* in order to apply a cubic Spline resampling, the source image
		 intensities have to be decomposed */
		SourceTYPE *intensityPtr = (SourceTYPE *)sourceImage->data;
	 	PrecisionTYPE *sourceCoefficients = (PrecisionTYPE *)malloc(sourceImage->nvox*sizeof(PrecisionTYPE));
		for(unsigned int i=0; i<sourceImage->nvox;i++)
			sourceCoefficients[i]=(PrecisionTYPE)intensityPtr[i];

		PrecisionTYPE pole = (PrecisionTYPE)(sqrt(3.0) - 2.0);

			// X axis first
		int number = sourceImage->nx;
		PrecisionTYPE *values=new PrecisionTYPE[number];
		int increment = 1;
		for(int i=0;i<sourceImage->ny*sourceImage->nz;i++){
			int start = i*sourceImage->nx;
			int end =  start + sourceImage->nx;
			extractLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
			intensitiesToSplineCoefficients<PrecisionTYPE>(values, number, pole);
			restoreLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
		}
		delete[] values;
		// Y axis then
		number = sourceImage->ny;
		values=new PrecisionTYPE[number];
		increment = sourceImage->nx;
		for(int i=0;i<sourceImage->nx*sourceImage->nz;i++){
			int start = i + i/sourceImage->nx * sourceImage->nx * (sourceImage->ny - 1);
			int end =  start + sourceImage->nx*sourceImage->ny;
			extractLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
			intensitiesToSplineCoefficients<PrecisionTYPE>(values, number, pole);
			restoreLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
		}
		delete[] values;
		if(targetImage->nz>1){
			// Z axis at last
			number = sourceImage->nz;
			values=new PrecisionTYPE[number];
			increment = sourceImage->nx*sourceImage->ny;
			for(int i=0;i<sourceImage->nx*sourceImage->ny;i++){
				int start = i;
				int end =  start + sourceImage->nx*sourceImage->ny*sourceImage->nz;
				extractLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
				intensitiesToSplineCoefficients<PrecisionTYPE>(values, number, pole);
				restoreLine<PrecisionTYPE>(start,end,increment,sourceCoefficients,values);
			}
			delete[] values;
		}

		if(targetImage->nz>1){
			CubicSplineGradientResultImage<PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(	sourceCoefficients,
																				  sourceImage,
																				  positionField,
																				  resultGradientImage,
                                                                                mask);
		}
		else{
			CubicSplineGradientResultImage2D<PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(	sourceCoefficients,
																						sourceImage,
																						positionField,
																						resultGradientImage,
                                                                                        mask);
		}
		free(sourceCoefficients);
	}
	else{ // trilinear interpolation [ by default ]
		SourceTYPE *intensityPtr = (SourceTYPE *)sourceImage->data;
		if(targetImage->nz>1){
			TrilinearGradientResultImage<PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(intensityPtr,
																				sourceImage,
																				positionField,
																				resultGradientImage,
                                                                                mask);
		}
		else{
			TrilinearGradientResultImage2D<PrecisionTYPE,SourceTYPE,GradientTYPE,FieldTYPE>(	intensityPtr,
																					sourceImage,
																					positionField,
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
								nifti_image *positionField,
                                int *mask,
								int interp
							)
{
	switch(resultGradientImage->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getSourceImageGradient3<PrecisionTYPE,FieldTYPE,SourceTYPE,float>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getSourceImageGradient3<PrecisionTYPE,FieldTYPE,SourceTYPE,double>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		default:
			printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template <class PrecisionTYPE, class FieldTYPE>
void reg_getSourceImageGradient1(nifti_image *targetImage,
								nifti_image *sourceImage,
								nifti_image *resultGradientImage,
								nifti_image *positionField,
                                int *mask,
								int interp
							)
{
	switch(sourceImage->datatype){
		case NIFTI_TYPE_UINT8:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,unsigned char>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_INT8:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,char>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,unsigned short>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_INT16:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,short>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,unsigned int>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_INT32:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,int>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,float>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getSourceImageGradient2<PrecisionTYPE,FieldTYPE,double>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		default:
			printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
extern "C++" template <class PrecisionTYPE>
void reg_getSourceImageGradient(	nifti_image *targetImage,
								nifti_image *sourceImage,
								nifti_image *resultGradientImage,
								nifti_image *positionField,
                                int *mask,
								int interp
							)
{
    // a mask array is created if no mask is specified
    bool MrPropreRule=false;
    if(mask==NULL){
        mask=(int *)calloc(targetImage->nvox,sizeof(int)); // voxels in the background are set to -1 so 0 will do the job here
        MrPropreRule=true;
    }

	switch(positionField->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getSourceImageGradient1<PrecisionTYPE,float>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getSourceImageGradient1<PrecisionTYPE,double>
				(targetImage,sourceImage,resultGradientImage,positionField,mask,interp);
			break;
		default:
			printf("err\treg_getSourceImageGradient\tDeformation field pixel type unsupported.");
			break;
	}
    if(MrPropreRule==true) free(mask);
}
/* *************************************************************** */
template void reg_getSourceImageGradient<float>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *, int);
template void reg_getSourceImageGradient<double>(nifti_image *, nifti_image *, nifti_image *, nifti_image *, int *, int);
/* *************************************************************** */
/* *************************************************************** */
template <class FieldTYPE, class JacobianTYPE>
void reg_getJacobianImage2(	nifti_image *positionField,
							nifti_image *jacobianImage)
{
	FieldTYPE *fieldPtrX = static_cast<FieldTYPE *>(positionField->data);
	FieldTYPE *fieldPtrY = &fieldPtrX[jacobianImage->nvox];
	FieldTYPE *fieldPtrZ = &fieldPtrY[jacobianImage->nvox];
	JacobianTYPE *jacobianPtr = static_cast<JacobianTYPE *>(jacobianImage->data);

	JacobianTYPE jacobianMatrix[3][3];

	int voxelIndex=0;
	for(int z=0; z<positionField->nz; z++){
		for(int y=0; y<positionField->ny; y++){
			for(int x=0; x<positionField->nx; x++){

				// derivative of along the X axis
				if(x==0){
					// forward difference
					jacobianMatrix[0][0]= (JacobianTYPE)((fieldPtrX[voxelIndex+1] - fieldPtrX[voxelIndex] ) / (positionField->dx));// Tx/dx
					jacobianMatrix[1][0]= (JacobianTYPE)((fieldPtrY[voxelIndex+1] - fieldPtrY[voxelIndex] ) / (positionField->dx));// Ty/dx
					jacobianMatrix[2][0]= (JacobianTYPE)((fieldPtrZ[voxelIndex+1] - fieldPtrZ[voxelIndex] ) / (positionField->dx));// Tz/dx
					
				}
				else if(x==positionField->nx-1){
					// backward difference
					jacobianMatrix[0][0]= (JacobianTYPE)((fieldPtrX[voxelIndex] - fieldPtrX[voxelIndex-1] ) / (positionField->dx));// Tx/dx
					jacobianMatrix[1][0]= (JacobianTYPE)((fieldPtrY[voxelIndex] - fieldPtrY[voxelIndex-1] ) / (positionField->dx));// Ty/dx
					jacobianMatrix[2][0]= (JacobianTYPE)((fieldPtrZ[voxelIndex] - fieldPtrZ[voxelIndex-1] ) / (positionField->dx));// Tz/dx
					
				}
				else{
					// symmetric derivative
					jacobianMatrix[0][0]= (JacobianTYPE)((fieldPtrX[voxelIndex+1] - fieldPtrX[voxelIndex-1] ) / (2.0*positionField->dx));// Tx/dx
					jacobianMatrix[1][0]= (JacobianTYPE)((fieldPtrY[voxelIndex+1] - fieldPtrY[voxelIndex-1] ) / (2.0*positionField->dx));// Ty/dx
					jacobianMatrix[2][0]= (JacobianTYPE)((fieldPtrZ[voxelIndex+1] - fieldPtrZ[voxelIndex-1] ) / (2.0*positionField->dx));// Tz/dx
				}

				// derivative of along the Y axis
				if(y==0){
					// forward difference
					jacobianMatrix[0][1]= (JacobianTYPE)((fieldPtrX[voxelIndex+positionField->nx] - fieldPtrX[voxelIndex] ) /
							(positionField->dy));// Tx/dy
					jacobianMatrix[1][1]= (JacobianTYPE)((fieldPtrY[voxelIndex+positionField->nx] - fieldPtrY[voxelIndex] ) /
							(positionField->dy));// Ty/dy
					jacobianMatrix[2][1]= (JacobianTYPE)((fieldPtrZ[voxelIndex+positionField->nx] - fieldPtrZ[voxelIndex] ) /
							(positionField->dy));// Tz/dy
					
				}
				else if(y==positionField->ny-1){
					// backward difference
					jacobianMatrix[0][1]= (JacobianTYPE)((fieldPtrX[voxelIndex] - fieldPtrX[voxelIndex-positionField->nx] ) /
							(positionField->dy));// Tx/dy
					jacobianMatrix[1][1]= (JacobianTYPE)((fieldPtrY[voxelIndex] - fieldPtrY[voxelIndex-positionField->nx] ) /
							(positionField->dy));// Ty/dy
					jacobianMatrix[2][1]= (JacobianTYPE)((fieldPtrZ[voxelIndex] - fieldPtrZ[voxelIndex-positionField->nx] ) /
							(positionField->dy));// Tz/dy
					
				}
				else{
					// symmetric derivative
					jacobianMatrix[0][1]= (JacobianTYPE)((fieldPtrX[voxelIndex+positionField->nx] - fieldPtrX[voxelIndex-positionField->nx] ) /
							(2.0*positionField->dy));// Tx/dy
					jacobianMatrix[1][1]= (JacobianTYPE)((fieldPtrY[voxelIndex+positionField->nx] - fieldPtrY[voxelIndex-positionField->nx] ) /
							(2.0*positionField->dy));// Ty/dy
					jacobianMatrix[2][1]= (JacobianTYPE)((fieldPtrZ[voxelIndex+positionField->nx] - fieldPtrZ[voxelIndex-positionField->nx] ) /
							(2.0*positionField->dy));// Tz/dy
				}

				// derivative of along the Z axis
				if(z==0){
					// forward difference
					jacobianMatrix[0][2]= (JacobianTYPE)((fieldPtrX[voxelIndex+positionField->nx*positionField->ny] -
							fieldPtrX[voxelIndex] ) /
							(positionField->dz));// Tx/dz
					jacobianMatrix[1][2]= (JacobianTYPE)((fieldPtrY[voxelIndex+positionField->nx*positionField->ny] -
							fieldPtrY[voxelIndex] ) /
							(positionField->dz));// Ty/dz
					jacobianMatrix[2][2]= (JacobianTYPE)((fieldPtrZ[voxelIndex+positionField->nx*positionField->ny] -
							fieldPtrZ[voxelIndex] ) /
							(positionField->dz));// Tz/dz
					
				}
				else if(z==positionField->nz-1){
					// backward difference
					jacobianMatrix[0][2]= (JacobianTYPE)((fieldPtrX[voxelIndex] -
							fieldPtrX[voxelIndex-positionField->nx*positionField->ny] ) /
							(positionField->dz));// Tx/dz
					jacobianMatrix[1][2]= (JacobianTYPE)((fieldPtrY[voxelIndex] -
							fieldPtrY[voxelIndex-positionField->nx*positionField->ny] ) /
							(positionField->dz));// Ty/dz
					jacobianMatrix[2][2]= (JacobianTYPE)((fieldPtrZ[voxelIndex] -
							fieldPtrZ[voxelIndex-positionField->nx*positionField->ny] ) /
							(positionField->dz));// Tz/dz
					
				}
				else{
					// symmetric derivative
					jacobianMatrix[0][2]= (JacobianTYPE)((fieldPtrX[voxelIndex+positionField->nx*positionField->ny] -
							fieldPtrX[voxelIndex-positionField->nx*positionField->ny] ) /
							(2.0*positionField->dz));// Tx/dz
					jacobianMatrix[1][2]= (JacobianTYPE)((fieldPtrY[voxelIndex+positionField->nx*positionField->ny] -
							fieldPtrY[voxelIndex-positionField->nx*positionField->ny] ) /
							(2.0*positionField->dz));// Ty/dz
					jacobianMatrix[2][2]= (JacobianTYPE)((fieldPtrZ[voxelIndex+positionField->nx*positionField->ny] -
							fieldPtrZ[voxelIndex-positionField->nx*positionField->ny] ) /
							(2.0*positionField->dz));// Tz/dz
				}

				JacobianTYPE jacobianValue = jacobianMatrix[0][0]*jacobianMatrix[1][1]*jacobianMatrix[2][2];
				jacobianValue += jacobianMatrix[0][1]*jacobianMatrix[1][2]*jacobianMatrix[2][0];
				jacobianValue += jacobianMatrix[0][2]*jacobianMatrix[1][0]*jacobianMatrix[2][1];

				jacobianValue -= jacobianMatrix[0][0]*jacobianMatrix[1][2]*jacobianMatrix[2][1];
				jacobianValue -= jacobianMatrix[0][1]*jacobianMatrix[1][0]*jacobianMatrix[2][2];
				jacobianValue -= jacobianMatrix[0][2]*jacobianMatrix[1][1]*jacobianMatrix[2][0];

				*jacobianPtr++ = jacobianValue;
				voxelIndex++;
			}
		}
	}
}
/* *************************************************************** */
template <class FieldTYPE>
void reg_getJacobianImage1(	nifti_image *positionField,
				nifti_image *jacobianImage)
{
	switch(jacobianImage->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getJacobianImage2<FieldTYPE,float>(positionField,jacobianImage);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getJacobianImage2<FieldTYPE,double>(positionField,jacobianImage);
			break;
		default:
			printf("err\treg_getSourceImageGradient\tJacobian image pixel type unsupported.");
			break;
	}
}
/* *************************************************************** */
void reg_getJacobianImage(	nifti_image *positionField,
				nifti_image *jacobianImage)
{
	switch(positionField->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getJacobianImage1<float>(positionField,jacobianImage);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getJacobianImage1<double>(positionField,jacobianImage);
			break;
		default:
			printf("err\treg_getSourceImageGradient\tDeformation field pixel type unsupported.");
			break;
	}
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_linearVelocityUpsampling_2D(nifti_image *image, int newDim[8], float newSpacing[8])
{
    /* The current image is stored and freed */
    ImageTYPE *currentValue = (ImageTYPE *)malloc(image->nvox * sizeof(ImageTYPE));
    memcpy(currentValue, image->data, image->nvox*image->nbyper);
    const int oldDim[8]={image->dim[0], image->dim[1], image->dim[2], image->dim[3],
        image->dim[4], image->dim[5], image->dim[6], image->dim[7]};

    free(image->data);

    for(int i=0;i<8;i++){
        image->dim[i]=newDim[i];
        image->pixdim[i]=newSpacing[i];
    }
    image->nx=image->dim[1];
    image->ny=image->dim[2];
    image->dx=image->pixdim[1];
    image->dy=image->pixdim[2];
    image->nvox=image->dim[1]*image->dim[2]*image->dim[3]*image->dim[4]*image->dim[5];
    image->data=(ImageTYPE *)malloc( image->nvox * sizeof(ImageTYPE) );

    ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);

    if(image->nt<1) image->nt=1;
    if(image->nu<1) image->nu=1;

	for(int ut=0;ut<image->nu*image->nt;ut++){
		ImageTYPE *newPtr = &imagePtr[ut*image->nx*image->ny];
		ImageTYPE *oldPtr = &currentValue[ut*oldDim[1]*oldDim[2]];
		for(int y=0; y<image->ny; y++){
			const int Y = (int)ceil((float)y/2.0f);
			for(int x=0; x<image->nx; x++){
				const int X = (int)ceil((float)x/2.0f);
				if(x/2 == X){
					if(y/2 == Y){
						*newPtr = (oldPtr[Y*oldDim[1]+X]
									 + oldPtr[Y*oldDim[1]+X+1]
									 + oldPtr[(Y+1)*oldDim[1]+X]
									 + oldPtr[(Y+1)*oldDim[1]+X+1]) /4.0f;
					}
					else{
						*newPtr = (oldPtr[Y*oldDim[1]+X]
									 + oldPtr[Y*oldDim[1]+X+1]) /2.0f;
					}
				}
				else{
					if(y/2 == Y){
						*newPtr = (oldPtr[Y*oldDim[1]+X]
									 + oldPtr[(Y+1)*oldDim[1]+X])/2.0f;
					}
					else{
						*newPtr = oldPtr[Y*oldDim[1]+X];
					}
				}
				newPtr++;
			}
		}
	}
	
	free(currentValue);
	return;
}
/* *************************************************************** */
template <class ImageTYPE>
void reg_linearVelocityUpsampling_3D(nifti_image *image, int newDim[8], float newSpacing[8])
{
	/* The current image is stored and freed */
	ImageTYPE *currentValue = (ImageTYPE *)malloc(image->nvox * sizeof(ImageTYPE));
	memcpy(currentValue, image->data, image->nvox*image->nbyper);
	const int oldDim[8]={image->dim[0], image->dim[1], image->dim[2], image->dim[3],
		image->dim[4], image->dim[5], image->dim[6], image->dim[7]};
	
    free(image->data);

    for(int i=0;i<8;i++){
        image->dim[i]=newDim[i];
        image->pixdim[i]=newSpacing[i];
    }
    image->nx=image->dim[1];
    image->ny=image->dim[2];
    image->nz=image->dim[3];
    image->dx=image->pixdim[1];
    image->dy=image->pixdim[2];
    image->dz=image->pixdim[3];
    image->nvox=image->dim[1]*image->dim[2]*image->dim[3]*image->dim[4]*image->dim[5];
    image->data=(ImageTYPE *)malloc( image->nvox * sizeof(ImageTYPE) );

	ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);
	
	if(image->nt<1) image->nt=1;
	if(image->nu<1) image->nu=1;
	
	for(int ut=0;ut<image->nu*image->nt;ut++){
		ImageTYPE *newPtr = &imagePtr[ut*image->nx*image->ny*image->nz];
		ImageTYPE *oldPtr = &currentValue[ut*oldDim[1]*oldDim[2]*oldDim[3]];
		for(int z=0; z<image->nz; z++){
			const int Z = (int)ceil((float)z/2.0f);
			if(z/2 == Z){
				for(int y=0; y<image->ny; y++){
					const int Y = (int)ceil((float)y/2.0f);
					if(y/2 == Y){
						for(int x=0; x<image->nx; x++){
							const int X = (int)ceil((float)x/2.0f);
							if(x/2 == X){
								// z' y' x' 
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[((Z+1)*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y+1)*oldDim[1]+X]+
											 oldPtr[((Z+1)*oldDim[2]+Y+1)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X+1]+
											 oldPtr[((Z+1)*oldDim[2]+Y)*oldDim[1]+X+1]+
											 oldPtr[(Z*oldDim[2]+Y+1)*oldDim[1]+X+1]+
											 oldPtr[((Z+1)*oldDim[2]+Y+1)*oldDim[1]+X+1] ) /8.0f;
							}
							else{ // (x/2==x)
								// z' y' x
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[((Z+1)*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y+1)*oldDim[1]+X]+
											 oldPtr[((Z+1)*oldDim[2]+Y+1)*oldDim[1]+X] ) /4.0f;
							} // (x/2==x)
						} // x loop
					}
					else{ // (y/2==Y)
						for(int x=0; x<image->nx; x++){
							const int X = (int)ceil((float)x/2.0f);
							if(x/2 == X){
								// z' y x'
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[((Z+1)*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X+1]+
											 oldPtr[((Z+1)*oldDim[2]+Y)*oldDim[1]+X+1] ) /4.0f;
							}
							else{ // (x/2==x)
								// z' y x
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[((Z+1)*oldDim[2]+Y)*oldDim[1]+X] ) /2.0f;
							} // (x/2==x)
						} // x loop
					} // (y/2==Y)
				} // y loop
			}
			else{ // (z/2==Z)
				for(int y=0; y<image->ny; y++){
					const int Y = (int)ceil((float)y/2.0f);
					if(y/2 == Y){
						for(int x=0; x<image->nx; x++){
							const int X = (int)ceil((float)x/2.0f);
							if(x/2 == X){
								// z y' x'
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y+1)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X+1]+
											 oldPtr[(Z*oldDim[2]+Y+1)*oldDim[1]+X+1] ) /4.0f;
							}
							else{ // (x/2==x)
								// z y' x
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]+
											 oldPtr[(Z*oldDim[2]+Y+1)*oldDim[1]+X] ) /2.0f;
							} // (x/2==x)
						} // x loop
					}
					else{ // (y/2==Y)
						for(int x=0; x<image->nx; x++){
							const int X = (int)ceil((float)x/2.0f);
							if(x/2 == X){
								// z y x'
								*newPtr++ = (oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X]
											 + oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X+1] ) /2.0f;
							}
							else{ // (x/2==x)
								// z y x
								*newPtr++ = oldPtr[(Z*oldDim[2]+Y)*oldDim[1]+X];
							} // (x/2==x)
						} // x loop
					} // (y/2==Y)
				} // y loop
			} // (z/2==Z)
		} // z loop
	} // ut
				

	
	free(currentValue);
	return;
}
/* *************************************************************** */
void reg_linearVelocityUpsampling(nifti_image *image, int newDim[8], float newSpacing[8])
{
	if(image->nz>1){
		switch(image->datatype){
			case NIFTI_TYPE_FLOAT32:
				reg_linearVelocityUpsampling_3D<float>(image, newDim, newSpacing);
				break;
			case NIFTI_TYPE_FLOAT64:
				reg_linearVelocityUpsampling_3D<double>(image, newDim, newSpacing);
				break;
			default:
				fprintf(stderr, "err\treg_linearVelocityUpsampling\tVoxel type unsupported.");
				break;
		}
	}
	else{
		switch(image->datatype){
			case NIFTI_TYPE_FLOAT32:
				reg_linearVelocityUpsampling_2D<float>(image, newDim, newSpacing);
				break;
			case NIFTI_TYPE_FLOAT64:
				reg_linearVelocityUpsampling_2D<double>(image, newDim, newSpacing);
				break;
			default:
				fprintf(stderr, "err\treg_linearVelocityUpsampling\tPixel type unsupported.");
				break;
		}
	}
}
/* *************************************************************** */
/* *************************************************************** */
#endif
