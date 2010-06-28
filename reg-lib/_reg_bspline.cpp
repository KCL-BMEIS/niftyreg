/*
 *  _reg_bspline.cpp
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_CPP
#define _REG_BSPLINE_CPP

#include "_reg_bspline.h"
#include <limits>

/* *************************************************************** */
/* *************************************************************** */

// No round() function available in windows.
#ifdef _WINDOWS
template<class PrecisionType>
int round(PrecisionType x)
{
   return int(x > 0.0 ? x + 0.5 : x - 0.5);
}
#endif

/* *************************************************************** */
/* *************************************************************** */

template<class PrecisionTYPE, class FieldTYPE>
void reg_bspline2D( nifti_image *splineControlPoint,
                    nifti_image *targetImage,
                    nifti_image *positionField,
                    int *mask,
                    int type)
{

#if _USE_SSE
    union u{
    __m128 m;
    float f[4];
    } val;
#else
    #ifdef _WINDOWS
        __declspec(align(16)) PrecisionTYPE temp[4];
    #else
        PrecisionTYPE temp[4] __attribute__((aligned(16)));
    #endif
#endif  

    FieldTYPE *controlPointPtrX = static_cast<FieldTYPE *>(splineControlPoint->data);
    FieldTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    FieldTYPE *fieldPtrX=static_cast<FieldTYPE *>(positionField->data);
    FieldTYPE *fieldPtrY=&fieldPtrX[targetImage->nx*targetImage->ny*targetImage->nz];

    int *maskPtr = &mask[0];

    PrecisionTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    PrecisionTYPE basis, FF, FFF, MF;

#ifdef _WINDOWS
    __declspec(align(16)) PrecisionTYPE yBasis[4];
    __declspec(align(16)) PrecisionTYPE xyBasis[16];

    __declspec(align(16)) PrecisionTYPE xControlPointCoordinates[16];
    __declspec(align(16)) PrecisionTYPE yControlPointCoordinates[16];
#else
    PrecisionTYPE yBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE xyBasis[16] __attribute__((aligned(16)));

    PrecisionTYPE xControlPointCoordinates[16] __attribute__((aligned(16)));
    PrecisionTYPE yControlPointCoordinates[16] __attribute__((aligned(16)));
#endif
	
    unsigned int coord;

    if(type == 2){ // Composition of deformation fields

		mat44 *targetMatrix_real_to_voxel;
		if(targetImage->sform_code>0)
			targetMatrix_real_to_voxel=&(targetImage->sto_ijk);
		else targetMatrix_real_to_voxel=&(targetImage->qto_ijk);

#ifdef _WINDOWS
		__declspec(align(16)) PrecisionTYPE xBasis[4];
#else
		PrecisionTYPE xBasis[4] __attribute__((aligned(16)));
#endif		
		// read the ijk sform or qform, as appropriate
		
        for(int y=0; y<positionField->ny; y++){
			for(int x=0; x<positionField->nx; x++){

				// The previous position at the current pixel position is read
				PrecisionTYPE xReal = (PrecisionTYPE)(*fieldPtrX);
				PrecisionTYPE yReal = (PrecisionTYPE)(*fieldPtrY);
				
				// From real to pixel position
				PrecisionTYPE xVoxel = targetMatrix_real_to_voxel->m[0][0]*xReal
					+ targetMatrix_real_to_voxel->m[0][1]*yReal + targetMatrix_real_to_voxel->m[0][3];
				PrecisionTYPE yVoxel = targetMatrix_real_to_voxel->m[1][0]*xReal
					+ targetMatrix_real_to_voxel->m[1][1]*yReal + targetMatrix_real_to_voxel->m[1][3];
				
				xVoxel = xVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:xVoxel;
				yVoxel = yVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:yVoxel;
				
				// The spline coefficients are computed
				int xPre=(int)((PrecisionTYPE)xVoxel/gridVoxelSpacing[0]);
				basis=(PrecisionTYPE)xVoxel/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
				if(basis<0.0) basis=0.0; //rounding error
				FF= basis*basis;
				FFF= FF*basis;
				MF=(PrecisionTYPE)(1.0-basis);
				xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
				xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
				xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
				xBasis[3] = (PrecisionTYPE)(FFF/6.0);
				
				int yPre=(int)((PrecisionTYPE)yVoxel/gridVoxelSpacing[1]);
				basis=(PrecisionTYPE)yVoxel/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
				if(basis<0.0) basis=0.0; //rounding error
				FF= basis*basis;
				FFF= FF*basis;
				MF=(PrecisionTYPE)(1.0-basis);
				yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
				yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
				yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
				yBasis[3] = (PrecisionTYPE)(FFF/6.0);
				
				// The control point postions are extracted
				coord=0;
				for(int Y=yPre; Y<yPre+4; Y++){
					unsigned int index=Y*splineControlPoint->nx;
					FieldTYPE *xPtr = &controlPointPtrX[index];
					FieldTYPE *yPtr = &controlPointPtrY[index];
					for(int X=xPre; X<xPre+4; X++){
						xControlPointCoordinates[coord] = (PrecisionTYPE)xPtr[X];
						yControlPointCoordinates[coord] = (PrecisionTYPE)yPtr[X];
						coord++;
					}
				}
				
				xReal=0.0;
                yReal=0.0;
				
                if(*maskPtr++>-1){
#if _USE_SSE
					coord=0;
                    for(unsigned int b=0; b<4; b++){
						for(unsigned int a=0; a<4; a++){
							xyBasis[coord++] = xBasis[a] * yBasis[b];
						}
					}
					
                    __m128 tempX =  _mm_set_ps1(0.0);
                    __m128 tempY =  _mm_set_ps1(0.0);
                    __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                    __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                    __m128 *ptrBasis   = (__m128 *) &xyBasis[0];
                    //addition and multiplication of the 16 basis value and CP position for each axis
                    for(unsigned int a=0; a<4; a++){
                        tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
                        tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
                        ptrBasis++;
                        ptrX++;
                        ptrY++;
                    }
                    //the values stored in SSE variables are transfered to normal float
                    val.m = tempX;
                    xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = tempY;
                    yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    for(unsigned int b=0; b<4; b++){
						for(unsigned int a=0; a<4; a++){
							PrecisionTYPE tempValue = xBasis[a] * yBasis[b];
							xReal += xControlPointCoordinates[b*4+a] * tempValue;
							yReal += yControlPointCoordinates[b*4+a] * tempValue;
						}
                    }
#endif
				}

				*fieldPtrX = (FieldTYPE)xReal;
				*fieldPtrY = (FieldTYPE)yReal;

                fieldPtrX++;
                fieldPtrY++;
			}
		}
    }
    else{
		
		mat44 *targetMatrix_voxel_to_real;
		if(targetImage->sform_code>0)
			targetMatrix_voxel_to_real=&(targetImage->sto_xyz);
		else targetMatrix_voxel_to_real=&(targetImage->qto_xyz);

		PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

        for(int y=0; y<positionField->ny; y++){

            int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
            basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
            yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            yBasis[3] = (PrecisionTYPE)(FFF/6.0);

            for(int x=0; x<positionField->nx; x++){

                int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
                basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                if(basis<0.0) basis=(PrecisionTYPE)(0.0); //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
#if _USE_SSE
                val.f[0] = (MF)*(MF)*(MF)/6.0;
                val.f[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
                val.f[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
                val.f[3] = FFF/6.0;
                __m128 tempCurrent=val.m;
                __m128* ptrBasis   = (__m128 *) &xyBasis[0];
                for(int a=0;a<4;a++){
                    val.m=_mm_set_ps1(yBasis[a]);
                    *ptrBasis=_mm_mul_ps(tempCurrent,val.m);
                    ptrBasis++;
                }
#else
                temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
                temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0);
                temp[3] = (PrecisionTYPE)(FFF/6.0);
                coord=0;
                for(int a=0;a<4;a++){
                    xyBasis[coord++]=temp[0]*yBasis[a];
                    xyBasis[coord++]=temp[1]*yBasis[a];
                    xyBasis[coord++]=temp[2]*yBasis[a];
                    xyBasis[coord++]=temp[3]*yBasis[a];
                }
#endif
                if(basis<=oldBasis || x==0){
                    coord=0;
                    for(int Y=yPre; Y<yPre+4; Y++){
                        unsigned int index=Y*splineControlPoint->nx;
                        FieldTYPE *xPtr = &controlPointPtrX[index];
                        FieldTYPE *yPtr = &controlPointPtrY[index];
                        for(int X=xPre; X<xPre+4; X++){
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xPtr[X];
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yPtr[X];
                            coord++;
                        }
                    }
                }
                oldBasis=basis;

                PrecisionTYPE xReal=0.0;
                PrecisionTYPE yReal=0.0;

                if(*maskPtr++>-1){
#if _USE_SSE
                    __m128 tempX =  _mm_set_ps1(0.0);
                    __m128 tempY =  _mm_set_ps1(0.0);
                    __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                    __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                    __m128 *ptrBasis   = (__m128 *) &xyBasis[0];
                    //addition and multiplication of the 64 basis value and CP displacement for each axis
                    for(unsigned int a=0; a<4; a++){
                        tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
                        tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
                        ptrBasis++;
                        ptrX++;
                        ptrY++;
                    }
                    //the values stored in SSE variables are transfered to normal float
                    val.m=tempX;
                    xReal=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m=tempY;
                    yReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    for(unsigned int i=0; i<16; i++){
                        xReal += xControlPointCoordinates[i] * xyBasis[i];
                        yReal += yControlPointCoordinates[i] * xyBasis[i];
                    }
#endif
                }// mask
                if(type==1){ // addition of deformation fields
					// The initial displacement needs to be extracted
					PrecisionTYPE xInitial = targetMatrix_voxel_to_real->m[0][0]*x
						+ targetMatrix_voxel_to_real->m[0][1]*y + targetMatrix_voxel_to_real->m[0][3];
					PrecisionTYPE yInitial = targetMatrix_voxel_to_real->m[1][0]*x
						+ targetMatrix_voxel_to_real->m[1][1]*y + targetMatrix_voxel_to_real->m[1][3];
					// The previous displacement is added to the computed position
                    *fieldPtrX = (FieldTYPE)(*fieldPtrX -xInitial) + (FieldTYPE)xReal;
                    *fieldPtrY = (FieldTYPE)(*fieldPtrY -yInitial) + (FieldTYPE)yReal;
                }
                else{ // starting from a blank deformation field
                    *fieldPtrX = (FieldTYPE)xReal;
                    *fieldPtrY = (FieldTYPE)yReal;
                }

                fieldPtrX++;
                fieldPtrY++;
            } // x
        } // y
    } // additive or blank deformation field

    return;
}
/* *************************************************************** */
template<class PrecisionTYPE, class FieldTYPE>
void reg_bspline3D( nifti_image *splineControlPoint,
                    nifti_image *targetImage,
                    nifti_image *positionField,
                    int *mask,
                    int type
                    )
{

#if _USE_SSE
    union u{
    __m128 m;
    float f[4];
    } val;
#else
    #ifdef _WINDOWS
        __declspec(align(16)) PrecisionTYPE temp[4];
    #else
        PrecisionTYPE temp[4] __attribute__((aligned(16)));
    #endif
#endif  

    FieldTYPE *controlPointPtrX = static_cast<FieldTYPE *>(splineControlPoint->data);
    FieldTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    FieldTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    
    FieldTYPE *fieldPtrX=static_cast<FieldTYPE *>(positionField->data);
    FieldTYPE *fieldPtrY=&fieldPtrX[positionField->nx*positionField->ny*positionField->nz];
    FieldTYPE *fieldPtrZ=&fieldPtrY[positionField->nx*positionField->ny*positionField->nz];

    int *maskPtr = &mask[0];
    
    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

#ifdef _WINDOWS
    __declspec(align(16)) PrecisionTYPE zBasis[4];
    __declspec(align(16)) PrecisionTYPE yzBasis[16];
    __declspec(align(16)) PrecisionTYPE xyzBasis[64];   

    __declspec(align(16)) PrecisionTYPE xControlPointCoordinates[64];
    __declspec(align(16)) PrecisionTYPE yControlPointCoordinates[64];
    __declspec(align(16)) PrecisionTYPE zControlPointCoordinates[64];
#else
    PrecisionTYPE zBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE yzBasis[16] __attribute__((aligned(16)));
    PrecisionTYPE xyzBasis[64] __attribute__((aligned(16)));    

    PrecisionTYPE xControlPointCoordinates[64] __attribute__((aligned(16)));
    PrecisionTYPE yControlPointCoordinates[64] __attribute__((aligned(16)));
    PrecisionTYPE zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif

    unsigned int coord;

    if(type == 2){ // Composition of deformation fields

		// read the ijk sform or qform, as appropriate
		mat44 *targetMatrix_real_to_voxel;
		if(targetImage->sform_code>0)
			targetMatrix_real_to_voxel=&(targetImage->sto_ijk);
		else targetMatrix_real_to_voxel=&(targetImage->qto_ijk);
		
#ifdef _WINDOWS
		__declspec(align(16)) PrecisionTYPE xBasis[4];
		__declspec(align(16)) PrecisionTYPE yBasis[4];
#else
		PrecisionTYPE xBasis[4] __attribute__((aligned(16)));
		PrecisionTYPE yBasis[4] __attribute__((aligned(16)));
#endif		
		
		for(int z=0; z<positionField->nz; z++){
			for(int y=0; y<positionField->ny; y++){
				for(int x=0; x<positionField->nx; x++){
					
					// The previous position at the current pixel position is read
					PrecisionTYPE xReal = (PrecisionTYPE)(*fieldPtrX);
					PrecisionTYPE yReal = (PrecisionTYPE)(*fieldPtrY);
					PrecisionTYPE zReal = (PrecisionTYPE)(*fieldPtrZ);

					// From real to pixel position
                    PrecisionTYPE xVoxel
                    = targetMatrix_real_to_voxel->m[0][0]*xReal
					+ targetMatrix_real_to_voxel->m[0][1]*yReal
					+ targetMatrix_real_to_voxel->m[0][2]*zReal
					+ targetMatrix_real_to_voxel->m[0][3];
                    PrecisionTYPE yVoxel
                    = targetMatrix_real_to_voxel->m[1][0]*xReal
					+ targetMatrix_real_to_voxel->m[1][1]*yReal
					+ targetMatrix_real_to_voxel->m[1][2]*zReal
					+ targetMatrix_real_to_voxel->m[1][3];
                    PrecisionTYPE zVoxel
                    = targetMatrix_real_to_voxel->m[2][0]*xReal
					+ targetMatrix_real_to_voxel->m[2][1]*yReal
					+ targetMatrix_real_to_voxel->m[2][2]*zReal
					+ targetMatrix_real_to_voxel->m[2][3];
					
					xVoxel = xVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:xVoxel;
					yVoxel = yVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:yVoxel;
                    zVoxel = zVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:zVoxel;

					// The spline coefficients are computed
					int xPre=(int)((PrecisionTYPE)xVoxel/gridVoxelSpacing[0]);
					basis=(PrecisionTYPE)xVoxel/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
					if(basis<0.0) basis=0.0; //rounding error
					FF= basis*basis;
					FFF= FF*basis;
					MF=(PrecisionTYPE)(1.0-basis);
					xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
					xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
					xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
					xBasis[3] = (PrecisionTYPE)(FFF/6.0);
					
					int yPre=(int)((PrecisionTYPE)yVoxel/gridVoxelSpacing[1]);
					basis=(PrecisionTYPE)yVoxel/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
					if(basis<0.0) basis=0.0; //rounding error
					FF= basis*basis;
					FFF= FF*basis;
					MF=(PrecisionTYPE)(1.0-basis);
					yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
					yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
					yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
					yBasis[3] = (PrecisionTYPE)(FFF/6.0);
					
					int zPre=(int)((PrecisionTYPE)zVoxel/gridVoxelSpacing[2]);
					basis=(PrecisionTYPE)zVoxel/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
					if(basis<0.0) basis=0.0; //rounding error
					FF= basis*basis;
					FFF= FF*basis;
					MF=(PrecisionTYPE)(1.0-basis);
					zBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
					zBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
					zBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
					zBasis[3] = (PrecisionTYPE)(FFF/6.0);
					
					// The control point postions are extracted
					coord=0;
					for(int Z=zPre; Z<zPre+4; Z++){
						int index = Z*splineControlPoint->nx*splineControlPoint->ny;
						FieldTYPE *xPtr = &controlPointPtrX[index];
						FieldTYPE *yPtr = &controlPointPtrY[index];
						FieldTYPE *zPtr = &controlPointPtrZ[index];
						for(int Y=yPre; Y<yPre+4; Y++){
							index = Y*splineControlPoint->nx;
							FieldTYPE *xxPtr = &xPtr[index];
							FieldTYPE *yyPtr = &yPtr[index];
							FieldTYPE *zzPtr = &zPtr[index];
							for(int X=xPre; X<xPre+4; X++){
									xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
									yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
									zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
								coord++;
							}
						}
					}
					
					xReal=0.0;
					yReal=0.0;
					zReal=0.0;
					
					if(*maskPtr++>-1){
	#if _USE_SSE
						coord=0;
						for(unsigned int c=0; c<4; c++){
							for(unsigned int b=0; b<4; b++){
								for(unsigned int a=0; a<4; a++){
									xyzBasis[coord++] = xBasis[a] * yBasis[b] * zBasis[c];
								}
							}
						}
						
						__m128 tempX =  _mm_set_ps1(0.0);
						__m128 tempY =  _mm_set_ps1(0.0);
						__m128 tempZ =  _mm_set_ps1(0.0);
						__m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
						__m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
						__m128 *ptrZ = (__m128 *) &yControlPointCoordinates[0];
						__m128 *ptrBasis   = (__m128 *) &xyzBasis[0];
						//addition and multiplication of the 16 basis value and CP position for each axis
						for(unsigned int a=0; a<16; a++){
							tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
							tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
							tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrZ), tempZ );
							ptrBasis++;
							ptrX++;
							ptrY++;
							ptrZ++;
						}
						//the values stored in SSE variables are transfered to normal float
						val.m = tempX;
						xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
						val.m = tempY;
						yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
						val.m = tempZ;
						zReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
	#else
						for(unsigned int c=0; c<4; c++){
							for(unsigned int b=0; b<4; b++){
								for(unsigned int a=0; a<4; a++){
									PrecisionTYPE tempValue = xBasis[a] * yBasis[b] * zBasis[c];
									unsigned int index=(4*c+b)*4+a;
									xReal += xControlPointCoordinates[index] * tempValue;
									yReal += yControlPointCoordinates[index] * tempValue;
									zReal += zControlPointCoordinates[index] * tempValue;
								}
							}
						}
	#endif
					}
					
					*fieldPtrX++ = (FieldTYPE)xReal;
					*fieldPtrY++ = (FieldTYPE)yReal;
					*fieldPtrZ++ = (FieldTYPE)zReal;
				}
			}
		}
    }
    else{

		mat44 *targetMatrix_voxel_to_real;
		if(targetImage->sform_code>0)
			targetMatrix_voxel_to_real=&(targetImage->sto_xyz);
		else targetMatrix_voxel_to_real=&(targetImage->qto_xyz);
		
        for(int z=0; z<positionField->nz; z++){

            int zPre=(int)((PrecisionTYPE)z/gridVoxelSpacing[2]);
            basis=(PrecisionTYPE)z/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            zBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
            zBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            zBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            zBasis[3] = (PrecisionTYPE)(FFF/6.0);
        
            for(int y=0; y<positionField->ny; y++){

                int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
                basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
                if(basis<0.0) basis=(PrecisionTYPE)(0.0); //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
#if _USE_SSE
                val.f[0] = (MF)*(MF)*(MF)/6.0;
                val.f[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
                val.f[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
                val.f[3] = FFF/6.0;
                __m128 tempCurrent=val.m;
                __m128* ptrBasis   = (__m128 *) &yzBasis[0];
                for(int a=0;a<4;a++){
                    val.m=_mm_set_ps1(zBasis[a]);
                    *ptrBasis=_mm_mul_ps(tempCurrent,val.m);
                    ptrBasis++;
                }
#else
                temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
                temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0);
                temp[3] = (PrecisionTYPE)(FFF/6.0);
                coord=0;
                for(int a=0;a<4;a++){
                    yzBasis[coord++]=temp[0]*zBasis[a];
                    yzBasis[coord++]=temp[1]*zBasis[a];
                    yzBasis[coord++]=temp[2]*zBasis[a];
                    yzBasis[coord++]=temp[3]*zBasis[a];
                }           
#endif

                for(int x=0; x<positionField->nx; x++){

                    int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
                    basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                    if(basis<0.0) basis=0.0; //rounding error
                    FF= basis*basis;
                    FFF= FF*basis;
                    MF=(PrecisionTYPE)(1.0-basis);
#if _USE_SSE
                    val.f[0] = (MF)*(MF)*(MF)/6.0;
                    val.f[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
                    val.f[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
                    val.f[3] = FFF/6.0;
                    tempCurrent=val.m;          
                    ptrBasis   = (__m128 *) &xyzBasis[0];
                    for(int a=0;a<16;++a){
                        val.m=_mm_set_ps1(yzBasis[a]);
                        *ptrBasis=_mm_mul_ps(tempCurrent,val.m);
                        ptrBasis++;
                    }
#else
                    temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
                    temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                    temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0);
                    temp[3] = (PrecisionTYPE)(FFF/6.0);
                    coord=0;
                    for(int a=0;a<16;a++){
                        xyzBasis[coord++]=temp[0]*yzBasis[a];
                        xyzBasis[coord++]=temp[1]*yzBasis[a];
                        xyzBasis[coord++]=temp[2]*yzBasis[a];
                        xyzBasis[coord++]=temp[3]*yzBasis[a];
                    }
#endif
                    if(basis<=oldBasis || x==0){
						coord=0;
						for(int Z=zPre; Z<zPre+4; Z++){
							int index = Z*splineControlPoint->nx*splineControlPoint->ny;
							FieldTYPE *xPtr = &controlPointPtrX[index];
							FieldTYPE *yPtr = &controlPointPtrY[index];
							FieldTYPE *zPtr = &controlPointPtrZ[index];
							for(int Y=yPre; Y<yPre+4; Y++){
								index = Y*splineControlPoint->nx;
								FieldTYPE *xxPtr = &xPtr[index];
								FieldTYPE *yyPtr = &yPtr[index];
								FieldTYPE *zzPtr = &zPtr[index];
								for(int X=xPre; X<xPre+4; X++){
									xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
									yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
									zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
									coord++;
								}
							}
						}
					}
                    oldBasis=basis;

                    PrecisionTYPE xReal=0.0;
                    PrecisionTYPE yReal=0.0;
                    PrecisionTYPE zReal=0.0;

                    if(*maskPtr++>-1){
#if _USE_SSE
                        __m128 tempX =  _mm_set_ps1(0.0);
                        __m128 tempY =  _mm_set_ps1(0.0);
                        __m128 tempZ =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];
                        __m128 *ptrBasis   = (__m128 *) &xyzBasis[0];
                        //addition and multiplication of the 64 basis value and CP displacement for each axis
                        for(unsigned int a=0; a<16; a++){
                            tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
                            tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
                            tempZ = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrZ), tempZ );
                            ptrBasis++;
                            ptrX++;
                            ptrY++;
                            ptrZ++;
                        }
                        //the values stored in SSE variables are transfered to normal float
                        val.m=tempX;
                        xReal=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m=tempY;
                        yReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m=tempZ;
                        zReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        for(unsigned int i=0; i<64; i++){
                            xReal += xControlPointCoordinates[i] * xyzBasis[i];
                            yReal += yControlPointCoordinates[i] * xyzBasis[i];
                            zReal += zControlPointCoordinates[i] * xyzBasis[i];
                        }
#endif
                    }// mask
                    if(type==1){ // addition of deformation fields
						// The initial displacement needs to be extracted
						PrecisionTYPE xInitial = targetMatrix_voxel_to_real->m[0][0]*x
						+ targetMatrix_voxel_to_real->m[0][1]*y
						+ targetMatrix_voxel_to_real->m[0][2]*z
						+ targetMatrix_voxel_to_real->m[0][3];
						PrecisionTYPE yInitial = targetMatrix_voxel_to_real->m[1][0]*x
						+ targetMatrix_voxel_to_real->m[1][1]*y
						+ targetMatrix_voxel_to_real->m[1][2]*z
						+ targetMatrix_voxel_to_real->m[1][3];
						PrecisionTYPE zInitial = targetMatrix_voxel_to_real->m[2][0]*x
						+ targetMatrix_voxel_to_real->m[2][1]*y
						+ targetMatrix_voxel_to_real->m[2][2]*z
						+ targetMatrix_voxel_to_real->m[2][3];
						
						// The previous displacement is added to the computed position
						*fieldPtrX = (FieldTYPE)(*fieldPtrX -xInitial) + (FieldTYPE)xReal;
						*fieldPtrY = (FieldTYPE)(*fieldPtrY -yInitial) + (FieldTYPE)yReal;
						*fieldPtrZ = (FieldTYPE)(*fieldPtrZ -zInitial) + (FieldTYPE)zReal;
                        *fieldPtrX += (FieldTYPE)xReal;
                        *fieldPtrY += (FieldTYPE)yReal;
                        *fieldPtrZ += (FieldTYPE)zReal;
                    }
                    else{ // starting from a blank deformation field
                        *fieldPtrX = (FieldTYPE)xReal;
                        *fieldPtrY = (FieldTYPE)yReal;
                        *fieldPtrZ = (FieldTYPE)zReal;
                    }

                    fieldPtrX++;
                    fieldPtrY++;
                    fieldPtrZ++;
                } // x
            } // y
        } // z
    } // additive or blank deformation field
    
    return;
}
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_bspline(   nifti_image *splineControlPoint,
                    nifti_image *targetImage,
                    nifti_image *positionField,
                    int *mask,
                    int type)
{
#if _USE_SSE
	if(sizeof(PrecisionTYPE) != sizeof(float)){
		printf("SSE computation has only been implemented for single precision.\n");
		printf("The deformation field is not computed\n");
		return;
	}
#endif
	if(splineControlPoint->datatype != positionField->datatype){
		printf("The spline control point image and the deformation field image are expected to be the same type\n");
		printf("The deformation field is not computed\n");
		return;	
	}
    bool MrPropre=false;
    if(mask==NULL){
        // Active voxel are all superior to -1, 0 thus will do !
        MrPropre=true;
        mask=(int *)calloc(targetImage->nvox, sizeof(int));
    }

    if(splineControlPoint->nz==1){
        switch(positionField->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline2D<PrecisionTYPE, float>(splineControlPoint, targetImage, positionField, mask, type);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline2D<PrecisionTYPE, double>(splineControlPoint, targetImage, positionField, mask, type);
                break;
            default:
                printf("Only single or double precision is implemented for deformation field\n");
                printf("The deformation field is not computed\n");
                break;
        }
    }
    else{
        switch(positionField->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline3D<PrecisionTYPE, float>(splineControlPoint, targetImage, positionField, mask, type);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline3D<PrecisionTYPE, double>(splineControlPoint, targetImage, positionField, mask, type);
                break;
            default:
                printf("Only single or double precision is implemented for deformation field\n");
                printf("The deformation field is not computed\n");
                break;
        }
    }
    if(MrPropre==true) free(mask);
	return;
}
template void reg_bspline<float>(nifti_image *, nifti_image *, nifti_image *, int *, int);
template void reg_bspline<double>(nifti_image *, nifti_image *, nifti_image *, int *, int);
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_bendingEnergyValue2D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    PrecisionTYPE temp[4],first[4],second[4];
    PrecisionTYPE yBasis[4],yFirst[4],ySecond[4];
    PrecisionTYPE basisXX[16], basisYY[16], basisXY[16];
    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

    PrecisionTYPE xControlPointCoordinates[16];
    PrecisionTYPE yControlPointCoordinates[16];

    PrecisionTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;
    
    PrecisionTYPE constraintValue=0;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
        basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(PrecisionTYPE)(1.0-basis);
        yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
        yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        yBasis[3] = (PrecisionTYPE)(FFF/6.0);
        yFirst[3] = (PrecisionTYPE)(FF / 2.0);
        yFirst[0] = (PrecisionTYPE)(basis - 1.0/2.0 - yFirst[3]);
        yFirst[2] = (PrecisionTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
        yFirst[1] = (PrecisionTYPE)(- yFirst[0] - yFirst[2] - yFirst[3]);
        ySecond[3]= basis;
        ySecond[0]= (PrecisionTYPE)(1.0 - ySecond[3]);
        ySecond[2]= (PrecisionTYPE)(ySecond[0] - 2.0*ySecond[3]);
        ySecond[1]= - ySecond[0] - ySecond[2] - ySecond[3];

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
            basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (PrecisionTYPE)(FFF/6.0);
            first[3]= (PrecisionTYPE)(FF / 2.0);
            first[0]= (PrecisionTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (PrecisionTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= - first[0] - first[2] - first[3];
            second[3]= basis;
            second[0]= (PrecisionTYPE)(1.0 - second[3]);
            second[2]= (PrecisionTYPE)(second[0] - 2.0*second[3]);
            second[1]= - second[0] - second[2] - second[3];


            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisXX[coord]=yBasis[b]*second[a];    // y * x"
                    basisYY[coord]=ySecond[b]*temp[a];      // y" * x
                    basisXY[coord]=yFirst[b]*first[a];     // y' * x'
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                coord=0;
				for(int Y=yPre; Y<yPre+4; Y++){
					unsigned int index=Y*splineControlPoint->nx;
					SplineTYPE *xPtr = &controlPointPtrX[index];
					SplineTYPE *yPtr = &controlPointPtrY[index];
					for(int X=xPre; X<xPre+4; X++){
						xControlPointCoordinates[coord] = (PrecisionTYPE)xPtr[X];
						yControlPointCoordinates[coord] = (PrecisionTYPE)yPtr[X];
						coord++;
					}
				}
            }

            PrecisionTYPE XX_x=0.0;
            PrecisionTYPE YY_x=0.0;
            PrecisionTYPE XY_x=0.0;
            PrecisionTYPE XX_y=0.0;
            PrecisionTYPE YY_y=0.0;
            PrecisionTYPE XY_y=0.0;

            for(int a=0; a<16; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }

            constraintValue += (PrecisionTYPE)(XX_x*XX_x + YY_x*YY_x + 2.0*XY_x*XY_x);
            constraintValue += (PrecisionTYPE)(XX_y*XX_y + YY_y*YY_y + 2.0*XY_y*XY_y);
        }
    }

    return (PrecisionTYPE)(constraintValue/(2.0*targetImage->nx*targetImage->ny));
        
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_bendingEnergyValue3D( nifti_image *splineControlPoint,
                        nifti_image *targetImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny];

    PrecisionTYPE temp[4],first[4],second[4];
    PrecisionTYPE zBasis[4],zFirst[4],zSecond[4];
    PrecisionTYPE tempXX[16], tempYY[16], tempZZ[16], tempXY[16], tempYZ[16], tempXZ[16];
    PrecisionTYPE basisXX[64], basisYY[64], basisZZ[64], basisXY[64], basisYZ[64], basisXZ[64];
    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

    PrecisionTYPE xControlPointCoordinates[64];
    PrecisionTYPE yControlPointCoordinates[64];
    PrecisionTYPE zControlPointCoordinates[64];

    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    unsigned int coord=0;

    PrecisionTYPE constraintValue=0;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((PrecisionTYPE)z/gridVoxelSpacing[2]);
        basis=(PrecisionTYPE)z/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(PrecisionTYPE)(1.0-basis);
        zBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
        zBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        zBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        zBasis[3] = (PrecisionTYPE)(FFF/6.0);
        zFirst[3] = (PrecisionTYPE)(FF / 2.0);
        zFirst[0] = (PrecisionTYPE)(basis - 1.0/2.0 - zFirst[3]);
        zFirst[2] = (PrecisionTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
        zFirst[1] = (PrecisionTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);
        zSecond[3]= basis;
        zSecond[0]= (PrecisionTYPE)(1.0 - zSecond[3]);
        zSecond[2]= (PrecisionTYPE)(zSecond[0] - 2.0*zSecond[3]);
        zSecond[1]= - zSecond[0] - zSecond[2] - zSecond[3];

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
            basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (PrecisionTYPE)(FFF/6.0);
            first[3]= (PrecisionTYPE)(FF / 2.0);
            first[0]= (PrecisionTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (PrecisionTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= - first[0] - first[2] - first[3];
            second[3]= basis;
            second[0]= (PrecisionTYPE)(1.0 - second[3]);
            second[2]= (PrecisionTYPE)(second[0] - 2.0*second[3]);
            second[1]= - second[0] - second[2] - second[3];

            coord=0;
            for(int c=0; c<4; c++){
                for(int b=0; b<4; b++){
                    tempXX[coord]=zBasis[c]*temp[b];    // z * y
                    tempYY[coord]=zBasis[c]*second[b];  // z * y"
                    tempZZ[coord]=zSecond[c]*temp[b];   // z" * y
                    tempXY[coord]=zBasis[c]*first[b];   // z * y'
                    tempYZ[coord]=zFirst[c]*first[b];   // z' * y'
                    tempXZ[coord]=zFirst[c]*temp[b];    // z' * y
                    coord++;
                }
            }

            for(int x=0; x<targetImage->nx; x++){

                int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
                basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
                temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                temp[3] = (PrecisionTYPE)(FFF/6.0);
                first[3]= (PrecisionTYPE)(FF / 2.0);
                first[0]= (PrecisionTYPE)(basis - 1.0/2.0 - first[3]);
                first[2]= (PrecisionTYPE)(1.0 + first[0] - 2.0*first[3]);
                first[1]= - first[0] - first[2] - first[3];
                second[3]= basis;
                second[0]= (PrecisionTYPE)(1.0 - second[3]);
                second[2]= (PrecisionTYPE)(second[0] - 2.0*second[3]);
                second[1]= - second[0] - second[2] - second[3];

                coord=0;
                for(int bc=0; bc<16; bc++){
                    for(int a=0; a<4; a++){
                        basisXX[coord]=tempXX[bc]*second[a];    // z * y * x"
                        basisYY[coord]=tempYY[bc]*temp[a];      // z * y" * x
                        basisZZ[coord]=tempZZ[bc]*temp[a];      // z" * y * x
                        basisXY[coord]=tempXY[bc]*first[a];     // z * y' * x'
                        basisYZ[coord]=tempYZ[bc]*temp[a];      // z' * y' * x
                        basisXZ[coord]=tempXZ[bc]*first[a];     // z' * y * x'
                        coord++;
                    }
                }

                if(basis<=oldBasis || x==0){
                    coord=0;
                    for(int Z=zPre; Z<zPre+4; Z++){
                        unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                        SplineTYPE *xPtr = &controlPointPtrX[index];
                        SplineTYPE *yPtr = &controlPointPtrY[index];
                        SplineTYPE *zPtr = &controlPointPtrZ[index];
                        for(int Y=yPre; Y<yPre+4; Y++){
                            index = Y*splineControlPoint->nx;
                            SplineTYPE *xxPtr = &xPtr[index];
                            SplineTYPE *yyPtr = &yPtr[index];
                            SplineTYPE *zzPtr = &zPtr[index];
                            for(int X=xPre; X<xPre+4; X++){
                                xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                                yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                                zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;

                PrecisionTYPE XX_x=0.0;
                PrecisionTYPE YY_x=0.0;
                PrecisionTYPE ZZ_x=0.0;
                PrecisionTYPE XY_x=0.0;
                PrecisionTYPE YZ_x=0.0;
                PrecisionTYPE XZ_x=0.0;
                PrecisionTYPE XX_y=0.0;
                PrecisionTYPE YY_y=0.0;
                PrecisionTYPE ZZ_y=0.0;
                PrecisionTYPE XY_y=0.0;
                PrecisionTYPE YZ_y=0.0;
                PrecisionTYPE XZ_y=0.0;
                PrecisionTYPE XX_z=0.0;
                PrecisionTYPE YY_z=0.0;
                PrecisionTYPE ZZ_z=0.0;
                PrecisionTYPE XY_z=0.0;
                PrecisionTYPE YZ_z=0.0;
                PrecisionTYPE XZ_z=0.0;

                for(int a=0; a<64; a++){
                    XX_x += basisXX[a]*xControlPointCoordinates[a];
                    YY_x += basisYY[a]*xControlPointCoordinates[a];
                    ZZ_x += basisZZ[a]*xControlPointCoordinates[a];
                    XY_x += basisXY[a]*xControlPointCoordinates[a];
                    YZ_x += basisYZ[a]*xControlPointCoordinates[a];
                    XZ_x += basisXZ[a]*xControlPointCoordinates[a];
                                    
                    XX_y += basisXX[a]*yControlPointCoordinates[a];
                    YY_y += basisYY[a]*yControlPointCoordinates[a];
                    ZZ_y += basisZZ[a]*yControlPointCoordinates[a];
                    XY_y += basisXY[a]*yControlPointCoordinates[a];
                    YZ_y += basisYZ[a]*yControlPointCoordinates[a];
                    XZ_y += basisXZ[a]*yControlPointCoordinates[a];
                                    
                    XX_z += basisXX[a]*zControlPointCoordinates[a];
                    YY_z += basisYY[a]*zControlPointCoordinates[a];
                    ZZ_z += basisZZ[a]*zControlPointCoordinates[a];
                    XY_z += basisXY[a]*zControlPointCoordinates[a];
                    YZ_z += basisYZ[a]*zControlPointCoordinates[a];
                    XZ_z += basisXZ[a]*zControlPointCoordinates[a];
                }
                
                constraintValue += (PrecisionTYPE)(XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x));
                constraintValue += (PrecisionTYPE)(XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y));
                constraintValue += (PrecisionTYPE)(XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z));
            }
        }
    }

    return (PrecisionTYPE)(constraintValue/(3.0*targetImage->nx*targetImage->ny*targetImage->nz));
        
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_bendingEnergyApproxValue2D(   nifti_image *splineControlPoint,
                                                        nifti_image *targetImage)
{
	
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    // As the contraint is only computed at the control point positions, the basis value of the spline are always the same 
    PrecisionTYPE normal[3];
    PrecisionTYPE first[3];
    PrecisionTYPE second[3];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);
    second[0] = (PrecisionTYPE)(1.0);
    second[1] = (PrecisionTYPE)(-2.0);
    second[2] = (PrecisionTYPE)(1.0);

    int coord=0;
    PrecisionTYPE constraintValue=0.0;

    PrecisionTYPE basisXX[9], basisYY[9], basisXY[9];

    coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisXX[coord]=normal[b]*second[a];   // y * x"
            basisYY[coord]=second[b]*normal[a];   // y" * x
            basisXY[coord]=first[b]*first[a];     // y' * x'
            coord++;
        }
    }

    PrecisionTYPE xControlPointCoordinates[9];
    PrecisionTYPE yControlPointCoordinates[9];

    for(int y=1;y<splineControlPoint->ny-1;y++){
        for(int x=1;x<splineControlPoint->nx-1;x++){

            coord=0;
            for(int Y=y-1; Y<y+2; Y++){
                unsigned int index=Y*splineControlPoint->nx;
                SplineTYPE *xPtr = &controlPointPtrX[index];
                SplineTYPE *yPtr = &controlPointPtrY[index];
                for(int X=x-1; X<x+2; X++){
                    xControlPointCoordinates[coord] = (PrecisionTYPE)xPtr[X];
                    yControlPointCoordinates[coord] = (PrecisionTYPE)yPtr[X];
                    coord++;
                }
            }

            PrecisionTYPE XX_x=0.0;
            PrecisionTYPE YY_x=0.0;
            PrecisionTYPE XY_x=0.0;
            PrecisionTYPE XX_y=0.0;
            PrecisionTYPE YY_y=0.0;
            PrecisionTYPE XY_y=0.0;

            for(int a=0; a<9; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }

            constraintValue += (PrecisionTYPE)(XX_x*XX_x + YY_x*YY_x + 2.0*XY_x*XY_x);
            constraintValue += (PrecisionTYPE)(XX_y*XX_y + YY_y*YY_y + 2.0*XY_y*XY_y);
        }
    }
    return (PrecisionTYPE)(constraintValue/(2.0*splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz));
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_bendingEnergyApproxValue3D(   nifti_image *splineControlPoint,
                            nifti_image *targetImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    // As the contraint is only computed at the control point positions, the basis value of the spline are always the same 
    PrecisionTYPE normal[3];
    PrecisionTYPE first[3];
    PrecisionTYPE second[3];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);
    second[0] = (PrecisionTYPE)(1.0);
    second[1] = (PrecisionTYPE)(-2.0);
    second[2] = (PrecisionTYPE)(1.0);
    
    int coord=0;
    PrecisionTYPE constraintValue=0.0;

    // There are six different values taken into account
    PrecisionTYPE tempXX[9], tempYY[9], tempZZ[9], tempXY[9], tempYZ[9], tempXZ[9];
    
    coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempXX[coord]=normal[c]*normal[b];  // z * y
            tempYY[coord]=normal[c]*second[b];  // z * y"
            tempZZ[coord]=second[c]*normal[b];  // z" * y
            tempXY[coord]=normal[c]*first[b];   // z * y'
            tempYZ[coord]=first[c]*first[b];    // z' * y'
            tempXZ[coord]=first[c]*normal[b];   // z' * y
            coord++;
        }
    }
    
    PrecisionTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    
    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisXX[coord]=tempXX[bc]*second[a];    // z * y * x"
            basisYY[coord]=tempYY[bc]*normal[a];    // z * y" * x
            basisZZ[coord]=tempZZ[bc]*normal[a];    // z" * y * x
            basisXY[coord]=tempXY[bc]*first[a]; // z * y' * x'
            basisYZ[coord]=tempYZ[bc]*normal[a];    // z' * y' * x
            basisXZ[coord]=tempXZ[bc]*first[a]; // z' * y * x'
            coord++;
        }
    }
    
    PrecisionTYPE xControlPointCoordinates[27];
    PrecisionTYPE yControlPointCoordinates[27];
    PrecisionTYPE zControlPointCoordinates[27];
    
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            for(int x=1;x<splineControlPoint->nx-1;x++){
                
                coord=0;
                for(int Z=z-1; Z<z+2; Z++){
                    unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                    SplineTYPE *xPtr = &controlPointPtrX[index];
                    SplineTYPE *yPtr = &controlPointPtrY[index];
                    SplineTYPE *zPtr = &controlPointPtrZ[index];
                    for(int Y=y-1; Y<y+2; Y++){
                        index = Y*splineControlPoint->nx;
                        SplineTYPE *xxPtr = &xPtr[index];
                        SplineTYPE *yyPtr = &yPtr[index];
                        SplineTYPE *zzPtr = &zPtr[index];
                        for(int X=x-1; X<x+2; X++){
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                            zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                            coord++;
                        }
                    }
                }
                
                PrecisionTYPE XX_x=0.0;
                PrecisionTYPE YY_x=0.0;
                PrecisionTYPE ZZ_x=0.0;
                PrecisionTYPE XY_x=0.0;
                PrecisionTYPE YZ_x=0.0;
                PrecisionTYPE XZ_x=0.0;
                PrecisionTYPE XX_y=0.0;
                PrecisionTYPE YY_y=0.0;
                PrecisionTYPE ZZ_y=0.0;
                PrecisionTYPE XY_y=0.0;
                PrecisionTYPE YZ_y=0.0;
                PrecisionTYPE XZ_y=0.0;
                PrecisionTYPE XX_z=0.0;
                PrecisionTYPE YY_z=0.0;
                PrecisionTYPE ZZ_z=0.0;
                PrecisionTYPE XY_z=0.0;
                PrecisionTYPE YZ_z=0.0;
                PrecisionTYPE XZ_z=0.0;
                
                for(int a=0; a<27; a++){
                    XX_x += basisXX[a]*xControlPointCoordinates[a];
                    YY_x += basisYY[a]*xControlPointCoordinates[a];
                    ZZ_x += basisZZ[a]*xControlPointCoordinates[a];
                    XY_x += basisXY[a]*xControlPointCoordinates[a];
                    YZ_x += basisYZ[a]*xControlPointCoordinates[a];
                    XZ_x += basisXZ[a]*xControlPointCoordinates[a];
                    
                    XX_y += basisXX[a]*yControlPointCoordinates[a];
                    YY_y += basisYY[a]*yControlPointCoordinates[a];
                    ZZ_y += basisZZ[a]*yControlPointCoordinates[a];
                    XY_y += basisXY[a]*yControlPointCoordinates[a];
                    YZ_y += basisYZ[a]*yControlPointCoordinates[a];
                    XZ_y += basisXZ[a]*yControlPointCoordinates[a];
                    
                    XX_z += basisXX[a]*zControlPointCoordinates[a];
                    YY_z += basisYY[a]*zControlPointCoordinates[a];
                    ZZ_z += basisZZ[a]*zControlPointCoordinates[a];
                    XY_z += basisXY[a]*zControlPointCoordinates[a];
                    YZ_z += basisYZ[a]*zControlPointCoordinates[a];
                    XZ_z += basisXZ[a]*zControlPointCoordinates[a];
                }

                constraintValue += (PrecisionTYPE)(XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x));
                constraintValue += (PrecisionTYPE)(XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y));
                constraintValue += (PrecisionTYPE)(XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z));
            }
        }
    }

    return (PrecisionTYPE)(constraintValue/(3.0*splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz));
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_bendingEnergy1(nifti_image *splineControlPoint,
					nifti_image *targetImage,
					int type)
{
	if(type==1){
        if(splineControlPoint->nz==1)
            return reg_bspline_bendingEnergyApproxValue2D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
        else return reg_bspline_bendingEnergyApproxValue3D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
	}
	else{
        if(splineControlPoint->nz==1)
            return reg_bspline_bendingEnergyValue2D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
		else return reg_bspline_bendingEnergyValue3D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
	}
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_bspline_bendingEnergy(	nifti_image *splineControlPoint,
					nifti_image *targetImage,
					int type)
{
	switch(splineControlPoint->datatype){
		case NIFTI_TYPE_FLOAT32:
			return reg_bspline_bendingEnergy1<PrecisionTYPE, float>(splineControlPoint, targetImage, type);
		case NIFTI_TYPE_FLOAT64:
			return reg_bspline_bendingEnergy1<PrecisionTYPE, double>(splineControlPoint, targetImage, type);
		default:
			printf("Only single or double precision is implemented for the bending energy\n");
			printf("The bending energy is not computed\n");
			return 0;
	}
}
/* *************************************************************** */
template float reg_bspline_bendingEnergy<float>(nifti_image *, nifti_image *, int);
template double reg_bspline_bendingEnergy<double>(nifti_image *, nifti_image *, int);
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobianValue2D(  nifti_image *splineControlPoint,
											nifti_image *targetImage)
{
	SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    PrecisionTYPE yBasis[4],yFirst[4],temp[4],first[4];
    PrecisionTYPE basisX[16], basisY[16];
    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

    PrecisionTYPE xControlPointCoordinates[16];
    PrecisionTYPE yControlPointCoordinates[16];

    PrecisionTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;

	mat33 reorient;
	reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
	reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
	reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
	mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
		spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
	}
    else{
		spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
	}
	reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
	mat33 jacobianMatrix;
	
//     unsigned int negativeDeterminant=0;
    PrecisionTYPE constraintValue=0;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
        basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(PrecisionTYPE)(1.0-basis);
        yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
        yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        yBasis[3] = (PrecisionTYPE)(FFF/6.0);
        yFirst[3]= (PrecisionTYPE)(FF / 2.0);
        yFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - yFirst[3]);
        yFirst[2]= (PrecisionTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
        yFirst[1]= - yFirst[0] - yFirst[2] - yFirst[3];

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
            basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (PrecisionTYPE)(FFF/6.0);
            first[3]= (PrecisionTYPE)(FF / 2.0);
            first[0]= (PrecisionTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (PrecisionTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= - first[0] - first[2] - first[3];

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*first[a];   // y * x'
                    basisY[coord]=yFirst[b]*temp[a];    // y'* x
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                coord=0;
                for(int Y=yPre; Y<yPre+4; Y++){
                    unsigned int index = Y*splineControlPoint->nx;
                    SplineTYPE *xxPtr = &controlPointPtrX[index];
                    SplineTYPE *yyPtr = &controlPointPtrY[index];
                    for(int X=xPre; X<xPre+4; X++){
                        xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                        yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                        coord++;
                    }
                }
            }
            oldBasis=basis;

            PrecisionTYPE Tx_x=0.0;
            PrecisionTYPE Ty_x=0.0;
            PrecisionTYPE Tx_y=0.0;
            PrecisionTYPE Ty_y=0.0;

            for(int a=0; a<16; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];

                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }

			memset(&jacobianMatrix, 0, sizeof(mat33));
			jacobianMatrix.m[2][2]=1.0f;
			jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
			jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
			jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
			jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
			
            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
			PrecisionTYPE detJac = nifti_mat33_determ(jacobianMatrix);
			if(detJac>0.0){
                PrecisionTYPE logJac = log(detJac);
                constraintValue += logJac*logJac;
            }
			else{
                return std::numeric_limits<PrecisionTYPE>::quiet_NaN();
            }
        }
    }
    return constraintValue/(PrecisionTYPE)targetImage->nvox;
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobianValue3D(  nifti_image *splineControlPoint,
											nifti_image *targetImage)
{
#if _USE_SSE
	if(sizeof(PrecisionTYPE)!=4){
		fprintf(stderr, "***ERROR***\treg_bspline_jacobianValue3D\n");
		fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
		exit(0);
	}
    union u{
		__m128 m;
		float f[4];
    } val;
#endif  
	
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    PrecisionTYPE zBasis[4],zFirst[4],temp[4],first[4];
    PrecisionTYPE tempX[16], tempY[16], tempZ[16];
    PrecisionTYPE basisX[64], basisY[64], basisZ[64];
    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

    PrecisionTYPE xControlPointCoordinates[64];
    PrecisionTYPE yControlPointCoordinates[64];
    PrecisionTYPE zControlPointCoordinates[64];

    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    unsigned int coord=0;

	mat33 reorient;
	reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
	reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
	reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
	mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
		spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
	}
    else{
		spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
	}
	reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
	mat33 jacobianMatrix;
	
    PrecisionTYPE constraintValue=0;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((PrecisionTYPE)z/gridVoxelSpacing[2]);
        basis=(PrecisionTYPE)z/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(PrecisionTYPE)(1.0-basis);
        zBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
        zBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        zBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        zBasis[3] = (PrecisionTYPE)(FFF/6.0);
        zFirst[3]= (PrecisionTYPE)(FF / 2.0);
        zFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - zFirst[3]);
        zFirst[2]= (PrecisionTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
        zFirst[1]= - zFirst[0] - zFirst[2] - zFirst[3];

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
            basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (PrecisionTYPE)(FFF/6.0);
            first[3]= (PrecisionTYPE)(FF / 2.0);
            first[0]= (PrecisionTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (PrecisionTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= - first[0] - first[2] - first[3];
			
#if _USE_SSE
            val.f[0]=temp[0];
            val.f[1]=temp[1];
            val.f[2]=temp[2];
            val.f[3]=temp[3];
            __m128 _yBasis=val.m; 
            val.f[0]=first[0];
            val.f[1]=first[1];
            val.f[2]=first[2];
            val.f[3]=first[3];
            __m128 _yFirst=val.m;          
            __m128 *ptrBasisX = (__m128 *) &tempX[0];
            __m128 *ptrBasisY = (__m128 *) &tempY[0];
            __m128 *ptrBasisZ = (__m128 *) &tempZ[0];
            for(int a=0;a<4;++a){
	            val.m=_mm_set_ps1(zBasis[a]);
	            *ptrBasisX=_mm_mul_ps(_yBasis,val.m);
	            *ptrBasisY=_mm_mul_ps(_yFirst,val.m);
	            val.m=_mm_set_ps1(zFirst[a]);
	            *ptrBasisZ=_mm_mul_ps(_yBasis,val.m);
	            ptrBasisX++;
	            ptrBasisY++;
	            ptrBasisZ++;
            }
#else
            coord=0;
            for(int c=0; c<4; c++){
                for(int b=0; b<4; b++){
                    tempX[coord]=zBasis[c]*temp[b]; // z * y
                    tempY[coord]=zBasis[c]*first[b];// z * y'
                    tempZ[coord]=zFirst[c]*temp[b]; // z'* y
                    coord++;
                }
            }
#endif
            for(int x=0; x<targetImage->nx; x++){

                int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
                basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                temp[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
                temp[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                temp[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                temp[3] = (PrecisionTYPE)(FFF/6.0);
                first[3]= (PrecisionTYPE)(FF / 2.0);
                first[0]= (PrecisionTYPE)(basis - 1.0/2.0 - first[3]);
                first[2]= (PrecisionTYPE)(1.0 + first[0] - 2.0*first[3]);
                first[1]= - first[0] - first[2] - first[3];

#if _USE_SSE
				val.f[0]=temp[0];
				val.f[1]=temp[1];
				val.f[2]=temp[2];
				val.f[3]=temp[3];
				__m128 _xBasis=val.m; 
				val.f[0]=first[0];
				val.f[1]=first[1];
				val.f[2]=first[2];
				val.f[3]=first[3];
				__m128 _xFirst=val.m;
				ptrBasisX = (__m128 *) &basisX[0];
				ptrBasisY = (__m128 *) &basisY[0];
				ptrBasisZ = (__m128 *) &basisZ[0];
				for(int a=0;a<16;++a){
					val.m=_mm_set_ps1(tempX[a]);
					*ptrBasisX=_mm_mul_ps(_xFirst,val.m);
					val.m=_mm_set_ps1(tempY[a]);
					*ptrBasisY=_mm_mul_ps(_xBasis,val.m);
					val.m=_mm_set_ps1(tempZ[a]);
					*ptrBasisZ=_mm_mul_ps(_xBasis,val.m);
					ptrBasisX++;
					ptrBasisY++;
					ptrBasisZ++;
				}
#else
                coord=0;
                for(int bc=0; bc<16; bc++){
                    for(int a=0; a<4; a++){
                        basisX[coord]=tempX[bc]*first[a];   // z * y * x'
                        basisY[coord]=tempY[bc]*temp[a];    // z * y'* x
                        basisZ[coord]=tempZ[bc]*temp[a];    // z'* y * x
                        coord++;
                    }
                }
#endif

                if(basis<=oldBasis || x==0){
                    coord=0;
                    for(int Z=zPre; Z<zPre+4; Z++){
                        unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                        SplineTYPE *xPtr = &controlPointPtrX[index];
                        SplineTYPE *yPtr = &controlPointPtrY[index];
                        SplineTYPE *zPtr = &controlPointPtrZ[index];
                        for(int Y=yPre; Y<yPre+4; Y++){
                            index = Y*splineControlPoint->nx;
                            SplineTYPE *xxPtr = &xPtr[index];
                            SplineTYPE *yyPtr = &yPtr[index];
                            SplineTYPE *zzPtr = &zPtr[index];
                            for(int X=xPre; X<xPre+4; X++){
                                xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                                yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                                zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;

                PrecisionTYPE Tx_x=0.0;
                PrecisionTYPE Ty_x=0.0;
                PrecisionTYPE Tz_x=0.0;
                PrecisionTYPE Tx_y=0.0;
                PrecisionTYPE Ty_y=0.0;
                PrecisionTYPE Tz_y=0.0;
                PrecisionTYPE Tx_z=0.0;
                PrecisionTYPE Ty_z=0.0;
                PrecisionTYPE Tz_z=0.0;

#if _USE_SSE	
				__m128 tempX_x =  _mm_set_ps1(0.0);
				__m128 tempX_y =  _mm_set_ps1(0.0);
				__m128 tempX_z =  _mm_set_ps1(0.0);
				__m128 tempY_x =  _mm_set_ps1(0.0);
				__m128 tempY_y =  _mm_set_ps1(0.0);
				__m128 tempY_z =  _mm_set_ps1(0.0);
				__m128 tempZ_x =  _mm_set_ps1(0.0);
				__m128 tempZ_y =  _mm_set_ps1(0.0);
				__m128 tempZ_z =  _mm_set_ps1(0.0);
				__m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
				__m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
				__m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];
				ptrBasisX   = (__m128 *) &basisX[0];
				ptrBasisY   = (__m128 *) &basisY[0];
				ptrBasisZ   = (__m128 *) &basisZ[0];
				//addition and multiplication of the 16 basis value and CP position for each axis
				for(unsigned int a=0; a<16; a++){
					tempX_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrX), tempX_x );
					tempX_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrX), tempX_y );
					tempX_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrX), tempX_z );
					
					tempY_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrY), tempY_x );
					tempY_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrY), tempY_y );
					tempY_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrY), tempY_z );
					
					tempZ_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrZ), tempZ_x );
					tempZ_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrZ), tempZ_y );
					tempZ_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrZ), tempZ_z );
					
					ptrBasisX++;
					ptrBasisY++;
					ptrBasisZ++;
					ptrX++;
					ptrY++;
					ptrZ++;
				}

				//the values stored in SSE variables are transfered to normal float
				val.m = tempX_x;
				Tx_x = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				val.m = tempX_y;
				Tx_y = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				val.m = tempX_z;
				Tx_z = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				
				val.m = tempY_x;
				Ty_x = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				val.m = tempY_y;
				Ty_y = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				val.m = tempY_z;
				Ty_z = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				
				val.m = tempZ_x;
				Tz_x = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				val.m = tempZ_y;
				Tz_y = val.f[0]+val.f[1]+val.f[2]+val.f[3];
				val.m = tempZ_z;
				Tz_z = val.f[0]+val.f[1]+val.f[2]+val.f[3];	
#else
                for(int a=0; a<64; a++){
                    Tx_x += basisX[a]*xControlPointCoordinates[a];
                    Tx_y += basisY[a]*xControlPointCoordinates[a];
                    Tx_z += basisZ[a]*xControlPointCoordinates[a];
					
                    Ty_x += basisX[a]*yControlPointCoordinates[a];
                    Ty_y += basisY[a]*yControlPointCoordinates[a];
                    Ty_z += basisZ[a]*yControlPointCoordinates[a];
					
                    Tz_x += basisX[a]*zControlPointCoordinates[a];
                    Tz_y += basisY[a]*zControlPointCoordinates[a];
                    Tz_z += basisZ[a]*zControlPointCoordinates[a];
                }
#endif				

				jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
				jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
				jacobianMatrix.m[0][2]= (float)(Tx_z / splineControlPoint->dz);
				jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
				jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
				jacobianMatrix.m[1][2]= (float)(Ty_z / splineControlPoint->dz);
				jacobianMatrix.m[2][0]= (float)(Tz_x / splineControlPoint->dx);
				jacobianMatrix.m[2][1]= (float)(Tz_y / splineControlPoint->dy);
				jacobianMatrix.m[2][2]= (float)(Tz_z / splineControlPoint->dz);
				
            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
				PrecisionTYPE detJac = nifti_mat33_determ(jacobianMatrix);

				if(detJac>0.0){
                    PrecisionTYPE logJac = log(detJac);
                    constraintValue += logJac*logJac;
//                     constraintValue += logValue;
				}
//                 else constraintValue += 1000.0;
				//else return NAN;
                else return std::numeric_limits<PrecisionTYPE>::quiet_NaN();
            }
        }
    }

    return constraintValue/(PrecisionTYPE)targetImage->nvox;
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobianApproxValue2D(  nifti_image *splineControlPoint,
                            nifti_image *targetImage
                            )
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same 
    PrecisionTYPE normal[3];
    PrecisionTYPE first[3];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);

    PrecisionTYPE constraintValue= (PrecisionTYPE)(0.0);

    PrecisionTYPE basisX[9], basisY[9];

    int coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisX[coord]=normal[b]*first[a];   // y * x'
            basisY[coord]=first[b]*normal[a];   // y'* x
            coord++;
        }
    }

    PrecisionTYPE xControlPointCoordinates[9];
    PrecisionTYPE yControlPointCoordinates[9];

	mat33 reorient;
	reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
	reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
	reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
	mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
		spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
	}
    else{
		spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
	}
	reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
	mat33 jacobianMatrix;
	
    for(int y=1;y<splineControlPoint->ny-2;y++){
        for(int x=1;x<splineControlPoint->nx-2;x++){

            coord=0;
            for(int Y=y-1; Y<y+2; Y++){
                unsigned int index = Y*splineControlPoint->nx;
                SplineTYPE *xxPtr = &controlPointPtrX[index];
                SplineTYPE *yyPtr = &controlPointPtrY[index];
                for(int X=x-1; X<x+2; X++){
                    xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                    yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                    coord++;
                }
            }

            PrecisionTYPE Tx_x=0.0;
            PrecisionTYPE Ty_x=0.0;
            PrecisionTYPE Tx_y=0.0;
            PrecisionTYPE Ty_y=0.0;

            for(int a=0; a<9; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];

                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }
			
			memset(&jacobianMatrix, 0, sizeof(mat33));
			jacobianMatrix.m[2][2]=1.0f;
			jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
			jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
			jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
			jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
			
            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
			PrecisionTYPE detJac = nifti_mat33_determ(jacobianMatrix);

			if(detJac>0.0){
                PrecisionTYPE logJac = log(detJac);
                constraintValue += logJac*logJac;
			}
            else return std::numeric_limits<PrecisionTYPE>::quiet_NaN();
        }
    }

    return constraintValue/(float)((splineControlPoint->nx-2)*(splineControlPoint->ny-2));
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobianApproxValue3D(  nifti_image *splineControlPoint,
                            nifti_image *targetImage
                            )
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same 
	float basisX[27] = {-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f, 
		-0.0555556f, 0.0000000f, 0.0555556f, -0.2222222f, 0.0000000f, 0.2222222f, -0.0555556f, 0.0000000f, 0.0555556f, 
		-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f};
	float basisY[27] = {-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f, 
		-0.0555556f, -0.2222222f, -0.0555556f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0555556f, 0.2222222f, 0.0555556f, 
		-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f};
	float basisZ[27] = {-0.0138889f, -0.0555556f, -0.0138889f, -0.0555556f, -0.2222222f, -0.0555556f, -0.0138889f, -0.0555556f, -0.0138889f, 
		0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 
		0.0138889f, 0.0555556f, 0.0138889f, 0.0555556f, 0.2222222f, 0.0555556f, 0.0138889f, 0.0555556f, 0.0138889f};
    PrecisionTYPE xControlPointCoordinates[27];
    PrecisionTYPE yControlPointCoordinates[27];
    PrecisionTYPE zControlPointCoordinates[27];

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
	spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
	spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
	spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
	spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
	spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
	spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
	spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
	spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
	spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
	spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
	spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
	spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
	spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
	spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
	spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
	spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
	spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
	spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    mat33 jacobianMatrix;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
	(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>
	(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    double constraintValue=0.0;
    int coord;
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            for(int x=1;x<splineControlPoint->nx-1;x++){

                coord=0;
                for(int Z=z-1; Z<z+2; Z++){
                    unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                    SplineTYPE *xPtr = &controlPointPtrX[index];
                    SplineTYPE *yPtr = &controlPointPtrY[index];
                    SplineTYPE *zPtr = &controlPointPtrZ[index];
                    for(int Y=y-1; Y<y+2; Y++){
                        index = Y*splineControlPoint->nx;
                        SplineTYPE *xxPtr = &xPtr[index];
                        SplineTYPE *yyPtr = &yPtr[index];
                        SplineTYPE *zzPtr = &zPtr[index];
                        for(int X=x-1; X<x+2; X++){
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                            zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                            coord++;
                        }
                    }
                }

                PrecisionTYPE Tx_x=0.0;
                PrecisionTYPE Ty_x=0.0;
                PrecisionTYPE Tz_x=0.0;
                PrecisionTYPE Tx_y=0.0;
                PrecisionTYPE Ty_y=0.0;
                PrecisionTYPE Tz_y=0.0;
                PrecisionTYPE Tx_z=0.0;
                PrecisionTYPE Ty_z=0.0;
                PrecisionTYPE Tz_z=0.0;

                for(int a=0; a<27; a++){
                    Tx_x += basisX[a]*xControlPointCoordinates[a];
                    Tx_y += basisY[a]*xControlPointCoordinates[a];
                    Tx_z += basisZ[a]*xControlPointCoordinates[a];

                    Ty_x += basisX[a]*yControlPointCoordinates[a];
                    Ty_y += basisY[a]*yControlPointCoordinates[a];
                    Ty_z += basisZ[a]*yControlPointCoordinates[a];

                    Tz_x += basisX[a]*zControlPointCoordinates[a];
                    Tz_y += basisY[a]*zControlPointCoordinates[a];
                    Tz_z += basisZ[a]*zControlPointCoordinates[a];
                }
				
				jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
				jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
				jacobianMatrix.m[0][2]= (float)(Tx_z / splineControlPoint->dz);
				jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
				jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
				jacobianMatrix.m[1][2]= (float)(Ty_z / splineControlPoint->dz);
				jacobianMatrix.m[2][0]= (float)(Tz_x / splineControlPoint->dx);
				jacobianMatrix.m[2][1]= (float)(Tz_y / splineControlPoint->dy);
				jacobianMatrix.m[2][2]= (float)(Tz_z / splineControlPoint->dz);
				
                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
				PrecisionTYPE detJac = nifti_mat33_determ(jacobianMatrix);

				if(detJac>0.0){
                    PrecisionTYPE logJac = log(detJac);
                    constraintValue += logJac*logJac;
				}
                else return std::numeric_limits<PrecisionTYPE>::quiet_NaN();
            }
        }
    }

    return constraintValue/(float)((splineControlPoint->nx-2)*(splineControlPoint->ny-2)*(splineControlPoint->nz-2));
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobian1(nifti_image *splineControlPoint,
					nifti_image *targetImage,
					int type
					)
{
    if(splineControlPoint->nz==1){
        if(type==1){
            return reg_bspline_jacobianApproxValue2D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
        }
        else{
            return reg_bspline_jacobianValue2D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
        }
    }
    else{
        if(type==1){
            return reg_bspline_jacobianApproxValue3D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
        }
        else{
            return reg_bspline_jacobianValue3D<PrecisionTYPE,SplineTYPE>(splineControlPoint, targetImage);
        }
    }
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_bspline_jacobian(	nifti_image *splineControlPoint,
					nifti_image *targetImage,
					int type
					)
{
	switch(splineControlPoint->datatype){
		case NIFTI_TYPE_FLOAT32:
			return reg_bspline_jacobian1<PrecisionTYPE, float>(splineControlPoint, targetImage, type);
		case NIFTI_TYPE_FLOAT64:
			return reg_bspline_jacobian1<PrecisionTYPE, double>(splineControlPoint, targetImage, type);
		default:
			printf("Only single or double precision is implemented for the jacobian value\n");
			printf("The jacobian value is not computed\n");
			return 0;
	}
}
/* *************************************************************** */
template float reg_bspline_jacobian<float>(nifti_image *, nifti_image *, int);
template double reg_bspline_jacobian<double>(nifti_image *, nifti_image *, int);
/* *************************************************************** */
/* *************************************************************** */
template<class NodeTYPE, class VoxelTYPE>
void reg_voxelCentric2NodeCentric2D(nifti_image *nodeImage,
									nifti_image *voxelImage,
									float weight
									)
{
	NodeTYPE *nodePtrX = static_cast<NodeTYPE *>(nodeImage->data);
	NodeTYPE *nodePtrY = &nodePtrX[nodeImage->nx*nodeImage->ny];

	VoxelTYPE *voxelPtrX = static_cast<VoxelTYPE *>(voxelImage->data);
	VoxelTYPE *voxelPtrY = &voxelPtrX[voxelImage->nx*voxelImage->ny];

	float ratio[2];
	ratio[0] = nodeImage->dx / voxelImage->dx;
	ratio[1] = nodeImage->dy / voxelImage->dy;

	for(int y=0;y<nodeImage->ny; y++){
		int Y = (int)round((float)(y-1) * ratio[1]);
		VoxelTYPE *yVoxelPtrX=&voxelPtrX[Y*voxelImage->nx];
		VoxelTYPE *yVoxelPtrY=&voxelPtrY[Y*voxelImage->nx];
		for(int x=0;x<nodeImage->nx; x++){
			int X = (int)round((float)(x-1) * ratio[0]);
			if( -1<Y && Y<voxelImage->ny && -1<X && X<voxelImage->nx){
				*nodePtrX++ = (NodeTYPE)(yVoxelPtrX[X] * weight);
				*nodePtrY++ = (NodeTYPE)(yVoxelPtrY[X] * weight);
			}
			else{
				*nodePtrX++ = 0.0;
				*nodePtrY++ = 0.0;
			}
		}
	}
}
/* *************************************************************** */
template<class NodeTYPE, class VoxelTYPE>
void reg_voxelCentric2NodeCentric3D(nifti_image *nodeImage,
									nifti_image *voxelImage,
									float weight
									)
{
	NodeTYPE *nodePtrX = static_cast<NodeTYPE *>(nodeImage->data);
	NodeTYPE *nodePtrY = &nodePtrX[nodeImage->nx*nodeImage->ny*nodeImage->nz];
	NodeTYPE *nodePtrZ = &nodePtrY[nodeImage->nx*nodeImage->ny*nodeImage->nz];

	VoxelTYPE *voxelPtrX = static_cast<VoxelTYPE *>(voxelImage->data);
	VoxelTYPE *voxelPtrY = &voxelPtrX[voxelImage->nx*voxelImage->ny*voxelImage->nz];
	VoxelTYPE *voxelPtrZ = &voxelPtrY[voxelImage->nx*voxelImage->ny*voxelImage->nz];

	float ratio[3];
	ratio[0] = nodeImage->dx / voxelImage->dx;
	ratio[1] = nodeImage->dy / voxelImage->dy;
	ratio[2] = nodeImage->dz / voxelImage->dz;

	for(int z=0;z<nodeImage->nz; z++){
		int Z = (int)round((float)(z-1) * ratio[2]);
		VoxelTYPE *zvoxelPtrX=&voxelPtrX[Z*voxelImage->nx*voxelImage->ny];
		VoxelTYPE *zvoxelPtrY=&voxelPtrY[Z*voxelImage->nx*voxelImage->ny];
		VoxelTYPE *zvoxelPtrZ=&voxelPtrZ[Z*voxelImage->nx*voxelImage->ny];
		for(int y=0;y<nodeImage->ny; y++){
			int Y = (int)round((float)(y-1) * ratio[1]);
			VoxelTYPE *yzvoxelPtrX=&zvoxelPtrX[Y*voxelImage->nx];
			VoxelTYPE *yzvoxelPtrY=&zvoxelPtrY[Y*voxelImage->nx];
			VoxelTYPE *yzvoxelPtrZ=&zvoxelPtrZ[Y*voxelImage->nx];
			for(int x=0;x<nodeImage->nx; x++){
				int X = (int)round((float)(x-1) * ratio[0]);
				if(-1<Z && Z<voxelImage->nz && -1<Y && Y<voxelImage->ny && -1<X && X<voxelImage->nx){
					*nodePtrX++ = (NodeTYPE)(yzvoxelPtrX[X]*weight);
					*nodePtrY++ = (NodeTYPE)(yzvoxelPtrY[X]*weight);
					*nodePtrZ++ = (NodeTYPE)(yzvoxelPtrZ[X]*weight);
				}
				else{
					*nodePtrX++ = 0.0;
					*nodePtrY++ = 0.0;
					*nodePtrZ++ = 0.0;
				}
			}
		}
	}
}
/* *************************************************************** */
extern "C++"
void reg_voxelCentric2NodeCentric(	nifti_image *nodeImage,
									nifti_image *voxelImage,
									float weight
									)
{
	// it is assumed than node[000] and voxel[000] are aligned.
	if(nodeImage->nz==1){	
		switch(nodeImage->datatype){
			case NIFTI_TYPE_FLOAT32:
				switch(voxelImage->datatype){
					case NIFTI_TYPE_FLOAT32:
						reg_voxelCentric2NodeCentric2D<float, float>(nodeImage, voxelImage, weight);
						break;
					case NIFTI_TYPE_FLOAT64:
						reg_voxelCentric2NodeCentric2D<float, double>(nodeImage, voxelImage, weight);
						break;
					default:
						printf("err\treg_voxelCentric2NodeCentric:v1\tdata type not supported\n");
						break;
				}
				break;
			case NIFTI_TYPE_FLOAT64:
				switch(voxelImage->datatype){
					case NIFTI_TYPE_FLOAT32:
						reg_voxelCentric2NodeCentric2D<double, float>(nodeImage, voxelImage, weight);
						break;
					case NIFTI_TYPE_FLOAT64:
						reg_voxelCentric2NodeCentric2D<double, double>(nodeImage, voxelImage, weight);
						break;
					default:
						printf("err\treg_voxelCentric2NodeCentric:v2\tdata type not supported\n");
						break;
				}
				break;
			default:
				printf("err\treg_voxelCentric2NodeCentric:n\tdata type not supported\n");
				break;
		}
	}
	else{
		switch(nodeImage->datatype){
			case NIFTI_TYPE_FLOAT32:
				switch(voxelImage->datatype){
					case NIFTI_TYPE_FLOAT32:
						reg_voxelCentric2NodeCentric3D<float, float>(nodeImage, voxelImage, weight);
						break;
					case NIFTI_TYPE_FLOAT64:
						reg_voxelCentric2NodeCentric3D<float, double>(nodeImage, voxelImage, weight);
						break;
					default:
						printf("err\treg_voxelCentric2NodeCentric:v1\tdata type not supported\n");
						break;
				}
				break;
			case NIFTI_TYPE_FLOAT64:
				switch(voxelImage->datatype){
					case NIFTI_TYPE_FLOAT32:
						reg_voxelCentric2NodeCentric3D<double, float>(nodeImage, voxelImage, weight);
						break;
					case NIFTI_TYPE_FLOAT64:
						reg_voxelCentric2NodeCentric3D<double, double>(nodeImage, voxelImage, weight);
						break;
					default:
						printf("err\treg_voxelCentric2NodeCentric:v2\tdata type not supported\n");
						break;
				}
				break;
			default:
				printf("err\treg_voxelCentric2NodeCentric:n\tdata type not supported\n");
				break;
		}
	}
}
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_approxBendingEnergyGradient3D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same 
    PrecisionTYPE normal[3];
    PrecisionTYPE first[3];
    PrecisionTYPE second[3];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);
    second[0] = (PrecisionTYPE)(1.0);
    second[1] = (PrecisionTYPE)(-2.0);
    second[2] = (PrecisionTYPE)(1.0);


    int coord;
    // There are six different values taken into account
    PrecisionTYPE tempXX[9], tempYY[9], tempZZ[9], tempXY[9], tempYZ[9], tempXZ[9];

    coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempXX[coord]=normal[c]*normal[b];  // z * y
            tempYY[coord]=normal[c]*second[b];  // z * y"
            tempZZ[coord]=second[c]*normal[b];  // z"* y
            tempXY[coord]=normal[c]*first[b];   // z * y'
            tempYZ[coord]=first[c]*first[b];    // z'* y'
            tempXZ[coord]=first[c]*normal[b];   // z'* y
            coord++;
        }
    }

    PrecisionTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];

    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisXX[coord]=tempXX[bc]*second[a];    // z * y * x"
            basisYY[coord]=tempYY[bc]*normal[a];    // z * y"* x
            basisZZ[coord]=tempZZ[bc]*normal[a];    // z"* y * x
            basisXY[coord]=tempXY[bc]*first[a];     // z * y'* x'
            basisYZ[coord]=tempYZ[bc]*normal[a];    // z'* y'* x
            basisXZ[coord]=tempXZ[bc]*first[a];     // z'* y * x'
            coord++;
        }
    }

    PrecisionTYPE nodeNumber = (PrecisionTYPE)(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
    PrecisionTYPE *derivativeValues = (PrecisionTYPE *)calloc(18*(int)nodeNumber, sizeof(PrecisionTYPE));
    PrecisionTYPE *derivativeValuesPtr;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[(unsigned int)nodeNumber]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[(unsigned int)nodeNumber]);

    PrecisionTYPE xControlPointCoordinates[27];
    PrecisionTYPE yControlPointCoordinates[27];
    PrecisionTYPE zControlPointCoordinates[27];

    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            derivativeValuesPtr = &derivativeValues[18*((z*splineControlPoint->ny+y)*splineControlPoint->nx+1)];
            for(int x=1;x<splineControlPoint->nx-1;x++){

                coord=0;
                for(int Z=z-1; Z<z+2; Z++){
                    unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                    SplineTYPE *xPtr = &controlPointPtrX[index];
                    SplineTYPE *yPtr = &controlPointPtrY[index];
                    SplineTYPE *zPtr = &controlPointPtrZ[index];
                    for(int Y=y-1; Y<y+2; Y++){
                        index = Y*splineControlPoint->nx;
                        SplineTYPE *xxPtr = &xPtr[index];
                        SplineTYPE *yyPtr = &yPtr[index];
                        SplineTYPE *zzPtr = &zPtr[index];
                        for(int X=x-1; X<x+2; X++){
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                            zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                            coord++;
                        }
                    }
                }

                PrecisionTYPE XX_x=0.0;
                PrecisionTYPE YY_x=0.0;
                PrecisionTYPE ZZ_x=0.0;
                PrecisionTYPE XY_x=0.0;
                PrecisionTYPE YZ_x=0.0;
                PrecisionTYPE XZ_x=0.0;
                PrecisionTYPE XX_y=0.0;
                PrecisionTYPE YY_y=0.0;
                PrecisionTYPE ZZ_y=0.0;
                PrecisionTYPE XY_y=0.0;
                PrecisionTYPE YZ_y=0.0;
                PrecisionTYPE XZ_y=0.0;
                PrecisionTYPE XX_z=0.0;
                PrecisionTYPE YY_z=0.0;
                PrecisionTYPE ZZ_z=0.0;
                PrecisionTYPE XY_z=0.0;
                PrecisionTYPE YZ_z=0.0;
                PrecisionTYPE XZ_z=0.0;

                for(int a=0; a<27; a++){
                    XX_x += basisXX[a]*xControlPointCoordinates[a];
                    YY_x += basisYY[a]*xControlPointCoordinates[a];
                    ZZ_x += basisZZ[a]*xControlPointCoordinates[a];
                    XY_x += basisXY[a]*xControlPointCoordinates[a];
                    YZ_x += basisYZ[a]*xControlPointCoordinates[a];
                    XZ_x += basisXZ[a]*xControlPointCoordinates[a];

                    XX_y += basisXX[a]*yControlPointCoordinates[a];
                    YY_y += basisYY[a]*yControlPointCoordinates[a];
                    ZZ_y += basisZZ[a]*yControlPointCoordinates[a];
                    XY_y += basisXY[a]*yControlPointCoordinates[a];
                    YZ_y += basisYZ[a]*yControlPointCoordinates[a];
                    XZ_y += basisXZ[a]*yControlPointCoordinates[a];

                    XX_z += basisXX[a]*zControlPointCoordinates[a];
                    YY_z += basisYY[a]*zControlPointCoordinates[a];
                    ZZ_z += basisZZ[a]*zControlPointCoordinates[a];
                    XY_z += basisXY[a]*zControlPointCoordinates[a];
                    YZ_z += basisYZ[a]*zControlPointCoordinates[a];
                    XZ_z += basisXZ[a]*zControlPointCoordinates[a];
                }
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*XX_x);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*XX_y);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*XX_z);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*YY_x);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*YY_y);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*YY_z);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*ZZ_x);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*ZZ_y);
                *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*ZZ_z);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XY_x);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XY_y);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XY_z);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*YZ_x);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*YZ_y);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*YZ_z);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XZ_x);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XZ_y);
                *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XZ_z);
            }
        }
    }

    SplineTYPE *gradientX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientY = &gradientX[(int)nodeNumber];
    SplineTYPE *gradientZ = &gradientY[(int)nodeNumber];
    SplineTYPE *gradientXPtr = &gradientX[0];
    SplineTYPE *gradientYPtr = &gradientY[0];
    SplineTYPE *gradientZPtr = &gradientZ[0];

    SplineTYPE approxRatio= weight * targetImage->nx*targetImage->ny*targetImage->nz
    / ( splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz );

    PrecisionTYPE gradientValue[3];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                gradientValue[0]=gradientValue[1]=gradientValue[2]=0.0;

                coord=0;
                for(int Z=z-1; Z<z+2; Z++){
                    for(int Y=y-1; Y<y+2; Y++){
                        for(int X=x-1; X<x+2; X++){
                            if(-1<X && -1<Y && -1<Z && X<splineControlPoint->nx && Y<splineControlPoint->ny && Z<splineControlPoint->nz){
                                derivativeValuesPtr = &derivativeValues[18 * ((Z*splineControlPoint->ny + Y)*splineControlPoint->nx + X)];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisXX[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXX[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXX[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisYY[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisYY[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisYY[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisZZ[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisZZ[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisZZ[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisXY[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXY[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXY[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisYZ[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisYZ[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisYZ[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisXZ[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXZ[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXZ[coord];
                            }
                            coord++;
                        }
                    }
                }
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                *gradientXPtr++ += (SplineTYPE)(approxRatio*gradientValue[0]);
                *gradientYPtr++ += (SplineTYPE)(approxRatio*gradientValue[1]);
                *gradientZPtr++ += (SplineTYPE)(approxRatio*gradientValue[2]);
            }
        }
    }

    free(derivativeValues);
}
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_approxBendingEnergyGradient2D(   nifti_image *splineControlPoint,
                                            nifti_image *targetImage,
                                            nifti_image *gradientImage,
                                            float weight)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same 
    PrecisionTYPE normal[3];
    PrecisionTYPE first[3];
    PrecisionTYPE second[3];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);
    second[0] = (PrecisionTYPE)(1.0);
    second[1] = (PrecisionTYPE)(-2.0);
    second[2] = (PrecisionTYPE)(1.0);


    int coord;
    // There are six different values taken into account
    PrecisionTYPE basisXX[9], basisYY[9], basisXY[9];

    coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisXX[coord]=normal[b]*second[a];    // y * x"
            basisYY[coord]=second[b]*normal[a];    // y"* x
            basisXY[coord]=first[b]*first[a];      // y'* x'
            coord++;
        }
    }

    PrecisionTYPE nodeNumber = (PrecisionTYPE)(splineControlPoint->nx*splineControlPoint->ny);
    PrecisionTYPE *derivativeValues = (PrecisionTYPE *)calloc(6*(int)nodeNumber, sizeof(PrecisionTYPE));

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[(unsigned int)nodeNumber]);

    PrecisionTYPE xControlPointCoordinates[9];
    PrecisionTYPE yControlPointCoordinates[9];
	
    PrecisionTYPE *derivativeValuesPtr = &derivativeValues[0];

    for(int y=1;y<splineControlPoint->ny-1;y++){
		derivativeValuesPtr = &derivativeValues[6*(y*splineControlPoint->nx+1)];
        for(int x=1;x<splineControlPoint->nx-1;x++){

			coord=0;
			for(int Y=y-1; Y<y+2; Y++){
				unsigned int index = Y*splineControlPoint->nx;
				SplineTYPE *xPtr = &controlPointPtrX[index];
				SplineTYPE *yPtr = &controlPointPtrY[index];
				
				for(int X=x-1; X<x+2; X++){
					xControlPointCoordinates[coord] = (PrecisionTYPE)xPtr[X];
					yControlPointCoordinates[coord] = (PrecisionTYPE)yPtr[X];
					
					coord++;
				}
			}

            PrecisionTYPE XX_x=0.0;
            PrecisionTYPE YY_x=0.0;
            PrecisionTYPE XY_x=0.0;
            PrecisionTYPE XX_y=0.0;
            PrecisionTYPE YY_y=0.0;
            PrecisionTYPE XY_y=0.0;

            for(int a=0; a<9; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }
            *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*XX_x);
            *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*XX_y);
            *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*YY_x);
            *derivativeValuesPtr++ = (PrecisionTYPE)(2.0*YY_y);
            *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XY_x);
            *derivativeValuesPtr++ = (PrecisionTYPE)(4.0*XY_y);
        }
    }

    SplineTYPE *gradientXPtr = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientYPtr = static_cast<SplineTYPE *>(&gradientXPtr[(int)nodeNumber]);

	SplineTYPE approxRatio= weight * targetImage->nx*targetImage->ny
	/ ( splineControlPoint->nx*splineControlPoint->ny );

    PrecisionTYPE gradientValue[2];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            gradientValue[0]=gradientValue[1]=0.0;

            coord=0;
            for(int Y=y-1; Y<y+2; Y++){
                for(int X=x-1; X<x+2; X++){
                    if(-1<X && -1<Y && X<splineControlPoint->nx && Y<splineControlPoint->ny){
                        derivativeValuesPtr = &derivativeValues[6 * (Y*splineControlPoint->nx + X)];
                        gradientValue[0] += (*derivativeValuesPtr++) * basisXX[coord];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXX[coord];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisYY[coord];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisYY[coord];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisXY[coord];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXY[coord];
                    }
                    coord++;
                }
            }
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            *gradientXPtr++ += (SplineTYPE)(approxRatio*gradientValue[0]);
            *gradientYPtr++ += (SplineTYPE)(approxRatio*gradientValue[1]);
        }
    }

    free(derivativeValues);
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE>
void reg_bspline_bendingEnergyGradient( nifti_image *splineControlPoint,
                                        nifti_image *targetImage,
                                        nifti_image *gradientImage,
                                        float weight)
{
	if(splineControlPoint->datatype != gradientImage->datatype){
		fprintf(stderr,"The spline control point image and the gradient image were expected to have the same datatype\n");
		fprintf(stderr,"The bending energy gradient has not computed\n");
	}
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_approxBendingEnergyGradient2D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                break;
                reg_bspline_approxBendingEnergyGradient2D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"The bending energy gradient has not been computed\n");
                break;
        }
        }else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_approxBendingEnergyGradient3D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                break;
                reg_bspline_approxBendingEnergyGradient3D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"The bending energy gradient has not been computed\n");
                break;
        }
    }
}
template void reg_bspline_bendingEnergyGradient<float>(nifti_image *, nifti_image *, nifti_image *, float);
template void reg_bspline_bendingEnergyGradient<double>(nifti_image *, nifti_image *, nifti_image *, float);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class SplineTYPE>
void computeJacobianMatrices_2D(nifti_image *targetImage,
                                nifti_image *splineControlPoint,
                                mat33 *invertedJacobianMatrices,
                                PrecisionTYPE *jacobianDeterminant)
{

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    PrecisionTYPE yBasis[4],yFirst[4],xBasis[4],xFirst[4];
    PrecisionTYPE basisX[16], basisY[16];
    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

    PrecisionTYPE xControlPointCoordinates[16];
    PrecisionTYPE yControlPointCoordinates[16];

    PrecisionTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    mat33 jacobianMatrix;

    // The inverted Jacobian matrices are first computed for every voxel
    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    PrecisionTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
        basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(PrecisionTYPE)(1.0-basis);
        yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
        yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        yBasis[3] = (PrecisionTYPE)(FFF/6.0);
        yFirst[3]= (PrecisionTYPE)(FF / 2.0);
        yFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - yFirst[3]);
        yFirst[2]= (PrecisionTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
        yFirst[1]= - yFirst[0] - yFirst[2] - yFirst[3];

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
            basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
            xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            xBasis[3] = (PrecisionTYPE)(FFF/6.0);
            xFirst[3]= (PrecisionTYPE)(FF / 2.0);
            xFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - xFirst[3]);
            xFirst[2]= (PrecisionTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
            xFirst[1]= - xFirst[0] - xFirst[2] - xFirst[3];

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*xFirst[a];   // y * x'
                    basisY[coord]=yFirst[b]*xBasis[a];    // y'* x
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                coord=0;
                for(int Y=yPre; Y<yPre+4; Y++){
                    unsigned int index = Y*splineControlPoint->nx;
                    SplineTYPE *xxPtr = &controlPointPtrX[index];
                    SplineTYPE *yyPtr = &controlPointPtrY[index];
                    for(int X=xPre; X<xPre+4; X++){
                        xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                        yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                        coord++;
                    }
                }
            }
            oldBasis=basis;

            PrecisionTYPE Tx_x=0.0;
            PrecisionTYPE Ty_x=0.0;
            PrecisionTYPE Tx_y=0.0;
            PrecisionTYPE Ty_y=0.0;

            for(int a=0; a<16; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];

                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }

            memset(&jacobianMatrix, 0, sizeof(mat33));
            jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
            jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
            jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
            jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
            jacobianMatrix.m[2][2]=1.0f;

            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);

            *jacobianDeterminantPtr++ = (jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1])
                                        - (jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0]);
            *invertedJacobianMatricesPtr++ = nifti_mat33_inverse(jacobianMatrix);
        }
    }
}
/* *************************************************************** */
template <class PrecisionTYPE, class SplineTYPE>
void computeJacobianMatrices_3D(nifti_image *targetImage,
                                nifti_image *splineControlPoint,
                                mat33 *invertedJacobianMatrices,
                                PrecisionTYPE *jacobianDeterminant)
{
#if _USE_SSE
    if(sizeof(PrecisionTYPE)!=4){
        fprintf(stderr, "***ERROR***\tcomputeJacobianMatrices_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    PrecisionTYPE yBasis[4],yFirst[4],xBasis[4],xFirst[4],zBasis[4],zFirst[4];
    PrecisionTYPE tempX[16], tempY[16], tempZ[16];
    PrecisionTYPE basisX[64], basisY[64], basisZ[64];
    PrecisionTYPE basis, FF, FFF, MF, oldBasis=(PrecisionTYPE)(1.1);

    PrecisionTYPE xControlPointCoordinates[64];
    PrecisionTYPE yControlPointCoordinates[64];
    PrecisionTYPE zControlPointCoordinates[64];

    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    unsigned int coord=0;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    reorient=nifti_mat33_inverse(desorient);
    mat33 jacobianMatrix;

    // The inverted Jacobian matrices are first computed for every voxel
    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    PrecisionTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((PrecisionTYPE)z/gridVoxelSpacing[2]);
        basis=(PrecisionTYPE)z/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(PrecisionTYPE)(1.0-basis);
        zBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
        zBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        zBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        zBasis[3] = (PrecisionTYPE)(FFF/6.0);
        zFirst[3]= (PrecisionTYPE)(FF / 2.0);
        zFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - zFirst[3]);
        zFirst[2]= (PrecisionTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
        zFirst[1]= - zFirst[0] - zFirst[2] - zFirst[3];

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((PrecisionTYPE)y/gridVoxelSpacing[1]);
            basis=(PrecisionTYPE)y/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
            yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            yBasis[3] = (PrecisionTYPE)(FFF/6.0);
            yFirst[3]= (PrecisionTYPE)(FF / 2.0);
            yFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - yFirst[3]);
            yFirst[2]= (PrecisionTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
            yFirst[1]= - yFirst[0] - yFirst[2] - yFirst[3];
#if _USE_SSE
            val.f[0]=yBasis[0];
            val.f[1]=yBasis[1];
            val.f[2]=yBasis[2];
            val.f[3]=yBasis[3];
            __m128 _yBasis=val.m;
            val.f[0]=yFirst[0];
            val.f[1]=yFirst[1];
            val.f[2]=yFirst[2];
            val.f[3]=yFirst[3];
            __m128 _yFirst=val.m;
            __m128 *ptrBasisX = (__m128 *) &tempX[0];
            __m128 *ptrBasisY = (__m128 *) &tempY[0];
            __m128 *ptrBasisZ = (__m128 *) &tempZ[0];
            for(int a=0;a<4;++a){
                val.m=_mm_set_ps1(zBasis[a]);
                *ptrBasisX=_mm_mul_ps(_yBasis,val.m);
                *ptrBasisY=_mm_mul_ps(_yFirst,val.m);
                val.m=_mm_set_ps1(zFirst[a]);
                *ptrBasisZ=_mm_mul_ps(_yBasis,val.m);
                ptrBasisX++;
                ptrBasisY++;
                ptrBasisZ++;
            }
#else
            coord=0;
            for(int c=0; c<4; c++){
                for(int b=0; b<4; b++){
                    tempX[coord]=zBasis[c]*yBasis[b]; // z * y
                    tempY[coord]=zBasis[c]*yFirst[b];// z * y'
                    tempZ[coord]=zFirst[c]*yBasis[b]; // z'* y
                    coord++;
                }
            }
#endif

            for(int x=0; x<targetImage->nx; x++){

                int xPre=(int)((PrecisionTYPE)x/gridVoxelSpacing[0]);
                basis=(PrecisionTYPE)x/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/6.0);
                xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                xBasis[3] = (PrecisionTYPE)(FFF/6.0);
                xFirst[3]= (PrecisionTYPE)(FF / 2.0);
                xFirst[0]= (PrecisionTYPE)(basis - 1.0/2.0 - xFirst[3]);
                xFirst[2]= (PrecisionTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                xFirst[1]= - xFirst[0] - xFirst[2] - xFirst[3];

#if _USE_SSE
                val.f[0]=xBasis[0];
                val.f[1]=xBasis[1];
                val.f[2]=xBasis[2];
                val.f[3]=xBasis[3];
                __m128 _xBasis=val.m;
                val.f[0]=xFirst[0];
                val.f[1]=xFirst[1];
                val.f[2]=xFirst[2];
                val.f[3]=xFirst[3];
                __m128 _xFirst=val.m;
                ptrBasisX = (__m128 *) &basisX[0];
                ptrBasisY = (__m128 *) &basisY[0];
                ptrBasisZ = (__m128 *) &basisZ[0];
                for(int a=0;a<16;++a){
                    val.m=_mm_set_ps1(tempX[a]);
                    *ptrBasisX=_mm_mul_ps(_xFirst,val.m);
                    val.m=_mm_set_ps1(tempY[a]);
                    *ptrBasisY=_mm_mul_ps(_xBasis,val.m);
                    val.m=_mm_set_ps1(tempZ[a]);
                    *ptrBasisZ=_mm_mul_ps(_xBasis,val.m);
                    ptrBasisX++;
                    ptrBasisY++;
                    ptrBasisZ++;
                }
#else
                coord=0;
                for(int bc=0; bc<16; bc++){
                    for(int a=0; a<4; a++){
                        basisX[coord]=tempX[bc]*xFirst[a];   // z * y * x'
                        basisY[coord]=tempY[bc]*xBasis[a];    // z * y'* x
                        basisZ[coord]=tempZ[bc]*xBasis[a];    // z'* y * x
                        coord++;
                    }
                }
#endif

                if(basis<=oldBasis || x==0){
                    coord=0;
                    for(int Z=zPre; Z<zPre+4; Z++){
                        unsigned int index = Z*splineControlPoint->nx*splineControlPoint->ny;
                        SplineTYPE *xPtr = &controlPointPtrX[index];
                        SplineTYPE *yPtr = &controlPointPtrY[index];
                        SplineTYPE *zPtr = &controlPointPtrZ[index];
                        for(int Y=yPre; Y<yPre+4; Y++){
                            unsigned int index = Y*splineControlPoint->nx;
                            SplineTYPE *xxPtr = &xPtr[index];
                            SplineTYPE *yyPtr = &yPtr[index];
                            SplineTYPE *zzPtr = &zPtr[index];
                            for(int X=xPre; X<xPre+4; X++){
                                xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                                yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                                zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;

                PrecisionTYPE Tx_x=0.0;
                PrecisionTYPE Ty_x=0.0;
                PrecisionTYPE Tz_x=0.0;
                PrecisionTYPE Tx_y=0.0;
                PrecisionTYPE Ty_y=0.0;
                PrecisionTYPE Tz_y=0.0;
                PrecisionTYPE Tx_z=0.0;
                PrecisionTYPE Ty_z=0.0;
                PrecisionTYPE Tz_z=0.0;

#if _USE_SSE
                __m128 tempX_x =  _mm_set_ps1(0.0);
                __m128 tempX_y =  _mm_set_ps1(0.0);
                __m128 tempX_z =  _mm_set_ps1(0.0);
                __m128 tempY_x =  _mm_set_ps1(0.0);
                __m128 tempY_y =  _mm_set_ps1(0.0);
                __m128 tempY_z =  _mm_set_ps1(0.0);
                __m128 tempZ_x =  _mm_set_ps1(0.0);
                __m128 tempZ_y =  _mm_set_ps1(0.0);
                __m128 tempZ_z =  _mm_set_ps1(0.0);
                __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];
                ptrBasisX   = (__m128 *) &basisX[0];
                ptrBasisY   = (__m128 *) &basisY[0];
                ptrBasisZ   = (__m128 *) &basisZ[0];
                //addition and multiplication of the 16 basis value and CP position for each axis
                for(unsigned int a=0; a<16; a++){
                    tempX_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrX), tempX_x );
                    tempX_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrX), tempX_y );
                    tempX_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrX), tempX_z );

                    tempY_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrY), tempY_x );
                    tempY_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrY), tempY_y );
                    tempY_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrY), tempY_z );

                    tempZ_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrZ), tempZ_x );
                    tempZ_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrZ), tempZ_y );
                    tempZ_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrZ), tempZ_z );

                    ptrBasisX++;
                    ptrBasisY++;
                    ptrBasisZ++;
                    ptrX++;
                    ptrY++;
                    ptrZ++;
                }

                //the values stored in SSE variables are transfered to normal float
                val.m = tempX_x;
                Tx_x = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempX_y;
                Tx_y = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempX_z;
                Tx_z = val.f[0]+val.f[1]+val.f[2]+val.f[3];

                val.m = tempY_x;
                Ty_x = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempY_y;
                Ty_y = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempY_z;
                Ty_z = val.f[0]+val.f[1]+val.f[2]+val.f[3];

                val.m = tempZ_x;
                Tz_x = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempZ_y;
                Tz_y = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempZ_z;
                Tz_z = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                for(int a=0; a<64; a++){
                    Tx_x += basisX[a]*xControlPointCoordinates[a];
                    Tx_y += basisY[a]*xControlPointCoordinates[a];
                    Tx_z += basisZ[a]*xControlPointCoordinates[a];

                    Ty_x += basisX[a]*yControlPointCoordinates[a];
                    Ty_y += basisY[a]*yControlPointCoordinates[a];
                    Ty_z += basisZ[a]*yControlPointCoordinates[a];

                    Tz_x += basisX[a]*zControlPointCoordinates[a];
                    Tz_y += basisY[a]*zControlPointCoordinates[a];
                    Tz_z += basisZ[a]*zControlPointCoordinates[a];
                }
#endif

                jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
                jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
                jacobianMatrix.m[0][2]= (float)(Tx_z / splineControlPoint->dz);
                jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
                jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
                jacobianMatrix.m[1][2]= (float)(Ty_z / splineControlPoint->dz);
                jacobianMatrix.m[2][0]= (float)(Tz_x / splineControlPoint->dx);
                jacobianMatrix.m[2][1]= (float)(Tz_y / splineControlPoint->dy);
                jacobianMatrix.m[2][2]= (float)(Tz_z / splineControlPoint->dz);

                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                *jacobianDeterminantPtr++ = nifti_mat33_determ(jacobianMatrix);
                *invertedJacobianMatricesPtr++ = nifti_mat33_inverse(jacobianMatrix);
            }
        }
    }
}
/* *************************************************************** */
template <class PrecisionTYPE, class SplineTYPE>
void computeApproximateJacobianMatrices_2D( nifti_image *splineControlPoint,
                                            mat33 *invertedJacobianMatrices,
                                            PrecisionTYPE *jacobianDeterminant)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    PrecisionTYPE normal[3];
    PrecisionTYPE first[3];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);

    PrecisionTYPE basisX[9], basisY[9];

    int coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisX[coord]=normal[b]*first[a];   // y * x'
            basisY[coord]=first[b]*normal[a];   // y'* x
            coord++;
        }
    }

    PrecisionTYPE xControlPointCoordinates[9];
    PrecisionTYPE yControlPointCoordinates[9];

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    mat33 jacobianMatrix;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
        (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    PrecisionTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* All the Jacobian matrices are computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    // Loop over (almost) each control point
    for(int y=1;y<splineControlPoint->ny-1;y++){
        unsigned int jacIndex = y*splineControlPoint->nx + 1;
        for(int x=1;x<splineControlPoint->nx-1;x++){

            // The control points are stored
            coord=0;
            for(int Y=y-1; Y<y+2; Y++){
                unsigned int index = Y*splineControlPoint->nx;
                SplineTYPE *xxPtr = &controlPointPtrX[index];
                SplineTYPE *yyPtr = &controlPointPtrY[index];
                for(int X=x-1; X<x+2; X++){
                    xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                    yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                    coord++;
                }
            }

            PrecisionTYPE Tx_x=(PrecisionTYPE)0.0;
            PrecisionTYPE Ty_x=(PrecisionTYPE)0.0;
            PrecisionTYPE Tx_y=(PrecisionTYPE)0.0;
            PrecisionTYPE Ty_y=(PrecisionTYPE)0.0;

            for(int a=0; a<9; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];
                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }

            memset(&jacobianMatrix, 0, sizeof(mat33));
            jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
            jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
            jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
            jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
            jacobianMatrix.m[2][2]=1.0f;

            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);

            jacobianDeterminantPtr[jacIndex] = (jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1])
                                        - (jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0]);
            invertedJacobianMatricesPtr[jacIndex] = nifti_mat33_inverse(jacobianMatrix);
            jacIndex++;
        } // x
    } // y
}
/* *************************************************************** */
template <class PrecisionTYPE, class SplineTYPE>
void computeApproximateJacobianMatrices_3D( nifti_image *splineControlPoint,
                                            mat33 *invertedJacobianMatrices,
                                            PrecisionTYPE *jacobianDeterminant)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
	float basisX[27] = {-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f, 
		-0.0555556f, 0.0000000f, 0.0555556f, -0.2222222f, 0.0000000f, 0.2222222f, -0.0555556f, 0.0000000f, 0.0555556f, 
		-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f};
	float basisY[27] = {-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f, 
		-0.0555556f, -0.2222222f, -0.0555556f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0555556f, 0.2222222f, 0.0555556f, 
		-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f};
	float basisZ[27] = {-0.0138889f, -0.0555556f, -0.0138889f, -0.0555556f, -0.2222222f, -0.0555556f, -0.0138889f, -0.0555556f, -0.0138889f, 
		0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 
		0.0138889f, 0.0555556f, 0.0138889f, 0.0555556f, 0.2222222f, 0.0555556f, 0.0138889f, 0.0555556f, 0.0138889f};
    int coord=0;
    PrecisionTYPE xControlPointCoordinates[27];
    PrecisionTYPE yControlPointCoordinates[27];
    PrecisionTYPE zControlPointCoordinates[27];

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    reorient=nifti_mat33_inverse(desorient);
    mat33 jacobianMatrix;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
        (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>
        (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* All the Jacobian matrices are computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    PrecisionTYPE *jacobianDeterminantPtr = jacobianDeterminant;

//	// Loop over (almost) each control point
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            unsigned int jacIndex = (z*splineControlPoint->ny+y)*splineControlPoint->nx+1;
            for(int x=1;x<splineControlPoint->nx-1;x++){

                // The control points are stored
                coord=0;
                for(int Z=z-1; Z<z+2; Z++){
                    unsigned int index = Z*splineControlPoint->nx*splineControlPoint->ny;
                    SplineTYPE *xPtr = &controlPointPtrX[index];
                    SplineTYPE *yPtr = &controlPointPtrY[index];
                    SplineTYPE *zPtr = &controlPointPtrZ[index];
                    for(int Y=y-1; Y<y+2; Y++){
                        unsigned int index = Y*splineControlPoint->nx;
                        SplineTYPE *xxPtr = &xPtr[index];
                        SplineTYPE *yyPtr = &yPtr[index];
                        SplineTYPE *zzPtr = &zPtr[index];
                        for(int X=x-1; X<x+2; X++){
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X];
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X];
                            zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X];
                            coord++;
                        }
                    }
                }

                PrecisionTYPE Tx_x=(PrecisionTYPE)0.0;
                PrecisionTYPE Ty_x=(PrecisionTYPE)0.0;
                PrecisionTYPE Tz_x=(PrecisionTYPE)0.0;
                PrecisionTYPE Tx_y=(PrecisionTYPE)0.0;
                PrecisionTYPE Ty_y=(PrecisionTYPE)0.0;
                PrecisionTYPE Tz_y=(PrecisionTYPE)0.0;
                PrecisionTYPE Tx_z=(PrecisionTYPE)0.0;
                PrecisionTYPE Ty_z=(PrecisionTYPE)0.0;
                PrecisionTYPE Tz_z=(PrecisionTYPE)0.0;

                for(int a=0; a<27; a++){
                    Tx_x += basisX[a]*xControlPointCoordinates[a];
                    Tx_y += basisY[a]*xControlPointCoordinates[a];
                    Tx_z += basisZ[a]*xControlPointCoordinates[a];
                    Ty_x += basisX[a]*yControlPointCoordinates[a];
                    Ty_y += basisY[a]*yControlPointCoordinates[a];
                    Ty_z += basisZ[a]*yControlPointCoordinates[a];
                    Tz_x += basisX[a]*zControlPointCoordinates[a];
                    Tz_y += basisY[a]*zControlPointCoordinates[a];
                    Tz_z += basisZ[a]*zControlPointCoordinates[a];
                }


                jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
                jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
                jacobianMatrix.m[0][2]= (float)(Tx_z / splineControlPoint->dz);
                jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
                jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
                jacobianMatrix.m[1][2]= (float)(Ty_z / splineControlPoint->dz);
                jacobianMatrix.m[2][0]= (float)(Tz_x / splineControlPoint->dx);
                jacobianMatrix.m[2][1]= (float)(Tz_y / splineControlPoint->dy);
                jacobianMatrix.m[2][2]= (float)(Tz_z / splineControlPoint->dz);

                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                jacobianDeterminantPtr[jacIndex] = nifti_mat33_determ(jacobianMatrix);
                invertedJacobianMatricesPtr[jacIndex] = nifti_mat33_inverse(jacobianMatrix);
                jacIndex++;
            } // x
        } // y
    } //z
}
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradient2D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(targetImage->nvox * sizeof(PrecisionTYPE));

    computeJacobianMatrices_2D<PrecisionTYPE, SplineTYPE>(  targetImage,
                                                            splineControlPoint,
                                                            invertedJacobianMatrices,
                                                            jacobianDeterminant);

    PrecisionTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    PrecisionTYPE basisValues[2];
    PrecisionTYPE xBasis, yBasis, basis;
    PrecisionTYPE xFirst, yFirst;
    unsigned int jacIndex;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;
	
	// The gradient are now computed for every control point
	SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny];

	for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){
			
			PrecisionTYPE jacobianConstraintX=(PrecisionTYPE)0.0;
            PrecisionTYPE jacobianConstraintY=(PrecisionTYPE)0.0;
			
			// Loop over all the control points in the surrounding area
            for(int pixelY=(int)ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)floor((y+1)*gridVoxelSpacing[1]); pixelY++){
				if(pixelY>-1 && pixelY<targetImage->ny){
					
					int yPre=(int)((PrecisionTYPE)pixelY/gridVoxelSpacing[1]);
					basis=(PrecisionTYPE)pixelY/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
					if(basis<0.0) basis=0.0; //rounding error
					
					switch(y-yPre){
						case 0:
                            yBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                            yFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
							break;
						case 1:
                            yBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                            yFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
							break;
						case 2:
                            yBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                            yFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
							break;
						case 3:
                            yBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                            yFirst=(PrecisionTYPE)(basis*basis/2.0);
							break;
						default:
                            yBasis=(PrecisionTYPE)0.0;
                            yFirst=(PrecisionTYPE)0.0;
							break;
					}
					
                    for(int pixelX=(int)ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)floor((x+1)*gridVoxelSpacing[0]); pixelX++){
						if(pixelX>-1 && pixelX<targetImage->nx){
							
							int xPre=(int)((PrecisionTYPE)pixelX/gridVoxelSpacing[0]);
							basis=(PrecisionTYPE)pixelX/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
							if(basis<0.0) basis=0.0; //rounding error

							switch(x-xPre){
								case 0:
                                    xBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                    xFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
									break;
								case 1:
                                    xBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                    xFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
									break;
								case 2:
                                    xBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                    xFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
									break;
								case 3:
                                    xBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                    xFirst=(PrecisionTYPE)(basis*basis/2.0);
									break;
								default:
                                    xBasis=(PrecisionTYPE)0.0;
                                    xFirst=(PrecisionTYPE)0.0;
									break;
							}
							
                            basisValues[0]= xFirst * yBasis;
                            basisValues[1]= xBasis * yFirst;
                            jacIndex = pixelY*targetImage->nx+pixelX;
                            PrecisionTYPE logDet=jacobianDeterminant[jacIndex];
                            jacobianMatrix = invertedJacobianMatrices[jacIndex];

                            if(logDet>(PrecisionTYPE)0.0){
                                /* derivative of the squared log of the Jacobian determinant */
                                logDet=(PrecisionTYPE)(2.0*log(logDet));
                                jacobianConstraintX += logDet * (jacobianMatrix.m[0][0]*basisValues[0] + jacobianMatrix.m[0][1]*basisValues[1]);
                                jacobianConstraintY += logDet * (jacobianMatrix.m[1][0]*basisValues[0] + jacobianMatrix.m[1][1]*basisValues[1]);
                            }
						}// if x
					}// x
				}// if y
			}// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            *gradientImagePtrX++ += weight *
                (desorient.m[0][0]*jacobianConstraintX +
                desorient.m[0][1]*jacobianConstraintY);
            *gradientImagePtrY++ += weight *
                (desorient.m[1][0]*jacobianConstraintX +
                desorient.m[1][1]*jacobianConstraintY);
		}
    }

    free(jacobianDeterminant);
    free(invertedJacobianMatrices);
	
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradientApprox2D(  nifti_image *splineControlPoint,
                                                        nifti_image *targetImage,
                                                        nifti_image *gradientImage,
                                                        float weight)
{
    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(jacobianNumber * sizeof(PrecisionTYPE));

    computeApproximateJacobianMatrices_2D<PrecisionTYPE, SplineTYPE>(   splineControlPoint,
                                                                        invertedJacobianMatrices,
                                                                        jacobianDeterminant);

    PrecisionTYPE basisValues[2];
    PrecisionTYPE xBasis, yBasis;
    PrecisionTYPE xFirst, yFirst;
    unsigned int jacIndex;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = static_cast<SplineTYPE *>
        (&gradientImagePtrX[gradientImage->nx*gradientImage->ny]);

    PrecisionTYPE approxRatio = weight * targetImage->nvox
        / ( splineControlPoint->nx*splineControlPoint->ny );

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            PrecisionTYPE jacobianConstraintX=(PrecisionTYPE)0.0;
            PrecisionTYPE jacobianConstraintY=(PrecisionTYPE)0.0;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)((y-1));pixelY<(int)((y+2)); pixelY++){
                if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                    switch(pixelY-y){
                        case -1:
                            yBasis=(PrecisionTYPE)(1.0/6.0);
                            yFirst=(PrecisionTYPE)(0.5);
                            break;
                        case 0:
                            yBasis=(PrecisionTYPE)(2.0/3.0);
                            yFirst=(PrecisionTYPE)(0.0);
                            break;
                        case 1:
                            yBasis=(PrecisionTYPE)(1.0/6.0);
                            yFirst=(PrecisionTYPE)(-0.5);
                            break;
                        default:
                            yBasis=(PrecisionTYPE)0.0;
                            yFirst=(PrecisionTYPE)0.0;
                            break;
                    }
                    for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                            switch(pixelX-x){
                                case -1:
                                    xBasis=(PrecisionTYPE)(1.0/6.0);
                                    xFirst=(PrecisionTYPE)(0.5);
                                    break;
                                case 0:
                                    xBasis=(PrecisionTYPE)(2.0/3.0);
                                    xFirst=(PrecisionTYPE)(0.0);
                                    break;
                                case 1:
                                    xBasis=(PrecisionTYPE)(1.0/6.0);
                                    xFirst=(PrecisionTYPE)(-0.5);
                                    break;
                                default:
                                    xBasis=(PrecisionTYPE)0.0;
                                    xFirst=(PrecisionTYPE)0.0;
                                    break;
                            }

                            basisValues[0] = xFirst * yBasis ;
                            basisValues[1] = xBasis * yFirst ;

                            jacIndex = pixelY*splineControlPoint->nx+pixelX;
                            PrecisionTYPE logDet=jacobianDeterminant[jacIndex];
                            jacobianMatrix = invertedJacobianMatrices[jacIndex];

                            if(logDet>(PrecisionTYPE)0.0){
                                /* derivative of the squared log of the Jacobian determinant */
                                logDet=(PrecisionTYPE)(2.0*log(logDet));
                                jacobianConstraintX += logDet * (jacobianMatrix.m[0][0]*basisValues[0] + jacobianMatrix.m[0][1]*basisValues[1]);
                                jacobianConstraintY += logDet * (jacobianMatrix.m[1][0]*basisValues[0] + jacobianMatrix.m[1][1]*basisValues[1]);
                            }
                        } // if x
                    }// x
                }// if y
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            *gradientImagePtrX++ += approxRatio *
                (desorient.m[0][0]*jacobianConstraintX +
                desorient.m[0][1]*jacobianConstraintY);
            *gradientImagePtrY++ += approxRatio *
                (desorient.m[1][0]*jacobianConstraintX +
                desorient.m[1][1]*jacobianConstraintY);
        }
    }
    free(invertedJacobianMatrices);
    free(jacobianDeterminant);
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradient3D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(targetImage->nvox * sizeof(PrecisionTYPE));

    computeJacobianMatrices_3D<PrecisionTYPE, SplineTYPE>(  targetImage,
                                                            splineControlPoint,
                                                            invertedJacobianMatrices,
                                                            jacobianDeterminant);

    /*  */
    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;

    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    PrecisionTYPE xBasis, yBasis, zBasis, basis;
    PrecisionTYPE xFirst, yFirst, zFirst;
    PrecisionTYPE basisValues[3];
    unsigned int jacIndex;
	
	// The gradient are now computed for every control point
	SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz];
    SplineTYPE *gradientImagePtrZ = &gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz];
	
	for(int z=0;z<splineControlPoint->nz;z++){
		for(int y=0;y<splineControlPoint->ny;y++){
			for(int x=0;x<splineControlPoint->nx;x++){
				
				PrecisionTYPE jacobianConstraintX=(PrecisionTYPE)0.0;
				PrecisionTYPE jacobianConstraintY=(PrecisionTYPE)0.0;
                PrecisionTYPE jacobianConstraintZ=(PrecisionTYPE)0.0;
				
                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)ceil((z-3)*gridVoxelSpacing[2]);pixelZ<=(int)floor((z+1)*gridVoxelSpacing[2]); pixelZ++){
					if(pixelZ>-1 && pixelZ<targetImage->nz){

						int zPre=(int)((PrecisionTYPE)pixelZ/gridVoxelSpacing[2]);
						basis=(PrecisionTYPE)pixelZ/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
                        if(basis<0.0) basis=0.0; //rounding error
						
						switch(z-zPre){
							case 0:
                                zBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                zFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
								break;
							case 1:
                                zBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                zFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
								break;
							case 2:
                                zBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                zFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
								break;
							case 3:
                                zBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                zFirst=(PrecisionTYPE)(basis*basis/2.0);
								break;
							default:
                                zBasis=(PrecisionTYPE)0.0;
                                zFirst=(PrecisionTYPE)0.0;
								break;
                        }
						
                        for(int pixelY=(int)ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)floor((y+1)*gridVoxelSpacing[1]); pixelY++){
							if(pixelY>-1 && pixelY<targetImage->ny){
						
								int yPre=(int)((PrecisionTYPE)pixelY/gridVoxelSpacing[1]);
								basis=(PrecisionTYPE)pixelY/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
								if(basis<0.0) basis=0.0; //rounding error
								
								switch(y-yPre){
									case 0:
                                        yBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                        yFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
										break;
									case 1:
                                        yBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                        yFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
										break;
									case 2:
                                        yBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                        yFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
										break;
									case 3:
                                        yBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                        yFirst=(PrecisionTYPE)(basis*basis/2.0);
										break;
									default:
                                        yBasis=(PrecisionTYPE)0.0;
                                        yFirst=(PrecisionTYPE)0.0;
										break;
								}
								
                                for(int pixelX=(int)ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)floor((x+1)*gridVoxelSpacing[0]); pixelX++){
									if(pixelX>-1 && pixelX<targetImage->nx){
										
										int xPre=(int)((PrecisionTYPE)pixelX/gridVoxelSpacing[0]);
										basis=(PrecisionTYPE)pixelX/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
										if(basis<0.0) basis=0.0; //rounding error
										
										switch(x-xPre){
											case 0:
                                                xBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                                xFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
												break;
											case 1:
                                                xBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                                xFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
												break;
											case 2:
                                                xBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                                xFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
												break;
											case 3:
                                                xBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                                xFirst=(PrecisionTYPE)(basis*basis/2.0);
												break;
											default:
                                                xBasis=(PrecisionTYPE)0.0;
                                                xFirst=(PrecisionTYPE)0.0;
												break;
										}

                                        basisValues[0]= xFirst * yBasis * zBasis ;
                                        basisValues[1]= xBasis * yFirst * zBasis ;
                                        basisValues[2]= xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+pixelX;
                                        jacobianMatrix = invertedJacobianMatrices[jacIndex];
                                        PrecisionTYPE detJac = jacobianDeterminant[jacIndex];
                                        if(detJac>0.0){
                                            detJac = 2.0 * log(detJac);
                                            jacobianConstraintX += detJac *
                                                                (   jacobianMatrix.m[0][0]*basisValues[0]
                                                                +   jacobianMatrix.m[0][1]*basisValues[1]
                                                                +   jacobianMatrix.m[0][2]*basisValues[2]);
                                            jacobianConstraintY += detJac *
                                                                (   jacobianMatrix.m[1][0]*basisValues[0]
                                                                +   jacobianMatrix.m[1][1]*basisValues[1]
                                                                +   jacobianMatrix.m[1][2]*basisValues[2]);
                                            jacobianConstraintZ += detJac *
                                                                (   jacobianMatrix.m[2][0]*basisValues[0]
                                                                +   jacobianMatrix.m[2][1]*basisValues[1]
                                                                +   jacobianMatrix.m[2][2]*basisValues[2]);
                                        }
									} // if x
								}// x
							}// if y
						}// y
					}// if z
				} // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                *gradientImagePtrX++ += weight *
                                     ( desorient.m[0][0]*jacobianConstraintX
                                     + desorient.m[0][1]*jacobianConstraintY
                                     + desorient.m[0][2]*jacobianConstraintZ);
                *gradientImagePtrY++ += weight *
                                     ( desorient.m[1][0]*jacobianConstraintX
                                     + desorient.m[1][1]*jacobianConstraintY
                                     + desorient.m[1][2]*jacobianConstraintZ);
                *gradientImagePtrZ++ += weight *
                                     ( desorient.m[2][0]*jacobianConstraintX
                                     + desorient.m[2][1]*jacobianConstraintY
                                     + desorient.m[2][2]*jacobianConstraintZ);
			}
		}
    }
    free(invertedJacobianMatrices);
    free(jacobianDeterminant);
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradientApprox3D(  nifti_image *splineControlPoint,
													            nifti_image *targetImage,
													            nifti_image *gradientImage,
                                                                float weight)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(jacobianNumber * sizeof(PrecisionTYPE));

    computeApproximateJacobianMatrices_3D<PrecisionTYPE, SplineTYPE>(   splineControlPoint,
                                                                        invertedJacobianMatrices,
                                                                        jacobianDeterminant);

    /*  */
    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;

    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    PrecisionTYPE xBasis, yBasis, zBasis;
    PrecisionTYPE xFirst, yFirst, zFirst;
    PrecisionTYPE basisValues[3];
    unsigned int jacIndex;

    // The gradient are now computed for every control point
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz];
    SplineTYPE *gradientImagePtrZ = &gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz];

	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	/* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    PrecisionTYPE approxRatio = weight * targetImage->nvox
        / ( jacobianNumber );
	
	for(int z=0;z<splineControlPoint->nz;z++){
		for(int y=0;y<splineControlPoint->ny;y++){
			for(int x=0;x<splineControlPoint->nx;x++){
				
                PrecisionTYPE jacobianConstraintX=(PrecisionTYPE)0.0;
                PrecisionTYPE jacobianConstraintY=(PrecisionTYPE)0.0;
                PrecisionTYPE jacobianConstraintZ=(PrecisionTYPE)0.0;

				// Loop over all the control points in the surrounding area
				for(int pixelZ=(int)((z-1));pixelZ<(int)((z+2)); pixelZ++){
					if(pixelZ>0 && pixelZ<splineControlPoint->nz-1){

						switch(pixelZ-z){
							case -1:
								zBasis=(PrecisionTYPE)(1.0/6.0);
								zFirst=(PrecisionTYPE)(0.5);
								break;
							case 0:
								zBasis=(PrecisionTYPE)(2.0/3.0);
								zFirst=(PrecisionTYPE)(0.0);
								break;
							case 1:
								zBasis=(PrecisionTYPE)(1.0/6.0);
								zFirst=(PrecisionTYPE)(-0.5);
								break;
							default:
								zBasis=(PrecisionTYPE)0.0;
								zFirst=(PrecisionTYPE)0.0;
								break;
						}
						for(int pixelY=(int)((y-1));pixelY<(int)((y+2)); pixelY++){
							if(pixelY>0 && pixelY<splineControlPoint->ny-1){

								switch(pixelY-y){
									case -1:
										yBasis=(PrecisionTYPE)(1.0/6.0);
										yFirst=(PrecisionTYPE)(0.5);
										break;
									case 0:
										yBasis=(PrecisionTYPE)(2.0/3.0);
										yFirst=(PrecisionTYPE)(0.0);
										break;
									case 1:
										yBasis=(PrecisionTYPE)(1.0/6.0);
										yFirst=(PrecisionTYPE)(-0.5);
										break;
									default:
										yBasis=(PrecisionTYPE)0.0;
										yFirst=(PrecisionTYPE)0.0;
										break;
								}
								for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
									if(pixelX>0 && pixelX<splineControlPoint->nx-1){

										switch(pixelX-x){
											case -1:
												xBasis=(PrecisionTYPE)(1.0/6.0);
												xFirst=(PrecisionTYPE)(0.5);
												break;
											case 0:
												xBasis=(PrecisionTYPE)(2.0/3.0);
												xFirst=(PrecisionTYPE)(0.0);
												break;
											case 1:
												xBasis=(PrecisionTYPE)(1.0/6.0);
												xFirst=(PrecisionTYPE)(-0.5);
												break;
											default:
												xBasis=(PrecisionTYPE)0.0;
												xFirst=(PrecisionTYPE)0.0;
												break;
										}

										basisValues[0] = xFirst * yBasis * zBasis ;
										basisValues[1] = xBasis * yFirst * zBasis ;
                                        basisValues[2] = xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*splineControlPoint->ny+pixelY)*splineControlPoint->nx+pixelX;
                                        jacobianMatrix = invertedJacobianMatrices[jacIndex];
                                        PrecisionTYPE detJac = jacobianDeterminant[jacIndex];

                                        if(detJac>0.0){
                                            detJac = 2.0 * log(detJac);
                                            jacobianConstraintX += detJac *
                                                                (   jacobianMatrix.m[0][0]*basisValues[0]
                                                                +   jacobianMatrix.m[0][1]*basisValues[1]
                                                                +   jacobianMatrix.m[0][2]*basisValues[2]);
                                            jacobianConstraintY += detJac *
                                                                (   jacobianMatrix.m[1][0]*basisValues[0]
                                                                +   jacobianMatrix.m[1][1]*basisValues[1]
                                                                +   jacobianMatrix.m[1][2]*basisValues[2]);
                                            jacobianConstraintZ += detJac *
                                                                (   jacobianMatrix.m[2][0]*basisValues[0]
                                                                +   jacobianMatrix.m[2][1]*basisValues[1]
                                                                +   jacobianMatrix.m[2][2]*basisValues[2]);
                                        }
                                    } // if x
                                }// x
                            }// if y
                        }// y
                    }// if z
                } // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                *gradientImagePtrX++ += approxRatio *
                                     ( desorient.m[0][0]*jacobianConstraintX
                                     + desorient.m[0][1]*jacobianConstraintY
                                     + desorient.m[0][2]*jacobianConstraintZ);
                *gradientImagePtrY++ += approxRatio *
                                     ( desorient.m[1][0]*jacobianConstraintX
                                     + desorient.m[1][1]*jacobianConstraintY
                                     + desorient.m[1][2]*jacobianConstraintZ);
                *gradientImagePtrZ++ += approxRatio *
                                     ( desorient.m[2][0]*jacobianConstraintX
                                     + desorient.m[2][1]*jacobianConstraintY
                                     + desorient.m[2][2]*jacobianConstraintZ);
            }
        }
    }
    free(invertedJacobianMatrices);
    free(jacobianDeterminant);
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE>
void reg_bspline_jacobianDeterminantGradient(	nifti_image *splineControlPoint,
						                        nifti_image *targetImage,
						                        nifti_image *gradientImage,
						                        float weight,
                                                bool approx)
{
	if(splineControlPoint->datatype != gradientImage->datatype){
		
		fprintf(stderr,"The spline control point image and the gradient image were expected to have the same datatype\n");
		fprintf(stderr,"The bending energy gradient has not computed\n");
	}
	
    if(splineControlPoint->nz==1){
		if(approx){
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradientApprox2D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage, gradientImage, weight);
				case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradientApprox2D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage, gradientImage, weight);
				default:
					fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The jacobian penalty gradient has not computed\n");
					exit(1);
			}		
		}
		else{
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradient2D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradient2D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The jacobian penalty gradient has not computed\n");
                    exit(1);
			}
		}
    }
	else{
		if(approx){
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradientApprox3D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradientApprox3D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The jacobian penalty gradient has not computed\n");
                    exit(1);
                    break;
			}		
		}
		else{
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradient3D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradient3D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The jacobian penalty gradient has not computed\n");
                    exit(1);
                    break;
			}
		}
	}
}
template void reg_bspline_jacobianDeterminantGradient<float>(nifti_image *, nifti_image *, nifti_image *, float, bool);
template void reg_bspline_jacobianDeterminantGradient<double>(nifti_image *, nifti_image *, nifti_image *, float, bool);
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_correctFolding_2D(nifti_image *splineControlPoint,
                                            nifti_image *targetImage)
{

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(targetImage->nvox * sizeof(PrecisionTYPE));

    computeJacobianMatrices_2D<PrecisionTYPE, SplineTYPE>(  targetImage,
                                                            splineControlPoint,
                                                            invertedJacobianMatrices,
                                                            jacobianDeterminant);

    // The current Penalty term value is computed
    double penaltyTerm =0.0;
    for(unsigned int i=0; i< targetImage->nvox; i++){
        double logDet = log(jacobianDeterminant[i]);
        penaltyTerm += logDet*logDet;
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(invertedJacobianMatrices);
        return (PrecisionTYPE)(penaltyTerm/(PrecisionTYPE)targetImage->nvox);
    }

    PrecisionTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    PrecisionTYPE basisValues[2];
    PrecisionTYPE xBasis, yBasis, basis;
    PrecisionTYPE xFirst, yFirst;
    unsigned int jacIndex;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;

    // The gradient are now computed for every control point
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            PrecisionTYPE foldingCorrectionX=(PrecisionTYPE)0.0;
            PrecisionTYPE foldingCorrectionY=(PrecisionTYPE)0.0;

            bool correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)ceil((y-2)*gridVoxelSpacing[1]);pixelY<(int)floor((y)*gridVoxelSpacing[1]); pixelY++){
                if(pixelY>-1 && pixelY<targetImage->ny){

                    int yPre=(int)((PrecisionTYPE)pixelY/gridVoxelSpacing[1]);
                    basis=(PrecisionTYPE)pixelY/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
                    if(basis<0.0) basis=0.0; //rounding error

                    switch(y-yPre){
                        case 0:
                            yBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                            yFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                            break;
                        case 1:
                            yBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                            yFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                            break;
                        case 2:
                            yBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                            yFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                            break;
                        case 3:
                            yBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                            yFirst=(PrecisionTYPE)(basis*basis/2.0);
                            break;
                        default:
                            yBasis=(PrecisionTYPE)0.0;
                            yFirst=(PrecisionTYPE)0.0;
                            break;
                    }

                    for(int pixelX=(int)ceil((x-2)*gridVoxelSpacing[0]);pixelX<(int)floor((x)*gridVoxelSpacing[0]); pixelX++){
                        if(pixelX>-1 && pixelX<targetImage->nx){

                            int xPre=(int)((PrecisionTYPE)pixelX/gridVoxelSpacing[0]);
                            basis=(PrecisionTYPE)pixelX/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                            if(basis<0.0) basis=0.0; //rounding error

                            switch(x-xPre){
                                case 0:
                                    xBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                    xFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                    break;
                                case 1:
                                    xBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                    xFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                    break;
                                case 2:
                                    xBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                    xFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                    break;
                                case 3:
                                    xBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                    xFirst=(PrecisionTYPE)(basis*basis/2.0);
                                    break;
                                default:
                                    xBasis=(PrecisionTYPE)0.0;
                                    xFirst=(PrecisionTYPE)0.0;
                                    break;
                            }

                            basisValues[0]= xFirst * yBasis;
                            basisValues[1]= xBasis * yFirst;

                            jacIndex = pixelY*targetImage->nx+pixelX;
                            PrecisionTYPE logDet=jacobianDeterminant[jacIndex];
                            jacobianMatrix = invertedJacobianMatrices[jacIndex];

                            if(logDet<=0.0){
                                /* derivative of the Jacobian determinant itself */
                                correctFolding=true;
                                foldingCorrectionX += logDet * (jacobianMatrix.m[0][0]*basisValues[0] + jacobianMatrix.m[0][1]*basisValues[1]);
                                foldingCorrectionY += logDet * (jacobianMatrix.m[1][0]*basisValues[0] + jacobianMatrix.m[1][1]*basisValues[1]);
                            }
                        }// if x
                    }// x
                }// if y
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            if(correctFolding){
                PrecisionTYPE gradient[2];
                gradient[0] = desorient.m[0][0]*foldingCorrectionX +
                    desorient.m[0][1]*foldingCorrectionY;
                gradient[1] = desorient.m[1][0]*foldingCorrectionX +
                    desorient.m[1][1]*foldingCorrectionY;
                PrecisionTYPE norm = 5.0 * sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
                if(norm>0.0){
                    const unsigned int id = y*splineControlPoint->nx+x;
                    controlPointPtrX[id] += (SplineTYPE)(splineControlPoint->dx*gradient[0]/norm);
                    controlPointPtrY[id] += (SplineTYPE)(splineControlPoint->dy*gradient[1]/norm);
                }
            }

        }
    }
    free(jacobianDeterminant);
    free(invertedJacobianMatrices);
    return std::numeric_limits<float>::quiet_NaN();

}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_correctFoldingApprox_2D(nifti_image *splineControlPoint,
                                                  nifti_image *targetImage)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(jacobianNumber * sizeof(PrecisionTYPE));

    computeApproximateJacobianMatrices_2D<PrecisionTYPE, SplineTYPE>(   splineControlPoint,
                                                                        invertedJacobianMatrices,
                                                                        jacobianDeterminant);

    // The current Penalty term value is computed
    unsigned int jacIndex;
    double penaltyTerm =0.0;
    for(int j=1; j< splineControlPoint->ny-1; j++){
        jacIndex = j*splineControlPoint->nx+1;
        for(int i=1; i< splineControlPoint->nx-1; i++){
            double logDet = log(jacobianDeterminant[jacIndex++]);
            penaltyTerm += logDet*logDet;
        }
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(invertedJacobianMatrices);
        return (PrecisionTYPE)(penaltyTerm/(PrecisionTYPE)jacobianNumber);
    }

    PrecisionTYPE basisValues[2];
    PrecisionTYPE xBasis, yBasis;
    PrecisionTYPE xFirst, yFirst;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;

    // The gradient are now computed for every control point
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            PrecisionTYPE foldingCorrectionX=(PrecisionTYPE)0.0;
            PrecisionTYPE foldingCorrectionY=(PrecisionTYPE)0.0;

            bool correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(y-1);pixelY<(y+2); pixelY++){
                if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                    switch(pixelY-y){
                        case -1:
                            yBasis=(PrecisionTYPE)(1.0/6.0);
                            yFirst=(PrecisionTYPE)(0.5);
                            break;
                        case 0:
                            yBasis=(PrecisionTYPE)(2.0/3.0);
                            yFirst=(PrecisionTYPE)(0.0);
                            break;
                        case 1:
                            yBasis=(PrecisionTYPE)(1.0/6.0);
                            yFirst=(PrecisionTYPE)(-0.5);
                            break;
                        default:
                            yBasis=(PrecisionTYPE)0.0;
                            yFirst=(PrecisionTYPE)0.0;
                            break;
                    }
                    for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                            switch(pixelX-x){
                                case -1:
                                    xBasis=(PrecisionTYPE)(1.0/6.0);
                                    xFirst=(PrecisionTYPE)(0.5);
                                    break;
                                case 0:
                                    xBasis=(PrecisionTYPE)(2.0/3.0);
                                    xFirst=(PrecisionTYPE)(0.0);
                                    break;
                                case 1:
                                    xBasis=(PrecisionTYPE)(1.0/6.0);
                                    xFirst=(PrecisionTYPE)(-0.5);
                                    break;
                                default:
                                    xBasis=(PrecisionTYPE)0.0;
                                    xFirst=(PrecisionTYPE)0.0;
                                    break;
                            }

                            basisValues[0] = xFirst * yBasis ;
                            basisValues[1] = xBasis * yFirst ;

                            jacIndex = pixelY*splineControlPoint->nx+pixelX;
                            PrecisionTYPE logDet=jacobianDeterminant[jacIndex];
                            jacobianMatrix = invertedJacobianMatrices[jacIndex];

                            if(logDet<=0.0){
                                /* derivative of the Jacobian determinant itself */
                                correctFolding=true;
                                foldingCorrectionX += logDet * (jacobianMatrix.m[0][0]*basisValues[0] + jacobianMatrix.m[0][1]*basisValues[1]);
                                foldingCorrectionY += logDet * (jacobianMatrix.m[1][0]*basisValues[0] + jacobianMatrix.m[1][1]*basisValues[1]);
                            }
                        }// if x
                    }// x
                }// if y
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            if(correctFolding){
                PrecisionTYPE gradient[2];
                gradient[0] = desorient.m[0][0]*foldingCorrectionX
                            + desorient.m[0][1]*foldingCorrectionY;
                gradient[1] = desorient.m[1][0]*foldingCorrectionX
                            + desorient.m[1][1]*foldingCorrectionY;
                PrecisionTYPE norm = 5.0 * sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
                if(norm>0.0){
                    const unsigned int id = y*splineControlPoint->nx+x;
                    controlPointPtrX[id] += splineControlPoint->dx*gradient[0]/norm;
                    controlPointPtrY[id] += splineControlPoint->dy*gradient[1]/norm;
                }
            }

        }
    }
    free(jacobianDeterminant);
    free(invertedJacobianMatrices);
    return std::numeric_limits<float>::quiet_NaN();
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_correctFolding_3D(nifti_image *splineControlPoint,
                                            nifti_image *targetImage)
{

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(targetImage->nvox * sizeof(PrecisionTYPE));

    computeJacobianMatrices_3D<PrecisionTYPE, SplineTYPE>(  targetImage,
                                                            splineControlPoint,
                                                            invertedJacobianMatrices,
                                                            jacobianDeterminant);

    /* The current Penalty term value is computed */
    double penaltyTerm =0.0;
    for(unsigned int i=0; i< targetImage->nvox; i++){
        double logDet = log(jacobianDeterminant[i]);
        penaltyTerm += logDet*logDet;
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(invertedJacobianMatrices);
        return (PrecisionTYPE)(penaltyTerm/(double)targetImage->nvox);
    }

    /*  */
    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    PrecisionTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    PrecisionTYPE basisValues[3];
    PrecisionTYPE xBasis, yBasis, zBasis, basis;
    PrecisionTYPE xFirst, yFirst, zFirst;
    unsigned int jacIndex;

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                PrecisionTYPE foldingCorrectionX=(PrecisionTYPE)0.0;
                PrecisionTYPE foldingCorrectionY=(PrecisionTYPE)0.0;
                PrecisionTYPE foldingCorrectionZ=(PrecisionTYPE)0.0;

                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)ceil((z-2)*gridVoxelSpacing[2]);pixelZ<(int)floor((z)*gridVoxelSpacing[2]); pixelZ++){
                    if(pixelZ>-1 && pixelZ<targetImage->nz){

                        int zPre=(int)((PrecisionTYPE)pixelZ/gridVoxelSpacing[2]);
                        basis=(PrecisionTYPE)pixelZ/gridVoxelSpacing[2]-(PrecisionTYPE)zPre;
                        if(basis<0.0) basis=0.0; //rounding error

                        switch(z-zPre){
                            case 0:
                                zBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                zFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                break;
                            case 1:
                                zBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                zFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                break;
                            case 2:
                                zBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                zFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                break;
                            case 3:
                                zBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                zFirst=(PrecisionTYPE)(basis*basis/2.0);
                                break;
                            default:
                                zBasis=(PrecisionTYPE)0.0;
                                zFirst=(PrecisionTYPE)0.0;
                                break;
                        }


                        for(int pixelY=(int)ceil((y-2)*gridVoxelSpacing[1]);pixelY<(int)floor((y)*gridVoxelSpacing[1]); pixelY++){
                            if(pixelY>-1 && pixelY<targetImage->ny){

                                int yPre=(int)((PrecisionTYPE)pixelY/gridVoxelSpacing[1]);
                                basis=(PrecisionTYPE)pixelY/gridVoxelSpacing[1]-(PrecisionTYPE)yPre;
                                if(basis<0.0) basis=0.0; //rounding error

                                switch(y-yPre){
                                    case 0:
                                        yBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                        yFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                        break;
                                    case 1:
                                        yBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                        yFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                        break;
                                    case 2:
                                        yBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                        yFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                        break;
                                    case 3:
                                        yBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                        yFirst=(PrecisionTYPE)(basis*basis/2.0);
                                        break;
                                    default:
                                        yBasis=(PrecisionTYPE)0.0;
                                        yFirst=(PrecisionTYPE)0.0;
                                        break;
                                }

                                for(int pixelX=(int)ceil((x-2)*gridVoxelSpacing[0]);pixelX<(int)floor((x)*gridVoxelSpacing[0]); pixelX++){
                                    if(pixelX>-1 && pixelX<targetImage->nx){

                                        int xPre=(int)((PrecisionTYPE)pixelX/gridVoxelSpacing[0]);
                                        basis=(PrecisionTYPE)pixelX/gridVoxelSpacing[0]-(PrecisionTYPE)xPre;
                                        if(basis<0.0) basis=0.0; //rounding error

                                        switch(x-xPre){
                                            case 0:
                                                xBasis=(PrecisionTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                                xFirst=(PrecisionTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                                break;
                                            case 1:
                                                xBasis=(PrecisionTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                                xFirst=(PrecisionTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                                break;
                                            case 2:
                                                xBasis=(PrecisionTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                                xFirst=(PrecisionTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                                break;
                                            case 3:
                                                xBasis=(PrecisionTYPE)(basis*basis*basis/6.0);
                                                xFirst=(PrecisionTYPE)(basis*basis/2.0);
                                                break;
                                            default:
                                                xBasis=(PrecisionTYPE)0.0;
                                                xFirst=(PrecisionTYPE)0.0;
                                                break;
                                        }

                                        basisValues[0]= xFirst * yBasis * zBasis ;
                                        basisValues[1]= xBasis * yFirst * zBasis ;
                                        basisValues[2]= xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+pixelX;
                                        PrecisionTYPE detJac = jacobianDeterminant[jacIndex];

                                        mat33 jacobianMatrix = invertedJacobianMatrices[jacIndex];

                                        if(detJac<=0.0){
                                            correctFolding=true;
                                            // Derivative of the jacobian itself
                                            foldingCorrectionX += detJac *
                                                ( jacobianMatrix.m[0][0]*basisValues[0]
                                                + jacobianMatrix.m[0][1]*basisValues[1]
                                                + jacobianMatrix.m[0][2]*basisValues[2]);
                                            foldingCorrectionY += detJac *
                                                ( jacobianMatrix.m[1][0]*basisValues[0]
                                                + jacobianMatrix.m[1][1]*basisValues[1]
                                                + jacobianMatrix.m[1][2]*basisValues[2]);
                                            foldingCorrectionZ += detJac *
                                                ( jacobianMatrix.m[2][0]*basisValues[0]
                                                + jacobianMatrix.m[2][1]*basisValues[1]
                                                + jacobianMatrix.m[2][2]*basisValues[2]);
                                        } // detJac<0.0
                                    } // if x
                                }// x
                            }// if y
                        }// y
                    }// if z
                } // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                if(correctFolding){
                    PrecisionTYPE gradient[3];
                    gradient[0] = desorient.m[0][0]*foldingCorrectionX
                                + desorient.m[0][1]*foldingCorrectionY
                                + desorient.m[0][2]*foldingCorrectionZ;
                    gradient[1] = desorient.m[1][0]*foldingCorrectionX
                                + desorient.m[1][1]*foldingCorrectionY
                                + desorient.m[1][2]*foldingCorrectionZ;
                    gradient[2] = desorient.m[2][0]*foldingCorrectionX
                                + desorient.m[2][1]*foldingCorrectionY
                                + desorient.m[2][2]*foldingCorrectionZ;
                    PrecisionTYPE norm = (PrecisionTYPE)(5.0 * sqrt(gradient[0]*gradient[0]
                                            + gradient[1]*gradient[1]
                                            + gradient[2]*gradient[2]));

                    if(norm>0.0){
                        const unsigned int id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        controlPointPtrX[id] += (SplineTYPE)(splineControlPoint->dx*gradient[0]/norm);
                        controlPointPtrY[id] += (SplineTYPE)(splineControlPoint->dy*gradient[1]/norm);
                        controlPointPtrZ[id] += (SplineTYPE)(splineControlPoint->dz*gradient[2]/norm);
                    }
                }
            }
        }
    }
    free(jacobianDeterminant);
    free(invertedJacobianMatrices);
    return std::numeric_limits<float>::quiet_NaN();
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_correctFoldingApprox_3D(  nifti_image *splineControlPoint,
                                                    nifti_image *targetImage)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    PrecisionTYPE *jacobianDeterminant=(PrecisionTYPE *)malloc(jacobianNumber * sizeof(PrecisionTYPE));

    computeApproximateJacobianMatrices_3D<PrecisionTYPE, SplineTYPE>(   splineControlPoint,
                                                                        invertedJacobianMatrices,
                                                                        jacobianDeterminant);

    // The current Penalty term value is computed
    unsigned int jacIndex;
    double penaltyTerm =0.0;
    for(int k=1; k<splineControlPoint->nz-1; k++){
        for(int j=1; j<splineControlPoint->ny-1; j++){
            jacIndex = (k*splineControlPoint->ny+j)*splineControlPoint->nx+1;
            for(int i=1; i<splineControlPoint->nx-1; i++){
                double logDet = log(jacobianDeterminant[jacIndex++]);
                penaltyTerm += logDet*logDet;
            }
        }
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(invertedJacobianMatrices);
        return (PrecisionTYPE)(penaltyTerm/(PrecisionTYPE)jacobianNumber);
    }

    PrecisionTYPE basisValues[3];
    PrecisionTYPE xBasis, yBasis, zBasis;
    PrecisionTYPE xFirst, yFirst, zFirst;

    mat33 reorient;
    reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    mat33 desorient=nifti_mat33_mul(spline_ijk, reorient);
    mat33 jacobianMatrix;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                PrecisionTYPE foldingCorrectionX=(PrecisionTYPE)0.0;
                PrecisionTYPE foldingCorrectionY=(PrecisionTYPE)0.0;
                PrecisionTYPE foldingCorrectionZ=(PrecisionTYPE)0.0;

                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)((z-1));pixelZ<(int)((z+2)); pixelZ++){
                    if(pixelZ>0 && pixelZ<splineControlPoint->nz-1){

                        switch(pixelZ-z){
                            case -1:
                                zBasis=(PrecisionTYPE)(1.0/6.0);
                                zFirst=(PrecisionTYPE)(0.5);
                                break;
                            case 0:
                                zBasis=(PrecisionTYPE)(2.0/3.0);
                                zFirst=(PrecisionTYPE)(0.0);
                                break;
                            case 1:
                                zBasis=(PrecisionTYPE)(1.0/6.0);
                                zFirst=(PrecisionTYPE)(-0.5);
                                break;
                            default:
                                zBasis=(PrecisionTYPE)0.0;
                                zFirst=(PrecisionTYPE)0.0;
                                break;
                        }
                        for(int pixelY=(int)((y-1));pixelY<(int)((y+2)); pixelY++){
                            if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                                switch(pixelY-y){
                                    case -1:
                                        yBasis=(PrecisionTYPE)(1.0/6.0);
                                        yFirst=(PrecisionTYPE)(0.5);
                                        break;
                                    case 0:
                                        yBasis=(PrecisionTYPE)(2.0/3.0);
                                        yFirst=(PrecisionTYPE)(0.0);
                                        break;
                                    case 1:
                                        yBasis=(PrecisionTYPE)(1.0/6.0);
                                        yFirst=(PrecisionTYPE)(-0.5);
                                        break;
                                    default:
                                        yBasis=(PrecisionTYPE)0.0;
                                        yFirst=(PrecisionTYPE)0.0;
                                        break;
                                }
                                for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                                    if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                                        switch(pixelX-x){
                                            case -1:
                                                xBasis=(PrecisionTYPE)(1.0/6.0);
                                                xFirst=(PrecisionTYPE)(0.5);
                                                break;
                                            case 0:
                                                xBasis=(PrecisionTYPE)(2.0/3.0);
                                                xFirst=(PrecisionTYPE)(0.0);
                                                break;
                                            case 1:
                                                xBasis=(PrecisionTYPE)(1.0/6.0);
                                                xFirst=(PrecisionTYPE)(-0.5);
                                                break;
                                            default:
                                                xBasis=(PrecisionTYPE)0.0;
                                                xFirst=(PrecisionTYPE)0.0;
                                                break;
                                        }

                                        basisValues[0] = xFirst * yBasis * zBasis ;
                                        basisValues[1] = xBasis * yFirst * zBasis ;
                                        basisValues[2] = xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*splineControlPoint->ny+pixelY)*splineControlPoint->nx+pixelX;
                                        jacobianMatrix = invertedJacobianMatrices[jacIndex];
                                        PrecisionTYPE detJac = jacobianDeterminant[jacIndex];

                                        if(detJac<=0.0){
                                            correctFolding=true;
                                            // Derivative of the jacobian itself
                                            foldingCorrectionX += detJac *
                                                ( jacobianMatrix.m[0][0]*basisValues[0]
                                                + jacobianMatrix.m[0][1]*basisValues[1]
                                                + jacobianMatrix.m[0][2]*basisValues[2]);
                                            foldingCorrectionY += detJac *
                                                ( jacobianMatrix.m[1][0]*basisValues[0]
                                                + jacobianMatrix.m[1][1]*basisValues[1]
                                                + jacobianMatrix.m[1][2]*basisValues[2]);
                                            foldingCorrectionZ += detJac *
                                                ( jacobianMatrix.m[2][0]*basisValues[0]
                                                + jacobianMatrix.m[2][1]*basisValues[1]
                                                + jacobianMatrix.m[2][2]*basisValues[2]);
                                        } // detJac<0.0
                                    } // if x
                                }// x
                            }// if y
                        }// y
                    }// if z
                } // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                if(correctFolding){
                    PrecisionTYPE gradient[3];
                    gradient[0] = desorient.m[0][0]*foldingCorrectionX
                                + desorient.m[0][1]*foldingCorrectionY
                                + desorient.m[0][2]*foldingCorrectionZ;
                    gradient[1] = desorient.m[1][0]*foldingCorrectionX
                                + desorient.m[1][1]*foldingCorrectionY
                                + desorient.m[1][2]*foldingCorrectionZ;
                    gradient[2] = desorient.m[2][0]*foldingCorrectionX
                                + desorient.m[2][1]*foldingCorrectionY
                                + desorient.m[2][2]*foldingCorrectionZ;
                    PrecisionTYPE norm = (PrecisionTYPE)(5.0 * sqrt(gradient[0]*gradient[0]
                                            + gradient[1]*gradient[1]
                                            + gradient[2]*gradient[2]));

                    if(norm>(PrecisionTYPE)0.0){
                        const unsigned int id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        controlPointPtrX[id] += (SplineTYPE)(splineControlPoint->dx*gradient[0]/norm);
                        controlPointPtrY[id] += (SplineTYPE)(splineControlPoint->dy*gradient[1]/norm);
                        controlPointPtrZ[id] += (SplineTYPE)(splineControlPoint->dz*gradient[2]/norm);
                    }
                }
            }
        }
    }
    free(jacobianDeterminant);
    free(invertedJacobianMatrices);
    return std::numeric_limits<float>::quiet_NaN();
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_bspline_correctFolding(	nifti_image *splineControlPoint,
                                            nifti_image *targetImage,
                                            bool approx)
{

    if(splineControlPoint->nz==1){
        if(approx){
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFoldingApprox_2D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage);
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFoldingApprox_2D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage);
                default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The bending energy gradient has not computed\n");
                    exit(1);
            }
        }
        else{
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFolding_2D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFolding_2D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage);
                    break;
                default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The bending energy gradient has not computed\n");
                    exit(1);
            }
        }
    }
    else{
        if(approx){
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFoldingApprox_3D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFoldingApprox_3D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage);
                    break;
                default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The bending energy gradient has not computed\n");
                    exit(1);
                    break;
            }
        }
        else{
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFolding_3D<PrecisionTYPE, float>
                        (splineControlPoint, targetImage);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFolding_3D<PrecisionTYPE, double>
                        (splineControlPoint, targetImage);
                    break;
                default:
                    fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"The bending energy gradient has not computed\n");
                    exit(1);
                    break;
            }
        }
    }
}
template float reg_bspline_correctFolding<float>(nifti_image *, nifti_image *, bool);
template double reg_bspline_correctFolding<double>(nifti_image *, nifti_image *, bool);
/* *************************************************************** */
/* *************************************************************** */
template<class SplineTYPE>
SplineTYPE GetValue(SplineTYPE *array, int *dim, int x, int y, int z)
{
	if(x<0 || x>= dim[1] || y<0 || y>= dim[2] || z<0 || z>= dim[3])
		return 0.0;
	return array[(z*dim[2]+y)*dim[1]+x];
}
/* *************************************************************** */
template<class SplineTYPE>
void SetValue(SplineTYPE *array, int *dim, int x, int y, int z, SplineTYPE value)
{
	if(x<0 || x>= dim[1] || y<0 || y>= dim[2] || z<0 || z>= dim[3])
		return;
	array[(z*dim[2]+y)*dim[1]+x] = value;
}
/* *************************************************************** */
extern "C++" template<class SplineTYPE>
void reg_bspline_refineControlPointGrid2D(  nifti_image *targetImage,
                                            nifti_image *splineControlPoint)
{
    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
    if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[1]=splineControlPoint->dim[1];
    oldDim[2]=splineControlPoint->dim[2];
    oldDim[3]=1;

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = 1.0f;

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+5;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+5;
    splineControlPoint->dim[3]=1;

    splineControlPoint->nvox=splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz*splineControlPoint->nt*splineControlPoint->nu;
    splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);

    gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]];

    for(int y=0; y<oldDim[2]; y++){
        int Y=2*y-1;
        if(Y<splineControlPoint->ny){
            for(int x=0; x<oldDim[1]; x++){
                int X=2*x-1;
                if(X<splineControlPoint->nx){

		/* X Axis */
			// 0 0
			SetValue(gridPtrX, splineControlPoint->dim, X, Y, 0,
			(GetValue(oldGridPtrX,oldDim,x-1,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
			GetValue(oldGridPtrX,oldDim,x-1,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
			+ 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
			GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) )
			+ 36.0f * GetValue(oldGridPtrX,oldDim,x,y,0) ) / 64.0f);
            // 1 0
			SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, 0,
			(GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
			GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
			+ 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) ) ) / 16.0f);
            // 0 1
			SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, 0,
			(GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x-1,y+1,0) +
			GetValue(oldGridPtrX,oldDim,x+1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
			+ 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) ) ) / 16.0f);
            // 1 1
			SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, 0,
			(GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
			GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0) ) / 4.0f);

		/* Y Axis */
			// 0 0
			SetValue(gridPtrY, splineControlPoint->dim, X, Y, 0,
			(GetValue(oldGridPtrY,oldDim,x-1,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
			GetValue(oldGridPtrY,oldDim,x-1,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
			+ 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
			GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) )
			+ 36.0f * GetValue(oldGridPtrY,oldDim,x,y,0) ) / 64.0f);
            // 1 0
			SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, 0,
			(GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
			GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
			+ 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) ) ) / 16.0f);
            // 0 1
			SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, 0,
			(GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x-1,y+1,0) +
			GetValue(oldGridPtrY,oldDim,x+1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
			+ 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) ) ) / 16.0f);
            // 1 1
			SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, 0,
			(GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
			GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0) ) / 4.0f);

                }
            }
        }
    }

    free(oldGrid);
}
/* *************************************************************** */
extern "C++" template<class SplineTYPE>
void reg_bspline_refineControlPointGrid3D(nifti_image *targetImage,
                    nifti_image *splineControlPoint)
{

    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
    if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[1]=splineControlPoint->dim[1];
    oldDim[2]=splineControlPoint->dim[2];
    oldDim[3]=splineControlPoint->dim[3];

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = splineControlPoint->pixdim[3] = splineControlPoint->dz / 2.0f;

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+5;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+5;
    splineControlPoint->dim[3]=splineControlPoint->nz=(int)floor(targetImage->nz*targetImage->dz/splineControlPoint->dz)+5;

    splineControlPoint->nvox=splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz*splineControlPoint->nt*splineControlPoint->nu;
    splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);
    
    gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *gridPtrZ = &gridPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]*oldDim[3]];
    SplineTYPE *oldGridPtrZ = &oldGridPtrY[oldDim[1]*oldDim[2]*oldDim[3]];


    for(int z=0; z<oldDim[3]; z++){
        int Z=2*z-1;
        if(Z<splineControlPoint->nz){
            for(int y=0; y<oldDim[2]; y++){
                int Y=2*y-1;
                if(Y<splineControlPoint->ny){
                    for(int x=0; x<oldDim[1]; x++){
                        int X=2*x-1;
                        if(X<splineControlPoint->nx){
            
                            /* X Axis */
                            // 0 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z,
                                (GetValue(oldGridPtrX,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) +
                                GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                                GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1)+
                                GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)
                                + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) )
                                + 36.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) )
                                + 216.0f * GetValue(oldGridPtrX,oldDim,x,y,z) ) / 512.0f);
            
                            // 1 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z,
                                ( GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1)) +
                                36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z)) ) / 128.0f);
            
                            // 0 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z,
                                ( GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x-1,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z)) ) / 128.0f);
            
                            // 1 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z,
                                (GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z-1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                                GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) ) ) / 32.0f);
            
                            // 0 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z+1,
                                ( GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y,z+1)) ) / 128.0f);
            
                            // 1 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z+1,
                                (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) ) ) / 32.0f);
            
                            // 0 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z+1,
                                (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x+1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) ) ) / 32.0f);
            
                            // 1 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z+1,
                                (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)) / 8.0f);
                            
            
                            /* Y Axis */
                            // 0 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z,
                                (GetValue(oldGridPtrY,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) +
                                GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                                GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1)+
                                GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)
                                + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) )
                                + 36.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) )
                                + 216.0f * GetValue(oldGridPtrY,oldDim,x,y,z) ) / 512.0f);
            
                            // 1 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z,
                                ( GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1)) +
                                36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z)) ) / 128.0f);
            
                            // 0 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z,
                                ( GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x-1,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z)) ) / 128.0f);
            
                            // 1 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z,
                                (GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z-1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                                GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) ) ) / 32.0f);
            
                            // 0 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z+1,
                                ( GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y,z+1)) ) / 128.0f);
            
                            // 1 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z+1,
                                (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) ) ) / 32.0f);
            
                            // 0 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z+1,
                                (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x+1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) ) ) / 32.0f);
            
                            // 1 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z+1,
                                (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)) / 8.0f);
            
                            /* Z Axis */
                            // 0 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z,
                                (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1)+
                                GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)
                                + 6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) )
                                + 36.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) )
                                + 216.0f * GetValue(oldGridPtrZ,oldDim,x,y,z) ) / 512.0f);
                            
                            // 1 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z,
                                ( GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1)) +
                                36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z)) ) / 128.0f);
                            
                            // 0 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z,
                                ( GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z)) ) / 128.0f);
                            
                            // 1 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z,
                                (GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) ) ) / 32.0f);
                            
                            // 0 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z+1,
                                ( GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y,z+1)) ) / 128.0f);
                            
                            // 1 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z+1,
                                (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) ) ) / 32.0f);
                            
                            // 0 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z+1,
                                (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) ) ) / 32.0f);
                            
                            // 1 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z+1,
                                (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)) / 8.0f);
                        }
                    }
                }
            }
        }
    }

    free(oldGrid);
}
/* *************************************************************** */
extern "C++"
void reg_bspline_refineControlPointGrid(	nifti_image *targetImage,
					nifti_image *splineControlPoint)
{
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_refineControlPointGrid2D<float>(targetImage,splineControlPoint);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_refineControlPointGrid2D<double>(targetImage,splineControlPoint);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"The bending energy gradient has not computed\n");
        }
    }else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_refineControlPointGrid3D<float>(targetImage,splineControlPoint);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_refineControlPointGrid3D<double>(targetImage,splineControlPoint);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"The bending energy gradient has not computed\n");
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class SplineTYPE, class JacobianTYPE>
void reg_bspline_GetJacobianMap2D(nifti_image *splineControlPoint,
                nifti_image *jacobianImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    JacobianTYPE *jacobianMapPtr = static_cast<JacobianTYPE *>(jacobianImage->data);

    JacobianTYPE yBasis[4],yFirst[4],temp[4],first[4];
    JacobianTYPE basisX[16], basisY[16];
    JacobianTYPE basis, FF, FFF, MF, oldBasis=(JacobianTYPE)(1.1);

    JacobianTYPE xControlPointCoordinates[16];
    JacobianTYPE yControlPointCoordinates[16];

    JacobianTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    unsigned int coord=0;
	
	/* In case the matrix is not diagonal, the jacobian has to be reoriented */
	mat33 reorient;
	reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
	reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
	reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
	mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
		spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
	}
    else{
		spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
	}
	reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
	
	mat33 jacobianMatrix;
	
    for(int y=0; y<jacobianImage->ny; y++){

        int yPre=(int)((JacobianTYPE)y/gridVoxelSpacing[1]);
        basis=(JacobianTYPE)y/gridVoxelSpacing[1]-(JacobianTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(JacobianTYPE)(1.0-basis);
        yBasis[0] = (JacobianTYPE)((MF)*(MF)*(MF)/6.0);
        yBasis[1] = (JacobianTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        yBasis[2] = (JacobianTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        yBasis[3] = (JacobianTYPE)(FFF/6.0);
        yFirst[3] = (JacobianTYPE)(FF / 2.0);
        yFirst[0]= (JacobianTYPE)(basis - 1.0/2.0 - yFirst[3]);
        yFirst[2]= (JacobianTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
        yFirst[1]= (JacobianTYPE)(- yFirst[0] - yFirst[2] - yFirst[3]);

        for(int x=0; x<jacobianImage->nx; x++){

            int xPre=(int)((JacobianTYPE)x/gridVoxelSpacing[0]);
            basis=(JacobianTYPE)x/gridVoxelSpacing[0]-(JacobianTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(JacobianTYPE)(1.0-basis);
            temp[0] = (JacobianTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (JacobianTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (JacobianTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (JacobianTYPE)(FFF/6.0);
            first[3]= (JacobianTYPE)(FF / 2.0);
            first[0]= (JacobianTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (JacobianTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= (JacobianTYPE)(- first[0] - first[2] - first[3]);

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*first[a];   // y * x'
                    basisY[coord]=yFirst[b]*temp[a];    // y'* x
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                coord=0;
                for(int Y=yPre; Y<yPre+4; Y++){
                    int index = Y*splineControlPoint->nx;
                    SplineTYPE *xPtr = &controlPointPtrX[index];
                    SplineTYPE *yPtr = &controlPointPtrY[index];
                    for(int X=xPre; X<xPre+4; X++){
                        xControlPointCoordinates[coord] = (JacobianTYPE)xPtr[X];
                        yControlPointCoordinates[coord] = (JacobianTYPE)yPtr[X];
                        coord++;
                    }
                }
            }
            oldBasis=basis;
            JacobianTYPE Tx_x=0.0;
            JacobianTYPE Ty_x=0.0;
            JacobianTYPE Tx_y=0.0;
            JacobianTYPE Ty_y=0.0;

            for(int a=0; a<16; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Ty_x += basisX[a]*yControlPointCoordinates[a];

                Tx_y += basisY[a]*xControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }
			
			memset(&jacobianMatrix, 0, sizeof(mat33));
			jacobianMatrix.m[2][2]=1.0f;
			jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
            jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
            jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
            jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
			
            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
			
			*jacobianMapPtr++ = nifti_mat33_determ(jacobianMatrix);
        }
    }
}
/* *************************************************************** */
template <class SplineTYPE, class JacobianTYPE>
void reg_bspline_GetJacobianMap3D(nifti_image *splineControlPoint,
                nifti_image *jacobianImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    JacobianTYPE *jacobianMapPtr = static_cast<JacobianTYPE *>(jacobianImage->data);

    JacobianTYPE zBasis[4],zFirst[4],temp[4],first[4];
    JacobianTYPE tempX[16], tempY[16], tempZ[16];
    JacobianTYPE basisX[64], basisY[64], basisZ[64];
    JacobianTYPE basis, FF, FFF, MF, oldBasis=(JacobianTYPE)(1.1);

    JacobianTYPE xControlPointCoordinates[64];
    JacobianTYPE yControlPointCoordinates[64];
    JacobianTYPE zControlPointCoordinates[64];

    JacobianTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;
    unsigned int coord=0;
	
	mat33 reorient;
	reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
	reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
	reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
	mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
		spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
	}
    else{
		spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
	}
	reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
	mat33 jacobianMatrix;

    for(int z=0; z<jacobianImage->nz; z++){

        int zPre=(int)((JacobianTYPE)z/gridVoxelSpacing[2]);
        basis=(JacobianTYPE)z/gridVoxelSpacing[2]-(JacobianTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(JacobianTYPE)(1.0-basis);
        zBasis[0] = (JacobianTYPE)((MF)*(MF)*(MF)/6.0);
        zBasis[1] = (JacobianTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        zBasis[2] = (JacobianTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        zBasis[3] = (JacobianTYPE)(FFF/6.0);
        zFirst[3] = (JacobianTYPE)(FF / 2.0);
        zFirst[0]= (JacobianTYPE)(basis - 1.0/2.0 - zFirst[3]);
        zFirst[2]= (JacobianTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
        zFirst[1]= (JacobianTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

        for(int y=0; y<jacobianImage->ny; y++){
            
            int yPre=(int)((JacobianTYPE)y/gridVoxelSpacing[1]);
            basis=(JacobianTYPE)y/gridVoxelSpacing[1]-(JacobianTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(JacobianTYPE)(1.0-basis);
            temp[0] = (JacobianTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (JacobianTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (JacobianTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (JacobianTYPE)(FFF/6.0);
            first[3]= (JacobianTYPE)(FF / 2.0);
            first[0]= (JacobianTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (JacobianTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= (JacobianTYPE)(- first[0] - first[2] - first[3]);
            
            coord=0;
            for(int c=0; c<4; c++){
                for(int b=0; b<4; b++){
                    tempX[coord]=zBasis[c]*temp[b]; // z * y
                    tempY[coord]=zBasis[c]*first[b];// z * y'
                    tempZ[coord]=zFirst[c]*temp[b]; // z'* y
                    coord++;
                }
            }
            
            for(int x=0; x<jacobianImage->nx; x++){
                
                int xPre=(int)((JacobianTYPE)x/gridVoxelSpacing[0]);
                basis=(JacobianTYPE)x/gridVoxelSpacing[0]-(JacobianTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(JacobianTYPE)(1.0-basis);
                temp[0] = (JacobianTYPE)((MF)*(MF)*(MF)/6.0);
                temp[1] = (JacobianTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                temp[2] = (JacobianTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                temp[3] = (JacobianTYPE)(FFF/6.0);
                first[3]= (JacobianTYPE)(FF / 2.0);
                first[0]= (JacobianTYPE)(basis - 1.0/2.0 - first[3]);
                first[2]= (JacobianTYPE)(1.0 + first[0] - 2.0*first[3]);
                first[1]= (JacobianTYPE)(- first[0] - first[2] - first[3]);
                
                coord=0;
                for(int bc=0; bc<16; bc++){
                    for(int a=0; a<4; a++){
                        basisX[coord]=tempX[bc]*first[a];   // z * y * x'
                        basisY[coord]=tempY[bc]*temp[a];    // z * y'* x
                        basisZ[coord]=tempZ[bc]*temp[a];    // z'* y * x
                        coord++;
                    }
                }
                
                if(basis<=oldBasis || x==0){
                    coord=0;
                    for(int Z=zPre; Z<zPre+4; Z++){
                        unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                        SplineTYPE *xPtr = &controlPointPtrX[index];
                        SplineTYPE *yPtr = &controlPointPtrY[index];
                        SplineTYPE *zPtr = &controlPointPtrZ[index];
                        for(int Y=yPre; Y<yPre+4; Y++){
                            index = Y*splineControlPoint->nx;
                            SplineTYPE *xxPtr = &xPtr[index];
                            SplineTYPE *yyPtr = &yPtr[index];
                            SplineTYPE *zzPtr = &zPtr[index];
                            for(int X=xPre; X<xPre+4; X++){
                                xControlPointCoordinates[coord] = (JacobianTYPE)xxPtr[X];
                                yControlPointCoordinates[coord] = (JacobianTYPE)yyPtr[X];
                                zControlPointCoordinates[coord] = (JacobianTYPE)zzPtr[X];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;
                
                JacobianTYPE Tx_x=0.0;
                JacobianTYPE Ty_x=0.0;
                JacobianTYPE Tz_x=0.0;
                JacobianTYPE Tx_y=0.0;
                JacobianTYPE Ty_y=0.0;
                JacobianTYPE Tz_y=0.0;
                JacobianTYPE Tx_z=0.0;
                JacobianTYPE Ty_z=0.0;
                JacobianTYPE Tz_z=0.0;
                
                for(int a=0; a<64; a++){
                    Tx_x += basisX[a]*xControlPointCoordinates[a];
                    Tx_y += basisY[a]*xControlPointCoordinates[a];
                    Tx_z += basisZ[a]*xControlPointCoordinates[a];
                    
                    Ty_x += basisX[a]*yControlPointCoordinates[a];
                    Ty_y += basisY[a]*yControlPointCoordinates[a];
                    Ty_z += basisZ[a]*yControlPointCoordinates[a];
                    
                    Tz_x += basisX[a]*zControlPointCoordinates[a];
                    Tz_y += basisY[a]*zControlPointCoordinates[a];
                    Tz_z += basisZ[a]*zControlPointCoordinates[a];
                }
                
				jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
				jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
				jacobianMatrix.m[0][2]= (float)(Tx_z / splineControlPoint->dz);
				jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
				jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
				jacobianMatrix.m[1][2]= (float)(Ty_z / splineControlPoint->dz);
				jacobianMatrix.m[2][0]= (float)(Tz_x / splineControlPoint->dx);
				jacobianMatrix.m[2][1]= (float)(Tz_y / splineControlPoint->dy);
				jacobianMatrix.m[2][2]= (float)(Tz_z / splineControlPoint->dz);
				
                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
				JacobianTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                *jacobianMapPtr++ = detJac;
            }
        }
    }
}
/* *************************************************************** */
template <class SplineTYPE>
void reg_bspline_GetJacobianMap1(   nifti_image *splineControlPoint,
				                    nifti_image *jacobianImage)
{
    if(splineControlPoint->nz==1){
        switch(jacobianImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_GetJacobianMap2D<SplineTYPE,float>(splineControlPoint, jacobianImage);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_GetJacobianMap2D<SplineTYPE,double>(splineControlPoint, jacobianImage);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the jacobian map image\n");
                fprintf(stderr,"The jacobian map has not computed\n");
        }
    }else{
        switch(jacobianImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_GetJacobianMap3D<SplineTYPE,float>(splineControlPoint, jacobianImage);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_GetJacobianMap3D<SplineTYPE,double>(splineControlPoint, jacobianImage);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the jacobian map image\n");
                fprintf(stderr,"The jacobian map has not computed\n");
        }
    }
}
/* *************************************************************** */
void reg_bspline_GetJacobianMap(	nifti_image *splineControlPoint,
				nifti_image *jacobianImage)
{
	switch(splineControlPoint->datatype){
		case NIFTI_TYPE_FLOAT32:
			return reg_bspline_GetJacobianMap1<float>(splineControlPoint, jacobianImage);
		case NIFTI_TYPE_FLOAT64:
			return reg_bspline_GetJacobianMap1<double>(splineControlPoint, jacobianImage);
		default:
			fprintf(stderr,"Only single or double precision is implemented for the control point image\n");
			fprintf(stderr,"The jacobian map has not computed\n");
	}
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianMatrix2D(nifti_image *splineControlPoint,
                                nifti_image *jacobianImage
                                )
{
    ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
    ImageTYPE *controlPointPtrY = static_cast<ImageTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    ImageTYPE *jacobianMatrixTxxPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    ImageTYPE *jacobianMatrixTyxPtr = &jacobianMatrixTxxPtr[jacobianImage->nx*jacobianImage->ny];

    ImageTYPE *jacobianMatrixTxyPtr = &jacobianMatrixTyxPtr[jacobianImage->nx*jacobianImage->ny];
    ImageTYPE *jacobianMatrixTyyPtr = &jacobianMatrixTxyPtr[jacobianImage->nx*jacobianImage->ny];

    ImageTYPE yBasis[4],yFirst[4],temp[4],first[4];
    ImageTYPE basisX[16], basisY[16];
    ImageTYPE basis, FF, FFF, MF, oldBasis=(ImageTYPE)(1.1);

    ImageTYPE xControlPointCoordinates[16];
    ImageTYPE yControlPointCoordinates[16];

    ImageTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    unsigned int coord=0;

	mat33 reorient;
	reorient.m[0][0]=splineControlPoint->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
	reorient.m[1][0]=0.0f; reorient.m[1][1]=splineControlPoint->dy; reorient.m[1][2]=0.0f;
	reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=splineControlPoint->dz;
	mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
		spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
	}
    else{
		spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
		spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
		spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
		spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
		spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
		spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
		spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
		spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
		spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
	}
	reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
	mat33 jacobianMatrix;
	
    for(int y=0; y<jacobianImage->ny; y++){

        int yPre=(int)((ImageTYPE)y/gridVoxelSpacing[1]);
        basis=(ImageTYPE)y/gridVoxelSpacing[1]-(ImageTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(ImageTYPE)(1.0-basis);
        yBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
        yBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        yBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        yBasis[3] = (ImageTYPE)(FFF/6.0);
        yFirst[3] = (ImageTYPE)(FF / 2.0);
        yFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - yFirst[3]);
        yFirst[2]= (ImageTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
        yFirst[1]= (ImageTYPE)(- yFirst[0] - yFirst[2] - yFirst[3]);

        for(int x=0; x<jacobianImage->nx; x++){

            int xPre=(int)((ImageTYPE)x/gridVoxelSpacing[0]);
            basis=(ImageTYPE)x/gridVoxelSpacing[0]-(ImageTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(ImageTYPE)(1.0-basis);
            temp[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
            temp[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            temp[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            temp[3] = (ImageTYPE)(FFF/6.0);
            first[3]= (ImageTYPE)(FF / 2.0);
            first[0]= (ImageTYPE)(basis - 1.0/2.0 - first[3]);
            first[2]= (ImageTYPE)(1.0 + first[0] - 2.0*first[3]);
            first[1]= (ImageTYPE)(- first[0] - first[2] - first[3]);

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*first[a];   // y * x'
                    basisY[coord]=yFirst[b]*temp[a];    // y'* x
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                coord=0;
                for(int Y=yPre; Y<yPre+4; Y++){
                    int index = Y*splineControlPoint->nx;
                    ImageTYPE *xPtr = &controlPointPtrX[index];
                    ImageTYPE *yPtr = &controlPointPtrY[index];
                    for(int X=xPre; X<xPre+4; X++){
                        xControlPointCoordinates[coord] = (ImageTYPE)xPtr[X];
                        yControlPointCoordinates[coord] = (ImageTYPE)yPtr[X];
                        coord++;
                    }
                }
            }
            oldBasis=basis;
            ImageTYPE Tx_x=0.0;
            ImageTYPE Ty_x=0.0;
            ImageTYPE Tx_y=0.0;
            ImageTYPE Ty_y=0.0;

            for(int a=0; a<16; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];

                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }
			
			memset(&jacobianMatrix, 0, sizeof(mat33));
			jacobianMatrix.m[2][2]=1.0f;
			jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
			jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
			jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
			jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
			
            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);

            *jacobianMatrixTxxPtr++ = jacobianMatrix.m[0][0];
            *jacobianMatrixTyxPtr++ = jacobianMatrix.m[0][1];

            *jacobianMatrixTxyPtr++ = jacobianMatrix.m[1][0];
            *jacobianMatrixTyyPtr++ = jacobianMatrix.m[1][1];
        }
    }
}
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianMatrix3D(   nifti_image *splineControlPoint,
                                        nifti_image *jacobianImage)
{
    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianImage->nvox * sizeof(mat33));
    ImageTYPE *jacobianDeterminant=(ImageTYPE *)malloc(jacobianImage->nvox * sizeof(ImageTYPE));

    computeJacobianMatrices_3D<ImageTYPE, ImageTYPE>(   jacobianImage,
                                                        splineControlPoint,
                                                        invertedJacobianMatrices,
                                                        jacobianDeterminant);
    
    ImageTYPE *jacobianMatrixTxxPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    ImageTYPE *jacobianMatrixTyxPtr = &jacobianMatrixTxxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    ImageTYPE *jacobianMatrixTzxPtr = &jacobianMatrixTyxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    
    ImageTYPE *jacobianMatrixTxyPtr = &jacobianMatrixTzxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    ImageTYPE *jacobianMatrixTyyPtr = &jacobianMatrixTxyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    ImageTYPE *jacobianMatrixTzyPtr = &jacobianMatrixTyyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    
    ImageTYPE *jacobianMatrixTxzPtr = &jacobianMatrixTzyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    ImageTYPE *jacobianMatrixTyzPtr = &jacobianMatrixTxzPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    ImageTYPE *jacobianMatrixTzzPtr = &jacobianMatrixTyzPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    
    for(int i=0; i<jacobianImage->nx*jacobianImage->ny*jacobianImage->nz; i++){
        *jacobianMatrixTxxPtr++ = invertedJacobianMatrices[i].m[0][0];
        *jacobianMatrixTxyPtr++ = invertedJacobianMatrices[i].m[0][1];
        *jacobianMatrixTxzPtr++ = invertedJacobianMatrices[i].m[0][2];

        *jacobianMatrixTyxPtr++ = invertedJacobianMatrices[i].m[1][0];
        *jacobianMatrixTyyPtr++ = invertedJacobianMatrices[i].m[1][1];
        *jacobianMatrixTyzPtr++ = invertedJacobianMatrices[i].m[1][2];

        *jacobianMatrixTzxPtr++ = invertedJacobianMatrices[i].m[2][0];
        *jacobianMatrixTzyPtr++ = invertedJacobianMatrices[i].m[2][1];
        *jacobianMatrixTzzPtr++ = invertedJacobianMatrices[i].m[2][2];
    }
    free(jacobianDeterminant);
    free(invertedJacobianMatrices);

}
/* *************************************************************** */
void reg_bspline_GetJacobianMatrix(nifti_image *splineControlPoint,
                                   nifti_image *jacobianImage)
{
    if(splineControlPoint->datatype != jacobianImage->datatype){
        fprintf(stderr, "Error:\treg_bspline_GetJacobianMatrix\n");
        fprintf(stderr, "Input images were expected to be from the same type\n");
        exit(1);
    }

    if(splineControlPoint->nz>1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_GetJacobianMatrix3D<float>(splineControlPoint, jacobianImage);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_GetJacobianMatrix3D<double>(splineControlPoint, jacobianImage);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the control point image\n");
                fprintf(stderr,"The jacobian matrix image has not been computed\n");
                exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_GetJacobianMatrix2D<float>(splineControlPoint, jacobianImage);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_GetJacobianMatrix2D<double>(splineControlPoint, jacobianImage);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the control point image\n");
                fprintf(stderr,"The jacobian matrix image has not been computed\n");
                exit(1);
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_initialiseControlPointGridWithAffine2D(	mat44 *affineTransformation,
															nifti_image *controlPointImage)
{
    DTYPE *CPPX=static_cast<DTYPE *>(controlPointImage->data);
    DTYPE *CPPY=&CPPX[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];

    mat44 *cppMatrix;
    if(controlPointImage->sform_code>0)
        cppMatrix=&(controlPointImage->sto_xyz);
    else cppMatrix=&(controlPointImage->qto_xyz);

    mat44 voxelToRealDeformed = reg_mat44_mul(affineTransformation, cppMatrix);

    float index[3];
    float position[3];
    index[2]=0;
    for(int y=0; y<controlPointImage->ny; y++){
        index[1]=(float)y;
        for(int x=0; x<controlPointImage->nx; x++){
            index[0]=(float)x;

            reg_mat44_mul(&voxelToRealDeformed, index, position);

            *CPPX++ = position[0];
            *CPPY++ = position[1];
        }
    }
}
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_initialiseControlPointGridWithAffine3D(	mat44 *affineTransformation,
							nifti_image *controlPointImage)
{
    DTYPE *CPPX=static_cast<DTYPE *>(controlPointImage->data);
    DTYPE *CPPY=&CPPX[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];
    DTYPE *CPPZ=&CPPY[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];

    mat44 *cppMatrix;
    if(controlPointImage->sform_code>0)
        cppMatrix=&(controlPointImage->sto_xyz);
    else cppMatrix=&(controlPointImage->qto_xyz);

    mat44 voxelToRealDeformed = reg_mat44_mul(affineTransformation, cppMatrix);

    float index[3];
    float position[3];
    for(int z=0; z<controlPointImage->nz; z++){
        index[2]=(float)z;
        for(int y=0; y<controlPointImage->ny; y++){
            index[1]=(float)y;
            for(int x=0; x<controlPointImage->nx; x++){
                index[0]=(float)x;

                reg_mat44_mul(&voxelToRealDeformed, index, position);

                *CPPX++ = position[0];
                *CPPY++ = position[1];
                *CPPZ++ = position[2];
            }
        }
    }
}
/* *************************************************************** */
int reg_bspline_initialiseControlPointGridWithAffine(   mat44 *affineTransformation,
                                                        nifti_image *controlPointImage)
{
	if(controlPointImage->nz==1){
		switch(controlPointImage->datatype){
			case NIFTI_TYPE_FLOAT32:
				reg_bspline_initialiseControlPointGridWithAffine2D<float>(affineTransformation, controlPointImage);
				break;
			case NIFTI_TYPE_FLOAT64:
				reg_bspline_initialiseControlPointGridWithAffine2D<double>(affineTransformation, controlPointImage);
				break;
			default:
				fprintf(stderr,"ERROR:\treg_bspline_initialiseControlPointGridWithAffine\n");
				fprintf(stderr,"ERROR:\tOnly single or double precision is implemented for the control point image\n");
				return 1;
		}
	}
	else{
		switch(controlPointImage->datatype){
			case NIFTI_TYPE_FLOAT32:
				reg_bspline_initialiseControlPointGridWithAffine3D<float>(affineTransformation, controlPointImage);
				break;
			case NIFTI_TYPE_FLOAT64:
				reg_bspline_initialiseControlPointGridWithAffine3D<double>(affineTransformation, controlPointImage);
				break;
			default:
				fprintf(stderr,"ERROR:\treg_bspline_initialiseControlPointGridWithAffine\n");
				fprintf(stderr,"ERROR:\tOnly single or double precision is implemented for the control point image\n");
				return 1;
		}
	}
	return 0;
}
/* *************************************************************** */
/* *************************************************************** */

/* *************************************************************** */
/* *************************************************************** */

/* *************************************************************** */
/* *************************************************************** */

/* *************************************************************** */
/* *************************************************************** */

#endif
