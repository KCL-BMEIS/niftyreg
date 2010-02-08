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

// No round() function available in windows.
#ifdef _WINDOWS
template<class PrecisionType>
int round(PrecisionType x)
{
   return int(x > 0.0 ? x + 0.5 : x - 0.5);
}
#endif

template<class PrecisionTYPE, class FieldTYPE>
void reg_bspline2D( nifti_image *splineControlPoint,
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
    FieldTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    FieldTYPE *fieldPtrX=static_cast<FieldTYPE *>(positionField->data);
    FieldTYPE *fieldPtrY=&fieldPtrX[targetImage->nvox];

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
				PrecisionTYPE xReal = *fieldPtrX;
				PrecisionTYPE yReal = *fieldPtrY;
				
				// From real to pixel position
				PrecisionTYPE xVoxel = targetMatrix_real_to_voxel->m[0][0]*xReal
					+ targetMatrix_real_to_voxel->m[0][1]*yReal + targetMatrix_real_to_voxel->m[0][3];
				PrecisionTYPE yVoxel = targetMatrix_real_to_voxel->m[1][0]*xReal
					+ targetMatrix_real_to_voxel->m[1][1]*yReal + targetMatrix_real_to_voxel->m[1][3];
				
				xVoxel = xVoxel<0.0?0.0:xVoxel;
				yVoxel = yVoxel<0.0?0.0:yVoxel;
				
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
//                    memset(xControlPointCoordinates, 0, 16*sizeof(PrecisionTYPE));
//                    memset(yControlPointCoordinates, 0, 16*sizeof(PrecisionTYPE));
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
                    *fieldPtrX = *fieldPtrX -xInitial + (FieldTYPE)xReal;
                    *fieldPtrY = *fieldPtrY -yInitial + (FieldTYPE)yReal;
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
    FieldTYPE *fieldPtrY=&fieldPtrX[targetImage->nvox];
    FieldTYPE *fieldPtrZ=&fieldPtrY[targetImage->nvox];

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
					PrecisionTYPE xReal = *fieldPtrX;
					PrecisionTYPE yReal = *fieldPtrY;
					PrecisionTYPE zReal = *fieldPtrZ;
					
					// From real to pixel position
					PrecisionTYPE xVoxel = targetMatrix_real_to_voxel->m[0][0]*xReal
					+ targetMatrix_real_to_voxel->m[0][1]*yReal
					+ targetMatrix_real_to_voxel->m[0][2]*zReal
					+ targetMatrix_real_to_voxel->m[0][3];
					PrecisionTYPE yVoxel = targetMatrix_real_to_voxel->m[1][0]*xReal
					+ targetMatrix_real_to_voxel->m[1][1]*yReal
					+ targetMatrix_real_to_voxel->m[1][2]*zReal
					+ targetMatrix_real_to_voxel->m[1][3];
					PrecisionTYPE zVoxel = targetMatrix_real_to_voxel->m[2][0]*xReal
					+ targetMatrix_real_to_voxel->m[2][1]*yReal
					+ targetMatrix_real_to_voxel->m[2][2]*zReal
					+ targetMatrix_real_to_voxel->m[2][3];
					
					xVoxel = xVoxel<0.0?0.0:xVoxel;
					yVoxel = yVoxel<0.0?0.0:yVoxel;
					zVoxel = zVoxel<0.0?0.0:zVoxel;
					
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
						unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
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
					
					*fieldPtrX = (FieldTYPE)xReal;
					*fieldPtrY = (FieldTYPE)yReal;
					*fieldPtrY = (FieldTYPE)yReal;
					
					fieldPtrX++;
					fieldPtrY++;
					fieldPtrZ++;
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
//                        memset(xControlPointCoordinates, 0, 64*sizeof(PrecisionTYPE));
//                        memset(yControlPointCoordinates, 0, 64*sizeof(PrecisionTYPE));
//                        memset(zControlPointCoordinates, 0, 64*sizeof(PrecisionTYPE));
                        coord=0;
                        for(int Z=zPre; Z<zPre+4; Z++){
                            unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
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
						*fieldPtrX = *fieldPtrX -xInitial + (FieldTYPE)xReal;
						*fieldPtrY = *fieldPtrY -yInitial + (FieldTYPE)yReal;
						*fieldPtrZ = *fieldPtrZ -zInitial + (FieldTYPE)zReal;
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
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

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
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

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

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1]
                )>0.0f?1.0f:-1.0f;

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
                        xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X]*orientation[0];
                        yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X]*orientation[1];
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

            Tx_x /= splineControlPoint->dx;
            Ty_x /= splineControlPoint->dx;

            Tx_y /= splineControlPoint->dy;
            Ty_y /= splineControlPoint->dy;

            PrecisionTYPE detJac =  Tx_x * Ty_y - Tx_y * Ty_x;
			if(detJac>0.0)
				constraintValue += log(detJac)*log(detJac);
			else constraintValue += 100.0;
        }
    }

    return constraintValue/(targetImage->nx*targetImage->ny*targetImage->nz);
        
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobianValue3D(  nifti_image *splineControlPoint,
                    nifti_image *targetImage)
{
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

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1] +
                (splineMatrix->m[0][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][2]*splineMatrix->m[0][2]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1] +
                (splineMatrix->m[1][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][2]*splineMatrix->m[1][2]
                )>0.0f?1.0f:-1.0f;
    orientation[2] = (  (splineMatrix->m[2][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][0]*splineMatrix->m[2][0] +
                (splineMatrix->m[2][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][1]*splineMatrix->m[2][1] +
                (splineMatrix->m[2][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][2]*splineMatrix->m[2][2]
                )>0.0f?1.0f:-1.0f;

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

            coord=0;
            for(int c=0; c<4; c++){
                for(int b=0; b<4; b++){
                    tempX[coord]=zBasis[c]*temp[b]; // z * y
                    tempY[coord]=zBasis[c]*first[b];// z * y'
                    tempZ[coord]=zFirst[c]*temp[b]; // z'* y
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
                                xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X]*orientation[0];
                                yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X]*orientation[1];
                                zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X]*orientation[2];
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

                Tx_x /= splineControlPoint->dx;
                Ty_x /= splineControlPoint->dx;
                Tz_x /= splineControlPoint->dx;

                Tx_y /= splineControlPoint->dy;
                Ty_y /= splineControlPoint->dy;
                Tz_y /= splineControlPoint->dy;

                Tx_z /= splineControlPoint->dz;
                Ty_z /= splineControlPoint->dz;
                Tz_z /= splineControlPoint->dz;

                PrecisionTYPE detJac =  Tz_z*(Ty_y*Tx_x - Tx_y*Ty_x) +
					Ty_z*(Tx_y*Tz_x - Tz_y*Tx_x) +
					Tx_z*(Tz_y*Ty_x - Ty_y*Tz_x);
				if(detJac>0.0)
					constraintValue += log(detJac)*log(detJac);
				else constraintValue += 100.0;
            }
        }
    }

    return constraintValue/(targetImage->nx*targetImage->ny*targetImage->nz);
        
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
    PrecisionTYPE normal[4];
    PrecisionTYPE first[4];
    normal[0] = (PrecisionTYPE)(1.0/6.0);
    normal[1] = (PrecisionTYPE)(2.0/3.0);
    normal[2] = (PrecisionTYPE)(1.0/6.0);
    normal[3] = (PrecisionTYPE)(0.0);
    first[0] = (PrecisionTYPE)(-0.5);
    first[1] = (PrecisionTYPE)(0.0);
    first[2] = (PrecisionTYPE)(0.5);
    first[3] = (PrecisionTYPE)(0.0);

    PrecisionTYPE constraintValue= (PrecisionTYPE)(0.0);

    PrecisionTYPE basisX[16], basisY[16];

    int coord=0;
    for(int b=0; b<4; b++){
        for(int a=0; a<4; a++){
            basisX[coord]=normal[b]*first[a];   // y * x'
            basisY[coord]=first[b]*normal[a];   // y'* x
            coord++;
        }
    }

    PrecisionTYPE xControlPointCoordinates[16];
    PrecisionTYPE yControlPointCoordinates[16];

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1]
                )>0.0f?1.0f:-1.0f;

    for(int y=1;y<splineControlPoint->ny-2;y++){
        for(int x=1;x<splineControlPoint->nx-2;x++){

            coord=0;
            for(int Y=y-1; Y<y+3; Y++){
                unsigned int index = Y*splineControlPoint->nx;
                SplineTYPE *xxPtr = &controlPointPtrX[index];
                SplineTYPE *yyPtr = &controlPointPtrY[index];
                for(int X=x-1; X<x+3; X++){
                    xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X]*orientation[0];
                    yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X]*orientation[1];
                    coord++;
                }
            }

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

            Tx_x /= splineControlPoint->dx;
            Ty_x /= splineControlPoint->dx;

            Tx_y /= splineControlPoint->dy;
            Ty_y /= splineControlPoint->dy;

            PrecisionTYPE detJac =  Tx_x * Ty_y - Tx_y * Ty_x;
			if(detJac>0.0)
				constraintValue += log(detJac)*log(detJac);
			else constraintValue += 100.0;
        }
    }

    return constraintValue/(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
}
/* *************************************************************** */
template<class PrecisionTYPE, class SplineTYPE>
PrecisionTYPE reg_bspline_jacobianApproxValue3D(  nifti_image *splineControlPoint,
                            nifti_image *targetImage
                            )
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

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

    // There are six different values taken into account
    PrecisionTYPE tempX[9], tempY[9], tempZ[9];

    int coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempX[coord]=normal[c]*normal[b];   // z * y
            tempY[coord]=normal[c]*first[b];    // z * y'
            tempZ[coord]=first[c]*normal[b];    // z'* y
            coord++;
        }
    }
    
    PrecisionTYPE basisX[27], basisY[27], basisZ[27];
    
    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisX[coord]=tempX[bc]*first[a];   // z * y * x'
            basisY[coord]=tempY[bc]*normal[a];  // z * y'* x
            basisZ[coord]=tempZ[bc]*normal[a];  // z'* y * x
            coord++;
        }
    }
    
    PrecisionTYPE xControlPointCoordinates[27];
    PrecisionTYPE yControlPointCoordinates[27];
    PrecisionTYPE zControlPointCoordinates[27];

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1] +
                (splineMatrix->m[0][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][2]*splineMatrix->m[0][2]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1] +
                (splineMatrix->m[1][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][2]*splineMatrix->m[1][2]
                )>0.0f?1.0f:-1.0f;
    orientation[2] = (  (splineMatrix->m[2][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][0]*splineMatrix->m[2][0] +
                (splineMatrix->m[2][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][1]*splineMatrix->m[2][1] +
                (splineMatrix->m[2][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][2]*splineMatrix->m[2][2]
                )>0.0f?1.0f:-1.0f;
    
    for(int z=1;z<splineControlPoint->nz-2;z++){
        for(int y=1;y<splineControlPoint->ny-2;y++){
            for(int x=1;x<splineControlPoint->nx-2;x++){
                
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
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X]*orientation[0];
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X]*orientation[1];
                            zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X]*orientation[2];
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
                
                Tx_x /= splineControlPoint->dx;
                Ty_x /= splineControlPoint->dx;
                Tz_x /= splineControlPoint->dx;
                
                Tx_y /= splineControlPoint->dy;
                Ty_y /= splineControlPoint->dy;
                Tz_y /= splineControlPoint->dy;
                
                Tx_z /= splineControlPoint->dz;
                Ty_z /= splineControlPoint->dz;
                Tz_z /= splineControlPoint->dz;

                PrecisionTYPE detJac =  Tz_z*(Ty_y*Tx_x - Tx_y*Ty_x) +
					Ty_z*(Tx_y*Tz_x - Tz_y*Tx_x) +
					Tx_z*(Tz_y*Ty_x - Ty_y*Tz_x);
				if(detJac>0.0)
					constraintValue += log(detJac)*log(detJac);
				else constraintValue += 100.0;
            }
        }
    }
    
    return constraintValue/(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
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
void reg_bspline_bendingEnergyGradient3D(   nifti_image *splineControlPoint,
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
            basisXY[coord]=tempXY[bc]*first[a]; // z * y'* x'
            basisYZ[coord]=tempYZ[bc]*normal[a];    // z'* y'* x
            basisXZ[coord]=tempXZ[bc]*first[a]; // z'* y * x'
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
    SplineTYPE *gradientY = static_cast<SplineTYPE *>(&gradientX[(int)nodeNumber]);
    SplineTYPE *gradientZ = static_cast<SplineTYPE *>(&gradientY[(int)nodeNumber]);
    SplineTYPE *gradientXPtr = &gradientX[0];
    SplineTYPE *gradientYPtr = &gradientY[0];
    SplineTYPE *gradientZPtr = &gradientZ[0];

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
                *gradientXPtr++ += weight*gradientValue[0];
                *gradientYPtr++ += weight*gradientValue[1];
                *gradientZPtr++ += weight*gradientValue[2];
            }
        }
    }

    free(derivativeValues);
}
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_bendingEnergyGradient2D(   nifti_image *splineControlPoint,
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
            *gradientXPtr++ += weight*gradientValue[0];
            *gradientYPtr++ += weight*gradientValue[1];
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
                reg_bspline_bendingEnergyGradient2D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                break;
                reg_bspline_bendingEnergyGradient2D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"The bending energy gradient has not been computed\n");
                break;
        }
        }else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_bendingEnergyGradient3D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                break;
                reg_bspline_bendingEnergyGradient3D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
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
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradient2D(nifti_image *splineControlPoint,
											   nifti_image *targetImage,
											   nifti_image *gradientImage,
											   float weight)
{
	//TODO
	fprintf(stderr, "The 2D gradient of the Jacobian determinant has not been implemented yet.\n");
	fprintf(stderr, "... Exit ...\n");
	exit(0);
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradientApprox2D(nifti_image *splineControlPoint,
													 nifti_image *targetImage,
													 nifti_image *gradientImage,
													 float weight)
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
	
    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
	
    float orientation[2];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
					  (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1]
					  )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
					  (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1]
					  )>0.0f?1.0f:-1.0f;
	
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);
	
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = static_cast<SplineTYPE *>(&gradientImagePtrX[splineControlPoint->nx*splineControlPoint->ny]);
	
	// Loop over each control point
    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){
			
			PrecisionTYPE jacobianContraintX=(PrecisionTYPE)0.0;
			PrecisionTYPE jacobianContraintY=(PrecisionTYPE)0.0;
			
			// Loop over all the control points in the surrounding area
			unsigned int basisCoord=0;
			for(int CPY=y-1;CPY<y+2; CPY++){
				for(int CPX=x-1;CPX<x+2; CPX++){
					
					// The control points are stored
					coord=0;
					for(int Y=CPY-1; Y<CPY+2; Y++){
						PrecisionTYPE diffY = (PrecisionTYPE)0.0;
						int index = Y*splineControlPoint->nx;
						if(Y<0){ // Border effect
							diffY=Y;
							index=0;
						}
						else if(Y>=splineControlPoint->ny){ // Border effect
							diffY=Y-splineControlPoint->ny+1;
							index=(splineControlPoint->ny-1)*splineControlPoint->nx;
						}
						SplineTYPE *xxPtr = &controlPointPtrX[index];
						SplineTYPE *yyPtr = &controlPointPtrY[index];
						
						for(int X=CPX-1; X<CPX+2; X++){
							PrecisionTYPE diffX = (PrecisionTYPE)0.0;
							index=X;
							if(X<0){ // Border effect
								diffX=X;
								index=0;
							}
							else if(X>=splineControlPoint->nx){ // Border effect
								index=splineControlPoint->nx-1;
								diffX=X-index;
							}
							// home-made sliding effect which reproduce the control point spacing
							xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[index]*orientation[0] + diffX*splineControlPoint->dx;
							yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[index]*orientation[1] + diffY*splineControlPoint->dy;
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

					Tx_x /= splineControlPoint->dx;
					Ty_x /= splineControlPoint->dx;

					Tx_y /= splineControlPoint->dy;
					Ty_y /= splineControlPoint->dy;

					PrecisionTYPE detJac =  Tx_x * Ty_y - Tx_y * Ty_x;
					
					PrecisionTYPE basisValues[2];
					basisValues[0] = basisX[basisCoord];
					basisValues[1] = basisY[basisCoord];
					
					PrecisionTYPE jacobianInverse[4];
					jacobianInverse[0]=detJac*Ty_y;
					jacobianInverse[1]=-detJac*Ty_x;
					jacobianInverse[2]=-detJac*Tx_y;
					jacobianInverse[3]=detJac*Tx_x;
					
					PrecisionTYPE twoLogDet=2.0 * log(detJac);
					if(detJac<=0.0) twoLogDet = -100;	
										
					jacobianContraintX -=	twoLogDet
						* (jacobianInverse[0]*basisValues[0]+jacobianInverse[1]*basisValues[1]);
					jacobianContraintY -=	twoLogDet
						* (jacobianInverse[2]*basisValues[0]+jacobianInverse[3]*basisValues[1]);
			
					basisCoord++;
				} // X
			} // Y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
			*gradientImagePtrX++ += weight*jacobianContraintX*orientation[0];
			*gradientImagePtrY++ += weight*jacobianContraintY*orientation[1];
        } // x
    } // y
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradient3D(nifti_image *splineControlPoint,
											   nifti_image *targetImage,
											   nifti_image *gradientImage,
											   float weight)
{
	//TODO
	fprintf(stderr, "The 3D gradient of the Jacobian determinant has not been implemented yet.\n");
	fprintf(stderr, "... Exit ...\n");
	exit(0);
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class SplineTYPE>
void reg_bspline_jacobianDeterminantGradientApprox3D(nifti_image *splineControlPoint,
													 nifti_image *targetImage,
													 nifti_image *gradientImage,
													 float weight)
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
	
    // There are 3 different values taken into account
    PrecisionTYPE tempX[9], tempY[9], tempZ[9];
    int coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempX[coord]=normal[c]*normal[b];   // z * y
            tempY[coord]=normal[c]*first[b];    // z * y'
            tempZ[coord]=first[c]*normal[b];    // z'* y
            coord++;
        }
    }
    
    PrecisionTYPE basisX[27], basisY[27], basisZ[27];
    
    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisX[coord]=tempX[bc]*first[a];   // z * y * x'
            basisY[coord]=tempY[bc]*normal[a];  // z * y'* x
            basisZ[coord]=tempZ[bc]*normal[a];  // z'* y * x
            coord++;
        }
    }
	
    PrecisionTYPE xControlPointCoordinates[27];
    PrecisionTYPE yControlPointCoordinates[27];
    PrecisionTYPE zControlPointCoordinates[27];
	
    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
	
    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
					  (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1] +
					  (splineMatrix->m[0][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][2]*splineMatrix->m[0][2]
					  )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
					  (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1] +
					  (splineMatrix->m[1][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][2]*splineMatrix->m[1][2]
					  )>0.0f?1.0f:-1.0f;
    orientation[2] = (  (splineMatrix->m[2][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][0]*splineMatrix->m[2][0] +
					  (splineMatrix->m[2][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][1]*splineMatrix->m[2][1] +
					  (splineMatrix->m[2][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][2]*splineMatrix->m[2][2]
					  )>0.0f?1.0f:-1.0f;
	
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	/* All the Jacobian matrix are computed */
	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	mat33 *inverseJacobianMatrices =
	(mat33 *)malloc((gradientImage->nx+2)
					*(gradientImage->ny+2)
					*(gradientImage->nz+2)
					* sizeof(mat33));
	mat33 *inverseJacobianMatricesPtr = &inverseJacobianMatrices[0];
	PrecisionTYPE *jacobianDeterminants = 
	(PrecisionTYPE *)malloc((gradientImage->nx+2)
							*(gradientImage->ny+2)
							*(gradientImage->nz+2)
							* sizeof(PrecisionTYPE));
	PrecisionTYPE *jacobianDeterminantsPtr = &jacobianDeterminants[0];
	// Loop over each control point
	for(int z=-1;z<gradientImage->nz+1;z++){
		for(int y=-1;y<gradientImage->ny+1;y++){
			for(int x=-1;x<gradientImage->nx+1;x++){
							
				// The control points are stored
				coord=0;
				for(int Z=z-1; Z<z+2; Z++){
					PrecisionTYPE diffZ = (PrecisionTYPE)0.0;
					int index = Z*splineControlPoint->nx*splineControlPoint->ny;
					if(Z<0){ // Border effect
						diffZ=Z;
						index=0;
					}
					else if(Z>=splineControlPoint->nz){ // Border effect
						diffZ=Z-splineControlPoint->nz+1;
						index=(splineControlPoint->nz-1)*splineControlPoint->nx*splineControlPoint->ny;
					}
					SplineTYPE *xPtr = &controlPointPtrX[index];
					SplineTYPE *yPtr = &controlPointPtrY[index];
					SplineTYPE *zPtr = &controlPointPtrZ[index];
					
					for(int Y=y-1; Y<y+2; Y++){
						PrecisionTYPE diffY = (PrecisionTYPE)0.0;
						int index = Y*splineControlPoint->nx;
						if(Y<0){ // Border effect
							diffY=Y;
							index=0;
						}
						else if(Y>=splineControlPoint->ny){ // Border effect
							diffY=Y-splineControlPoint->ny+1;
							index=(splineControlPoint->ny-1)*splineControlPoint->nx;
						}
						SplineTYPE *xxPtr = &xPtr[index];
						SplineTYPE *yyPtr = &yPtr[index];
						SplineTYPE *zzPtr = &zPtr[index];
						
						for(int X=x-1; X<x+2; X++){
							PrecisionTYPE diffX = (PrecisionTYPE)0.0;
							index=X;
							if(X<0){ // Border effect
								diffX=X;
								index=0;
							}
							else if(X>=splineControlPoint->nx){ // Border effect
								index=splineControlPoint->nx-1;
								diffX=X-index;
							}
							// home-made sliding effect which reproduce the control point spacing
							xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[index]*orientation[0] + diffX*splineControlPoint->dx;
							yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[index]*orientation[1] + diffY*splineControlPoint->dy;
							zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[index]*orientation[2] + diffZ*splineControlPoint->dz;
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

				mat33 jacobianMatrix;
				jacobianMatrix.m[0][0] = Tx_x/splineControlPoint->dx;
				jacobianMatrix.m[0][1] = Tx_y/splineControlPoint->dy;
				jacobianMatrix.m[0][2] = Tx_z/splineControlPoint->dz;
				jacobianMatrix.m[1][0] = Ty_x/splineControlPoint->dx;
				jacobianMatrix.m[1][1] = Ty_y/splineControlPoint->dy;
				jacobianMatrix.m[1][2] = Ty_z/splineControlPoint->dz;
				jacobianMatrix.m[2][0] = Tz_x/splineControlPoint->dx;
				jacobianMatrix.m[2][1] = Tz_y/splineControlPoint->dy;
				jacobianMatrix.m[2][2] = Tz_z/splineControlPoint->dz;

				*jacobianDeterminantsPtr++ = nifti_mat33_determ(jacobianMatrix);
				*inverseJacobianMatricesPtr++ = nifti_mat33_inverse(jacobianMatrix);

			} // x
		} // y
	} //z

	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
	/* The actual gradient are now computed */
	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = static_cast<SplineTYPE *>(&gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz]);
    SplineTYPE *gradientImagePtrZ = static_cast<SplineTYPE *>(&gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz]);
	// Loop over each control point
	for(int z=0;z<gradientImage->nz;z++){
		for(int y=0;y<gradientImage->ny;y++){
			for(int x=0;x<gradientImage->nx;x++){
				
				PrecisionTYPE jacobianContraintX=(PrecisionTYPE)0.0;
				PrecisionTYPE jacobianContraintY=(PrecisionTYPE)0.0;
				PrecisionTYPE jacobianContraintZ=(PrecisionTYPE)0.0;
				
				// Loop over all the control points in the surrounding area
				unsigned int basisCoord=0;
				for(int CPZ=z-1;CPZ<z+2; CPZ++){
					for(int CPY=y-1;CPY<y+2; CPY++){
						for(int CPX=x-1;CPX<x+2; CPX++){
							
							
//							
//							// The control points are stored
//							coord=0;
//							for(int Z=CPZ-1; Z<CPZ+2; Z++){
//								PrecisionTYPE diffZ = (PrecisionTYPE)0.0;
//								int index = Z*splineControlPoint->nx*splineControlPoint->ny;
//								if(Z<0){ // Border effect
//									diffZ=Z;
//									index=0;
//								}
//								else if(Z>=splineControlPoint->nz){ // Border effect
//									diffZ=Z-splineControlPoint->nz+1;
//									index=(splineControlPoint->nz-1)*splineControlPoint->nx*splineControlPoint->ny;
//								}
//								SplineTYPE *xPtr = &controlPointPtrX[index];
//								SplineTYPE *yPtr = &controlPointPtrY[index];
//								SplineTYPE *zPtr = &controlPointPtrZ[index];
//								
//								for(int Y=CPY-1; Y<CPY+2; Y++){
//									PrecisionTYPE diffY = (PrecisionTYPE)0.0;
//									int index = Y*splineControlPoint->nx;
//									if(Y<0){ // Border effect
//										diffY=Y;
//										index=0;
//									}
//									else if(Y>=splineControlPoint->ny){ // Border effect
//										diffY=Y-splineControlPoint->ny+1;
//										index=(splineControlPoint->ny-1)*splineControlPoint->nx;
//									}
//									SplineTYPE *xxPtr = &xPtr[index];
//									SplineTYPE *yyPtr = &yPtr[index];
//									SplineTYPE *zzPtr = &zPtr[index];
//									
//									for(int X=x-1; X<x+2; X++){
//										PrecisionTYPE diffX = (PrecisionTYPE)0.0;
//										index=X;
//										if(X<0){ // Border effect
//											diffX=X;
//											index=0;
//										}
//										else if(X>=splineControlPoint->nx){ // Border effect
//											index=splineControlPoint->nx-1;
//											diffX=X-index;
//										}
//										// home-made sliding effect which reproduce the control point spacing
//										xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[index]*orientation[0] + diffX*splineControlPoint->dx;
//										yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[index]*orientation[1] + diffY*splineControlPoint->dy;
//										zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[index]*orientation[2] + diffZ*splineControlPoint->dz;
//										coord++;
//									}
//								}
//							}
//							
//							PrecisionTYPE Tx_x=(PrecisionTYPE)0.0;
//							PrecisionTYPE Ty_x=(PrecisionTYPE)0.0;
//							PrecisionTYPE Tz_x=(PrecisionTYPE)0.0;
//							PrecisionTYPE Tx_y=(PrecisionTYPE)0.0;
//							PrecisionTYPE Ty_y=(PrecisionTYPE)0.0;
//							PrecisionTYPE Tz_y=(PrecisionTYPE)0.0;
//							PrecisionTYPE Tx_z=(PrecisionTYPE)0.0;
//							PrecisionTYPE Ty_z=(PrecisionTYPE)0.0;
//							PrecisionTYPE Tz_z=(PrecisionTYPE)0.0;
//							
//							for(int a=0; a<27; a++){
//								Tx_x += basisX[a]*xControlPointCoordinates[a];
//								Tx_y += basisY[a]*xControlPointCoordinates[a];
//								Tx_z += basisZ[a]*xControlPointCoordinates[a];
//								Ty_x += basisX[a]*yControlPointCoordinates[a];
//								Ty_y += basisY[a]*yControlPointCoordinates[a];
//								Ty_z += basisZ[a]*yControlPointCoordinates[a];
//								Tz_x += basisX[a]*zControlPointCoordinates[a];
//								Tz_y += basisY[a]*zControlPointCoordinates[a];
//								Tz_z += basisZ[a]*zControlPointCoordinates[a];
//							}
//							
//							mat33 jacobianMatrix;
//							jacobianMatrix.m[0][0] = Tx_x/splineControlPoint->dx;
//							jacobianMatrix.m[0][1] = Tx_y/splineControlPoint->dy;
//							jacobianMatrix.m[0][2] = Tx_z/splineControlPoint->dz;
//							jacobianMatrix.m[1][0] = Ty_x/splineControlPoint->dx;
//							jacobianMatrix.m[1][1] = Ty_y/splineControlPoint->dy;
//							jacobianMatrix.m[1][2] = Ty_z/splineControlPoint->dz;
//							jacobianMatrix.m[2][0] = Tz_x/splineControlPoint->dx;
//							jacobianMatrix.m[2][1] = Tz_y/splineControlPoint->dy;
//							jacobianMatrix.m[2][2] = Tz_z/splineControlPoint->dz;
//							
//							*jacobianDeterminantsPtr++ = nifti_mat33_determ(jacobianMatrix);
//							*inverseJacobianMatricesPtr++ = nifti_mat33_inverse(jacobianMatrix);
							
							
							
							
							unsigned int jacobianIndex =
								((CPZ+1)*(gradientImage->ny+2)+(CPY+1))
								*(gradientImage->nx+2)+CPX+1;
							
							PrecisionTYPE twoLogDet=2.0 * log(jacobianDeterminants[jacobianIndex]);
							if(twoLogDet != twoLogDet) twoLogDet = -100;

							mat33 jacobianMatrix=inverseJacobianMatrices[jacobianIndex];

							PrecisionTYPE basisValues[3];
							basisValues[0] = basisX[basisCoord];
							basisValues[1] = basisY[basisCoord];
							basisValues[2] = basisZ[basisCoord];

							jacobianContraintX -=	twoLogDet
							* ( jacobianMatrix.m[0][0]*basisValues[0]
							   + jacobianMatrix.m[1][0]*basisValues[1]
							   + jacobianMatrix.m[2][0]*basisValues[2]);
							jacobianContraintY -=	twoLogDet
							* ( jacobianMatrix.m[0][1]*basisValues[0]
							   + jacobianMatrix.m[1][1]*basisValues[1]
							   + jacobianMatrix.m[2][1]*basisValues[2]);
							jacobianContraintZ -=	twoLogDet
							* ( jacobianMatrix.m[0][2]*basisValues[0]
							   + jacobianMatrix.m[1][2]*basisValues[1]
							   + jacobianMatrix.m[2][2]*basisValues[2]);
							
							basisCoord++;
						} // X
					} // Y
				} // Z
				// (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
				*gradientImagePtrX++ += weight*jacobianContraintX*orientation[0];
				*gradientImagePtrY++ += weight*jacobianContraintY*orientation[1];
				*gradientImagePtrZ++ += weight*jacobianContraintZ*orientation[2];
			} // x
		} // y
	} //z
	free(inverseJacobianMatrices);
	free(jacobianDeterminants);
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
					reg_bspline_jacobianDeterminantGradientApprox2D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_bspline_jacobianDeterminantGradientApprox2D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
					fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
					fprintf(stderr,"The bending energy gradient has not computed\n");
					break;
			}		
		}
		else{
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
					reg_bspline_jacobianDeterminantGradient2D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_bspline_jacobianDeterminantGradient2D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
					fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
					fprintf(stderr,"The bending energy gradient has not computed\n");
					break;
			}
		}
    }
	else{
		if(approx){
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
					reg_bspline_jacobianDeterminantGradientApprox3D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_bspline_jacobianDeterminantGradientApprox3D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
					fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
					fprintf(stderr,"The bending energy gradient has not computed\n");
					break;
			}		
		}
		else{
			switch(splineControlPoint->datatype){
				case NIFTI_TYPE_FLOAT32:
					reg_bspline_jacobianDeterminantGradient3D<PrecisionTYPE, float>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				case NIFTI_TYPE_FLOAT64:
					reg_bspline_jacobianDeterminantGradient3D<PrecisionTYPE, double>(splineControlPoint, targetImage, gradientImage, weight);
					break;
				default:
					fprintf(stderr,"Only single or double precision is implemented for the Jacobian determinant gradient\n");
					fprintf(stderr,"The bending energy gradient has not computed\n");
					break;
			}
		}
	}
}
template void reg_bspline_jacobianDeterminantGradient<float>(nifti_image *, nifti_image *, nifti_image *, float, bool);
template void reg_bspline_jacobianDeterminantGradient<double>(nifti_image *, nifti_image *, nifti_image *, float, bool);
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

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+4;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+4;
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

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+4;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+4;
    splineControlPoint->dim[3]=splineControlPoint->nz=(int)floor(targetImage->nz*targetImage->dz/splineControlPoint->dz)+4;

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

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[2];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1]
                )>0.0f?1.0f:-1.0f;
    unsigned int coord=0;

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
                        xControlPointCoordinates[coord] = (JacobianTYPE)xPtr[X]*orientation[0];
                        yControlPointCoordinates[coord] = (JacobianTYPE)yPtr[X]*orientation[1];
                        coord++;
                    }
                }
            }
            oldBasis=basis;
            JacobianTYPE X_x=0.0;
            JacobianTYPE Y_x=0.0;
            JacobianTYPE X_y=0.0;
            JacobianTYPE Y_y=0.0;

            for(int a=0; a<16; a++){
                X_x += basisX[a]*xControlPointCoordinates[a];
                Y_x += basisY[a]*xControlPointCoordinates[a];

                X_y += basisX[a]*yControlPointCoordinates[a];
                Y_y += basisY[a]*yControlPointCoordinates[a];
            }

            X_x /= splineControlPoint->dx;
            Y_x /= splineControlPoint->dx;

            X_y /= splineControlPoint->dy;
            Y_y /= splineControlPoint->dy;

            *jacobianMapPtr++ = X_x * Y_y - X_y * Y_x;
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

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1] +
                (splineMatrix->m[0][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][2]*splineMatrix->m[0][2]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1] +
                (splineMatrix->m[1][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][2]*splineMatrix->m[1][2]
                )>0.0f?1.0f:-1.0f;
    orientation[2] = (  (splineMatrix->m[2][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][0]*splineMatrix->m[2][0] +
                (splineMatrix->m[2][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][1]*splineMatrix->m[2][1] +
                (splineMatrix->m[2][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][2]*splineMatrix->m[2][2]
                )>0.0f?1.0f:-1.0f;
    unsigned int coord=0;
    
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
                                xControlPointCoordinates[coord] = (JacobianTYPE)xxPtr[X]*orientation[0];
                                yControlPointCoordinates[coord] = (JacobianTYPE)yyPtr[X]*orientation[1];
                                zControlPointCoordinates[coord] = (JacobianTYPE)zzPtr[X]*orientation[2];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;
                
                JacobianTYPE X_x=0.0;
                JacobianTYPE Y_x=0.0;
                JacobianTYPE Z_x=0.0;
                JacobianTYPE X_y=0.0;
                JacobianTYPE Y_y=0.0;
                JacobianTYPE Z_y=0.0;
                JacobianTYPE X_z=0.0;
                JacobianTYPE Y_z=0.0;
                JacobianTYPE Z_z=0.0;
                
                for(int a=0; a<64; a++){
                    X_x += basisX[a]*xControlPointCoordinates[a];
                    Y_x += basisY[a]*xControlPointCoordinates[a];
                    Z_x += basisZ[a]*xControlPointCoordinates[a];
                    
                    X_y += basisX[a]*yControlPointCoordinates[a];
                    Y_y += basisY[a]*yControlPointCoordinates[a];
                    Z_y += basisZ[a]*yControlPointCoordinates[a];
                    
                    X_z += basisX[a]*zControlPointCoordinates[a];
                    Y_z += basisY[a]*zControlPointCoordinates[a];
                    Z_z += basisZ[a]*zControlPointCoordinates[a];
                }
                
                X_x /= splineControlPoint->dx;
                Y_x /= splineControlPoint->dx;
                Z_x /= splineControlPoint->dx;
                
                X_y /= splineControlPoint->dy;
                Y_y /= splineControlPoint->dy;
                Z_y /= splineControlPoint->dy;
                
                X_z /= splineControlPoint->dz;
                Y_z /= splineControlPoint->dz;
                Z_z /= splineControlPoint->dz;

                JacobianTYPE jacobianValue = X_x * Y_y * Z_z + X_y * Y_z * Z_x + X_z * Y_x * Z_y;
                jacobianValue -= X_z * Y_y * Z_x + X_y * Y_x * Z_z + X_x * Y_z * Z_y;
                
                *jacobianMapPtr++ = jacobianValue;
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
template <class SplineTYPE, class JacobianTYPE>
void reg_bspline_GetJacobianMatrix2D(nifti_image *splineControlPoint,
                                nifti_image *jacobianImage
                                )
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    JacobianTYPE *jacobianMatrixTxxPtr = static_cast<JacobianTYPE *>(jacobianImage->data);
    JacobianTYPE *jacobianMatrixTyxPtr = &jacobianMatrixTxxPtr[jacobianImage->nx*jacobianImage->ny];

    JacobianTYPE *jacobianMatrixTxyPtr = &jacobianMatrixTyxPtr[jacobianImage->nx*jacobianImage->ny];
    JacobianTYPE *jacobianMatrixTyyPtr = &jacobianMatrixTxyPtr[jacobianImage->nx*jacobianImage->ny];

    JacobianTYPE yBasis[4],yFirst[4],temp[4],first[4];
    JacobianTYPE basisX[16], basisY[16];
    JacobianTYPE basis, FF, FFF, MF, oldBasis=(JacobianTYPE)(1.1);

    JacobianTYPE xControlPointCoordinates[16];
    JacobianTYPE yControlPointCoordinates[16];

    JacobianTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[2];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1]
                )>0.0f?1.0f:-1.0f;

    unsigned int coord=0;

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
                        xControlPointCoordinates[coord] = (JacobianTYPE)xPtr[X]*orientation[0];
                        yControlPointCoordinates[coord] = (JacobianTYPE)yPtr[X]*orientation[1];
                        coord++;
                    }
                }
            }
            oldBasis=basis;
            JacobianTYPE X_x=0.0;
            JacobianTYPE Y_x=0.0;
            JacobianTYPE X_y=0.0;
            JacobianTYPE Y_y=0.0;

            for(int a=0; a<16; a++){
                X_x += basisX[a]*xControlPointCoordinates[a];
                Y_x += basisY[a]*xControlPointCoordinates[a];

                X_y += basisX[a]*yControlPointCoordinates[a];
                Y_y += basisY[a]*yControlPointCoordinates[a];
            }

            X_x /= splineControlPoint->dx;
            Y_x /= splineControlPoint->dx;

            X_y /= splineControlPoint->dy;
            Y_y /= splineControlPoint->dy;

            *jacobianMatrixTxxPtr++ = X_x;
            *jacobianMatrixTyxPtr++ = Y_x;

            *jacobianMatrixTxyPtr++ = X_y;
            *jacobianMatrixTyyPtr++ = Y_y;
        }
    }
}
/* *************************************************************** */
template <class SplineTYPE, class JacobianTYPE>
void reg_bspline_GetJacobianMatrix3D(nifti_image *splineControlPoint,
                                nifti_image *jacobianImage
                                )
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    
    JacobianTYPE *jacobianMatrixTxxPtr = static_cast<JacobianTYPE *>(jacobianImage->data);
    JacobianTYPE *jacobianMatrixTyxPtr = &jacobianMatrixTxxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    JacobianTYPE *jacobianMatrixTzxPtr = &jacobianMatrixTyxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    
    JacobianTYPE *jacobianMatrixTxyPtr = &jacobianMatrixTzxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    JacobianTYPE *jacobianMatrixTyyPtr = &jacobianMatrixTxyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    JacobianTYPE *jacobianMatrixTzyPtr = &jacobianMatrixTyyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    
    JacobianTYPE *jacobianMatrixTxzPtr = &jacobianMatrixTzyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    JacobianTYPE *jacobianMatrixTyzPtr = &jacobianMatrixTxzPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    JacobianTYPE *jacobianMatrixTzzPtr = &jacobianMatrixTyzPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    
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

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    float orientation[3];
    orientation[0] = (  (splineMatrix->m[0][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][0]*splineMatrix->m[0][0] +
                (splineMatrix->m[0][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][1]*splineMatrix->m[0][1] +
                (splineMatrix->m[0][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[0][2]*splineMatrix->m[0][2]
                )>0.0f?1.0f:-1.0f;
    orientation[1] = (  (splineMatrix->m[1][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][0]*splineMatrix->m[1][0] +
                (splineMatrix->m[1][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][1]*splineMatrix->m[1][1] +
                (splineMatrix->m[1][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[1][2]*splineMatrix->m[1][2]
                )>0.0f?1.0f:-1.0f;
    orientation[2] = (  (splineMatrix->m[2][0]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][0]*splineMatrix->m[2][0] +
                (splineMatrix->m[2][1]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][1]*splineMatrix->m[2][1] +
                (splineMatrix->m[2][2]>0.0f?1.0f:-1.0f)*splineMatrix->m[2][2]*splineMatrix->m[2][2]
                )>0.0f?1.0f:-1.0f;

    unsigned int coord=0;
    
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
        zFirst[0] = (JacobianTYPE)(basis - 1.0/2.0 - zFirst[3]);
        zFirst[2] = (JacobianTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
        zFirst[1] = (JacobianTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);
        
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
                                xControlPointCoordinates[coord] = (JacobianTYPE)xxPtr[X]*orientation[0];
                                yControlPointCoordinates[coord] = (JacobianTYPE)yyPtr[X]*orientation[1];
                                zControlPointCoordinates[coord] = (JacobianTYPE)zzPtr[X]*orientation[2];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;
                
                JacobianTYPE X_x=0.0;
                JacobianTYPE Y_x=0.0;
                JacobianTYPE Z_x=0.0;
                JacobianTYPE X_y=0.0;
                JacobianTYPE Y_y=0.0;
                JacobianTYPE Z_y=0.0;
                JacobianTYPE X_z=0.0;
                JacobianTYPE Y_z=0.0;
                JacobianTYPE Z_z=0.0;
                
                for(int a=0; a<64; a++){
                    X_x += basisX[a]*xControlPointCoordinates[a];
                    Y_x += basisY[a]*xControlPointCoordinates[a];
                    Z_x += basisZ[a]*xControlPointCoordinates[a];
                    
                    X_y += basisX[a]*yControlPointCoordinates[a];
                    Y_y += basisY[a]*yControlPointCoordinates[a];
                    Z_y += basisZ[a]*yControlPointCoordinates[a];
                    
                    X_z += basisX[a]*zControlPointCoordinates[a];
                    Y_z += basisY[a]*zControlPointCoordinates[a];
                    Z_z += basisZ[a]*zControlPointCoordinates[a];
                }
                
                X_x /= splineControlPoint->dx;
                Y_x /= splineControlPoint->dx;
                Z_x /= splineControlPoint->dx;
                
                X_y /= splineControlPoint->dy;
                Y_y /= splineControlPoint->dy;
                Z_y /= splineControlPoint->dy;
                
                X_z /= splineControlPoint->dz;
                Y_z /= splineControlPoint->dz;
                Z_z /= splineControlPoint->dz;

                *jacobianMatrixTxxPtr++ = X_x;
                *jacobianMatrixTyxPtr++ = Y_x;
                *jacobianMatrixTzxPtr++ = Z_x;
                
                *jacobianMatrixTxyPtr++ = X_y;
                *jacobianMatrixTyyPtr++ = Y_y;
                *jacobianMatrixTzyPtr++ = Z_y;
                
                *jacobianMatrixTxzPtr++ = X_z;
                *jacobianMatrixTyzPtr++ = Y_z;
                *jacobianMatrixTzzPtr++ = Z_z;
            }
        }
    }
}
/* *************************************************************** */
template <class SplineTYPE>
void reg_bspline_GetJacobianMatrix1(nifti_image *splineControlPoint,
								nifti_image *jacobianImage
								)
{
    if(splineControlPoint->nz==1){
        switch(jacobianImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_GetJacobianMatrix2D<SplineTYPE,float>(splineControlPoint, jacobianImage);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_GetJacobianMatrix2D<SplineTYPE,double>(splineControlPoint, jacobianImage);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the jacobian matrix image\n");
                fprintf(stderr,"The jacobian matrix image has not computed\n");
        }
    }
    else{
        switch(jacobianImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_GetJacobianMatrix3D<SplineTYPE,float>(splineControlPoint, jacobianImage);
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_GetJacobianMatrix3D<SplineTYPE,double>(splineControlPoint, jacobianImage);
            default:
                fprintf(stderr,"Only single or double precision is implemented for the jacobian matrix image\n");
                fprintf(stderr,"The jacobian matrix image has not computed\n");
        }
    }
}
/* *************************************************************** */
void reg_bspline_GetJacobianMatrix(	nifti_image *splineControlPoint,
							   nifti_image *jacobianImage
							   )
{
	switch(splineControlPoint->datatype){
		case NIFTI_TYPE_FLOAT32:
			return reg_bspline_GetJacobianMatrix1<float>(splineControlPoint, jacobianImage);
		case NIFTI_TYPE_FLOAT64:
			return reg_bspline_GetJacobianMatrix1<double>(splineControlPoint, jacobianImage);
		default:
			fprintf(stderr,"Only single or double precision is implemented for the control point image\n");
			fprintf(stderr,"The jacobian matrix image has not been computed\n");
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
template<class PrecisionTYPE>
void reg_square_cpp_2D(nifti_image *splineControlPoint)
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
	
	// The input cpp is saved
	PrecisionTYPE *tempCPPStorage = (PrecisionTYPE *)malloc(splineControlPoint->nvox * sizeof(PrecisionTYPE));
	memcpy(tempCPPStorage, splineControlPoint->data, splineControlPoint->nvox*splineControlPoint->nbyper);
	
    PrecisionTYPE *controlPointPtrX = tempCPPStorage;
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];
	
    PrecisionTYPE *outCPPPtrX=static_cast<PrecisionTYPE *>(splineControlPoint->data);
    PrecisionTYPE *outCPPPtrY=&outCPPPtrX[splineControlPoint->nx*splineControlPoint->ny];
	
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
		
	mat44 *matrix_real_to_voxel;
	if(splineControlPoint->sform_code>0)
		matrix_real_to_voxel=&(splineControlPoint->sto_ijk);
	else matrix_real_to_voxel=&(splineControlPoint->qto_ijk);
	
	mat44 *matrix_voxel_to_real;
    if(splineControlPoint->sform_code>0) matrix_voxel_to_real=&(splineControlPoint->sto_xyz);
    else matrix_voxel_to_real=&(splineControlPoint->qto_xyz);
		
#ifdef _WINDOWS
	__declspec(align(16)) PrecisionTYPE xBasis[4];
#else
	PrecisionTYPE xBasis[4] __attribute__((aligned(16)));
#endif		
	// read the ijk sform or qform, as appropriate
	
	for(int y=0; y<splineControlPoint->ny; y++){
		for(int x=0; x<splineControlPoint->nx; x++){
			
			// Get the initial control point position
			PrecisionTYPE xInit = matrix_voxel_to_real->m[0][0]*(PrecisionTYPE)x
			+ matrix_voxel_to_real->m[0][1]*(PrecisionTYPE)y
			+ matrix_voxel_to_real->m[0][3];
			PrecisionTYPE yInit = matrix_voxel_to_real->m[1][0]*(PrecisionTYPE)x
			+ matrix_voxel_to_real->m[1][1]*(PrecisionTYPE)y
			+ matrix_voxel_to_real->m[1][3];
			
			// Get the control point actual position
			PrecisionTYPE xReal = *outCPPPtrX + xInit;
			PrecisionTYPE yReal = *outCPPPtrY + yInit;
			
			// Get the voxel based control point position
			PrecisionTYPE xVoxel = matrix_real_to_voxel->m[0][0]*xReal
			+ matrix_real_to_voxel->m[0][1]*yReal + matrix_real_to_voxel->m[0][3];
			PrecisionTYPE yVoxel = matrix_real_to_voxel->m[1][0]*xReal
			+ matrix_real_to_voxel->m[1][1]*yReal + matrix_real_to_voxel->m[1][3];
			
			xVoxel = xVoxel<0.0?0.0:xVoxel;
			yVoxel = yVoxel<0.0?0.0:yVoxel;
			
			printf("x[%i] y[%i] / [%g][%g] -> [%g][%g]\n", x, y, xReal, yReal, xVoxel, yVoxel);
//			exit(0);
			
			// The spline coefficients are computed
			int xPre=(int)(floor(xVoxel)-1);
			basis=(PrecisionTYPE)xVoxel-(PrecisionTYPE)xPre;
			if(basis<0.0) basis=0.0; //rounding error
			FF= basis*basis;
			FFF= FF*basis;
			MF=(PrecisionTYPE)(1.0-basis);
			xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
			xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
			xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
			xBasis[3] = (PrecisionTYPE)(FFF/6.0);
			
			int yPre=(int)(floor(yVoxel)-1);
			basis=(PrecisionTYPE)yVoxel-(PrecisionTYPE)yPre;
			if(basis<0.0) basis=0.0; //rounding error
			FF= basis*basis;
			FFF= FF*basis;
			MF=(PrecisionTYPE)(1.0-basis);
			yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
			yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
			yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
			yBasis[3] = (PrecisionTYPE)(FFF/6.0);
			
			// The control points are stored
			coord=0;
			memset(xControlPointCoordinates, 0, 16*sizeof(PrecisionTYPE));
			memset(yControlPointCoordinates, 0, 16*sizeof(PrecisionTYPE));
			for(int Y=yPre; Y<yPre+3; Y++){
				int index = Y*splineControlPoint->nx;
				PrecisionTYPE *xxPtr = &controlPointPtrX[index];
				PrecisionTYPE *yyPtr = &controlPointPtrY[index];
				
				for(int X=xPre; X<xPre+3; X++){
					// home-made sliding effect which reproduce the control point spacing
					if(X>-1 && Y>-1 && X<splineControlPoint->nx && Y<splineControlPoint->ny){
						xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[index];
						yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[index];
					}
					coord++;
				}
			}
			
			xReal=0.0;
			yReal=0.0;

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
			
			*outCPPPtrX++ = (PrecisionTYPE)xReal;
			*outCPPPtrY++ = (PrecisionTYPE)yReal;
		}
	}
	free(tempCPPStorage);
	return;
}
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_square_cpp_3D(nifti_image *splineControlPoint)
{
	// The input cpp is saved
	PrecisionTYPE *tempCPPStorage = (PrecisionTYPE *)malloc(splineControlPoint->nvox * sizeof(PrecisionTYPE));
	memcpy(tempCPPStorage, splineControlPoint->data, splineControlPoint->nvox*splineControlPoint->nbyper);
	
	
}
/* *************************************************************** */
template<class PrecisionTYPE>
int reg_square_cpp(nifti_image *splineControlPoint)
{
		
	switch(splineControlPoint->nu){
		case 2:
			reg_square_cpp_2D<PrecisionTYPE>(splineControlPoint);
			break;
		case 3:
			reg_square_cpp_3D<PrecisionTYPE>(splineControlPoint);
			break;
		default:
			fprintf(stderr,"ERROR:\treg_square_cpp\n");
			fprintf(stderr,"ERROR:\tOnly implemented for 2 or 3D images\n");
			return 1;
	}
	return 0;
}
template int reg_square_cpp<float>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_getDisplacementFromPosition_2D(nifti_image *splineControlPoint)
{
	PrecisionTYPE *controlPointPtrX = static_cast<PrecisionTYPE *>(splineControlPoint->data);
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];
	
	mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
	
	
	for(int y=0; y<splineControlPoint->ny; y++){
		for(int x=0; x<splineControlPoint->nx; x++){
			
			// Get the initial control point position
			PrecisionTYPE xInit = splineMatrix->m[0][0]*(PrecisionTYPE)x
			+ splineMatrix->m[0][1]*(PrecisionTYPE)y
			+ splineMatrix->m[0][3];
			PrecisionTYPE yInit = splineMatrix->m[1][0]*(PrecisionTYPE)x
			+ splineMatrix->m[1][1]*(PrecisionTYPE)y
			+ splineMatrix->m[1][3];
			
			// The initial position is subtracted from every values
			*controlPointPtrX++ -= xInit;
			*controlPointPtrY++ -= yInit;
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_getDisplacementFromPosition_3D(nifti_image *splineControlPoint)
{
	PrecisionTYPE *controlPointPtrX = static_cast<PrecisionTYPE *>(splineControlPoint->data);
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    PrecisionTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
	
	mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
	
	
	for(int z=0; z<splineControlPoint->nz; z++){
		for(int y=0; y<splineControlPoint->ny; y++){
			for(int x=0; x<splineControlPoint->nx; x++){
			
				// Get the initial control point position
				PrecisionTYPE xInit = splineMatrix->m[0][0]*(PrecisionTYPE)x
				+ splineMatrix->m[0][1]*(PrecisionTYPE)y
				+ splineMatrix->m[0][2]*(PrecisionTYPE)z
				+ splineMatrix->m[0][3];
				PrecisionTYPE yInit = splineMatrix->m[1][0]*(PrecisionTYPE)x
				+ splineMatrix->m[1][1]*(PrecisionTYPE)y
				+ splineMatrix->m[1][2]*(PrecisionTYPE)z
				+ splineMatrix->m[1][3];
				PrecisionTYPE zInit = splineMatrix->m[2][0]*(PrecisionTYPE)x
				+ splineMatrix->m[2][1]*(PrecisionTYPE)y
				+ splineMatrix->m[2][2]*(PrecisionTYPE)z
				+ splineMatrix->m[2][3];
				
				// The initial position is subtracted from every values
				*controlPointPtrX++ -= xInit;
				*controlPointPtrY++ -= yInit;
				*controlPointPtrZ++ -= zInit;
			}
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE>
int reg_getDisplacementFromPosition(nifti_image *splineControlPoint)
{		
	switch(splineControlPoint->nu){
		case 2:
			reg_getDisplacementFromPosition_2D<PrecisionTYPE>(splineControlPoint);
			break;
		case 3:
			reg_getDisplacementFromPosition_3D<PrecisionTYPE>(splineControlPoint);
			break;
		default:
			fprintf(stderr,"ERROR:\treg_getDisplacementFromPosition\n");
			fprintf(stderr,"ERROR:\tOnly implemented for 2 or 3D images\n");
			return 1;
	}
	return 0;
}
template int reg_getDisplacementFromPosition<float>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_getPositionFromDisplacement_2D(nifti_image *splineControlPoint)
{
	PrecisionTYPE *controlPointPtrX = static_cast<PrecisionTYPE *>(splineControlPoint->data);
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];
	
	mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
	
	
	for(int y=0; y<splineControlPoint->ny; y++){
		for(int x=0; x<splineControlPoint->nx; x++){
			
			// Get the initial control point position
			PrecisionTYPE xInit = splineMatrix->m[0][0]*(PrecisionTYPE)x
			+ splineMatrix->m[0][1]*(PrecisionTYPE)y
			+ splineMatrix->m[0][3];
			PrecisionTYPE yInit = splineMatrix->m[1][0]*(PrecisionTYPE)x
			+ splineMatrix->m[1][1]*(PrecisionTYPE)y
			+ splineMatrix->m[1][3];
			
			// The initial position is added from every values
			*controlPointPtrX++ += xInit;
			*controlPointPtrY++ += yInit;
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_getPositionFromDisplacement_3D(nifti_image *splineControlPoint)
{
	PrecisionTYPE *controlPointPtrX = static_cast<PrecisionTYPE *>(splineControlPoint->data);
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    PrecisionTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
	
	mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
	
	
	for(int z=0; z<splineControlPoint->nz; z++){
		for(int y=0; y<splineControlPoint->ny; y++){
			for(int x=0; x<splineControlPoint->nx; x++){
				
				// Get the initial control point position
				PrecisionTYPE xInit = splineMatrix->m[0][0]*(PrecisionTYPE)x
				+ splineMatrix->m[0][1]*(PrecisionTYPE)y
				+ splineMatrix->m[0][2]*(PrecisionTYPE)z
				+ splineMatrix->m[0][3];
				PrecisionTYPE yInit = splineMatrix->m[1][0]*(PrecisionTYPE)x
				+ splineMatrix->m[1][1]*(PrecisionTYPE)y
				+ splineMatrix->m[1][2]*(PrecisionTYPE)z
				+ splineMatrix->m[1][3];
				PrecisionTYPE zInit = splineMatrix->m[2][0]*(PrecisionTYPE)x
				+ splineMatrix->m[2][1]*(PrecisionTYPE)y
				+ splineMatrix->m[2][2]*(PrecisionTYPE)z
				+ splineMatrix->m[2][3];
				
				// The initial position is subtracted from every values
				*controlPointPtrX++ += xInit;
				*controlPointPtrY++ += yInit;
				*controlPointPtrZ++ += zInit;
			}
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE>
int reg_getPositionFromDisplacement(nifti_image *splineControlPoint)
{
	switch(splineControlPoint->nu){
		case 2:
			reg_getPositionFromDisplacement_2D<PrecisionTYPE>(splineControlPoint);
			break;
		case 3:
			reg_getPositionFromDisplacement_3D<PrecisionTYPE>(splineControlPoint);
			break;
		default:
			fprintf(stderr,"ERROR:\treg_getPositionFromDisplacement\n");
			fprintf(stderr,"ERROR:\tOnly implemented for 2 or 3D images\n");
			return 1;
	}
	return 0;}
template int reg_getPositionFromDisplacement<float>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
#endif