/*
 *  _reg_bspline_comp.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_COMP_CPP
#define _REG_BSPLINE_COMP_CPP

#include "_reg_bspline_comp.h"
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

void reg_spline_scaling_squaring(   nifti_image *velocityFieldImage,
                                    nifti_image *controlPointImage)
{

    // The velocity field is copied to the cpp image
    nifti_image *nodePositionImage=nifti_copy_nim_info(controlPointImage);
    nodePositionImage->data=(void *)calloc(controlPointImage->nvox, controlPointImage->nbyper);

    memcpy(nodePositionImage->data, velocityFieldImage->data,
           controlPointImage->nvox*controlPointImage->nbyper);

    // The control point image is decomposed
    reg_spline_Interpolant2Interpolator(nodePositionImage,
                                        controlPointImage);
//         memcpy(controlPointImage->data, nodePositionImage->data,
//             controlPointImage->nvox*controlPointImage->nbyper);

    // Squaring approach
    for(unsigned int i=0; i<SQUARING_VALUE; i++){
        reg_spline_cppComposition(  nodePositionImage,
                                    controlPointImage,
                                    1.0f,
                                    0);
        // The control point image is decomposed
        reg_spline_Interpolant2Interpolator(nodePositionImage,
                                            controlPointImage);
//         memcpy(controlPointImage->data, nodePositionImage->data,
//             controlPointImage->nvox*controlPointImage->nbyper);
    }
    nifti_image_free(nodePositionImage);

    switch(controlPointImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_getPositionFromDisplacement<float>(controlPointImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_getPositionFromDisplacement<double>(controlPointImage);
                break;
            default:
                fprintf(stderr, "reg_getPositionFromDisplacement can only be performed on single or double precision control point grid ... Exit");
                exit(0);
                return;
    }
}

template<class PrecisionTYPE>
void reg_spline_cppComposition_2D(  nifti_image *positionGridImage,
                                    nifti_image *decomposedGridImage,
                                    float ratio,
                                    bool type
                                    )
{   
#if _USE_SSE
    union u{
        __m128 m;
        float f[4];
    } val;
#endif  
    
    PrecisionTYPE *outCPPPtrX = static_cast<PrecisionTYPE *>(positionGridImage->data);
    PrecisionTYPE *outCPPPtrY = &outCPPPtrX[positionGridImage->nx*positionGridImage->ny];
    
    PrecisionTYPE *controlPointPtrX = static_cast<PrecisionTYPE *>(decomposedGridImage->data);
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[decomposedGridImage->nx*decomposedGridImage->ny];
    
    PrecisionTYPE basis, FF, FFF, MF;
    
#ifdef _WINDOWS
    __declspec(align(16)) PrecisionTYPE xBasis[4];
    __declspec(align(16)) PrecisionTYPE yBasis[4];
#if _USE_SSE
    __declspec(align(16)) PrecisionTYPE xyBasis[16];
#endif
    
    __declspec(align(16)) PrecisionTYPE xControlPointCoordinates[16];
    __declspec(align(16)) PrecisionTYPE yControlPointCoordinates[16];
#else
    PrecisionTYPE xBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE yBasis[4] __attribute__((aligned(16)));
#if _USE_SSE
    PrecisionTYPE xyBasis[16] __attribute__((aligned(16)));
#endif
    
    PrecisionTYPE xControlPointCoordinates[16] __attribute__((aligned(16)));
    PrecisionTYPE yControlPointCoordinates[16] __attribute__((aligned(16)));
#endif
    
    unsigned int coord;
    
    // read the xyz/ijk sform or qform, as appropriate
    mat44 *matrix_real_to_voxel=NULL;
    mat44 *matrix_voxel_to_real=NULL;
    if(decomposedGridImage->sform_code>0){
        matrix_real_to_voxel=&(decomposedGridImage->sto_ijk);
        matrix_voxel_to_real=&(decomposedGridImage->sto_xyz);
    }
    else{
        matrix_real_to_voxel=&(decomposedGridImage->qto_ijk);
        matrix_voxel_to_real=&(decomposedGridImage->qto_xyz);
    }

    for(int y=0; y<decomposedGridImage->ny; y++){
        for(int x=0; x<decomposedGridImage->nx; x++){
            
            // Get the initial control point position
            PrecisionTYPE xReal=0;
            PrecisionTYPE yReal=0;
            if(type==0){ // displacement are assumed, position otherwise
                xReal = matrix_voxel_to_real->m[0][0]*(PrecisionTYPE)x
                + matrix_voxel_to_real->m[0][1]*(PrecisionTYPE)y
                + matrix_voxel_to_real->m[0][3];
                yReal = matrix_voxel_to_real->m[1][0]*(PrecisionTYPE)x
                + matrix_voxel_to_real->m[1][1]*(PrecisionTYPE)y
                + matrix_voxel_to_real->m[1][3];
            }
            
            // Get the control point actual position
            xReal += *outCPPPtrX;
            yReal += *outCPPPtrY;
            
            // Get the voxel based control point position
            PrecisionTYPE xVoxel = matrix_real_to_voxel->m[0][0]*xReal
            + matrix_real_to_voxel->m[0][1]*yReal + matrix_real_to_voxel->m[0][3];
            PrecisionTYPE yVoxel = matrix_real_to_voxel->m[1][0]*xReal
            + matrix_real_to_voxel->m[1][1]*yReal + matrix_real_to_voxel->m[1][3];
            
            xVoxel = xVoxel<0.0?0.0:xVoxel;
            yVoxel = yVoxel<0.0?0.0:yVoxel;
            
            // The spline coefficients are computed
            int xPre=(int)(floor(xVoxel));
            basis=(PrecisionTYPE)xVoxel-(PrecisionTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(PrecisionTYPE)(1.0-basis);
            xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
            xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            xBasis[3] = (PrecisionTYPE)(FFF/6.0);
            
            int yPre=(int)(floor(yVoxel));
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
            for(int Y=yPre-1; Y<yPre+3; Y++){
                if(Y>-1 && Y<decomposedGridImage->ny){
                    int index = Y*decomposedGridImage->nx;
                    PrecisionTYPE *xPtr = &controlPointPtrX[index];
                    PrecisionTYPE *yPtr = &controlPointPtrY[index];
                    for(int X=xPre-1; X<xPre+3; X++){
                        if(X>-1 && X<decomposedGridImage->nx){
                            xControlPointCoordinates[coord] = (PrecisionTYPE)xPtr[X]*(PrecisionTYPE)ratio;
                            yControlPointCoordinates[coord] = (PrecisionTYPE)yPtr[X]*(PrecisionTYPE)ratio;
                        }
                        coord++;
                    }
                }
                else coord+=4;
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
            *outCPPPtrX++ += (PrecisionTYPE)xReal;
            *outCPPPtrY++ += (PrecisionTYPE)yReal;
        }
    }
    return;
}
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_spline_cppComposition_3D(  nifti_image *positionGridImage,
                                    nifti_image *decomposedGridImage,
                                    float ratio,
                                    bool type)
{
#if _USE_SSE
    union u{
        __m128 m;
        float f[4];
    } val;
#endif  
    
    PrecisionTYPE *outCPPPtrX = static_cast<PrecisionTYPE *>(positionGridImage->data);
    PrecisionTYPE *outCPPPtrY = &outCPPPtrX[positionGridImage->nx*positionGridImage->ny*positionGridImage->nz];
    PrecisionTYPE *outCPPPtrZ = &outCPPPtrY[positionGridImage->nx*positionGridImage->ny*positionGridImage->nz];
    
    PrecisionTYPE *controlPointPtrX = static_cast<PrecisionTYPE *>(decomposedGridImage->data);
    PrecisionTYPE *controlPointPtrY = &controlPointPtrX[decomposedGridImage->nx*decomposedGridImage->ny*decomposedGridImage->nz];
    PrecisionTYPE *controlPointPtrZ = &controlPointPtrY[decomposedGridImage->nx*decomposedGridImage->ny*decomposedGridImage->nz];
    
    PrecisionTYPE basis, FF, FFF, MF;
    
#ifdef _WINDOWS
    __declspec(align(16)) PrecisionTYPE xBasis[4];
    __declspec(align(16)) PrecisionTYPE yBasis[4];
    __declspec(align(16)) PrecisionTYPE zBasis[4];
#if _USE_SSE
    __declspec(align(16)) PrecisionTYPE xyzBasis[64];
#endif
    
    __declspec(align(16)) PrecisionTYPE xControlPointCoordinates[64];
    __declspec(align(16)) PrecisionTYPE yControlPointCoordinates[64];
    __declspec(align(16)) PrecisionTYPE zControlPointCoordinates[64];
#else
    PrecisionTYPE xBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE yBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE zBasis[4] __attribute__((aligned(16)));
#if _USE_SSE
    PrecisionTYPE xyzBasis[64] __attribute__((aligned(16)));
#endif
    
    PrecisionTYPE xControlPointCoordinates[64] __attribute__((aligned(16)));
    PrecisionTYPE yControlPointCoordinates[64] __attribute__((aligned(16)));
    PrecisionTYPE zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif
    
    unsigned int coord;
    
    // read the xyz/ijk sform or qform, as appropriate
    mat44 *matrix_real_to_voxel=NULL;
    mat44 *matrix_voxel_to_real=NULL;
    if(decomposedGridImage->sform_code>0){
        matrix_real_to_voxel=&(decomposedGridImage->sto_ijk);
        matrix_voxel_to_real=&(decomposedGridImage->sto_xyz);
    }
    else{
        matrix_real_to_voxel=&(decomposedGridImage->qto_ijk);
        matrix_voxel_to_real=&(decomposedGridImage->qto_xyz);
    }

    for(int z=0; z<decomposedGridImage->nz; z++){
        for(int y=0; y<decomposedGridImage->ny; y++){
            for(int x=0; x<decomposedGridImage->nx; x++){
                
                // Get the initial control point position
                PrecisionTYPE xReal=0;
                PrecisionTYPE yReal=0;
                PrecisionTYPE zReal=0;
                if(type==0){ // displacement are assumed, position otherwise
                    xReal = matrix_voxel_to_real->m[0][0]*(PrecisionTYPE)x
                    + matrix_voxel_to_real->m[0][1]*(PrecisionTYPE)y
                    + matrix_voxel_to_real->m[0][2]*(PrecisionTYPE)z
                    + matrix_voxel_to_real->m[0][3];
                    yReal = matrix_voxel_to_real->m[1][0]*(PrecisionTYPE)x
                    + matrix_voxel_to_real->m[1][1]*(PrecisionTYPE)y
                    + matrix_voxel_to_real->m[1][2]*(PrecisionTYPE)z
                    + matrix_voxel_to_real->m[1][3];
                    zReal = matrix_voxel_to_real->m[2][0]*(PrecisionTYPE)x
                    + matrix_voxel_to_real->m[2][1]*(PrecisionTYPE)y
                    + matrix_voxel_to_real->m[2][2]*(PrecisionTYPE)z
                    + matrix_voxel_to_real->m[2][3];
                }
                
                // Get the control point actual position
                xReal += *outCPPPtrX;
                yReal += *outCPPPtrY;
                zReal += *outCPPPtrZ;
                
                // Get the voxel based control point position
                PrecisionTYPE xVoxel = matrix_real_to_voxel->m[0][0]*xReal
                + matrix_real_to_voxel->m[0][1]*yReal
                + matrix_real_to_voxel->m[0][2]*zReal
                + matrix_real_to_voxel->m[0][3];
                PrecisionTYPE yVoxel = matrix_real_to_voxel->m[1][0]*xReal
                + matrix_real_to_voxel->m[1][1]*yReal
                + matrix_real_to_voxel->m[1][2]*zReal
                + matrix_real_to_voxel->m[1][3];
                PrecisionTYPE zVoxel = matrix_real_to_voxel->m[2][0]*xReal
                + matrix_real_to_voxel->m[2][1]*yReal
                + matrix_real_to_voxel->m[2][2]*zReal
                + matrix_real_to_voxel->m[2][3];
                
                xVoxel = xVoxel<0.0?0.0:xVoxel;
                yVoxel = yVoxel<0.0?0.0:yVoxel;
                zVoxel = zVoxel<0.0?0.0:zVoxel;
                
                // The spline coefficients are computed
                int xPre=(int)(floor(xVoxel));
                basis=(PrecisionTYPE)xVoxel-(PrecisionTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
                xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                xBasis[3] = (PrecisionTYPE)(FFF/6.0);
                
                int yPre=(int)(floor(yVoxel));
                basis=(PrecisionTYPE)yVoxel-(PrecisionTYPE)yPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
                yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                yBasis[3] = (PrecisionTYPE)(FFF/6.0);
                
                int zPre=(int)(floor(zVoxel));
                basis=(PrecisionTYPE)zVoxel-(PrecisionTYPE)zPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                zBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
                zBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                zBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                zBasis[3] = (PrecisionTYPE)(FFF/6.0);
                
                // The control points are stored
                coord=0;
                memset(xControlPointCoordinates, 0, 64*sizeof(PrecisionTYPE));
                memset(yControlPointCoordinates, 0, 64*sizeof(PrecisionTYPE));
                memset(zControlPointCoordinates, 0, 64*sizeof(PrecisionTYPE));
                for(int Z=zPre-1; Z<zPre+3; Z++){
                    if(Z>-1 && Z<decomposedGridImage->nz){
                        int index = Z*decomposedGridImage->nx*decomposedGridImage->ny;
                        PrecisionTYPE *xPtr = &controlPointPtrX[index];
                        PrecisionTYPE *yPtr = &controlPointPtrY[index];
                        PrecisionTYPE *zPtr = &controlPointPtrZ[index];
                        for(int Y=yPre-1; Y<yPre+3; Y++){
                            if(Y>-1 && Y<decomposedGridImage->ny){
                                index = Y*decomposedGridImage->nx;
                                PrecisionTYPE *xxPtr = &xPtr[index];
                                PrecisionTYPE *yyPtr = &yPtr[index];
                                PrecisionTYPE *zzPtr = &zPtr[index];
                                for(int X=xPre-1; X<xPre+3; X++){
                                    if(X>-1 && X<decomposedGridImage->nx){
                                        xControlPointCoordinates[coord] = (PrecisionTYPE)xxPtr[X]*(PrecisionTYPE)ratio;
                                        yControlPointCoordinates[coord] = (PrecisionTYPE)yyPtr[X]*(PrecisionTYPE)ratio;
                                        zControlPointCoordinates[coord] = (PrecisionTYPE)zzPtr[X]*(PrecisionTYPE)ratio;
                                    }
                                    coord++;
                                }
                            }
                            else coord+=4;
                        }
                    }
                    else coord+=16;
                }
                
                xReal=0.0;
                yReal=0.0;
                zReal=0.0;
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
                __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];
                __m128 *ptrBasis   = (__m128 *) &xyzBasis[0];
                //addition and multiplication of the 16 basis value and CP position for each axis
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
                val.m = tempX;
                xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempY;
                yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempZ;
                zReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                coord=0;
                for(unsigned int c=0; c<4; c++){
                    for(unsigned int b=0; b<4; b++){
                        for(unsigned int a=0; a<4; a++){
                            PrecisionTYPE tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                            xReal += xControlPointCoordinates[coord] * tempValue;
                            yReal += yControlPointCoordinates[coord] * tempValue;
                            zReal += zControlPointCoordinates[coord] * tempValue;
                            coord++;
                        }
                    }
                }
#endif
                *outCPPPtrX++ += (PrecisionTYPE)xReal;
                *outCPPPtrY++ += (PrecisionTYPE)yReal;
                *outCPPPtrZ++ += (PrecisionTYPE)zReal;
            }
        }
    }
    return;
}
/* *************************************************************** */
int reg_spline_cppComposition(  nifti_image *positionGridImage,
                                nifti_image *decomposedGridImage,
                                float ratio,
                                bool type)
{
    if(positionGridImage->datatype != decomposedGridImage->datatype){
        fprintf(stderr,"ERROR:\treg_square_cpp\n");
        fprintf(stderr,"ERROR:\tInput and output image do not have the same data type\n");
        return 1;       
    }
    
    if(positionGridImage->nz>1){
        switch(positionGridImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_spline_cppComposition_3D<float>(positionGridImage, decomposedGridImage, ratio, type);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_spline_cppComposition_3D<double>(positionGridImage, decomposedGridImage, ratio, type);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_spline_cppComposition 3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for single or double floating images\n");
                return 1;
        }
    }
    else{
        switch(positionGridImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_spline_cppComposition_2D<float>(positionGridImage, decomposedGridImage, ratio, type);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_spline_cppComposition_2D<double>(positionGridImage, decomposedGridImage, ratio, type);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_spline_cppComposition 2D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for single or double floating images\n");
                return 1;
        }
    }
    return 0;
}
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
template int reg_getDisplacementFromPosition<double>(nifti_image *);
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
template int reg_getPositionFromDisplacement<double>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void extractLine(int start, int end, int increment,const ImageTYPE *image, double *values)
{
    unsigned int index = 0;
    for(int i=start; i<end; i+=increment)
        values[index++] = (double)image[i];
}
/* *************************************************************** */
template <class ImageTYPE>
void restoreLine(int start, int end, int increment, ImageTYPE *image, const double *values)
{
    unsigned int index = 0;
    for(int i=start; i<end; i+=increment)
        image[i] = (ImageTYPE)values[index++];
}
/* *************************************************************** */
void intensitiesToSplineCoefficients(double *values, int number, double pole)
{
    // Border are set to zero
    double currentPole = pole;
    double currentOpposite = pow(pole,(double)(2.0*(double)number-1.0));
    double sum=0.0;
    for(short i=1; i<number; i++){
        sum += (currentPole - currentOpposite) * values[i];
        currentPole *= pole;
        currentOpposite /= pole;
    }
    values[0] = (double)((values[0] - pole*pole*(values[0] + sum)) / (1.0 - pow(pole,(double)(2.0*(double)number+2.0))));
    
    //other values forward
    for(int i=1; i<number; i++){
        values[i] += pole * values[i-1];
    }
    
    double ipp=(double)(1.0-pole); ipp*=ipp;
    
    //last value
    values[number-1] = ipp * values[number-1];
    
    //other values backward
    for(int i=number-2; 0<=i; i--){
        values[i] = pole * values[i+1] + ipp*values[i];
    }
    return;
}
/* *************************************************************** */
template <class ImageTYPE>
int reg_spline_Interpolant2Interpolator_2D(nifti_image *inputImage,
                                           nifti_image *outputImage)
{
    /* in order to apply a cubic Spline resampling, the source image
     intensities have to be decomposed */
    ImageTYPE *inputPtrX = static_cast<ImageTYPE *>(inputImage->data);
    ImageTYPE *inputPtrY = &inputPtrX[inputImage->nx*inputImage->ny];
    
    ImageTYPE *outputPtrX = static_cast<ImageTYPE *>(outputImage->data);
    ImageTYPE *outputPtrY = &outputPtrX[outputImage->nx*outputImage->ny];
    
    double pole = (double)(sqrt(3.0) - 2.0);
    int increment;
    double *values=NULL;
    
    // X axis first
    values=new double[inputImage->nx];
    increment = 1;
    for(int i=0;i<inputImage->ny;i++){
        int start = i*inputImage->nx;
        int end =  start + inputImage->nx;
        
        extractLine(start,end,increment,inputPtrX,values);
        intensitiesToSplineCoefficients(values, inputImage->nx, pole);
        restoreLine(start,end,increment,outputPtrX,values);
        
        extractLine(start,end,increment,inputPtrY,values);
        intensitiesToSplineCoefficients(values, inputImage->nx, pole);
        restoreLine(start,end,increment,outputPtrY,values);
    }
    delete[] values;

    // Y axis then
    values=new double[inputImage->ny];
    increment = inputImage->nx;
    for(int i=0;i<inputImage->nx;i++){
        int start = i + i/inputImage->nx * inputImage->nx * (inputImage->ny - 1);
        int end =  start + inputImage->nx*inputImage->ny;
        
        extractLine(start,end,increment,outputPtrX,values);
        intensitiesToSplineCoefficients(values, inputImage->ny, pole);
        restoreLine(start,end,increment,outputPtrX,values);
        
        extractLine(start,end,increment,outputPtrY,values);
        intensitiesToSplineCoefficients(values, inputImage->ny, pole);
        restoreLine(start,end,increment,outputPtrY,values);
    }
    delete[] values;
    
    return 0;
}
/* *************************************************************** */
template <class ImageTYPE>
int reg_spline_Interpolant2Interpolator_3D(nifti_image *inputImage,
                                           nifti_image *outputImage)
{
    /* in order to apply a cubic Spline resampling, the source image
     intensities have to be decomposed */
    ImageTYPE *inputPtrX = static_cast<ImageTYPE *>(inputImage->data);
    ImageTYPE *inputPtrY = &inputPtrX[inputImage->nx*inputImage->ny*inputImage->nz];
    ImageTYPE *inputPtrZ = &inputPtrY[inputImage->nx*inputImage->ny*inputImage->nz];
    
    ImageTYPE *outputPtrX = static_cast<ImageTYPE *>(outputImage->data);
    ImageTYPE *outputPtrY = &outputPtrX[outputImage->nx*outputImage->ny*outputImage->nz];
    ImageTYPE *outputPtrZ = &outputPtrY[outputImage->nx*outputImage->ny*outputImage->nz];
    
    double pole = (double)(sqrt(3.0) - 2.0);
    int increment;
    double *values=NULL;
    
    // X axis first
    values=new double[inputImage->nx];
    increment = 1;
    for(int i=0;i<inputImage->ny*inputImage->nz;i++){
        int start = i*inputImage->nx;
        int end =  start + inputImage->nx;
        
        extractLine(start,end,increment,inputPtrX,values);
        intensitiesToSplineCoefficients(values, inputImage->nx, pole);
        restoreLine(start,end,increment,outputPtrX,values);
        
        extractLine(start,end,increment,inputPtrY,values);
        intensitiesToSplineCoefficients(values, inputImage->nx, pole);
        restoreLine(start,end,increment,outputPtrY,values);
        
        extractLine(start,end,increment,inputPtrZ,values);
        intensitiesToSplineCoefficients(values, inputImage->nx, pole);
        restoreLine(start,end,increment,outputPtrZ,values);
    }
    delete[] values;
    // Y axis
    values=new double[inputImage->ny];
    increment = inputImage->nx;
    for(int i=0;i<inputImage->nx*inputImage->nz;i++){
        int start = i + i/inputImage->nx * inputImage->nx * (inputImage->ny - 1);
        int end =  start + inputImage->nx*inputImage->ny;
        
        extractLine(start,end,increment,outputPtrX,values);
        intensitiesToSplineCoefficients(values, inputImage->ny, pole);
        restoreLine(start,end,increment,outputPtrX,values);
        
        extractLine(start,end,increment,outputPtrY,values);
        intensitiesToSplineCoefficients(values, inputImage->ny, pole);
        restoreLine(start,end,increment,outputPtrY,values);
        
        extractLine(start,end,increment,outputPtrZ,values);
        intensitiesToSplineCoefficients(values, inputImage->ny, pole);
        restoreLine(start,end,increment,outputPtrZ,values);
    }
    delete[] values;
    // Z axis
    values=new double[inputImage->nz];
    increment = inputImage->nx*inputImage->ny;
    for(int i=0;i<inputImage->nx*inputImage->ny;i++){
        int start = i;
        int end =  start + inputImage->nx*inputImage->ny*inputImage->nz;
        
        extractLine(start,end,increment,outputPtrX,values);
        intensitiesToSplineCoefficients(values, inputImage->nz, pole);
        restoreLine(start,end,increment,outputPtrX,values);
        
        extractLine(start,end,increment,outputPtrY,values);
        intensitiesToSplineCoefficients(values, inputImage->nz, pole);
        restoreLine(start,end,increment,outputPtrY,values);
        
        extractLine(start,end,increment,outputPtrZ,values);
        intensitiesToSplineCoefficients(values, inputImage->nz, pole);
        restoreLine(start,end,increment,outputPtrZ,values);
    }
    delete[] values;
    return 0;
}
/* *************************************************************** */
int reg_spline_Interpolant2Interpolator(nifti_image *inputImage,
                                        nifti_image *outputImage)
{
    if(inputImage->datatype != outputImage->datatype){
        fprintf(stderr,"ERROR:\treg_spline_Interpolant2Interpolator\n");
        fprintf(stderr,"ERROR:\tInput and output image do not have the same data type\n");
        return 1;       
    }
    
    if(inputImage->nz>1){
        switch(inputImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_spline_Interpolant2Interpolator_3D<float>(inputImage, outputImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_spline_Interpolant2Interpolator_3D<double>(inputImage, outputImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_spline_Interpolant2Interpolator\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                return 1;
                break;
        }
    }
    else{
        switch(inputImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_spline_Interpolant2Interpolator_2D<float>(inputImage, outputImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_spline_Interpolant2Interpolator_2D<double>(inputImage, outputImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_spline_Interpolant2Interpolator\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                return 1;
                break;
        }
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_3D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage)
{
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "***ERROR***\treg_bspline_GetJacobianMapFromVelocityField_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    // The jacobian map is initialise to 1 everywhere
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++) jacPtr[i]=(ImageTYPE)1.0;

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);

    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE zBasis[4],zFirst[4],temp[4],first[4];
    ImageTYPE tempX[16], tempY[16], tempZ[16];
    ImageTYPE basisX[64], basisY[64], basisZ[64];
    ImageTYPE basis, FF, FFF, MF, oldBasis=(ImageTYPE)(1.1);

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
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

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
        ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

        unsigned int jacIndex=0;
        for(int z=0; z<jacobianImage->nz; z++){

            int zPre=(int)((ImageTYPE)z/gridVoxelSpacing[2]);
            basis=(ImageTYPE)z/gridVoxelSpacing[2]-(ImageTYPE)zPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(ImageTYPE)(1.0-basis);
            zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
            zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            zBasis[3] = (ImageTYPE)(FFF/6.0);
            zFirst[3] = (ImageTYPE)(FF / 2.0);
            zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
            zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
            zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

            for(int y=0; y<jacobianImage->ny; y++){

                int yPre=(int)((ImageTYPE)y/gridVoxelSpacing[1]);
                basis=(ImageTYPE)y/gridVoxelSpacing[1]-(ImageTYPE)yPre;
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
                            ImageTYPE *xPtr = &controlPointPtrX[index];
                            ImageTYPE *yPtr = &controlPointPtrY[index];
                            ImageTYPE *zPtr = &controlPointPtrZ[index];
                            for(int Y=yPre; Y<yPre+4; Y++){
                                index = Y*splineControlPoint->nx;
                                ImageTYPE *xxPtr = &xPtr[index];
                                ImageTYPE *yyPtr = &yPtr[index];
                                ImageTYPE *zzPtr = &zPtr[index];
                                for(int X=xPre; X<xPre+4; X++){
                                    xControlPointCoordinates[coord] = (ImageTYPE)xxPtr[X];
                                    yControlPointCoordinates[coord] = (ImageTYPE)yyPtr[X];
                                    zControlPointCoordinates[coord] = (ImageTYPE)zzPtr[X];
                                    coord++;
                                }
                            }
                        }
                    }
                    oldBasis=basis;

                    ImageTYPE Tx_x=0.0;
                    ImageTYPE Ty_x=0.0;
                    ImageTYPE Tz_x=0.0;
                    ImageTYPE Tx_y=0.0;
                    ImageTYPE Ty_y=0.0;
                    ImageTYPE Tz_y=0.0;
                    ImageTYPE Tx_z=0.0;
                    ImageTYPE Ty_z=0.0;
                    ImageTYPE Tz_z=0.0;
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
                    jacobianMatrix.m[0][0]= Tx_x / splineControlPoint->dx;
                    jacobianMatrix.m[0][1]= Tx_y / splineControlPoint->dy;
                    jacobianMatrix.m[0][2]= Tx_z / splineControlPoint->dz;
                    jacobianMatrix.m[1][0]= Ty_x / splineControlPoint->dx;
                    jacobianMatrix.m[1][1]= Ty_y / splineControlPoint->dy;
                    jacobianMatrix.m[1][2]= Ty_z / splineControlPoint->dz;
                    jacobianMatrix.m[2][0]= Tz_x / splineControlPoint->dx;
                    jacobianMatrix.m[2][1]= Tz_y / splineControlPoint->dy;
                    jacobianMatrix.m[2][2]= Tz_z / splineControlPoint->dz;

                    jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                    ImageTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                    jacPtr[jacIndex++] *= detJac;
                }
            }
        }
        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    }

    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);
}
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_2D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage)
{
    // The jacobian map is initialise to 1 everywhere
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++) jacPtr[i]=(ImageTYPE)1.0;

    // Two control point image is allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE yBasis[4],yFirst[4],temp[4],first[4];
    ImageTYPE basisX[16], basisY[16];
    ImageTYPE basis, FF, FFF, MF, oldBasis=(ImageTYPE)(1.1);

    ImageTYPE xControlPointCoordinates[16];
    ImageTYPE yControlPointCoordinates[16];

    ImageTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);
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

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

        unsigned int jacIndex=0;

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
            yFirst[3]= (ImageTYPE)(FF / 2.0);
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
                        unsigned int index = Y*splineControlPoint->nx;
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

                jacobianMatrix.m[0][0]= Tx_x / splineControlPoint->dx;
                jacobianMatrix.m[0][1]= Tx_y / splineControlPoint->dy;
                jacobianMatrix.m[0][2]= 0.0;
                jacobianMatrix.m[1][0]= Ty_x / splineControlPoint->dx;
                jacobianMatrix.m[1][1]= Ty_y / splineControlPoint->dy;
                jacobianMatrix.m[1][2]= 0.0;
                jacobianMatrix.m[2][0]= 0.0;
                jacobianMatrix.m[2][1]= 0.0;
                jacobianMatrix.m[2][2]= 1.0;

                jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                ImageTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                jacPtr[jacIndex++] *= detJac;
            }
        }
        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    }
    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);
}
/* *************************************************************** */
int reg_bspline_GetJacobianMapFromVelocityField(nifti_image* velocityFieldImage,
                                                nifti_image* jacobianImage)
{
    if(velocityFieldImage->datatype != jacobianImage->datatype){
        fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
        fprintf(stderr,"ERROR:\tInput and output image do not have the same data type\n");
        return 1;       
    }
    
    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                return 1;
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField_2D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                return 1;
                break;
        }
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
double reg_bspline_GetJacobianValueFromVelocityField(   nifti_image* velocityFieldImage,
                                                        nifti_image* resultImage,
                                                        bool approx)
{
    if(approx!=0){
            fprintf(stderr,"ERROR:\treg_bspline_GetJacobianValueFromVelocityField\n");
            fprintf(stderr,"ERROR:\tThe approximated version has not been implemented yet\n");
            exit(1);
    }

    // An image to contain the Jacobian map is allocated
    nifti_image *jacobianImage = nifti_copy_nim_info(resultImage);
    jacobianImage->datatype = velocityFieldImage->datatype;
    switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                jacobianImage->nbyper = sizeof(float);
                break;
            case NIFTI_TYPE_FLOAT64:
                jacobianImage->nbyper = sizeof(double);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
    }
    jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField_2D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    // The value in the array are then integrated
    double jacobianNormalisedSum=0.0;
    float *singlePtr=NULL;
    double *doublePtr=NULL;
    switch(velocityFieldImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            singlePtr = static_cast<float *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                float temp = logf(singlePtr[i]);
                if(temp!=temp) return std::numeric_limits<float>::quiet_NaN();
                temp *= temp;
                jacobianNormalisedSum += (double)temp;
            }
            break;
        case NIFTI_TYPE_FLOAT64:
            doublePtr = static_cast<double *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                double temp = logf(doublePtr[i]);
                if(temp!=temp) return std::numeric_limits<double>::quiet_NaN();
                temp *= temp;
                jacobianNormalisedSum += temp;
            }
            break;
    }
    nifti_image_free(jacobianImage);

    return jacobianNormalisedSum = jacobianNormalisedSum / (double)resultImage->nvox;

}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bsplineComp_correctFolding2D(nifti_image *velocityFieldImage,
                                    nifti_image *targetImage)
{
    fprintf(stderr, "reg_bsplineComp_correctFolding2D needs to be implemented. Exit ...\n");
    exit(1);
}
/* *************************************************************** */
template <class ImageTYPE>
void reg_bsplineComp_correctFolding3D(  nifti_image *velocityFieldImage,
                                        nifti_image *jacobianImage)
{
    ImageTYPE *jacobianImagePtr=static_cast<ImageTYPE *>(jacobianImage->data);

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = velocityFieldImage->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = velocityFieldImage->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = velocityFieldImage->dz / jacobianImage->dz;

    ImageTYPE *velocityFieldImagePtrX = static_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *velocityFieldImagePtrY = &velocityFieldImagePtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    ImageTYPE *velocityFieldImagePtrZ = &velocityFieldImagePtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    for(unsigned int z=0; z<jacobianImage->nz; z++){
        int zPre=(int)((ImageTYPE)z/gridVoxelSpacing[2]);
        for(unsigned int y=0; y<jacobianImage->ny; y++){
            int yPre=(int)((ImageTYPE)y/gridVoxelSpacing[1]);
            for(unsigned int x=0; x<jacobianImage->nx; x++){
                int xPre=(int)((ImageTYPE)x/gridVoxelSpacing[0]);

                ImageTYPE detJac = jacobianImagePtr[(z*jacobianImage->ny+y)*jacobianImage->nx+x];

                if(detJac<0.1){
                    for(int c=0;c<2;c++){
                        const unsigned int three=zPre+1+c;
                        for(int b=0;b<2;b++){
                            const unsigned int two=yPre+1+b;
                            for(int a=0;a<2;a++){
                                const unsigned int one=xPre+1+a;
                                const unsigned int controlPointCoord[7][3] =
                                {{one,two,three},
                                    {one-1,two,three},
                                    {one+1,two,three},
                                    {one,two-1,three},
                                    {one,two+1,three},
                                    {one,two,three-1},
                                    {one,two,three+1}};
                                unsigned int controlPointIndex[7];
                                ImageTYPE position[7][3];
                                for(unsigned int i=0;i<7;i++){
                                    controlPointIndex[i] = (controlPointCoord[i][2]*velocityFieldImage->ny+controlPointCoord[i][1])
                                        * velocityFieldImage->nx+controlPointCoord[i][0];
                                    // The position are extracted
                                    position[i][0] = velocityFieldImagePtrX[controlPointIndex[i]];
                                    position[i][1] = velocityFieldImagePtrY[controlPointIndex[i]];
                                    position[i][2] = velocityFieldImagePtrZ[controlPointIndex[i]];
                                }
                                velocityFieldImagePtrX[controlPointIndex[0]]= (position[2][0] + position[1][0])/2.0;
                                velocityFieldImagePtrY[controlPointIndex[0]]= (position[4][1] + position[3][1])/2.0;
                                velocityFieldImagePtrZ[controlPointIndex[0]]= (position[6][2] + position[5][2])/2.0;
                            }
                        }
                    }
                    x+=int(fabs(gridVoxelSpacing[0]-1));
                }
            }
        }
    }
}
/* *************************************************************** */
void reg_bsplineComp_correctFolding(nifti_image *velocityFieldImage,
                                    nifti_image *resultImage)
{
    // A Jacobian map image is first computed using the squaring approach
    nifti_image *jacobianImage = nifti_copy_nim_info(resultImage);
    jacobianImage->datatype = velocityFieldImage->datatype;
    switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                jacobianImage->nbyper = sizeof(float);
                break;
            case NIFTI_TYPE_FLOAT64:
                jacobianImage->nbyper = sizeof(double);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
    }
    jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField_2D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }

    // The previously computed Jacobian map is now used to assess the folded area
    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bsplineComp_correctFolding3D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bsplineComp_correctFolding3D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bsplineComp_correctFolding3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bsplineComp_correctFolding2D<float>(velocityFieldImage, jacobianImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bsplineComp_correctFolding2D<double>(velocityFieldImage, jacobianImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bsplineComp_correctFolding2D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    nifti_image_free(jacobianImage);
    return;
}
/* *************************************************************** */
/* *************************************************************** */
#endif
