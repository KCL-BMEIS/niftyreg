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

/* *************************************************************** */
/* *************************************************************** */

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
            
            xVoxel = xVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:xVoxel;
            yVoxel = yVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:yVoxel;
            
            // The spline coefficients are computed
            int xPre=(int)(floor(xVoxel));
            basis=(PrecisionTYPE)xVoxel-(PrecisionTYPE)xPre;
            if(basis<(PrecisionTYPE)0.0) basis=(PrecisionTYPE)0.0; //rounding error
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
    __declspec(align(16)) PrecisionTYPE xControlPointCoordinates[64];
    __declspec(align(16)) PrecisionTYPE yControlPointCoordinates[64];
    __declspec(align(16)) PrecisionTYPE zControlPointCoordinates[64];
#else
    PrecisionTYPE xBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE yBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE zBasis[4] __attribute__((aligned(16)));
    PrecisionTYPE xControlPointCoordinates[64] __attribute__((aligned(16)));
    PrecisionTYPE yControlPointCoordinates[64] __attribute__((aligned(16)));
    PrecisionTYPE zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif
    
    unsigned int coord;
    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;
    
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

                xVoxel = xVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:xVoxel;
                yVoxel = yVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:yVoxel;
                zVoxel = zVoxel<(PrecisionTYPE)0.0?(PrecisionTYPE)0.0:zVoxel;

                // The spline coefficients are computed
                xPre=(int)(floor(xVoxel));
                basis=(PrecisionTYPE)xVoxel-(PrecisionTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                xBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
                xBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                xBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                xBasis[3] = (PrecisionTYPE)(FFF/6.0);

                yPre=(int)(floor(yVoxel));
                basis=(PrecisionTYPE)yVoxel-(PrecisionTYPE)yPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(PrecisionTYPE)(1.0-basis);
                yBasis[0] = (PrecisionTYPE)((MF)*(MF)*(MF)/(6.0));
                yBasis[1] = (PrecisionTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                yBasis[2] = (PrecisionTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                yBasis[3] = (PrecisionTYPE)(FFF/6.0);

                zPre=(int)(floor(zVoxel));
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
                if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
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
                    xPreOld=xPre;
                    yPreOld=yPre;
                    zPreOld=zPre;
                }

                xReal=0.0;
                yReal=0.0;
                zReal=0.0;
#if _USE_SSE
                val.f[0] = xBasis[0];
                val.f[1] = xBasis[1];
                val.f[2] = xBasis[2];
                val.f[3] = xBasis[3];
                __m128 _xBasis_sse = val.m;

                __m128 tempX =  _mm_set_ps1(0.0);
                __m128 tempY =  _mm_set_ps1(0.0);
                __m128 tempZ =  _mm_set_ps1(0.0);
                __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                for(unsigned int c=0; c<4; c++){
                    for(unsigned int b=0; b<4; b++){
                        __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                        __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                        __m128 _temp_basis   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                        __m128 _basis       = _mm_mul_ps(_temp_basis, _xBasis_sse);
                        tempX = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), tempX );
                        tempY = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), tempY );
                        tempZ = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), tempZ );
                        ptrX++;
                        ptrY++;
                        ptrZ++;
                    }
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
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetApproxJacobianMapFromVelocityField_2D(  nifti_image* velocityFieldImage,
                                                            nifti_image* jacobianImage)
{
    // The jacobian map is initialise to 1 everywhere
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++) jacPtr[i]=(ImageTYPE)1.0;

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4];
    ImageTYPE basisX[16], basisY[16];
    ImageTYPE basis, FF, FFF, MF, oldBasis=(ImageTYPE)(1.1);

    ImageTYPE xControlPointCoordinates[16];
    ImageTYPE yControlPointCoordinates[16];

    int xPre, yPre, oldXPre=-1, oldYPre=-1;

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

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX=(ImageTYPE *)malloc(jacobianImage->nvox);
    ImageTYPE *deformationFieldArrayY=(ImageTYPE *)malloc(jacobianImage->nvox);
    // The initial deformation field is initialised with the nifti header
    mat44* jac_xyz_matrix = NULL;
    if(jacobianImage->sform_code>0)
        jac_xyz_matrix= &(jacobianImage->sto_xyz);
    else jac_xyz_matrix= &(jacobianImage->qto_xyz);
    unsigned int jacIndex = 0;
    for(int y=0; y>jacobianImage->ny; y++){
        for(int x=0; x>jacobianImage->nx; x++){
            deformationFieldArrayX[jacIndex] = jac_xyz_matrix->m[0][0]*x
                + jac_xyz_matrix->m[0][1]*y + jac_xyz_matrix->m[0][3];
            deformationFieldArrayY[jacIndex] = jac_xyz_matrix->m[1][0]*x
                + jac_xyz_matrix->m[1][1]*y + jac_xyz_matrix->m[1][3];
            jacIndex++;
        }
    }

    /* The real to voxel matrix will be used */
    mat44* jac_ijk_matrix = NULL;
    if(jacobianImage->sform_code>0)
        jac_ijk_matrix= &(jacobianImage->sto_ijk);
    else jac_ijk_matrix= &(jacobianImage->qto_ijk);

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

        jacIndex=0;

        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){

                ImageTYPE realPosition[2];
                realPosition[0]=deformationFieldArrayX[jacIndex];
                realPosition[1]=deformationFieldArrayY[jacIndex];

                ImageTYPE voxelPosition[2];
                voxelPosition[0] = jac_ijk_matrix->m[0][0]*realPosition[0]
                    + jac_ijk_matrix->m[0][1]*realPosition[1] + jac_ijk_matrix->m[0][3];
                voxelPosition[1] = jac_ijk_matrix->m[1][0]*realPosition[0]
                    + jac_ijk_matrix->m[1][1]*realPosition[1] + jac_ijk_matrix->m[1][3];

                xPre=x-1;
                yPre=y-1;

                ImageTYPE detJac = 1.0f;

                if( xPre>-1 && (xPre+4)<splineControlPoint->nx &&
                    yPre>-1 && (yPre+4)<splineControlPoint->ny){

                    basis=(ImageTYPE)voxelPosition[0]-(ImageTYPE)(xPre+1);
                    if(basis<0.0) basis=0.0; //rounding error
                    FF= basis*basis;
                    FFF= FF*basis;
                    MF=(ImageTYPE)(1.0-basis);
                    xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                    xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                    xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                    xBasis[3] = (ImageTYPE)(FFF/6.0);
                    xFirst[3]= (ImageTYPE)(FF / 2.0);
                    xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                    xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                    xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                    basis=(ImageTYPE)voxelPosition[1]-(ImageTYPE)(yPre+1);
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

                    coord=0;
                    for(int b=0; b<4; b++){
                        for(int a=0; a<4; a++){
                            basisX[coord]=yBasis[b]*xFirst[a];   // y * x'
                            basisY[coord]=yFirst[b]*xBasis[a];    // y'* x
                            coord++;
                        }
                    }
                    if(xPre != oldXPre || yPre != oldYPre){
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
                        oldXPre=xPre;
                        oldYPre=yPre;
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

                    jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                    jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                    jacobianMatrix.m[0][2]= 0.0f;
                    jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                    jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                    jacobianMatrix.m[1][2]= 0.0f;
                    jacobianMatrix.m[2][0]= 0.0f;
                    jacobianMatrix.m[2][1]= 0.0f;
                    jacobianMatrix.m[2][2]= 1.0f;

                    jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                    detJac = nifti_mat33_determ(jacobianMatrix);
                }

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
template <class ImageTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_2D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage)
{
    // The jacobian map is initialise to 1 everywhere
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++) jacPtr[i]=(ImageTYPE)1.0;

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4];
    ImageTYPE basisX[16], basisY[16];
    ImageTYPE basis, FF, FFF, MF, oldBasis=(ImageTYPE)(1.1);

    ImageTYPE xControlPointCoordinates[16];
    ImageTYPE yControlPointCoordinates[16];

    int xPre, yPre, oldXPre=-1, oldYPre=-1;

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

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX=(ImageTYPE *)malloc(jacobianImage->nvox);
    ImageTYPE *deformationFieldArrayY=(ImageTYPE *)malloc(jacobianImage->nvox);
    // The initial deformation field is initialised with the nifti header
    mat44* jac_xyz_matrix = NULL;
    if(jacobianImage->sform_code>0)
        jac_xyz_matrix= &(jacobianImage->sto_xyz);
    else jac_xyz_matrix= &(jacobianImage->qto_xyz);
    unsigned int jacIndex = 0;
    for(int y=0; y>jacobianImage->ny; y++){
        for(int x=0; x>jacobianImage->nx; x++){
            deformationFieldArrayX[jacIndex] = jac_xyz_matrix->m[0][0]*x
                + jac_xyz_matrix->m[0][1]*y + jac_xyz_matrix->m[0][3];
            deformationFieldArrayY[jacIndex] = jac_xyz_matrix->m[1][0]*x
                + jac_xyz_matrix->m[1][1]*y + jac_xyz_matrix->m[1][3];
            jacIndex++;
        }
    }

    /* The real to voxel matrix will be used */
    mat44* jac_ijk_matrix = NULL;
    if(jacobianImage->sform_code>0)
        jac_ijk_matrix= &(jacobianImage->sto_ijk);
    else jac_ijk_matrix= &(jacobianImage->qto_ijk);

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

        jacIndex=0;

        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){

                ImageTYPE realPosition[2];
                realPosition[0]=deformationFieldArrayX[jacIndex];
                realPosition[1]=deformationFieldArrayY[jacIndex];

                ImageTYPE voxelPosition[2];
                voxelPosition[0] = jac_ijk_matrix->m[0][0]*realPosition[0]
                    + jac_ijk_matrix->m[0][1]*realPosition[1] + jac_ijk_matrix->m[0][3];
                voxelPosition[1] = jac_ijk_matrix->m[1][0]*realPosition[0]
                    + jac_ijk_matrix->m[1][1]*realPosition[1] + jac_ijk_matrix->m[1][3];

                xPre=(int)((ImageTYPE)x/gridVoxelSpacing[0]);
                yPre=(int)((ImageTYPE)y/gridVoxelSpacing[1]);

                ImageTYPE detJac = 1.0f;

                if( xPre>-1 && (xPre+4)<splineControlPoint->nx &&
                    yPre>-1 && (yPre+4)<splineControlPoint->ny){

                    basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)xPre;
                    if(basis<0.0) basis=0.0; //rounding error
                    FF= basis*basis;
                    FFF= FF*basis;
                    MF=(ImageTYPE)(1.0-basis);
                    xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                    xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                    xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                    xBasis[3] = (ImageTYPE)(FFF/6.0);
                    xFirst[3]= (ImageTYPE)(FF / 2.0);
                    xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                    xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                    xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                    basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)yPre;
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

                    coord=0;
                    for(int b=0; b<4; b++){
                        for(int a=0; a<4; a++){
                            basisX[coord]=yBasis[b]*xFirst[a];   // y * x'
                            basisY[coord]=yFirst[b]*xBasis[a];    // y'* x
                            coord++;
                        }
                    }
                    if(xPre != oldXPre || yPre != oldYPre){
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
                        oldXPre=xPre;
                        oldYPre=yPre;
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

                    jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                    jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                    jacobianMatrix.m[0][2]= 0.0f;
                    jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                    jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                    jacobianMatrix.m[1][2]= 0.0f;
                    jacobianMatrix.m[2][0]= 0.0f;
                    jacobianMatrix.m[2][1]= 0.0f;
                    jacobianMatrix.m[2][2]= 1.0f;

                    jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                    detJac = nifti_mat33_determ(jacobianMatrix);
                }

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
template <class ImageTYPE>
void reg_bspline_GetApproxJacobianMapFromVelocityField_3D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage)
{
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "***ERROR***\treg_bspline_GetApproxJacobianMapFromVelocityField_3D\n");
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

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis, FF, FFF, MF;

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

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

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayZ = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));

    /* The real to voxel matrix will be used */
    mat44 *jac_ijk_matrix = NULL;
    mat44 *jac_xyz_matrix = NULL;
    if(jacobianImage->sform_code>0){
        jac_ijk_matrix= &(jacobianImage->sto_ijk);
        jac_xyz_matrix= &(jacobianImage->sto_xyz);
    }
    else{
        jac_ijk_matrix= &(jacobianImage->qto_ijk);
        jac_xyz_matrix= &(jacobianImage->qto_xyz);
    }
#if USE_SSE
    val.f[0] = jac_ijk_matrix->m[0][0];
    val.f[1] = jac_ijk_matrix->m[0][1];
    val.f[2] = jac_ijk_matrix->m[0][2];
    val.f[3] = jac_ijk_matrix->m[0][3];
    __m128 _jac_ijk_matrix_sse_x = val.m;
    val.f[0] = jac_ijk_matrix->m[1][0];
    val.f[1] = jac_ijk_matrix->m[1][1];
    val.f[2] = jac_ijk_matrix->m[1][2];
    val.f[3] = jac_ijk_matrix->m[1][3];
    __m128 _jac_ijk_matrix_sse_y = val.m;
    val.f[0] = jac_ijk_matrix->m[2][0];
    val.f[1] = jac_ijk_matrix->m[2][1];
    val.f[2] = jac_ijk_matrix->m[2][2];
    val.f[3] = jac_ijk_matrix->m[2][3];
    __m128 _jac_ijk_matrix_sse_z = val.m;
#endif

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int z=0; z<jacobianImage->nz; z++){
        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*x
                    + jac_xyz_matrix->m[0][1]*y
                    + jac_xyz_matrix->m[0][2]*z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*x
                    + jac_xyz_matrix->m[1][1]*y
                    + jac_xyz_matrix->m[1][2]*z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*x
                    + jac_xyz_matrix->m[2][1]*y
                    + jac_xyz_matrix->m[2][2]*z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }
    unsigned int coord=0;

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
        ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

        jacIndex=0;
        for(int z=0; z<jacobianImage->nz; z++){
            for(int y=0; y<jacobianImage->ny; y++){
                for(int x=0; x<jacobianImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if USE_SSE
                    val.f[0] = realPosition[0];
                    val.f[1] = realPosition[1];
                    val.f[2] = realPosition[2];
                    val.f[3] = 1.0;
                    __m128 _realPosition_sse = val.m;
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_x,_realPosition_sse);
                    voxelPosition[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_y,_realPosition_sse);
                    voxelPosition[1]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_z,_realPosition_sse);
                    voxelPosition[2]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    voxelPosition[0]
                        = jac_ijk_matrix->m[0][0]*realPosition[0]
                        + jac_ijk_matrix->m[0][1]*realPosition[1]
                        + jac_ijk_matrix->m[0][2]*realPosition[2]
                        + jac_ijk_matrix->m[0][3];
                    voxelPosition[1]
                        = jac_ijk_matrix->m[1][0]*realPosition[0]
                        + jac_ijk_matrix->m[1][1]*realPosition[1]
                        + jac_ijk_matrix->m[1][2]*realPosition[2]
                        + jac_ijk_matrix->m[1][3];
                    voxelPosition[2]
                        = jac_ijk_matrix->m[2][0]*realPosition[0]
                        + jac_ijk_matrix->m[2][1]*realPosition[1]
                        + jac_ijk_matrix->m[2][2]*realPosition[2]
                        + jac_ijk_matrix->m[2][3];
#endif

                    xPre=(int)floor(voxelPosition[0]-1);
                    yPre=(int)floor(voxelPosition[1]-1);
                    zPre=(int)floor(voxelPosition[2]-1);

                    if( xPre>-1 && xPre<splineControlPoint->nx-3 &&
                        yPre>-1 && yPre<splineControlPoint->ny-3 &&
                        zPre>-1 && zPre<splineControlPoint->nz-3 ){

                        ImageTYPE detJac = 1.0f;

                        basis=(ImageTYPE)voxelPosition[0]-(ImageTYPE)(xPre+1);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        xBasis[3] = (ImageTYPE)(FFF/6.0);
                        xFirst[3]= (ImageTYPE)(FF / 2.0);
                        xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                        xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                        xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                        basis=(ImageTYPE)voxelPosition[1]-(ImageTYPE)(yPre+1);
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

                        basis=(ImageTYPE)voxelPosition[2]-(ImageTYPE)(zPre+1);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        zBasis[3] = (ImageTYPE)(FFF/6.0);
                        zFirst[3]= (ImageTYPE)(FF / 2.0);
                        zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
                        zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
                        zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

                        ImageTYPE Tx_x=0.0;
                        ImageTYPE Ty_x=0.0;
                        ImageTYPE Tz_x=0.0;
                        ImageTYPE Tx_y=0.0;
                        ImageTYPE Ty_y=0.0;
                        ImageTYPE Tz_y=0.0;
                        ImageTYPE Tx_z=0.0;
                        ImageTYPE Ty_z=0.0;
                        ImageTYPE Tz_z=0.0;
                        ImageTYPE newPositionX = 0.0;
                        ImageTYPE newPositionY = 0.0;
                        ImageTYPE newPositionZ = 0.0;

                        if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
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
                            xPreOld=xPre;
                            yPreOld=yPre;
                            zPreOld=zPre;
                        }

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
                        __m128 pos_x =  _mm_set_ps1(0.0);
                        __m128 pos_y =  _mm_set_ps1(0.0);
                        __m128 pos_z =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        __m128 _xBasis_sse = val.m;
                        val.f[0] = xFirst[0];
                        val.f[1] = xFirst[1];
                        val.f[2] = xFirst[2];
                        val.f[3] = xFirst[3];
                        __m128 _xFirst_sse = val.m;

                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                __m128 _yFirst_sse  = _mm_set_ps1(yFirst[b]);

                                __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                __m128 _zFirst_sse  = _mm_set_ps1(zFirst[c]);

                                __m128 _temp_sseX   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                                __m128 _temp_sseY   = _mm_mul_ps(_yFirst_sse, _zBasis_sse);
                                __m128 _temp_sseZ   = _mm_mul_ps(_yBasis_sse, _zFirst_sse);

                                __m128 _basisX      = _mm_mul_ps(_temp_sseX, _xFirst_sse);
                                __m128 _basisY      = _mm_mul_ps(_temp_sseY, _xBasis_sse);
                                __m128 _basisZ      = _mm_mul_ps(_temp_sseZ, _xBasis_sse);
                                __m128 _basis       = _mm_mul_ps(_temp_sseX, _xBasis_sse);

                                pos_x = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), pos_x );
                                tempX_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrX), tempX_x );
                                tempX_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrX), tempX_y );
                                tempX_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrX), tempX_z );

                                pos_y = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), pos_y );
                                tempY_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrY), tempY_x );
                                tempY_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrY), tempY_y );
                                tempY_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrY), tempY_z );

                                pos_z = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), pos_z );
                                tempZ_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrZ), tempZ_x );
                                tempZ_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrZ), tempZ_y );
                                tempZ_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrZ), tempZ_z );

                                ptrX++;
                                ptrY++;
                                ptrZ++;
                            }
                        }

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
                        val.m = pos_x;
                        newPositionX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_y;
                        newPositionY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_z;
                        newPositionZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        coord=0;
                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                for(int a=0; a<4; a++){
                                    ImageTYPE basisX= tempBasisX*xFirst[a];   // z * y * x'
                                    ImageTYPE basisY= tempBasisY*xBasis[a];   // z * y'* x
                                    ImageTYPE basisZ= tempBasisZ*xBasis[a];   // z'* y * x
                                    basis = tempBasisX*xBasis[a];   // z * y * x
                                    Tx_x += basisX*xControlPointCoordinates[coord];
                                    Tx_y += basisY*xControlPointCoordinates[coord];
                                    Tx_z += basisZ*xControlPointCoordinates[coord];
                                    Ty_x += basisX*yControlPointCoordinates[coord];
                                    Ty_y += basisY*yControlPointCoordinates[coord];
                                    Ty_z += basisZ*yControlPointCoordinates[coord];
                                    Tz_x += basisX*zControlPointCoordinates[coord];
                                    Tz_y += basisY*zControlPointCoordinates[coord];
                                    Tz_z += basisZ*zControlPointCoordinates[coord];
                                    newPositionX += basis*xControlPointCoordinates[coord];
                                    newPositionY += basis*yControlPointCoordinates[coord];
                                    newPositionZ += basis*zControlPointCoordinates[coord];
                                    coord++;
                                }
                            }
                        }
#endif

                        deformationFieldArrayX[jacIndex] = newPositionX;
                        deformationFieldArrayY[jacIndex] = newPositionY;
                        deformationFieldArrayZ[jacIndex] = newPositionZ;

                        jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)splineControlPoint->dz);

                        jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                        detJac = nifti_mat33_determ(jacobianMatrix);
                        jacPtr[jacIndex] *= detJac;
                    } // not in the range
                    jacIndex++;
                } // x
            } // y
        } // z
        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    } // squaring step
    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);
    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);
}
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

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis, FF, FFF, MF;

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;

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

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayZ = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));

    /* The real to voxel matrix will be used */
    mat44 *jac_ijk_matrix = NULL;
    mat44 *jac_xyz_matrix = NULL;
    if(jacobianImage->sform_code>0){
        jac_ijk_matrix= &(jacobianImage->sto_ijk);
        jac_xyz_matrix= &(jacobianImage->sto_xyz);
    }
    else{
        jac_ijk_matrix= &(jacobianImage->qto_ijk);
        jac_xyz_matrix= &(jacobianImage->qto_xyz);
    }

#if USE_SSE
    val.f[0] = jac_ijk_matrix->m[0][0];
    val.f[1] = jac_ijk_matrix->m[0][1];
    val.f[2] = jac_ijk_matrix->m[0][2];
    val.f[3] = jac_ijk_matrix->m[0][3];
    __m128 _jac_ijk_matrix_sse_x = val.m;
    val.f[0] = jac_ijk_matrix->m[1][0];
    val.f[1] = jac_ijk_matrix->m[1][1];
    val.f[2] = jac_ijk_matrix->m[1][2];
    val.f[3] = jac_ijk_matrix->m[1][3];
    __m128 _jac_ijk_matrix_sse_y = val.m;
    val.f[0] = jac_ijk_matrix->m[2][0];
    val.f[1] = jac_ijk_matrix->m[2][1];
    val.f[2] = jac_ijk_matrix->m[2][2];
    val.f[3] = jac_ijk_matrix->m[2][3];
    __m128 _jac_ijk_matrix_sse_z = val.m;
#endif

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int z=0; z<jacobianImage->nz; z++){
        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*x
                    + jac_xyz_matrix->m[0][1]*y
                    + jac_xyz_matrix->m[0][2]*z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*x
                    + jac_xyz_matrix->m[1][1]*y
                    + jac_xyz_matrix->m[1][2]*z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*x
                    + jac_xyz_matrix->m[2][1]*y
                    + jac_xyz_matrix->m[2][2]*z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }
    unsigned int coord=0;

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
        ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

        jacIndex=0;
        for(int z=0; z<jacobianImage->nz; z++){
            for(int y=0; y<jacobianImage->ny; y++){
                for(int x=0; x<jacobianImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if USE_SSE
                    val.f[0] = realPosition[0];
                    val.f[1] = realPosition[1];
                    val.f[2] = realPosition[2];
                    val.f[3] = 1.0;
                    __m128 _realPosition_sse = val.m;
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_x,_realPosition_sse);
                    voxelPosition[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_y,_realPosition_sse);
                    voxelPosition[1]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_z,_realPosition_sse);
                    voxelPosition[2]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    voxelPosition[0]
                        = jac_ijk_matrix->m[0][0]*realPosition[0]
                        + jac_ijk_matrix->m[0][1]*realPosition[1]
                        + jac_ijk_matrix->m[0][2]*realPosition[2]
                        + jac_ijk_matrix->m[0][3];
                    voxelPosition[1]
                        = jac_ijk_matrix->m[1][0]*realPosition[0]
                        + jac_ijk_matrix->m[1][1]*realPosition[1]
                        + jac_ijk_matrix->m[1][2]*realPosition[2]
                        + jac_ijk_matrix->m[1][3];
                    voxelPosition[2]
                        = jac_ijk_matrix->m[2][0]*realPosition[0]
                        + jac_ijk_matrix->m[2][1]*realPosition[1]
                        + jac_ijk_matrix->m[2][2]*realPosition[2]
                        + jac_ijk_matrix->m[2][3];
#endif
                    xPre=(int)floor((ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]);
                    yPre=(int)floor((ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]);
                    zPre=(int)floor((ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]);


                    if( xPre>-1 && xPre<splineControlPoint->nx+3 &&
                        yPre>-1 && yPre<splineControlPoint->ny+3 &&
                        zPre>-1 && zPre<splineControlPoint->nz+3 ){

                        basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        xBasis[3] = (ImageTYPE)(FFF/6.0);
                        xFirst[3]= (ImageTYPE)(FF / 2.0);
                        xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                        xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                        xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                        basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
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

                        basis=(ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]-(ImageTYPE)(zPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        zBasis[3] = (ImageTYPE)(FFF/6.0);
                        zFirst[3]= (ImageTYPE)(FF / 2.0);
                        zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
                        zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
                        zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

                        ImageTYPE Tx_x=0.0;
                        ImageTYPE Ty_x=0.0;
                        ImageTYPE Tz_x=0.0;
                        ImageTYPE Tx_y=0.0;
                        ImageTYPE Ty_y=0.0;
                        ImageTYPE Tz_y=0.0;
                        ImageTYPE Tx_z=0.0;
                        ImageTYPE Ty_z=0.0;
                        ImageTYPE Tz_z=0.0;
                        ImageTYPE newPositionX = 0.0;
                        ImageTYPE newPositionY = 0.0;
                        ImageTYPE newPositionZ = 0.0;

                        if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
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
                            xPreOld=xPre;
                            yPreOld=yPre;
                            zPreOld=zPre;
                        }

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
                        __m128 pos_x =  _mm_set_ps1(0.0);
                        __m128 pos_y =  _mm_set_ps1(0.0);
                        __m128 pos_z =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        __m128 _xBasis_sse = val.m;
                        val.f[0] = xFirst[0];
                        val.f[1] = xFirst[1];
                        val.f[2] = xFirst[2];
                        val.f[3] = xFirst[3];
                        __m128 _xFirst_sse = val.m;

                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                __m128 _yFirst_sse  = _mm_set_ps1(yFirst[b]);

                                __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                __m128 _zFirst_sse  = _mm_set_ps1(zFirst[c]);

                                __m128 _temp_sseX   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                                __m128 _temp_sseY   = _mm_mul_ps(_yFirst_sse, _zBasis_sse);
                                __m128 _temp_sseZ   = _mm_mul_ps(_yBasis_sse, _zFirst_sse);

                                __m128 _basisX      = _mm_mul_ps(_temp_sseX, _xFirst_sse);
                                __m128 _basisY      = _mm_mul_ps(_temp_sseY, _xBasis_sse);
                                __m128 _basisZ      = _mm_mul_ps(_temp_sseZ, _xBasis_sse);
                                __m128 _basis       = _mm_mul_ps(_temp_sseX, _xBasis_sse);

                                pos_x = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), pos_x );
                                tempX_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrX), tempX_x );
                                tempX_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrX), tempX_y );
                                tempX_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrX), tempX_z );

                                pos_y = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), pos_y );
                                tempY_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrY), tempY_x );
                                tempY_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrY), tempY_y );
                                tempY_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrY), tempY_z );

                                pos_z = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), pos_z );
                                tempZ_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrZ), tempZ_x );
                                tempZ_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrZ), tempZ_y );
                                tempZ_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrZ), tempZ_z );

                                ptrX++;
                                ptrY++;
                                ptrZ++;
                            }
                        }

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
                        val.m = pos_x;
                        newPositionX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_y;
                        newPositionY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_z;
                        newPositionZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        coord=0;
                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                for(int a=0; a<4; a++){
                                    ImageTYPE basisX= tempBasisX*xFirst[a];   // z * y * x'
                                    ImageTYPE basisY= tempBasisY*xBasis[a];   // z * y'* x
                                    ImageTYPE basisZ= tempBasisZ*xBasis[a];   // z'* y * x
                                    basis = tempBasisX*xBasis[a];   // z * y * x
                                    Tx_x += basisX*xControlPointCoordinates[coord];
                                    Tx_y += basisY*xControlPointCoordinates[coord];
                                    Tx_z += basisZ*xControlPointCoordinates[coord];
                                    Ty_x += basisX*yControlPointCoordinates[coord];
                                    Ty_y += basisY*yControlPointCoordinates[coord];
                                    Ty_z += basisZ*yControlPointCoordinates[coord];
                                    Tz_x += basisX*zControlPointCoordinates[coord];
                                    Tz_y += basisY*zControlPointCoordinates[coord];
                                    Tz_z += basisZ*zControlPointCoordinates[coord];
                                    newPositionX += basis*xControlPointCoordinates[coord];
                                    newPositionY += basis*yControlPointCoordinates[coord];
                                    newPositionZ += basis*zControlPointCoordinates[coord];
                                    coord++;
                                }
                            }
                        }
#endif

                        deformationFieldArrayX[jacIndex] += newPositionX;
                        deformationFieldArrayY[jacIndex] += newPositionY;
                        deformationFieldArrayZ[jacIndex] += newPositionZ;

                        jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)splineControlPoint->dz);

                        jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                        jacobianMatrix.m[0][0]++;
                        jacobianMatrix.m[1][1]++;
                        jacobianMatrix.m[2][2]++;
                        ImageTYPE detJac = nifti_mat33_determ(jacobianMatrix);
                        jacPtr[jacIndex] *= detJac;
                    } // not in the range
                    jacIndex++;
                } // x
            } // y
        } // z

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    } // squaring step
    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);
    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);

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
    // An image to contain the Jacobian map is allocated
    nifti_image *jacobianImage = NULL;
    if(!approx){
        jacobianImage = nifti_copy_nim_info(resultImage);
    }
    else{
        jacobianImage = nifti_copy_nim_info(velocityFieldImage);
        jacobianImage->dim[0]=3;
        jacobianImage->dim[5]=jacobianImage->nu=1;
        jacobianImage->nvox=jacobianImage->nx*jacobianImage->ny*jacobianImage->nz;
    }
    jacobianImage->datatype = velocityFieldImage->datatype;
    switch(jacobianImage->datatype){
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

    if(approx){
        if(velocityFieldImage->nz>1){
            switch(velocityFieldImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_GetApproxJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_GetApproxJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage);
                    break;
                default:
                    fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
                    fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                    exit(1);
                    break;
            }
        }
        else{
            switch(velocityFieldImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_GetApproxJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_GetApproxJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage);
                    break;
                default:
                    fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
                    fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                    exit(1);
                    break;
            }
        }
    }
    else{
        if(velocityFieldImage->nz>1){
            switch(velocityFieldImage->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage);
                    break;
                default:
                    fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
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
                    fprintf(stderr,"ERROR:\treg_bspline_GetJacobianMapFromVelocityField\n");
                    fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                    exit(1);
                    break;
            }
        }

    }

    // The value in the array are then integrated
    double jacobianNormalisedSum=0.0;
    switch(velocityFieldImage->datatype){
        case NIFTI_TYPE_FLOAT32:{
            float *singlePtr = static_cast<float *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                float temp = log(singlePtr[i]);
                jacobianNormalisedSum += (double)(temp*temp);
            }
            break;}
        case NIFTI_TYPE_FLOAT64:{
            double *doublePtr = static_cast<double *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                double temp = log(doublePtr[i]);
                jacobianNormalisedSum += temp*temp;
            }
            break;}
    }

    if(approx)
        jacobianNormalisedSum /= (double)((jacobianImage->nx-2)*(jacobianImage->ny-2)*(jacobianImage->nz-2));
    else jacobianNormalisedSum /= (double)jacobianImage->nvox;
    nifti_image_free(jacobianImage);
    return jacobianNormalisedSum;

}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianGradient_3D(  nifti_image *velocityFieldImage,
                                            nifti_image *targetImage,
                                            nifti_image *gradientImage,
                                            float weight)
{
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "***ERROR***\treg_bspline_GetJacobianGradient_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis, FF, FFF, MF;

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE basisX[64];
    ImageTYPE basisY[64];
    ImageTYPE basisZ[64];

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = velocityFieldImage->dx / targetImage->dx;
    gridVoxelSpacing[1] = velocityFieldImage->dy / targetImage->dy;
    gridVoxelSpacing[2] = velocityFieldImage->dz / targetImage->dz;

    mat33 desorient;
    desorient.m[0][0]=splineControlPoint->dx; desorient.m[0][1]=0.0f; desorient.m[0][2]=0.0f;
    desorient.m[1][0]=0.0f; desorient.m[1][1]=splineControlPoint->dy; desorient.m[1][2]=0.0f;
    desorient.m[2][0]=0.0f; desorient.m[2][1]=0.0f; desorient.m[2][2]=splineControlPoint->dz;
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
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, desorient));
    mat33 jacobianMatrix;

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(targetImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(targetImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayZ = (ImageTYPE *)malloc(targetImage->nvox*sizeof(ImageTYPE));

    /* The real to voxel matrix will be used */
    mat44 *jac_ijk_matrix = NULL;
    mat44 *jac_xyz_matrix = NULL;
    if(targetImage->sform_code>0){
        jac_ijk_matrix= &(targetImage->sto_ijk);
        jac_xyz_matrix= &(targetImage->sto_xyz);
    }
    else{
        jac_ijk_matrix= &(targetImage->qto_ijk);
        jac_xyz_matrix= &(targetImage->qto_xyz);
    }
#if USE_SSE
    val.f[0] = jac_ijk_matrix->m[0][0];
    val.f[1] = jac_ijk_matrix->m[0][1];
    val.f[2] = jac_ijk_matrix->m[0][2];
    val.f[3] = jac_ijk_matrix->m[0][3];
    __m128 _jac_ijk_matrix_sse_x = val.m;
    val.f[0] = jac_ijk_matrix->m[1][0];
    val.f[1] = jac_ijk_matrix->m[1][1];
    val.f[2] = jac_ijk_matrix->m[1][2];
    val.f[3] = jac_ijk_matrix->m[1][3];
    __m128 _jac_ijk_matrix_sse_y = val.m;
    val.f[0] = jac_ijk_matrix->m[2][0];
    val.f[1] = jac_ijk_matrix->m[2][1];
    val.f[2] = jac_ijk_matrix->m[2][2];
    val.f[3] = jac_ijk_matrix->m[2][3];
    __m128 _jac_ijk_matrix_sse_z = val.m;
#endif

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int z=0; z<targetImage->nz; z++){
        for(int y=0; y<targetImage->ny; y++){
            for(int x=0; x<targetImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*x
                    + jac_xyz_matrix->m[0][1]*y
                    + jac_xyz_matrix->m[0][2]*z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*x
                    + jac_xyz_matrix->m[1][1]*y
                    + jac_xyz_matrix->m[1][2]*z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*x
                    + jac_xyz_matrix->m[2][1]*y
                    + jac_xyz_matrix->m[2][2]*z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }
    unsigned int coord=0;

    ImageTYPE *gradientImagePtrX=static_cast<ImageTYPE *>(gradientImage->data);
    ImageTYPE *gradientImagePtrY=&gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nx];
    ImageTYPE *gradientImagePtrZ=&gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nx];

    ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
    ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){
        
        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        jacIndex=0;
        for(int z=0; z<targetImage->nz; z++){
            for(int y=0; y<targetImage->ny; y++){
                for(int x=0; x<targetImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if USE_SSE
                    val.f[0] = realPosition[0];
                    val.f[1] = realPosition[1];
                    val.f[2] = realPosition[2];
                    val.f[3] = 1.0;
                    __m128 _realPosition_sse = val.m;
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_x,_realPosition_sse);
                    voxelPosition[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_y,_realPosition_sse);
                    voxelPosition[1]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_z,_realPosition_sse);
                    voxelPosition[2]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    voxelPosition[0]
                        = jac_ijk_matrix->m[0][0]*realPosition[0]
                        + jac_ijk_matrix->m[0][1]*realPosition[1]
                        + jac_ijk_matrix->m[0][2]*realPosition[2]
                        + jac_ijk_matrix->m[0][3];
                    voxelPosition[1]
                        = jac_ijk_matrix->m[1][0]*realPosition[0]
                        + jac_ijk_matrix->m[1][1]*realPosition[1]
                        + jac_ijk_matrix->m[1][2]*realPosition[2]
                        + jac_ijk_matrix->m[1][3];
                    voxelPosition[2]
                        = jac_ijk_matrix->m[2][0]*realPosition[0]
                        + jac_ijk_matrix->m[2][1]*realPosition[1]
                        + jac_ijk_matrix->m[2][2]*realPosition[2]
                        + jac_ijk_matrix->m[2][3];
#endif

                    xPre=(int)floor((ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]);
                    yPre=(int)floor((ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]);
                    zPre=(int)floor((ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]);

                    ImageTYPE detJac = 1.0f;

                    if( xPre>-1 && (xPre+3)<splineControlPoint->nx &&
                        yPre>-1 && (yPre+3)<splineControlPoint->ny &&
                        zPre>-1 && (zPre+3)<splineControlPoint->nz ){

                        basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        xBasis[3] = (ImageTYPE)(FFF/6.0);
                        xFirst[3]= (ImageTYPE)(FF / 2.0);
                        xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                        xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                        xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                        basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
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

                        basis=(ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]-(ImageTYPE)(zPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        zBasis[3] = (ImageTYPE)(FFF/6.0);
                        zFirst[3]= (ImageTYPE)(FF / 2.0);
                        zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
                        zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
                        zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

                        ImageTYPE Tx_x=0.0;
                        ImageTYPE Ty_x=0.0;
                        ImageTYPE Tz_x=0.0;
                        ImageTYPE Tx_y=0.0;
                        ImageTYPE Ty_y=0.0;
                        ImageTYPE Tz_y=0.0;
                        ImageTYPE Tx_z=0.0;
                        ImageTYPE Ty_z=0.0;
                        ImageTYPE Tz_z=0.0;
                        ImageTYPE newPositionX = 0.0;
                        ImageTYPE newPositionY = 0.0;
                        ImageTYPE newPositionZ = 0.0;

                        if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
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
                            xPreOld=xPre;
                            yPreOld=yPre;
                            zPreOld=zPre;
                        }

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
                        __m128 pos_x =  _mm_set_ps1(0.0);
                        __m128 pos_y =  _mm_set_ps1(0.0);
                        __m128 pos_z =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                        __m128 *ptrBasisX = (__m128 *) &basisX[0];
                        __m128 *ptrBasisY = (__m128 *) &basisY[0];
                        __m128 *ptrBasisZ = (__m128 *) &basisZ[0];

                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        __m128 _xBasis_sse = val.m;
                        val.f[0] = xFirst[0];
                        val.f[1] = xFirst[1];
                        val.f[2] = xFirst[2];
                        val.f[3] = xFirst[3];
                        __m128 _xFirst_sse = val.m;

                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                __m128 _yFirst_sse  = _mm_set_ps1(yFirst[b]);

                                __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                __m128 _zFirst_sse  = _mm_set_ps1(zFirst[c]);

                                __m128 _temp_sseX   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                                __m128 _temp_sseY   = _mm_mul_ps(_yFirst_sse, _zBasis_sse);
                                __m128 _temp_sseZ   = _mm_mul_ps(_yBasis_sse, _zFirst_sse);

                                *ptrBasisX          = _mm_mul_ps(_temp_sseX, _xFirst_sse);
                                *ptrBasisY          = _mm_mul_ps(_temp_sseY, _xBasis_sse);
                                *ptrBasisZ          = _mm_mul_ps(_temp_sseZ, _xBasis_sse);
                                __m128 _basis       = _mm_mul_ps(_temp_sseX, _xBasis_sse);

                                pos_x = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), pos_x );
                                tempX_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrX), tempX_x );
                                tempX_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrX), tempX_y );
                                tempX_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrX), tempX_z );

                                pos_y = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), pos_y );
                                tempY_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrY), tempY_x );
                                tempY_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrY), tempY_y );
                                tempY_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrY), tempY_z );

                                pos_z = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), pos_z );
                                tempZ_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrZ), tempZ_x );
                                tempZ_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrZ), tempZ_y );
                                tempZ_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrZ), tempZ_z );

                                ptrX++;
                                ptrY++;
                                ptrZ++;
                                ptrBasisX++;
                                ptrBasisY++;
                                ptrBasisZ++;
                            }
                        }

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
                        val.m = pos_x;
                        newPositionX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_y;
                        newPositionY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_z;
                        newPositionZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        coord=0;
                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                for(int a=0; a<4; a++){
                                    basisX[coord] = tempBasisX*xFirst[a];   // z * y * x'
                                    basisY[coord] = tempBasisY*xBasis[a];   // z * y'* x
                                    basisZ[coord] = tempBasisZ*xBasis[a];   // z'* y * x
                                    basis = tempBasisX*xBasis[a];   // z * y * x
                                    Tx_x += basisX[coord]*xControlPointCoordinates[coord];
                                    Tx_y += basisY[coord]*xControlPointCoordinates[coord];
                                    Tx_z += basisZ[coord]*xControlPointCoordinates[coord];
                                    Ty_x += basisX[coord]*yControlPointCoordinates[coord];
                                    Ty_y += basisY[coord]*yControlPointCoordinates[coord];
                                    Ty_z += basisZ[coord]*yControlPointCoordinates[coord];
                                    Tz_x += basisX[coord]*zControlPointCoordinates[coord];
                                    Tz_y += basisY[coord]*zControlPointCoordinates[coord];
                                    Tz_z += basisZ[coord]*zControlPointCoordinates[coord];
                                    newPositionX += basis*xControlPointCoordinates[coord];
                                    newPositionY += basis*yControlPointCoordinates[coord];
                                    newPositionZ += basis*zControlPointCoordinates[coord];
                                    coord++;
                                }
                            }
                        }
#endif

                        deformationFieldArrayX[jacIndex] = newPositionX;
                        deformationFieldArrayY[jacIndex] = newPositionY;
                        deformationFieldArrayZ[jacIndex] = newPositionZ;

                        jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)splineControlPoint->dz);

                        jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                        detJac = nifti_mat33_determ(jacobianMatrix);
                        jacobianMatrix=nifti_mat33_inverse(jacobianMatrix);

                        if(detJac>0){
                            detJac = 2.0f * log(detJac); // otherwise the gradient of the determinant itself is considered
//                            detJac *= -1.0; // otherwise the gradient of the determinant itself is considered
                        }
                        coord = 0;
                        for(int Z=zPre; Z<zPre+4; Z++){
                            unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                            ImageTYPE *xPtr = &gradientImagePtrX[index];
                            ImageTYPE *yPtr = &gradientImagePtrY[index];
                            ImageTYPE *zPtr = &gradientImagePtrZ[index];
                            for(int Y=yPre; Y<yPre+4; Y++){
                                index = Y*splineControlPoint->nx;
                                ImageTYPE *xxPtr = &xPtr[index];
                                ImageTYPE *yyPtr = &yPtr[index];
                                ImageTYPE *zzPtr = &zPtr[index];
                                for(int X=xPre; X<xPre+4; X++){
                                    ImageTYPE gradientValueX = detJac *
                                        ( jacobianMatrix.m[0][0] * basisX[coord]
                                        + jacobianMatrix.m[0][1] * basisY[coord]
                                        + jacobianMatrix.m[0][2] * basisZ[coord]);
                                    ImageTYPE gradientValueY = detJac *
                                        ( jacobianMatrix.m[1][0] * basisX[coord]
                                        + jacobianMatrix.m[1][1] * basisY[coord]
                                        + jacobianMatrix.m[1][2] * basisZ[coord]);
                                    ImageTYPE gradientValueZ = detJac *
                                        ( jacobianMatrix.m[2][0] * basisX[coord]
                                        + jacobianMatrix.m[2][1] * basisY[coord]
                                        + jacobianMatrix.m[2][2] * basisZ[coord]);

                                    xxPtr[X] += weight *
                                        (desorient.m[0][0]*gradientValueX +
                                        desorient.m[0][1]*gradientValueY +
                                        desorient.m[0][2]*gradientValueZ);
                                    yyPtr[X] += weight *
                                        (desorient.m[1][0]*gradientValueX +
                                        desorient.m[1][1]*gradientValueY +
                                        desorient.m[1][2]*gradientValueZ);
                                    zzPtr[X] += weight *
                                        (desorient.m[2][0]*gradientValueX +
                                        desorient.m[2][1]*gradientValueY +
                                        desorient.m[2][2]*gradientValueZ);
                                    coord++;
                                } // a
                            } // b
                        } // c

                    } // Not in the range
                    jacIndex++;
                } // x
            } // y
        } // z

        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    } // squaring step

    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);

    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);
}
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetApproxJacobianGradient_3D(nifti_image *velocityFieldImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "***ERROR***\treg_bspline_GetApproxJacobianGradient_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    const unsigned int jacobianNumber = velocityFieldImage->nx * velocityFieldImage->ny * velocityFieldImage->nz;

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis, FF, FFF, MF;

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE basisX[64];
    ImageTYPE basisY[64];
    ImageTYPE basisZ[64];

    mat33 desorient;
    desorient.m[0][0]=splineControlPoint->dx; desorient.m[0][1]=0.0f; desorient.m[0][2]=0.0f;
    desorient.m[1][0]=0.0f; desorient.m[1][1]=splineControlPoint->dy; desorient.m[1][2]=0.0f;
    desorient.m[2][0]=0.0f; desorient.m[2][1]=0.0f; desorient.m[2][2]=splineControlPoint->dz;
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
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, desorient));
    mat33 jacobianMatrix;

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(jacobianNumber*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(jacobianNumber*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayZ = (ImageTYPE *)malloc(jacobianNumber*sizeof(ImageTYPE));

    /* The real to voxel matrix will be used */
    mat44 *jac_ijk_matrix = NULL;
    mat44 *jac_xyz_matrix = NULL;
    if(velocityFieldImage->sform_code>0){
        jac_ijk_matrix= &(velocityFieldImage->sto_ijk);
        jac_xyz_matrix= &(velocityFieldImage->sto_xyz);
    }
    else{
        jac_ijk_matrix= &(velocityFieldImage->qto_ijk);
        jac_xyz_matrix= &(velocityFieldImage->qto_xyz);
    }
#if USE_SSE
    val.f[0] = jac_ijk_matrix->m[0][0];
    val.f[1] = jac_ijk_matrix->m[0][1];
    val.f[2] = jac_ijk_matrix->m[0][2];
    val.f[3] = jac_ijk_matrix->m[0][3];
    __m128 _jac_ijk_matrix_sse_x = val.m;
    val.f[0] = jac_ijk_matrix->m[1][0];
    val.f[1] = jac_ijk_matrix->m[1][1];
    val.f[2] = jac_ijk_matrix->m[1][2];
    val.f[3] = jac_ijk_matrix->m[1][3];
    __m128 _jac_ijk_matrix_sse_y = val.m;
    val.f[0] = jac_ijk_matrix->m[2][0];
    val.f[1] = jac_ijk_matrix->m[2][1];
    val.f[2] = jac_ijk_matrix->m[2][2];
    val.f[3] = jac_ijk_matrix->m[2][3];
    __m128 _jac_ijk_matrix_sse_z = val.m;
#endif

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int z=0; z<velocityFieldImage->nz; z++){
        for(int y=0; y<velocityFieldImage->ny; y++){
            for(int x=0; x<velocityFieldImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*x
                    + jac_xyz_matrix->m[0][1]*y
                    + jac_xyz_matrix->m[0][2]*z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*x
                    + jac_xyz_matrix->m[1][1]*y
                    + jac_xyz_matrix->m[1][2]*z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*x
                    + jac_xyz_matrix->m[2][1]*y
                    + jac_xyz_matrix->m[2][2]*z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }
    unsigned int coord=0;

    ImageTYPE *gradientImagePtrX=static_cast<ImageTYPE *>(gradientImage->data);
    ImageTYPE *gradientImagePtrY=&gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nx];
    ImageTYPE *gradientImagePtrZ=&gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nx];

    ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
    ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        jacIndex=0;
        for(int z=0; z<velocityFieldImage->nz; z++){
            for(int y=0; y<velocityFieldImage->ny; y++){
                for(int x=0; x<velocityFieldImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if USE_SSE
                    val.f[0] = realPosition[0];
                    val.f[1] = realPosition[1];
                    val.f[2] = realPosition[2];
                    val.f[3] = 1.0;
                    __m128 _realPosition_sse = val.m;
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_x,_realPosition_sse);
                    voxelPosition[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_y,_realPosition_sse);
                    voxelPosition[1]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_z,_realPosition_sse);
                    voxelPosition[2]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    voxelPosition[0]
                        = jac_ijk_matrix->m[0][0]*realPosition[0]
                        + jac_ijk_matrix->m[0][1]*realPosition[1]
                        + jac_ijk_matrix->m[0][2]*realPosition[2]
                        + jac_ijk_matrix->m[0][3];
                    voxelPosition[1]
                        = jac_ijk_matrix->m[1][0]*realPosition[0]
                        + jac_ijk_matrix->m[1][1]*realPosition[1]
                        + jac_ijk_matrix->m[1][2]*realPosition[2]
                        + jac_ijk_matrix->m[1][3];
                    voxelPosition[2]
                        = jac_ijk_matrix->m[2][0]*realPosition[0]
                        + jac_ijk_matrix->m[2][1]*realPosition[1]
                        + jac_ijk_matrix->m[2][2]*realPosition[2]
                        + jac_ijk_matrix->m[2][3];
#endif

                    xPre=(int)floor(voxelPosition[0]);
                    yPre=(int)floor(voxelPosition[1]);
                    zPre=(int)floor(voxelPosition[2]);

                    ImageTYPE detJac = 1.0f;

                    if( xPre>0 && xPre<splineControlPoint->nx-2 &&
                        yPre>0 && yPre<splineControlPoint->ny-2 &&
                        zPre>0 && zPre<splineControlPoint->nz-2 ){

                        basis=(ImageTYPE)voxelPosition[0]-(ImageTYPE)(xPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        xBasis[3] = (ImageTYPE)(FFF/6.0);
                        xFirst[3]= (ImageTYPE)(FF / 2.0);
                        xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                        xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                        xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                        basis=(ImageTYPE)voxelPosition[1]-(ImageTYPE)(yPre);
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

                        basis=(ImageTYPE)voxelPosition[2]-(ImageTYPE)(zPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        zBasis[3] = (ImageTYPE)(FFF/6.0);
                        zFirst[3]= (ImageTYPE)(FF / 2.0);
                        zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
                        zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
                        zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

                        ImageTYPE Tx_x=0.0;
                        ImageTYPE Ty_x=0.0;
                        ImageTYPE Tz_x=0.0;
                        ImageTYPE Tx_y=0.0;
                        ImageTYPE Ty_y=0.0;
                        ImageTYPE Tz_y=0.0;
                        ImageTYPE Tx_z=0.0;
                        ImageTYPE Ty_z=0.0;
                        ImageTYPE Tz_z=0.0;
                        ImageTYPE newPositionX = 0.0;
                        ImageTYPE newPositionY = 0.0;
                        ImageTYPE newPositionZ = 0.0;

                        if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
                            coord=0;
                            for(int Z=zPre-1; Z<zPre+3; Z++){
                                unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                                ImageTYPE *xPtr = &controlPointPtrX[index];
                                ImageTYPE *yPtr = &controlPointPtrY[index];
                                ImageTYPE *zPtr = &controlPointPtrZ[index];
                                for(int Y=yPre-1; Y<yPre+3; Y++){
                                    index = Y*splineControlPoint->nx;
                                    ImageTYPE *xxPtr = &xPtr[index];
                                    ImageTYPE *yyPtr = &yPtr[index];
                                    ImageTYPE *zzPtr = &zPtr[index];
                                    for(int X=xPre-1; X<xPre+3; X++){
                                        xControlPointCoordinates[coord] = (ImageTYPE)xxPtr[X];
                                        yControlPointCoordinates[coord] = (ImageTYPE)yyPtr[X];
                                        zControlPointCoordinates[coord] = (ImageTYPE)zzPtr[X];
                                        coord++;
                                    }
                                }
                            }
                            xPreOld=xPre;
                            yPreOld=yPre;
                            zPreOld=zPre;
                        }

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
                        __m128 pos_x =  _mm_set_ps1(0.0);
                        __m128 pos_y =  _mm_set_ps1(0.0);
                        __m128 pos_z =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                        __m128 *ptrBasisX = (__m128 *) &basisX[0];
                        __m128 *ptrBasisY = (__m128 *) &basisY[0];
                        __m128 *ptrBasisZ = (__m128 *) &basisZ[0];

                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        __m128 _xBasis_sse = val.m;
                        val.f[0] = xFirst[0];
                        val.f[1] = xFirst[1];
                        val.f[2] = xFirst[2];
                        val.f[3] = xFirst[3];
                        __m128 _xFirst_sse = val.m;

                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                __m128 _yFirst_sse  = _mm_set_ps1(yFirst[b]);

                                __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                __m128 _zFirst_sse  = _mm_set_ps1(zFirst[c]);

                                __m128 _temp_sseX   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                                __m128 _temp_sseY   = _mm_mul_ps(_yFirst_sse, _zBasis_sse);
                                __m128 _temp_sseZ   = _mm_mul_ps(_yBasis_sse, _zFirst_sse);

                                *ptrBasisX          = _mm_mul_ps(_temp_sseX, _xFirst_sse);
                                *ptrBasisY          = _mm_mul_ps(_temp_sseY, _xBasis_sse);
                                *ptrBasisZ          = _mm_mul_ps(_temp_sseZ, _xBasis_sse);
                                __m128 _basis       = _mm_mul_ps(_temp_sseX, _xBasis_sse);

                                pos_x = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), pos_x );
                                tempX_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrX), tempX_x );
                                tempX_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrX), tempX_y );
                                tempX_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrX), tempX_z );

                                pos_y = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), pos_y );
                                tempY_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrY), tempY_x );
                                tempY_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrY), tempY_y );
                                tempY_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrY), tempY_z );

                                pos_z = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), pos_z );
                                tempZ_x = _mm_add_ps(_mm_mul_ps(*ptrBasisX, *ptrZ), tempZ_x );
                                tempZ_y = _mm_add_ps(_mm_mul_ps(*ptrBasisY, *ptrZ), tempZ_y );
                                tempZ_z = _mm_add_ps(_mm_mul_ps(*ptrBasisZ, *ptrZ), tempZ_z );

                                ptrX++;
                                ptrY++;
                                ptrZ++;
                                ptrBasisX++;
                                ptrBasisY++;
                                ptrBasisZ++;
                            }
                        }

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
                        val.m = pos_x;
                        newPositionX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_y;
                        newPositionY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_z;
                        newPositionZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        coord=0;
                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                for(int a=0; a<4; a++){
                                    basisX[coord] = tempBasisX*xFirst[a];   // z * y * x'
                                    basisY[coord] = tempBasisY*xBasis[a];   // z * y'* x
                                    basisZ[coord] = tempBasisZ*xBasis[a];   // z'* y * x
                                    basis = tempBasisX*xBasis[a];   // z * y * x
                                    Tx_x += basisX[coord]*xControlPointCoordinates[coord];
                                    Tx_y += basisY[coord]*xControlPointCoordinates[coord];
                                    Tx_z += basisZ[coord]*xControlPointCoordinates[coord];
                                    Ty_x += basisX[coord]*yControlPointCoordinates[coord];
                                    Ty_y += basisY[coord]*yControlPointCoordinates[coord];
                                    Ty_z += basisZ[coord]*yControlPointCoordinates[coord];
                                    Tz_x += basisX[coord]*zControlPointCoordinates[coord];
                                    Tz_y += basisY[coord]*zControlPointCoordinates[coord];
                                    Tz_z += basisZ[coord]*zControlPointCoordinates[coord];
                                    newPositionX += basis*xControlPointCoordinates[coord];
                                    newPositionY += basis*yControlPointCoordinates[coord];
                                    newPositionZ += basis*zControlPointCoordinates[coord];
                                    coord++;
                                }
                            }
                        }
#endif

                        deformationFieldArrayX[jacIndex] = newPositionX;
                        deformationFieldArrayY[jacIndex] = newPositionY;
                        deformationFieldArrayZ[jacIndex] = newPositionZ;

                        jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)splineControlPoint->dz);

                        jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                        detJac = nifti_mat33_determ(jacobianMatrix);
                        jacobianMatrix=nifti_mat33_inverse(jacobianMatrix);

                        if(detJac>0){
                            detJac = 2.0f * log(detJac); // otherwise the gradient of the determinant itself is considered
                        }// otherwise the gradient of the determinant itself is considered
                        coord = 0;
                        for(int Z=zPre-1; Z<zPre+3; Z++){
                            unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                            ImageTYPE *xPtr = &gradientImagePtrX[index];
                            ImageTYPE *yPtr = &gradientImagePtrY[index];
                            ImageTYPE *zPtr = &gradientImagePtrZ[index];
                            for(int Y=yPre-1; Y<yPre+3; Y++){
                                index = Y*splineControlPoint->nx;
                                ImageTYPE *xxPtr = &xPtr[index];
                                ImageTYPE *yyPtr = &yPtr[index];
                                ImageTYPE *zzPtr = &zPtr[index];
                                for(int X=xPre-1; X<xPre+3; X++){
                                    ImageTYPE gradientValueX = detJac *
                                        ( jacobianMatrix.m[0][0] * basisX[coord]
                                        + jacobianMatrix.m[0][1] * basisY[coord]
                                        + jacobianMatrix.m[0][2] * basisZ[coord]);
                                    ImageTYPE gradientValueY = detJac *
                                        ( jacobianMatrix.m[1][0] * basisX[coord]
                                        + jacobianMatrix.m[1][1] * basisY[coord]
                                        + jacobianMatrix.m[1][2] * basisZ[coord]);
                                    ImageTYPE gradientValueZ = detJac *
                                        ( jacobianMatrix.m[2][0] * basisX[coord]
                                        + jacobianMatrix.m[2][1] * basisY[coord]
                                        + jacobianMatrix.m[2][2] * basisZ[coord]);

                                    xxPtr[X] += weight *
                                        (desorient.m[0][0]*gradientValueX +
                                        desorient.m[0][1]*gradientValueY +
                                        desorient.m[0][2]*gradientValueZ);
                                    yyPtr[X] += weight *
                                        (desorient.m[1][0]*gradientValueX +
                                        desorient.m[1][1]*gradientValueY +
                                        desorient.m[1][2]*gradientValueZ);
                                    zzPtr[X] += weight *
                                        (desorient.m[2][0]*gradientValueX +
                                        desorient.m[2][1]*gradientValueY +
                                        desorient.m[2][2]*gradientValueZ);
                                    coord++;
                                } // a
                            } // b
                        } // c

                    } // Not in the range
                    jacIndex++;
                } // x
            } // y
        } // z

        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    } // squaring step

    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);

    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
double reg_bspline_CorrectFoldingFromVelocityField_3D(  nifti_image* velocityFieldImage,
                                                        nifti_image* targetImage)
{
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "***ERROR***\treg_bspline_CorrectFoldingFromVelocityField_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    // The jacobian map is initialise to 1 everywhere
    nifti_image *jacobianImage = nifti_copy_nim_info(targetImage);
    jacobianImage->data = (void *)malloc(jacobianImage->nvox*jacobianImage->nbyper);
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++)
        jacPtr[i]=(ImageTYPE)1.0;

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis, FF, FFF, MF;

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;

    mat33 desorient;
    desorient.m[0][0]=splineControlPoint->dx; desorient.m[0][1]=0.0f; desorient.m[0][2]=0.0f;
    desorient.m[1][0]=0.0f; desorient.m[1][1]=splineControlPoint->dy; desorient.m[1][2]=0.0f;
    desorient.m[2][0]=0.0f; desorient.m[2][1]=0.0f; desorient.m[2][2]=splineControlPoint->dz;
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
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, desorient));
    mat33 jacobianMatrix;

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayZ = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));

    /* The real to voxel matrix will be used */
    mat44 *jac_ijk_matrix = NULL;
    mat44 *jac_xyz_matrix = NULL;
    if(jacobianImage->sform_code>0){
        jac_ijk_matrix= &(jacobianImage->sto_ijk);
        jac_xyz_matrix= &(jacobianImage->sto_xyz);
    }
    else{
        jac_ijk_matrix= &(jacobianImage->qto_ijk);
        jac_xyz_matrix= &(jacobianImage->qto_xyz);
    }
#if USE_SSE
    val.f[0] = jac_ijk_matrix->m[0][0];
    val.f[1] = jac_ijk_matrix->m[0][1];
    val.f[2] = jac_ijk_matrix->m[0][2];
    val.f[3] = jac_ijk_matrix->m[0][3];
    __m128 _jac_ijk_matrix_sse_x = val.m;
    val.f[0] = jac_ijk_matrix->m[1][0];
    val.f[1] = jac_ijk_matrix->m[1][1];
    val.f[2] = jac_ijk_matrix->m[1][2];
    val.f[3] = jac_ijk_matrix->m[1][3];
    __m128 _jac_ijk_matrix_sse_y = val.m;
    val.f[0] = jac_ijk_matrix->m[2][0];
    val.f[1] = jac_ijk_matrix->m[2][1];
    val.f[2] = jac_ijk_matrix->m[2][2];
    val.f[3] = jac_ijk_matrix->m[2][3];
    __m128 _jac_ijk_matrix_sse_z = val.m;
#endif

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int z=0; z<jacobianImage->nz; z++){
        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*x
                    + jac_xyz_matrix->m[0][1]*y
                    + jac_xyz_matrix->m[0][2]*z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*x
                    + jac_xyz_matrix->m[1][1]*y
                    + jac_xyz_matrix->m[1][2]*z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*x
                    + jac_xyz_matrix->m[2][1]*y
                    + jac_xyz_matrix->m[2][2]*z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }

    ImageTYPE *veloPtrX = static_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *veloPtrY = &veloPtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    ImageTYPE *veloPtrZ = &veloPtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    ImageTYPE ratio = (ImageTYPE)(2.0 / (ImageTYPE)(SCALING_VALUE*targetImage->nvox));

    unsigned int coord=0;
    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
        ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

        jacIndex=0;
        for(int z=0; z<jacobianImage->nz; z++){
            for(int y=0; y<jacobianImage->ny; y++){
                for(int x=0; x<jacobianImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if USE_SSE
                    val.f[0] = realPosition[0];
                    val.f[1] = realPosition[1];
                    val.f[2] = realPosition[2];
                    val.f[3] = 1.0;
                    __m128 _realPosition_sse = val.m;
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_x,_realPosition_sse);
                    voxelPosition[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_y,_realPosition_sse);
                    voxelPosition[1]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_z,_realPosition_sse);
                    voxelPosition[2]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    voxelPosition[0]
                        = jac_ijk_matrix->m[0][0]*realPosition[0]
                        + jac_ijk_matrix->m[0][1]*realPosition[1]
                        + jac_ijk_matrix->m[0][2]*realPosition[2]
                        + jac_ijk_matrix->m[0][3];
                    voxelPosition[1]
                        = jac_ijk_matrix->m[1][0]*realPosition[0]
                        + jac_ijk_matrix->m[1][1]*realPosition[1]
                        + jac_ijk_matrix->m[1][2]*realPosition[2]
                        + jac_ijk_matrix->m[1][3];
                    voxelPosition[2]
                        = jac_ijk_matrix->m[2][0]*realPosition[0]
                        + jac_ijk_matrix->m[2][1]*realPosition[1]
                        + jac_ijk_matrix->m[2][2]*realPosition[2]
                        + jac_ijk_matrix->m[2][3];
#endif

                    xPre=(int)floor((ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]);
                    yPre=(int)floor((ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]);
                    zPre=(int)floor((ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]);

                    ImageTYPE detJac = 1.0f;

                    if( xPre>-1 && (xPre+3)<splineControlPoint->nx &&
                        yPre>-1 && (yPre+3)<splineControlPoint->ny &&
                        zPre>-1 && (zPre+3)<splineControlPoint->nz ){

                        basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        xBasis[3] = (ImageTYPE)(FFF/6.0);
                        xFirst[3]= (ImageTYPE)(FF / 2.0);
                        xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                        xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                        xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                        basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
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

                        basis=(ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]-(ImageTYPE)(zPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        zBasis[3] = (ImageTYPE)(FFF/6.0);
                        zFirst[3]= (ImageTYPE)(FF / 2.0);
                        zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
                        zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
                        zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

                        ImageTYPE Tx_x=0.0;
                        ImageTYPE Ty_x=0.0;
                        ImageTYPE Tz_x=0.0;
                        ImageTYPE Tx_y=0.0;
                        ImageTYPE Ty_y=0.0;
                        ImageTYPE Tz_y=0.0;
                        ImageTYPE Tx_z=0.0;
                        ImageTYPE Ty_z=0.0;
                        ImageTYPE Tz_z=0.0;
                        ImageTYPE newPositionX = 0.0;
                        ImageTYPE newPositionY = 0.0;
                        ImageTYPE newPositionZ = 0.0;

                        if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
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
                            xPreOld=xPre;
                            yPreOld=yPre;
                            zPreOld=zPre;
                        }

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
                        __m128 pos_x =  _mm_set_ps1(0.0);
                        __m128 pos_y =  _mm_set_ps1(0.0);
                        __m128 pos_z =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        __m128 _xBasis_sse = val.m;
                        val.f[0] = xFirst[0];
                        val.f[1] = xFirst[1];
                        val.f[2] = xFirst[2];
                        val.f[3] = xFirst[3];
                        __m128 _xFirst_sse = val.m;

                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                __m128 _yFirst_sse  = _mm_set_ps1(yFirst[b]);

                                __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                __m128 _zFirst_sse  = _mm_set_ps1(zFirst[c]);

                                __m128 _temp_sseX   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                                __m128 _temp_sseY   = _mm_mul_ps(_yFirst_sse, _zBasis_sse);
                                __m128 _temp_sseZ   = _mm_mul_ps(_yBasis_sse, _zFirst_sse);

                                __m128 _basisX      = _mm_mul_ps(_temp_sseX, _xFirst_sse);
                                __m128 _basisY      = _mm_mul_ps(_temp_sseY, _xBasis_sse);
                                __m128 _basisZ      = _mm_mul_ps(_temp_sseZ, _xBasis_sse);
                                __m128 _basis       = _mm_mul_ps(_temp_sseX, _xBasis_sse);

                                pos_x = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), pos_x );
                                tempX_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrX), tempX_x );
                                tempX_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrX), tempX_y );
                                tempX_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrX), tempX_z );

                                pos_y = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), pos_y );
                                tempY_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrY), tempY_x );
                                tempY_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrY), tempY_y );
                                tempY_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrY), tempY_z );

                                pos_z = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), pos_z );
                                tempZ_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrZ), tempZ_x );
                                tempZ_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrZ), tempZ_y );
                                tempZ_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrZ), tempZ_z );

                                ptrX++;
                                ptrY++;
                                ptrZ++;
                            }
                        }

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
                        val.m = pos_x;
                        newPositionX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_y;
                        newPositionY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_z;
                        newPositionZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        coord=0;
                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                for(int a=0; a<4; a++){
                                    ImageTYPE basisX= tempBasisX*xFirst[a];   // z * y * x'
                                    ImageTYPE basisY= tempBasisY*xBasis[a];   // z * y'* x
                                    ImageTYPE basisZ= tempBasisZ*xBasis[a];   // z'* y * x
                                    basis = tempBasisX*xBasis[a];   // z * y * x
                                    Tx_x += basisX*xControlPointCoordinates[coord];
                                    Tx_y += basisY*xControlPointCoordinates[coord];
                                    Tx_z += basisZ*xControlPointCoordinates[coord];
                                    Ty_x += basisX*yControlPointCoordinates[coord];
                                    Ty_y += basisY*yControlPointCoordinates[coord];
                                    Ty_z += basisZ*yControlPointCoordinates[coord];
                                    Tz_x += basisX*zControlPointCoordinates[coord];
                                    Tz_y += basisY*zControlPointCoordinates[coord];
                                    Tz_z += basisZ*zControlPointCoordinates[coord];
                                    newPositionX += basis*xControlPointCoordinates[coord];
                                    newPositionY += basis*yControlPointCoordinates[coord];
                                    newPositionZ += basis*zControlPointCoordinates[coord];
                                    coord++;
                                }
                            }
                        }
#endif

                        deformationFieldArrayX[jacIndex] = newPositionX;
                        deformationFieldArrayY[jacIndex] = newPositionY;
                        deformationFieldArrayZ[jacIndex] = newPositionZ;

                        jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)splineControlPoint->dz);
                        jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)splineControlPoint->dx);
                        jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)splineControlPoint->dy);
                        jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)splineControlPoint->dz);

                        jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                        detJac = nifti_mat33_determ(jacobianMatrix);
                        jacobianMatrix=nifti_mat33_inverse(jacobianMatrix);

                        if(detJac < 0){
                            for(int c=1; c<3; c++){
                                int Z=zPre+c;
                                unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                                ImageTYPE *xPtr = &veloPtrX[index];
                                ImageTYPE *yPtr = &veloPtrY[index];
                                ImageTYPE *zPtr = &veloPtrZ[index];
                                for(int b=1; b<3; b++){
                                    int Y=yPre+b;
                                    index = Y*splineControlPoint->nx;
                                    ImageTYPE *xxPtr = &xPtr[index];
                                    ImageTYPE *yyPtr = &yPtr[index];
                                    ImageTYPE *zzPtr = &zPtr[index];
                                    ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                    ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                    ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                    for(int a=1; a<3; a++){
                                        int X=xPre+a;
                                        ImageTYPE basisX= tempBasisX*xFirst[a];   // z * y * x'
                                        ImageTYPE basisY= tempBasisY*xBasis[a];   // z * y'* x
                                        ImageTYPE basisZ= tempBasisZ*xBasis[a];   // z'* y * x

                                        ImageTYPE gradientValueX = detJac *
                                            ( jacobianMatrix.m[0][0] * basisX
                                            + jacobianMatrix.m[0][1] * basisY
                                            + jacobianMatrix.m[0][2] * basisZ);
                                        ImageTYPE gradientValueY = detJac *
                                            ( jacobianMatrix.m[1][0] * basisX
                                            + jacobianMatrix.m[1][1] * basisY
                                            + jacobianMatrix.m[1][2] * basisZ);
                                        ImageTYPE gradientValueZ = detJac *
                                            ( jacobianMatrix.m[2][0] * basisX
                                            + jacobianMatrix.m[2][1] * basisY
                                            + jacobianMatrix.m[2][2] * basisZ);

                                        xxPtr[X] += ratio *
                                            (desorient.m[0][0]*gradientValueX +
                                            desorient.m[0][1]*gradientValueY +
                                            desorient.m[0][2]*gradientValueZ);
                                        yyPtr[X] += ratio *
                                            (desorient.m[1][0]*gradientValueX +
                                            desorient.m[1][1]*gradientValueY +
                                            desorient.m[1][2]*gradientValueZ);
                                        zzPtr[X] += ratio *
                                            (desorient.m[2][0]*gradientValueX +
                                            desorient.m[2][1]*gradientValueY +
                                            desorient.m[2][2]*gradientValueZ);
                                    }
                                }
                            }

                        }
                    } // not in the range
                    jacPtr[jacIndex++] *= detJac;
                } // x
            } // y
        } // z

        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    } // squaring step

    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);

    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);

    double averagedJacobianValue=0.0;
    for(unsigned int i=0; i<jacobianImage->nvox; i++){
        double logDet = log(jacPtr[i]);
        averagedJacobianValue+=logDet*logDet;
    }
    averagedJacobianValue /= (ImageTYPE)jacobianImage->nvox;
    nifti_image_free(jacobianImage);
    return (ImageTYPE)averagedJacobianValue;
}
/* *************************************************************** */

template <class ImageTYPE>
double reg_bspline_CorrectFoldingFromApproxVelocityField_3D(    nifti_image* velocityFieldImage,
                                                                nifti_image* targetImage)
{
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "***ERROR***\treg_bspline_CorrectFoldingFromApproxVelocityField_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    // The jacobian map is initialise to 1 everywhere
    nifti_image *jacobianImage = nifti_copy_nim_info(velocityFieldImage);
    jacobianImage->dim[0]=3;
    jacobianImage->dim[5]=jacobianImage->nu=1;
    jacobianImage->nvox = jacobianImage->nx * jacobianImage->ny * jacobianImage->nz;
    jacobianImage->data = (void *)malloc(jacobianImage->nvox*jacobianImage->nbyper);
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++)
        jacPtr[i]=(ImageTYPE)1.0;

    // Two control point image are allocated
    nifti_image *splineControlPoint = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint->data=(void *)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    nifti_image *splineControlPoint2 = nifti_copy_nim_info(velocityFieldImage);
    splineControlPoint2->data=(void *)malloc(splineControlPoint2->nvox * splineControlPoint2->nbyper);
    memcpy(splineControlPoint2->data, velocityFieldImage->data, splineControlPoint2->nvox * splineControlPoint2->nbyper);

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis, FF, FFF, MF;

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;

    mat33 desorient;
    desorient.m[0][0]=splineControlPoint->dx; desorient.m[0][1]=0.0f; desorient.m[0][2]=0.0f;
    desorient.m[1][0]=0.0f; desorient.m[1][1]=splineControlPoint->dy; desorient.m[1][2]=0.0f;
    desorient.m[2][0]=0.0f; desorient.m[2][1]=0.0f; desorient.m[2][2]=splineControlPoint->dz;
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
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, desorient));
    mat33 jacobianMatrix;

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayZ = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));

    /* The real to voxel matrix will be used */
    mat44 *jac_ijk_matrix = NULL;
    mat44 *jac_xyz_matrix = NULL;
    if(jacobianImage->sform_code>0){
        jac_ijk_matrix= &(jacobianImage->sto_ijk);
        jac_xyz_matrix= &(jacobianImage->sto_xyz);
    }
    else{
        jac_ijk_matrix= &(jacobianImage->qto_ijk);
        jac_xyz_matrix= &(jacobianImage->qto_xyz);
    }
#if USE_SSE
    val.f[0] = jac_ijk_matrix->m[0][0];
    val.f[1] = jac_ijk_matrix->m[0][1];
    val.f[2] = jac_ijk_matrix->m[0][2];
    val.f[3] = jac_ijk_matrix->m[0][3];
    __m128 _jac_ijk_matrix_sse_x = val.m;
    val.f[0] = jac_ijk_matrix->m[1][0];
    val.f[1] = jac_ijk_matrix->m[1][1];
    val.f[2] = jac_ijk_matrix->m[1][2];
    val.f[3] = jac_ijk_matrix->m[1][3];
    __m128 _jac_ijk_matrix_sse_y = val.m;
    val.f[0] = jac_ijk_matrix->m[2][0];
    val.f[1] = jac_ijk_matrix->m[2][1];
    val.f[2] = jac_ijk_matrix->m[2][2];
    val.f[3] = jac_ijk_matrix->m[2][3];
    __m128 _jac_ijk_matrix_sse_z = val.m;
#endif

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int z=0; z<jacobianImage->nz; z++){
        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*x
                    + jac_xyz_matrix->m[0][1]*y
                    + jac_xyz_matrix->m[0][2]*z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*x
                    + jac_xyz_matrix->m[1][1]*y
                    + jac_xyz_matrix->m[1][2]*z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*x
                    + jac_xyz_matrix->m[2][1]*y
                    + jac_xyz_matrix->m[2][2]*z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }

    ImageTYPE *veloPtrX = static_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *veloPtrY = &veloPtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    ImageTYPE *veloPtrZ = &veloPtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    ImageTYPE ratio = (ImageTYPE)(2.0 / (ImageTYPE)(SCALING_VALUE*targetImage->nvox));

    unsigned int coord=0;
    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<SQUARING_VALUE;l++){

        reg_spline_Interpolant2Interpolator(splineControlPoint2,
                                            splineControlPoint);

        reg_getPositionFromDisplacement<ImageTYPE>(splineControlPoint);

        ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(splineControlPoint->data);
        ImageTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
        ImageTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

        jacIndex=0;
        for(int z=0; z<jacobianImage->nz; z++){
            for(int y=0; y<jacobianImage->ny; y++){
                for(int x=0; x<jacobianImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if USE_SSE
                    val.f[0] = realPosition[0];
                    val.f[1] = realPosition[1];
                    val.f[2] = realPosition[2];
                    val.f[3] = 1.0;
                    __m128 _realPosition_sse = val.m;
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_x,_realPosition_sse);
                    voxelPosition[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_y,_realPosition_sse);
                    voxelPosition[1]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = _mm_mul_ps(_jac_ijk_matrix_sse_z,_realPosition_sse);
                    voxelPosition[2]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    voxelPosition[0]
                        = jac_ijk_matrix->m[0][0]*realPosition[0]
                        + jac_ijk_matrix->m[0][1]*realPosition[1]
                        + jac_ijk_matrix->m[0][2]*realPosition[2]
                        + jac_ijk_matrix->m[0][3];
                    voxelPosition[1]
                        = jac_ijk_matrix->m[1][0]*realPosition[0]
                        + jac_ijk_matrix->m[1][1]*realPosition[1]
                        + jac_ijk_matrix->m[1][2]*realPosition[2]
                        + jac_ijk_matrix->m[1][3];
                    voxelPosition[2]
                        = jac_ijk_matrix->m[2][0]*realPosition[0]
                        + jac_ijk_matrix->m[2][1]*realPosition[1]
                        + jac_ijk_matrix->m[2][2]*realPosition[2]
                        + jac_ijk_matrix->m[2][3];
#endif

                    xPre=(int)floor((ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]);
                    yPre=(int)floor((ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]);
                    zPre=(int)floor((ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]);

                    ImageTYPE detJac = 1.0f;

                    if( xPre>-1 && (xPre+3)<splineControlPoint->nx &&
                        yPre>-1 && (yPre+3)<splineControlPoint->ny &&
                        zPre>-1 && (zPre+3)<splineControlPoint->nz ){

                        basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        xBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        xBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        xBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        xBasis[3] = (ImageTYPE)(FFF/6.0);
                        xFirst[3]= (ImageTYPE)(FF / 2.0);
                        xFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - xFirst[3]);
                        xFirst[2]= (ImageTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
                        xFirst[1]= (ImageTYPE)(- xFirst[0] - xFirst[2] - xFirst[3]);

                        basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
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

                        basis=(ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]-(ImageTYPE)(zPre);
                        FF= basis*basis;
                        FFF= FF*basis;
                        MF=(ImageTYPE)(1.0-basis);
                        zBasis[0] = (ImageTYPE)((MF)*(MF)*(MF)/6.0);
                        zBasis[1] = (ImageTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                        zBasis[2] = (ImageTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                        zBasis[3] = (ImageTYPE)(FFF/6.0);
                        zFirst[3]= (ImageTYPE)(FF / 2.0);
                        zFirst[0]= (ImageTYPE)(basis - 1.0/2.0 - zFirst[3]);
                        zFirst[2]= (ImageTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
                        zFirst[1]= (ImageTYPE)(- zFirst[0] - zFirst[2] - zFirst[3]);

                        ImageTYPE Tx_x=0.0;
                        ImageTYPE Ty_x=0.0;
                        ImageTYPE Tz_x=0.0;
                        ImageTYPE Tx_y=0.0;
                        ImageTYPE Ty_y=0.0;
                        ImageTYPE Tz_y=0.0;
                        ImageTYPE Tx_z=0.0;
                        ImageTYPE Ty_z=0.0;
                        ImageTYPE Tz_z=0.0;
                        ImageTYPE newPositionX = 0.0;
                        ImageTYPE newPositionY = 0.0;
                        ImageTYPE newPositionZ = 0.0;

                        if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
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
                            xPreOld=xPre;
                            yPreOld=yPre;
                            zPreOld=zPre;
                        }

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
                        __m128 pos_x =  _mm_set_ps1(0.0);
                        __m128 pos_y =  _mm_set_ps1(0.0);
                        __m128 pos_z =  _mm_set_ps1(0.0);
                        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
                        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
                        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];

                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        __m128 _xBasis_sse = val.m;
                        val.f[0] = xFirst[0];
                        val.f[1] = xFirst[1];
                        val.f[2] = xFirst[2];
                        val.f[3] = xFirst[3];
                        __m128 _xFirst_sse = val.m;

                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                __m128 _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                __m128 _yFirst_sse  = _mm_set_ps1(yFirst[b]);

                                __m128 _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                __m128 _zFirst_sse  = _mm_set_ps1(zFirst[c]);

                                __m128 _temp_sseX   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                                __m128 _temp_sseY   = _mm_mul_ps(_yFirst_sse, _zBasis_sse);
                                __m128 _temp_sseZ   = _mm_mul_ps(_yBasis_sse, _zFirst_sse);

                                __m128 _basisX      = _mm_mul_ps(_temp_sseX, _xFirst_sse);
                                __m128 _basisY      = _mm_mul_ps(_temp_sseY, _xBasis_sse);
                                __m128 _basisZ      = _mm_mul_ps(_temp_sseZ, _xBasis_sse);
                                __m128 _basis       = _mm_mul_ps(_temp_sseX, _xBasis_sse);

                                pos_x = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), pos_x );
                                tempX_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrX), tempX_x );
                                tempX_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrX), tempX_y );
                                tempX_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrX), tempX_z );

                                pos_y = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), pos_y );
                                tempY_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrY), tempY_x );
                                tempY_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrY), tempY_y );
                                tempY_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrY), tempY_z );

                                pos_z = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), pos_z );
                                tempZ_x = _mm_add_ps(_mm_mul_ps(_basisX, *ptrZ), tempZ_x );
                                tempZ_y = _mm_add_ps(_mm_mul_ps(_basisY, *ptrZ), tempZ_y );
                                tempZ_z = _mm_add_ps(_mm_mul_ps(_basisZ, *ptrZ), tempZ_z );

                                ptrX++;
                                ptrY++;
                                ptrZ++;
                            }
                        }

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
                        val.m = pos_x;
                        newPositionX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_y;
                        newPositionY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = pos_z;
                        newPositionZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        coord=0;
                        for(int c=0; c<4; c++){
                            for(int b=0; b<4; b++){
                                ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                for(int a=0; a<4; a++){
                                    ImageTYPE basisX= tempBasisX*xFirst[a];   // z * y * x'
                                    ImageTYPE basisY= tempBasisY*xBasis[a];   // z * y'* x
                                    ImageTYPE basisZ= tempBasisZ*xBasis[a];   // z'* y * x
                                    basis = tempBasisX*xBasis[a];   // z * y * x
                                    Tx_x += basisX*xControlPointCoordinates[coord];
                                    Tx_y += basisY*xControlPointCoordinates[coord];
                                    Tx_z += basisZ*xControlPointCoordinates[coord];
                                    Ty_x += basisX*yControlPointCoordinates[coord];
                                    Ty_y += basisY*yControlPointCoordinates[coord];
                                    Ty_z += basisZ*yControlPointCoordinates[coord];
                                    Tz_x += basisX*zControlPointCoordinates[coord];
                                    Tz_y += basisY*zControlPointCoordinates[coord];
                                    Tz_z += basisZ*zControlPointCoordinates[coord];
                                    newPositionX += basis*xControlPointCoordinates[coord];
                                    newPositionY += basis*yControlPointCoordinates[coord];
                                    newPositionZ += basis*zControlPointCoordinates[coord];
                                    coord++;
                                }
                            }
                        }
#endif

                        deformationFieldArrayX[jacIndex] = newPositionX;
                        deformationFieldArrayY[jacIndex] = newPositionY;
                        deformationFieldArrayZ[jacIndex] = newPositionZ;

                        jacobianMatrix.m[0][0]= (float)(Tx_x / splineControlPoint->dx);
                        jacobianMatrix.m[0][1]= (float)(Tx_y / splineControlPoint->dy);
                        jacobianMatrix.m[0][2]= (float)(Tx_z / splineControlPoint->dz);
                        jacobianMatrix.m[1][0]= (float)(Ty_x / splineControlPoint->dx);
                        jacobianMatrix.m[1][1]= (float)(Ty_y / splineControlPoint->dy);
                        jacobianMatrix.m[1][2]= (float)(Ty_z / splineControlPoint->dz);
                        jacobianMatrix.m[2][0]= (float)(Tz_x / splineControlPoint->dx);
                        jacobianMatrix.m[2][1]= (float)(Tz_y / splineControlPoint->dy);
                        jacobianMatrix.m[2][2]= (float)(Tz_z / splineControlPoint->dz);

                        jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                        detJac = nifti_mat33_determ(jacobianMatrix);
                        jacobianMatrix=nifti_mat33_inverse(jacobianMatrix);

                        if(detJac < 0){
                            for(int c=1; c<3; c++){
                                int Z=zPre+c;
                                unsigned int index=Z*splineControlPoint->nx*splineControlPoint->ny;
                                ImageTYPE *xPtr = &veloPtrX[index];
                                ImageTYPE *yPtr = &veloPtrY[index];
                                ImageTYPE *zPtr = &veloPtrZ[index];
                                for(int b=1; b<3; b++){
                                    int Y=yPre+b;
                                    index = Y*splineControlPoint->nx;
                                    ImageTYPE *xxPtr = &xPtr[index];
                                    ImageTYPE *yyPtr = &yPtr[index];
                                    ImageTYPE *zzPtr = &zPtr[index];
                                    ImageTYPE tempBasisX=zBasis[c]*yBasis[b];
                                    ImageTYPE tempBasisY=zBasis[c]*yFirst[b];
                                    ImageTYPE tempBasisZ=zFirst[c]*yBasis[b];
                                    for(int a=1; a<3; a++){
                                        int X=xPre+a;
                                        ImageTYPE basisX= tempBasisX*xFirst[a];   // z * y * x'
                                        ImageTYPE basisY= tempBasisY*xBasis[a];   // z * y'* x
                                        ImageTYPE basisZ= tempBasisZ*xBasis[a];   // z'* y * x

                                        ImageTYPE gradientValueX = detJac *
                                            ( jacobianMatrix.m[0][0] * basisX
                                            + jacobianMatrix.m[0][1] * basisY
                                            + jacobianMatrix.m[0][2] * basisZ);
                                        ImageTYPE gradientValueY = detJac *
                                            ( jacobianMatrix.m[1][0] * basisX
                                            + jacobianMatrix.m[1][1] * basisY
                                            + jacobianMatrix.m[1][2] * basisZ);
                                        ImageTYPE gradientValueZ = detJac *
                                            ( jacobianMatrix.m[2][0] * basisX
                                            + jacobianMatrix.m[2][1] * basisY
                                            + jacobianMatrix.m[2][2] * basisZ);

                                        xxPtr[X] += ratio *
                                            (desorient.m[0][0]*gradientValueX +
                                            desorient.m[0][1]*gradientValueY +
                                            desorient.m[0][2]*gradientValueZ);
                                        yyPtr[X] += ratio *
                                            (desorient.m[1][0]*gradientValueX +
                                            desorient.m[1][1]*gradientValueY +
                                            desorient.m[1][2]*gradientValueZ);
                                        zzPtr[X] += ratio *
                                            (desorient.m[2][0]*gradientValueX +
                                            desorient.m[2][1]*gradientValueY +
                                            desorient.m[2][2]*gradientValueZ);
                                    }
                                }
                            }

                        }
                    } // not in the range
                    jacPtr[jacIndex++] *= detJac;
                } // x
            } // y
        } // z

        reg_getDisplacementFromPosition<ImageTYPE>(splineControlPoint);

        reg_spline_cppComposition(  splineControlPoint2,
                                    splineControlPoint,
                                    1.0f,
                                    0);
    } // squaring step

    nifti_image_free(splineControlPoint);
    nifti_image_free(splineControlPoint2);

    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);

    double averagedJacobianValue=0.0;
    for(unsigned int i=0; i<jacobianImage->nvox; i++){
        double logDet = log(jacPtr[i]);
        averagedJacobianValue+=logDet*logDet;
    }
    averagedJacobianValue /= (ImageTYPE)jacobianImage->nvox;
    nifti_image_free(jacobianImage);
    return (ImageTYPE)averagedJacobianValue;
}
/* *************************************************************** */
void reg_bspline_GetJacobianGradientFromVelocityField(  nifti_image* velocityFieldImage,
                                                        nifti_image* resultImage,
                                                        nifti_image* gradientImage,
                                                        float weight,
                                                        bool approx)
{
    // The Jacobian-based penalty term gradient is computed
    if(approx){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetApproxJacobianGradient_3D<float>
                    (velocityFieldImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetApproxJacobianGradient_3D<double>
                    (velocityFieldImage, gradientImage, weight);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianGradientFromVelocityField\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianGradient_3D<float>
                    (velocityFieldImage, resultImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianGradient_3D<double>
                    (velocityFieldImage, resultImage, gradientImage, weight);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_GetJacobianGradientFromVelocityField\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
}
/* *************************************************************** */
double reg_bspline_CorrectFoldingFromVelocityField( nifti_image* velocityFieldImage,
                                                    nifti_image* targetImage,
                                                    bool approx)
{
    // The Jacobian-based folding correction is computed
    if(approx){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_CorrectFoldingFromApproxVelocityField_3D<float>
                    (velocityFieldImage, targetImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_CorrectFoldingFromApproxVelocityField_3D<double>
                        (velocityFieldImage, targetImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_CorrectFoldingFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_CorrectFoldingFromVelocityField_3D<float>
                    (velocityFieldImage, targetImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_CorrectFoldingFromVelocityField_3D<double>
                        (velocityFieldImage, targetImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_CorrectFoldingFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class SplineTYPE>
SplineTYPE reg_bspline_CorrectApproximatedFoldingFromApproxCPP_3D(  nifti_image *splineControlPoint,
                                                                    nifti_image *velocityFieldImage,
                                                                    nifti_image *targetImage)
{
    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(jacobianNumber * sizeof(SplineTYPE));

    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE normal[3];
    SplineTYPE first[3];
    normal[0] = (SplineTYPE)(1.0/6.0);
    normal[1] = (SplineTYPE)(2.0/3.0);
    normal[2] = (SplineTYPE)(1.0/6.0);
    first[0] = (SplineTYPE)(-0.5);
    first[1] = (SplineTYPE)(0.0);
    first[2] = (SplineTYPE)(0.5);

    // There are 3 different values taken into account
    SplineTYPE tempX[9], tempY[9], tempZ[9];
    int coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempX[coord]=normal[c]*normal[b];   // z * y
            tempY[coord]=normal[c]*first[b];    // z * y'
            tempZ[coord]=first[c]*normal[b];    // z'* y
            coord++;
        }
    }

    SplineTYPE basisX[27], basisY[27], basisZ[27];

    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisX[coord]=tempX[bc]*first[a];   // z * y * x'
            basisY[coord]=tempY[bc]*normal[a];  // z * y'* x
            basisZ[coord]=tempZ[bc]*normal[a];  // z'* y * x
            coord++;
        }
    }

    SplineTYPE xControlPointCoordinates[27];
    SplineTYPE yControlPointCoordinates[27];
    SplineTYPE zControlPointCoordinates[27];

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
    SplineTYPE *jacobianDeterminantPtr = jacobianDeterminant;

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
                            xControlPointCoordinates[coord] = (SplineTYPE)xxPtr[X];
                            yControlPointCoordinates[coord] = (SplineTYPE)yyPtr[X];
                            zControlPointCoordinates[coord] = (SplineTYPE)zzPtr[X];
                            coord++;
                        }
                    }
                }

                SplineTYPE Tx_x=(SplineTYPE)0.0;
                SplineTYPE Ty_x=(SplineTYPE)0.0;
                SplineTYPE Tz_x=(SplineTYPE)0.0;
                SplineTYPE Tx_y=(SplineTYPE)0.0;
                SplineTYPE Ty_y=(SplineTYPE)0.0;
                SplineTYPE Tz_y=(SplineTYPE)0.0;
                SplineTYPE Tx_z=(SplineTYPE)0.0;
                SplineTYPE Ty_z=(SplineTYPE)0.0;
                SplineTYPE Tz_z=(SplineTYPE)0.0;

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


                jacobianMatrix.m[0][0]= (float)(Tx_x / (SplineTYPE)splineControlPoint->dx);
                jacobianMatrix.m[0][1]= (float)(Tx_y / (SplineTYPE)splineControlPoint->dy);
                jacobianMatrix.m[0][2]= (float)(Tx_z / (SplineTYPE)splineControlPoint->dz);
                jacobianMatrix.m[1][0]= (float)(Ty_x / (SplineTYPE)splineControlPoint->dx);
                jacobianMatrix.m[1][1]= (float)(Ty_y / (SplineTYPE)splineControlPoint->dy);
                jacobianMatrix.m[1][2]= (float)(Ty_z / (SplineTYPE)splineControlPoint->dz);
                jacobianMatrix.m[2][0]= (float)(Tz_x / (SplineTYPE)splineControlPoint->dx);
                jacobianMatrix.m[2][1]= (float)(Tz_y / (SplineTYPE)splineControlPoint->dy);
                jacobianMatrix.m[2][2]= (float)(Tz_z / (SplineTYPE)splineControlPoint->dz);

                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                jacobianDeterminantPtr[jacIndex] = nifti_mat33_determ(jacobianMatrix);
                invertedJacobianMatricesPtr[jacIndex] = nifti_mat33_inverse(jacobianMatrix);
                jacIndex++;
            } // x
        } // y
    } //z
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
        return (SplineTYPE)(penaltyTerm/(SplineTYPE)jacobianNumber);
    }

    SplineTYPE basisValues[3];
    SplineTYPE xBasis, yBasis, zBasis;
    SplineTYPE xFirst, yFirst, zFirst;

    SplineTYPE *velocityFieldPtrX = static_cast<SplineTYPE *>(velocityFieldImage->data);
    SplineTYPE *velocityFieldPtrY = &velocityFieldPtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    SplineTYPE *velocityFieldPtrZ = &velocityFieldPtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                SplineTYPE foldingCorrectionX=(SplineTYPE)0.0;
                SplineTYPE foldingCorrectionY=(SplineTYPE)0.0;
                SplineTYPE foldingCorrectionZ=(SplineTYPE)0.0;

                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)((z-1));pixelZ<(int)((z+2)); pixelZ++){
                    if(pixelZ>0 && pixelZ<splineControlPoint->nz-1){

                        switch(pixelZ-z){
                            case -1:
                                zBasis=(SplineTYPE)(1.0/6.0);
                                zFirst=(SplineTYPE)(0.5);
                                break;
                            case 0:
                                zBasis=(SplineTYPE)(2.0/3.0);
                                zFirst=(SplineTYPE)(0.0);
                                break;
                            case 1:
                                zBasis=(SplineTYPE)(1.0/6.0);
                                zFirst=(SplineTYPE)(-0.5);
                                break;
                            default:
                                zBasis=(SplineTYPE)0.0;
                                zFirst=(SplineTYPE)0.0;
                                break;
                        }
                        for(int pixelY=(int)((y-1));pixelY<(int)((y+2)); pixelY++){
                            if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                                switch(pixelY-y){
                                    case -1:
                                        yBasis=(SplineTYPE)(1.0/6.0);
                                        yFirst=(SplineTYPE)(0.5);
                                        break;
                                    case 0:
                                        yBasis=(SplineTYPE)(2.0/3.0);
                                        yFirst=(SplineTYPE)(0.0);
                                        break;
                                    case 1:
                                        yBasis=(SplineTYPE)(1.0/6.0);
                                        yFirst=(SplineTYPE)(-0.5);
                                        break;
                                    default:
                                        yBasis=(SplineTYPE)0.0;
                                        yFirst=(SplineTYPE)0.0;
                                        break;
                                }
                                for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                                    if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                                        switch(pixelX-x){
                                            case -1:
                                                xBasis=(SplineTYPE)(1.0/6.0);
                                                xFirst=(SplineTYPE)(0.5);
                                                break;
                                            case 0:
                                                xBasis=(SplineTYPE)(2.0/3.0);
                                                xFirst=(SplineTYPE)(0.0);
                                                break;
                                            case 1:
                                                xBasis=(SplineTYPE)(1.0/6.0);
                                                xFirst=(SplineTYPE)(-0.5);
                                                break;
                                            default:
                                                xBasis=(SplineTYPE)0.0;
                                                xFirst=(SplineTYPE)0.0;
                                                break;
                                        }

                                        basisValues[0] = xFirst * yBasis * zBasis ;
                                        basisValues[1] = xBasis * yFirst * zBasis ;
                                        basisValues[2] = xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*splineControlPoint->ny+pixelY)*splineControlPoint->nx+pixelX;
                                        jacobianMatrix = invertedJacobianMatrices[jacIndex];
                                        SplineTYPE detJac = jacobianDeterminant[jacIndex];

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
                    SplineTYPE gradient[3];
                    gradient[0] = desorient.m[0][0]*foldingCorrectionX
                                + desorient.m[0][1]*foldingCorrectionY
                                + desorient.m[0][2]*foldingCorrectionZ;
                    gradient[1] = desorient.m[1][0]*foldingCorrectionX
                                + desorient.m[1][1]*foldingCorrectionY
                                + desorient.m[1][2]*foldingCorrectionZ;
                    gradient[2] = desorient.m[2][0]*foldingCorrectionX
                                + desorient.m[2][1]*foldingCorrectionY
                                + desorient.m[2][2]*foldingCorrectionZ;
                    SplineTYPE norm = (SplineTYPE)(5.0 * sqrt(gradient[0]*gradient[0]
                                            + gradient[1]*gradient[1]
                                            + gradient[2]*gradient[2])
                                            / (float)SCALING_VALUE);

                    if(norm>0.0){
                        const unsigned int id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        velocityFieldPtrX[id] += splineControlPoint->dx*gradient[0]/norm;
                        velocityFieldPtrY[id] += splineControlPoint->dy*gradient[1]/norm;
                        velocityFieldPtrZ[id] += splineControlPoint->dz*gradient[2]/norm;
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
template<class SplineTYPE>
SplineTYPE reg_bspline_CorrectApproximatedFoldingFromCPP_3D(    nifti_image *splineControlPoint,
                                                                nifti_image *velocityFieldImage,
                                                                nifti_image *targetImage)
{

#if _USE_SSE
    if(sizeof(SplineTYPE)!=4){
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

    SplineTYPE *velocityFieldPtrX = static_cast<SplineTYPE *>(velocityFieldImage->data);
    SplineTYPE *velocityFieldPtrY = &velocityFieldPtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    SplineTYPE *velocityFieldPtrZ = &velocityFieldPtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    SplineTYPE *jacValues = (SplineTYPE *)malloc(targetImage->nvox * sizeof(SplineTYPE));
    mat33 *jacInvertedMatrices = (mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    SplineTYPE *jacPtr = &jacValues[0];
    mat33 *matricesPtr = &jacInvertedMatrices[0];

    SplineTYPE yBasis[4],yFirst[4],xBasis[4],xFirst[4],zBasis[4],zFirst[4];
    SplineTYPE tempX[16], tempY[16], tempZ[16];
    SplineTYPE basisX[64], basisY[64], basisZ[64];
    SplineTYPE basis, FF, FFF, MF, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[64];
    SplineTYPE yControlPointCoordinates[64];
    SplineTYPE zControlPointCoordinates[64];

    SplineTYPE gridVoxelSpacing[3];
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

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((SplineTYPE)z/gridVoxelSpacing[2]);
        basis=(SplineTYPE)z/gridVoxelSpacing[2]-(SplineTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        FF= basis*basis;
        FFF= FF*basis;
        MF=(SplineTYPE)(1.0-basis);
        zBasis[0] = (SplineTYPE)((MF)*(MF)*(MF)/6.0);
        zBasis[1] = (SplineTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
        zBasis[2] = (SplineTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
        zBasis[3] = (SplineTYPE)(FFF/6.0);
        zFirst[3]= (SplineTYPE)(FF / 2.0);
        zFirst[0]= (SplineTYPE)(basis - 1.0/2.0 - zFirst[3]);
        zFirst[2]= (SplineTYPE)(1.0 + zFirst[0] - 2.0*zFirst[3]);
        zFirst[1]= - zFirst[0] - zFirst[2] - zFirst[3];

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
            basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            FF= basis*basis;
            FFF= FF*basis;
            MF=(SplineTYPE)(1.0-basis);
            yBasis[0] = (SplineTYPE)((MF)*(MF)*(MF)/6.0);
            yBasis[1] = (SplineTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
            yBasis[2] = (SplineTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
            yBasis[3] = (SplineTYPE)(FFF/6.0);
            yFirst[3]= (SplineTYPE)(FF / 2.0);
            yFirst[0]= (SplineTYPE)(basis - 1.0/2.0 - yFirst[3]);
            yFirst[2]= (SplineTYPE)(1.0 + yFirst[0] - 2.0*yFirst[3]);
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

                int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
                basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                FF= basis*basis;
                FFF= FF*basis;
                MF=(SplineTYPE)(1.0-basis);
                xBasis[0] = (SplineTYPE)((MF)*(MF)*(MF)/6.0);
                xBasis[1] = (SplineTYPE)((3.0*FFF - 6.0*FF +4.0)/6.0);
                xBasis[2] = (SplineTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
                xBasis[3] = (SplineTYPE)(FFF/6.0);
                xFirst[3]= (SplineTYPE)(FF / 2.0);
                xFirst[0]= (SplineTYPE)(basis - 1.0/2.0 - xFirst[3]);
                xFirst[2]= (SplineTYPE)(1.0 + xFirst[0] - 2.0*xFirst[3]);
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
                                xControlPointCoordinates[coord] = (SplineTYPE)xxPtr[X];
                                yControlPointCoordinates[coord] = (SplineTYPE)yyPtr[X];
                                zControlPointCoordinates[coord] = (SplineTYPE)zzPtr[X];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;

                SplineTYPE Tx_x=0.0;
                SplineTYPE Ty_x=0.0;
                SplineTYPE Tz_x=0.0;
                SplineTYPE Tx_y=0.0;
                SplineTYPE Ty_y=0.0;
                SplineTYPE Tz_y=0.0;
                SplineTYPE Tx_z=0.0;
                SplineTYPE Ty_z=0.0;
                SplineTYPE Tz_z=0.0;

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

                jacobianMatrix.m[0][0]= (float)(Tx_x / (SplineTYPE)splineControlPoint->dx);
                jacobianMatrix.m[0][1]= (float)(Tx_y / (SplineTYPE)splineControlPoint->dy);
                jacobianMatrix.m[0][2]= (float)(Tx_z / (SplineTYPE)splineControlPoint->dz);
                jacobianMatrix.m[1][0]= (float)(Ty_x / (SplineTYPE)splineControlPoint->dx);
                jacobianMatrix.m[1][1]= (float)(Ty_y / (SplineTYPE)splineControlPoint->dy);
                jacobianMatrix.m[1][2]= (float)(Ty_z / (SplineTYPE)splineControlPoint->dz);
                jacobianMatrix.m[2][0]= (float)(Tz_x / (SplineTYPE)splineControlPoint->dx);
                jacobianMatrix.m[2][1]= (float)(Tz_y / (SplineTYPE)splineControlPoint->dy);
                jacobianMatrix.m[2][2]= (float)(Tz_z / (SplineTYPE)splineControlPoint->dz);

                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                SplineTYPE detJac = nifti_mat33_determ(jacobianMatrix);
                *jacPtr++ = detJac;
                *matricesPtr++ = nifti_mat33_inverse(jacobianMatrix);
            } // x
        } // y
    } // z
    double averagedJacobianValue=0.0;
    for(unsigned int i=0; i<targetImage->nvox; i++){
        double logDet = log(jacValues[i]);
        averagedJacobianValue+=logDet*logDet;
    }
    averagedJacobianValue /= (SplineTYPE)targetImage->nvox;
    if(averagedJacobianValue==averagedJacobianValue){
        free(jacInvertedMatrices);
        free(jacValues);
        return (SplineTYPE)averagedJacobianValue;
    }

    SplineTYPE basisValues[3];
    SplineTYPE xBasisSingle, yBasisSingle, zBasisSingle;
    SplineTYPE xFirstSingle, yFirstSingle, zFirstSingle;
    unsigned int jacIndex;

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                SplineTYPE foldingCorrectionX=(SplineTYPE)0.0;
                SplineTYPE foldingCorrectionY=(SplineTYPE)0.0;
                SplineTYPE foldingCorrectionZ=(SplineTYPE)0.0;

                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)ceil((z-2)*gridVoxelSpacing[2]);pixelZ<(int)floor((z)*gridVoxelSpacing[2]); pixelZ++){
                    if(pixelZ>-1 && pixelZ<targetImage->nz){

                        int zPre=(int)((SplineTYPE)pixelZ/gridVoxelSpacing[2]);
                        basis=(SplineTYPE)pixelZ/gridVoxelSpacing[2]-(SplineTYPE)zPre;
                        if(basis<0.0) basis=0.0; //rounding error

                        switch(z-zPre){
                            case 0:
                                zBasisSingle=(SplineTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                zFirstSingle=(SplineTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                break;
                            case 1:
                                zBasisSingle=(SplineTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                zFirstSingle=(SplineTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                break;
                            case 2:
                                zBasisSingle=(SplineTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                zFirstSingle=(SplineTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                break;
                            case 3:
                                zBasisSingle=(SplineTYPE)(basis*basis*basis/6.0);
                                zFirstSingle=(SplineTYPE)(basis*basis/2.0);
                                break;
                            default:
                                zBasisSingle=(SplineTYPE)0.0;
                                zFirstSingle=(SplineTYPE)0.0;
                                break;
                        }


                        for(int pixelY=(int)ceil((y-2)*gridVoxelSpacing[1]);pixelY<(int)floor((y)*gridVoxelSpacing[1]); pixelY++){
                            if(pixelY>-1 && pixelY<targetImage->ny){

                                int yPre=(int)((SplineTYPE)pixelY/gridVoxelSpacing[1]);
                                basis=(SplineTYPE)pixelY/gridVoxelSpacing[1]-(SplineTYPE)yPre;
                                if(basis<0.0) basis=0.0; //rounding error

                                switch(y-yPre){
                                    case 0:
                                        yBasisSingle=(SplineTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                        yFirstSingle=(SplineTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                        break;
                                    case 1:
                                        yBasisSingle=(SplineTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                        yFirstSingle=(SplineTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                        break;
                                    case 2:
                                        yBasisSingle=(SplineTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                        yFirstSingle=(SplineTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                        break;
                                    case 3:
                                        yBasisSingle=(SplineTYPE)(basis*basis*basis/6.0);
                                        yFirstSingle=(SplineTYPE)(basis*basis/2.0);
                                        break;
                                    default:
                                        yBasisSingle=(SplineTYPE)0.0;
                                        yFirstSingle=(SplineTYPE)0.0;
                                        break;
                                }

                                for(int pixelX=(int)ceil((x-2)*gridVoxelSpacing[0]);pixelX<(int)floor((x)*gridVoxelSpacing[0]); pixelX++){
                                    if(pixelX>-1 && pixelX<targetImage->nx){

                                        int xPre=(int)((SplineTYPE)pixelX/gridVoxelSpacing[0]);
                                        basis=(SplineTYPE)pixelX/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                                        if(basis<0.0) basis=0.0; //rounding error

                                        switch(x-xPre){
                                            case 0:
                                                xBasisSingle=(SplineTYPE)((basis-1.0)*(basis-1.0)*(basis-1.0)/(6.0));
                                                xFirstSingle=(SplineTYPE)((-basis*basis + 2.0*basis - 1.0) / 2.0);
                                                break;
                                            case 1:
                                                xBasisSingle=(SplineTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                                                xFirstSingle=(SplineTYPE)((3.0*basis*basis - 4.0*basis) / 2.0);
                                                break;
                                            case 2:
                                                xBasisSingle=(SplineTYPE)((-3.0*basis*basis*basis + 3.0*basis*basis + 3.0*basis + 1.0)/6.0);
                                                xFirstSingle=(SplineTYPE)((-3.0*basis*basis + 2.0*basis + 1.0) / 2.0);
                                                break;
                                            case 3:
                                                xBasisSingle=(SplineTYPE)(basis*basis*basis/6.0);
                                                xFirstSingle=(SplineTYPE)(basis*basis/2.0);
                                                break;
                                            default:
                                                xBasisSingle=(SplineTYPE)0.0;
                                                xFirstSingle=(SplineTYPE)0.0;
                                                break;
                                        }

                                        basisValues[0]= xFirstSingle * yBasisSingle * zBasisSingle ;
                                        basisValues[1]= xBasisSingle * yFirstSingle * zBasisSingle ;
                                        basisValues[2]= xBasisSingle * yBasisSingle * zFirstSingle ;

                                        jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+pixelX;
                                        SplineTYPE detJac = jacValues[jacIndex];

                                        mat33 jacobianMatrix = jacInvertedMatrices[jacIndex];

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
                    SplineTYPE gradient[3];
                    gradient[0] = desorient.m[0][0]*foldingCorrectionX
                                + desorient.m[0][1]*foldingCorrectionY
                                + desorient.m[0][2]*foldingCorrectionZ;
                    gradient[1] = desorient.m[1][0]*foldingCorrectionX
                                + desorient.m[1][1]*foldingCorrectionY
                                + desorient.m[1][2]*foldingCorrectionZ;
                    gradient[2] = desorient.m[2][0]*foldingCorrectionX
                                + desorient.m[2][1]*foldingCorrectionY
                                + desorient.m[2][2]*foldingCorrectionZ;
                    SplineTYPE norm = (SplineTYPE)(5.0 * sqrt(gradient[0]*gradient[0]
                                            + gradient[1]*gradient[1]
                                            + gradient[2]*gradient[2])
                                            * SCALING_VALUE);
                    if(norm>0.0){
                        const unsigned int id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        velocityFieldPtrX[id] += splineControlPoint->dx*gradient[0]/norm;
                        velocityFieldPtrY[id] += splineControlPoint->dy*gradient[1]/norm;
                        velocityFieldPtrZ[id] += splineControlPoint->dz*gradient[2]/norm;
                    }
                }
            }
        }
    }
    free(jacInvertedMatrices);
    free(jacValues);
    return std::numeric_limits<float>::quiet_NaN();
}
/* *************************************************************** */
double reg_bspline_CorrectApproximatedFoldingFromCPP(   nifti_image* controlPointImage,
                                                        nifti_image* velocityFieldImage,
                                                        nifti_image* targetImage,
                                                        bool approx
                                                        )
{
    // The Jacobian-based folding correction is computed
    if(approx){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_CorrectApproximatedFoldingFromApproxCPP_3D<float>
                        (controlPointImage, velocityFieldImage, targetImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_CorrectApproximatedFoldingFromApproxCPP_3D<double>
                        (controlPointImage, velocityFieldImage, targetImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_CorrectFoldingFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                return reg_bspline_CorrectApproximatedFoldingFromCPP_3D<float>
                    (controlPointImage, velocityFieldImage, targetImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                return reg_bspline_CorrectApproximatedFoldingFromCPP_3D<double>
                        (controlPointImage, velocityFieldImage, targetImage);
                break;
            default:
                fprintf(stderr,"ERROR:\treg_bspline_CorrectFoldingFromVelocityField_3D\n");
                fprintf(stderr,"ERROR:\tOnly implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
}

#endif
