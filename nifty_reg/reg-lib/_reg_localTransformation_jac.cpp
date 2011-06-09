/*
 *  _reg_localTransformation_jac.cpp
 *  
 *
 *  Created by Marc Modat on 10/05/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTransformation.h"

#define _USE_SQUARE_LOG_JAC

/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void get_BSplineBasisValue(DTYPE basis, int index, DTYPE &value)
{
    switch(index){
        case 0: value = (DTYPE)((1.0-basis)*(1.0-basis)*(1.0-basis)/6.0);
                break;
        case 1: value = (DTYPE)((3.0*basis*basis*basis - 6.0*basis*basis + 4.0)/6.0);
                break;
        case 2: value = (DTYPE)((3.0*basis*basis - 3.0*basis*basis*basis + 3.0*basis + 1.0)/6.0);
                break;
        case 3: value = (DTYPE)(basis*basis*basis/6.0);
                break;
        default: value = (DTYPE)0;
                break;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void get_BSplineBasisValue(DTYPE basis, int index, DTYPE &value, DTYPE &first)
{
    get_BSplineBasisValue<DTYPE>(basis, index, value);
    switch(index){
        case 0: first = (DTYPE)((2.0*basis - basis*basis - 1.0)/2.0);
                break;
        case 1: first = (DTYPE)((3.0*basis*basis - 4.0*basis)/2.0);
                break;
        case 2: first = (DTYPE)((2.0*basis - 3.0*basis*basis + 1.0)/2.0);
                break;
        case 3: first = (DTYPE)(basis*basis/2.0);
                break;
        default: first = (DTYPE)0;
                break;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void get_BSplineBasisValue(DTYPE basis, int index, DTYPE &value, DTYPE &first, DTYPE &second)
{
    get_BSplineBasisValue<DTYPE>(basis, index, value, first);
    switch(index){
        case 0: second = (DTYPE)(1.0 - basis);
                break;
        case 1: second = (DTYPE)(3.0*basis -2.0);
                break;
        case 2: second = (DTYPE)(1.0 - 3.0*basis);
                break;
        case 3: second = (DTYPE)(basis);
                break;
        default: second = (DTYPE)0;
                break;
    }
}

/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void addJacobianGradientValues(mat33 jacobianMatrix,
                               double detJac,
                               DTYPE basisX,
                               DTYPE basisY,
                               DTYPE *jacobianConstraint)
{
    jacobianConstraint[0] += detJac * (jacobianMatrix.m[1][1]*basisX - jacobianMatrix.m[1][0]*basisY);
    jacobianConstraint[1] += detJac * (jacobianMatrix.m[0][0]*basisY - jacobianMatrix.m[0][1]*basisX);
}
/* *************************************************************** */
template <class DTYPE>
void addJacobianGradientValues(mat33 jacobianMatrix,
                               double detJac,
                               DTYPE basisX,
                               DTYPE basisY,
                               DTYPE basisZ,
                               DTYPE *jacobianConstraint)
{
    jacobianConstraint[0] += detJac * (
                            basisX * (jacobianMatrix.m[1][1]*jacobianMatrix.m[2][2] - jacobianMatrix.m[1][2]*jacobianMatrix.m[2][1]) +
                            basisY * (jacobianMatrix.m[1][2]*jacobianMatrix.m[2][0] - jacobianMatrix.m[1][0]*jacobianMatrix.m[2][2]) +
                            basisZ * (jacobianMatrix.m[1][0]*jacobianMatrix.m[2][1] - jacobianMatrix.m[1][1]*jacobianMatrix.m[2][0]) );

    jacobianConstraint[1] += detJac * (
                            basisX * (jacobianMatrix.m[0][2]*jacobianMatrix.m[2][1] - jacobianMatrix.m[0][1]*jacobianMatrix.m[2][2]) +
                            basisY * (jacobianMatrix.m[0][0]*jacobianMatrix.m[2][2] - jacobianMatrix.m[0][2]*jacobianMatrix.m[2][0]) +
                            basisZ * (jacobianMatrix.m[0][1]*jacobianMatrix.m[2][0] - jacobianMatrix.m[0][0]*jacobianMatrix.m[2][1]) );

    jacobianConstraint[2] += detJac * (
                            basisX * (jacobianMatrix.m[0][1]*jacobianMatrix.m[1][2] - jacobianMatrix.m[0][2]*jacobianMatrix.m[1][1]) +
                            basisY * (jacobianMatrix.m[0][2]*jacobianMatrix.m[1][0] - jacobianMatrix.m[0][0]*jacobianMatrix.m[1][2]) +
                            basisZ * (jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1] - jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0]) );
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_jacobianValue2D(nifti_image *splineControlPoint,
                                   nifti_image *targetImage)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>
                                   (splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>
                                   (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    DTYPE yBasis[4],yFirst[4],temp[4],first[4];
    DTYPE basisX[16], basisY[16];
    DTYPE basis, oldBasis=(DTYPE)(1.1);

    DTYPE xControlPointCoordinates[16];
    DTYPE yControlPointCoordinates[16];

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    double constraintValue=0;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
        basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<DTYPE>(basis, yBasis, yFirst);

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
            basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<DTYPE>(basis, temp, first);

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*first[a];   // y * x'
                    basisY[coord]=yFirst[b]*temp[a];    // y'* x
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                get_GridValues<DTYPE>(xPre,
                                           yPre,
                                           splineControlPoint,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates,
                                           false);
            }
            oldBasis=basis;

            DTYPE Tx_x=0.0;
            DTYPE Ty_x=0.0;
            DTYPE Tx_y=0.0;
            DTYPE Ty_y=0.0;

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
            double detJac = nifti_mat33_determ(jacobianMatrix);
            if(detJac>0.0){
                double logJac = log(detJac);
#ifdef _USE_SQUARE_LOG_JAC
                constraintValue += logJac*logJac;
#else
                constraintValue +=  fabs(logJac);
#endif
            }
            else return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return constraintValue/(double)targetImage->nvox;
}
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_jacobianValue3D(nifti_image *splineControlPoint,
                                   nifti_image *targetImage)
{
#if _USE_SSE
    if(sizeof(DTYPE)!=4){
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_jacobianValue3D\n");
        fprintf(stderr, "[NiftyReg ERROR] The SSE implementation assume single precision... Exit\n");
        exit(1);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    DTYPE *controlPointPtrX = static_cast<DTYPE *>
                                   (splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>
                                   (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    DTYPE *controlPointPtrZ = static_cast<DTYPE *>
                                   (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    DTYPE zBasis[4],zFirst[4],temp[4],first[4];
    DTYPE tempX[16], tempY[16], tempZ[16];
    DTYPE basisX[64], basisY[64], basisZ[64];
    DTYPE basis, oldBasis=(DTYPE)(1.1);

    DTYPE xControlPointCoordinates[64];
    DTYPE yControlPointCoordinates[64];
    DTYPE zControlPointCoordinates[64];

    DTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    double constraintValue=0;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((DTYPE)z/gridVoxelSpacing[2]);
        basis=(DTYPE)z/gridVoxelSpacing[2]-(DTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<DTYPE>(basis, zBasis, zFirst);

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
            basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<DTYPE>(basis, temp, first);

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
            unsigned int coord=0;
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

                int xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
                basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                Get_BSplineBasisValues<DTYPE>(basis, temp, first);

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
                    get_GridValues<DTYPE>(xPre,
                                               yPre,
                                               zPre,
                                               splineControlPoint,
                                               controlPointPtrX,
                                               controlPointPtrY,
                                               controlPointPtrZ,
                                               xControlPointCoordinates,
                                               yControlPointCoordinates,
                                               zControlPointCoordinates,
                                               false);
                }
                oldBasis=basis;

                DTYPE Tx_x=0.0;
                DTYPE Ty_x=0.0;
                DTYPE Tz_x=0.0;
                DTYPE Tx_y=0.0;
                DTYPE Ty_y=0.0;
                DTYPE Tz_y=0.0;
                DTYPE Tx_z=0.0;
                DTYPE Ty_z=0.0;
                DTYPE Tz_z=0.0;

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
                double detJac = nifti_mat33_determ(jacobianMatrix);

                if(detJac>0.0){
                    double logJac = log(detJac);
#ifdef _USE_SQUARE_LOG_JAC
                    constraintValue += logJac*logJac;
#else
                    constraintValue +=  fabs(log(detJac));
#endif
                }
                else return std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    return constraintValue/(double)targetImage->nvox;
}
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_jacobianApproxValue2D(nifti_image *splineControlPoint)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    DTYPE basisX[9], basisY[9], constraintValue=0, xControlPointCoordinates[9], yControlPointCoordinates[9];
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0.0, 0.5};
    unsigned int coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisX[coord]=first[a]*normal[b];
            basisY[coord]=normal[a]*first[b];
            coord++;
        }
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    for(int y=1;y<splineControlPoint->ny-2;y++){
        for(int x=1;x<splineControlPoint->nx-2;x++){

            get_GridValuesApprox<DTYPE>(x-1,
                                        y-1,
                                        splineControlPoint,
                                        controlPointPtrX,
                                        controlPointPtrY,
                                        xControlPointCoordinates,
                                        yControlPointCoordinates,
                                        true);

            DTYPE Tx_x=0.0;
            DTYPE Ty_x=0.0;
            DTYPE Tx_y=0.0;
            DTYPE Ty_y=0.0;

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
            DTYPE detJac = jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1]-
                           jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0];

            if(detJac>0.0){
                double logJac = log(detJac);
#ifdef _USE_SQUARE_LOG_JAC
                constraintValue += logJac*logJac;
#else
                constraintValue +=  fabs(log(detJac));
#endif
            }
            else return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return constraintValue/(double)((splineControlPoint->nx-2)*(splineControlPoint->ny-2));
}
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_jacobianApproxValue3D(nifti_image *splineControlPoint)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    float basisX[27], basisY[27], basisZ[27];
    DTYPE xControlPointCoordinates[27], yControlPointCoordinates[27], zControlPointCoordinates[27];
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0, 0.5};
    // There are six different values taken into account
    DTYPE tempX[9], tempY[9], tempZ[9];
    int coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempX[coord]=normal[c]*normal[b];  // z * y
            tempY[coord]=normal[c]*first[b];  // z * y"
            tempZ[coord]=first[c]*normal[b];  // z"* y
            coord++;
        }
    }
    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisX[coord]=tempX[bc]*first[a];    // z * y * x"
            basisY[coord]=tempY[bc]*normal[a];    // z * y"* x
            basisZ[coord]=tempZ[bc]*normal[a];    // z"* y * x
            coord++;
        }
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>
        (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    DTYPE *controlPointPtrZ = static_cast<DTYPE *>
        (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    double constraintValue=0.0;
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            for(int x=1;x<splineControlPoint->nx-1;x++){

                get_GridValuesApprox<DTYPE>(x-1,
                                                 y-1,
                                                 z-1,
                                                 splineControlPoint,
                                                 controlPointPtrX,
                                                 controlPointPtrY,
                                                 controlPointPtrZ,
                                                 xControlPointCoordinates,
                                                 yControlPointCoordinates,
                                                 zControlPointCoordinates,
                                                 false);

                DTYPE Tx_x=0.0;
                DTYPE Ty_x=0.0;
                DTYPE Tz_x=0.0;
                DTYPE Tx_y=0.0;
                DTYPE Ty_y=0.0;
                DTYPE Tz_y=0.0;
                DTYPE Tx_z=0.0;
                DTYPE Ty_z=0.0;
                DTYPE Tz_z=0.0;

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
                DTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                if(detJac>0.0){
                    double logJac = log(detJac);
#ifdef _USE_SQUARE_LOG_JAC
                    constraintValue += logJac*logJac;
#else
                    constraintValue +=  fabs(log(detJac));
#endif
                }
                else return std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    return constraintValue/(double)((splineControlPoint->nx-2)*(splineControlPoint->ny-2)*(splineControlPoint->nz-2));
}
/* *************************************************************** */
extern "C++"
double reg_bspline_jacobian(nifti_image *splineControlPoint,
                            nifti_image *targetImage,
                            bool approx
                            )
{
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                if(approx)
                    return reg_bspline_jacobianApproxValue2D<float>(splineControlPoint);
                else return reg_bspline_jacobianValue2D<float>(splineControlPoint, targetImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                if(approx)
                    return reg_bspline_jacobianApproxValue2D<double>(splineControlPoint);
                else return reg_bspline_jacobianValue2D<double>(splineControlPoint, targetImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the jacobian value\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian value is not computed\n");
                exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
            if(approx)
                    return reg_bspline_jacobianApproxValue3D<float>(splineControlPoint);
                else return reg_bspline_jacobianValue3D<float>(splineControlPoint, targetImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                if(approx)
                    return reg_bspline_jacobianApproxValue3D<double>(splineControlPoint);
                else return reg_bspline_jacobianValue3D<double>(splineControlPoint, targetImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the jacobian value\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian value is not computed\n");
                exit(1);
        }

    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_computeJacobianMatrices_2D(nifti_image *targetImage,
                                nifti_image *splineControlPoint,
                                mat33 *jacobianMatrices,
                                DTYPE *jacobianDeterminant)
{

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    DTYPE yBasis[4],yFirst[4],xBasis[4],xFirst[4];
    DTYPE basisX[16], basisY[16], basis;
    int oldXpre=9999999, oldYpre=9999999;

    DTYPE xControlPointCoordinates[16];
    DTYPE yControlPointCoordinates[16];

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    unsigned int index=0, coord;
    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
        basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<DTYPE>(basis, yBasis, yFirst);

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
            basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<DTYPE>(basis, xBasis, xFirst);

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*xFirst[a];   // y * x'
                    basisY[coord]=yFirst[b]*xBasis[a];    // y'* x
                    coord++;
                }
            }

            if(xPre!=oldXpre || yPre!=oldYpre){
                get_GridValues<DTYPE>(xPre,
                                           yPre,
                                           splineControlPoint,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates,
                                           false);
                oldXpre=xPre;oldYpre=yPre;
            }

            DTYPE Tx_x=0.0;
            DTYPE Ty_x=0.0;
            DTYPE Tx_y=0.0;
            DTYPE Ty_y=0.0;

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

            if(jacobianDeterminant!=NULL)
                jacobianDeterminant[index] = jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1] -
                                             jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0];
            if(jacobianMatrices!=NULL)
                jacobianMatrices[index] = jacobianMatrix;
            index++;
        }
    }
}
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_computeJacobianMatrices_3D(nifti_image *targetImage,
                                nifti_image *splineControlPoint,
                                mat33 *jacobianMatrices,
                                DTYPE *jacobianDeterminant)
{
#if _USE_SSE
    if(sizeof(DTYPE)!=4){
        fprintf(stderr, "[NiftyReg ERROR] computeJacobianMatrices_3D\n");
        fprintf(stderr, "[NiftyReg ERROR] The SSE implementation assume single precision... Exit\n");
        exit(1);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    DTYPE yBasis[4], yFirst[4], xBasis[4], xFirst[4] ,zBasis[4] ,zFirst[4], basis;
    DTYPE tempX[16], tempY[16], tempZ[16], basisX[64], basisY[64], basisZ[64];
    int oldXpre=999999, oldYpre=999999, oldZpre=999999;

    DTYPE xControlPointCoordinates[64];
    DTYPE yControlPointCoordinates[64];
    DTYPE zControlPointCoordinates[64];

    DTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    int index=0;
#ifndef _USE_SSE
    int coord;
#endif
    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((DTYPE)z/gridVoxelSpacing[2]);
        basis=(DTYPE)z/gridVoxelSpacing[2]-(DTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<DTYPE>(basis, zBasis, zFirst);

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
            basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<DTYPE>(basis, yBasis, yFirst);
#ifdef _USE_SSE
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
                    tempY[coord]=zBasis[c]*yFirst[b]; // z * y'
                    tempZ[coord]=zFirst[c]*yBasis[b]; // z'* y
                    coord++;
                }
            }
#endif

            for(int x=0; x<targetImage->nx; x++){

                int xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
                basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                Get_BSplineBasisValues<DTYPE>(basis, xBasis, xFirst);

#ifdef _USE_SSE
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

                if(xPre!=oldXpre || yPre!=oldYpre || zPre!=oldZpre){
                    get_GridValues<DTYPE>(xPre,
                                               yPre,
                                               zPre,
                                               splineControlPoint,
                                               controlPointPtrX,
                                               controlPointPtrY,
                                               controlPointPtrZ,
                                               xControlPointCoordinates,
                                               yControlPointCoordinates,
                                               zControlPointCoordinates,
                                               false);
                    oldXpre=xPre; oldYpre=yPre; oldZpre=zPre;
                }

                DTYPE Tx_x=0.0;
                DTYPE Ty_x=0.0;
                DTYPE Tz_x=0.0;
                DTYPE Tx_y=0.0;
                DTYPE Ty_y=0.0;
                DTYPE Tz_y=0.0;
                DTYPE Tx_z=0.0;
                DTYPE Ty_z=0.0;
                DTYPE Tz_z=0.0;

#ifdef _USE_SSE
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

                if(jacobianDeterminant!=NULL)
                    jacobianDeterminant[index] = nifti_mat33_determ(jacobianMatrix);
                if(jacobianMatrices!=NULL)
                    jacobianMatrices[index] = jacobianMatrix;
                index++;
            }
        }
    }
}
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_computeApproximateJacobianMatrices_2D( nifti_image *splineControlPoint,
                                                        mat33 *jacobianMatrices,
                                                        DTYPE *jacobianDeterminant)
{
    unsigned int jacobianNumber = splineControlPoint->nx*splineControlPoint->ny;
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    DTYPE basisX[9], basisY[9], xControlPointCoordinates[9], yControlPointCoordinates[9];
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0.0, 0.5};
    unsigned int coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisX[coord]=first[a]*normal[b];
            basisY[coord]=normal[a]*first[b];
            coord++;
        }
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    jacobianMatrix.m[0][0]=jacobianMatrix.m[1][1]=jacobianMatrix.m[2][2]=1.f;
    jacobianMatrix.m[1][0]=jacobianMatrix.m[0][1]=jacobianMatrix.m[0][2]=0.f;
    jacobianMatrix.m[2][0]=jacobianMatrix.m[2][1]=jacobianMatrix.m[1][2]=0.f;
    for(unsigned int i=0;i<jacobianNumber;++i){
        memcpy(&jacobianMatrices[i], &jacobianMatrix, sizeof(mat33));
        jacobianDeterminant[i]=1.f;
    }

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[jacobianNumber];

    mat33 *jacobianMatricesPtr = jacobianMatrices;
    DTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* All the Jacobian matrices are computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    // Loop over (almost) each control point
    for(int y=1;y<splineControlPoint->ny-1;y++){
        unsigned int jacIndex = y*splineControlPoint->nx + 1;
        for(int x=1;x<splineControlPoint->nx-1;x++){

            // The control points are stored
            get_GridValuesApprox<DTYPE>(x-1,
                                        y-1,
                                        splineControlPoint,
                                        controlPointPtrX,
                                        controlPointPtrY,
                                        xControlPointCoordinates,
                                        yControlPointCoordinates,
                                        true);

            DTYPE Tx_x=(DTYPE)0.0;
            DTYPE Ty_x=(DTYPE)0.0;
            DTYPE Tx_y=(DTYPE)0.0;
            DTYPE Ty_y=(DTYPE)0.0;

            for(int a=0; a<9; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];
                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }

            jacobianMatrix.m[0][0] = (float)(Tx_x / splineControlPoint->dx);
            jacobianMatrix.m[0][1] = (float)(Tx_y / splineControlPoint->dy);
            jacobianMatrix.m[0][2] = 0.0f;
            jacobianMatrix.m[1][0] = (float)(Ty_x / splineControlPoint->dx);
            jacobianMatrix.m[1][1] = (float)(Ty_y / splineControlPoint->dy);
            jacobianMatrix.m[1][2] = 0.0f;
            jacobianMatrix.m[2][0] = 0.0f;
            jacobianMatrix.m[2][1] = 0.0f;
            jacobianMatrix.m[2][2] = 1.0f;

            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);

            jacobianDeterminantPtr[jacIndex] = (jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1])
                                             - (jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0]);
            jacobianMatricesPtr[jacIndex] = jacobianMatrix;
            jacIndex++;
        } // x
    } // y
}
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_computeApproximateJacobianMatrices_3D( nifti_image *splineControlPoint,
                                                        mat33 *jacobianMatrices,
                                                        DTYPE *jacobianDeterminant)
{
    // As the contraint is only computed at the voxel position, the basis values of the spline are always the same
    float basisX[27], basisY[27], basisZ[27];
    DTYPE xControlPointCoordinates[27], yControlPointCoordinates[27], zControlPointCoordinates[27];
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0, 0.5};
    // There are six different values taken into account
    DTYPE tempX[9], tempY[9], tempZ[9];
    int coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempX[coord]=normal[c]*normal[b];  // z * y
            tempY[coord]=normal[c]*first[b];  // z * y"
            tempZ[coord]=first[c]*normal[b];  // z"* y
            coord++;
        }
    }
    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisX[coord]=tempX[bc]*first[a];    // z * y * x"
            basisY[coord]=tempY[bc]*normal[a];    // z * y"* x
            basisZ[coord]=tempZ[bc]*normal[a];    // z"* y * x
            coord++;
        }
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    jacobianMatrix.m[0][0]=jacobianMatrix.m[1][1]=jacobianMatrix.m[2][2]=1.f;
    jacobianMatrix.m[1][0]=jacobianMatrix.m[0][1]=jacobianMatrix.m[0][2]=0.f;
    jacobianMatrix.m[2][0]=jacobianMatrix.m[2][1]=jacobianMatrix.m[1][2]=0.f;
    for(unsigned int i=0;i<splineControlPoint->nvox/3;++i){
        jacobianMatrices[i]=jacobianMatrix;
        jacobianDeterminant[i]=1.f;
    }

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* All the Jacobian matrices are computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    mat33 *jacobianMatricesPtr = jacobianMatrices;
    DTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    // Loop over (almost) each control point
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            unsigned int jacIndex = (z*splineControlPoint->ny+y)*splineControlPoint->nx+1;
            for(int x=1;x<splineControlPoint->nx-1;x++){

                // The control points are stored
                get_GridValuesApprox<DTYPE>(x-1,
                                                 y-1,
                                                 z-1,
                                                 splineControlPoint,
                                                 controlPointPtrX,
                                                 controlPointPtrY,
                                                 controlPointPtrZ,
                                                 xControlPointCoordinates,
                                                 yControlPointCoordinates,
                                                 zControlPointCoordinates,
                                                 false);

                DTYPE Tx_x=(DTYPE)0.0; DTYPE Ty_x=(DTYPE)0.0; DTYPE Tz_x=(DTYPE)0.0;
                DTYPE Tx_y=(DTYPE)0.0; DTYPE Ty_y=(DTYPE)0.0; DTYPE Tz_y=(DTYPE)0.0;
                DTYPE Tx_z=(DTYPE)0.0; DTYPE Ty_z=(DTYPE)0.0; DTYPE Tz_z=(DTYPE)0.0;

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
                jacobianMatricesPtr[jacIndex] = jacobianMatrix;
                jacIndex++;
            } // x
        } // y
    } //z
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_bspline_jacobianDeterminantGradient2D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    mat33 *jacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(targetImage->nvox * sizeof(DTYPE));

    reg_bspline_computeJacobianMatrices_2D<DTYPE>(targetImage,
                                                  splineControlPoint,
                                                  jacobianMatrices,
                                                  jacobianDeterminant);

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    DTYPE basisValues[2];
    DTYPE xBasis=0, yBasis=0, basis;
    DTYPE xFirst=0, yFirst=0;
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The gradient are now computed for every control point
    DTYPE *gradientImagePtrX = static_cast<DTYPE *>(gradientImage->data);
    DTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            DTYPE jacobianConstraint[2]={0, 0};

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)reg_ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)reg_ceil((y+1)*gridVoxelSpacing[1]); ++pixelY){
                if(pixelY>-1 && pixelY<targetImage->ny){

                    int yPre=(int)((DTYPE)pixelY/gridVoxelSpacing[1]);
                    basis=(DTYPE)pixelY/gridVoxelSpacing[1]-(DTYPE)yPre;
                    get_BSplineBasisValue<DTYPE>(basis,y-yPre,yBasis,yFirst);
                    if(yBasis!=0||yFirst!=0){

                        for(int pixelX=(int)reg_ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)reg_ceil((x+1)*gridVoxelSpacing[0]); ++pixelX){
                            if(pixelX>-1 && pixelX<targetImage->nx){

                                int xPre=(int)((DTYPE)pixelX/gridVoxelSpacing[0]);
                                basis=(DTYPE)pixelX/gridVoxelSpacing[0]-(DTYPE)xPre;
                                get_BSplineBasisValue<DTYPE>(basis,x-xPre,xBasis,xFirst);

                                jacIndex = pixelY*targetImage->nx+pixelX;
                                double detJac=jacobianDeterminant[jacIndex];

                                if(detJac>0.0 && (xBasis!=0||xFirst!=0)){

                                    jacobianMatrix = jacobianMatrices[jacIndex];
                                    basisValues[0]= xFirst * yBasis;
                                    basisValues[1]= xBasis * yFirst;
#ifdef _USE_SQUARE_LOG_JAC
                                    detJac= 2.0*log(detJac) /detJac;
#else
                                    detJac = (log(detJac)>0?1.0:-1.0) / detjac;
#endif
                                    addJacobianGradientValues(jacobianMatrix,
                                                              detJac,
                                                              basisValues[0],
                                                              basisValues[1],
                                                              jacobianConstraint);
                                }
                             }// if x
                        }// x
                    }
                }// if y
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            *gradientImagePtrX++ += weight *
                (desorient.m[0][0]*jacobianConstraint[0] +
                desorient.m[0][1]*jacobianConstraint[1]);
            *gradientImagePtrY++ += weight *
                (desorient.m[1][0]*jacobianConstraint[0] +
                desorient.m[1][1]*jacobianConstraint[1]);
        }
    }
    free(jacobianDeterminant);
    free(jacobianMatrices);

}
/* *************************************************************** */
template<class DTYPE>
void reg_bspline_jacobianDeterminantGradientApprox2D(nifti_image *splineControlPoint,
                                                     nifti_image *targetImage,
                                                     nifti_image *gradientImage,
                                                     float weight
                                                     )
{
    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny;

    mat33 *jacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(jacobianNumber * sizeof(DTYPE));
    reg_bspline_computeApproximateJacobianMatrices_2D<DTYPE>(splineControlPoint,
                                                 jacobianMatrices,
                                                 jacobianDeterminant);


    DTYPE basisX[9], basisY[9];
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0.0, 0.5};
    unsigned int coord=0;
     // INVERTED ON PURPOSE
    for(int b=2; b>-1; --b){
        for(int a=2; a>-1; --a){
            basisX[coord]=first[a]*normal[b];
            basisY[coord]=normal[a]*first[b];
            coord++;
        }
    }
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    DTYPE *gradientImagePtrX = static_cast<DTYPE *>(gradientImage->data);
    DTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny];

    DTYPE approxRatio = weight * (DTYPE)(targetImage->nx*targetImage->ny)
        / (DTYPE)(jacobianNumber);

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            DTYPE jacobianConstraint[2]={0. , 0.};

            // Loop over all the control points in the surrounding area
            coord=0;
            for(int pixelY=(int)(y-1);pixelY<(int)(y+2); ++pixelY){
                if(pixelY>-1 && pixelY<splineControlPoint->ny){

                    for(int pixelX=(int)(x-1);pixelX<(int)(x+2); ++pixelX){
                        if(pixelX>-1 && pixelX<splineControlPoint->nx){

                            jacIndex = pixelY*splineControlPoint->nx+pixelX;
                            double detJac=(double)jacobianDeterminant[jacIndex];

                            if(detJac>0.0){

                                jacobianMatrix = jacobianMatrices[jacIndex];

#ifdef _USE_SQUARE_LOG_JAC
                                /* derivative of the squared log of the Jacobian determinant */
                                detJac = 2.0 * log(detJac) / detJac;
#else
                                detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                                addJacobianGradientValues(jacobianMatrix,
                                                          detJac,
                                                          basisX[coord],
                                                          basisY[coord],
                                                          jacobianConstraint);
                            }
                        } // if x
                        coord++;
                    }// x
                }// if y
                else coord+=3;
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            *gradientImagePtrX++ += approxRatio * (desorient.m[0][0]*jacobianConstraint[0] + desorient.m[0][1]*jacobianConstraint[1]);
            *gradientImagePtrY++ += approxRatio * (desorient.m[1][0]*jacobianConstraint[0] + desorient.m[1][1]*jacobianConstraint[1]);
        }
    }
    free(jacobianMatrices);
    free(jacobianDeterminant);
}
/* *************************************************************** */
template<class DTYPE>
void reg_bspline_jacobianDeterminantGradient3D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    mat33 *jacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(targetImage->nvox * sizeof(DTYPE));

    reg_bspline_computeJacobianMatrices_3D<DTYPE>(targetImage,
                                           splineControlPoint,
                                           jacobianMatrices,
                                           jacobianDeterminant);

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    DTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    DTYPE xBasis=0, yBasis=0, zBasis=0, basis;
    DTYPE xFirst=0, yFirst=0, zFirst=0;
    DTYPE basisValues[3];
    unsigned int jacIndex;

    // The gradient are now computed for every control point
    DTYPE *gradientImagePtrX = static_cast<DTYPE *>(gradientImage->data);
    DTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz];
    DTYPE *gradientImagePtrZ = &gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                DTYPE jacobianConstraint[3]={0., 0., 0.};

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)reg_ceil((z-3)*gridVoxelSpacing[2]);pixelZ<=(int)reg_ceil((z+1)*gridVoxelSpacing[2]); pixelZ++){
                    if(pixelZ>-1 && pixelZ<targetImage->nz){

                        int zPre=(int)((DTYPE)pixelZ/gridVoxelSpacing[2]);
                        basis=(DTYPE)pixelZ/gridVoxelSpacing[2]-(DTYPE)zPre;
                        get_BSplineBasisValue<DTYPE>(basis,z-zPre,zBasis,zFirst);

                        for(int pixelY=(int)reg_ceil((y-3)*gridVoxelSpacing[1]);pixelY<=(int)reg_ceil((y+1)*gridVoxelSpacing[1]); pixelY++){
                            if(pixelY>-1 && pixelY<targetImage->ny && (zFirst!=0 || zBasis!=0)){

                                int yPre=(int)((DTYPE)pixelY/gridVoxelSpacing[1]);
                                basis=(DTYPE)pixelY/gridVoxelSpacing[1]-(DTYPE)yPre;
                                get_BSplineBasisValue<DTYPE>(basis,y-yPre,yBasis,yFirst);

                                jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+(int)reg_ceil((x-3)*gridVoxelSpacing[0]);

                                for(int pixelX=(int)reg_ceil((x-3)*gridVoxelSpacing[0]);pixelX<=(int)reg_ceil((x+1)*gridVoxelSpacing[0]); pixelX++){
                                    if(pixelX>-1 && pixelX<targetImage->nx && (yFirst!=0 || yBasis!=0)){

                                        double detJac = jacobianDeterminant[jacIndex];

                                        int xPre=(int)((DTYPE)pixelX/gridVoxelSpacing[0]);
                                        basis=(DTYPE)pixelX/gridVoxelSpacing[0]-(DTYPE)xPre;
                                        get_BSplineBasisValue<DTYPE>(basis,x-xPre,xBasis,xFirst);

                                        if(detJac>0.0 && (xBasis!=0 ||xFirst!=0)){

                                            jacobianMatrix = jacobianMatrices[jacIndex];

                                            basisValues[0] = xFirst * yBasis * zBasis ;
                                            basisValues[1] = xBasis * yFirst * zBasis ;
                                            basisValues[2] = xBasis * yBasis * zFirst ;

                                            jacobianMatrix = jacobianMatrices[jacIndex];
#ifdef _USE_SQUARE_LOG_JAC
                                            detJac= 2.0*log(detJac) / detJac;
#else
                                            detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                                            addJacobianGradientValues<DTYPE>(jacobianMatrix,
                                                                                  detJac,
                                                                                  basisValues[0],
                                                                                  basisValues[1],
                                                                                  basisValues[2],
                                                                                  jacobianConstraint);
                                        }
                                    } // if x
                                    jacIndex++;
                                }// x
                            }// if y
                        }// y
                    }// if z
                } // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                *gradientImagePtrX++ += weight *
                                     ( desorient.m[0][0]*jacobianConstraint[0]
                                     + desorient.m[0][1]*jacobianConstraint[1]
                                     + desorient.m[0][2]*jacobianConstraint[2]);
                *gradientImagePtrY++ += weight *
                                     ( desorient.m[1][0]*jacobianConstraint[0]
                                     + desorient.m[1][1]*jacobianConstraint[1]
                                     + desorient.m[1][2]*jacobianConstraint[2]);
                *gradientImagePtrZ++ += weight *
                                     ( desorient.m[2][0]*jacobianConstraint[0]
                                     + desorient.m[2][1]*jacobianConstraint[1]
                                     + desorient.m[2][2]*jacobianConstraint[2]);
            }
        }
    }
    free(jacobianMatrices);
    free(jacobianDeterminant);
}
/* *************************************************************** */
template<class DTYPE>
void reg_bspline_jacobianDeterminantGradientApprox3D(nifti_image *splineControlPoint,
                                                     nifti_image *referenceImage,
                                                     nifti_image *gradientImage,
                                                     float weight)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *jacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(jacobianNumber * sizeof(DTYPE));

    reg_bspline_computeApproximateJacobianMatrices_3D<DTYPE>(splineControlPoint,
                                                      jacobianMatrices,
                                                      jacobianDeterminant);

    DTYPE basisX[27], basisY[27], basisZ[27];
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0.0, 0.5};
    unsigned int coord=0;
     // INVERTED ON PURPOSE
    for(int c=2; c>-1; --c){
        for(int b=2; b>-1; --b){
            for(int a=2; a>-1; --a){
                basisX[coord]=normal[c]*normal[b]*first[a];
                basisY[coord]=normal[c]*first[b]*normal[a];
                basisZ[coord]=first[c]*normal[b]*normal[a];
                coord++;
            }
        }
    }
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    DTYPE *gradientImagePtrX = static_cast<DTYPE *>(gradientImage->data);
    DTYPE *gradientImagePtrY = &gradientImagePtrX[jacobianNumber];
    DTYPE *gradientImagePtrZ = &gradientImagePtrY[jacobianNumber];

    DTYPE approxRatio = weight * (DTYPE)(referenceImage->nx*referenceImage->ny*referenceImage->nz)
                             / (DTYPE)jacobianNumber;

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                DTYPE jacobianConstraint[3]={0, 0, 0};

                // Loop over all the control points in the surrounding area
                coord=0;
                for(int pixelZ=(int)(z-1); pixelZ<(int)(z+2); ++pixelZ){
                    if(pixelZ>0 && pixelZ<splineControlPoint->nz-1){

                        for(int pixelY=(int)(y-1); pixelY<(int)(y+2); ++pixelY){
                            if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                                jacIndex = (pixelZ*splineControlPoint->ny+pixelY)*splineControlPoint->nx+x-1;
                                for(int pixelX=(int)(x-1); pixelX<(int)(x+2); ++pixelX){
                                    if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                                        double detJac = (double)jacobianDeterminant[jacIndex];

                                        if(detJac>0.0){
                                            jacobianMatrix = jacobianMatrices[jacIndex];
#ifdef _USE_SQUARE_LOG_JAC
                                            detJac = 2.0*log(detJac) / detJac;
#else
                                            detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                                            addJacobianGradientValues<DTYPE>(jacobianMatrix,
                                                                             detJac,
                                                                             basisX[coord],
                                                                             basisY[coord],
                                                                             basisZ[coord],
                                                                             jacobianConstraint);
                                        }
                                    } // if x
                                    coord++;
                                    jacIndex++;
                                }// x
                            }// if y
                            else coord+=3;
                        }// y
                    }// if z
                    else coord+=9;
                } // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                *gradientImagePtrX++ += approxRatio *
                                     ( desorient.m[0][0]*jacobianConstraint[0]
                                     + desorient.m[0][1]*jacobianConstraint[1]
                                     + desorient.m[0][2]*jacobianConstraint[2]);
                *gradientImagePtrY++ += approxRatio *
                                     ( desorient.m[1][0]*jacobianConstraint[0]
                                     + desorient.m[1][1]*jacobianConstraint[1]
                                     + desorient.m[1][2]*jacobianConstraint[2]);
                *gradientImagePtrZ++ += approxRatio *
                                     ( desorient.m[2][0]*jacobianConstraint[0]
                                     + desorient.m[2][1]*jacobianConstraint[1]
                                     + desorient.m[2][2]*jacobianConstraint[2]);
            }
        }
    }
    free(jacobianMatrices);
    free(jacobianDeterminant);
}
/* *************************************************************** */
extern "C++"
void reg_bspline_jacobianDeterminantGradient(nifti_image *splineControlPoint,
                                             nifti_image *targetImage,
                                             nifti_image *gradientImage,
                                             float weight,
                                             bool approx)
{
    if(splineControlPoint->datatype != gradientImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] The spline control point image and the gradient image were expected to have the same datatype\n");
        fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
        exit(1);
    }

    if(splineControlPoint->nz==1){
        if(approx){
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradientApprox2D<float>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradientApprox2D<double>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The jacobian penalty gradient has not computed\n");
                    exit(1);
            }
        }
        else{
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradient2D<float>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradient2D<double>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The jacobian penalty gradient has not computed\n");
                    exit(1);
            }
        }
    }
    else{
        if(approx){
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradientApprox3D<float>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradientApprox3D<double>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The jacobian penalty gradient has not computed\n");
                    exit(1);
            }
        }
        else{
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    reg_bspline_jacobianDeterminantGradient3D<float>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    reg_bspline_jacobianDeterminantGradient3D<double>
                        (splineControlPoint, targetImage, gradientImage, weight);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The jacobian penalty gradient has not computed\n");
                    exit(1);
            }
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_correctFolding_2D(nifti_image *splineControlPoint,
                                     nifti_image *targetImage)
{

    mat33 *jacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(targetImage->nvox * sizeof(DTYPE));

    reg_bspline_computeJacobianMatrices_2D<DTYPE>(targetImage,
                                           splineControlPoint,
                                           jacobianMatrices,
                                           jacobianDeterminant);

    // The current Penalty term value is computed
    double penaltyTerm =0.0;
    for(unsigned int i=0; i< targetImage->nvox; i++){
        double logDet = log(jacobianDeterminant[i]);
        penaltyTerm += logDet*logDet;
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(jacobianMatrices);
        return penaltyTerm/(double)targetImage->nvox;
    }

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    DTYPE basisValues[2];
    DTYPE xBasis=0, yBasis=0, basis;
    DTYPE xFirst=0, yFirst=0;
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The gradient are now computed for every control point
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            DTYPE foldingCorrection[2]={0.0, 0.0};

            bool correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)reg_ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)reg_floor((y+1)*gridVoxelSpacing[1]); pixelY++){
                if(pixelY>-1 && pixelY<targetImage->ny){


                    for(int pixelX=(int)reg_ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)reg_floor((x+1)*gridVoxelSpacing[0]); pixelX++){
                        if(pixelX>-1 && pixelX<targetImage->nx){

                            jacIndex = pixelY*targetImage->nx+pixelX;
                            double detJac=jacobianDeterminant[jacIndex];

                            if(detJac<=0.0){

                                int yPre=(int)((DTYPE)pixelY/gridVoxelSpacing[1]);
                                basis=(DTYPE)pixelY/gridVoxelSpacing[1]-(DTYPE)yPre;
                                get_BSplineBasisValue<DTYPE>(basis, y-yPre,yBasis,yFirst);

                                int xPre=(int)((DTYPE)pixelX/gridVoxelSpacing[0]);
                                basis=(DTYPE)pixelX/gridVoxelSpacing[0]-(DTYPE)xPre;
                                get_BSplineBasisValue<DTYPE>(basis, x-xPre,xBasis,xFirst);

                                basisValues[0]= xFirst * yBasis;
                                basisValues[1]= xBasis * yFirst;

                                jacobianMatrix = jacobianMatrices[jacIndex];

                                /* derivative of the Jacobian determinant itself */
                                correctFolding=true;
                                addJacobianGradientValues<DTYPE>(jacobianMatrix,
                                                                      1.0,
                                                                      basisValues[0],
                                                                      basisValues[1],
                                                                      foldingCorrection);
                            }
                        }// if x
                    }// x
                }// if y
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            if(correctFolding){
                DTYPE gradient[2];
                gradient[0] = desorient.m[0][0]*foldingCorrection[0] +
                              desorient.m[0][1]*foldingCorrection[1];
                gradient[1] = desorient.m[1][0]*foldingCorrection[0] +
                              desorient.m[1][1]*foldingCorrection[1];
                DTYPE norm = 5.0 * sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
                if(norm>0.0){
                    const unsigned int id = y*splineControlPoint->nx+x;
                    controlPointPtrX[id] += (DTYPE)(splineControlPoint->dx*gradient[0]/norm);
                    controlPointPtrY[id] += (DTYPE)(splineControlPoint->dy*gradient[1]/norm);
                }
            }

        }
    }
    free(jacobianDeterminant);
    free(jacobianMatrices);
    return std::numeric_limits<double>::quiet_NaN();

}
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_correctFoldingApprox_2D(nifti_image *splineControlPoint)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny;

    mat33 *jacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(jacobianNumber * sizeof(DTYPE));

    reg_bspline_computeApproximateJacobianMatrices_2D<DTYPE>(splineControlPoint,
                                                      jacobianMatrices,
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
        free(jacobianMatrices);
        jacobianNumber = (splineControlPoint->nx-2) * (splineControlPoint->ny-2);
        return penaltyTerm/(double)jacobianNumber;
    }

    DTYPE basisValues[2];
    DTYPE xBasis=0, yBasis=0;
    DTYPE xFirst=0, yFirst=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The gradient are now computed for every control point
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            DTYPE foldingCorrection[2]={0., 0.};
            bool correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(y-1);pixelY<(y+2); pixelY++){
                if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                    for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                            jacIndex = pixelY*splineControlPoint->nx+pixelX;
                            DTYPE logDet=jacobianDeterminant[jacIndex];

                            if(logDet<=0.0){

                                get_BSplineBasisValue<DTYPE>(0, y-pixelY+1,yBasis,yFirst);
                                get_BSplineBasisValue<DTYPE>(0, x-pixelX+1,xBasis,xFirst);

                                basisValues[0] = xFirst * yBasis ;
                                basisValues[1] = xBasis * yFirst ;

                                jacobianMatrix = jacobianMatrices[jacIndex];

                                /* derivative of the Jacobian determinant itself */
                                correctFolding=true;
                                addJacobianGradientValues<DTYPE>(jacobianMatrix,
                                                                      1.0,
                                                                      basisValues[0],
                                                                      basisValues[1],
                                                                      foldingCorrection);
                            }
                        }// if x
                    }// x
                }// if y
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            if(correctFolding){
                DTYPE gradient[2];
                gradient[0] = desorient.m[0][0]*foldingCorrection[0]
                            + desorient.m[0][1]*foldingCorrection[1];
                gradient[1] = desorient.m[1][0]*foldingCorrection[0]
                            + desorient.m[1][1]*foldingCorrection[1];
                DTYPE norm = 5.0 * sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
                if(norm>0.0){
                    const unsigned int id = y*splineControlPoint->nx+x;
                    controlPointPtrX[id] += splineControlPoint->dx*gradient[0]/norm;
                    controlPointPtrY[id] += splineControlPoint->dy*gradient[1]/norm;
                }
            }

        }
    }
    free(jacobianDeterminant);
    free(jacobianMatrices);
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_correctFolding_3D(nifti_image *splineControlPoint,
                                     nifti_image *targetImage)
{

    mat33 *jacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(targetImage->nvox * sizeof(DTYPE));

    reg_bspline_computeJacobianMatrices_3D<DTYPE>(targetImage,
                                           splineControlPoint,
                                           jacobianMatrices,
                                           jacobianDeterminant);

    /* The current Penalty term value is computed */
    double penaltyTerm =0.0;
    for(unsigned int i=0; i< targetImage->nvox; i++){
        double logDet = log(jacobianDeterminant[i]);
        penaltyTerm += fabs(logDet);
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(jacobianMatrices);
        return penaltyTerm/(double)targetImage->nvox;
    }

    /*  */
    mat33 reorient, desorient;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    DTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    DTYPE basisValues[3];
    DTYPE xBasis=0, yBasis=0, zBasis=0, basis;
    DTYPE xFirst=0, yFirst=0, zFirst=0;
    unsigned int jacIndex;

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                DTYPE foldingCorrection[3]={0., 0., 0.};
                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)reg_ceil((z-3)*gridVoxelSpacing[2]);pixelZ<(int)reg_floor((z+1)*gridVoxelSpacing[2]); pixelZ++){
                    if(pixelZ>-1 && pixelZ<targetImage->nz){

                        for(int pixelY=(int)reg_ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)reg_floor((y+1)*gridVoxelSpacing[1]); pixelY++){
                            if(pixelY>-1 && pixelY<targetImage->ny){

                                for(int pixelX=(int)reg_ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)reg_floor((x+1)*gridVoxelSpacing[0]); pixelX++){
                                    if(pixelX>-1 && pixelX<targetImage->nx){

                                        jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+pixelX;
                                        double detJac = jacobianDeterminant[jacIndex];

                                        if(detJac<=0.0){

                                            mat33 jacobianMatrix = jacobianMatrices[jacIndex];

                                            int zPre=(int)((DTYPE)pixelZ/gridVoxelSpacing[2]);
                                            basis=(DTYPE)pixelZ/gridVoxelSpacing[2]-(DTYPE)zPre;
                                            get_BSplineBasisValue<DTYPE>(basis, z-zPre,zBasis,zFirst);

                                            int yPre=(int)((DTYPE)pixelY/gridVoxelSpacing[1]);
                                            basis=(DTYPE)pixelY/gridVoxelSpacing[1]-(DTYPE)yPre;
                                            get_BSplineBasisValue<DTYPE>(basis, y-yPre,yBasis,yFirst);

                                            int xPre=(int)((DTYPE)pixelX/gridVoxelSpacing[0]);
                                            basis=(DTYPE)pixelX/gridVoxelSpacing[0]-(DTYPE)xPre;
                                            get_BSplineBasisValue<DTYPE>(basis, x-xPre,xBasis,xFirst);

                                            basisValues[0]= xFirst * yBasis * zBasis ;
                                            basisValues[1]= xBasis * yFirst * zBasis ;
                                            basisValues[2]= xBasis * yBasis * zFirst ;

                                            correctFolding=true;
                                            addJacobianGradientValues<DTYPE>(jacobianMatrix,
                                                                                  1.0,
                                                                                  basisValues[0],
                                                                                  basisValues[1],
                                                                                  basisValues[2],
                                                                                  foldingCorrection);
                                        } // detJac<0.0
                                    } // if x
                                }// x
                            }// if y
                        }// y
                    }// if z
                } // z
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                if(correctFolding){
                    DTYPE gradient[3];
                    gradient[0] = desorient.m[0][0]*foldingCorrection[0]
                                + desorient.m[0][1]*foldingCorrection[1]
                                + desorient.m[0][2]*foldingCorrection[2];
                    gradient[1] = desorient.m[1][0]*foldingCorrection[0]
                                + desorient.m[1][1]*foldingCorrection[1]
                                + desorient.m[1][2]*foldingCorrection[2];
                    gradient[2] = desorient.m[2][0]*foldingCorrection[0]
                                + desorient.m[2][1]*foldingCorrection[1]
                                + desorient.m[2][2]*foldingCorrection[2];
                    DTYPE norm = (DTYPE)(5.0 * sqrt(gradient[0]*gradient[0] +
                                                              gradient[1]*gradient[1] +
                                                              gradient[2]*gradient[2]));

                    if(norm>0.0){
                        const unsigned int id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        controlPointPtrX[id] += (DTYPE)(splineControlPoint->dx*gradient[0]/norm);
                        controlPointPtrY[id] += (DTYPE)(splineControlPoint->dy*gradient[1]/norm);
                        controlPointPtrZ[id] += (DTYPE)(splineControlPoint->dz*gradient[2]/norm);
                    }
                }
            }
        }
    }
    free(jacobianDeterminant);
    free(jacobianMatrices);
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
template<class DTYPE>
double reg_bspline_correctFoldingApprox_3D(nifti_image *splineControlPoint)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *jacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(jacobianNumber * sizeof(DTYPE));

    reg_bspline_computeApproximateJacobianMatrices_3D<DTYPE>(splineControlPoint,
                                                      jacobianMatrices,
                                                      jacobianDeterminant);

    // The current Penalty term value is computed
    unsigned int jacIndex;
    double penaltyTerm =0.0;
    for(int k=1; k<splineControlPoint->nz-1; k++){
        for(int j=1; j<splineControlPoint->ny-1; j++){
            jacIndex = (k*splineControlPoint->ny+j)*splineControlPoint->nx+1;
            for(int i=1; i<splineControlPoint->nx-1; i++){
                double logDet = log(jacobianDeterminant[jacIndex++]);
                penaltyTerm += fabs(logDet);
            }
        }
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(jacobianMatrices);
        jacobianNumber = (splineControlPoint->nx-2) * (splineControlPoint->ny-2) * (splineControlPoint->nz-2);
        return penaltyTerm/(double)jacobianNumber;
    }

    DTYPE basisValues[3];
    DTYPE xBasis=0, yBasis=0, zBasis=0;
    DTYPE xFirst=0, yFirst=0, zFirst=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                DTYPE foldingCorrection[3]={0., 0., 0.};
                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)((z-1));pixelZ<(int)((z+2)); pixelZ++){
                    if(pixelZ>0 && pixelZ<splineControlPoint->nz-1){


                        for(int pixelY=(int)((y-1));pixelY<(int)((y+2)); pixelY++){
                            if(pixelY>0 && pixelY<splineControlPoint->ny-1){


                                    for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                                        jacIndex = (pixelZ*splineControlPoint->ny+pixelY)*splineControlPoint->nx+pixelX;
                                        double detJac = jacobianDeterminant[jacIndex];

                                        if(detJac<=0.0){

                                            get_BSplineBasisValue<DTYPE>(0, z-pixelZ+1, zBasis, zFirst);
                                            get_BSplineBasisValue<DTYPE>(0, y-pixelY+1, yBasis, yFirst);
                                            get_BSplineBasisValue<DTYPE>(0, x-pixelX+1, xBasis, xFirst);

                                            basisValues[0] = xFirst * yBasis * zBasis ;
                                            basisValues[1] = xBasis * yFirst * zBasis ;
                                            basisValues[2] = xBasis * yBasis * zFirst ;

                                            jacobianMatrix = jacobianMatrices[jacIndex];

                                            correctFolding=true;
                                            addJacobianGradientValues<DTYPE>(jacobianMatrix,
                                                                                  1.0,
                                                                                  basisValues[0],
                                                                                  basisValues[1],
                                                                                  basisValues[2],
                                                                                  foldingCorrection);
                                        } // detJac<0.0
                                    } // if x
                                }// x
                            }// if y
                        }// y
                    }// if z
                } // z
                if(correctFolding){
                    DTYPE gradient[3];
                    gradient[0] = desorient.m[0][0]*foldingCorrection[0]
                                + desorient.m[0][1]*foldingCorrection[1]
                                + desorient.m[0][2]*foldingCorrection[2];
                    gradient[1] = desorient.m[1][0]*foldingCorrection[0]
                                + desorient.m[1][1]*foldingCorrection[1]
                                + desorient.m[1][2]*foldingCorrection[2];
                    gradient[2] = desorient.m[2][0]*foldingCorrection[0]
                                + desorient.m[2][1]*foldingCorrection[1]
                                + desorient.m[2][2]*foldingCorrection[2];
                    DTYPE norm = (DTYPE)(5.0 * sqrt(gradient[0]*gradient[0]
                                        + gradient[1]*gradient[1]
                                        + gradient[2]*gradient[2]));

                    if(norm>(DTYPE)0.0){
                        const unsigned int id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        controlPointPtrX[id] += (DTYPE)(splineControlPoint->dx*gradient[0]/norm);
                        controlPointPtrY[id] += (DTYPE)(splineControlPoint->dy*gradient[1]/norm);
                        controlPointPtrZ[id] += (DTYPE)(splineControlPoint->dz*gradient[2]/norm);
                    }
                }
            }
        }
    }
    free(jacobianDeterminant);
    free(jacobianMatrices);
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
extern "C++"
double reg_bspline_correctFolding(nifti_image *splineControlPoint,
                                  nifti_image *targetImage,
                                  bool approx)
{

    if(splineControlPoint->nz==1){
        if(approx){
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFoldingApprox_2D<float>
                        (splineControlPoint);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFoldingApprox_2D<double>
                        (splineControlPoint);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
                    exit(1);
            }
        }
        else{
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFolding_2D<float>
                        (splineControlPoint, targetImage);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFolding_2D<double>
                        (splineControlPoint, targetImage);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
                    exit(1);
            }
        }
    }
    else{
        if(approx){
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFoldingApprox_3D<float>
                        (splineControlPoint);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFoldingApprox_3D<double>
                        (splineControlPoint);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
                    exit(1);
            }
        }
        else{
            switch(splineControlPoint->datatype){
                case NIFTI_TYPE_FLOAT32:
                    return reg_bspline_correctFolding_3D<float>
                        (splineControlPoint, targetImage);
                    break;
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFolding_3D<double>
                        (splineControlPoint, targetImage);
                    break;
#endif
                default:
                    fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the Jacobian determinant gradient\n");
                    fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
                    exit(1);
            }
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_GetJacobianMap2D(nifti_image *splineControlPoint,
                                  nifti_image *jacobianImage)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    DTYPE *jacobianMapPtr = static_cast<DTYPE *>(jacobianImage->data);

    DTYPE yBasis[4],yFirst[4],temp[4],first[4];
    DTYPE basisX[16], basisY[16];
    DTYPE basis, oldBasis=(DTYPE)(1.1);

    DTYPE xControlPointCoordinates[16];
    DTYPE yControlPointCoordinates[16];

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    unsigned int coord=0;

    /* In case the matrix is not diagonal, the jacobian has to be reoriented */
    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    for(int y=0; y<jacobianImage->ny; y++){

        int yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
        basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<DTYPE>(basis, yBasis, yFirst);

        for(int x=0; x<jacobianImage->nx; x++){

            int xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
            basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<DTYPE>(basis, temp, first);

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
                    DTYPE *xPtr = &controlPointPtrX[index];
                    DTYPE *yPtr = &controlPointPtrY[index];
                    for(int X=xPre; X<xPre+4; X++){
                        xControlPointCoordinates[coord] = (DTYPE)xPtr[X];
                        yControlPointCoordinates[coord] = (DTYPE)yPtr[X];
                        coord++;
                    }
                }
            }
            oldBasis=basis;
            DTYPE Tx_x=0.0;
            DTYPE Ty_x=0.0;
            DTYPE Tx_y=0.0;
            DTYPE Ty_y=0.0;

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
template <class DTYPE>
void reg_bspline_GetJacobianMap3D(nifti_image *splineControlPoint,
                                  nifti_image *jacobianImage)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    DTYPE *controlPointPtrZ = static_cast<DTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    DTYPE *jacobianMapPtr = static_cast<DTYPE *>(jacobianImage->data);

    DTYPE zBasis[4],zFirst[4],temp[4],first[4];
    DTYPE tempX[16], tempY[16], tempZ[16];
    DTYPE basisX[64], basisY[64], basisZ[64];
    DTYPE basis, oldBasis=(DTYPE)(1.1);

    DTYPE xControlPointCoordinates[64];
    DTYPE yControlPointCoordinates[64];
    DTYPE zControlPointCoordinates[64];

    DTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;
    unsigned int coord=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    for(int z=0; z<jacobianImage->nz; z++){

        int zPre=(int)((DTYPE)z/gridVoxelSpacing[2]);
        basis=(DTYPE)z/gridVoxelSpacing[2]-(DTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<DTYPE>(basis, zBasis, zFirst);

        for(int y=0; y<jacobianImage->ny; y++){

            int yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
            basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<DTYPE>(basis, temp, first);

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

                int xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
                basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                Get_BSplineBasisValues<DTYPE>(basis, temp, first);

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
                        DTYPE *xPtr = &controlPointPtrX[index];
                        DTYPE *yPtr = &controlPointPtrY[index];
                        DTYPE *zPtr = &controlPointPtrZ[index];
                        for(int Y=yPre; Y<yPre+4; Y++){
                            index = Y*splineControlPoint->nx;
                            DTYPE *xxPtr = &xPtr[index];
                            DTYPE *yyPtr = &yPtr[index];
                            DTYPE *zzPtr = &zPtr[index];
                            for(int X=xPre; X<xPre+4; X++){
                                xControlPointCoordinates[coord] = (DTYPE)xxPtr[X];
                                yControlPointCoordinates[coord] = (DTYPE)yyPtr[X];
                                zControlPointCoordinates[coord] = (DTYPE)zzPtr[X];
                                coord++;
                            }
                        }
                    }
                }
                oldBasis=basis;

                DTYPE Tx_x=0.0;
                DTYPE Ty_x=0.0;
                DTYPE Tz_x=0.0;
                DTYPE Tx_y=0.0;
                DTYPE Ty_y=0.0;
                DTYPE Tz_y=0.0;
                DTYPE Tx_z=0.0;
                DTYPE Ty_z=0.0;
                DTYPE Tz_z=0.0;

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
                DTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                *jacobianMapPtr++ = detJac;
            }
        }
    }
}
/* *************************************************************** */
void reg_bspline_GetJacobianMap(nifti_image *splineControlPoint,
                                nifti_image *jacobianImage)
{
    if(splineControlPoint->nz==1){
        switch(jacobianImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMap2D<float>(splineControlPoint, jacobianImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMap2D<double>(splineControlPoint, jacobianImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the jacobian map image\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian map has not computed\n");
                exit(1);
        }
    }else{
        switch(jacobianImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMap3D<float>(splineControlPoint, jacobianImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMap3D<double>(splineControlPoint, jacobianImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the jacobian map image\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian map has not computed\n");
                exit(1);
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_GetJacobianMatrix(nifti_image *splineControlPoint,
                                   nifti_image *jacobianImage)
{
    if(splineControlPoint->datatype != jacobianImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_GetJacobianMatrix\n");
        fprintf(stderr, "[NiftyReg ERROR] Input images were expected to be from the same type\n");
        exit(1);
    }

    unsigned int voxelNumber = jacobianImage->nx*jacobianImage->ny*jacobianImage->nz;

    mat33 *jacobianMatrices=(mat33 *)malloc(voxelNumber * sizeof(mat33));

    if(splineControlPoint->nz>1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_computeJacobianMatrices_3D<float>(jacobianImage,splineControlPoint,jacobianMatrices, NULL);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_computeJacobianMatrices_3D<double>(jacobianImage,splineControlPoint,jacobianMatrices, NULL);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the control point image\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian matrix image has not been computed\n");
                exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_computeJacobianMatrices_2D<float>(jacobianImage,splineControlPoint,jacobianMatrices, NULL);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_computeJacobianMatrices_2D<double>(jacobianImage,splineControlPoint,jacobianMatrices, NULL);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the control point image\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian matrix image has not been computed\n");
                exit(1);
        }
    }

    // The matrix are copied back in the nifti image
    switch(splineControlPoint->datatype){
        case NIFTI_TYPE_FLOAT32:
        {
            float *jacPtrXX=NULL;
            float *jacPtrXY=NULL;
            float *jacPtrXZ=NULL;
            float *jacPtrYX=NULL;
            float *jacPtrYY=NULL;
            float *jacPtrYZ=NULL;
            float *jacPtrZX=NULL;
            float *jacPtrZY=NULL;
            float *jacPtrZZ=NULL;
            if(jacobianImage->nz>1){
                jacPtrXX=static_cast<float *>(jacobianImage->data);
                jacPtrXY=&jacPtrXX[voxelNumber];
                jacPtrXZ=&jacPtrXY[voxelNumber];
                jacPtrYX=&jacPtrXZ[voxelNumber];
                jacPtrYY=&jacPtrYX[voxelNumber];
                jacPtrYZ=&jacPtrYY[voxelNumber];
                jacPtrZX=&jacPtrYZ[voxelNumber];
                jacPtrZY=&jacPtrZX[voxelNumber];
                jacPtrZZ=&jacPtrZY[voxelNumber];
            }
            else{
                jacPtrXX=static_cast<float *>(jacobianImage->data);
                jacPtrXY=&jacPtrXX[voxelNumber];
                jacPtrYX=&jacPtrYX[voxelNumber];
                jacPtrYY=&jacPtrXY[voxelNumber];
            }
            for(unsigned int i=0;i<voxelNumber;++i){
                jacPtrXX[i]=jacobianMatrices[i].m[0][0];
                jacPtrXY[i]=jacobianMatrices[i].m[0][1];
                jacPtrYX[i]=jacobianMatrices[i].m[1][0];
                jacPtrYY[i]=jacobianMatrices[i].m[1][1];
                if(jacobianImage->nz>1){
                    jacPtrXZ[i]=jacobianMatrices[i].m[0][2];
                    jacPtrYZ[i]=jacobianMatrices[i].m[1][2];
                    jacPtrZX[i]=jacobianMatrices[i].m[2][0];
                    jacPtrZY[i]=jacobianMatrices[i].m[2][1];
                    jacPtrZZ[i]=jacobianMatrices[i].m[2][2];
                }
            }
        }
        break;
        case NIFTI_TYPE_FLOAT64:
        {
            double *jacPtrXX=NULL;
            double *jacPtrXY=NULL;
            double *jacPtrXZ=NULL;
            double *jacPtrYX=NULL;
            double *jacPtrYY=NULL;
            double *jacPtrYZ=NULL;
            double *jacPtrZX=NULL;
            double *jacPtrZY=NULL;
            double *jacPtrZZ=NULL;
            if(jacobianImage->nz>1){
                jacPtrXX=static_cast<double *>(jacobianImage->data);
                jacPtrXY=&jacPtrXX[voxelNumber];
                jacPtrXZ=&jacPtrXY[voxelNumber];
                jacPtrYX=&jacPtrXZ[voxelNumber];
                jacPtrYY=&jacPtrYX[voxelNumber];
                jacPtrYZ=&jacPtrYY[voxelNumber];
                jacPtrZX=&jacPtrYZ[voxelNumber];
                jacPtrZY=&jacPtrZX[voxelNumber];
                jacPtrZZ=&jacPtrZY[voxelNumber];
            }
            else{
                jacPtrXX=static_cast<double *>(jacobianImage->data);
                jacPtrXY=&jacPtrXX[voxelNumber];
                jacPtrYX=&jacPtrYX[voxelNumber];
                jacPtrYY=&jacPtrXY[voxelNumber];
            }
            for(unsigned int i=0;i<voxelNumber;++i){
                jacPtrXX[i]=jacobianMatrices[i].m[0][0];
                jacPtrXY[i]=jacobianMatrices[i].m[0][1];
                jacPtrYX[i]=jacobianMatrices[i].m[1][0];
                jacPtrYY[i]=jacobianMatrices[i].m[1][1];
                if(jacobianImage->nz>1){
                    jacPtrXZ[i]=jacobianMatrices[i].m[0][2];
                    jacPtrYZ[i]=jacobianMatrices[i].m[1][2];
                    jacPtrZX[i]=jacobianMatrices[i].m[2][0];
                    jacPtrZY[i]=jacobianMatrices[i].m[2][1];
                    jacPtrZZ[i]=jacobianMatrices[i].m[2][2];
                }
            }
        }
        break;
    }
    free(jacobianMatrices);
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_defField_getJacobianMap2D(nifti_image *deformationField,
                                   nifti_image *jacobianDeterminant,
                                   nifti_image *jacobianMatrices)
{
    DTYPE *jacDetPtr = NULL;
    DTYPE *jacXXPtr = NULL;
    DTYPE *jacXYPtr = NULL;
    DTYPE *jacYXPtr = NULL;
    DTYPE *jacYYPtr = NULL;

    if(jacobianDeterminant!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDeterminant->data);

    unsigned int pixelNumber=deformationField->nx*deformationField->ny;
    if(jacobianMatrices!=NULL){
        jacXXPtr = static_cast<DTYPE *>(jacobianMatrices->data);
        jacXYPtr = &jacXXPtr[pixelNumber];
        jacYXPtr = &jacXYPtr[pixelNumber];
        jacYYPtr = &jacYXPtr[pixelNumber];
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(deformationField, &desorient, &reorient);

    mat44 *real2voxel=NULL;
    mat44 *voxel2real=NULL;
    if(deformationField->sform_code){
        real2voxel=&(deformationField->sto_ijk);
        voxel2real=&(deformationField->sto_xyz);
    }
    else{
        real2voxel=&(deformationField->qto_ijk);
        voxel2real=&(deformationField->qto_xyz);
    }

    DTYPE *deformationPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *deformationPtrY = &deformationPtrX[pixelNumber];

    unsigned int currentIndex=0;
    for(int y=0;y<deformationField->ny;++y){
        for(int x=0;x<deformationField->nx;++x){

            // Extract the current voxel deformation
            DTYPE realPosition[2]={deformationPtrX[currentIndex],
                                   deformationPtrY[currentIndex]};

            // Get the corresponding voxel position
            DTYPE voxelPosition[2];
            voxelPosition[0]=real2voxel->m[0][0] * realPosition[0] +
                             real2voxel->m[0][1] * realPosition[1] +
                             real2voxel->m[0][3];
            voxelPosition[1]=real2voxel->m[1][0] * realPosition[0] +
                             real2voxel->m[1][1] * realPosition[1] +
                             real2voxel->m[1][3];

            // Compute the relative positions
            int previous[2];
            previous[0]=(int)floor(voxelPosition[0]);
            previous[1]=(int)floor(voxelPosition[1]);
            DTYPE basisX[2], basisY[2], first[2]={-1.0,1.0};
            basisX[1]=voxelPosition[0]-(DTYPE)previous[0];basisX[0]=1.0-basisX[1];
            basisY[1]=voxelPosition[1]-(DTYPE)previous[1];basisY[0]=1.0-basisY[1];

            DTYPE firstX, firstY, defX, defY;
            memset(&jacobianMatrix,0,sizeof(mat33));
            for(int b=0;b<2;++b){
                int currentY=previous[1]+b;
                for(int a=0;a<2;++a){
                    int currentX=previous[0]+a;

                    firstX=first[a]*basisY[b];
                    firstY=basisX[a]*first[b];

                    if(currentX>-1 && currentX<deformationField->nx &&
                       currentY>-1 && currentY<deformationField->ny){
                        // Uses the deformation field if voxel is in its space
                        unsigned int index=currentY*deformationField->nx+currentX;
                        defX = deformationPtrX[index];
                        defY = deformationPtrY[index];
                    }
                    else{
                        // Uses the deformation field affine transformation
                        defX = voxel2real->m[0][0] * currentX +
                               voxel2real->m[0][1] * currentY +
                               voxel2real->m[0][3];
                        defY = voxel2real->m[1][0] * currentX +
                               voxel2real->m[1][1] * currentY +
                               voxel2real->m[1][3];
                    }//in space

                    jacobianMatrix.m[0][0] += firstX*defX;
                    jacobianMatrix.m[0][1] += firstY*defX;
                    jacobianMatrix.m[1][0] += firstX*defY;
                    jacobianMatrix.m[1][1] += firstY*defY;
                }//a
            }//b
            jacobianMatrix.m[0][0] /= deformationField->dx;
            jacobianMatrix.m[0][1] /= deformationField->dy;
            jacobianMatrix.m[1][0] /= deformationField->dx;
            jacobianMatrix.m[1][1] /= deformationField->dy;
            jacobianMatrix.m[2][2]=1.f;

            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);

            if(jacobianDeterminant!=NULL)
                jacDetPtr[currentIndex] = nifti_mat33_determ(jacobianMatrix);
            if(jacobianMatrices!=NULL){
                jacXXPtr[currentIndex]=jacobianMatrix.m[0][0];
                jacXYPtr[currentIndex]=jacobianMatrix.m[0][1];
                jacYXPtr[currentIndex]=jacobianMatrix.m[1][0];
                jacYYPtr[currentIndex]=jacobianMatrix.m[1][1];
            }

            currentIndex++;

        }// x jacImage
    }//y jacImage
}
/* *************************************************************** */
template <class DTYPE>
void reg_defField_getJacobianMap3D(nifti_image *deformationField,
                                   nifti_image *jacobianDeterminant,
                                   nifti_image *jacobianMatrices)
{
    DTYPE *jacDetPtr = NULL;
    DTYPE *jacXXPtr = NULL;
    DTYPE *jacXYPtr = NULL;
    DTYPE *jacXZPtr = NULL;
    DTYPE *jacYXPtr = NULL;
    DTYPE *jacYYPtr = NULL;
    DTYPE *jacYZPtr = NULL;
    DTYPE *jacZXPtr = NULL;
    DTYPE *jacZYPtr = NULL;
    DTYPE *jacZZPtr = NULL;

    if(jacobianDeterminant!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDeterminant->data);

    unsigned int voxelNumber=deformationField->nx*deformationField->ny*deformationField->nz;
    if(jacobianMatrices!=NULL){
        jacXXPtr = static_cast<DTYPE *>(jacobianMatrices->data);
        jacXYPtr = &jacXXPtr[voxelNumber];
        jacXZPtr = &jacXYPtr[voxelNumber];
        jacYXPtr = &jacXZPtr[voxelNumber];
        jacYYPtr = &jacYXPtr[voxelNumber];
        jacYZPtr = &jacYYPtr[voxelNumber];
        jacZXPtr = &jacYZPtr[voxelNumber];
        jacZYPtr = &jacZXPtr[voxelNumber];
        jacZZPtr = &jacZYPtr[voxelNumber];
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(deformationField, &desorient, &reorient);

    mat44 *real2voxel=NULL;
    mat44 *voxel2real=NULL;
    if(deformationField->sform_code){
        real2voxel=&(deformationField->sto_ijk);
        voxel2real=&(deformationField->sto_xyz);
    }
    else{
        real2voxel=&(deformationField->qto_ijk);
        voxel2real=&(deformationField->qto_xyz);
    }

    DTYPE *deformationPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *deformationPtrY = &deformationPtrX[voxelNumber];
    DTYPE *deformationPtrZ = &deformationPtrY[voxelNumber];

    unsigned int currentIndex=0;
    for(int z=0;z<deformationField->nz;++z){
        for(int y=0;y<deformationField->ny;++y){
            for(int x=0;x<deformationField->nx;++x){

                // Extract the current voxel deformation
                DTYPE realPosition[3]={deformationPtrX[currentIndex],
                                       deformationPtrY[currentIndex],
                                       deformationPtrZ[currentIndex]};

                // Get the corresponding voxel position
                DTYPE voxelPosition[3];
                voxelPosition[0]=real2voxel->m[0][0] * realPosition[0] +
                                 real2voxel->m[0][1] * realPosition[1] +
                                 real2voxel->m[0][2] * realPosition[2] +
                                 real2voxel->m[0][3];
                voxelPosition[1]=real2voxel->m[1][0] * realPosition[0] +
                                 real2voxel->m[1][1] * realPosition[1] +
                                 real2voxel->m[1][2] * realPosition[2] +
                                 real2voxel->m[1][3];
                voxelPosition[2]=real2voxel->m[2][0] * realPosition[0] +
                                 real2voxel->m[2][1] * realPosition[1] +
                                 real2voxel->m[2][2] * realPosition[2] +
                                 real2voxel->m[2][3];

                // Compute the relative positions
                int previous[3];
                previous[0]=(int)floor(voxelPosition[0]);
                previous[1]=(int)floor(voxelPosition[1]);
                previous[2]=(int)floor(voxelPosition[2]);
                DTYPE basisX[2], basisY[2], basisZ[2], first[2]={-1.0,1.0};
                basisX[1]=voxelPosition[0]-(DTYPE)previous[0];basisX[0]=1.0-basisX[1];
                basisY[1]=voxelPosition[1]-(DTYPE)previous[1];basisY[0]=1.0-basisY[1];
                basisZ[1]=voxelPosition[2]-(DTYPE)previous[2];basisZ[0]=1.0-basisZ[1];

                DTYPE firstX, firstY, firstZ, defX, defY, defZ;
                memset(&jacobianMatrix,0,sizeof(mat33));
                for(int c=0;c<2;++c){
                    int currentZ=previous[2]+c;
                    for(int b=0;b<2;++b){
                        int currentY=previous[1]+b;
                        for(int a=0;a<2;++a){
                            int currentX=previous[0]+a;

                            firstX=first[a]*basisY[b]*basisZ[c];
                            firstY=basisX[a]*first[b]*basisZ[c];
                            firstZ=basisX[a]*basisY[b]*first[c];

                            if(currentX>-1 && currentX<deformationField->nx &&
                               currentY>-1 && currentY<deformationField->ny &&
                               currentZ>-1 && currentZ<deformationField->nz){
                                // Uses the deformation field if voxel is in its space
                                unsigned int index=(currentZ*deformationField->ny+currentY)*deformationField->nx+currentX;
                                defX = deformationPtrX[index];
                                defY = deformationPtrY[index];
                                defZ = deformationPtrZ[index];
                            }
                            else{
                                // Uses the deformation field affine transformation
                                defX = voxel2real->m[0][0] * currentX +
                                       voxel2real->m[0][1] * currentY +
                                       voxel2real->m[0][2] * currentZ +
                                       voxel2real->m[0][3];
                                defY = voxel2real->m[1][0] * currentX +
                                       voxel2real->m[1][1] * currentY +
                                       voxel2real->m[1][2] * currentZ +
                                       voxel2real->m[1][3];
                                defZ = voxel2real->m[2][0] * currentX +
                                       voxel2real->m[2][1] * currentY +
                                       voxel2real->m[2][2] * currentZ +
                                       voxel2real->m[2][3];
                            }//in space

                            jacobianMatrix.m[0][0] += firstX*defX;
                            jacobianMatrix.m[0][1] += firstY*defX;
                            jacobianMatrix.m[0][2] += firstZ*defX;
                            jacobianMatrix.m[1][0] += firstX*defY;
                            jacobianMatrix.m[1][1] += firstY*defY;
                            jacobianMatrix.m[1][2] += firstZ*defY;
                            jacobianMatrix.m[2][0] += firstX*defZ;
                            jacobianMatrix.m[2][1] += firstY*defZ;
                            jacobianMatrix.m[2][2] += firstZ*defZ;
                        }//a
                    }//b
                }//c
                jacobianMatrix.m[0][0] /= deformationField->dx;
                jacobianMatrix.m[0][1] /= deformationField->dy;
                jacobianMatrix.m[0][2] /= deformationField->dz;
                jacobianMatrix.m[1][0] /= deformationField->dx;
                jacobianMatrix.m[1][1] /= deformationField->dy;
                jacobianMatrix.m[1][2] /= deformationField->dz;
                jacobianMatrix.m[2][0] /= deformationField->dx;
                jacobianMatrix.m[2][1] /= deformationField->dy;
                jacobianMatrix.m[2][2] /= deformationField->dz;

                jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                if(jacobianDeterminant!=NULL)
                    jacDetPtr[currentIndex] = nifti_mat33_determ(jacobianMatrix);
                if(jacobianMatrices!=NULL){
                    jacXXPtr[currentIndex]=jacobianMatrix.m[0][0];
                    jacXYPtr[currentIndex]=jacobianMatrix.m[0][1];
                    jacXZPtr[currentIndex]=jacobianMatrix.m[0][2];
                    jacYXPtr[currentIndex]=jacobianMatrix.m[1][0];
                    jacYYPtr[currentIndex]=jacobianMatrix.m[1][1];
                    jacYZPtr[currentIndex]=jacobianMatrix.m[1][2];
                    jacZXPtr[currentIndex]=jacobianMatrix.m[2][0];
                    jacZYPtr[currentIndex]=jacobianMatrix.m[2][1];
                    jacZZPtr[currentIndex]=jacobianMatrix.m[2][2];
                }

                currentIndex++;

            }// x jacImage
        }//y jacImage
    }//z jacImage
}
/* *************************************************************** */
void reg_defField_getJacobianMap(nifti_image *deformationField,
                                 nifti_image *jacobianImage)
{
    if(deformationField->datatype!=jacobianImage->datatype){
        printf("[NiftyReg ERROR] reg_defField_getJacobianMap\n");
        printf("[NiftyReg ERROR] Both input images have different type. Exit\n");
        exit(1);
    }
    switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            if(deformationField->nz>1)
                reg_defField_getJacobianMap3D<float>(deformationField,jacobianImage,NULL);
            else reg_defField_getJacobianMap2D<float>(deformationField,jacobianImage,NULL);
            break;
#ifdef _NR_DEV
        case NIFTI_TYPE_FLOAT64:
            if(deformationField->nz>1)
                reg_defField_getJacobianMap3D<double>(deformationField,jacobianImage,NULL);
            else reg_defField_getJacobianMap2D<double>(deformationField,jacobianImage,NULL);
            break;
#endif
        default:
            printf("[NiftyReg ERROR] reg_defField_getJacobianMap\n");
            printf("[NiftyReg ERROR] Voxel type unsupported.\n");
            exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_getJacobianMatrix(nifti_image *deformationField,
                                    nifti_image *jacobianImage)
{
    if(deformationField->datatype!=jacobianImage->datatype){
        printf("[NiftyReg ERROR] reg_defField_getJacobianMap\n");
        printf("[NiftyReg ERROR] Both input images have different type. Exit\n");
        exit(1);
    }
    switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            if(deformationField->nz>1)
                reg_defField_getJacobianMap3D<float>(deformationField,NULL,jacobianImage);
            else reg_defField_getJacobianMap2D<float>(deformationField,NULL,jacobianImage);
            break;
#ifdef _NR_DEV
        case NIFTI_TYPE_FLOAT64:
            if(deformationField->nz>1)
                reg_defField_getJacobianMap3D<double>(deformationField,NULL,jacobianImage);
            else reg_defField_getJacobianMap2D<double>(deformationField,NULL,jacobianImage);
            break;
#endif
        default:
            printf("[NiftyReg ERROR] reg_defField_getJacobianMap\n");
            printf("[NiftyReg ERROR] Voxel type unsupported.\n");
            exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_2D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage)
{
    // The initial deformation fields are allocated
    nifti_image *deformationFieldA = nifti_copy_nim_info(jacobianImage);
    deformationFieldA->dim[0]=deformationFieldA->ndim=5;
    deformationFieldA->dim[1]=deformationFieldA->nx=jacobianImage->nx;
    deformationFieldA->dim[2]=deformationFieldA->ny=jacobianImage->ny;
    deformationFieldA->dim[3]=deformationFieldA->nz=jacobianImage->nz;
    deformationFieldA->dim[4]=deformationFieldA->nt=1;
    deformationFieldA->pixdim[4]=deformationFieldA->dt=1.0;
    deformationFieldA->dim[5]=deformationFieldA->nu=velocityFieldImage->nu;
    deformationFieldA->pixdim[5]=deformationFieldA->du=1.0;
    deformationFieldA->dim[6]=deformationFieldA->nv=1;
    deformationFieldA->pixdim[6]=deformationFieldA->dv=1.0;
    deformationFieldA->dim[7]=deformationFieldA->nw=1;
    deformationFieldA->pixdim[7]=deformationFieldA->dw=1.0;
    deformationFieldA->nvox=deformationFieldA->nx *
                            deformationFieldA->ny *
                            deformationFieldA->nz *
                            deformationFieldA->nt *
                            deformationFieldA->nu;
    deformationFieldA->nbyper = jacobianImage->nbyper;
    deformationFieldA->datatype = jacobianImage->datatype;
    deformationFieldA->data = (void *)calloc(deformationFieldA->nvox, deformationFieldA->nbyper);
    nifti_image *deformationFieldB = nifti_copy_nim_info(deformationFieldA);
    deformationFieldB->data = (void *)calloc(deformationFieldB->nvox, deformationFieldB->nbyper);

    // The velocity field is scaled
    nifti_image *scaledVelocityField=nifti_copy_nim_info(velocityFieldImage);
    scaledVelocityField->data=(void *)malloc(scaledVelocityField->nvox*scaledVelocityField->nbyper);
    memcpy(scaledVelocityField->data, velocityFieldImage->data, scaledVelocityField->nvox*scaledVelocityField->nbyper);
    reg_getDisplacementFromDeformation(scaledVelocityField);
    reg_tools_addSubMulDivValue(scaledVelocityField, scaledVelocityField, pow(2,scaledVelocityField->pixdim[5]), 3);
    reg_getDeformationFromDisplacement(scaledVelocityField);

    // The initial deformation field is computed
    reg_spline(scaledVelocityField,
               deformationFieldA,
               deformationFieldA,
               NULL, // mask
               false, //composition
               true // bspline
               );
    nifti_image_free(scaledVelocityField);

    // The Jacobian determinant values are initialised to 1
    DTYPE *jacobianPtr = static_cast<DTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;++i) jacobianPtr[i]=1;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(deformationFieldA, &desorient, &reorient);

    mat44 *real2voxel=NULL;
    mat44 *voxel2real=NULL;
    if(deformationFieldA->sform_code){
        real2voxel=&(deformationFieldA->sto_ijk);
        voxel2real=&(deformationFieldA->sto_xyz);
    }
    else{
        real2voxel=&(deformationFieldA->qto_ijk);
        voxel2real=&(deformationFieldA->qto_xyz);
    }

    DTYPE *deformationAPtrX = static_cast<DTYPE *>(deformationFieldA->data);
    DTYPE *deformationAPtrY = &deformationAPtrX[jacobianImage->nvox];

    DTYPE *deformationBPtrX = static_cast<DTYPE *>(deformationFieldB->data);
    DTYPE *deformationBPtrY = &deformationBPtrX[jacobianImage->nvox];

    for(unsigned int i=0;i<velocityFieldImage->pixdim[5];++i){

        // The jacobian determinant is computed at every voxel
        unsigned int currentIndex=0;
            for(int y=0;y<jacobianImage->ny;++y){
                for(int x=0;x<jacobianImage->nx;++x){

                    // Extract the current voxel deformation
                    DTYPE realPosition[2]={
                        deformationAPtrX[currentIndex],
                        deformationAPtrY[currentIndex]
                        };

                    // Get the corresponding voxel position
                    DTYPE voxelPosition[2];
                    voxelPosition[0]=real2voxel->m[0][0] * realPosition[0] +
                                     real2voxel->m[0][1] * realPosition[1] +
                                     real2voxel->m[0][3];
                    voxelPosition[1]=real2voxel->m[1][0] * realPosition[0] +
                                     real2voxel->m[1][1] * realPosition[1] +
                                     real2voxel->m[1][3];

                    // Compute the relative positions
                    int previous[2];
                    previous[0]=(int)floor(voxelPosition[0]);
                    previous[1]=(int)floor(voxelPosition[1]);
                    DTYPE basisX[2], basisY[2], first[2]={-1.0,1.0};
                    basisX[1]=voxelPosition[0]-(DTYPE)previous[0];basisX[0]=1.0-basisX[1];
                    basisY[1]=voxelPosition[1]-(DTYPE)previous[1];basisY[0]=1.0-basisY[1];

                    DTYPE defX, defY, firstX, firstY, basis;
                    memset(&jacobianMatrix,0,sizeof(mat33));
                    DTYPE newDefX=0.0, newDefY=0.0;
                        for(int b=0;b<2;++b){
                            int currentY=previous[1]+b;
                            for(int a=0;a<2;++a){
                                int currentX=previous[0]+a;

                                firstX=first[a]*basisY[b];
                                firstY=basisX[a]*first[b];
                                basis=basisX[a]*basisY[b];

                                if(currentX>-1 && currentX<deformationFieldA->nx &&
                                   currentY>-1 && currentY<deformationFieldA->ny){
                                    // Uses the deformation field if voxel is in its space
                                    unsigned int index=currentY*deformationFieldA->nx+currentX;
                                    defX = deformationAPtrX[index];
                                    defY = deformationAPtrY[index];
                                }
                                else{
                                    // Uses the deformation field affine transformation
                                    defX = voxel2real->m[0][0] * currentX +
                                           voxel2real->m[0][1] * currentY +
                                           voxel2real->m[0][3];
                                    defY = voxel2real->m[1][0] * currentX +
                                           voxel2real->m[1][1] * currentY +
                                           voxel2real->m[1][3];
                                }//in space

                                newDefX += basis * defX;
                                newDefY += basis * defY;
                                jacobianMatrix.m[0][0] += firstX*defX;
                                jacobianMatrix.m[0][1] += firstY*defX;
                                jacobianMatrix.m[1][0] += firstX*defY;
                                jacobianMatrix.m[1][1] += firstY*defY;
                            }//a
                        }//b
                    jacobianMatrix.m[0][0] /= deformationFieldA->dx;
                    jacobianMatrix.m[0][1] /= deformationFieldA->dy;
                    jacobianMatrix.m[1][0] /= deformationFieldA->dx;
                    jacobianMatrix.m[1][1] /= deformationFieldA->dy;
                    jacobianMatrix.m[2][2]=1.f;

                    jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                    DTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                    jacobianPtr[currentIndex] *= detJac;
                    deformationBPtrX[currentIndex]=newDefX;
                    deformationBPtrY[currentIndex]=newDefY;
                    currentIndex++;

                }// x jacImage
            }//y jacImage
        if(i!=velocityFieldImage->pixdim[5]-1)
            memcpy(deformationFieldA->data,deformationFieldB->data,
                   deformationFieldA->nvox*deformationFieldA->nbyper);
    }//composition step

    nifti_image_free(deformationFieldA);
    nifti_image_free(deformationFieldB);

}
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_3D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage)
{
    // The initial deformation fields are allocated
    nifti_image *deformationFieldA = nifti_copy_nim_info(jacobianImage);
    deformationFieldA->dim[0]=deformationFieldA->ndim=5;
    deformationFieldA->dim[1]=deformationFieldA->nx=jacobianImage->nx;
    deformationFieldA->dim[2]=deformationFieldA->ny=jacobianImage->ny;
    deformationFieldA->dim[3]=deformationFieldA->nz=jacobianImage->nz;
    deformationFieldA->dim[4]=deformationFieldA->nt=1;
    deformationFieldA->pixdim[4]=deformationFieldA->dt=1.0;
    deformationFieldA->dim[5]=deformationFieldA->nu=velocityFieldImage->nu;
    deformationFieldA->pixdim[5]=deformationFieldA->du=1.0;
    deformationFieldA->dim[6]=deformationFieldA->nv=1;
    deformationFieldA->pixdim[6]=deformationFieldA->dv=1.0;
    deformationFieldA->dim[7]=deformationFieldA->nw=1;
    deformationFieldA->pixdim[7]=deformationFieldA->dw=1.0;
    deformationFieldA->nvox=deformationFieldA->nx *
                            deformationFieldA->ny *
                            deformationFieldA->nz *
                            deformationFieldA->nt *
                            deformationFieldA->nu;
    deformationFieldA->nbyper = jacobianImage->nbyper;
    deformationFieldA->datatype = jacobianImage->datatype;
    deformationFieldA->data = (void *)calloc(deformationFieldA->nvox, deformationFieldA->nbyper);
    nifti_image *deformationFieldB = nifti_copy_nim_info(deformationFieldA);
    deformationFieldB->data = (void *)calloc(deformationFieldB->nvox, deformationFieldB->nbyper);

    // The initial deformation field is computed
    reg_spline(velocityFieldImage,
               jacobianImage,
               deformationFieldA,
               NULL, // mask
               false, //composition
               true // bspline
               );

    // The Jacobian determinant values are initialised to 1
    DTYPE *jacobianPtr = static_cast<DTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;++i) jacobianPtr[i]=1;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(jacobianImage, &desorient, &reorient);

    mat44 *real2voxel=NULL;
    mat44 *voxel2real=NULL;
    if(deformationFieldA->sform_code){
        real2voxel=&(deformationFieldA->sto_ijk);
        voxel2real=&(deformationFieldA->sto_xyz);
    }
    else{
        real2voxel=&(deformationFieldA->qto_ijk);
        voxel2real=&(deformationFieldA->qto_xyz);
    }

    DTYPE *deformationAPtrX = static_cast<DTYPE *>(deformationFieldA->data);
    DTYPE *deformationAPtrY = &deformationAPtrX[jacobianImage->nvox];
    DTYPE *deformationAPtrZ = &deformationAPtrY[jacobianImage->nvox];

    DTYPE *deformationBPtrX = static_cast<DTYPE *>(deformationFieldB->data);
    DTYPE *deformationBPtrY = &deformationBPtrX[jacobianImage->nvox];
    DTYPE *deformationBPtrZ = &deformationBPtrY[jacobianImage->nvox];

    for(unsigned int i=0;i<velocityFieldImage->pixdim[5];++i){

        // The jacobian determinant is computed at every voxel
        unsigned int currentIndex=0;
        for(int z=0;z<jacobianImage->nz;++z){
            for(int y=0;y<jacobianImage->ny;++y){
                for(int x=0;x<jacobianImage->nx;++x){

                    // Extract the current voxel deformation
                    DTYPE realPosition[3]={
                        deformationAPtrX[currentIndex],
                        deformationAPtrY[currentIndex],
                        deformationAPtrZ[currentIndex]
                        };

                    // Get the corresponding voxel position
                    DTYPE voxelPosition[3];
                    voxelPosition[0]=real2voxel->m[0][0] * realPosition[0] +
                                     real2voxel->m[0][1] * realPosition[1] +
                                     real2voxel->m[0][2] * realPosition[2] +
                                     real2voxel->m[0][3];
                    voxelPosition[1]=real2voxel->m[1][0] * realPosition[0] +
                                     real2voxel->m[1][1] * realPosition[1] +
                                     real2voxel->m[1][2] * realPosition[2] +
                                     real2voxel->m[1][3];
                    voxelPosition[2]=real2voxel->m[2][0] * realPosition[0] +
                                     real2voxel->m[2][1] * realPosition[1] +
                                     real2voxel->m[2][2] * realPosition[2] +
                                     real2voxel->m[2][3];

                    // Compute the relative positions
                    int previous[3];
                    previous[0]=(int)floor(voxelPosition[0]);
                    previous[1]=(int)floor(voxelPosition[1]);
                    previous[2]=(int)floor(voxelPosition[2]);
                    DTYPE basisX[2], basisY[2], basisZ[2], first[2]={-1,1};
                    basisX[1]=voxelPosition[0]-(DTYPE)previous[0];basisX[0]=1.-basisX[1];
                    basisY[1]=voxelPosition[1]-(DTYPE)previous[1];basisY[0]=1.-basisY[1];
                    basisZ[1]=voxelPosition[2]-(DTYPE)previous[2];basisZ[0]=1.-basisZ[1];

                    DTYPE defX, defY, defZ, basisCoeff[4];
                    memset(&jacobianMatrix,0,sizeof(mat33));
                    DTYPE newDefX=0.0, newDefY=0.0, newDefZ=0.0;
                    for(int c=0;c<2;++c){
                        int currentZ=previous[2]+c;
                        for(int b=0;b<2;++b){
                            int currentY=previous[1]+b;
                            for(int a=0;a<2;++a){
                                int currentX=previous[0]+a;

                                basisCoeff[0]=basisX[a]*basisY[b]*basisZ[c];
                                basisCoeff[1]=first[a]*basisY[b]*basisZ[c];
                                basisCoeff[2]=basisX[a]*first[b]*basisZ[c];
                                basisCoeff[3]=basisX[a]*basisY[b]*first[c];

                                if(currentX>-1 && currentX<deformationFieldA->nx &&
                                   currentY>-1 && currentY<deformationFieldA->ny &&
                                   currentZ>-1 && currentZ<deformationFieldA->nz){
                                    // Uses the deformation field if voxel is in its space
                                    int index=(currentZ*deformationFieldA->ny+currentY)
                                              *deformationFieldA->nx+currentX;
                                    defX = deformationAPtrX[index];
                                    defY = deformationAPtrY[index];
                                    defZ = deformationAPtrZ[index];
                                }
                                else{
                                    // Uses the deformation field affine transformation
                                    defX = voxel2real->m[0][0] * currentX +
                                           voxel2real->m[0][1] * currentY +
                                           voxel2real->m[0][2] * currentZ +
                                           voxel2real->m[0][3];
                                    defY = voxel2real->m[1][0] * currentX +
                                           voxel2real->m[1][1] * currentY +
                                           voxel2real->m[1][2] * currentZ +
                                           voxel2real->m[1][3];
                                    defZ = voxel2real->m[2][0] * currentX +
                                           voxel2real->m[2][1] * currentY +
                                           voxel2real->m[2][2] * currentZ +
                                           voxel2real->m[2][3];
                                }//padding

                                newDefX += basisCoeff[0] * defX;
                                newDefY += basisCoeff[0] * defY;
                                newDefZ += basisCoeff[0] * defZ;
                                jacobianMatrix.m[0][0] += basisCoeff[1]*defX;
                                jacobianMatrix.m[0][1] += basisCoeff[2]*defX;
                                jacobianMatrix.m[0][2] += basisCoeff[3]*defX;
                                jacobianMatrix.m[1][0] += basisCoeff[1]*defY;
                                jacobianMatrix.m[1][1] += basisCoeff[2]*defY;
                                jacobianMatrix.m[1][2] += basisCoeff[3]*defY;
                                jacobianMatrix.m[2][0] += basisCoeff[1]*defZ;
                                jacobianMatrix.m[2][1] += basisCoeff[2]*defZ;
                                jacobianMatrix.m[2][2] += basisCoeff[3]*defZ;
                            }//a
                        }//b
                    }//c
                    jacobianMatrix.m[0][0] /= deformationFieldA->dx;
                    jacobianMatrix.m[0][1] /= deformationFieldA->dy;
                    jacobianMatrix.m[0][2] /= deformationFieldA->dz;
                    jacobianMatrix.m[1][0] /= deformationFieldA->dx;
                    jacobianMatrix.m[1][1] /= deformationFieldA->dy;
                    jacobianMatrix.m[1][2] /= deformationFieldA->dz;
                    jacobianMatrix.m[2][0] /= deformationFieldA->dx;
                    jacobianMatrix.m[2][1] /= deformationFieldA->dy;
                    jacobianMatrix.m[2][2] /= deformationFieldA->dz;

                    jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);
                    DTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                    jacobianPtr[currentIndex] *= detJac;
                    deformationBPtrX[currentIndex]=newDefX;
                    deformationBPtrY[currentIndex]=newDefY;
                    deformationBPtrZ[currentIndex]=newDefZ;
                    currentIndex++;

                }// x jacImage
            }//y jacImage
        }//z jacImage
        if(i!=velocityFieldImage->pixdim[5]-1)
            memcpy(deformationFieldA->data,deformationFieldB->data,
                   deformationFieldA->nvox*deformationFieldA->nbyper);
    }//composition step

    nifti_image_free(deformationFieldA);
    nifti_image_free(deformationFieldB);

}
/* *************************************************************** */
int reg_bspline_GetJacobianMapFromVelocityField(nifti_image* velocityFieldImage,
                                                nifti_image* jacobianImage)
{
    if(velocityFieldImage->datatype != jacobianImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField\n");
        fprintf(stderr,"[NiftyReg ERROR] Input and output image do not have the same data type\n");
        return 1;
    }
    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField_3D\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
                return 1;
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField_3D\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
                return 1;
                break;
        }
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
