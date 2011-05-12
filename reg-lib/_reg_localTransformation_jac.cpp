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

/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_BSplineBasisValue(DTYPE basis, int index, DTYPE &value)
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
void Get_BSplineBasisValue(DTYPE basis, int index, DTYPE &value, DTYPE &first)
{
    Get_BSplineBasisValue<DTYPE>(basis, index, value);
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
void Get_BSplineBasisValue(DTYPE basis, int index, DTYPE &value, DTYPE &first, DTYPE &second)
{
    Get_BSplineBasisValue<DTYPE>(basis, index, value, first);
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
template<class SplineTYPE>
double reg_bspline_jacobianValue2D(nifti_image *splineControlPoint,
                                   nifti_image *targetImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>
                                   (splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
                                   (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    SplineTYPE yBasis[4],yFirst[4],temp[4],first[4];
    SplineTYPE basisX[16], basisY[16];
    SplineTYPE basis, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[16];
    SplineTYPE yControlPointCoordinates[16];

    SplineTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    double constraintValue=0;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
        basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, yBasis, yFirst);

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
            basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, temp, first);

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*first[a];   // y * x'
                    basisY[coord]=yFirst[b]*temp[a];    // y'* x
                    coord++;
                }
            }

            if(basis<=oldBasis || x==0){
                get_GridValues<SplineTYPE>(xPre,
                                           yPre,
                                           splineControlPoint,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates,
                                           false);
            }
            oldBasis=basis;

            SplineTYPE Tx_x=0.0;
            SplineTYPE Ty_x=0.0;
            SplineTYPE Tx_y=0.0;
            SplineTYPE Ty_y=0.0;

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
//                double logJac = log(detJac);
//                constraintValue += logJac*logJac;
                constraintValue +=  fabs(log(detJac));
            }
            else return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return constraintValue/(double)targetImage->nvox;
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_jacobianValue3D(nifti_image *splineControlPoint,
                                   nifti_image *targetImage)
{
#if _USE_SSE
    if(sizeof(SplineTYPE)!=4){
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_jacobianValue3D\n");
        fprintf(stderr, "[NiftyReg ERROR] The SSE implementation assume single precision... Exit\n");
        exit(1);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>
                                   (splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
                                   (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>
                                   (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    SplineTYPE zBasis[4],zFirst[4],temp[4],first[4];
    SplineTYPE tempX[16], tempY[16], tempZ[16];
    SplineTYPE basisX[64], basisY[64], basisZ[64];
    SplineTYPE basis, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[64];
    SplineTYPE yControlPointCoordinates[64];
    SplineTYPE zControlPointCoordinates[64];

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    double constraintValue=0;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((SplineTYPE)z/gridVoxelSpacing[2]);
        basis=(SplineTYPE)z/gridVoxelSpacing[2]-(SplineTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, zBasis, zFirst);

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
            basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, temp, first);

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

                int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
                basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                Get_BSplineBasisValues<SplineTYPE>(basis, temp, first);

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
                    get_GridValues<SplineTYPE>(xPre,
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
//                    double logJac = log(detJac);
//                    constraintValue += logJac*logJac;
                    constraintValue +=  fabs(log(detJac));
                }
                else return std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    return constraintValue/(double)targetImage->nvox;
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_jacobianApproxValue2D(nifti_image *splineControlPoint)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE basisX[9], basisY[9], constraintValue=0, xControlPointCoordinates[9], yControlPointCoordinates[9];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0.0, 0.5};
    unsigned int coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisX[coord]=normal[b]*first[a];   // y * x'
            basisY[coord]=first[b]*normal[a];   // y'* x
            coord++;
        }
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    for(int y=1;y<splineControlPoint->ny-2;y++){
        for(int x=1;x<splineControlPoint->nx-2;x++){

            get_GridValuesApprox<SplineTYPE>(x-1,
                                             y-1,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false);

            SplineTYPE Tx_x=0.0;
            SplineTYPE Ty_x=0.0;
            SplineTYPE Tx_y=0.0;
            SplineTYPE Ty_y=0.0;

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
            SplineTYPE detJac = nifti_mat33_determ(jacobianMatrix);

            if(detJac>0.0){
//                PrecisionTYPE logJac = log(detJac);
//                constraintValue += logJac*logJac;
                constraintValue +=  fabs(log(detJac));
            }
            else return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return constraintValue/(double)((splineControlPoint->nx-2)*(splineControlPoint->ny-2));
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_jacobianApproxValue3D(nifti_image *splineControlPoint)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    float basisX[27], basisY[27], basisZ[27];
    SplineTYPE xControlPointCoordinates[27], yControlPointCoordinates[27], zControlPointCoordinates[27];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    // There are six different values taken into account
    SplineTYPE tempX[9], tempY[9], tempZ[9];
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

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
        (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>
        (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    double constraintValue=0.0;
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            for(int x=1;x<splineControlPoint->nx-1;x++){

                get_GridValuesApprox<SplineTYPE>(x-1,
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

                SplineTYPE Tx_x=0.0;
                SplineTYPE Ty_x=0.0;
                SplineTYPE Tz_x=0.0;
                SplineTYPE Tx_y=0.0;
                SplineTYPE Ty_y=0.0;
                SplineTYPE Tz_y=0.0;
                SplineTYPE Tx_z=0.0;
                SplineTYPE Ty_z=0.0;
                SplineTYPE Tz_z=0.0;

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
                SplineTYPE detJac = nifti_mat33_determ(jacobianMatrix);

                if(detJac>0.0){
//                    double logJac = log(detJac);
//                    constraintValue += logJac*logJac;
                    constraintValue +=  fabs(log(detJac));
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
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                if(approx)
                    return reg_bspline_jacobianApproxValue2D<double>(splineControlPoint);
                else return reg_bspline_jacobianValue2D<double>(splineControlPoint, targetImage);
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
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                if(approx)
                    return reg_bspline_jacobianApproxValue3D<double>(splineControlPoint);
                else return reg_bspline_jacobianValue3D<double>(splineControlPoint, targetImage);
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
template <class SplineTYPE>
void computeJacobianMatrices_2D(nifti_image *targetImage,
                                nifti_image *splineControlPoint,
                                mat33 *invertedJacobianMatrices,
                                SplineTYPE *jacobianDeterminant)
{

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    SplineTYPE yBasis[4],yFirst[4],xBasis[4],xFirst[4];
    SplineTYPE basisX[16], basisY[16], basis;
    int oldXpre=9999999, oldYpre=9999999;

    SplineTYPE xControlPointCoordinates[16];
    SplineTYPE yControlPointCoordinates[16];

    SplineTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The inverted Jacobian matrices are first computed for every voxel
    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    SplineTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
        basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, yBasis, yFirst);

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
            basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, xBasis, xFirst);

            coord=0;
            for(int b=0; b<4; b++){
                for(int a=0; a<4; a++){
                    basisX[coord]=yBasis[b]*xFirst[a];   // y * x'
                    basisY[coord]=yFirst[b]*xBasis[a];    // y'* x
                    coord++;
                }
            }

            if(xPre!=oldXpre || yPre!=oldYpre){
                get_GridValues<SplineTYPE>(xPre,
                                           yPre,
                                           splineControlPoint,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates,
                                           false);
                oldXpre=xPre;oldYpre=yPre;
            }

            SplineTYPE Tx_x=0.0;
            SplineTYPE Ty_x=0.0;
            SplineTYPE Tx_y=0.0;
            SplineTYPE Ty_y=0.0;

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
template <class SplineTYPE>
void computeJacobianMatrices_3D(nifti_image *targetImage,
                                nifti_image *splineControlPoint,
                                mat33 *invertedJacobianMatrices,
                                SplineTYPE *jacobianDeterminant)
{
#if _USE_SSE
    if(sizeof(SplineTYPE)!=4){
        fprintf(stderr, "[NiftyReg ERROR] computeJacobianMatrices_3D\n");
        fprintf(stderr, "[NiftyReg ERROR] The SSE implementation assume single precision... Exit\n");
        exit(1);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    SplineTYPE yBasis[4], yFirst[4], xBasis[4], xFirst[4] ,zBasis[4] ,zFirst[4], basis;
    SplineTYPE tempX[16], tempY[16], tempZ[16], basisX[64], basisY[64], basisZ[64];
    int oldXpre=999999, oldYpre=999999, oldZpre=999999;

    SplineTYPE xControlPointCoordinates[64];
    SplineTYPE yControlPointCoordinates[64];
    SplineTYPE zControlPointCoordinates[64];

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The inverted Jacobian matrices are first computed for every voxel
    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    SplineTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((SplineTYPE)z/gridVoxelSpacing[2]);
        basis=(SplineTYPE)z/gridVoxelSpacing[2]-(SplineTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, zBasis, zFirst);

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
            basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, yBasis, yFirst);
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
            unsigned int coord=0;
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
                Get_BSplineBasisValues<SplineTYPE>(basis, xBasis, xFirst);

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

                if(xPre!=oldXpre || yPre!=oldYpre || zPre!=oldZpre){
                    get_GridValues<SplineTYPE>(xPre,
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
template <class SplineTYPE>
void computeApproximateJacobianMatrices_2D( nifti_image *splineControlPoint,
                                            mat33 *invertedJacobianMatrices,
                                            SplineTYPE *jacobianDeterminant)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE basisX[9], basisY[9], xControlPointCoordinates[9], yControlPointCoordinates[9];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0.0, 0.5};
    unsigned int coord=0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisX[coord]=normal[b]*first[a];   // y * x'
            basisY[coord]=first[b]*normal[a];   // y'* x
            coord++;
        }
    }

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
        (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    mat33 *invertedJacobianMatricesPtr = invertedJacobianMatrices;
    SplineTYPE *jacobianDeterminantPtr = jacobianDeterminant;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* All the Jacobian matrices are computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    // Loop over (almost) each control point
    for(int y=1;y<splineControlPoint->ny-1;y++){
        unsigned int jacIndex = y*splineControlPoint->nx + 1;
        for(int x=1;x<splineControlPoint->nx-1;x++){

            // The control points are stored
            get_GridValuesApprox<SplineTYPE>(x-1,
                                             y-1,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false);

            SplineTYPE Tx_x=(SplineTYPE)0.0;
            SplineTYPE Ty_x=(SplineTYPE)0.0;
            SplineTYPE Tx_y=(SplineTYPE)0.0;
            SplineTYPE Ty_y=(SplineTYPE)0.0;

            for(int a=0; a<9; a++){
                Tx_x += basisX[a]*xControlPointCoordinates[a];
                Tx_y += basisY[a]*xControlPointCoordinates[a];
                Ty_x += basisX[a]*yControlPointCoordinates[a];
                Ty_y += basisY[a]*yControlPointCoordinates[a];
            }

            memset(&jacobianMatrix, 0, sizeof(mat33));
            jacobianMatrix.m[0][0] = (float)(Tx_x / splineControlPoint->dx);
            jacobianMatrix.m[0][1] = (float)(Tx_y / splineControlPoint->dy);
            jacobianMatrix.m[1][0] = (float)(Ty_x / splineControlPoint->dx);
            jacobianMatrix.m[1][1] = (float)(Ty_y / splineControlPoint->dy);
            jacobianMatrix.m[2][2] = 1.0f;

            jacobianMatrix=nifti_mat33_mul(reorient,jacobianMatrix);

            jacobianDeterminantPtr[jacIndex] = (jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1])
                                        - (jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0]);
            invertedJacobianMatricesPtr[jacIndex] = nifti_mat33_inverse(jacobianMatrix);
            jacIndex++;
        } // x
    } // y
}
/* *************************************************************** */
template <class SplineTYPE>
void computeApproximateJacobianMatrices_3D( nifti_image *splineControlPoint,
                                            mat33 *invertedJacobianMatrices,
                                            SplineTYPE *jacobianDeterminant)
{
    // As the contraint is only computed at the voxel position, the basis values of the spline are always the same
    float basisX[27], basisY[27], basisZ[27];
    SplineTYPE xControlPointCoordinates[27], yControlPointCoordinates[27], zControlPointCoordinates[27];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    // There are six different values taken into account
    SplineTYPE tempX[9], tempY[9], tempZ[9];
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

    // Loop over (almost) each control point
    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            unsigned int jacIndex = (z*splineControlPoint->ny+y)*splineControlPoint->nx+1;
            for(int x=1;x<splineControlPoint->nx-1;x++){

                // The control points are stored
                get_GridValuesApprox<SplineTYPE>(x-1,
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
template<class SplineTYPE>
void reg_bspline_jacobianDeterminantGradient2D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(targetImage->nvox * sizeof(SplineTYPE));

    computeJacobianMatrices_2D<SplineTYPE>(targetImage,
                                           splineControlPoint,
                                           invertedJacobianMatrices,
                                           jacobianDeterminant);

    SplineTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    SplineTYPE basisValues[2];
    SplineTYPE xBasis=0, yBasis=0, basis;
    SplineTYPE xFirst=0, yFirst=0;
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The gradient are now computed for every control point
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            SplineTYPE jacobianConstraintX=(SplineTYPE)0.0;
            SplineTYPE jacobianConstraintY=(SplineTYPE)0.0;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)reg_ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)reg_floor((y+1)*gridVoxelSpacing[1]); pixelY++){
                if(pixelY>-1 && pixelY<targetImage->ny){

                    int yPre=(int)((SplineTYPE)pixelY/gridVoxelSpacing[1]);
                    basis=(SplineTYPE)pixelY/gridVoxelSpacing[1]-(SplineTYPE)yPre;
                    if(basis<0.0) basis=0.0; //rounding error
                    Get_BSplineBasisValue<SplineTYPE>(basis,y-yPre,yBasis,yFirst);

                    for(int pixelX=(int)reg_ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)reg_floor((x+1)*gridVoxelSpacing[0]); pixelX++){
                        if(pixelX>-1 && pixelX<targetImage->nx){

                            int xPre=(int)((SplineTYPE)pixelX/gridVoxelSpacing[0]);
                            basis=(SplineTYPE)pixelX/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                            if(basis<0.0) basis=0.0; //rounding error
                            Get_BSplineBasisValue<SplineTYPE>(basis,x-xPre,xBasis,xFirst);

                            basisValues[0]= xFirst * yBasis;
                            basisValues[1]= xBasis * yFirst;
                            jacIndex = pixelY*targetImage->nx+pixelX;
                            SplineTYPE detJac=jacobianDeterminant[jacIndex];
                            jacobianMatrix = invertedJacobianMatrices[jacIndex];

                            if(detJac>(SplineTYPE)0.0){
                                /* derivative of the squared log of the Jacobian determinant */
//                                logDet=(double)(2.0*log(logDet));
                                detJac = log(detJac)>0?1.0:-1.0;
                                jacobianConstraintX += detJac * (jacobianMatrix.m[0][0]*basisValues[0] +
                                                                 jacobianMatrix.m[0][1]*basisValues[1]);
                                jacobianConstraintY += detJac * (jacobianMatrix.m[1][0]*basisValues[0] +
                                                                 jacobianMatrix.m[1][1]*basisValues[1]);
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
template<class SplineTYPE>
void reg_bspline_jacobianDeterminantGradientApprox2D(  nifti_image *splineControlPoint,
                                                        nifti_image *targetImage,
                                                        nifti_image *gradientImage,
                                                        float weight)
{
    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(jacobianNumber * sizeof(SplineTYPE));

    computeApproximateJacobianMatrices_2D<SplineTYPE>(splineControlPoint,
                                                      invertedJacobianMatrices,
                                                      jacobianDeterminant);

    SplineTYPE basisValues[2];
    SplineTYPE xBasis=0, yBasis=0;
    SplineTYPE xFirst=0, yFirst=0;
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>
                                    (gradientImage->data);
    SplineTYPE *gradientImagePtrY = static_cast<SplineTYPE *>
                                    (&gradientImagePtrX[gradientImage->nx*gradientImage->ny]);

    SplineTYPE approxRatio = weight * (SplineTYPE)(targetImage->nx*targetImage->ny)
        / (SplineTYPE)(jacobianNumber);

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            SplineTYPE jacobianConstraintX=(SplineTYPE)0.0;
            SplineTYPE jacobianConstraintY=(SplineTYPE)0.0;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)(y-1);pixelY<(int)(y+2); pixelY++){
                if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                    Get_BSplineBasisValue<SplineTYPE>(0,y-pixelY+1, yBasis, yFirst);

                    for(int pixelX=(int)(x-1);pixelX<(int)(x+2); pixelX++){
                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                            jacIndex = pixelY*splineControlPoint->nx+pixelX;
                            SplineTYPE detJac=jacobianDeterminant[jacIndex];

                            if(detJac>(SplineTYPE)0.0){
                                Get_BSplineBasisValue<SplineTYPE>(0,x-pixelX+1,xBasis,xFirst);

                                basisValues[0] = xFirst * yBasis ;
                                basisValues[1] = xBasis * yFirst ;

                                jacobianMatrix = invertedJacobianMatrices[jacIndex];

                                /* derivative of the squared log of the Jacobian determinant */
//                                logDet=(SplineTYPE)(2.0*log(logDet));
                                detJac = log(detJac)>0?1.0:-1.0;
                                jacobianConstraintX += detJac * (jacobianMatrix.m[0][0]*basisValues[0] + jacobianMatrix.m[0][1]*basisValues[1]);
                                jacobianConstraintY += detJac * (jacobianMatrix.m[1][0]*basisValues[0] + jacobianMatrix.m[1][1]*basisValues[1]);
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
template<class SplineTYPE>
void reg_bspline_jacobianDeterminantGradient3D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(targetImage->nvox * sizeof(SplineTYPE));

    computeJacobianMatrices_3D<SplineTYPE>(targetImage,
                                           splineControlPoint,
                                           invertedJacobianMatrices,
                                           jacobianDeterminant);

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    SplineTYPE xBasis=0, yBasis=0, zBasis=0, basis;
    SplineTYPE xFirst=0, yFirst=0, zFirst=0;
    SplineTYPE basisValues[3];
    unsigned int jacIndex;

    // The gradient are now computed for every control point
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz];
    SplineTYPE *gradientImagePtrZ = &gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                SplineTYPE jacobianConstraintX=(SplineTYPE)0.0;
                SplineTYPE jacobianConstraintY=(SplineTYPE)0.0;
                SplineTYPE jacobianConstraintZ=(SplineTYPE)0.0;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)reg_ceil((z-3)*gridVoxelSpacing[2]);pixelZ<=(int)reg_floor((z+1)*gridVoxelSpacing[2]); pixelZ++){
                    if(pixelZ>-1 && pixelZ<targetImage->nz){

                        int zPre=(int)((SplineTYPE)pixelZ/gridVoxelSpacing[2]);
                        basis=(SplineTYPE)pixelZ/gridVoxelSpacing[2]-(SplineTYPE)zPre;
                        if(basis<0.0) basis=0.0; //rounding error
                        Get_BSplineBasisValue<SplineTYPE>(basis,z-zPre,zBasis,zFirst);

                        for(int pixelY=(int)reg_ceil((y-3)*gridVoxelSpacing[1]);pixelY<(int)reg_floor((y+1)*gridVoxelSpacing[1]); pixelY++){
                            if(pixelY>-1 && pixelY<targetImage->ny){

                                int yPre=(int)((SplineTYPE)pixelY/gridVoxelSpacing[1]);
                                basis=(SplineTYPE)pixelY/gridVoxelSpacing[1]-(SplineTYPE)yPre;
                                if(basis<0.0) basis=0.0; //rounding error
                                Get_BSplineBasisValue<SplineTYPE>(basis,y-yPre,yBasis,yFirst);

                                for(int pixelX=(int)reg_ceil((x-3)*gridVoxelSpacing[0]);pixelX<(int)reg_floor((x+1)*gridVoxelSpacing[0]); pixelX++){
                                    if(pixelX>-1 && pixelX<targetImage->nx){

                                        jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+pixelX;
                                        jacobianMatrix = invertedJacobianMatrices[jacIndex];
                                        SplineTYPE detJac = jacobianDeterminant[jacIndex];
                                        if(detJac>0.0){

                                            int xPre=(int)((SplineTYPE)pixelX/gridVoxelSpacing[0]);
                                            basis=(SplineTYPE)pixelX/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                                            if(basis<0.0) basis=0.0; //rounding error
                                            Get_BSplineBasisValue<SplineTYPE>(basis,x-xPre,xBasis,xFirst);

                                            basisValues[0]= xFirst * yBasis * zBasis ;
                                            basisValues[1]= xBasis * yFirst * zBasis ;
                                            basisValues[2]= xBasis * yBasis * zFirst ;

                                            jacobianMatrix = invertedJacobianMatrices[jacIndex];
//                                            detJac = 2.0 * log(detJac);
                                            detJac = log(detJac)>0?1.0:-1.0;
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
template<class SplineTYPE>
void reg_bspline_jacobianDeterminantGradientApprox3D(nifti_image *splineControlPoint,
                                                     nifti_image *targetImage,
                                                     nifti_image *gradientImage,
                                                     float weight)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(jacobianNumber * sizeof(SplineTYPE));

    computeApproximateJacobianMatrices_3D<SplineTYPE>(splineControlPoint,
                                                      invertedJacobianMatrices,
                                                      jacobianDeterminant);

    /*  */
    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    SplineTYPE xBasis=0, yBasis=0, zBasis=0;
    SplineTYPE xFirst=0, yFirst=0, zFirst=0;
    SplineTYPE basisValues[3];
    unsigned int jacIndex;

    // The gradient are now computed for every control point
    SplineTYPE *gradientImagePtrX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientImagePtrY = &gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz];
    SplineTYPE *gradientImagePtrZ = &gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz];

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    SplineTYPE approxRatio = weight * (SplineTYPE)targetImage->nvox  / (SplineTYPE)jacobianNumber;

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                SplineTYPE jacobianConstraintX=(SplineTYPE)0.0;
                SplineTYPE jacobianConstraintY=(SplineTYPE)0.0;
                SplineTYPE jacobianConstraintZ=(SplineTYPE)0.0;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)((z-1));pixelZ<(int)((z+3)); pixelZ++){
                    if(pixelZ>0 && pixelZ<splineControlPoint->nz-1){

                        Get_BSplineBasisValue<SplineTYPE>(0, z-pixelZ+1, zBasis,zFirst);

                        for(int pixelY=(int)((y-1));pixelY<(int)((y+3)); pixelY++){
                            if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                                Get_BSplineBasisValue<SplineTYPE>(0, y-pixelY+1,yBasis,yFirst);

                                for(int pixelX=(int)((x-1));pixelX<(int)((x+3)); pixelX++){
                                    if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                                        Get_BSplineBasisValue<SplineTYPE>(0, x-pixelX+1,xBasis,xFirst);

                                        basisValues[0] = xFirst * yBasis * zBasis ;
                                        basisValues[1] = xBasis * yFirst * zBasis ;
                                        basisValues[2] = xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*splineControlPoint->ny+pixelY)*splineControlPoint->nx+pixelX;
                                        jacobianMatrix = invertedJacobianMatrices[jacIndex];
                                        SplineTYPE detJac = jacobianDeterminant[jacIndex];

                                        if(detJac>0.0){
//                                            detJac = 2.0 * log(detJac);
                                            detJac = log(detJac)>0?1.0:-1.0;
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
template<class SplineTYPE>
double reg_bspline_correctFolding_2D(nifti_image *splineControlPoint,
                                     nifti_image *targetImage)
{

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(targetImage->nvox * sizeof(SplineTYPE));

    computeJacobianMatrices_2D<SplineTYPE>(targetImage,
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
        return penaltyTerm/(double)targetImage->nvox;
    }

    SplineTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    SplineTYPE basisValues[2];
    SplineTYPE xBasis=0, yBasis=0, basis;
    SplineTYPE xFirst=0, yFirst=0;
    unsigned int jacIndex;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The gradient are now computed for every control point
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            SplineTYPE foldingCorrectionX=(SplineTYPE)0.0;
            SplineTYPE foldingCorrectionY=(SplineTYPE)0.0;

            bool correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(int)reg_ceil((y-2)*gridVoxelSpacing[1]);pixelY<(int)reg_floor((y)*gridVoxelSpacing[1]); pixelY++){
                if(pixelY>-1 && pixelY<targetImage->ny){

                    int yPre=(int)((SplineTYPE)pixelY/gridVoxelSpacing[1]);
                    basis=(SplineTYPE)pixelY/gridVoxelSpacing[1]-(SplineTYPE)yPre;
                    if(basis<0.0) basis=0.0; //rounding error
                    Get_BSplineBasisValue<SplineTYPE>(basis, y-yPre,yBasis,yFirst);

                    for(int pixelX=(int)reg_ceil((x-2)*gridVoxelSpacing[0]);pixelX<(int)reg_floor((x)*gridVoxelSpacing[0]); pixelX++){
                        if(pixelX>-1 && pixelX<targetImage->nx){

                            jacIndex = pixelY*targetImage->nx+pixelX;
                            SplineTYPE logDet=jacobianDeterminant[jacIndex];

                            if(logDet<=0.0){
                                int xPre=(int)((SplineTYPE)pixelX/gridVoxelSpacing[0]);
                                basis=(SplineTYPE)pixelX/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                                if(basis<0.0) basis=0.0; //rounding error
                                Get_BSplineBasisValue<SplineTYPE>(basis, x-xPre,xBasis,xFirst);

                                basisValues[0]= xFirst * yBasis;
                                basisValues[1]= xBasis * yFirst;

                                jacobianMatrix = invertedJacobianMatrices[jacIndex];

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
                SplineTYPE gradient[2];
                gradient[0] = desorient.m[0][0]*foldingCorrectionX +
                    desorient.m[0][1]*foldingCorrectionY;
                gradient[1] = desorient.m[1][0]*foldingCorrectionX +
                    desorient.m[1][1]*foldingCorrectionY;
                SplineTYPE norm = 5.0 * sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
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
    return std::numeric_limits<double>::quiet_NaN();

}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_correctFoldingApprox_2D(nifti_image *splineControlPoint)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(jacobianNumber * sizeof(SplineTYPE));

    computeApproximateJacobianMatrices_2D<SplineTYPE>(splineControlPoint,
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
        jacobianNumber = (splineControlPoint->nx-2) * (splineControlPoint->ny-2);
        return penaltyTerm/(double)jacobianNumber;
    }

    SplineTYPE basisValues[2];
    SplineTYPE xBasis=0, yBasis=0;
    SplineTYPE xFirst=0, yFirst=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    // The gradient are now computed for every control point
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
    /* The actual gradient are now computed */
    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            SplineTYPE foldingCorrectionX=(SplineTYPE)0.0;
            SplineTYPE foldingCorrectionY=(SplineTYPE)0.0;

            bool correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(int pixelY=(y-1);pixelY<(y+2); pixelY++){
                if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                    Get_BSplineBasisValue<SplineTYPE>(0, y-pixelY+1,yBasis,yFirst);

                    for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                            jacIndex = pixelY*splineControlPoint->nx+pixelX;
                            SplineTYPE logDet=jacobianDeterminant[jacIndex];

                            if(logDet<=0.0){
                                Get_BSplineBasisValue<SplineTYPE>(0, x-pixelX+1,xBasis,xFirst);
                                basisValues[0] = xFirst * yBasis ;
                                basisValues[1] = xBasis * yFirst ;

                                jacobianMatrix = invertedJacobianMatrices[jacIndex];

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
                SplineTYPE gradient[2];
                gradient[0] = desorient.m[0][0]*foldingCorrectionX
                            + desorient.m[0][1]*foldingCorrectionY;
                gradient[1] = desorient.m[1][0]*foldingCorrectionX
                            + desorient.m[1][1]*foldingCorrectionY;
                SplineTYPE norm = 5.0 * sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
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
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_correctFolding_3D(nifti_image *splineControlPoint,
                                     nifti_image *targetImage)
{

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(targetImage->nvox * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(targetImage->nvox * sizeof(SplineTYPE));

    computeJacobianMatrices_3D<SplineTYPE>(targetImage,
                                           splineControlPoint,
                                           invertedJacobianMatrices,
                                           jacobianDeterminant);

    /* The current Penalty term value is computed */
    double penaltyTerm =0.0;
    for(unsigned int i=0; i< targetImage->nvox; i++){
        double logDet = log(jacobianDeterminant[i]);
        penaltyTerm += fabs(logDet);
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(invertedJacobianMatrices);
        return penaltyTerm/(double)targetImage->nvox;
    }

    /*  */
    mat33 reorient, desorient;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    SplineTYPE basisValues[3];
    SplineTYPE xBasis=0, yBasis=0, zBasis=0, basis;
    SplineTYPE xFirst=0, yFirst=0, zFirst=0;
    unsigned int jacIndex;

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                SplineTYPE foldingCorrectionX=(SplineTYPE)0.0;
                SplineTYPE foldingCorrectionY=(SplineTYPE)0.0;
                SplineTYPE foldingCorrectionZ=(SplineTYPE)0.0;

                bool correctFolding=false;

                // Loop over all the control points in the surrounding area
                for(int pixelZ=(int)reg_ceil((z-2)*gridVoxelSpacing[2]);pixelZ<(int)reg_floor((z)*gridVoxelSpacing[2]); pixelZ++){
                    if(pixelZ>-1 && pixelZ<targetImage->nz){

                        int zPre=(int)((SplineTYPE)pixelZ/gridVoxelSpacing[2]);
                        basis=(SplineTYPE)pixelZ/gridVoxelSpacing[2]-(SplineTYPE)zPre;
                        if(basis<0.0) basis=0.0; //rounding error
                        Get_BSplineBasisValue<SplineTYPE>(basis, z-zPre,zBasis,zFirst);

                        for(int pixelY=(int)reg_ceil((y-2)*gridVoxelSpacing[1]);pixelY<(int)reg_floor((y)*gridVoxelSpacing[1]); pixelY++){
                            if(pixelY>-1 && pixelY<targetImage->ny){

                                int yPre=(int)((SplineTYPE)pixelY/gridVoxelSpacing[1]);
                                basis=(SplineTYPE)pixelY/gridVoxelSpacing[1]-(SplineTYPE)yPre;
                                if(basis<0.0) basis=0.0; //rounding error
                                Get_BSplineBasisValue<SplineTYPE>(basis, y-yPre,yBasis,yFirst);

                                for(int pixelX=(int)reg_ceil((x-2)*gridVoxelSpacing[0]);pixelX<(int)reg_floor((x)*gridVoxelSpacing[0]); pixelX++){
                                    if(pixelX>-1 && pixelX<targetImage->nx){

                                        int xPre=(int)((SplineTYPE)pixelX/gridVoxelSpacing[0]);
                                        basis=(SplineTYPE)pixelX/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                                        if(basis<0.0) basis=0.0; //rounding error
                                        Get_BSplineBasisValue<SplineTYPE>(basis, x-xPre,xBasis,xFirst);

                                        basisValues[0]= xFirst * yBasis * zBasis ;
                                        basisValues[1]= xBasis * yFirst * zBasis ;
                                        basisValues[2]= xBasis * yBasis * zFirst ;

                                        jacIndex = (pixelZ*targetImage->ny+pixelY)*targetImage->nx+pixelX;
                                        SplineTYPE detJac = jacobianDeterminant[jacIndex];

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
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_correctFoldingApprox_3D(nifti_image *splineControlPoint)
{

    unsigned int jacobianNumber = splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;

    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianNumber * sizeof(mat33));
    SplineTYPE *jacobianDeterminant=(SplineTYPE *)malloc(jacobianNumber * sizeof(SplineTYPE));

    computeApproximateJacobianMatrices_3D<SplineTYPE>(splineControlPoint,
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
                penaltyTerm += fabs(logDet);
            }
        }
    }
    if(penaltyTerm==penaltyTerm){
        free(jacobianDeterminant);
        free(invertedJacobianMatrices);
        jacobianNumber = (splineControlPoint->nx-2) * (splineControlPoint->ny-2) * (splineControlPoint->nz-2);
        return penaltyTerm/(double)jacobianNumber;
    }

    SplineTYPE basisValues[3];
    SplineTYPE xBasis=0, yBasis=0, zBasis=0;
    SplineTYPE xFirst=0, yFirst=0, zFirst=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

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

                        Get_BSplineBasisValue<SplineTYPE>(0, z-pixelZ+1, zBasis, zFirst);

                        for(int pixelY=(int)((y-1));pixelY<(int)((y+2)); pixelY++){
                            if(pixelY>0 && pixelY<splineControlPoint->ny-1){

                                Get_BSplineBasisValue<SplineTYPE>(0, y-pixelY+1, yBasis, yFirst);

                                    for(int pixelX=(int)((x-1));pixelX<(int)((x+2)); pixelX++){
                                        if(pixelX>0 && pixelX<splineControlPoint->nx-1){

                                        Get_BSplineBasisValue<SplineTYPE>(0, x-pixelX+1, xBasis, xFirst);

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
                                        + gradient[2]*gradient[2]));

                    if(norm>(SplineTYPE)0.0){
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
#ifdef _NR_DEV
                case NIFTI_TYPE_FLOAT64:
                    return reg_bspline_correctFoldingApprox_2D<double>
                        (splineControlPoint);
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
#endif
                    break;
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
template <class SplineTYPE>
void reg_bspline_GetJacobianMap2D(nifti_image *splineControlPoint,
                                  nifti_image *jacobianImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    SplineTYPE *jacobianMapPtr = static_cast<SplineTYPE *>(jacobianImage->data);

    SplineTYPE yBasis[4],yFirst[4],temp[4],first[4];
    SplineTYPE basisX[16], basisY[16];
    SplineTYPE basis, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[16];
    SplineTYPE yControlPointCoordinates[16];

    SplineTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    unsigned int coord=0;

    /* In case the matrix is not diagonal, the jacobian has to be reoriented */
    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    for(int y=0; y<jacobianImage->ny; y++){

        int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
        basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, yBasis, yFirst);

        for(int x=0; x<jacobianImage->nx; x++){

            int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
            basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, temp, first);

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
                        xControlPointCoordinates[coord] = (SplineTYPE)xPtr[X];
                        yControlPointCoordinates[coord] = (SplineTYPE)yPtr[X];
                        coord++;
                    }
                }
            }
            oldBasis=basis;
            SplineTYPE Tx_x=0.0;
            SplineTYPE Ty_x=0.0;
            SplineTYPE Tx_y=0.0;
            SplineTYPE Ty_y=0.0;

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
template <class SplineTYPE>
void reg_bspline_GetJacobianMap3D(nifti_image *splineControlPoint,
                                  nifti_image *jacobianImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    SplineTYPE *jacobianMapPtr = static_cast<SplineTYPE *>(jacobianImage->data);

    SplineTYPE zBasis[4],zFirst[4],temp[4],first[4];
    SplineTYPE tempX[16], tempY[16], tempZ[16];
    SplineTYPE basisX[64], basisY[64], basisZ[64];
    SplineTYPE basis, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[64];
    SplineTYPE yControlPointCoordinates[64];
    SplineTYPE zControlPointCoordinates[64];

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / jacobianImage->dz;
    unsigned int coord=0;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(splineControlPoint, &desorient, &reorient);

    for(int z=0; z<jacobianImage->nz; z++){

        int zPre=(int)((SplineTYPE)z/gridVoxelSpacing[2]);
        basis=(SplineTYPE)z/gridVoxelSpacing[2]-(SplineTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, zBasis, zFirst);

        for(int y=0; y<jacobianImage->ny; y++){

            int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
            basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, temp, first);

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

                int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
                basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                Get_BSplineBasisValues<SplineTYPE>(basis, temp, first);

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
                SplineTYPE detJac = nifti_mat33_determ(jacobianMatrix);

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
template <class DTYPE>
void reg_bspline_GetJacobianMatrix2D(nifti_image *splineControlPoint,
                                     nifti_image *jacobianImage)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = static_cast<DTYPE *>(&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny]);

    DTYPE *jacobianMatrixTxxPtr = static_cast<DTYPE *>(jacobianImage->data);
    DTYPE *jacobianMatrixTyxPtr = &jacobianMatrixTxxPtr[jacobianImage->nx*jacobianImage->ny];

    DTYPE *jacobianMatrixTxyPtr = &jacobianMatrixTyxPtr[jacobianImage->nx*jacobianImage->ny];
    DTYPE *jacobianMatrixTyyPtr = &jacobianMatrixTxyPtr[jacobianImage->nx*jacobianImage->ny];

    DTYPE yBasis[4],yFirst[4],temp[4],first[4];
    DTYPE basisX[16], basisY[16];
    DTYPE basis, oldBasis=(DTYPE)(1.1);

    DTYPE xControlPointCoordinates[16];
    DTYPE yControlPointCoordinates[16];

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / jacobianImage->dy;

    unsigned int coord=0;

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
template <class DTYPE>
void reg_bspline_GetJacobianMatrix3D(nifti_image *splineControlPoint,
                                     nifti_image *jacobianImage)
{
    mat33 *invertedJacobianMatrices=(mat33 *)malloc(jacobianImage->nvox * sizeof(mat33));
    DTYPE *jacobianDeterminant=(DTYPE *)malloc(jacobianImage->nvox * sizeof(DTYPE));

    computeJacobianMatrices_3D<DTYPE>(jacobianImage,
                                      splineControlPoint,
                                      invertedJacobianMatrices,
                                      jacobianDeterminant);

    DTYPE *jacobianMatrixTxxPtr = static_cast<DTYPE *>(jacobianImage->data);
    DTYPE *jacobianMatrixTyxPtr = &jacobianMatrixTxxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    DTYPE *jacobianMatrixTzxPtr = &jacobianMatrixTyxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];

    DTYPE *jacobianMatrixTxyPtr = &jacobianMatrixTzxPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    DTYPE *jacobianMatrixTyyPtr = &jacobianMatrixTxyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    DTYPE *jacobianMatrixTzyPtr = &jacobianMatrixTyyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];

    DTYPE *jacobianMatrixTxzPtr = &jacobianMatrixTzyPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    DTYPE *jacobianMatrixTyzPtr = &jacobianMatrixTxzPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];
    DTYPE *jacobianMatrixTzzPtr = &jacobianMatrixTyzPtr[jacobianImage->nx*jacobianImage->ny*jacobianImage->nz];

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
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_GetJacobianMatrix\n");
        fprintf(stderr, "[NiftyReg ERROR] Input images were expected to be from the same type\n");
        exit(1);
    }

    if(splineControlPoint->nz>1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMatrix3D<float>(splineControlPoint, jacobianImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMatrix3D<double>(splineControlPoint, jacobianImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the control point image\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian matrix image has not been computed\n");
                exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMatrix2D<float>(splineControlPoint, jacobianImage);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMatrix2D<double>(splineControlPoint, jacobianImage);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the control point image\n");
                fprintf(stderr,"[NiftyReg ERROR] The jacobian matrix image has not been computed\n");
                exit(1);
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getJacobianMapFromDeformationField1(nifti_image *deformationField,
                                             nifti_image *jacobianImage)
{
    DTYPE *fieldPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *fieldPtrY = &fieldPtrX[jacobianImage->nvox];
    DTYPE *fieldPtrZ = NULL;
    if(deformationField->nz>1)
        fieldPtrZ = &fieldPtrY[jacobianImage->nvox];
    DTYPE *jacobianPtr = static_cast<DTYPE *>(jacobianImage->data);

    DTYPE jacobianMatrix[3][3];
    memset(jacobianMatrix,0,sizeof(jacobianMatrix));

    int voxelIndex=0;
    for(int z=0; z<deformationField->nz; z++){
        for(int y=0; y<deformationField->ny; y++){
            for(int x=0; x<deformationField->nx; x++){

                // derivative of along the X axis
                if(x==0){
                    // forward difference
                    jacobianMatrix[0][0]= (DTYPE)((fieldPtrX[voxelIndex+1] - fieldPtrX[voxelIndex] ) /
                                          (deformationField->dx));// Tx/dx
                    jacobianMatrix[1][0]= (DTYPE)((fieldPtrY[voxelIndex+1] - fieldPtrY[voxelIndex] ) /
                                          (deformationField->dx));// Ty/dx
                    if(deformationField->nz>1)
                        jacobianMatrix[2][0]= (DTYPE)((fieldPtrZ[voxelIndex+1] - fieldPtrZ[voxelIndex] ) /
                                              (deformationField->dx));// Tz/dx
                }
                else if(x==deformationField->nx-1){
                    // backward difference
                    jacobianMatrix[0][0]= (DTYPE)((fieldPtrX[voxelIndex] - fieldPtrX[voxelIndex-1] ) /
                                          (deformationField->dx));// Tx/dx
                    jacobianMatrix[1][0]= (DTYPE)((fieldPtrY[voxelIndex] - fieldPtrY[voxelIndex-1] ) /
                                          (deformationField->dx));// Ty/dx
                    if(deformationField->nz>1)
                        jacobianMatrix[2][0]= (DTYPE)((fieldPtrZ[voxelIndex] - fieldPtrZ[voxelIndex-1] ) /
                                              (deformationField->dx));// Tz/dx
                }
                else{
                    // symmetric derivative
                    jacobianMatrix[0][0]= (DTYPE)((fieldPtrX[voxelIndex+1] - fieldPtrX[voxelIndex-1] ) /
                                          (2.0*deformationField->dx));// Tx/dx
                    jacobianMatrix[1][0]= (DTYPE)((fieldPtrY[voxelIndex+1] - fieldPtrY[voxelIndex-1] ) /
                                          (2.0*deformationField->dx));// Ty/dx
                    if(deformationField->nz>1)
                        jacobianMatrix[2][0]= (DTYPE)((fieldPtrZ[voxelIndex+1] - fieldPtrZ[voxelIndex-1] ) /
                                              (2.0*deformationField->dx));// Tz/dx
                }

                // derivative of along the Y axis
                if(y==0){
                    // forward difference
                    jacobianMatrix[0][1]= (DTYPE)((fieldPtrX[voxelIndex+deformationField->nx] - fieldPtrX[voxelIndex] ) /
                                          (deformationField->dy));// Tx/dy
                    jacobianMatrix[1][1]= (DTYPE)((fieldPtrY[voxelIndex+deformationField->nx] - fieldPtrY[voxelIndex] ) /
                                          (deformationField->dy));// Ty/dy
                    if(deformationField->nz>1)
                        jacobianMatrix[2][1]= (DTYPE)((fieldPtrZ[voxelIndex+deformationField->nx] - fieldPtrZ[voxelIndex] ) /
                                              (deformationField->dy));// Tz/dy
                }
                else if(y==deformationField->ny-1){
                    // backward difference
                    jacobianMatrix[0][1]= (DTYPE)((fieldPtrX[voxelIndex] - fieldPtrX[voxelIndex-deformationField->nx] ) /
                                          (deformationField->dy));// Tx/dy
                    jacobianMatrix[1][1]= (DTYPE)((fieldPtrY[voxelIndex] - fieldPtrY[voxelIndex-deformationField->nx] ) /
                                          (deformationField->dy));// Ty/dy
                    if(deformationField->nz>1)
                        jacobianMatrix[2][1]= (DTYPE)((fieldPtrZ[voxelIndex] - fieldPtrZ[voxelIndex-deformationField->nx] ) /
                                              (deformationField->dy));// Tz/dy
                }
                else{
                    // symmetric derivative
                    jacobianMatrix[0][1]= (DTYPE)((fieldPtrX[voxelIndex+deformationField->nx] - fieldPtrX[voxelIndex-deformationField->nx] ) /
                                          (2.0*deformationField->dy));// Tx/dy
                    jacobianMatrix[1][1]= (DTYPE)((fieldPtrY[voxelIndex+deformationField->nx] - fieldPtrY[voxelIndex-deformationField->nx] ) /
                                          (2.0*deformationField->dy));// Ty/dy
                    if(deformationField->nz>1)
                        jacobianMatrix[2][1]= (DTYPE)((fieldPtrZ[voxelIndex+deformationField->nx] - fieldPtrZ[voxelIndex-deformationField->nx] ) /
                                              (2.0*deformationField->dy));// Tz/dy
                }

                // derivative of along the Z axis
                if(deformationField->nz>1){
                    if(z==0){
                        // forward difference
                        jacobianMatrix[0][2]= (DTYPE)((fieldPtrX[voxelIndex+deformationField->nx*deformationField->ny] -
                                              fieldPtrX[voxelIndex] ) / (deformationField->dz));// Tx/dz
                        jacobianMatrix[1][2]= (DTYPE)((fieldPtrY[voxelIndex+deformationField->nx*deformationField->ny] -
                                              fieldPtrY[voxelIndex] ) / (deformationField->dz));// Ty/dz
                        jacobianMatrix[2][2]= (DTYPE)((fieldPtrZ[voxelIndex+deformationField->nx*deformationField->ny] -
                                              fieldPtrZ[voxelIndex] ) / (deformationField->dz));// Tz/dz

                    }
                    else if(z==deformationField->nz-1){
                        // backward difference
                        jacobianMatrix[0][2]= (DTYPE)((fieldPtrX[voxelIndex] -
                                              fieldPtrX[voxelIndex-deformationField->nx*deformationField->ny] ) /
                                              (deformationField->dz));// Tx/dz
                        jacobianMatrix[1][2]= (DTYPE)((fieldPtrY[voxelIndex] -
                                              fieldPtrY[voxelIndex-deformationField->nx*deformationField->ny] ) /
                                              (deformationField->dz));// Ty/dz
                        jacobianMatrix[2][2]= (DTYPE)((fieldPtrZ[voxelIndex] -
                                              fieldPtrZ[voxelIndex-deformationField->nx*deformationField->ny] ) /
                                              (deformationField->dz));// Tz/dz

                    }
                    else{
                        // symmetric derivative
                        jacobianMatrix[0][2]= (DTYPE)((fieldPtrX[voxelIndex+deformationField->nx*deformationField->ny] -
                                              fieldPtrX[voxelIndex-deformationField->nx*deformationField->ny] ) /
                                              (2.0*deformationField->dz));// Tx/dz
                        jacobianMatrix[1][2]= (DTYPE)((fieldPtrY[voxelIndex+deformationField->nx*deformationField->ny] -
                                              fieldPtrY[voxelIndex-deformationField->nx*deformationField->ny] ) /
                                              (2.0*deformationField->dz));// Ty/dz
                        jacobianMatrix[2][2]= (DTYPE)((fieldPtrZ[voxelIndex+deformationField->nx*deformationField->ny] -
                                              fieldPtrZ[voxelIndex-deformationField->nx*deformationField->ny] ) /
                                              (2.0*deformationField->dz));// Tz/dz
                    }
                }

                DTYPE jacobianValue = 1;

                if(deformationField->nz>1){
                    jacobianValue =  jacobianMatrix[0][0]*jacobianMatrix[1][1]*jacobianMatrix[2][2];
                    jacobianValue += jacobianMatrix[0][1]*jacobianMatrix[1][2]*jacobianMatrix[2][0];
                    jacobianValue += jacobianMatrix[0][2]*jacobianMatrix[1][0]*jacobianMatrix[2][1];
                    jacobianValue -= jacobianMatrix[0][0]*jacobianMatrix[1][2]*jacobianMatrix[2][1];
                    jacobianValue -= jacobianMatrix[0][1]*jacobianMatrix[1][0]*jacobianMatrix[2][2];
                    jacobianValue -= jacobianMatrix[0][2]*jacobianMatrix[1][1]*jacobianMatrix[2][0];
                }
                else{
                    jacobianValue =  jacobianMatrix[0][0]*jacobianMatrix[1][1];
                    jacobianValue -= jacobianMatrix[0][1]*jacobianMatrix[1][0];
                }

                *jacobianPtr++ = jacobianValue;
                voxelIndex++;
            }
        }
    }
}
/* *************************************************************** */
void reg_getJacobianMapFromDeformationField(nifti_image *deformationField,
                                            nifti_image *jacobianImage)
{
    if(deformationField->datatype!=jacobianImage->datatype){
        printf("[NiftyReg ERROR] reg_getSourceImageGradient");
        printf("[NiftyReg ERROR] Both input images have different type. Exit");
        exit(1);
    }
    switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_getJacobianMapFromDeformationField1<float>(deformationField,jacobianImage);
            break;
#ifdef _NR_DEV
        case NIFTI_TYPE_FLOAT64:
            reg_getJacobianMapFromDeformationField1<double>(deformationField,jacobianImage);
            break;
#endif
        default:
            printf("[NiftyReg ERROR] reg_getSourceImageGradient");
            printf("[NiftyReg ERROR] Voxel type unsupported.");
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

    // The initial deformation field is computed
    reg_spline(velocityFieldImage,
               deformationFieldA,
               deformationFieldA,
               NULL, // mask
               false, //composition
               true // bspline
               );

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
            default:
#endif
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
            default:
#endif
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
