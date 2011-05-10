/*
 *  _reg_localTransformation_be.cpp
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
template<class SplineTYPE>
double reg_bspline_bendingEnergyValue2D(nifti_image *splineControlPoint,
                                        nifti_image *targetImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    SplineTYPE temp[4],first[4],second[4];
    SplineTYPE yBasis[4],yFirst[4],ySecond[4];
    SplineTYPE basisXX[16], basisYY[16], basisXY[16];
    SplineTYPE basis, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[16];
    SplineTYPE yControlPointCoordinates[16];

    SplineTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;

    unsigned int coord=0;

    double constraintValue=0;

    for(int y=0; y<targetImage->ny; y++){

        int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
        basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, yBasis, yFirst, ySecond);

        for(int x=0; x<targetImage->nx; x++){

            int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
            basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, temp, first, second);

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

            SplineTYPE XX_x=0.0;
            SplineTYPE YY_x=0.0;
            SplineTYPE XY_x=0.0;
            SplineTYPE XX_y=0.0;
            SplineTYPE YY_y=0.0;
            SplineTYPE XY_y=0.0;

            for(int a=0; a<16; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }

            constraintValue += (double)(XX_x*XX_x + YY_x*YY_x + 2.0*XY_x*XY_x);
            constraintValue += (double)(XX_y*XX_y + YY_y*YY_y + 2.0*XY_y*XY_y);
        }
    }

    return constraintValue/(double)(2.0*targetImage->nx*targetImage->ny);

}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_bendingEnergyValue3D(nifti_image *splineControlPoint,
                                        nifti_image *targetImage)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];
    SplineTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny];

    SplineTYPE temp[4],first[4],second[4];
    SplineTYPE zBasis[4],zFirst[4],zSecond[4];
    SplineTYPE tempXX[16], tempYY[16], tempZZ[16], tempXY[16], tempYZ[16], tempXZ[16];
    SplineTYPE basisXX[64], basisYY[64], basisZZ[64], basisXY[64], basisYZ[64], basisXZ[64];
    SplineTYPE basis, oldBasis=(SplineTYPE)(1.1);

    SplineTYPE xControlPointCoordinates[64];
    SplineTYPE yControlPointCoordinates[64];
    SplineTYPE zControlPointCoordinates[64];

    SplineTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / targetImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / targetImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / targetImage->dz;

    unsigned int coord=0;

    double constraintValue=0;

    for(int z=0; z<targetImage->nz; z++){

        int zPre=(int)((SplineTYPE)z/gridVoxelSpacing[2]);
        basis=(SplineTYPE)z/gridVoxelSpacing[2]-(SplineTYPE)zPre;
        if(basis<0.0) basis=0.0; //rounding error
        Get_BSplineBasisValues<SplineTYPE>(basis, zBasis, zFirst, zSecond);

        for(int y=0; y<targetImage->ny; y++){

            int yPre=(int)((SplineTYPE)y/gridVoxelSpacing[1]);
            basis=(SplineTYPE)y/gridVoxelSpacing[1]-(SplineTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            Get_BSplineBasisValues<SplineTYPE>(basis, temp, first, second);

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

                int xPre=(int)((SplineTYPE)x/gridVoxelSpacing[0]);
                basis=(SplineTYPE)x/gridVoxelSpacing[0]-(SplineTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                Get_BSplineBasisValues<SplineTYPE>(basis, temp, first, second);

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

                SplineTYPE XX_x=0.0;
                SplineTYPE YY_x=0.0;
                SplineTYPE ZZ_x=0.0;
                SplineTYPE XY_x=0.0;
                SplineTYPE YZ_x=0.0;
                SplineTYPE XZ_x=0.0;
                SplineTYPE XX_y=0.0;
                SplineTYPE YY_y=0.0;
                SplineTYPE ZZ_y=0.0;
                SplineTYPE XY_y=0.0;
                SplineTYPE YZ_y=0.0;
                SplineTYPE XZ_y=0.0;
                SplineTYPE XX_z=0.0;
                SplineTYPE YY_z=0.0;
                SplineTYPE ZZ_z=0.0;
                SplineTYPE XY_z=0.0;
                SplineTYPE YZ_z=0.0;
                SplineTYPE XZ_z=0.0;

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

                constraintValue += (double)(XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x));
                constraintValue += (double)(XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y));
                constraintValue += (double)(XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z));
            }
        }
    }

    return constraintValue/(double)(3.0*targetImage->nx*targetImage->ny*targetImage->nz);

}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_bendingEnergyApproxValue2D(nifti_image *splineControlPoint)
{

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    // As the contraint is only computed at the control point positions, the basis value of the spline are always the same
    SplineTYPE basisXX[9], basisYY[9], basisXY[9];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    int coord = 0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisXX[coord] = second[a] * normal[b];
            basisYY[coord] = normal[a] * second[b];
            basisXY[coord] = first[a] * first[b];
            coord++;
        }
    }

    SplineTYPE constraintValue=0.0;

    SplineTYPE xControlPointCoordinates[9];
    SplineTYPE yControlPointCoordinates[9];

    for(int y=1;y<splineControlPoint->ny-1;y++){
        for(int x=1;x<splineControlPoint->nx-1;x++){

            get_GridValuesApprox<SplineTYPE>(x-1,
                                             y-1,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false);

            SplineTYPE XX_x=0.0;
            SplineTYPE YY_x=0.0;
            SplineTYPE XY_x=0.0;
            SplineTYPE XX_y=0.0;
            SplineTYPE YY_y=0.0;
            SplineTYPE XY_y=0.0;

            for(int a=0; a<9; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }

            constraintValue += (double)(XX_x*XX_x + YY_x*YY_x + 2.0*XY_x*XY_x);
            constraintValue += (double)(XX_y*XX_y + YY_y*YY_y + 2.0*XY_y*XY_y);
        }
    }
    return constraintValue/(double)(2.0*splineControlPoint->nx*splineControlPoint->ny);
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_bendingEnergyApproxValue3D(nifti_image *splineControlPoint)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>
       (splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
       (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>
       (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    // As the contraint is only computed at the control point positions, the basis value of the spline are always the same
    SplineTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    // There are six different values taken into account
    SplineTYPE tempXX[9], tempYY[9], tempZZ[9], tempXY[9], tempYZ[9], tempXZ[9];
    int coord=0;
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

    double constraintValue=0.0;

    SplineTYPE xControlPointCoordinates[27];
    SplineTYPE yControlPointCoordinates[27];
    SplineTYPE zControlPointCoordinates[27];

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

                SplineTYPE XX_x=0.0, YY_x=0.0, ZZ_x=0.0;
                SplineTYPE XY_x=0.0, YZ_x=0.0, XZ_x=0.0;
                SplineTYPE XX_y=0.0, YY_y=0.0, ZZ_y=0.0;
                SplineTYPE XY_y=0.0, YZ_y=0.0, XZ_y=0.0;
                SplineTYPE XX_z=0.0, YY_z=0.0, ZZ_z=0.0;
                SplineTYPE XY_z=0.0, YZ_z=0.0, XZ_z=0.0;

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

                constraintValue += (double)(XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x));
                constraintValue += (double)(XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y));
                constraintValue += (double)(XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z));
            }
        }
    }

    return constraintValue/(double)(3.0*splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
}
/* *************************************************************** */
extern "C++"
double reg_bspline_bendingEnergy(nifti_image *splineControlPoint,
                                 nifti_image *targetImage,
                                 bool approx)
{
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                if(approx)
                    return reg_bspline_bendingEnergyApproxValue2D<float>
                        (splineControlPoint);
                else return reg_bspline_bendingEnergyValue2D<float>
                        (splineControlPoint, targetImage);
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                if(approx)
                    return reg_bspline_bendingEnergyApproxValue2D<double>
                        (splineControlPoint);
                else return reg_bspline_bendingEnergyValue2D<double>
                        (splineControlPoint, targetImage);
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy\n");
                fprintf(stderr,"[NiftyReg ERROR] The bending energy is not computed\n");
                exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                if(approx)
                    return reg_bspline_bendingEnergyApproxValue3D<float>
                        (splineControlPoint);
                else return reg_bspline_bendingEnergyValue3D<float>
                        (splineControlPoint, targetImage);
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                if(approx)
                    return reg_bspline_bendingEnergyApproxValue3D<double>
                        (splineControlPoint);
                else return reg_bspline_bendingEnergyValue3D<double>
                        (splineControlPoint, targetImage);
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy\n");
                fprintf(stderr,"[NiftyReg ERROR] The bending energy is not computed\n");
                exit(1);
        }

    }
}
/* *************************************************************** */
/* *************************************************************** */

template<class SplineTYPE>
void reg_bspline_approxBendingEnergyGradient3D( nifti_image *splineControlPoint,
                                                nifti_image *targetImage,
                                                nifti_image *gradientImage,
                                                float weight)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    // There are six different values taken into account
    SplineTYPE tempXX[9], tempYY[9], tempZZ[9], tempXY[9], tempYZ[9], tempXZ[9];
    int coord=0;
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

    SplineTYPE nodeNumber = (SplineTYPE)(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
    SplineTYPE *derivativeValues = (SplineTYPE *)calloc(18*(int)nodeNumber, sizeof(SplineTYPE));
    SplineTYPE *derivativeValuesPtr;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[(unsigned int)nodeNumber]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[(unsigned int)nodeNumber]);

    SplineTYPE xControlPointCoordinates[27];
    SplineTYPE yControlPointCoordinates[27];
    SplineTYPE zControlPointCoordinates[27];

    for(int z=1;z<splineControlPoint->nz-1;z++){
        for(int y=1;y<splineControlPoint->ny-1;y++){
            derivativeValuesPtr = &derivativeValues[18*((z*splineControlPoint->ny+y)*splineControlPoint->nx+1)];
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
                SplineTYPE XX_x=0.0;
                SplineTYPE YY_x=0.0;
                SplineTYPE ZZ_x=0.0;
                SplineTYPE XY_x=0.0;
                SplineTYPE YZ_x=0.0;
                SplineTYPE XZ_x=0.0;
                SplineTYPE XX_y=0.0;
                SplineTYPE YY_y=0.0;
                SplineTYPE ZZ_y=0.0;
                SplineTYPE XY_y=0.0;
                SplineTYPE YZ_y=0.0;
                SplineTYPE XZ_y=0.0;
                SplineTYPE XX_z=0.0;
                SplineTYPE YY_z=0.0;
                SplineTYPE ZZ_z=0.0;
                SplineTYPE XY_z=0.0;
                SplineTYPE YZ_z=0.0;
                SplineTYPE XZ_z=0.0;

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
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_x);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_y);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_z);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_x);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_y);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_z);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*ZZ_x);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*ZZ_y);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*ZZ_z);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_x);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_y);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_z);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*YZ_x);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*YZ_y);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*YZ_z);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XZ_x);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XZ_y);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XZ_z);
            }
        }
    }

    SplineTYPE *gradientX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientY = &gradientX[(int)nodeNumber];
    SplineTYPE *gradientZ = &gradientY[(int)nodeNumber];
    SplineTYPE *gradientXPtr = &gradientX[0];
    SplineTYPE *gradientYPtr = &gradientY[0];
    SplineTYPE *gradientZPtr = &gradientZ[0];

    SplineTYPE approxRatio = weight * (SplineTYPE)(targetImage->nx*targetImage->ny*targetImage->nz)
    / (SplineTYPE)(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);

    SplineTYPE gradientValue[3];

    for(int z=0;z<splineControlPoint->nz;z++){
        for(int y=0;y<splineControlPoint->ny;y++){
            for(int x=0;x<splineControlPoint->nx;x++){

                gradientValue[0]=gradientValue[1]=gradientValue[2]=0.0;

                unsigned int coord=0;
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
template<class SplineTYPE>
void reg_bspline_approxBendingEnergyGradient2D(   nifti_image *splineControlPoint,
                                            nifti_image *targetImage,
                                            nifti_image *gradientImage,
                                            float weight)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE basisXX[9], basisYY[9], basisXY[9];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    int coord = 0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisXX[coord] = second[a] * normal[b];
            basisYY[coord] = normal[a] * second[b];
            basisXY[coord] = first[a] * first[b];
            coord++;
        }
    }

    SplineTYPE nodeNumber = (SplineTYPE)(splineControlPoint->nx*splineControlPoint->ny);
    SplineTYPE *derivativeValues = (SplineTYPE *)calloc(6*(int)nodeNumber, sizeof(SplineTYPE));

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[(unsigned int)nodeNumber]);

    SplineTYPE xControlPointCoordinates[9];
    SplineTYPE yControlPointCoordinates[9];

    SplineTYPE *derivativeValuesPtr = &derivativeValues[0];

    for(int y=1;y<splineControlPoint->ny-1;y++){
        derivativeValuesPtr = &derivativeValues[6*(y*splineControlPoint->nx+1)];
        for(int x=1;x<splineControlPoint->nx-1;x++){

            get_GridValuesApprox<SplineTYPE>(x-1,
                                             y-1,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false);

            SplineTYPE XX_x=0.0;
            SplineTYPE YY_x=0.0;
            SplineTYPE XY_x=0.0;
            SplineTYPE XX_y=0.0;
            SplineTYPE YY_y=0.0;
            SplineTYPE XY_y=0.0;

            for(int a=0; a<9; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }
            *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_x);
            *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_y);
            *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_x);
            *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_y);
            *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_x);
            *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_y);
        }
    }

    SplineTYPE *gradientXPtr = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientYPtr = static_cast<SplineTYPE *>(&gradientXPtr[(int)nodeNumber]);

    SplineTYPE approxRatio= weight * (SplineTYPE)(targetImage->nx*targetImage->ny)
    / (SplineTYPE)( splineControlPoint->nx*splineControlPoint->ny );

    SplineTYPE gradientValue[2];

    for(int y=0;y<splineControlPoint->ny;y++){
        for(int x=0;x<splineControlPoint->nx;x++){

            gradientValue[0]=gradientValue[1]=0.0;

            unsigned int coord=0;
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
extern "C++"
void reg_bspline_bendingEnergyGradient( nifti_image *splineControlPoint,
                                        nifti_image *targetImage,
                                        nifti_image *gradientImage,
                                        float weight)
{
    if(splineControlPoint->datatype != gradientImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] The spline control point image and the gradient image were expected to have the same datatype\n");
        fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
        exit(1);
    }
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_approxBendingEnergyGradient2D<float>
                    (splineControlPoint, targetImage, gradientImage, weight);
                break;
            case NIFTI_TYPE_FLOAT64:
                break;
                reg_bspline_approxBendingEnergyGradient2D<double>
                    (splineControlPoint, targetImage, gradientImage, weight);
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not been computed\n");
                exit(1);
        }
        }else{
        switch(splineControlPoint->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_approxBendingEnergyGradient3D<float>
                    (splineControlPoint, targetImage, gradientImage, weight);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                break;
#endif
                reg_bspline_approxBendingEnergyGradient3D<double>
                    (splineControlPoint, targetImage, gradientImage, weight);
            default:
                fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
                fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not been computed\n");
                exit(1);
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
