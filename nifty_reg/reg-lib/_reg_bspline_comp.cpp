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

/* *************************************************************** */
template<class DTYPE>
void reg_spline_cppComposition_2D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool disp,
                                  bool bspline)
{
    // REMINDER T(x)=Grid1(Grid2(x))

#if _USE_SSE
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    DTYPE *outCPPPtrX = static_cast<DTYPE *>(grid2->data);
    DTYPE *outCPPPtrY = &outCPPPtrX[grid2->nx*grid2->ny];

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(grid1->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[grid1->nx*grid1->ny];

    DTYPE basis;

#ifdef _WINDOWS
    __declspec(align(16)) DTYPE xBasis[4];
    __declspec(align(16)) DTYPE yBasis[4];
#if _USE_SSE
    __declspec(align(16)) DTYPE xyBasis[16];
#endif

    __declspec(align(16)) DTYPE xControlPointCoordinates[16];
    __declspec(align(16)) DTYPE yControlPointCoordinates[16];
#else
    DTYPE xBasis[4] __attribute__((aligned(16)));
    DTYPE yBasis[4] __attribute__((aligned(16)));
#if _USE_SSE
    DTYPE xyBasis[16] __attribute__((aligned(16)));
#endif

    DTYPE xControlPointCoordinates[16] __attribute__((aligned(16)));
    DTYPE yControlPointCoordinates[16] __attribute__((aligned(16)));
#endif

    unsigned int coord;

    // read the xyz/ijk sform or qform, as appropriate
    mat44 *matrix_real_to_voxel=NULL;
    mat44 *matrix_voxel_to_real=NULL;
    if(grid1->sform_code>0){
        matrix_real_to_voxel=&(grid1->sto_ijk);
    }
    else{
        matrix_real_to_voxel=&(grid1->qto_ijk);
    }
    if(grid2->sform_code>0){
        matrix_voxel_to_real=&(grid2->sto_xyz);
    }
    else{
        matrix_voxel_to_real=&(grid2->qto_xyz);
    }

    for(int y=0; y<grid2->ny; y++){
        for(int x=0; x<grid2->nx; x++){

            // Get the initial control point position
            DTYPE xReal=0;
            DTYPE yReal=0;
            if(disp){ // displacement are assumed, position otherwise
                xReal = matrix_voxel_to_real->m[0][0]*(DTYPE)x
                    + matrix_voxel_to_real->m[0][1]*(DTYPE)y
                    + matrix_voxel_to_real->m[0][3];
                yReal = matrix_voxel_to_real->m[1][0]*(DTYPE)x
                    + matrix_voxel_to_real->m[1][1]*(DTYPE)y
                    + matrix_voxel_to_real->m[1][3];
            }

            // Get the control point actual position
            xReal += *outCPPPtrX;
            yReal += *outCPPPtrY;

            // Get the voxel based control point position
            DTYPE xVoxel = matrix_real_to_voxel->m[0][0]*xReal
            + matrix_real_to_voxel->m[0][1]*yReal + matrix_real_to_voxel->m[0][3];
            DTYPE yVoxel = matrix_real_to_voxel->m[1][0]*xReal
            + matrix_real_to_voxel->m[1][1]*yReal + matrix_real_to_voxel->m[1][3];

            xVoxel = xVoxel<(DTYPE)0.0?(DTYPE)0.0:xVoxel;
            yVoxel = yVoxel<(DTYPE)0.0?(DTYPE)0.0:yVoxel;

            // The spline coefficients are computed
            int xPre=(int)(floor(xVoxel));
            basis=(DTYPE)xVoxel-(DTYPE)xPre;
            if(bspline)
                Get_BSplineBasisValues<DTYPE>(basis, xBasis);
            else Get_SplineBasisValues<DTYPE>(basis, yBasis);

            int yPre=(int)(floor(yVoxel));
            basis=(DTYPE)yVoxel-(DTYPE)yPre;
            if(bspline)
                Get_BSplineBasisValues<DTYPE>(basis, yBasis);
            else Get_SplineBasisValues<DTYPE>(basis, yBasis);

            // The control points are stored
            get_splineDisplacement<DTYPE>(xPre,
                                          yPre,
                                          grid2,
                                          controlPointPtrX,
                                          controlPointPtrY,
                                          xControlPointCoordinates,
                                          yControlPointCoordinates);
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
            coord=0;
            for(unsigned int b=0; b<4; b++){
                for(unsigned int a=0; a<4; a++){
                    DTYPE tempValue = xBasis[a] * yBasis[b];
                    xReal += xControlPointCoordinates[coord] * tempValue;
                    yReal += yControlPointCoordinates[coord] * tempValue;
                    coord++;
                }
            }
#endif
            *outCPPPtrX++ += xReal;
            *outCPPPtrY++ += yReal;
        }
    }
    return;
}
/* *************************************************************** */
template<class DTYPE>
void reg_spline_cppComposition_3D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool disp,
                                  bool bspline)
{
    // REMINDER T(x)=Grid1(Grid2(x))

#if _USE_SSE
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    DTYPE *outCPPPtrX = static_cast<DTYPE *>(grid2->data);
    DTYPE *outCPPPtrY = &outCPPPtrX[grid2->nx*grid2->ny*grid2->nz];
    DTYPE *outCPPPtrZ = &outCPPPtrY[grid2->nx*grid2->ny*grid2->nz];

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(grid1->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[grid1->nx*grid1->ny*grid1->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[grid1->nx*grid1->ny*grid1->nz];

    DTYPE basis;

#ifdef _WINDOWS
    __declspec(align(16)) DTYPE xBasis[4];
    __declspec(align(16)) DTYPE yBasis[4];
    __declspec(align(16)) DTYPE zBasis[4];
    __declspec(align(16)) DTYPE xControlPointCoordinates[64];
    __declspec(align(16)) DTYPE yControlPointCoordinates[64];
    __declspec(align(16)) DTYPE zControlPointCoordinates[64];
#else
    DTYPE xBasis[4] __attribute__((aligned(16)));
    DTYPE yBasis[4] __attribute__((aligned(16)));
    DTYPE zBasis[4] __attribute__((aligned(16)));
    DTYPE xControlPointCoordinates[64] __attribute__((aligned(16)));
    DTYPE yControlPointCoordinates[64] __attribute__((aligned(16)));
    DTYPE zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif

    int xPre, xPreOld=1, yPre, yPreOld=1, zPre, zPreOld=1;

    // read the xyz/ijk sform or qform, as appropriate
    // read the xyz/ijk sform or qform, as appropriate
    mat44 *matrix_real_to_voxel=NULL;
    mat44 *matrix_voxel_to_real=NULL;
    if(grid1->sform_code>0){
        matrix_real_to_voxel=&(grid1->sto_ijk);
    }
    else{
        matrix_real_to_voxel=&(grid1->qto_ijk);
    }
    if(grid2->sform_code>0){
        matrix_voxel_to_real=&(grid2->sto_xyz);
    }
    else{
        matrix_voxel_to_real=&(grid2->qto_xyz);
    }

    for(int z=0; z<grid2->nz; z++){
        for(int y=0; y<grid2->ny; y++){
            for(int x=0; x<grid2->nx; x++){

                // Get the initial control point position
                DTYPE xReal=0;
                DTYPE yReal=0;
                DTYPE zReal=0;
                if(disp){ // Grid2 contains displacements
                    xReal = matrix_voxel_to_real->m[0][0]*(DTYPE)x
                    + matrix_voxel_to_real->m[0][1]*(DTYPE)y
                    + matrix_voxel_to_real->m[0][2]*(DTYPE)z
                    + matrix_voxel_to_real->m[0][3];
                    yReal = matrix_voxel_to_real->m[1][0]*(DTYPE)x
                    + matrix_voxel_to_real->m[1][1]*(DTYPE)y
                    + matrix_voxel_to_real->m[1][2]*(DTYPE)z
                    + matrix_voxel_to_real->m[1][3];
                    zReal = matrix_voxel_to_real->m[2][0]*(DTYPE)x
                    + matrix_voxel_to_real->m[2][1]*(DTYPE)y
                    + matrix_voxel_to_real->m[2][2]*(DTYPE)z
                    + matrix_voxel_to_real->m[2][3];
                }

                // Get the control point actual position
                xReal += *outCPPPtrX;
                yReal += *outCPPPtrY;
                zReal += *outCPPPtrZ;

                // Get the voxel based control point position
                DTYPE xVoxel =
                  matrix_real_to_voxel->m[0][0]*xReal
                + matrix_real_to_voxel->m[0][1]*yReal
                + matrix_real_to_voxel->m[0][2]*zReal
                + matrix_real_to_voxel->m[0][3];
                DTYPE yVoxel =
                  matrix_real_to_voxel->m[1][0]*xReal
                + matrix_real_to_voxel->m[1][1]*yReal
                + matrix_real_to_voxel->m[1][2]*zReal
                + matrix_real_to_voxel->m[1][3];
                DTYPE zVoxel =
                  matrix_real_to_voxel->m[2][0]*xReal
                + matrix_real_to_voxel->m[2][1]*yReal
                + matrix_real_to_voxel->m[2][2]*zReal
                + matrix_real_to_voxel->m[2][3];

                xVoxel = xVoxel<(DTYPE)0.0?(DTYPE)0.0:xVoxel;
                yVoxel = yVoxel<(DTYPE)0.0?(DTYPE)0.0:yVoxel;
                zVoxel = zVoxel<(DTYPE)0.0?(DTYPE)0.0:zVoxel;

                // The spline coefficients are computed
                xPre=(int)(floor(xVoxel));
                basis=(DTYPE)xVoxel-(DTYPE)xPre;
                Get_BSplineBasisValues<DTYPE>(basis, xBasis);

                yPre=(int)(floor(yVoxel));
                basis=(DTYPE)yVoxel-(DTYPE)yPre;
                Get_BSplineBasisValues<DTYPE>(basis, yBasis);

                zPre=(int)(floor(zVoxel));
                basis=(DTYPE)zVoxel-(DTYPE)zPre;
                Get_BSplineBasisValues<DTYPE>(basis, zBasis);

                // The control points are stored
                if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
                    get_splineDisplacement(xPre,
                                           yPre,
                                           zPre,
                                           grid1,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           controlPointPtrZ,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates,
                                           zControlPointCoordinates);
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
                unsigned int coord=0;
                for(unsigned int c=0; c<4; c++){
                    for(unsigned int b=0; b<4; b++){
                        for(unsigned int a=0; a<4; a++){
                            DTYPE tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                            xReal += xControlPointCoordinates[coord] * tempValue;
                            yReal += yControlPointCoordinates[coord] * tempValue;
                            zReal += zControlPointCoordinates[coord] * tempValue;
                            coord++;
                        }
                    }
                }
#endif
                *outCPPPtrX++ += xReal;
                *outCPPPtrY++ += yReal;
                *outCPPPtrZ++ += zReal;
            }
        }
    }
    return;
}
/* *************************************************************** */
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool disp,
                              bool bspline)
{
    // REMINDER T(x)=Grid1(Grid2(x))

    if(grid1->datatype != grid2->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition\n");
        fprintf(stderr,"[NiftyReg ERROR] Both input images do not have the same type\n");
        exit(1);
    }

#if _USE_SSE
    if(grid1->datatype != NIFTI_TYPE_FLOAT32){
        fprintf(stderr,"[NiftyReg ERROR] SSE computation has only been implemented for single precision.\n");
        fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
        exit(1);
    }
#endif

    if(grid1->nz>1){
        switch(grid1->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_spline_cppComposition_3D<float>
                        (grid1, grid2, disp, bspline);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_spline_cppComposition_3D<double>
                        (grid1, grid2, disp, bspline);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition 3D\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double floating images\n");
                return 1;
        }
    }
    else{
        switch(grid1->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_spline_cppComposition_2D<float>
                        (grid1, grid2, disp, bspline);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_spline_cppComposition_2D<double>
                        (grid1, grid2, disp, bspline);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition 2D\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
                return 1;
        }
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_2D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage,
                                                    bool approx)
{
    // The jacobian map is initialise to 1 everywhere
    ImageTYPE *jacPtr = static_cast<ImageTYPE *>(jacobianImage->data);
    for(unsigned int i=0;i<jacobianImage->nvox;i++) jacPtr[i]=(ImageTYPE)1.0;

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4];
    ImageTYPE basis;

    ImageTYPE xControlPointCoordinates[16];
    ImageTYPE yControlPointCoordinates[16];

    ImageTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = velocityFieldImage->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = velocityFieldImage->dy / jacobianImage->dy;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(velocityFieldImage, &desorient, &reorient);

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(jacobianImage->nvox*sizeof(ImageTYPE));

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

    // The initial deformation field is initialised with the nifti header
    int jacIndex = 0;
    for(int y=0; y<jacobianImage->ny; y++){
        for(int x=0; x<jacobianImage->nx; x++){
            deformationFieldArrayX[jacIndex]
                = jac_xyz_matrix->m[0][0]*(ImageTYPE)x
                + jac_xyz_matrix->m[0][1]*(ImageTYPE)y
                + jac_xyz_matrix->m[0][3];
            deformationFieldArrayY[jacIndex]
                = jac_xyz_matrix->m[1][0]*(ImageTYPE)x
                + jac_xyz_matrix->m[1][1]*(ImageTYPE)y
                + jac_xyz_matrix->m[1][3];
            jacIndex++;
        }
    }
    int coord=0;

    ImageTYPE *controlPointPtrX = reinterpret_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *controlPointPtrY = &controlPointPtrX[velocityFieldImage->nx*velocityFieldImage->ny];

    for(int l=0;l<velocityFieldImage->pixdim[5];l++){

        int xPre, yPre;
        int xPreOld=-99;
        int yPreOld=-99;

        jacIndex=0;
        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){

                ImageTYPE realPosition[3];
                realPosition[0] = deformationFieldArrayX[jacIndex];
                realPosition[1] = deformationFieldArrayY[jacIndex];

                ImageTYPE voxelPosition[2];
                voxelPosition[0]
                    = jac_ijk_matrix->m[0][0]*realPosition[0]
                    + jac_ijk_matrix->m[0][1]*realPosition[1]
                    + jac_ijk_matrix->m[0][3];
                voxelPosition[1]
                    = jac_ijk_matrix->m[1][0]*realPosition[0]
                    + jac_ijk_matrix->m[1][1]*realPosition[1]
                    + jac_ijk_matrix->m[1][3];

                xPre=(int)floor((ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]);
                yPre=(int)floor((ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]);

                basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                Get_BSplineBasisValues<ImageTYPE>(basis, xBasis,xFirst);

                basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
                Get_BSplineBasisValues<ImageTYPE>(basis, yBasis,yFirst);

                ImageTYPE Tx_x=0.0;
                ImageTYPE Ty_x=0.0;
                ImageTYPE Tx_y=0.0;
                ImageTYPE Ty_y=0.0;
                ImageTYPE newDisplacementX = 0.0;
                ImageTYPE newDisplacementY = 0.0;
                if(approx){
                    xPre--;
                    yPre--;
                }
                if(xPre!=xPreOld || yPre!=yPreOld){
                    get_splineDisplacement(xPre,
                                           yPre,
                                           velocityFieldImage,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates);
                    xPreOld=xPre;
                    yPreOld=yPre;
                }

                coord=0;
                for(int b=0; b<4; b++){
                    for(int a=0; a<4; a++){
                        ImageTYPE basisX= yBasis[b]*xFirst[a];   // y * x'
                        ImageTYPE basisY= yFirst[b]*xBasis[a];   // y'* x
                        basis = yBasis[b]*xBasis[a];   // y * x
                        Tx_x += basisX*xControlPointCoordinates[coord];
                        Tx_y += basisY*xControlPointCoordinates[coord];
                        Ty_x += basisX*yControlPointCoordinates[coord];
                        Ty_y += basisY*yControlPointCoordinates[coord];
                        newDisplacementX += basis*xControlPointCoordinates[coord];
                        newDisplacementY += basis*yControlPointCoordinates[coord];
                        coord++;
                    }
                }

                deformationFieldArrayX[jacIndex] += newDisplacementX;
                deformationFieldArrayY[jacIndex] += newDisplacementY;

                jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)velocityFieldImage->dx);
                jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)velocityFieldImage->dy);
                jacobianMatrix.m[0][2]= 0.f;
                jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)velocityFieldImage->dx);
                jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)velocityFieldImage->dy);
                jacobianMatrix.m[1][2]= 0.f;
                jacobianMatrix.m[2][0]= 0.f;
                jacobianMatrix.m[2][1]= 0.f;
                jacobianMatrix.m[2][2]= 0.f;

                jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                jacobianMatrix.m[0][0]++;
                jacobianMatrix.m[1][1]++;
                jacobianMatrix.m[2][2]++;
                ImageTYPE detJac = nifti_mat33_determ(jacobianMatrix);
                jacPtr[jacIndex] *= detJac;
                jacIndex++;
            } // x
        } // y
    }
    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacobianMapFromVelocityField_3D(nifti_image* velocityFieldImage,
                                                    nifti_image* jacobianImage,
                                                    bool approx)
{
    //TOCHECK
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField_3D\n");
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

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis;

    ImageTYPE xControlPointCoordinates[64];
    ImageTYPE yControlPointCoordinates[64];
    ImageTYPE zControlPointCoordinates[64];

    ImageTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = velocityFieldImage->dx / jacobianImage->dx;
    gridVoxelSpacing[1] = velocityFieldImage->dy / jacobianImage->dy;
    gridVoxelSpacing[2] = velocityFieldImage->dz / jacobianImage->dz;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(velocityFieldImage, &desorient, &reorient);

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

#if _USE_SSE
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
    int jacIndex = 0;
    for(int z=0; z<jacobianImage->nz; z++){
        for(int y=0; y<jacobianImage->ny; y++){
            for(int x=0; x<jacobianImage->nx; x++){
                deformationFieldArrayX[jacIndex]
                    = jac_xyz_matrix->m[0][0]*(ImageTYPE)x
                    + jac_xyz_matrix->m[0][1]*(ImageTYPE)y
                    + jac_xyz_matrix->m[0][2]*(ImageTYPE)z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*(ImageTYPE)x
                    + jac_xyz_matrix->m[1][1]*(ImageTYPE)y
                    + jac_xyz_matrix->m[1][2]*(ImageTYPE)z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*(ImageTYPE)x
                    + jac_xyz_matrix->m[2][1]*(ImageTYPE)y
                    + jac_xyz_matrix->m[2][2]*(ImageTYPE)z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }

    ImageTYPE *controlPointPtrX = reinterpret_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *controlPointPtrY = &controlPointPtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    ImageTYPE *controlPointPtrZ = &controlPointPtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<velocityFieldImage->pixdim[5];l++){

        int xPre, yPre, zPre;
        int xPreOld=-99;
        int yPreOld=-99;
        int zPreOld=-99;

        jacIndex=0;
        for(int z=0; z<jacobianImage->nz; z++){
            for(int y=0; y<jacobianImage->ny; y++){
                for(int x=0; x<jacobianImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if _USE_SSE
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

                    basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                    Get_BSplineBasisValues<ImageTYPE>(basis, xBasis,xFirst);

                    basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
                    Get_BSplineBasisValues<ImageTYPE>(basis, yBasis,yFirst);

                    basis=(ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]-(ImageTYPE)(zPre);
                    Get_BSplineBasisValues<ImageTYPE>(basis, zBasis,zFirst);

                    ImageTYPE Tx_x=0.0;
                    ImageTYPE Ty_x=0.0;
                    ImageTYPE Tz_x=0.0;
                    ImageTYPE Tx_y=0.0;
                    ImageTYPE Ty_y=0.0;
                    ImageTYPE Tz_y=0.0;
                    ImageTYPE Tx_z=0.0;
                    ImageTYPE Ty_z=0.0;
                    ImageTYPE Tz_z=0.0;
                    ImageTYPE newDisplacementX = 0.0;
                    ImageTYPE newDisplacementY = 0.0;
                    ImageTYPE newDisplacementZ = 0.0;
                    if(approx){
                        xPre--;
                        yPre--;
                        zPre--;
                    }
                    if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
                        get_splineDisplacement(xPre,
                                               yPre,
                                               zPre,
                                               velocityFieldImage,
                                               controlPointPtrX,
                                               controlPointPtrY,
                                               controlPointPtrZ,
                                               xControlPointCoordinates,
                                               yControlPointCoordinates,
                                               zControlPointCoordinates);
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
                    newDisplacementX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = pos_y;
                    newDisplacementY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = pos_z;
                    newDisplacementZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    int coord=0;
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
                                newDisplacementX += basis*xControlPointCoordinates[coord];
                                newDisplacementY += basis*yControlPointCoordinates[coord];
                                newDisplacementZ += basis*zControlPointCoordinates[coord];
                                coord++;
                            }
                        }
                    }
#endif

                    deformationFieldArrayX[jacIndex] += newDisplacementX;
                    deformationFieldArrayY[jacIndex] += newDisplacementY;
                    deformationFieldArrayZ[jacIndex] += newDisplacementZ;

                    jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)velocityFieldImage->dx);
                    jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)velocityFieldImage->dy);
                    jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)velocityFieldImage->dz);
                    jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)velocityFieldImage->dx);
                    jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)velocityFieldImage->dy);
                    jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)velocityFieldImage->dz);
                    jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)velocityFieldImage->dx);
                    jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)velocityFieldImage->dy);
                    jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)velocityFieldImage->dz);

                    jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                    jacobianMatrix.m[0][0]++;
                    jacobianMatrix.m[1][1]++;
                    jacobianMatrix.m[2][2]++;
                    ImageTYPE detJac = nifti_mat33_determ(jacobianMatrix);
                    jacPtr[jacIndex] *= detJac;
                    jacIndex++;
                } // x
            } // y
        } // z

    }
    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);

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
                reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage,0);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage,0);
                break;
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
                reg_bspline_GetJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage,0);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage,0);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField_2D\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
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
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
                exit(1);
                break;
    }
    jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_3D<float>(velocityFieldImage, jacobianImage, approx);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_3D<double>(velocityFieldImage, jacobianImage, approx);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacobianMapFromVelocityField_2D<float>(velocityFieldImage, jacobianImage, approx);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacobianMapFromVelocityField_2D<double>(velocityFieldImage, jacobianImage, approx);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianMapFromVelocityField\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
                exit(1);
                break;
        }
    }

    // The value in the array are then integrated
    double jacobianNormalisedSum=0.0;
    switch(velocityFieldImage->datatype){
        case NIFTI_TYPE_FLOAT32:{
            float *singlePtr = static_cast<float *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                float temp = singlePtr[i];
//                if(temp<1.f)
                    temp = log(temp);
//                else temp--;
                jacobianNormalisedSum += (double)(temp*temp);
            }
            break;}
        case NIFTI_TYPE_FLOAT64:{
            double *doublePtr = static_cast<double *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                double temp = doublePtr[i];
//                if(temp<1.f)
                    temp = log(temp);
//                else temp--;
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
void reg_bspline_GetJacGradientFromVel_2D(  nifti_image *velocityFieldImage,
                                            nifti_image *targetImage,
                                            nifti_image *gradientImage,
                                            float weight,
                                            bool approx)
{

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4];
    ImageTYPE basis;

    ImageTYPE xControlPointCoordinates[16];
    ImageTYPE yControlPointCoordinates[16];

    ImageTYPE basisX[16];
    ImageTYPE basisY[16];

    ImageTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = velocityFieldImage->dx / targetImage->dx;
    gridVoxelSpacing[1] = velocityFieldImage->dy / targetImage->dy;

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(velocityFieldImage, &desorient, &reorient);

    // The deformation field array is allocated
    ImageTYPE *deformationFieldArrayX = (ImageTYPE *)malloc(targetImage->nvox*sizeof(ImageTYPE));
    ImageTYPE *deformationFieldArrayY = (ImageTYPE *)malloc(targetImage->nvox*sizeof(ImageTYPE));

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

    // The initial deformation field is initialised with the nifti header
    unsigned int jacIndex = 0;
    for(int y=0; y<targetImage->ny; y++){
        for(int x=0; x<targetImage->nx; x++){
            deformationFieldArrayX[jacIndex]
                = jac_xyz_matrix->m[0][0]*(ImageTYPE)x
                + jac_xyz_matrix->m[0][1]*(ImageTYPE)y
                + jac_xyz_matrix->m[0][3];
            deformationFieldArrayY[jacIndex]
                = jac_xyz_matrix->m[1][0]*(ImageTYPE)x
                + jac_xyz_matrix->m[1][1]*(ImageTYPE)y
                + jac_xyz_matrix->m[1][3];
            jacIndex++;
        }
    }
    unsigned int coord=0;

    ImageTYPE *gradientImagePtrX=static_cast<ImageTYPE *>(gradientImage->data);
    ImageTYPE *gradientImagePtrY=&gradientImagePtrX[gradientImage->nx*gradientImage->ny];

    ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *controlPointPtrY = &controlPointPtrX[velocityFieldImage->nx*velocityFieldImage->ny];

    for(int l=0;l<velocityFieldImage->pixdim[5];l++){

        int xPre, xPreOld=-99, yPre, yPreOld=-99;

        jacIndex=0;
        for(int y=0; y<targetImage->ny; y++){
            for(int x=0; x<targetImage->nx; x++){

                ImageTYPE realPosition[2];
                realPosition[0] = deformationFieldArrayX[jacIndex];
                realPosition[1] = deformationFieldArrayY[jacIndex];

                ImageTYPE voxelPosition[2];
                voxelPosition[0]
                    = jac_ijk_matrix->m[0][0]*realPosition[0]
                    + jac_ijk_matrix->m[0][1]*realPosition[1]
                    + jac_ijk_matrix->m[0][3];
                voxelPosition[1]
                    = jac_ijk_matrix->m[1][0]*realPosition[0]
                    + jac_ijk_matrix->m[1][1]*realPosition[1]
                    + jac_ijk_matrix->m[1][3];

                xPre=(int)floor((ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]);
                yPre=(int)floor((ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]);

                basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                Get_BSplineBasisValues<ImageTYPE>(basis, xBasis,xFirst);

                basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
                Get_BSplineBasisValues<ImageTYPE>(basis, yBasis,yFirst);

                ImageTYPE Tx_x=0.0;
                ImageTYPE Ty_x=0.0;
                ImageTYPE Tx_y=0.0;
                ImageTYPE Ty_y=0.0;
                ImageTYPE newDisplacementX = 0.0;
                ImageTYPE newDisplacementY = 0.0;
                if(approx){
                    xPre--;
                    yPre--;
                }
                if(xPre!=xPreOld || yPre!=yPreOld){
                    get_splineDisplacement(xPre,
                                           yPre,
                                           velocityFieldImage,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates);
                    xPreOld=xPre;
                    yPreOld=yPre;
                }


                coord=0;
                for(int b=0; b<4; b++){
                    for(int a=0; a<4; a++){
                        basisX[coord] = yBasis[b]*xFirst[a];   // y * x'
                        basisY[coord] = yFirst[b]*xBasis[a];   // y'* x
                        basis = yBasis[b]*xBasis[a];   // y * x
                        Tx_x += basisX[coord]*xControlPointCoordinates[coord];
                        Tx_y += basisY[coord]*xControlPointCoordinates[coord];
                        Ty_x += basisX[coord]*yControlPointCoordinates[coord];
                        Ty_y += basisY[coord]*yControlPointCoordinates[coord];
                        newDisplacementX += basis*xControlPointCoordinates[coord];
                        newDisplacementY += basis*yControlPointCoordinates[coord];
                        coord++;
                    }
                }

                deformationFieldArrayX[jacIndex] += newDisplacementX;
                deformationFieldArrayY[jacIndex] += newDisplacementY;

                jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)velocityFieldImage->dx);
                jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)velocityFieldImage->dy);
                jacobianMatrix.m[0][2]= 0.f;
                jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)velocityFieldImage->dx);
                jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)velocityFieldImage->dy);
                jacobianMatrix.m[1][2]= 0.f;
                jacobianMatrix.m[2][0]= 0.f;
                jacobianMatrix.m[2][1]= 0.f;
                jacobianMatrix.m[2][2]= 1.f;

                jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                jacobianMatrix.m[0][0]=1.f;
                jacobianMatrix.m[1][1]=1.f;
                jacobianMatrix.m[2][2]=1.f;
                ImageTYPE detJac = nifti_mat33_determ(jacobianMatrix);
                jacobianMatrix=nifti_mat33_inverse(jacobianMatrix);

                if(detJac>0){
                    detJac = 2.0f*log(detJac);
                }
                coord = 0;
                for(int Y=yPre; Y<yPre+4; Y++){
                    if(Y>-1 && Y<velocityFieldImage->ny){
                        unsigned int index = Y*velocityFieldImage->nx;
                        ImageTYPE *xxPtr = &gradientImagePtrX[index];
                        ImageTYPE *yyPtr = &gradientImagePtrY[index];
                        for(int X=xPre; X<xPre+4; X++){
                            if(X>-1 && X<velocityFieldImage->nx){
                                ImageTYPE gradientValueX = detJac *
                                    ( jacobianMatrix.m[0][0] * basisX[coord]
                                    + jacobianMatrix.m[0][1] * basisY[coord]);
                                ImageTYPE gradientValueY = detJac *
                                    ( jacobianMatrix.m[1][0] * basisX[coord]
                                    + jacobianMatrix.m[1][1] * basisY[coord]);

                                xxPtr[X] += weight *
                                    (desorient.m[0][0]*gradientValueX +
                                    desorient.m[0][1]*gradientValueY);
                                yyPtr[X] += weight *
                                    (desorient.m[1][0]*gradientValueX +
                                    desorient.m[1][1]*gradientValueY);
                            }
                            coord++;
                        } // a
                    }
                    else coord+=4;
                } // b

                jacIndex++;
            } // x
        } // y
    }

    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
}
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
void reg_bspline_GetJacGradientFromVel_3D(  nifti_image *velocityFieldImage,
                                            nifti_image *targetImage,
                                            nifti_image *gradientImage,
                                            float weight,
                                            bool approx)
{
    //TOCHECK
#if _USE_SSE
    if(sizeof(ImageTYPE)!=4){
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_GetJacobianGradient_3D\n");
        fprintf(stderr, "The SSE implementation assume single precision... Exit\n");
        exit(0);
    }
    union u{
        __m128 m;
        float f[4];
    } val;
#endif

    ImageTYPE xBasis[4],xFirst[4],yBasis[4],yFirst[4],zBasis[4],zFirst[4];
    ImageTYPE basis;

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

    mat33 reorient, desorient, jacobianMatrix;
    getReorientationMatrix(velocityFieldImage, &desorient, &reorient);

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
#if _USE_SSE
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
                    = jac_xyz_matrix->m[0][0]*(ImageTYPE)x
                    + jac_xyz_matrix->m[0][1]*(ImageTYPE)y
                    + jac_xyz_matrix->m[0][2]*(ImageTYPE)z
                    + jac_xyz_matrix->m[0][3];
                deformationFieldArrayY[jacIndex]
                    = jac_xyz_matrix->m[1][0]*(ImageTYPE)x
                    + jac_xyz_matrix->m[1][1]*(ImageTYPE)y
                    + jac_xyz_matrix->m[1][2]*(ImageTYPE)z
                    + jac_xyz_matrix->m[1][3];
                deformationFieldArrayZ[jacIndex]
                    = jac_xyz_matrix->m[2][0]*(ImageTYPE)x
                    + jac_xyz_matrix->m[2][1]*(ImageTYPE)y
                    + jac_xyz_matrix->m[2][2]*(ImageTYPE)z
                    + jac_xyz_matrix->m[2][3];
                jacIndex++;
            }
        }
    }
    unsigned int coord=0;

    ImageTYPE *gradientImagePtrX=static_cast<ImageTYPE *>(gradientImage->data);
    ImageTYPE *gradientImagePtrY=&gradientImagePtrX[gradientImage->nx*gradientImage->ny*gradientImage->nz];
    ImageTYPE *gradientImagePtrZ=&gradientImagePtrY[gradientImage->nx*gradientImage->ny*gradientImage->nz];

    ImageTYPE *controlPointPtrX = static_cast<ImageTYPE *>(velocityFieldImage->data);
    ImageTYPE *controlPointPtrY = &controlPointPtrX[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];
    ImageTYPE *controlPointPtrZ = &controlPointPtrY[velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz];

    ImageTYPE detJac;

    // The jacobian map is updated and then the deformation is composed
    for(int l=0;l<velocityFieldImage->pixdim[5];l++){

        int xPre, xPreOld=-99, yPre, yPreOld=-99, zPre, zPreOld=-99;

        jacIndex=0;
        for(int z=0; z<targetImage->nz; z++){
            for(int y=0; y<targetImage->ny; y++){
                for(int x=0; x<targetImage->nx; x++){

                    ImageTYPE realPosition[3];
                    realPosition[0] = deformationFieldArrayX[jacIndex];
                    realPosition[1] = deformationFieldArrayY[jacIndex];
                    realPosition[2] = deformationFieldArrayZ[jacIndex];

                    ImageTYPE voxelPosition[3];
#if _USE_SSE
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

                    basis=(ImageTYPE)voxelPosition[0]/gridVoxelSpacing[0]-(ImageTYPE)(xPre);
                    Get_BSplineBasisValues<ImageTYPE>(basis, xBasis,xFirst);

                    basis=(ImageTYPE)voxelPosition[1]/gridVoxelSpacing[1]-(ImageTYPE)(yPre);
                    Get_BSplineBasisValues<ImageTYPE>(basis, yBasis,yFirst);

                    basis=(ImageTYPE)voxelPosition[2]/gridVoxelSpacing[2]-(ImageTYPE)(zPre);
                    Get_BSplineBasisValues<ImageTYPE>(basis, zBasis,zFirst);

                    ImageTYPE Tx_x=0.0;
                    ImageTYPE Ty_x=0.0;
                    ImageTYPE Tz_x=0.0;
                    ImageTYPE Tx_y=0.0;
                    ImageTYPE Ty_y=0.0;
                    ImageTYPE Tz_y=0.0;
                    ImageTYPE Tx_z=0.0;
                    ImageTYPE Ty_z=0.0;
                    ImageTYPE Tz_z=0.0;
                    ImageTYPE newDisplacementX = 0.0;
                    ImageTYPE newDisplacementY = 0.0;
                    ImageTYPE newDisplacementZ = 0.0;
                    if(approx){
                        xPre--;
                        yPre--;
                        zPre--;
                    }
                    if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
                        coord=0;
                        get_splineDisplacement(xPre,
                                               yPre,
                                               zPre,
                                               velocityFieldImage,
                                               controlPointPtrX,
                                               controlPointPtrY,
                                               controlPointPtrZ,
                                               xControlPointCoordinates,
                                               yControlPointCoordinates,
                                               zControlPointCoordinates);
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
                    newDisplacementX = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = pos_y;
                    newDisplacementY = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m = pos_z;
                    newDisplacementZ = val.f[0]+val.f[1]+val.f[2]+val.f[3];
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
                                newDisplacementX += basis*xControlPointCoordinates[coord];
                                newDisplacementY += basis*yControlPointCoordinates[coord];
                                newDisplacementZ += basis*zControlPointCoordinates[coord];
                                coord++;
                            }
                        }
                    }
#endif

                    deformationFieldArrayX[jacIndex] += newDisplacementX;
                    deformationFieldArrayY[jacIndex] += newDisplacementY;
                    deformationFieldArrayZ[jacIndex] += newDisplacementZ;

                    jacobianMatrix.m[0][0]= (float)(Tx_x / (ImageTYPE)velocityFieldImage->dx);
                    jacobianMatrix.m[0][1]= (float)(Tx_y / (ImageTYPE)velocityFieldImage->dy);
                    jacobianMatrix.m[0][2]= (float)(Tx_z / (ImageTYPE)velocityFieldImage->dz);
                    jacobianMatrix.m[1][0]= (float)(Ty_x / (ImageTYPE)velocityFieldImage->dx);
                    jacobianMatrix.m[1][1]= (float)(Ty_y / (ImageTYPE)velocityFieldImage->dy);
                    jacobianMatrix.m[1][2]= (float)(Ty_z / (ImageTYPE)velocityFieldImage->dz);
                    jacobianMatrix.m[2][0]= (float)(Tz_x / (ImageTYPE)velocityFieldImage->dx);
                    jacobianMatrix.m[2][1]= (float)(Tz_y / (ImageTYPE)velocityFieldImage->dy);
                    jacobianMatrix.m[2][2]= (float)(Tz_z / (ImageTYPE)velocityFieldImage->dz);

                    jacobianMatrix=nifti_mat33_mul(jacobianMatrix,reorient);
                    jacobianMatrix.m[0][0]=1.f;
                    jacobianMatrix.m[1][1]=1.f;
                    jacobianMatrix.m[2][2]=1.f;
                    detJac = nifti_mat33_determ(jacobianMatrix);
                    jacobianMatrix=nifti_mat33_inverse(jacobianMatrix);

                    if(detJac>0){
                        detJac = log(detJac)>0?1:-1;
                    }
                    coord = 0;
                    for(int Z=zPre; Z<zPre+4; Z++){
                        if(Z>-1 && Z<velocityFieldImage->nz){
                            unsigned int index=Z*velocityFieldImage->nx*velocityFieldImage->ny;
                            ImageTYPE *xPtr = &gradientImagePtrX[index];
                            ImageTYPE *yPtr = &gradientImagePtrY[index];
                            ImageTYPE *zPtr = &gradientImagePtrZ[index];
                            for(int Y=yPre; Y<yPre+4; Y++){
                                if(Y>-1 && Y<velocityFieldImage->ny){
                                    index = Y*velocityFieldImage->nx;
                                    ImageTYPE *xxPtr = &xPtr[index];
                                    ImageTYPE *yyPtr = &yPtr[index];
                                    ImageTYPE *zzPtr = &zPtr[index];
                                    for(int X=xPre; X<xPre+4; X++){
                                        if(X>-1 && X<velocityFieldImage->nx){
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
                                        }
                                        coord++;
                                    }
                                }
                                else coord+=4;
                            }
                        }
                        else coord+=16;
                    }
                    jacIndex++;
                } // x
            } // y
        } // z
    }
    free(deformationFieldArrayX);
    free(deformationFieldArrayY);
    free(deformationFieldArrayZ);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_GetJacobianGradientFromVelocityField(  nifti_image* velocityFieldImage,
                                                        nifti_image* resultImage,
                                                        nifti_image* gradientImage,
                                                        float weight,
                                                        bool approx)
{
    // The Jacobian-based penalty term gradient is computed
    if(velocityFieldImage->nz>1){
        switch(velocityFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_bspline_GetJacGradientFromVel_3D<float>
                    (velocityFieldImage, resultImage, gradientImage, weight, approx);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_bspline_GetJacGradientFromVel_3D<double>
                    (velocityFieldImage, resultImage, gradientImage, weight, approx);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianGradientFromVelocityField\n");
                fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
                exit(1);
                break;
        }
    }
    else{
        switch(velocityFieldImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_GetJacGradientFromVel_2D<float>
                (velocityFieldImage, resultImage, gradientImage, weight, approx);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_GetJacGradientFromVel_2D<double>
                (velocityFieldImage, resultImage, gradientImage, weight, approx);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_bspline_GetJacobianGradientFromVelocityField\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision\n");
            exit(1);
            break;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                             nifti_image *deformationFieldImage,
                                             int *currentMask)
{
    nifti_image *tempGrid = nifti_copy_nim_info(velocityFieldGrid);
    tempGrid->data=(void *)malloc(tempGrid->nvox*tempGrid->nbyper);
    memcpy(tempGrid->data, velocityFieldGrid->data,
        tempGrid->nvox * tempGrid->nbyper);
    reg_getDeformationFromDisplacement(tempGrid);
    reg_bspline(tempGrid,
                deformationFieldImage,
                deformationFieldImage,
                currentMask,
                false, //composition
                true // bspline
                );
    for(int i=1; i<velocityFieldGrid->pixdim[5];++i){
        reg_bspline(tempGrid,
                    deformationFieldImage,
                    deformationFieldImage,
                    currentMask,
                    true, //composition
                    true // bspline
                    );
    }
    nifti_image_free(tempGrid);

}
/* *************************************************************** */
/* *************************************************************** */
void reg_getControlPointPositionFromVelocityGrid(nifti_image *velocityFieldGrid,
                                                 nifti_image *controlPointGrid)
{
    //TODO
}
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
template <class DTYPE>
int Interpolant2Interpolator(nifti_image *inputImage,
                             nifti_image *outputImage)
{
    if(inputImage->nz<1) inputImage->nz=1;
    if(outputImage->nz<1) outputImage->nz=1;

    DTYPE *inputPtrX = static_cast<DTYPE *>(inputImage->data);
    DTYPE *inputPtrY = &inputPtrX[inputImage->nx*inputImage->ny*inputImage->nz];
    DTYPE *inputPtrZ = NULL;
    if(inputImage->nz>1)
        inputPtrZ = &inputPtrY[inputImage->nx*inputImage->ny*inputImage->nz];

    DTYPE *outputPtrX = static_cast<DTYPE *>(outputImage->data);
    DTYPE *outputPtrY = &outputPtrX[outputImage->nx*outputImage->ny*outputImage->nz];
    DTYPE *outputPtrZ = NULL;
    if(inputImage->nz>1)
        outputPtrZ = &outputPtrY[outputImage->nx*outputImage->ny*outputImage->nz];

    double pole = (double)(sqrt(3.0) - 2.0);
    int increment;

    // X axis first
    double *values=new double[inputImage->nx];
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

        if(inputImage->nz>1){
            extractLine(start,end,increment,inputPtrZ,values);
            intensitiesToSplineCoefficients(values, inputImage->nx, pole);
            restoreLine(start,end,increment,outputPtrZ,values);
        }
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

        if(inputImage->nz>1){
            extractLine(start,end,increment,outputPtrZ,values);
            intensitiesToSplineCoefficients(values, inputImage->ny, pole);
            restoreLine(start,end,increment,outputPtrZ,values);
        }
    }
    delete[] values;

    // Z axis
    if(inputImage->nz>1){
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
    }
    return 0;
}
/* *************************************************************** */
int reg_spline_Interpolant2Interpolator(nifti_image *inputImage,
                                        nifti_image *outputImage)
{
    if(inputImage->datatype != outputImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_spline_Interpolant2Interpolator\n");
        fprintf(stderr,"[NiftyReg ERROR] Input and output image do not have the same data type. EXIT\n");
        exit(1);
    }
    for(int i=1;i<8;++i){
        if(inputImage->dim[i]!=outputImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_spline_Interpolant2Interpolator\n");
            fprintf(stderr,"[NiftyReg ERROR] Both images are expected to have the same size. EXIT\n");
            exit(1);
        }
    }

    switch(inputImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            Interpolant2Interpolator<float>(inputImage, outputImage);
            break;
        case NIFTI_TYPE_FLOAT64:
            Interpolant2Interpolator<double>(inputImage, outputImage);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_spline_Interpolant2Interpolator\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for float or double precision. EXIT\n");
            exit(1);
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */

#endif
