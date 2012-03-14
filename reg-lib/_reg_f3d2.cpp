/*
 *  _reg_f3d2.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D2_CPP
#define _REG_F3D2_CPP

#include "_reg_f3d2.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d2<T>::reg_f3d2(int refTimePoint,int floTimePoint)
    :reg_f3d_sym<T>::reg_f3d_sym(refTimePoint,floTimePoint)
{
    this->executableName=(char *)"NiftyReg F3D2";
    this->stepNumber=6;
    this->inverseConsistencyWeight=0;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2 constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d2<T>::~reg_f3d2()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2 destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::SetCompositionStepNumber(int s)
{
    this->stepNumber = s;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d2<T>::Initisalise_f3d()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2::Initialise_f3d() called\n");
#endif

    reg_f3d_sym<T>::Initisalise_f3d();

    // Convert the deformation field into velocity field
    this->controlPointGrid->intent_code=this->stepNumber;
    this->backwardControlPointGrid->intent_code=-this->stepNumber;
    memset(this->controlPointGrid->intent_name, 0, 16);
    memset(this->backwardControlPointGrid->intent_name, 0, 16);
    strcpy(this->controlPointGrid->intent_name,"NREG_VEL_STEP");
    strcpy(this->backwardControlPointGrid->intent_name,"NREG_VEL_STEP");
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetDeformationField()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration forward\n");
#endif
    reg_bspline_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                    this->deformationFieldImage);
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration backward\n");
#endif
    reg_bspline_getDeformationFieldFromVelocityGrid(this->backwardControlPointGrid,
                                                    this->backwardDeformationFieldImage);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyErrorField()
{
    if(this->inverseConsistencyWeight<=0 || this->useInverseConsitency==false) return;

    if(this->similarityWeight<=0){
        reg_bspline_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                        this->deformationFieldImage);
        reg_bspline_getDeformationFieldFromVelocityGrid(this->backwardControlPointGrid,
                                                        this->backwardDeformationFieldImage);
    }
    nifti_image *tempForwardDeformationField=nifti_copy_nim_info(this->deformationFieldImage);
    nifti_image *tempBackwardDeformationField=nifti_copy_nim_info(this->backwardDeformationFieldImage);
    tempForwardDeformationField->data=(void *)malloc(tempForwardDeformationField->nbyper *
                                                     tempForwardDeformationField->nvox);
    tempBackwardDeformationField->data=(void *)malloc(tempBackwardDeformationField->nbyper *
                                                     tempBackwardDeformationField->nvox);
    memcpy(tempForwardDeformationField->data,this->deformationFieldImage,
           tempForwardDeformationField->nbyper *tempForwardDeformationField->nvox);
    memcpy(tempBackwardDeformationField->data,this->backwardDeformationFieldImage,
           tempBackwardDeformationField->nbyper *tempBackwardDeformationField->nvox);

    reg_defField_compose(tempBackwardDeformationField,
                         this->deformationFieldImage,
                         this->currentMask);
    reg_getDisplacementFromDeformation(this->deformationFieldImage);
    reg_defField_compose(tempForwardDeformationField,
                         this->backwardDeformationFieldImage,
                         this->currentFloatingMask);
    reg_getDisplacementFromDeformation(this->backwardDeformationFieldImage);
    nifti_image_free(tempForwardDeformationField);
    nifti_image_free(tempBackwardDeformationField);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyGradient()
{
    if(this->inverseConsistencyWeight<=0 || this->useInverseConsitency==false) return;

#warning TODO

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::UpdateControlPointPosition(T scale)
{
    this->RestoreCurrentControlPoint();

    // The velocity field is here updated using the BCH formulation
    mat33 *jacobianMatricesGrad=NULL;
    mat33 *jacobianMatricesVel=NULL;
    mat33 *jacobianMatricesBracket=NULL;

    nifti_image *lieBracket1=NULL;

    size_t node;

    /************************/
    /**** Forward update ****/
    /************************/

#ifndef NDEBUG
    printf("[NiftyReg f3d2] Update the forward control point grid using BCH approximation\n");
#endif

    size_t nodeNumber=this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz;

    // Scale the gradient image
    nifti_image *forwardScaledGradient=nifti_copy_nim_info(this->nodeBasedGradientImage);
    forwardScaledGradient->data=(void *)malloc(forwardScaledGradient->nvox*forwardScaledGradient->nbyper);
    reg_tools_addSubMulDivValue(this->nodeBasedGradientImage,
                                forwardScaledGradient,
                                scale,
                                2); // *(scale)

    // Allocate a nifti_image to store the new velocity field
    nifti_image *forwardNewVelocityField=nifti_copy_nim_info(this->controlPointGrid);
    forwardNewVelocityField->data=(void *)malloc(forwardNewVelocityField->nvox*forwardNewVelocityField->nbyper);

    // First computation of the BCH: new = grad + cpp
    reg_tools_addSubMulDivImages(forwardScaledGradient,
                                 this->controlPointGrid,
                                 forwardNewVelocityField,
                                 0); // addition

    // Convert the gradient into a deformation field to compute its Jacobian matrices
    reg_getDeformationFromDisplacement(forwardScaledGradient);

    // Allocate nifti_images to store the Lie bracket [grad,vel]
    lieBracket1=nifti_copy_nim_info(this->controlPointGrid);
    lieBracket1->data=(void *)malloc(lieBracket1->nvox*lieBracket1->nbyper);
/*
    // Allocate some arrays to store jacobian matrices
    jacobianMatricesGrad=(mat33 *)malloc(nodeNumber*sizeof(mat33));
    jacobianMatricesVel=(mat33 *)malloc(nodeNumber*sizeof(mat33));
    jacobianMatricesBracket=(mat33 *)malloc(nodeNumber*sizeof(mat33));

    // Compute Jacobian matrices
    reg_bspline_GetJacobianMatricesFromVelocityField(forwardScaledGradient,
                                                     forwardScaledGradient,
                                                     jacobianMatricesGrad);
    reg_getDisplacementFromDeformation(forwardScaledGradient);
    reg_bspline_GetJacobianMatricesFromVelocityField(this->controlPointGrid,
                                                     this->controlPointGrid,
                                                     jacobianMatricesVel);
    reg_getDisplacementFromDeformation(this->controlPointGrid);

    // Compute the first Lie bracket [grad,cpp] and add it to the new velocity grid
    // new += 0.5 [grad,cpp]
    T *lieBracket1PtrX=static_cast<T *>(lieBracket1->data);
    T *lieBracket1PtrY=&lieBracket1PtrX[nodeNumber];
    T *newVelocityPtrX=static_cast<T *>(forwardNewVelocityField->data);
    T *newVelocityPtrY=&newVelocityPtrX[nodeNumber];
    T *currentVelFieldPtrX=static_cast<T *>(this->controlPointGrid->data);
    T *currentVelFieldPtrY=&currentVelFieldPtrX[nodeNumber];
    T *scaledGradFieldPtrX=static_cast<T *>(forwardScaledGradient->data);
    T *scaledGradFieldPtrY=&scaledGradFieldPtrX[nodeNumber];
    T *lieBracket1PtrZ=NULL;
    T *newVelocityPtrZ=NULL;
    T *currentVelFieldPtrZ=NULL;
    T *scaledGradFieldPtrZ=NULL;

    if(this->controlPointGrid->nz>1){
        lieBracket1PtrZ=&lieBracket1PtrY[nodeNumber];
        newVelocityPtrZ=&newVelocityPtrY[nodeNumber];
        currentVelFieldPtrZ=&currentVelFieldPtrY[nodeNumber];
        scaledGradFieldPtrZ=&scaledGradFieldPtrY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node) \
        shared(nodeNumber,jacobianMatricesGrad,jacobianMatricesVel,scaledGradFieldPtrX, \
        scaledGradFieldPtrY,scaledGradFieldPtrZ,newVelocityPtrX,newVelocityPtrY, \
        newVelocityPtrZ,lieBracket1PtrX,lieBracket1PtrY,lieBracket1PtrZ, \
        currentVelFieldPtrX,currentVelFieldPtrY,currentVelFieldPtrZ)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            lieBracket1PtrX[node] =
                    jacobianMatricesGrad[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[0][1] * currentVelFieldPtrY[node] +
                    jacobianMatricesGrad[node].m[0][2] * currentVelFieldPtrZ[node] -
                    jacobianMatricesVel[node].m[0][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[0][1] * scaledGradFieldPtrY[node] -
                    jacobianMatricesVel[node].m[0][2] * scaledGradFieldPtrZ[node];
            lieBracket1PtrY[node] =
                    jacobianMatricesGrad[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[1][1] * currentVelFieldPtrY[node] +
                    jacobianMatricesGrad[node].m[1][2] * currentVelFieldPtrZ[node] -
                    jacobianMatricesVel[node].m[1][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[1][1] * scaledGradFieldPtrY[node] -
                    jacobianMatricesVel[node].m[1][2] * scaledGradFieldPtrZ[node];
            lieBracket1PtrZ[node] =
                    jacobianMatricesGrad[node].m[2][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[2][1] * currentVelFieldPtrY[node] +
                    jacobianMatricesGrad[node].m[2][2] * currentVelFieldPtrZ[node] -
                    jacobianMatricesVel[node].m[2][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[2][1] * scaledGradFieldPtrY[node] -
                    jacobianMatricesVel[node].m[2][2] * scaledGradFieldPtrZ[node];
            newVelocityPtrX[node] += 0.5f * lieBracket1PtrX[node];
            newVelocityPtrY[node] += 0.5f * lieBracket1PtrY[node];
            newVelocityPtrZ[node] += 0.5f * lieBracket1PtrZ[node];
        }
    }
    else{
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node) \
        shared(nodeNumber,jacobianMatricesGrad,jacobianMatricesVel,scaledGradFieldPtrX, \
        scaledGradFieldPtrY,newVelocityPtrX,newVelocityPtrY, \
        lieBracket1PtrX,lieBracket1PtrY,currentVelFieldPtrX, currentVelFieldPtrY)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            lieBracket1PtrX[node] =
                    jacobianMatricesGrad[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[0][1] * currentVelFieldPtrY[node] -
                    jacobianMatricesVel[node].m[0][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[0][1] * scaledGradFieldPtrY[node];
            lieBracket1PtrY[node] =
                    jacobianMatricesGrad[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[1][1] * currentVelFieldPtrY[node] -
                    jacobianMatricesVel[node].m[1][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[1][1] * scaledGradFieldPtrY[node];
            newVelocityPtrX[node] += 0.5f * lieBracket1PtrX[node];
            newVelocityPtrY[node] += 0.5f * lieBracket1PtrY[node];
        }
    }

    // Compute the Jacobian matrices from the the previously computed Lie bracket
    reg_getDeformationFromDisplacement(lieBracket1);
    reg_bspline_GetJacobianMatricesFromVelocityField(lieBracket1,
                                                     lieBracket1,
                                                     jacobianMatricesBracket);
    reg_getDisplacementFromDeformation(lieBracket1);

    // Update the velocity field
    if(this->controlPointGrid->nz>1){
        lieBracket1PtrZ=&lieBracket1PtrY[nodeNumber];
        newVelocityPtrZ=&newVelocityPtrY[nodeNumber];
        currentVelFieldPtrZ=&currentVelFieldPtrY[nodeNumber];
        scaledGradFieldPtrZ=&scaledGradFieldPtrY[nodeNumber];
        T tempLieBracket[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,tempLieBracket) \
        shared(nodeNumber,jacobianMatricesGrad,jacobianMatricesVel,jacobianMatricesBracket, \
        lieBracket1PtrX,lieBracket1PtrY,lieBracket1PtrZ, \
        currentVelFieldPtrX,currentVelFieldPtrY,currentVelFieldPtrZ, \
        newVelocityPtrX,newVelocityPtrY,newVelocityPtrZ, \
        scaledGradFieldPtrX,scaledGradFieldPtrY,scaledGradFieldPtrZ)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){

            tempLieBracket[0] =
                    jacobianMatrices1[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[0][1] * lieBracket1PtrY[node] +
                    jacobianMatrices1[node].m[0][2] * lieBracket1PtrZ[node] -
                    jacobianMatrices3[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * currentVelFieldPtrY[node] +
                    jacobianMatrices3[node].m[0][2] * currentVelFieldPtrZ[node];
            tempLieBracket[1] =
                    jacobianMatrices1[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[1][1] * lieBracket1PtrY[node] +
                    jacobianMatrices1[node].m[1][2] * lieBracket1PtrZ[node] -
                    jacobianMatrices3[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * currentVelFieldPtrY[node] +
                    jacobianMatrices3[node].m[1][2] * currentVelFieldPtrZ[node];
            tempLieBracket[2] =
                    jacobianMatrices1[node].m[2][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[2][1] * lieBracket1PtrY[node] +
                    jacobianMatrices1[node].m[2][2] * lieBracket1PtrZ[node] -
                    jacobianMatrices3[node].m[2][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[2][1] * currentVelFieldPtrY[node] +
                    jacobianMatrices3[node].m[2][2] * currentVelFieldPtrZ[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;
            newVelocityPtrZ[node] += tempLieBracket[2] / 12.f;

            tempLieBracket[0] =
                    jacobianMatrices3->m[0][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3->m[0][1] * scaledGradFieldPtrY[node] +
                    jacobianMatrices3->m[0][2] * scaledGradFieldPtrZ[node] -
                    jacobianMatrices2->m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2->m[0][1] * lieBracket1PtrY[node] +
                    jacobianMatrices2->m[0][2] * lieBracket1PtrZ[node];
            tempLieBracket[1] =
                    jacobianMatrices3->m[1][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3->m[1][1] * scaledGradFieldPtrY[node] +
                    jacobianMatrices3->m[1][2] * scaledGradFieldPtrZ[node] -
                    jacobianMatrices2->m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2->m[1][1] * lieBracket1PtrY[node] +
                    jacobianMatrices2->m[1][2] * lieBracket1PtrZ[node];
            tempLieBracket[2] =
                    jacobianMatrices3->m[2][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3->m[2][1] * scaledGradFieldPtrY[node] +
                    jacobianMatrices3->m[2][2] * scaledGradFieldPtrZ[node] -
                    jacobianMatrices2->m[2][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2->m[2][1] * lieBracket1PtrY[node] +
                    jacobianMatrices2->m[2][2] * lieBracket1PtrZ[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;
            newVelocityPtrZ[node] += tempLieBracket[2] / 12.f;
        }
    }
    else{
        T tempLieBracket[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,tempLieBracket) \
        shared(nodeNumber,jacobianMatricesGrad,jacobianMatricesVel,jacobianMatricesBracket, \
        lieBracket1PtrX,lieBracket1PtrY,currentVelFieldPtrX,currentVelFieldPtrY, \
        newVelocityPtrX,newVelocityPtrY,scaledGradFieldPtrX,scaledGradFieldPtrY)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){

            tempLieBracket[0] =
                    jacobianMatrices1[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[0][1] * lieBracket1PtrY[node] -
                    jacobianMatrices3[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * currentVelFieldPtrY[node];
            tempLieBracket[1] =
                    jacobianMatrices1[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[1][1] * lieBracket1PtrY[node] -
                    jacobianMatrices3[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * currentVelFieldPtrY[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;

            tempLieBracket[0] =
                    jacobianMatrices3[node].m[0][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * scaledGradFieldPtrY[node] -
                    jacobianMatrices2[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[0][1] * lieBracket1PtrY[node];
            tempLieBracket[1] =
                    jacobianMatrices3[node].m[1][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * scaledGradFieldPtrY[node] -
                    jacobianMatrices2[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[1][1] * lieBracket1PtrY[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;
        }
    }

    // Transfer the newly computed velocity field and clean the temporary nifti_images
    free(jacobianMatricesGrad);
    free(jacobianMatricesVel);
    free(jacobianMatricesBracket);
    nifti_image_free(lieBracket1);lieBracket1=NULL;
*/
    memcpy(this->controlPointGrid->data,forwardNewVelocityField->data,
           this->controlPointGrid->nbyper * this->controlPointGrid->nvox);
    nifti_image_free(forwardNewVelocityField);forwardNewVelocityField=NULL;
    nifti_image_free(forwardScaledGradient);forwardScaledGradient=NULL;

    /************************/
    /**** Backward update ***/
    /************************/

#ifndef NDEBUG
    printf("[NiftyReg f3d2] Update the backward control point grid using BCH approximation\n");
#endif

    nodeNumber=this->backwardControlPointGrid->nx*this->backwardControlPointGrid->ny*this->backwardControlPointGrid->nz;

    // Scale the gradient image
    nifti_image *backwardScaledGradient=nifti_copy_nim_info(this->backwardNodeBasedGradientImage);
    backwardScaledGradient->data=(void *)malloc(backwardScaledGradient->nvox*backwardScaledGradient->nbyper);
    reg_tools_addSubMulDivValue(this->backwardNodeBasedGradientImage,
                                backwardScaledGradient,
                                -scale,
                                2); // *(-scale)

    // Allocate a nifti_image to store the new velocity field
    nifti_image *backwardNewVelocityField=nifti_copy_nim_info(this->backwardControlPointGrid);
    backwardNewVelocityField->data=(void *)malloc(backwardNewVelocityField->nvox*backwardNewVelocityField->nbyper);

    // First computation of the BCH: new = grad + cpp
    reg_tools_addSubMulDivImages(backwardScaledGradient,
                                 this->backwardControlPointGrid,
                                 backwardNewVelocityField,
                                 0); // addition
/*
    // Convert the gradient into a deformation field to compute its Jacobian matrices
    reg_getDeformationFromDisplacement(backwardScaledGradient);

    // Allocate nifti_images to store the Lie bracket [grad,vel]
    lieBracket1=nifti_copy_nim_info(this->backwardControlPointGrid);
    lieBracket1->data=(void *)malloc(lieBracket1->nvox*lieBracket1->nbyper);

    // Allocate some arrays to store jacobian matrices
    jacobianMatricesGrad=(mat33 *)malloc(nodeNumber*sizeof(mat33));
    jacobianMatricesVel=(mat33 *)malloc(nodeNumber*sizeof(mat33));
    jacobianMatricesBracket=(mat33 *)malloc(nodeNumber*sizeof(mat33));

    // Compute Jacobian matrices
    reg_bspline_GetJacobianMatricesFromVelocityField(backwardScaledGradient,
                                                     backwardScaledGradient,
                                                     jacobianMatricesGrad);
    reg_getDisplacementFromDeformation(backwardScaledGradient);
    reg_bspline_GetJacobianMatricesFromVelocityField(this->backwardControlPointGrid,
                                                     this->backwardControlPointGrid,
                                                     jacobianMatricesVel);
    reg_getDisplacementFromDeformation(this->backwardControlPointGrid);

    // Compute the first Lie bracket [grad,cpp] and add it to the new velocity grid
    // new += 0.5 [grad,cpp]
    lieBracket1PtrX=static_cast<T *>(lieBracket1->data);
    lieBracket1PtrY=&lieBracket1PtrX[nodeNumber];
    newVelocityPtrX=static_cast<T *>(backwardNewVelocityField->data);
    newVelocityPtrY=&newVelocityPtrX[nodeNumber];
    currentVelFieldPtrX=static_cast<T *>(this->backwardControlPointGrid->data);
    currentVelFieldPtrY=&currentVelFieldPtrX[nodeNumber];
    scaledGradFieldPtrX=static_cast<T *>(backwardScaledGradient->data);
    scaledGradFieldPtrY=&scaledGradFieldPtrX[nodeNumber];
    lieBracket1PtrZ=NULL;
    newVelocityPtrZ=NULL;
    currentVelFieldPtrZ=NULL;
    scaledGradFieldPtrZ=NULL;

    if(this->controlPointGrid->nz>1){
        lieBracket1PtrZ=&lieBracket1PtrY[nodeNumber];
        newVelocityPtrZ=&newVelocityPtrY[nodeNumber];
        currentVelFieldPtrZ=&currentVelFieldPtrY[nodeNumber];
        scaledGradFieldPtrZ=&scaledGradFieldPtrY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node) \
        shared(nodeNumber,jacobianMatricesGrad,jacobianMatricesVel,scaledGradFieldPtrX, \
        scaledGradFieldPtrY,scaledGradFieldPtrZ,newVelocityPtrX,newVelocityPtrY, \
        newVelocityPtrZ,lieBracket1PtrX,lieBracket1PtrY,lieBracket1PtrZ, \
        currentVelFieldPtrX,currentVelFieldPtrY,currentVelFieldPtrZ)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            lieBracket1PtrX[node] =
                    jacobianMatricesGrad[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[0][1] * currentVelFieldPtrY[node] +
                    jacobianMatricesGrad[node].m[0][2] * currentVelFieldPtrZ[node] -
                    jacobianMatricesVel[node].m[0][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[0][1] * scaledGradFieldPtrY[node] -
                    jacobianMatricesVel[node].m[0][2] * scaledGradFieldPtrZ[node];
            lieBracket1PtrY[node] =
                    jacobianMatricesGrad[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[1][1] * currentVelFieldPtrY[node] +
                    jacobianMatricesGrad[node].m[1][2] * currentVelFieldPtrZ[node] -
                    jacobianMatricesVel[node].m[1][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[1][1] * scaledGradFieldPtrY[node] -
                    jacobianMatricesVel[node].m[1][2] * scaledGradFieldPtrZ[node];
            lieBracket1PtrZ[node] =
                    jacobianMatricesGrad[node].m[2][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[2][1] * currentVelFieldPtrY[node] +
                    jacobianMatricesGrad[node].m[2][2] * currentVelFieldPtrZ[node] -
                    jacobianMatricesVel[node].m[2][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[2][1] * scaledGradFieldPtrY[node] -
                    jacobianMatricesVel[node].m[2][2] * scaledGradFieldPtrZ[node];
            newVelocityPtrX[node] += 0.5f * lieBracket1PtrX[node];
            newVelocityPtrY[node] += 0.5f * lieBracket1PtrY[node];
            newVelocityPtrZ[node] += 0.5f * lieBracket1PtrZ[node];
        }
    }
    else{
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node) \
        shared(nodeNumber,jacobianMatricesGrad,jacobianMatricesVel,scaledGradFieldPtrX, \
        scaledGradFieldPtrY,newVelocityPtrX,newVelocityPtrY, \
        lieBracket1PtrX,lieBracket1PtrY,currentVelFieldPtrX, currentVelFieldPtrY)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            lieBracket1PtrX[node] =
                    jacobianMatricesGrad[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[0][1] * currentVelFieldPtrY[node] -
                    jacobianMatricesVel[node].m[0][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[0][1] * scaledGradFieldPtrY[node];
            lieBracket1PtrY[node] =
                    jacobianMatricesGrad[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatricesGrad[node].m[1][1] * currentVelFieldPtrY[node] -
                    jacobianMatricesVel[node].m[1][0] * scaledGradFieldPtrX[node] -
                    jacobianMatricesVel[node].m[1][1] * scaledGradFieldPtrY[node];
            newVelocityPtrX[node] += 0.5f * lieBracket1PtrX[node];
            newVelocityPtrY[node] += 0.5f * lieBracket1PtrY[node];
        }
    }

    // Compute the Jacobian matrices from the the previously computed Lie bracket
    reg_getDeformationFromDisplacement(lieBracket1);
    reg_bspline_GetJacobianMatricesFromVelocityField(lieBracket1,
                                                     lieBracket1,
                                                     jacobianMatrices3);
    reg_getDisplacementFromDeformation(lieBracket1);

    // Update the velocity field
    if(this->backwardControlPointGrid->nz>1){
        lieBracket1PtrZ=&lieBracket1PtrY[nodeNumber];
        newVelocityPtrZ=&newVelocityPtrY[nodeNumber];
        currentVelFieldPtrZ=&currentVelFieldPtrY[nodeNumber];
        scaledGradFieldPtrZ=&scaledGradFieldPtrY[nodeNumber];
        T tempLieBracket[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,tempLieBracket) \
        shared(nodeNumber,jacobianMatrices1,jacobianMatrices2,jacobianMatrices3, \
        lieBracket1PtrX,lieBracket1PtrY,lieBracket1PtrZ, \
        currentVelFieldPtrX,currentVelFieldPtrY,currentVelFieldPtrZ, \
        newVelocityPtrX,newVelocityPtrY,newVelocityPtrZ, \
        scaledGradFieldPtrX,scaledGradFieldPtrY,scaledGradFieldPtrZ)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){

            tempLieBracket[0] =
                    jacobianMatrices1[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[0][1] * lieBracket1PtrY[node] +
                    jacobianMatrices1[node].m[0][2] * lieBracket1PtrZ[node] -
                    jacobianMatrices3[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * currentVelFieldPtrY[node] +
                    jacobianMatrices3[node].m[0][2] * currentVelFieldPtrZ[node];
            tempLieBracket[1] =
                    jacobianMatrices1[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[1][1] * lieBracket1PtrY[node] +
                    jacobianMatrices1[node].m[1][2] * lieBracket1PtrZ[node] -
                    jacobianMatrices3[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * currentVelFieldPtrY[node] +
                    jacobianMatrices3[node].m[1][2] * currentVelFieldPtrZ[node];
            tempLieBracket[2] =
                    jacobianMatrices1[node].m[2][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[2][1] * lieBracket1PtrY[node] +
                    jacobianMatrices1[node].m[2][2] * lieBracket1PtrZ[node] -
                    jacobianMatrices3[node].m[2][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[2][1] * currentVelFieldPtrY[node] +
                    jacobianMatrices3[node].m[2][2] * currentVelFieldPtrZ[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;
            newVelocityPtrZ[node] += tempLieBracket[2] / 12.f;

            tempLieBracket[0] =
                    jacobianMatrices3[node].m[0][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * scaledGradFieldPtrY[node] +
                    jacobianMatrices3[node].m[0][2] * scaledGradFieldPtrZ[node] -
                    jacobianMatrices2[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[0][1] * lieBracket1PtrY[node] +
                    jacobianMatrices2[node].m[0][2] * lieBracket1PtrZ[node];
            tempLieBracket[1] =
                    jacobianMatrices3[node].m[1][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * scaledGradFieldPtrY[node] +
                    jacobianMatrices3[node].m[1][2] * scaledGradFieldPtrZ[node] -
                    jacobianMatrices2[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[1][1] * lieBracket1PtrY[node] +
                    jacobianMatrices2[node].m[1][2] * lieBracket1PtrZ[node];
            tempLieBracket[2] =
                    jacobianMatrices3[node].m[2][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[2][1] * scaledGradFieldPtrY[node] +
                    jacobianMatrices3[node].m[2][2] * scaledGradFieldPtrZ[node] -
                    jacobianMatrices2[node].m[2][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[2][1] * lieBracket1PtrY[node] +
                    jacobianMatrices2[node].m[2][2] * lieBracket1PtrZ[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;
            newVelocityPtrZ[node] += tempLieBracket[2] / 12.f;
        }
    }
    else{
        T tempLieBracket[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,tempLieBracket) \
        shared(nodeNumber,jacobianMatrices1,jacobianMatrices2,jacobianMatrices3, \
        lieBracket1PtrX,lieBracket1PtrY,currentVelFieldPtrX,currentVelFieldPtrY, \
        newVelocityPtrX,newVelocityPtrY,offset, scaledGradFieldPtrX,scaledGradFieldPtrY)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){

            tempLieBracket[0] =
                    jacobianMatrices1[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[0][1] * lieBracket1PtrY[node] -
                    jacobianMatrices3[node].m[0][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * currentVelFieldPtrY[node];
            tempLieBracket[1] =
                    jacobianMatrices1[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices1[node].m[1][1] * lieBracket1PtrY[node] -
                    jacobianMatrices3[node].m[1][0] * currentVelFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * currentVelFieldPtrY[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;

            tempLieBracket[0] =
                    jacobianMatrices3[node].m[0][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[0][1] * scaledGradFieldPtrY[node] -
                    jacobianMatrices2[node].m[0][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[0][1] * lieBracket1PtrY[node];
            tempLieBracket[1] =
                    jacobianMatrices3[node].m[1][0] * scaledGradFieldPtrX[node] +
                    jacobianMatrices3[node].m[1][1] * scaledGradFieldPtrY[node] -
                    jacobianMatrices2[node].m[1][0] * lieBracket1PtrX[node] +
                    jacobianMatrices2[node].m[1][1] * lieBracket1PtrY[node];
            newVelocityPtrX[node] += tempLieBracket[0] / 12.f ;
            newVelocityPtrY[node] += tempLieBracket[1] / 12.f;
        }
    }

    // Transfer the newly computed velocity field and clean the temporary nifti_images
    free(jacobianMatricesGrad);
    free(jacobianMatricesVel);
    free(jacobianMatricesBracket);
    nifti_image_free(lieBracket1);lieBracket1=NULL;
*/
    memcpy(this->backwardControlPointGrid->data,backwardNewVelocityField->data,
           this->backwardControlPointGrid->nbyper * this->backwardControlPointGrid->nvox);
    nifti_image_free(backwardNewVelocityField);backwardNewVelocityField=NULL;
    nifti_image_free(backwardScaledGradient);backwardScaledGradient=NULL;

return;
    /****************************/
    /******** Symmetrise ********/
    /****************************/

#ifndef NDEBUG
    printf("[NiftyReg f3d2] Ensure symmetry\n");
#endif

    // In order to ensure symmetry the forward and backward velocity fields
    // are averaged in both image spaces: reference and floating

    // Propagate the forward transformation within the backward space
    nifti_image *forward2backward = nifti_copy_nim_info(this->backwardControlPointGrid);
    nifti_image *forward2backwardDEF = nifti_copy_nim_info(this->backwardControlPointGrid);
    forward2backward->data=(void *)malloc(forward2backward->nvox*forward2backward->nbyper);
    forward2backwardDEF->data=(void *)malloc(forward2backwardDEF->nvox*forward2backwardDEF->nbyper);
    mat44 affine_forward2backward;
    mat44 *vox2real_bck=NULL;
    if(this->backwardControlPointGrid->sform_code>0)
        vox2real_bck=&this->backwardControlPointGrid->sto_xyz;
    else vox2real_bck=&this->backwardControlPointGrid->qto_xyz;
    mat44 *real2vox_for=NULL;
    if(this->controlPointGrid->sform_code>0)
        real2vox_for=&this->controlPointGrid->sto_ijk;
    else real2vox_for=&this->controlPointGrid->qto_ijk;
    affine_forward2backward=reg_mat44_mul(real2vox_for,vox2real_bck);
    reg_affine_positionField(&affine_forward2backward, // affine transformation
                             this->backwardControlPointGrid, // reference space
                             forward2backwardDEF); // deformation field
    reg_getDisplacementFromDeformation(this->controlPointGrid); // in order to use a zero padding
    reg_resampleSourceImage(this->backwardControlPointGrid, // reference
                            this->controlPointGrid, // floating
                            forward2backward, // warped
                            forward2backwardDEF, // deformation field
                            NULL, // no mask
                            3, // cubic spline interpolation
                            0.f // padding
                            );
    nifti_image_free(forward2backwardDEF);forward2backwardDEF=NULL;

    // Propagate the backward transformation within the forward space
    nifti_image *backward2forward = nifti_copy_nim_info(this->controlPointGrid);
    nifti_image *backward2forwardDEF = nifti_copy_nim_info(this->controlPointGrid);
    backward2forward->data=(void *)malloc(backward2forward->nvox*backward2forward->nbyper);
    backward2forwardDEF->data=(void *)malloc(backward2forwardDEF->nvox*backward2forwardDEF->nbyper);
    mat44 affine_backward2forward;
    mat44 *vox2real_for=NULL;
    if(this->controlPointGrid->sform_code>0)
        vox2real_for=&this->controlPointGrid->sto_xyz;
    else vox2real_for=&this->controlPointGrid->qto_xyz;
    mat44 *real2vox_bck=NULL;
    if(this->backwardControlPointGrid->sform_code>0)
        real2vox_bck=&this->backwardControlPointGrid->sto_ijk;
    else real2vox_bck=&this->backwardControlPointGrid->qto_ijk;
    affine_backward2forward=reg_mat44_mul(real2vox_bck,vox2real_for);
    reg_affine_positionField(&affine_backward2forward, // affine transformation
                             this->controlPointGrid, // reference space
                             backward2forwardDEF); // deformation field
    reg_getDisplacementFromDeformation(this->backwardControlPointGrid); // in order to use a zero padding
    reg_resampleSourceImage(this->controlPointGrid, // reference
                            this->backwardControlPointGrid, // floating
                            backward2forward, // warped
                            backward2forwardDEF, // deformation field
                            NULL, // no mask
                            3, // cubic spline interpolation
                            0.f // padding
                            );
    nifti_image_free(backward2forwardDEF);backward2forwardDEF=NULL;

    // Average the transformations in the backward space
    nodeNumber=this->backwardControlPointGrid->nx*this->backwardControlPointGrid->ny*this->backwardControlPointGrid->nz;
    T *currentVelFieldPtrX=static_cast<T *>(this->backwardControlPointGrid->data);
    T *currentVelFieldPtrY=&currentVelFieldPtrX[nodeNumber];
    T *propVelFieldPtrX=static_cast<T *>(forward2backward->data);
    T *propVelFieldPtrY=&propVelFieldPtrX[nodeNumber];

    if(this->backwardControlPointGrid->nz>1){
        T *currentVelFieldPtrZ=&currentVelFieldPtrY[nodeNumber];
        T *propVelFieldPtrZ=&propVelFieldPtrY[nodeNumber];
        T reoriented[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,reoriented) \
        shared(affine_forward2backward,nodeNumber, \
        propVelFieldPtrX,propVelFieldPtrY,propVelFieldPtrZ, \
        currentVelFieldPtrX,currentVelFieldPtrY,currentVelFieldPtrZ)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            reoriented[0] =
                    affine_forward2backward.m[0][0] * propVelFieldPtrX[node] +
                    affine_forward2backward.m[0][1] * propVelFieldPtrY[node] +
                    affine_forward2backward.m[0][2] * propVelFieldPtrZ[node];
            reoriented[1] =
                    affine_forward2backward.m[1][0] * propVelFieldPtrX[node] +
                    affine_forward2backward.m[1][1] * propVelFieldPtrY[node] +
                    affine_forward2backward.m[1][2] * propVelFieldPtrZ[node];
            reoriented[2] =
                    affine_forward2backward.m[2][0] * propVelFieldPtrX[node] +
                    affine_forward2backward.m[2][1] * propVelFieldPtrY[node] +
                    affine_forward2backward.m[2][2] * propVelFieldPtrZ[node];
            // The transformation will be negated while performing the exponentiation
            currentVelFieldPtrX[node] = ( reoriented[0] - currentVelFieldPtrX[node] ) / 2.;
            currentVelFieldPtrY[node] = ( reoriented[1] - currentVelFieldPtrY[node] ) / 2.;
            currentVelFieldPtrZ[node] = ( reoriented[2] - currentVelFieldPtrZ[node] ) / 2.;
        }
    }
    else{
        T reoriented[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,reoriented) \
        shared(affine_forward2backward,nodeNumber, \
        propVelFieldPtrX,propVelFieldPtrY, \
        currentVelFieldPtrX,currentVelFieldPtrY)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            reoriented[0] =
                    affine_forward2backward.m[0][0] * propVelFieldPtrX[node] +
                    affine_forward2backward.m[0][1] * propVelFieldPtrY[node];
            reoriented[1] =
                    affine_forward2backward.m[1][0] * propVelFieldPtrX[node] +
                    affine_forward2backward.m[1][1] * propVelFieldPtrY[node];
            currentVelFieldPtrX[node] = ( currentVelFieldPtrX[node] - reoriented[0] ) / 2.;
            currentVelFieldPtrY[node] = ( currentVelFieldPtrY[node] - reoriented[1] ) / 2.;
        }
    }
    nifti_image_free(forward2backward);forward2backward=NULL;

    // Average the transformations in the forward space
    nodeNumber=this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz;
    currentVelFieldPtrX=static_cast<T *>(this->controlPointGrid->data);
    currentVelFieldPtrY=&currentVelFieldPtrX[nodeNumber];
    propVelFieldPtrX=static_cast<T *>(backward2forward->data);
    propVelFieldPtrY=&propVelFieldPtrX[nodeNumber];

    if(this->backwardControlPointGrid->nz>1){
        T *currentVelFieldPtrZ=&currentVelFieldPtrY[nodeNumber];
        T *propVelFieldPtrZ=&propVelFieldPtrY[nodeNumber];
        T reoriented[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,reoriented) \
        shared(affine_backward2forward,nodeNumber, \
        propVelFieldPtrX,propVelFieldPtrY,propVelFieldPtrZ, \
        currentVelFieldPtrX,currentVelFieldPtrY,currentVelFieldPtrZ)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            reoriented[0] =
                    affine_backward2forward.m[0][0] * propVelFieldPtrX[node] +
                    affine_backward2forward.m[0][1] * propVelFieldPtrY[node] +
                    affine_backward2forward.m[0][2] * propVelFieldPtrZ[node];
            reoriented[1] =
                    affine_backward2forward.m[1][0] * propVelFieldPtrX[node] +
                    affine_backward2forward.m[1][1] * propVelFieldPtrY[node] +
                    affine_backward2forward.m[1][2] * propVelFieldPtrZ[node];
            reoriented[2] =
                    affine_backward2forward.m[2][0] * propVelFieldPtrX[node] +
                    affine_backward2forward.m[2][1] * propVelFieldPtrY[node] +
                    affine_backward2forward.m[2][2] * propVelFieldPtrZ[node];
            currentVelFieldPtrX[node] = ( currentVelFieldPtrX[node] - reoriented[0] ) / 2.;
            currentVelFieldPtrY[node] = ( currentVelFieldPtrY[node] - reoriented[1] ) / 2.;
            currentVelFieldPtrZ[node] = ( currentVelFieldPtrZ[node] - reoriented[2] ) / 2.;
        }
    }
    else{
        T reoriented[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        private(node,reoriented) \
        shared(affine_backward2forward,nodeNumber, \
        propVelFieldPtrX,propVelFieldPtrY,currentVelFieldPtrX,currentVelFieldPtrY)
#endif // _OPENMP
        for(node=0;node<nodeNumber;++node){
            reoriented[0] =
                    affine_backward2forward.m[0][0] * propVelFieldPtrX[node] +
                    affine_backward2forward.m[0][1] * propVelFieldPtrY[node];
            reoriented[1] =
                    affine_backward2forward.m[1][0] * propVelFieldPtrX[node] +
                    affine_backward2forward.m[1][1] * propVelFieldPtrY[node];
            currentVelFieldPtrX[node] = ( currentVelFieldPtrX[node] - reoriented[0] ) / 2.;
            currentVelFieldPtrY[node] = ( currentVelFieldPtrY[node] - reoriented[1] ) / 2.;
        }
    }
    nifti_image_free(forward2backward);forward2backward=NULL;

    reg_getDeformationFromDisplacement(this->controlPointGrid);
    reg_getDeformationFromDisplacement(this->backwardControlPointGrid);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image *reg_f3d2<T>::GetWarpedImage()
{
    // The initial images are used
    if(this->inputReference==NULL ||
       this->inputFloating==NULL ||
       this->controlPointGrid==NULL){
        fprintf(stderr,"[NiftyReg ERROR] reg_f3d2::GetWarpedImage()\n");
        fprintf(stderr," * The reference, floating and control point grid images have to be defined\n");
    }

    this->currentReference = this->inputReference;
    this->currentFloating = this->inputFloating;
    this->currentMask = NULL;
    this->currentFloatingMask = NULL;

    reg_f3d2<T>::AllocateWarped();
    reg_f3d2<T>::AllocateDeformationField();

    reg_f3d2<T>::WarpFloatingImage(3); // cubic spline interpolation

    reg_f3d2<T>::ClearDeformationField();

    nifti_image *resultImage = nifti_copy_nim_info(this->warped);
    resultImage->cal_min=this->inputFloating->cal_min;
    resultImage->cal_max=this->inputFloating->cal_max;
    resultImage->scl_slope=this->inputFloating->scl_slope;
    resultImage->scl_inter=this->inputFloating->scl_inter;
    resultImage->data=(void *)malloc(resultImage->nvox*resultImage->nbyper);
    memcpy(resultImage->data, this->warped->data, resultImage->nvox*resultImage->nbyper);

    reg_f3d2<T>::ClearWarped();
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
