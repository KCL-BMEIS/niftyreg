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
    :reg_f3d<T>::reg_f3d(refTimePoint,floTimePoint)
{
    fprintf(stderr,"F3D2 is work under progress and should not be used at the moment.\n");
    fprintf(stderr,"I am currently doing modification to it.\n");
    fprintf(stderr,"EXIT.\n"); exit(1);

    this->controlPointPositionGrid=NULL;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2 constructor called\n");
#endif
    printf("[NiftyReg F3D] A stationnary velocity field is use to parametrise the deformation (F3D2)\n");
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d2<T>::~reg_f3d2()
{
    if(this->controlPointPositionGrid!=NULL)
        nifti_image_free(this->controlPointPositionGrid);
    this->controlPointPositionGrid=NULL;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2 destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::GetDeformationField()
{
    reg_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                            this->deformationFieldImage,
                                            this->currentMask);
//    reg_getControlPointPositionFromVelocityGrid<T>(this->controlPointGrid,
//                                                   this->controlPointPositionGrid);
//    reg_bspline<T>( this->controlPointPositionGrid,
//                    this->currentReference,
//                    this->deformationFieldImage,
//                    this->currentMask,
//                    0);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d2<T>::ComputeBendingEnergyPenaltyTerm()
{
    memcpy(this->controlPointPositionGrid->data, this->controlPointGrid->data,
            this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
    reg_getDeformationFromDisplacement(this->controlPointPositionGrid);
    double value = reg_bspline_bendingEnergy<T>(this->controlPointPositionGrid,
                                                this->currentReference,
                                                this->bendingEnergyApproximation
                                                );
    return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::GetBendingEnergyGradient()
{
    memcpy(this->controlPointPositionGrid->data, this->controlPointGrid->data,
            this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
    reg_getDeformationFromDisplacement(this->controlPointPositionGrid);
    reg_bspline_bendingEnergyGradient<T>(   this->controlPointPositionGrid,
                                            this->currentReference,
                                            this->nodeBasedMeasureGradientImage,
                                            this->bendingEnergyWeight);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d2<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
    double value;
    do{
        if(type==2){
            value = reg_bspline_GetJacobianValueFromVelocityField(this->controlPointGrid,
                                                                  this->currentReference,
                                                                  false);
        }
        else{
            value = reg_bspline_GetJacobianValueFromVelocityField(this->controlPointGrid,
                                                                  this->currentReference,
                                                                  this->jacobianLogApproximation);
        }

        if(value!=value){
            printf("Folding correction needed\n");
            return value;
        }
    }
    while(value!=value);
    return (double)this->jacobianLogWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::GetJacobianBasedGradient()
{
    reg_bspline_GetJacobianGradientFromVelocityField(this->controlPointGrid,
                                                     this->currentReference,
                                                     this->nodeBasedMeasureGradientImage,
                                                     this->jacobianLogWeight,
                                                     this->jacobianLogApproximation);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::UpdateControlPointPosition(T scale)
{
    T scaledScale = scale/(T)this->stepNumber;

    unsigned int nodeNumber = this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz;
    if(this->currentReference->nz==1){
        T *controlPointValuesX = static_cast<T *>(this->controlPointGrid->data);
        T *controlPointValuesY = &controlPointValuesX[nodeNumber];
        T *bestControlPointValuesX = &this->bestControlPointPosition[0];
        T *bestControlPointValuesY = &bestControlPointValuesX[nodeNumber];
        T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
        T *gradientValuesY = &gradientValuesX[nodeNumber];
        for(unsigned int i=0; i<nodeNumber;i++){
            *controlPointValuesX++ = *bestControlPointValuesX++ + scaledScale * *gradientValuesX++;
            *controlPointValuesY++ = *bestControlPointValuesY++ + scaledScale * *gradientValuesY++;
        }
    }
    else{
        T *controlPointValuesX = static_cast<T *>(this->controlPointGrid->data);
        T *controlPointValuesY = &controlPointValuesX[nodeNumber];
        T *controlPointValuesZ = &controlPointValuesY[nodeNumber];
        T *bestControlPointValuesX = &this->bestControlPointPosition[0];
        T *bestControlPointValuesY = &bestControlPointValuesX[nodeNumber];
        T *bestControlPointValuesZ = &bestControlPointValuesY[nodeNumber];
        T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
        T *gradientValuesY = &gradientValuesX[nodeNumber];
        T *gradientValuesZ = &gradientValuesY[nodeNumber];
        for(unsigned int i=0; i<nodeNumber;i++){
            *controlPointValuesX++ = *bestControlPointValuesX++ + scaledScale * *gradientValuesX++;
            *controlPointValuesY++ = *bestControlPointValuesY++ + scaledScale * *gradientValuesY++;
            *controlPointValuesZ++ = *bestControlPointValuesZ++ + scaledScale * *gradientValuesZ++;
        }
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::AllocateCurrentInputImage(int level)
{
    // The number of step is store in the pixdim[5]
    this->controlPointGrid->pixdim[5]=this->stepNumber;
    this->controlPointGrid->du=this->stepNumber;

    if(level==0){
        // Velocity field is set to 0
        if(this->inputControlPointGrid==NULL){
            memset(this->controlPointGrid->data, 0,
                   this->controlPointGrid->nvox*this->controlPointGrid->nbyper);
        }
    }
    else{
        reg_bspline_refineControlPointGrid(this->currentReference, this->controlPointGrid);
    }

    this->controlPointPositionGrid=nifti_copy_nim_info(this->controlPointGrid);
    this->controlPointPositionGrid->data=(void *)malloc(this->controlPointPositionGrid->nvox*
                                                        this->controlPointPositionGrid->nbyper);

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::ClearCurrentInputImage()
{
    reg_f3d<T>::ClearCurrentInputImage();
    if(this->controlPointPositionGrid!=NULL)
        nifti_image_free(this->controlPointPositionGrid);
    this->controlPointPositionGrid=NULL;
    return 0;
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
        fprintf(stderr,"[NiftyReg ERROR] reg_f3d::GetWarpedImage()\n");
        fprintf(stderr," * The reference, floating and control point grid images have to be defined\n");
    }

    this->currentReference = this->inputReference;
    this->currentFloating = this->inputFloating;

    reg_f3d2<T>::AllocateWarped();
    reg_f3d2<T>::AllocateDeformationField();
//    this->controlPointPositionGrid=nifti_copy_nim_info(this->controlPointGrid);
//    this->controlPointPositionGrid->data=(void *)malloc(this->controlPointPositionGrid->nvox*
//                                                        this->controlPointPositionGrid->nbyper);

    reg_f3d2<T>::WarpFloatingImage(3); // cubic spline interpolation

    this->controlPointPositionGrid=NULL;
    reg_f3d2<T>::ClearDeformationField();

    nifti_image *resultImage = nifti_copy_nim_info(this->warped);
    resultImage->data=(void *)malloc(resultImage->nvox*resultImage->nbyper);
    memcpy(resultImage->data, this->warped->data, resultImage->nvox*resultImage->nbyper);

    reg_f3d2<T>::ClearWarped();
//    nifti_image_free(this->controlPointPositionGrid);this->controlPointPositionGrid=NULL;
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
