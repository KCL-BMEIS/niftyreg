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
    this->executableName=(char *)"NiftyReg F3D2";
    this->inverseDeformationFieldImage=NULL;
    this->negatedControlPointGrid=NULL;
//    this->useSymmetry=false;

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
int reg_f3d2<T>::AllocateDeformationField()
{
    this->ClearDeformationField();
    reg_f3d<T>::AllocateDeformationField();

//    if(this->useSymmetry){
//        this->inverseDeformationFieldImage = nifti_copy_nim_info(this->currentFloating);
//        this->inverseDeformationFieldImage->dim[0]=this->inverseDeformationFieldImage->ndim=5;
//        this->inverseDeformationFieldImage->dim[1]=this->inverseDeformationFieldImage->nx=this->currentFloating->nx;
//        this->inverseDeformationFieldImage->dim[2]=this->inverseDeformationFieldImage->ny=this->currentFloating->ny;
//        this->inverseDeformationFieldImage->dim[3]=this->inverseDeformationFieldImage->nz=this->currentFloating->nz;
//        this->inverseDeformationFieldImage->dim[4]=this->inverseDeformationFieldImage->nt=1;
//        this->inverseDeformationFieldImage->pixdim[4]=this->inverseDeformationFieldImage->dt=1.0;
//        if(this->currentFloating->nz==1)
//            this->inverseDeformationFieldImage->dim[5]=this->inverseDeformationFieldImage->nu=2;
//        else this->inverseDeformationFieldImage->dim[5]=this->inverseDeformationFieldImage->nu=3;
//        this->inverseDeformationFieldImage->pixdim[5]=this->inverseDeformationFieldImage->du=1.0;
//        this->inverseDeformationFieldImage->dim[6]=this->inverseDeformationFieldImage->nv=1;
//        this->inverseDeformationFieldImage->pixdim[6]=this->inverseDeformationFieldImage->dv=1.0;
//        this->inverseDeformationFieldImage->dim[7]=this->inverseDeformationFieldImage->nw=1;
//        this->inverseDeformationFieldImage->pixdim[7]=this->inverseDeformationFieldImage->dw=1.0;
//        this->inverseDeformationFieldImage->nvox=this->inverseDeformationFieldImage->nx *
//                                            this->inverseDeformationFieldImage->ny *
//                                            this->inverseDeformationFieldImage->nz *
//                                            this->inverseDeformationFieldImage->nt *
//                                            this->inverseDeformationFieldImage->nu;
//        this->inverseDeformationFieldImage->nbyper = this->controlPointGrid->nbyper;
//        this->inverseDeformationFieldImage->datatype = this->controlPointGrid->datatype;
//        this->inverseDeformationFieldImage->data = (void *)calloc(this->inverseDeformationFieldImage->nvox,
//                                                                  this->inverseDeformationFieldImage->nbyper);
//    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::ClearDeformationField()
{
    reg_f3d<T>::ClearDeformationField();
//    if(this->inverseDeformationFieldImage!=NULL){
//        nifti_image_free(this->inverseDeformationFieldImage);
//        this->inverseDeformationFieldImage=NULL;
//    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::GetDeformationField()
{
//    if(this->useSymmetry){
//        memcpy(this->negatedControlPointGrid->data,this->controlPointGrid->data,
//               this->negatedControlPointGrid->nvox*this->negatedControlPointGrid->nbyper);
//        reg_getDisplacementFromDeformation(this->negatedControlPointGrid);
//        reg_tools_addSubMulDivValue(this->negatedControlPointGrid,this->negatedControlPointGrid,-1.0f,2);
//        reg_getDeformationFromDisplacement(this->negatedControlPointGrid);
//    }

    if(this->f3d2AppFreeStep){
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration performed without approximation\n");
#endif
        reg_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                this->deformationFieldImage,
                                                this->currentMask,
                                                false // approximation
                                                );
//        if(this->useSymmetry){
//            reg_getDeformationFieldFromVelocityGrid(this->negatedControlPointGrid,
//                                                    this->inverseDeformationFieldImage,
//                                                    NULL,
//                                                    false // approximation
//                                                    );
//        }
    }
    else{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration performed with approximation\n");
#endif
        reg_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                this->deformationFieldImage,
                                                this->currentMask,
                                                true // approximation
                                                );
//        if(this->useSymmetry){
//            reg_getDeformationFieldFromVelocityGrid(this->negatedControlPointGrid,
//                                                    this->inverseDeformationFieldImage,
//                                                    NULL,
//                                                    true // approximation
//                                                    );
//        }
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::AllocateCurrentInputImage(int level)
{
    if(this->affineTransformation!=NULL){
        fprintf(stderr, "[NiftyReg ERROR] The velocity field parametrisation does not handle affine input\n");
        fprintf(stderr, "[NiftyReg ERROR] Please update your source image sform using reg_transform\n");
        fprintf(stderr, "[NiftyReg ERROR] and use the updated source image as an input\n.");
        exit(1);
    }

    // The number of step is store in the pixdim[5]
    if(this->inputControlPointGrid==NULL){
        this->controlPointGrid->pixdim[5]=this->stepNumber;
        this->controlPointGrid->du=this->stepNumber;
    }
    else this->stepNumber=this->controlPointGrid->du;

#ifdef NDEBUG
    if(this->verbose){
#endif
        printf("[%s] Velocity field integration with %i steps,\n",
               this->executableName, (int)pow(2,this->controlPointGrid->du));
        printf("[%s] squaring approximation is performed using %i steps\n",
               this->executableName, (int)this->controlPointGrid->du);
#ifdef NDEBUG
    }
#endif
    reg_f3d<T>::AllocateCurrentInputImage(level);

    this->f3d2AppFreeStep=false;
//    this->negatedControlPointGrid=nifti_copy_nim_info(this->controlPointGrid);
//    this->negatedControlPointGrid->data=(void *)malloc(this->negatedControlPointGrid->nvox*
//                                                       this->negatedControlPointGrid->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::ClearCurrentInputImage()
{
    reg_f3d<T>::ClearCurrentInputImage();
//    if(this->negatedControlPointGrid!=NULL){
//        nifti_image_free(this->negatedControlPointGrid);
//        this->negatedControlPointGrid=NULL;
//    }
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
        fprintf(stderr,"[NiftyReg ERROR] reg_f3d2::GetWarpedImage()\n");
        fprintf(stderr," * The reference, floating and control point grid images have to be defined\n");
    }

    this->currentReference = this->inputReference;
    this->currentFloating = this->inputFloating;

    reg_f3d2<T>::AllocateWarped();
    reg_f3d2<T>::AllocateDeformationField();

    reg_f3d2<T>::WarpFloatingImage(3); // cubic spline interpolation

    reg_f3d2<T>::ClearDeformationField();

    nifti_image *resultImage = nifti_copy_nim_info(this->warped);
    resultImage->data=(void *)malloc(resultImage->nvox*resultImage->nbyper);
    memcpy(resultImage->data, this->warped->data, resultImage->nvox*resultImage->nbyper);

    reg_f3d2<T>::ClearWarped();
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d2<T>::CheckStoppingCriteria(bool convergence)
{
    if(convergence){
        if(this->f3d2AppFreeStep==false){
            this->f3d2AppFreeStep=true;
#ifdef NDEBUG
            if(this->verbose)
#endif
            printf("[%s] Squaring is now performed without approximation\n",
                   this->executableName);
        }
        else return 1;
    }
    else{
        if( this->currentIteration>=(this->maxiterationNumber-(float)this->maxiterationNumber*0.1f) ){
            if(this->f3d2AppFreeStep==false){
                this->f3d2AppFreeStep=true;
#ifdef NDEBUG
                if(this->verbose)
#endif
                printf("[%s] Squaring is now performed without approximation\n",
                       this->executableName);
            }
        }
        else if(this->currentIteration>=this->maxiterationNumber) return 1;
    }

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
