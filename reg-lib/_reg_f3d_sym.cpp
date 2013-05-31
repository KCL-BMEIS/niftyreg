/*
 *  _reg_f3_symd.cpp
 *
 *
 *  Created by Marc Modat on 10/11/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_SYM_CPP
#define _REG_F3D_SYM_CPP

#include "_reg_f3d_sym.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_sym<T>::reg_f3d_sym(int refTimePoint,int floTimePoint)
    :reg_f3d<T>::reg_f3d(refTimePoint,floTimePoint)
{
    this->executableName=(char *)"NiftyReg F3D SYM";

    this->backwardControlPointGrid=NULL;
    this->backwardWarped=NULL;
    this->backwardWarpedGradientImage=NULL;
    this->backwardDeformationFieldImage=NULL;
    this->backwardVoxelBasedMeasureGradientImage=NULL;
    this->backwardTransformationGradient=NULL;

    this->backwardProbaJointHistogram=NULL;
    this->backwardLogJointHistogram=NULL;

    this->floatingMaskImage=NULL;
    this->currentFloatingMask=NULL;
    this->floatingMaskPyramid=NULL;
    this->backwardActiveVoxelNumber=NULL;

    this->inverseConsistencyWeight=0.1;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_sym constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_sym<T>::~reg_f3d_sym()
{
    if(this->backwardControlPointGrid!=NULL){
        nifti_image_free(this->backwardControlPointGrid);
        this->backwardControlPointGrid=NULL;
    }

    if(this->floatingMaskPyramid!=NULL){
        if(this->usePyramid){
            for(unsigned int i=0;i<this->levelToPerform;i++){
                if(this->floatingMaskPyramid[i]!=NULL){
                    free(this->floatingMaskPyramid[i]);
                    this->floatingMaskPyramid[i]=NULL;
                }
            }
        }
        else{
            if(this->floatingMaskPyramid[0]!=NULL){
                free(this->floatingMaskPyramid[0]);
                this->floatingMaskPyramid[0]=NULL;
            }
        }
        free(this->floatingMaskPyramid);
        floatingMaskPyramid=NULL;
    }

    if(this->backwardActiveVoxelNumber!=NULL){
        free(this->backwardActiveVoxelNumber);
        this->backwardActiveVoxelNumber=NULL;
    }

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_sym destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::SetFloatingMask(nifti_image *m)
{
    this->floatingMaskImage = m;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::SetInverseConsistencyWeight(T w)
{
    this->inverseConsistencyWeight = w;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_sym<T>::InitialiseCurrentLevel()
{
    T maxStepSize=reg_f3d<T>::InitialiseCurrentLevel();
    if(this->currentLevel!=0)
        reg_spline_refineControlPointGrid(this->currentFloating, this->backwardControlPointGrid);

    if(this->usePyramid){
        this->currentFloatingMask = this->floatingMaskPyramid[this->currentLevel];
    }
    else{
        this->currentFloatingMask = this->floatingMaskPyramid[0];
    }

    maxStepSize = this->currentFloating->dx>maxStepSize?this->currentFloating->dx:maxStepSize;
    maxStepSize = this->currentFloating->dy>maxStepSize?this->currentFloating->dy:maxStepSize;
    if(this->currentReference->ndim>2)
        maxStepSize = (this->currentFloating->dz>maxStepSize)?this->currentFloating->dz:maxStepSize;

    return maxStepSize;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearCurrentInputImage()
{
    reg_f3d<T>::ClearCurrentInputImage();
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateWarped()
{
    this->ClearWarped();

    reg_f3d<T>::AllocateWarped();
    if(this->currentFloating==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The floating image is not defined\n");
        reg_exit(1);
    }
    this->backwardWarped = nifti_copy_nim_info(this->currentFloating);
    this->backwardWarped->dim[0]=this->backwardWarped->ndim=this->currentReference->ndim;
    this->backwardWarped->dim[4]=this->backwardWarped->nt=this->currentReference->nt;
    this->backwardWarped->pixdim[4]=this->backwardWarped->dt=1.0;
    this->backwardWarped->nvox =
            (size_t)this->backwardWarped->nx *
            (size_t)this->backwardWarped->ny *
            (size_t)this->backwardWarped->nz *
            (size_t)this->backwardWarped->nt;
    this->backwardWarped->datatype = this->currentReference->datatype;
    this->backwardWarped->nbyper = this->currentReference->nbyper;
    this->backwardWarped->data = (void *)calloc(this->backwardWarped->nvox, this->backwardWarped->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearWarped()
{
    reg_f3d<T>::ClearWarped();
    if(this->backwardWarped!=NULL){
        nifti_image_free(this->backwardWarped);
        this->backwardWarped=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateDeformationField()
{
    this->ClearDeformationField();

    reg_f3d<T>::AllocateDeformationField();
    if(this->currentFloating==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The floating image is not defined\n");
        reg_exit(1);
    }
    if(this->backwardControlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        reg_exit(1);
    }
    this->backwardDeformationFieldImage = nifti_copy_nim_info(this->currentFloating);
    this->backwardDeformationFieldImage->dim[0]=this->backwardDeformationFieldImage->ndim=5;
    this->backwardDeformationFieldImage->dim[1]=this->backwardDeformationFieldImage->nx=this->currentFloating->nx;
    this->backwardDeformationFieldImage->dim[2]=this->backwardDeformationFieldImage->ny=this->currentFloating->ny;
    this->backwardDeformationFieldImage->dim[3]=this->backwardDeformationFieldImage->nz=this->currentFloating->nz;
    this->backwardDeformationFieldImage->dim[4]=this->backwardDeformationFieldImage->nt=1;
    this->backwardDeformationFieldImage->pixdim[4]=this->backwardDeformationFieldImage->dt=1.0;
    if(this->currentFloating->nz==1)
        this->backwardDeformationFieldImage->dim[5]=this->backwardDeformationFieldImage->nu=2;
    else this->backwardDeformationFieldImage->dim[5]=this->backwardDeformationFieldImage->nu=3;
    this->backwardDeformationFieldImage->pixdim[5]=this->backwardDeformationFieldImage->du=1.0;
    this->backwardDeformationFieldImage->dim[6]=this->backwardDeformationFieldImage->nv=1;
    this->backwardDeformationFieldImage->pixdim[6]=this->backwardDeformationFieldImage->dv=1.0;
    this->backwardDeformationFieldImage->dim[7]=this->backwardDeformationFieldImage->nw=1;
    this->backwardDeformationFieldImage->pixdim[7]=this->backwardDeformationFieldImage->dw=1.0;
    this->backwardDeformationFieldImage->nvox =
            (size_t)this->backwardDeformationFieldImage->nx *
            (size_t)this->backwardDeformationFieldImage->ny *
            (size_t)this->backwardDeformationFieldImage->nz *
            (size_t)this->backwardDeformationFieldImage->nt *
            (size_t)this->backwardDeformationFieldImage->nu;
    this->backwardDeformationFieldImage->nbyper = this->backwardControlPointGrid->nbyper;
    this->backwardDeformationFieldImage->datatype = this->backwardControlPointGrid->datatype;
    this->backwardDeformationFieldImage->data = (void *)calloc(this->backwardDeformationFieldImage->nvox, this->backwardDeformationFieldImage->nbyper);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearDeformationField()
{
    reg_f3d<T>::ClearDeformationField();
    if(this->backwardDeformationFieldImage!=NULL){
        nifti_image_free(this->backwardDeformationFieldImage);
        this->backwardDeformationFieldImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateWarpedGradient()
{
    this->ClearWarpedGradient();

    reg_f3d<T>::AllocateWarpedGradient();
    if(this->backwardDeformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        reg_exit(1);
    }
    this->backwardWarpedGradientImage = nifti_copy_nim_info(this->backwardDeformationFieldImage);
    this->backwardWarpedGradientImage->dim[0]=this->backwardWarpedGradientImage->ndim=5;
    this->backwardWarpedGradientImage->nt = this->backwardWarpedGradientImage->dim[4] = this->currentReference->nt;
    this->backwardWarpedGradientImage->nvox =
            (size_t)this->backwardWarpedGradientImage->nx *
            (size_t)this->backwardWarpedGradientImage->ny *
            (size_t)this->backwardWarpedGradientImage->nz *
            (size_t)this->backwardWarpedGradientImage->nt *
            (size_t)this->backwardWarpedGradientImage->nu;
    this->backwardWarpedGradientImage->data = (void *)calloc(this->backwardWarpedGradientImage->nvox, this->backwardWarpedGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearWarpedGradient()
{
    reg_f3d<T>::ClearWarpedGradient();
    if(this->backwardWarpedGradientImage!=NULL){
        nifti_image_free(this->backwardWarpedGradientImage);
        this->backwardWarpedGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateVoxelBasedMeasureGradient()
{
    this->ClearVoxelBasedMeasureGradient();

    reg_f3d<T>::AllocateVoxelBasedMeasureGradient();
    if(this->backwardDeformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        reg_exit(1);
    }
    this->backwardVoxelBasedMeasureGradientImage = nifti_copy_nim_info(this->backwardDeformationFieldImage);
    this->backwardVoxelBasedMeasureGradientImage->data =
            (void *)calloc(this->backwardVoxelBasedMeasureGradientImage->nvox,
                           this->backwardVoxelBasedMeasureGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearVoxelBasedMeasureGradient()
{
    reg_f3d<T>::ClearVoxelBasedMeasureGradient();
    if(this->backwardVoxelBasedMeasureGradientImage!=NULL){
        nifti_image_free(this->backwardVoxelBasedMeasureGradientImage);
        this->backwardVoxelBasedMeasureGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateTransformationGradient()
{
    this->ClearTransformationGradient();

    reg_f3d<T>::AllocateTransformationGradient();
    if(this->backwardControlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        reg_exit(1);
    }
    this->backwardTransformationGradient = nifti_copy_nim_info(this->backwardControlPointGrid);
    this->backwardTransformationGradient->data =
            (void *)calloc(this->backwardTransformationGradient->nvox,
                           this->backwardTransformationGradient->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearTransformationGradient()
{
    reg_f3d<T>::ClearTransformationGradient();
    if(this->backwardTransformationGradient!=NULL){
        nifti_image_free(this->backwardTransformationGradient);
        this->backwardTransformationGradient=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateJointHistogram()
{
    this->ClearJointHistogram();

    reg_f3d<T>::AllocateJointHistogram();
    this->backwardProbaJointHistogram = (double *)malloc(this->totalBinNumber*sizeof(double));
    this->backwardLogJointHistogram = (double *)malloc(this->totalBinNumber*sizeof(double));
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearJointHistogram()
{
    reg_f3d<T>::ClearJointHistogram();
    if(this->backwardProbaJointHistogram!=NULL){
        free(this->backwardProbaJointHistogram);
        this->backwardProbaJointHistogram=NULL;
    }
    if(this->backwardLogJointHistogram!=NULL){
        free(this->backwardLogJointHistogram);
        this->backwardLogJointHistogram=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::CheckParameters()
{

    reg_f3d<T>::CheckParameters();

    if(this->affineTransformation!=NULL){
        fprintf(stderr, "[NiftyReg F3D_SYM ERROR] The inverse consistency parametrisation does not handle affine input\n");
        fprintf(stderr, "[NiftyReg F3D_SYM ERROR] Please update your floating image sform using reg_transform\n");
        fprintf(stderr, "[NiftyReg F3D_SYM ERROR] and use the updated floating image as an input\n.");
        reg_exit(1);
    }

    // CHECK THE FLOATING MASK DIMENSION IF IT IS DEFINED
    if(this->floatingMaskImage!=NULL){
        if(this->inputFloating->nx != this->floatingMaskImage->nx ||
                this->inputFloating->ny != this->floatingMaskImage->ny ||
                this->inputFloating->nz != this->floatingMaskImage->nz){
            fprintf(stderr,"* The floating mask image has different x, y or z dimension than the floating image.\n");
            reg_exit(1);
        }
    }

    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
    T penaltySum=
            this->bendingEnergyWeight
            +this->linearEnergyWeight0
            +this->linearEnergyWeight1
            +this->L2NormWeight
            +this->jacobianLogWeight
            +this->inverseConsistencyWeight;
    if(penaltySum>=1){
        this->similarityWeight=0;
        this->bendingEnergyWeight /= penaltySum;
        this->linearEnergyWeight0 /= penaltySum;
        this->linearEnergyWeight1 /= penaltySum;
        this->L2NormWeight /= penaltySum;
        this->jacobianLogWeight /= penaltySum;
        this->inverseConsistencyWeight /= penaltySum;
    }
    else this->similarityWeight=1.0 - penaltySum;

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::Initisalise()
{
    reg_f3d<T>::Initisalise();

    /* allocate the backward control point image */

    /* Convert the spacing from voxel to mm if necessary */
    float spacingInMillimeter[3]={this->spacing[0],this->spacing[1],this->spacing[2]};
    if(this->usePyramid){
        if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->floatingPyramid[this->levelToPerform-1]->dx;
        if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->floatingPyramid[this->levelToPerform-1]->dy;
        if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->floatingPyramid[this->levelToPerform-1]->dz;
    }
    else{
        if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->floatingPyramid[0]->dx;
        if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->floatingPyramid[0]->dy;
        if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->floatingPyramid[0]->dz;
    }

    // Define the spacing for the first level
    float gridSpacing[3];
    gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(this->levelToPerform-1));
    gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(this->levelToPerform-1));
    gridSpacing[2] = 1.0f;
    if(this->floatingPyramid[0]->nz>1)
        gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(this->levelToPerform-1));

    // Create and allocate the control point image
    reg_createControlPointGrid<T>(&this->backwardControlPointGrid,
                                  this->floatingPyramid[0],
                                  gridSpacing);

    // the backward control point is initialised using an affine transformation
    mat44 matrixAffine;
    matrixAffine.m[0][0]=1.f;
    matrixAffine.m[0][1]=0.f;
    matrixAffine.m[0][2]=0.f;
    matrixAffine.m[0][3]=0.f;
    matrixAffine.m[1][0]=0.f;
    matrixAffine.m[1][1]=1.f;
    matrixAffine.m[1][2]=0.f;
    matrixAffine.m[1][3]=0.f;
    matrixAffine.m[2][0]=0.f;
    matrixAffine.m[2][1]=0.f;
    matrixAffine.m[2][2]=1.f;
    matrixAffine.m[2][3]=0.f;
    matrixAffine.m[3][0]=0.f;
    matrixAffine.m[3][1]=0.f;
    matrixAffine.m[3][2]=0.f;
    matrixAffine.m[3][3]=1.f;
    if(reg_spline_initialiseControlPointGridWithAffine(&matrixAffine, this->controlPointGrid))
        reg_exit(1);
    if(reg_spline_initialiseControlPointGridWithAffine(&matrixAffine, this->backwardControlPointGrid))
        reg_exit(1);

    // Set the floating mask image pyramid
    if(this->usePyramid){
        this->floatingMaskPyramid = (int **)malloc(this->levelToPerform*sizeof(int *));
        this->backwardActiveVoxelNumber= (int *)malloc(this->levelToPerform*sizeof(int));
    }
    else{
        this->floatingMaskPyramid = (int **)malloc(sizeof(int *));
        this->backwardActiveVoxelNumber= (int *)malloc(sizeof(int));
    }

    if(this->usePyramid){
        if (this->floatingMaskImage!=NULL)
            reg_createMaskPyramid<T>(this->floatingMaskImage,
                                     this->floatingMaskPyramid,
                                     this->levelNumber,
                                     this->levelToPerform,
                                     this->backwardActiveVoxelNumber);
        else{
            for(unsigned int l=0;l<this->levelToPerform;++l){
                this->backwardActiveVoxelNumber[l]=this->floatingPyramid[l]->nx*this->floatingPyramid[l]->ny*this->floatingPyramid[l]->nz;
                this->floatingMaskPyramid[l]=(int *)calloc(backwardActiveVoxelNumber[l],sizeof(int));
            }
        }
    }
    else{ // no pyramid
        if (this->floatingMaskImage!=NULL)
            reg_createMaskPyramid<T>(this->floatingMaskImage, this->floatingMaskPyramid, 1, 1, this->backwardActiveVoxelNumber);
        else{
            this->backwardActiveVoxelNumber[0]=this->floatingPyramid[0]->nx*this->floatingPyramid[0]->ny*this->floatingPyramid[0]->nz;
            this->floatingMaskPyramid[0]=(int *)calloc(backwardActiveVoxelNumber[0],sizeof(int));
        }
    }

#ifdef NDEBUG
    if(this->verbose){
#endif
    printf("[%s]\n", this->executableName);
    printf("[%s] Inverse consistency error penalty term weight: %g\n",
           this->executableName, this->inverseConsistencyWeight);
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_sym::Initialise_f3d() done\n");
#endif

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetDeformationField()
{
    reg_f3d<T>::GetDeformationField();
    if(this->backwardDeformationFieldImage!=NULL)
        reg_spline_getDeformationField(this->backwardControlPointGrid,
                                       this->backwardDeformationFieldImage,
                                       this->currentFloatingMask,
                                       false, //composition
                                       true // bspline
                                       );
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::WarpFloatingImage(int inter)
{
    // Compute the deformation fields
    this->GetDeformationField();

    // Resample the floating image
    reg_resampleImage(this->currentFloating, // input image
                      this->warped, // warped input image
                      this->deformationFieldImage, // deformation field
                      this->currentMask, // mask
                      inter, // interpolation
                      this->warpedPaddingValue); // padding value
    // Resample the reference image
    reg_resampleImage(this->currentReference, // input image
                      this->backwardWarped, // warped input image
                      this->backwardDeformationFieldImage, // deformation field
                      this->currentFloatingMask, // mask
                      inter, // interpolation type
                      this->warpedPaddingValue); // padding value
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeSimilarityMeasure()
{

    double measure=0.;
    if(this->useSSD){
        // forward
        measure = -reg_getSSD(this->currentReference,
                              this->warped,
                              NULL,
                              this->currentMask);
        // backward
        measure+= -reg_getSSD(this->currentFloating,
                              this->backwardWarped,
                              NULL,
                              this->currentFloatingMask);
        if(this->usePyramid)
            measure /= this->maxSSD[this->currentLevel];
        else measure /= this->maxSSD[0];
    }
    else if(this->useKLD){
        // forward
        measure = -reg_getKLDivergence(this->currentReference,
                                       this->warped,
                                       NULL,
                                       this->currentMask);
        // backward
        measure+= -reg_getKLDivergence(this->currentFloating,
                                       this->backwardWarped,
                                       NULL,
                                       this->currentFloatingMask);
    }
    else{
        // forward
        reg_getEntropies(this->currentReference,
                         this->warped,
                         this->referenceBinNumber,
                         this->floatingBinNumber,
                         this->probaJointHistogram,
                         this->logJointHistogram,
                         this->entropies,
                         this->currentMask,
                         this->approxParzenWindow);
        // backward
        reg_getEntropies(this->currentFloating,
                         this->backwardWarped,
                         this->floatingBinNumber,
                         this->referenceBinNumber,
                         this->backwardProbaJointHistogram,
                         this->backwardLogJointHistogram,
                         this->backwardEntropies,
                         this->currentFloatingMask,
                         this->approxParzenWindow);
        // overall measure
        measure = (this->entropies[0]+this->entropies[1])/this->entropies[2] +
                  (this->backwardEntropies[0]+this->backwardEntropies[1])/this->backwardEntropies[2];
//        printf("NMI A: ( %f + %f )/ %f = %f\n",
//               this->entropies[0], this->entropies[1], this->entropies[2],
//               (this->entropies[0]+this->entropies[1])/this->entropies[2]);
//        printf("NMI B: ( %f + %f )/ %f = %f\n",
//               this->backwardEntropies[0], this->backwardEntropies[1], this->backwardEntropies[2],
//               (this->backwardEntropies[0]+this->backwardEntropies[1])/this->backwardEntropies[2]);
    }
    return double(this->similarityWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
    if (this->jacobianLogWeight<=0) return 0.;

    double forwardPenaltyTerm=reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(type);

    double backwardPenaltyTerm=0.;

    if(type==2){
        backwardPenaltyTerm = reg_spline_getJacobianPenaltyTerm(this->backwardControlPointGrid,
                                                                this->currentFloating,
                                                                false);
    }
    else{
        backwardPenaltyTerm = reg_spline_getJacobianPenaltyTerm(this->backwardControlPointGrid,
                                                                this->currentFloating,
                                                                this->jacobianLogApproximation);
    }
    unsigned int maxit=5;
    if(type>0) maxit=20;
    unsigned int it=0;
    while(backwardPenaltyTerm!=backwardPenaltyTerm && it<maxit){
        if(type==2){
            backwardPenaltyTerm = reg_spline_correctFolding(this->backwardControlPointGrid,
                                               this->currentFloating,
                                               false);
        }
        else{
            backwardPenaltyTerm = reg_spline_correctFolding(this->backwardControlPointGrid,
                                               this->currentFloating,
                                               this->jacobianLogApproximation);
        }
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Folding correction - Backward transformation\n");
#endif
        it++;
    }
    if(type>0 && it>0){
        if(backwardPenaltyTerm!=backwardPenaltyTerm){
            this->optimiser->RestoreBestDOF();
#ifndef NDEBUG
            fprintf(stderr, "[NiftyReg ERROR] The backward transformation folding correction scheme failed\n");
#endif
        }
        else{
#ifdef NDEBUG
            if(this->verbose){
#endif
                printf("[%s] Backward transformation folding correction, %i step(s)\n", this->executableName, it);
#ifdef NDEBUG
            }
#endif
        }
    }
    backwardPenaltyTerm *= (double)this->jacobianLogWeight;

    return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeBendingEnergyPenaltyTerm()
{
    if (this->bendingEnergyWeight<=0) return 0.;

    double forwardPenaltyTerm=reg_f3d<T>::ComputeBendingEnergyPenaltyTerm();

    double value = reg_spline_approxBendingEnergy(this->backwardControlPointGrid);
    return forwardPenaltyTerm + this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeLinearEnergyPenaltyTerm()
{
    if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return 0.;

    double forwardPenaltyTerm=reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();

    double values_le[2]={0.,0.};
    reg_spline_linearEnergy(this->backwardControlPointGrid, values_le);

    double backwardPenaltyTerm = this->linearEnergyWeight0*values_le[0] +
                                 this->linearEnergyWeight1*values_le[1];

    return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeL2NormDispPenaltyTerm()
{
    if(this->L2NormWeight<=0) return 0.;

    // Compute the L2 norm penalty term along the forward direction
    double forwardPenaltyTerm=reg_f3d<T>::ComputeL2NormDispPenaltyTerm();

    // Compute the L2 norm penalty term along the backward direction
    double backwardPenaltyTerm = (double)this->L2NormWeight *
            reg_spline_L2norm_displacement(this->backwardControlPointGrid);

    // Return the sum of the forward and backward squared L2 norm of the displacement
    return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetVoxelBasedGradient()
{
    // The intensity gradient is first computed - floating warped into reference
    reg_getImageGradient(this->currentFloating,
                         this->warpedGradientImage,
                         this->deformationFieldImage,
                         this->currentMask,
                         this->interpolation,
                         this->warpedPaddingValue);

    // The intensity gradient is first computed - reference warped into floating
    reg_getImageGradient(this->currentReference,
                         this->backwardWarpedGradientImage,
                         this->backwardDeformationFieldImage,
                         this->currentFloatingMask,
                         this->interpolation,
                         this->warpedPaddingValue);

    if(this->useSSD){
        T localMaxSSD=this->maxSSD[0];
        if(this->usePyramid)
            localMaxSSD=this->maxSSD[this->currentLevel];
        // Compute the voxel based SSD gradient - forward
        reg_getVoxelBasedSSDGradient(this->currentReference,
                                     this->warped,
                                     this->warpedGradientImage,
                                     this->voxelBasedMeasureGradientImage,
                                     NULL,
                                     localMaxSSD,
                                     this->currentMask
                                     );
        // Compute the voxel based SSD gradient - backward
        reg_getVoxelBasedSSDGradient(this->currentFloating,
                                     this->backwardWarped,
                                     this->backwardWarpedGradientImage,
                                     this->backwardVoxelBasedMeasureGradientImage,
                                     NULL,
                                     localMaxSSD,
                                     this->currentFloatingMask
                                     );
    }
    else if(this->useKLD){
        // Compute the voxel based KL divergence gradient - forward
        reg_getKLDivergenceVoxelBasedGradient(this->currentReference,
                                              this->warped,
                                              this->warpedGradientImage,
                                              this->voxelBasedMeasureGradientImage,
                                              NULL,
                                              this->currentMask
                                              );
        // Compute the voxel based KL divergence gradient - backward
        reg_getKLDivergenceVoxelBasedGradient(this->currentFloating,
                                              this->backwardWarped,
                                              this->backwardWarpedGradientImage,
                                              this->backwardVoxelBasedMeasureGradientImage,
                                              NULL,
                                              this->currentFloatingMask
                                              );
    }
    else{
        // Compute the voxel based NMI gradient - forward
        reg_getVoxelBasedNMIGradientUsingPW(this->currentReference,
                                            this->warped,
                                            this->warpedGradientImage,
                                            this->referenceBinNumber,
                                            this->floatingBinNumber,
                                            this->logJointHistogram,
                                            this->entropies,
                                            this->voxelBasedMeasureGradientImage,
                                            this->currentMask,
                                            this->approxParzenWindow);
        // Compute the voxel based NMI gradient - backward
        reg_getVoxelBasedNMIGradientUsingPW(this->currentFloating,
                                            this->backwardWarped,
                                            this->backwardWarpedGradientImage,
                                            this->floatingBinNumber,
                                            this->referenceBinNumber,
                                            this->backwardLogJointHistogram,
                                            this->backwardEntropies,
                                            this->backwardVoxelBasedMeasureGradientImage,
                                            this->currentFloatingMask,
                                            this->approxParzenWindow);
    }

    return;
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetSimilarityMeasureGradient()
{
    reg_f3d<T>::GetSimilarityMeasureGradient();

    // The voxel based NMI gradient is convolved with a spline kernel
    float spacingVoxel[3]={
        this->backwardControlPointGrid->dx/this->currentFloating->dx,
        this->backwardControlPointGrid->dy/this->currentFloating->dy,
        this->backwardControlPointGrid->dz/this->currentFloating->dz};
    reg_tools_CubicSplineKernelConvolution(this->backwardVoxelBasedMeasureGradientImage,
                                           spacingVoxel);

    // The node based NMI gradient is extracted
    reg_voxelCentric2NodeCentric(this->backwardTransformationGradient,
                                 this->backwardVoxelBasedMeasureGradientImage,
                                 this->similarityWeight,
                                 false);

    /* The gradient is converted from voxel space to real space */
    mat44 *referenceMatrix_xyz=NULL;
	size_t controlPointNumber=
			(size_t)this->backwardControlPointGrid->nx *
            this->backwardControlPointGrid->ny *
			this->backwardControlPointGrid->nz;
#ifdef _WIN32
	int  i;
#else
	size_t  i;
#endif
    if(this->currentReference->sform_code>0)
        referenceMatrix_xyz = &(this->currentReference->sto_xyz);
    else referenceMatrix_xyz = &(this->currentReference->qto_xyz);
    if(this->currentFloating->nz==1){
        T *gradientValuesX = static_cast<T *>(this->backwardTransformationGradient->data);
        T *gradientValuesY = &gradientValuesX[controlPointNumber];
        T newGradientValueX, newGradientValueY;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gradientValuesX, gradientValuesY, referenceMatrix_xyz, controlPointNumber) \
    private(newGradientValueX, newGradientValueY, i)
#endif
        for(i=0; i<controlPointNumber; i++){
            newGradientValueX =
                    gradientValuesX[i] * referenceMatrix_xyz->m[0][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[0][1];
            newGradientValueY =
                    gradientValuesX[i] * referenceMatrix_xyz->m[1][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[1][1];
            gradientValuesX[i] = newGradientValueX;
            gradientValuesY[i] = newGradientValueY;
        }
    }
    else{
        T *gradientValuesX = static_cast<T *>(this->backwardTransformationGradient->data);
        T *gradientValuesY = &gradientValuesX[controlPointNumber];
        T *gradientValuesZ = &gradientValuesY[controlPointNumber];
        T newGradientValueX, newGradientValueY, newGradientValueZ;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gradientValuesX, gradientValuesY, gradientValuesZ, referenceMatrix_xyz, controlPointNumber) \
    private(newGradientValueX, newGradientValueY, newGradientValueZ, i)
#endif
        for(i=0; i<controlPointNumber; i++){

            newGradientValueX =
                    gradientValuesX[i] * referenceMatrix_xyz->m[0][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[0][1] +
                    gradientValuesZ[i] * referenceMatrix_xyz->m[0][2];
            newGradientValueY =
                    gradientValuesX[i] * referenceMatrix_xyz->m[1][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[1][1] +
                    gradientValuesZ[i] * referenceMatrix_xyz->m[1][2];
            newGradientValueZ =
                    gradientValuesX[i] * referenceMatrix_xyz->m[2][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[2][1] +
                    gradientValuesZ[i] * referenceMatrix_xyz->m[2][2];
            gradientValuesX[i] = newGradientValueX;
            gradientValuesY[i] = newGradientValueY;
            gradientValuesZ[i] = newGradientValueZ;
        }
    }

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetJacobianBasedGradient()
{
    if(this->jacobianLogWeight<=0) return;

    reg_f3d<T>::GetJacobianBasedGradient();

    reg_spline_getJacobianPenaltyTermGradient(this->backwardControlPointGrid,
                                              this->currentFloating,
                                              this->backwardTransformationGradient,
                                              this->jacobianLogWeight,
                                              this->jacobianLogApproximation);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetBendingEnergyGradient()
{
    if(this->bendingEnergyWeight<=0) return;

    reg_f3d<T>::GetBendingEnergyGradient();
    reg_spline_approxBendingEnergyGradient(this->backwardControlPointGrid,
                                           this->backwardTransformationGradient,
                                           this->bendingEnergyWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetLinearEnergyGradient()
{
    if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return;

    reg_f3d<T>::GetLinearEnergyGradient();

    reg_spline_linearEnergyGradient(this->backwardControlPointGrid,
                                     this->currentFloating,
                                     this->transformationGradient,
                                     this->linearEnergyWeight0,
                                     this->linearEnergyWeight1);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetL2NormDispGradient()
{
    if(this->L2NormWeight<=0) return;

    reg_f3d<T>::GetL2NormDispGradient();

    reg_spline_L2norm_dispGradient(this->backwardControlPointGrid,
                                    this->currentFloating,
                                    this->backwardTransformationGradient,
                                    this->L2NormWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SetGradientImageToZero()
{
    reg_f3d<T>::SetGradientImageToZero();

    T* nodeGradPtr = static_cast<T *>(this->backwardTransformationGradient->data);
    for(size_t i=0; i<this->backwardTransformationGradient->nvox; ++i)
        *nodeGradPtr++=0;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SmoothGradient()
{
    if(this->gradientSmoothingSigma!=0){
        reg_f3d<T>::SmoothGradient();
        // The gradient is smoothed using a Gaussian kernel if it is required
        reg_gaussianSmoothing<T>(this->backwardTransformationGradient,
                                 fabs(this->gradientSmoothingSigma),
                                 NULL);
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetApproximatedGradient()
{
    reg_f3d<T>::GetApproximatedGradient();

    // Loop over every control points
    T *gridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
    T *gradPtr = static_cast<T *>(this->backwardTransformationGradient->data);
    T eps = this->currentFloating->dx/1000.f;
    for(size_t i=0; i<this->backwardControlPointGrid->nvox;i++)
    {
        T currentValue = this->optimiser->GetBestDOF_b()[i];
        gridPtr[i] = currentValue+eps;
        double valPlus = this->GetObjectiveFunctionValue();
        gridPtr[i] = currentValue-eps;
        double valMinus = this->GetObjectiveFunctionValue();
        gridPtr[i] = currentValue;
        gradPtr[i] = -(T)((valPlus - valMinus ) / (2.0*eps));
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_sym<T>::NormaliseGradient()
{
    // The forward gradient max length is computed
    T forwardMaxValue = reg_f3d<T>::NormaliseGradient();

    // The backward gradient max length is computed
    T maxGradValue=0;
    size_t voxNumber = this->backwardTransformationGradient->nx *
            this->backwardTransformationGradient->ny *
            this->backwardTransformationGradient->nz;
    T *bckPtrX = static_cast<T *>(this->backwardTransformationGradient->data);
    T *bckPtrY = &bckPtrX[voxNumber];
    if(this->backwardTransformationGradient->nz>1){
        T *bckPtrZ = &bckPtrY[voxNumber];
		for(size_t i=0; i<voxNumber; i++){
            T valX=0,valY=0,valZ=0;
            if(this->optimiseX==true)
                valX = *bckPtrX++;
            if(this->optimiseY==true)
                valY = *bckPtrY++;
            if(this->optimiseZ==true)
                valZ = *bckPtrZ++;
            T length = (T)(sqrt(valX*valX + valY*valY + valZ*valZ));
            maxGradValue = (length>maxGradValue)?length:maxGradValue;
        }
    }
    else{
		for(size_t i=0; i<voxNumber; i++){
            T valX=0,valY=0;
            if(this->optimiseX==true)
                valX = *bckPtrX++;
            if(this->optimiseY==true)
                valY = *bckPtrY++;
            T length = (T)(sqrt(valX*valX + valY*valY));
            maxGradValue = (length>maxGradValue)?length:maxGradValue;
        }
    }

    // The largest value between the forward and backward gradient is kept
    maxGradValue = maxGradValue>forwardMaxValue?maxGradValue:forwardMaxValue;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Objective function gradient maximal length: %g\n", maxGradValue);
#endif

    // The forward gradient is normalised
    T *forPtrX = static_cast<T *>(this->transformationGradient->data);
    for(size_t i=0;i<this->transformationGradient->nvox;++i){
        *forPtrX++ /= maxGradValue;
    }
    // The backward gradient is normalised
    bckPtrX = static_cast<T *>(this->backwardTransformationGradient->data);
    for(size_t i=0;i<this->backwardTransformationGradient->nvox;++i){
        *bckPtrX++ /= maxGradValue;
    }

    // Returns the largest gradient distance
    return maxGradValue;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetObjectiveFunctionGradient()
{
    if(!this->useApproxGradient){
        // Compute the gradient of the similarity measure
        if(this->similarityWeight>0){
            this->WarpFloatingImage(this->interpolation);
            this->ComputeSimilarityMeasure();
            this->GetSimilarityMeasureGradient();
        }
        else{
            this->SetGradientImageToZero();
        }
    }
    this->optimiser->IncrementCurrentIterationNumber();

    // Smooth the gradient if require
    this->SmoothGradient();

    if(!this->useApproxGradient){
        // Compute the penalty term gradients if required
        this->GetBendingEnergyGradient();
        this->GetJacobianBasedGradient();
        this->GetLinearEnergyGradient();
        this->GetL2NormDispGradient();
        this->GetInverseConsistencyGradient();
    }
    else this->GetApproximatedGradient();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::DisplayCurrentLevelParameters()
{
    reg_f3d<T>::DisplayCurrentLevelParameters();
#ifdef NDEBUG
        if(this->verbose){
#endif
            printf("[%s] Current backward control point image\n", this->executableName);
            printf("[%s] \t* image dimension: %i x %i x %i\n", this->executableName,
                   this->backwardControlPointGrid->nx, this->backwardControlPointGrid->ny,
                   this->backwardControlPointGrid->nz);
            printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
                   this->backwardControlPointGrid->dx, this->backwardControlPointGrid->dy,
                   this->backwardControlPointGrid->dz);
#ifdef NDEBUG
        }
#endif

#ifndef NDEBUG

        if(this->backwardControlPointGrid->sform_code>0)
            reg_mat44_disp(&(this->backwardControlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP sform");
        else reg_mat44_disp(&(this->backwardControlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP qform");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetInverseConsistencyErrorField(bool forceAll)
{
    if (this->inverseConsistencyWeight<=0) return;

    if(this->similarityWeight<=0 || forceAll){
        reg_spline_getDeformationField(this->controlPointGrid,
                                       this->deformationFieldImage,
                                       this->currentMask,
                                       false, // composition
                                       true // use B-Spline
                                       );
        reg_spline_getDeformationField(this->backwardControlPointGrid,
                                       this->backwardDeformationFieldImage,
                                       this->currentFloatingMask,
                                       false, // composition
                                       true // use B-Spline
                                       );
    }

    reg_spline_getDeformationField(this->backwardControlPointGrid,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->backwardDeformationFieldImage,
                                   this->currentFloatingMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->deformationFieldImage);
    reg_getDisplacementFromDeformation(this->backwardDeformationFieldImage);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
double reg_f3d_sym<T>::GetInverseConsistencyPenaltyTerm()
{
    if (this->inverseConsistencyWeight<=0) return 0.;

    this->GetInverseConsistencyErrorField(false);

    double ferror=0.;
    size_t voxelNumber=this->deformationFieldImage->nx *
            this->deformationFieldImage->ny *
            this->deformationFieldImage->nz;
    T *dispPtrX=static_cast<T *>(this->deformationFieldImage->data);
    T *dispPtrY=&dispPtrX[voxelNumber];
    if(this->deformationFieldImage->nz>1){
        T *dispPtrZ=&dispPtrY[voxelNumber];
        for(size_t i=0; i<voxelNumber; ++i){
            if(this->currentMask[i]>-1){
                double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]) + reg_pow2(dispPtrZ[i]);
                ferror += dist;
            }
        }
    }
    else{
        for(size_t i=0; i<voxelNumber; ++i){
            if(this->currentMask[i]>-1){
                double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]);
                ferror += dist;
            }
        }
    }

    double berror=0.;
    voxelNumber=this->backwardDeformationFieldImage->nx *
            this->backwardDeformationFieldImage->ny *
            this->backwardDeformationFieldImage->nz;
    dispPtrX=static_cast<T *>(this->backwardDeformationFieldImage->data);
    dispPtrY=&dispPtrX[voxelNumber];
    if(this->backwardDeformationFieldImage->nz>1){
        T *dispPtrZ=&dispPtrY[voxelNumber];
        for(size_t i=0; i<voxelNumber; ++i){
            if(this->currentFloatingMask[i]>-1){
                double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]) + reg_pow2(dispPtrZ[i]);
                berror += dist;
            }
        }
    }
    else{
        for(size_t i=0; i<voxelNumber; ++i){
            if(this->currentFloatingMask[i]>-1){
                double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]);
                berror += dist;
            }
        }
    }
    double error = ferror/double(this->activeVoxelNumber[this->currentLevel])
                 + berror / (double)(this->backwardActiveVoxelNumber[this->currentLevel]);
    return double(this->inverseConsistencyWeight) * error;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetInverseConsistencyGradient()
{
    if(this->inverseConsistencyWeight<=0) return;

    // Note: I simplified the gradient computation in order to include
    // only d(B(F(x)))/d(forwardNode) and d(F(B(x)))/d(backwardNode)
    // I ignored d(F(B(x)))/d(forwardNode) and d(B(F(x)))/d(backwardNode)
    // cause it would only be an approximation since I don't have the
    // real inverses
    this->GetInverseConsistencyErrorField(true);

    // The forward inverse consistency field is masked
    size_t forwardVoxelNumber=
            this->deformationFieldImage->nx *
            this->deformationFieldImage->ny *
            this->deformationFieldImage->nz ;
    T *defPtrX=static_cast<T* >(this->deformationFieldImage->data);
    T *defPtrY=&defPtrX[forwardVoxelNumber];
    T *defPtrZ=&defPtrY[forwardVoxelNumber];
    for(size_t i=0; i<forwardVoxelNumber; ++i){
        if(this->currentMask[i]<0){
            defPtrX[i]=0;
            defPtrY[i]=0;
            if(this->deformationFieldImage->nz>1)
                defPtrZ[i]=0;
        }
    }
    // The backward inverse consistency field is masked
    size_t backwardVoxelNumber =
            this->backwardDeformationFieldImage->nx *
            this->backwardDeformationFieldImage->ny *
            this->backwardDeformationFieldImage->nz ;
    defPtrX=static_cast<T* >(this->backwardDeformationFieldImage->data);
    defPtrY=&defPtrX[backwardVoxelNumber];
    defPtrZ=&defPtrY[backwardVoxelNumber];
    for(size_t i=0; i<backwardVoxelNumber; ++i){
        if(this->currentFloatingMask[i]<0){
            defPtrX[i]=0;
            defPtrY[i]=0;
            if(this->backwardDeformationFieldImage->nz>1)
                defPtrZ[i]=0;
        }
    }

    // We convolve the inverse consistency map with a cubic B-Spline kernel
    float spacingVoxel[3];
    spacingVoxel[0]=this->controlPointGrid->dx/this->currentReference->dx;
    spacingVoxel[1]=this->controlPointGrid->dy/this->currentReference->dy;
    spacingVoxel[2]=this->controlPointGrid->dz/this->currentReference->dz;
    reg_tools_CubicSplineKernelConvolution(this->deformationFieldImage, spacingVoxel);
    // The forward inverse consistency gradient is extracted at the node position
    reg_voxelCentric2NodeCentric(this->transformationGradient,
                                 this->deformationFieldImage, //tempVoxelIC,
                                 2.f * this->inverseConsistencyWeight / (float)(this->activeVoxelNumber[this->currentLevel]),
                                 true); // update?

    // We convolve the inverse consistency map with a cubic B-Spline kernel
    spacingVoxel[0]=this->backwardControlPointGrid->dx/this->currentFloating->dx;
    spacingVoxel[1]=this->backwardControlPointGrid->dy/this->currentFloating->dy;
    spacingVoxel[2]=this->backwardControlPointGrid->dz/this->currentFloating->dz;
    reg_tools_CubicSplineKernelConvolution(this->backwardDeformationFieldImage, spacingVoxel);
    // The backward inverse consistency gradient is extracted at the node position
    reg_voxelCentric2NodeCentric(this->backwardTransformationGradient,
                                 this->backwardDeformationFieldImage, //tempVoxelIC,
                                 2.f * this->inverseConsistencyWeight / (float)(this->backwardActiveVoxelNumber[this->currentLevel]),
                                 true); // update?

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::UpdateParameters(float scale)
{
    // Update first the forward transformation
    reg_f3d<T>::UpdateParameters(scale);

    // Create some pointers to the relevant arrays
    T *currentDOF_b=this->optimiser->GetCurrentDOF_b();
    T *bestDOF_b=this->optimiser->GetBestDOF_b();
    T *gradient_b=this->optimiser->GetGradient_b();

    // Update the control point position
    if(this->optimiser->GetOptimiseX()==true &&
       this->optimiser->GetOptimiseY()==true &&
       this->optimiser->GetOptimiseZ()==true)
    {
        // Update the values for all axis displacement
        for(size_t i=0;i<this->optimiser->GetDOFNumber_b();++i){
            currentDOF_b[i] =bestDOF_b[i] + scale * gradient_b[i];
        }
    }
    else
    {
        size_t voxNumber_b = this->optimiser->GetVoxNumber_b();
        // Update the values for the x-axis displacement
        if(this->optimiser->GetOptimiseX()==true){
            for(size_t i=0;i<voxNumber_b;++i){
                currentDOF_b[i] =bestDOF_b[i] + scale * gradient_b[i];
            }
        }
        // Update the values for the y-axis displacement
        if(this->optimiser->GetOptimiseY()==true){
            T *currentDOFY_b=&currentDOF_b[voxNumber_b];
            T *bestDOFY_b=&bestDOF_b[voxNumber_b];
            T *gradientY_b=&gradient_b[voxNumber_b];
            for(size_t i=0;i<voxNumber_b;++i){
                currentDOFY_b[i] = bestDOFY_b[i] + scale * gradientY_b[i];
            }
        }
        // Update the values for the z-axis displacement
        if(this->optimiser->GetOptimiseZ()==true && this->optimiser->GetNDim()>2){
            T *currentDOFZ_b=&currentDOF_b[2*voxNumber_b];
            T *bestDOFZ_b=&bestDOF_b[2*voxNumber_b];
            T *gradientZ_b=&gradient_b[2*voxNumber_b];
            for(size_t i=0;i<voxNumber_b;++i){
                currentDOFZ_b[i] = bestDOFZ_b[i] + scale * gradientZ_b[i];
            }
        }
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SetOptimiser()
{
    if(this->useConjGradient)
        this->optimiser=new reg_conjugateGradient<T>();
    else this->optimiser=new reg_optimiser<T>();
    this->optimiser->Initialise(this->controlPointGrid->nvox,
                                this->controlPointGrid->nz>1?3:2,
                                this->optimiseX,
                                this->optimiseY,
                                this->optimiseZ,
                                this->maxiterationNumber,
                                0, // currentIterationNumber
                                this,
                                static_cast<T *>(this->controlPointGrid->data),
                                static_cast<T *>(this->transformationGradient->data),
                                this->backwardControlPointGrid->nvox,
                                static_cast<T *>(this->backwardControlPointGrid->data),
                                static_cast<T *>(this->backwardTransformationGradient->data));
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::PrintCurrentObjFunctionValue(T currentSize)
{
    if(!this->verbose) return;

    printf("[%s] [%i] Current objective function: %g",
           this->executableName,
           (int)this->optimiser->GetCurrentIterationNumber(),
           this->optimiser->GetBestObjFunctionValue());
    if(this->useSSD)
        printf(" = (wSSD)%g", this->bestWMeasure);
    else if(this->useKLD)
        printf(" = (wKLD)%g", this->bestWMeasure);
    else printf(" = (wNMI)%g", this->bestWMeasure);
    if(this->bendingEnergyWeight>0)
        printf(" - (wBE)%.2e", this->bestWBE);
    if(this->linearEnergyWeight0>0 || this->linearEnergyWeight1>0)
        printf(" - (wLE)%.2e", this->bestWLE);
    if(this->L2NormWeight>0)
        printf(" - (wL2)%.2e", this->bestWL2);
    if(this->jacobianLogWeight>0)
        printf(" - (wJAC)%.2e", this->bestWJac);
    if(bestIC!=0)
        printf(" - (IC)%.2e", this->bestIC);
    printf(" [+ %g mm]\n", currentSize);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::UpdateBestObjFunctionValue()
{
    reg_f3d<T>::UpdateBestObjFunctionValue();
    this->bestIC=this->currentIC;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::PrintInitialObjFunctionValue()
{
    if(!this->verbose) return;
    reg_f3d<T>::PrintInitialObjFunctionValue();
    printf("[%s] Initial Inverse consistency value: %g\n", this->executableName, this->bestIC);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::GetObjectiveFunctionValue()
{
    this->currentWJac = this->ComputeJacobianBasedPenaltyTerm(1); // 20 iterations

    this->currentWBE = this->ComputeBendingEnergyPenaltyTerm();

    this->currentWLE = this->ComputeLinearEnergyPenaltyTerm();

    this->currentWL2 = this->ComputeL2NormDispPenaltyTerm();

    // Compute initial similarity measure
    this->currentWMeasure = 0.0;
    if(this->similarityWeight>0){
        this->WarpFloatingImage(this->interpolation);
        this->currentWMeasure = this->ComputeSimilarityMeasure();
    }

    // Compute the Inverse consistency penalty term if required
    this->currentIC = this->GetInverseConsistencyPenaltyTerm();

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] (wMeasure) %g | (wBE) %g | (wLE) %g | (wL2) %g | (wJac) %g | (wIC) %g \n",
           this->currentWMeasure,
           this->currentWBE,
           this->currentWLE,
           this->currentWL2,
           this->currentWJac,
           this->currentIC);
#endif

    // Store the global objective function value
    return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWL2 - this->currentWJac - this->currentIC;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image **reg_f3d_sym<T>::GetWarpedImage()
{
    // The initial images are used
    if(this->inputReference==NULL ||
       this->inputFloating==NULL ||
       this->controlPointGrid==NULL ||
       this->backwardControlPointGrid==NULL){
        fprintf(stderr,"[NiftyReg ERROR] reg_f3d_sym::GetWarpedImage()\n");
        fprintf(stderr," * The reference, floating and both control point grid images have to be defined\n");
    }

    reg_f3d_sym<T>::currentReference = this->inputReference;
    reg_f3d_sym<T>::currentFloating = this->inputFloating;
    reg_f3d_sym<T>::currentMask = NULL;
    reg_f3d_sym<T>::currentFloatingMask = NULL;

    reg_f3d_sym<T>::AllocateWarped();
    reg_f3d_sym<T>::AllocateDeformationField();

    reg_f3d_sym<T>::WarpFloatingImage(3); // cubic spline interpolation

    reg_f3d_sym<T>::ClearDeformationField();

    nifti_image **resultImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
    resultImage[0] = nifti_copy_nim_info(this->warped);
    resultImage[0]->cal_min=this->inputFloating->cal_min;
    resultImage[0]->cal_max=this->inputFloating->cal_max;
    resultImage[0]->scl_slope=this->inputFloating->scl_slope;
    resultImage[0]->scl_inter=this->inputFloating->scl_inter;
    resultImage[0]->data=(void *)malloc(resultImage[0]->nvox*resultImage[0]->nbyper);
    memcpy(resultImage[0]->data, this->warped->data, resultImage[0]->nvox*resultImage[0]->nbyper);

    resultImage[1] = nifti_copy_nim_info(this->backwardWarped);
    resultImage[1]->cal_min=this->inputReference->cal_min;
    resultImage[1]->cal_max=this->inputReference->cal_max;
    resultImage[1]->scl_slope=this->inputReference->scl_slope;
    resultImage[1]->scl_inter=this->inputReference->scl_inter;
    resultImage[1]->data=(void *)malloc(resultImage[1]->nvox*resultImage[1]->nbyper);
    memcpy(resultImage[1]->data, this->backwardWarped->data, resultImage[1]->nvox*resultImage[1]->nbyper);

    reg_f3d_sym<T>::ClearWarped();
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image * reg_f3d_sym<T>::GetBackwardControlPointPositionImage()
{
    // Create a control point grid nifti image
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->backwardControlPointGrid);
    // Allocate the new image data array
    returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
    // Copy the final backward control point grid image
    memcpy(returnedControlPointGrid->data, this->backwardControlPointGrid->data,
           returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
    // Return the new control point grid
    return returnedControlPointGrid;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
