/*
 *  _reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_CPP
#define _REG_F3D_CPP

#include "_reg_f3d.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d<T>::reg_f3d(int refTimePoint,int floTimePoint)
{
    this->executableName=(char *)"NiftyReg F3D";
    this->referenceTimePoint=refTimePoint;
    this->floatingTimePoint=floTimePoint;
    this->inputReference=NULL; // pointer to external
    this->inputFloating=NULL; // pointer to external
    this->inputControlPointGrid=NULL; // pointer to external
    this->maskImage=NULL; // pointer to external
    this->affineTransformation=NULL;  // pointer to external
    this->controlPointGrid=NULL;
    this->referenceMask=NULL;
    this->bendingEnergyWeight=0.01;
    this->linearEnergyWeight0=0.;
    this->linearEnergyWeight1=0.;
    this->linearEnergyWeight2=0.;
    this->jacobianLogWeight=0.;
    this->jacobianLogApproximation=true;
    this->maxiterationNumber=300;
    this->referenceSmoothingSigma=0.;
    this->floatingSmoothingSigma=0.;
    this->referenceThresholdUp=new float[this->referenceTimePoint];
    this->referenceThresholdLow=new float[this->referenceTimePoint];
    this->floatingThresholdUp=new float[this->floatingTimePoint];
    this->floatingThresholdLow=new float[this->floatingTimePoint];
    this->referenceBinNumber=new unsigned int[this->referenceTimePoint];
    this->floatingBinNumber=new unsigned int[this->floatingTimePoint];
    for(int i=0; i<this->referenceTimePoint; i++){
        this->referenceThresholdUp[i]=std::numeric_limits<T>::max();
        this->referenceThresholdLow[i]=-std::numeric_limits<T>::max();
        this->referenceBinNumber[i]=64;
    }
    for(int i=0; i<this->floatingTimePoint; i++){
        this->floatingThresholdUp[i]=std::numeric_limits<T>::max();
        this->floatingThresholdLow[i]=-std::numeric_limits<T>::max();
        this->floatingBinNumber[i]=64;
    }
    this->warpedPaddingValue=std::numeric_limits<T>::quiet_NaN();
    this->spacing[0]=-5;
    this->spacing[1]=std::numeric_limits<T>::quiet_NaN();
    this->spacing[2]=std::numeric_limits<T>::quiet_NaN();
    this->levelNumber=3;
    this->levelToPerform=0;
    this->gradientSmoothingSigma=0;
    this->verbose=true;
    this->useSSD=false;
    this->useConjGradient=true;
    this->maxSSD=NULL;
    this->entropies[0]=this->entropies[1]=this->entropies[2]=this->entropies[3]=0.;
    this->currentIteration=0;
    this->usePyramid=true;
    //	this->threadNumber=1;

    this->initialised=false;
    this->referencePyramid=NULL;
    this->floatingPyramid=NULL;
    this->maskPyramid=NULL;
    this->activeVoxelNumber=NULL;
    this->currentReference=NULL;
    this->currentFloating=NULL;
    this->currentMask=NULL;
    this->warped=NULL;
    this->deformationFieldImage=NULL;
    this->warpedGradientImage=NULL;
    this->voxelBasedMeasureGradientImage=NULL;
    this->nodeBasedMeasureGradientImage=NULL;
    this->conjugateG=NULL;
    this->conjugateH=NULL;
    this->bestControlPointPosition=NULL;
    this->probaJointHistogram=NULL;
    this->logJointHistogram=NULL;

    this->interpolation=1;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d<T>::~reg_f3d()
{
    this->ClearWarped();
    this->ClearWarpedGradient();
    this->ClearDeformationField();
    this->ClearBestControlPointArray();
    this->ClearConjugateGradientVariables();
    this->ClearJointHistogram();
    this->ClearNodeBasedMeasureGradient();
    this->ClearVoxelBasedMeasureGradient();
    if(this->controlPointGrid!=NULL){
        nifti_image_free(this->controlPointGrid);
        this->controlPointGrid=NULL;
    }
    if(this->referencePyramid!=NULL){
        if(this->usePyramid){
            for(unsigned int i=0;i<levelToPerform;i++){
                if(referencePyramid[i]!=NULL){
                    nifti_image_free(referencePyramid[i]);
                    referencePyramid[i]=NULL;
                }
            }
        }
        else{
            if(referencePyramid[0]!=NULL){
                nifti_image_free(referencePyramid[0]);
                referencePyramid[0]=NULL;
            }
        }
        free(referencePyramid);
        referencePyramid=NULL;
    }
    if(this->maskPyramid!=NULL){
        if(this->usePyramid){
            for(unsigned int i=0;i<levelToPerform;i++){
                if(this->maskPyramid[i]!=NULL){
                    free(this->maskPyramid[i]);
                    this->maskPyramid[i]=NULL;
                }
            }
        }
        else{
            if(this->maskPyramid[0]!=NULL){
                free(this->maskPyramid[0]);
                this->maskPyramid[0]=NULL;
            }
        }
        free(this->maskPyramid);
        maskPyramid=NULL;
    }
    if(this->floatingPyramid!=NULL){
        if(this->usePyramid){
            for(unsigned int i=0;i<levelToPerform;i++){
                if(floatingPyramid[i]!=NULL){
                    nifti_image_free(floatingPyramid[i]);
                    floatingPyramid[i]=NULL;
                }
            }
        }
        else{
            if(floatingPyramid[0]!=NULL){
                nifti_image_free(floatingPyramid[0]);
                floatingPyramid[0]=NULL;
            }
        }
        free(floatingPyramid);
        floatingPyramid=NULL;
    }
    if(this->activeVoxelNumber!=NULL){
        free(activeVoxelNumber);
        this->activeVoxelNumber=NULL;
    }
    if(this->referenceThresholdUp!=NULL){delete []this->referenceThresholdUp;this->referenceThresholdUp=NULL;}
    if(this->referenceThresholdLow!=NULL){delete []this->referenceThresholdLow;this->referenceThresholdLow=NULL;}
    if(this->referenceBinNumber!=NULL){delete []this->referenceBinNumber;this->referenceBinNumber=NULL;}
    if(this->floatingThresholdUp!=NULL){delete []this->floatingThresholdUp;this->floatingThresholdUp=NULL;}
    if(this->floatingThresholdLow!=NULL){delete []this->floatingThresholdLow;this->floatingThresholdLow=NULL;}
    if(this->floatingBinNumber!=NULL){delete []this->floatingBinNumber;this->floatingBinNumber=NULL;}
    if(this->floatingBinNumber!=NULL){delete []this->activeVoxelNumber;this->activeVoxelNumber=NULL;}
    if(this->maxSSD!=NULL){delete []this->maxSSD;this->maxSSD=NULL;}
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetReferenceImage(nifti_image *r)
{
    this->inputReference = r;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetFloatingImage(nifti_image *f)
{
    this->inputFloating = f;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetMaximalIterationNumber(unsigned int dance)
{
    this->maxiterationNumber=dance;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetReferenceBinNumber(int l, unsigned int v)
{
    this->referenceBinNumber[l] = v;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetFloatingBinNumber(int l, unsigned int v)
{
    this->floatingBinNumber[l] = v;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetControlPointGridImage(nifti_image *cp)
{
    this->inputControlPointGrid = cp;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetReferenceMask(nifti_image *m)
{
    this->maskImage = m;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetAffineTransformation(mat44 *a)
{
    this->affineTransformation=a;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetBendingEnergyWeight(T be)
{
    this->bendingEnergyWeight = be;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetLinearEnergyWeights(T w0, T w1, T w2)
{
    this->linearEnergyWeight0=w0;
    this->linearEnergyWeight1=w1;
    this->linearEnergyWeight2=w2;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetJacobianLogWeight(T j)
{
    this->jacobianLogWeight = j;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::ApproximateJacobianLog()
{
    this->jacobianLogApproximation = true;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::DoNotApproximateJacobianLog()
{
    this->jacobianLogApproximation = false;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetReferenceSmoothingSigma(T s)
{
    this->referenceSmoothingSigma = s;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetFloatingSmoothingSigma(T s)
{
    this->floatingSmoothingSigma = s;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetReferenceThresholdUp(unsigned int i, T t)
{
    this->referenceThresholdUp[i] = t;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetReferenceThresholdLow(unsigned int i, T t)
{
    this->referenceThresholdLow[i] = t;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetFloatingThresholdUp(unsigned int i, T t)
{
    this->floatingThresholdUp[i] = t;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetFloatingThresholdLow(unsigned int i, T t)
{
    this->floatingThresholdLow[i] = t;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetWarpedPaddingValue(T p)
{
    this->warpedPaddingValue = p;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetSpacing(unsigned int i, T s)
{
    this->spacing[i] = s;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetLevelNumber(unsigned int l)
{
    this->levelNumber = l;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetLevelToPerform(unsigned int l)
{
    this->levelToPerform = l;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::SetGradientSmoothingSigma(T g)
{
    this->gradientSmoothingSigma = g;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::UseSSD()
{
    this->useSSD = true;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::DoNotUseSSD()
{
    this->useSSD = false;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::UseConjugateGradient()
{
    this->useConjGradient = true;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::DoNotUseConjugateGradient()
{
    this->useConjGradient = false;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::PrintOutInformation()
{
    this->verbose = true;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::DoNotPrintOutInformation()
{
    this->verbose = false;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
//template<class T>
//int reg_f3d<T>::SetThreadNumber(int t)
//{
//	this->threadNumber = t;
//	return 0;
//}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::DoNotUsePyramidalApproach()
{
    this->usePyramid=false;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::UseNeareatNeighborInterpolation()
{
    this->interpolation=0;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::UseLinearInterpolation()
{
    this->interpolation=1;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::UseCubicSplineInterpolation()
{
    this->interpolation=3;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateCurrentInputImage(int level)
{
    if(level!=0)
        reg_bspline_refineControlPointGrid(this->currentReference, this->controlPointGrid);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearCurrentInputImage()
{
    this->currentReference=NULL;
    this->currentMask=NULL;
    this->currentFloating=NULL;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateWarped()
{
    if(this->currentReference==NULL)
        return 1;
    reg_f3d<T>::ClearWarped();
    this->warped = nifti_copy_nim_info(this->currentReference);
    this->warped->dim[0]=this->warped->ndim=this->currentFloating->ndim;
    this->warped->dim[4]=this->warped->nt=this->currentFloating->nt;
    this->warped->pixdim[4]=this->warped->dt=1.0;
    this->warped->nvox = this->warped->nx *
            this->warped->ny *
            this->warped->nz *
            this->warped->nt;
    this->warped->datatype = this->currentFloating->datatype;
    this->warped->nbyper = this->currentFloating->nbyper;
    this->warped->data = (void *)calloc(this->warped->nvox, this->warped->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearWarped()
{
    if(this->warped!=NULL){
        nifti_image_free(this->warped);
        this->warped=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateDeformationField()
{
    if(this->currentReference==NULL)
        return 1;
    if(this->controlPointGrid==NULL)
        return 1;
    reg_f3d<T>::ClearDeformationField();
    this->deformationFieldImage = nifti_copy_nim_info(this->currentReference);
    this->deformationFieldImage->dim[0]=this->deformationFieldImage->ndim=5;
    this->deformationFieldImage->dim[1]=this->deformationFieldImage->nx=this->currentReference->nx;
    this->deformationFieldImage->dim[2]=this->deformationFieldImage->ny=this->currentReference->ny;
    this->deformationFieldImage->dim[3]=this->deformationFieldImage->nz=this->currentReference->nz;
    this->deformationFieldImage->dim[4]=this->deformationFieldImage->nt=1;
    this->deformationFieldImage->pixdim[4]=this->deformationFieldImage->dt=1.0;
    if(this->currentReference->nz==1)
        this->deformationFieldImage->dim[5]=this->deformationFieldImage->nu=2;
    else this->deformationFieldImage->dim[5]=this->deformationFieldImage->nu=3;
    this->deformationFieldImage->pixdim[5]=this->deformationFieldImage->du=1.0;
    this->deformationFieldImage->dim[6]=this->deformationFieldImage->nv=1;
    this->deformationFieldImage->pixdim[6]=this->deformationFieldImage->dv=1.0;
    this->deformationFieldImage->dim[7]=this->deformationFieldImage->nw=1;
    this->deformationFieldImage->pixdim[7]=this->deformationFieldImage->dw=1.0;
    this->deformationFieldImage->nvox=	this->deformationFieldImage->nx *
            this->deformationFieldImage->ny *
            this->deformationFieldImage->nz *
            this->deformationFieldImage->nt *
            this->deformationFieldImage->nu;
    this->deformationFieldImage->nbyper = this->controlPointGrid->nbyper;
    this->deformationFieldImage->datatype = this->controlPointGrid->datatype;
    this->deformationFieldImage->data = (void *)calloc(this->deformationFieldImage->nvox, this->deformationFieldImage->nbyper);

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearDeformationField()
{
    if(this->deformationFieldImage!=NULL){
        nifti_image_free(this->deformationFieldImage);
        this->deformationFieldImage=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateWarpedGradient()
{
    if(this->deformationFieldImage==NULL){
        return 1;
    }
    reg_f3d<T>::ClearWarpedGradient();
    this->warpedGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
    this->warpedGradientImage->dim[0]=this->warpedGradientImage->ndim=5;
    this->warpedGradientImage->nt = this->warpedGradientImage->dim[4] = this->currentFloating->nt;
    this->warpedGradientImage->nvox =	this->warpedGradientImage->nx *
            this->warpedGradientImage->ny *
            this->warpedGradientImage->nz *
            this->warpedGradientImage->nt *
            this->warpedGradientImage->nu;
    this->warpedGradientImage->data = (void *)calloc(this->warpedGradientImage->nvox, this->warpedGradientImage->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearWarpedGradient()
{
    if(this->warpedGradientImage!=NULL){
        nifti_image_free(this->warpedGradientImage);
        this->warpedGradientImage=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateVoxelBasedMeasureGradient()
{
    if(this->deformationFieldImage==NULL){
        return 1;
    }
    reg_f3d<T>::ClearVoxelBasedMeasureGradient();
    this->voxelBasedMeasureGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
    this->voxelBasedMeasureGradientImage->data = (void *)calloc(this->voxelBasedMeasureGradientImage->nvox,
                                                                this->voxelBasedMeasureGradientImage->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearVoxelBasedMeasureGradient()
{
    if(this->voxelBasedMeasureGradientImage!=NULL){
        nifti_image_free(this->voxelBasedMeasureGradientImage);
        this->voxelBasedMeasureGradientImage=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateNodeBasedMeasureGradient()
{
    if(this->controlPointGrid==NULL){
        return 1;
    }
    reg_f3d<T>::ClearNodeBasedMeasureGradient();
    this->nodeBasedMeasureGradientImage = nifti_copy_nim_info(this->controlPointGrid);
    this->nodeBasedMeasureGradientImage->data = (void *)calloc(this->nodeBasedMeasureGradientImage->nvox,
                                                               this->nodeBasedMeasureGradientImage->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearNodeBasedMeasureGradient()
{
    if(this->nodeBasedMeasureGradientImage!=NULL){
        nifti_image_free(this->nodeBasedMeasureGradientImage);
        this->nodeBasedMeasureGradientImage=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateConjugateGradientVariables()
{
    if(this->nodeBasedMeasureGradientImage==NULL)
        return 1;
    reg_f3d<T>::ClearConjugateGradientVariables();
    this->conjugateG = (T *)calloc(this->nodeBasedMeasureGradientImage->nvox, sizeof(T));
    this->conjugateH = (T *)calloc(this->nodeBasedMeasureGradientImage->nvox, sizeof(T));
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearConjugateGradientVariables()
{
    if(this->conjugateG!=NULL){
        free(this->conjugateG);
        this->conjugateG=NULL;
    }
    if(this->conjugateH!=NULL){
        free(this->conjugateH);
        this->conjugateH=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateBestControlPointArray()
{
    if(this->controlPointGrid==NULL)
        return 1;
    reg_f3d<T>::ClearBestControlPointArray();
    this->bestControlPointPosition = (T *)malloc(this->nodeBasedMeasureGradientImage->nvox*
                                                 this->nodeBasedMeasureGradientImage->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearBestControlPointArray()
{
    if(this->bestControlPointPosition!=NULL){
        free(this->bestControlPointPosition);
        this->bestControlPointPosition=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::AllocateJointHistogram()
{
    reg_f3d<T>::ClearJointHistogram();
    unsigned int histogramSize[3]={1,1,1};
    for(int i=0;i<this->currentReference->nt;i++){
        histogramSize[0] *= this->referenceBinNumber[i];
        histogramSize[1] *= this->referenceBinNumber[i];
    }
    for(int i=0;i<this->currentFloating->nt;i++){
        histogramSize[0] *= this->floatingBinNumber[i];
        histogramSize[2] *= this->floatingBinNumber[i];
    }
    histogramSize[0] += histogramSize[1] + histogramSize[2];
    this->totalBinNumber = histogramSize[0];
    this->probaJointHistogram = (double *)malloc(histogramSize[0]*sizeof(double));
    this->logJointHistogram = (double *)malloc(histogramSize[0]*sizeof(double));
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ClearJointHistogram()
{
    if(this->probaJointHistogram!=NULL){
        free(this->probaJointHistogram);
        this->probaJointHistogram=NULL;
    }
    if(this->logJointHistogram!=NULL){
        free(this->logJointHistogram);
        this->logJointHistogram=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::SaveCurrentControlPoint()
{
    memcpy(this->bestControlPointPosition, this->controlPointGrid->data,
           this->controlPointGrid->nvox*this->controlPointGrid->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::RestoreCurrentControlPoint()
{
    memcpy(this->controlPointGrid->data, this->bestControlPointPosition,
           this->controlPointGrid->nvox*this->controlPointGrid->nbyper);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::CheckParameters_f3d()
{
    // CHECK THAT BOTH INPUT IMAGES ARE DEFINED
    if(this->inputReference==NULL){
        fprintf(stderr,"[NiftyReg ERROR] No reference image has been defined.\n");
        return 1;
    }
    if(this->inputFloating==NULL){
        fprintf(stderr,"[NiftyReg ERROR] No floating image has been defined.\n");
        return 1;
    }

    if(this->useSSD){
        if(inputReference->nt>1 || inputFloating->nt>1){
            fprintf(stderr,"[NiftyReg ERROR] SSD is not available for multi-spectral registration.\n");
            return 1;
        }
    }

    // CHECK THE MASK DIMENSION IF IT IS DEFINED
    if(this->maskImage!=NULL){
        if(this->inputReference->nx != maskImage->nx ||
                this->inputReference->ny != maskImage->ny ||
                this->inputReference->nz != maskImage->nz)
            fprintf(stderr,"* The mask image has different x, y or z dimension than the reference image.\n");
    }

    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if(this->levelToPerform>0){
        this->levelToPerform=this->levelToPerform<this->levelNumber?this->levelToPerform:this->levelNumber;
    }
    else this->levelToPerform=this->levelNumber;

    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
    T penaltySum=this->bendingEnergyWeight+this->linearEnergyWeight0+this->linearEnergyWeight1+this->linearEnergyWeight2+this->jacobianLogWeight;
    if(penaltySum>=1) this->similarityWeight=0;
    else this->similarityWeight=1.0 - penaltySum;
    penaltySum+=this->similarityWeight;
    this->similarityWeight /= penaltySum;
    this->bendingEnergyWeight /= penaltySum;
    this->linearEnergyWeight0 /= penaltySum;
    this->linearEnergyWeight1 /= penaltySum;
    this->linearEnergyWeight2 /= penaltySum;
    this->jacobianLogWeight /= penaltySum;

    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if(this->levelToPerform==0 || this->levelToPerform>this->levelNumber)
        this->levelToPerform=this->levelNumber;

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::Initisalise_f3d()
{
    this->CheckParameters_f3d();

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::Initialise_f3d() called\n");
#endif
    // CREATE THE PYRAMIDE IMAGES
    nifti_image **tempMaskImagePyramid=NULL;
    if(this->usePyramid){
        this->referencePyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
        this->floatingPyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
        tempMaskImagePyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
        this->maskPyramid = (int **)malloc(this->levelToPerform*sizeof(int *));
        this->activeVoxelNumber= (int *)malloc(this->levelToPerform*sizeof(int));
    }
    else{
        this->referencePyramid = (nifti_image **)malloc(sizeof(nifti_image *));
        this->floatingPyramid = (nifti_image **)malloc(sizeof(nifti_image *));
        tempMaskImagePyramid = (nifti_image **)malloc(sizeof(nifti_image *));
        this->maskPyramid = (int **)malloc(sizeof(int *));
        this->activeVoxelNumber= (int *)malloc(sizeof(int));
    }

    // FINEST LEVEL OF REGISTRATION
    // Reference image is copied and converted to type T
    if(this->usePyramid){
        this->referencePyramid[this->levelToPerform-1]=nifti_copy_nim_info(this->inputReference);
        this->referencePyramid[this->levelToPerform-1]->data = (T *)calloc(this->referencePyramid[this->levelToPerform-1]->nvox,
                                                                           this->referencePyramid[this->levelToPerform-1]->nbyper);
        memcpy(this->referencePyramid[this->levelToPerform-1]->data, this->inputReference->data,
               this->referencePyramid[this->levelToPerform-1]->nvox* this->referencePyramid[this->levelToPerform-1]->nbyper);
        reg_changeDatatype<T>(this->referencePyramid[this->levelToPerform-1]);

        // Floating image is copied and converted to type T
        this->floatingPyramid[this->levelToPerform-1]=nifti_copy_nim_info(this->inputFloating);
        this->floatingPyramid[this->levelToPerform-1]->data = (T *)calloc(this->floatingPyramid[this->levelToPerform-1]->nvox,
                                                                          this->floatingPyramid[this->levelToPerform-1]->nbyper);
        memcpy(this->floatingPyramid[this->levelToPerform-1]->data, this->inputFloating->data,
               this->floatingPyramid[this->levelToPerform-1]->nvox* this->floatingPyramid[this->levelToPerform-1]->nbyper);
        reg_changeDatatype<T>(this->floatingPyramid[this->levelToPerform-1]);

        // Mask image is copied and converted to type unsigned char image
        if(this->maskImage!=NULL){
            tempMaskImagePyramid[this->levelToPerform-1]=nifti_copy_nim_info(this->maskImage);
            tempMaskImagePyramid[this->levelToPerform-1]->data = (T *)calloc(tempMaskImagePyramid[this->levelToPerform-1]->nvox,
                                                                             tempMaskImagePyramid[this->levelToPerform-1]->nbyper);
            memcpy(tempMaskImagePyramid[this->levelToPerform-1]->data, this->maskImage->data,
                   tempMaskImagePyramid[this->levelToPerform-1]->nvox* tempMaskImagePyramid[this->levelToPerform-1]->nbyper);
            reg_tool_binarise_image(tempMaskImagePyramid[this->levelToPerform-1]);
            reg_changeDatatype<unsigned char>(tempMaskImagePyramid[this->levelToPerform-1]);
        }
        else tempMaskImagePyramid[this->levelToPerform-1]=NULL;

        // Images are downsampled if appropriate
        for(unsigned int l=this->levelToPerform; l<this->levelNumber; l++){
            // Reference image
            bool referenceDownsampleAxis[8]={false,true,true,true,false,false,false,false};
            if((this->referencePyramid[this->levelToPerform-1]->nx/2) < 32) referenceDownsampleAxis[1]=false;
            if((this->referencePyramid[this->levelToPerform-1]->ny/2) < 32) referenceDownsampleAxis[2]=false;
            if((this->referencePyramid[this->levelToPerform-1]->nz/2) < 32) referenceDownsampleAxis[3]=false;
            reg_downsampleImage<T>(this->referencePyramid[this->levelToPerform-1], 1, referenceDownsampleAxis);
            // Mask image
            if(tempMaskImagePyramid[this->levelToPerform-1]!=NULL)
                reg_downsampleImage<T>(tempMaskImagePyramid[this->levelToPerform-1], 0, referenceDownsampleAxis);
            //Floating image
            bool floatingDownsampleAxis[8]={false,true,true,true,false,false,false,false};
            if((this->floatingPyramid[this->levelToPerform-1]->nx/2) < 32) floatingDownsampleAxis[1]=false;
            if((this->floatingPyramid[this->levelToPerform-1]->ny/2) < 32) floatingDownsampleAxis[2]=false;
            if((this->floatingPyramid[this->levelToPerform-1]->nz/2) < 32) floatingDownsampleAxis[3]=false;
            reg_downsampleImage<T>(this->floatingPyramid[this->levelToPerform-1], 1, floatingDownsampleAxis);
        }
        // Create a target mask here with the same dimension
        this->activeVoxelNumber[this->levelToPerform-1]=this->referencePyramid[this->levelToPerform-1]->nx *
                this->referencePyramid[this->levelToPerform-1]->ny *
                this->referencePyramid[this->levelToPerform-1]->nz;
        this->maskPyramid[this->levelToPerform-1]=(int *)calloc(this->activeVoxelNumber[this->levelToPerform-1], sizeof(int));
        if(tempMaskImagePyramid[this->levelToPerform-1]!=NULL){
            reg_tool_binaryImage2int(tempMaskImagePyramid[this->levelToPerform-1],
                                     this->maskPyramid[this->levelToPerform-1],
                                     this->activeVoxelNumber[this->levelToPerform-1]);
        }

        // Images for each subsequent levels are allocated and downsampled if appropriate
        for(int l=this->levelToPerform-2; l>=0; l--){
            // Allocation of the reference image
            this->referencePyramid[l]=nifti_copy_nim_info(this->referencePyramid[l+1]);
            this->referencePyramid[l]->data = (T *)calloc(  this->referencePyramid[l]->nvox,
                                                          this->referencePyramid[l]->nbyper);
            memcpy( this->referencePyramid[l]->data, this->referencePyramid[l+1]->data,
                   this->referencePyramid[l]->nvox* this->referencePyramid[l]->nbyper);

            // Allocation of the floating image
            this->floatingPyramid[l]=nifti_copy_nim_info(this->floatingPyramid[l+1]);
            this->floatingPyramid[l]->data = (T *)calloc(   this->floatingPyramid[l]->nvox,
                                                         this->floatingPyramid[l]->nbyper);
            memcpy( this->floatingPyramid[l]->data, this->floatingPyramid[l+1]->data,
                   this->floatingPyramid[l]->nvox* this->floatingPyramid[l]->nbyper);

            // Allocation of the mask image
            if(this->maskImage!=NULL){
                tempMaskImagePyramid[l]=nifti_copy_nim_info(tempMaskImagePyramid[l+1]);
                tempMaskImagePyramid[l]->data = (unsigned char *)calloc(tempMaskImagePyramid[l]->nvox,
                                                                        tempMaskImagePyramid[l]->nbyper);
                memcpy(tempMaskImagePyramid[l]->data, tempMaskImagePyramid[l+1]->data,
                       tempMaskImagePyramid[l]->nvox* tempMaskImagePyramid[l]->nbyper);
            }
            else tempMaskImagePyramid[l]=NULL;

            // Downsample the reference image
            bool referenceDownsampleAxis[8]={false,true,true,true,false,false,false,false};
            if((this->referencePyramid[l]->nx/2) < 32) referenceDownsampleAxis[1]=false;
            if((this->referencePyramid[l]->ny/2) < 32) referenceDownsampleAxis[2]=false;
            if((this->referencePyramid[l]->nz/2) < 32) referenceDownsampleAxis[3]=false;
            reg_downsampleImage<T>(this->referencePyramid[l], 1, referenceDownsampleAxis);
            // Downsample the mask image
            if(tempMaskImagePyramid[l]!=NULL)
                reg_downsampleImage<T>(tempMaskImagePyramid[l], 0, referenceDownsampleAxis);
            // Downsample the floating image
            bool floatingDownsampleAxis[8]={false,true,true,true,false,false,false,false};
            if((this->floatingPyramid[l]->nx/2) < 32) floatingDownsampleAxis[1]=false;
            if((this->floatingPyramid[l]->ny/2) < 32) floatingDownsampleAxis[2]=false;
            if((this->floatingPyramid[l]->nz/2) < 32) floatingDownsampleAxis[3]=false;
            reg_downsampleImage<T>(this->floatingPyramid[l], 1, floatingDownsampleAxis);

            // Create a target mask here with the same dimension
            this->activeVoxelNumber[l]=this->referencePyramid[l]->nx *
                    this->referencePyramid[l]->ny *
                    this->referencePyramid[l]->nz;
            this->maskPyramid[l]=(int *)calloc(activeVoxelNumber[l], sizeof(int));
            if(tempMaskImagePyramid[l]!=NULL){
                reg_tool_binaryImage2int(tempMaskImagePyramid[l],
                                         this->maskPyramid[l],
                                         this->activeVoxelNumber[l]);
            }
        }
        for(unsigned int l=0; l<this->levelToPerform; l++){
            nifti_image_free(tempMaskImagePyramid[l]);
        }
        free(tempMaskImagePyramid);

        // SMOOTH THE INPUT IMAGES IF REQUIRED
        for(unsigned int l=0; l<this->levelToPerform; l++){
            if(this->referenceSmoothingSigma!=0.0){
                bool smoothAxis[8]={false,true,true,true,false,false,false,false};
                reg_gaussianSmoothing<T>(this->referencePyramid[l], this->referenceSmoothingSigma, smoothAxis);
            }
            if(this->floatingSmoothingSigma!=0.0){
                bool smoothAxis[8]={false,true,true,true,false,false,false,false};
                reg_gaussianSmoothing<T>(this->floatingPyramid[l], this->floatingSmoothingSigma, smoothAxis);
            }
        }

        if(this->useSSD){
            this->maxSSD=new T[this->levelToPerform];
            // THRESHOLD THE INPUT IMAGES IF REQUIRED
            for(unsigned int l=0; l<this->levelToPerform; l++){
                reg_thresholdImage<T>(referencePyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
                reg_thresholdImage<T>(floatingPyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
                // The maximal difference image is extracted for normalisation
                T tempMaxSSD1 = (referencePyramid[l]->cal_min - floatingPyramid[l]->cal_max) *
                        (referencePyramid[l]->cal_min - floatingPyramid[l]->cal_max);
                T tempMaxSSD2 = (referencePyramid[l]->cal_max - floatingPyramid[l]->cal_min) *
                        (referencePyramid[l]->cal_max - floatingPyramid[l]->cal_min);
                this->maxSSD[l]=tempMaxSSD1>tempMaxSSD2?tempMaxSSD1:tempMaxSSD2;
            }
        }
        else{
            // RESCALE THE INPUT IMAGE INTENSITY
            /* the target and source are resampled between 2 and bin-3
             * The images are then shifted by two which is the suport of the spline used
             * by the parzen window filling of the joint histogram */

            float referenceRescalingArrayDown[10];
            float referenceRescalingArrayUp[10];
            float floatingRescalingArrayDown[10];
            float floatingRescalingArrayUp[10];
            for(int t=0;t<this->referencePyramid[0]->nt;t++){
                // INCREASE THE BIN SIZES
                this->referenceBinNumber[t] += 4;
                referenceRescalingArrayDown[t] = 2.f;
                referenceRescalingArrayUp[t] = this->referenceBinNumber[t]-3;
            }
            for(int t=0;t<this->floatingPyramid[0]->nt;t++){
                // INCREASE THE BIN SIZES
                this->floatingBinNumber[t] += 4;
                floatingRescalingArrayDown[t] = 2.f;
                floatingRescalingArrayUp[t] = this->floatingBinNumber[t]-3;
            }
            for(unsigned int l=0; l<this->levelToPerform; l++){
                reg_intensityRescale(this->referencePyramid[l],referenceRescalingArrayDown,referenceRescalingArrayUp,
                                     this->referenceThresholdLow, this->referenceThresholdUp);
                reg_intensityRescale(this->floatingPyramid[l],floatingRescalingArrayDown,floatingRescalingArrayUp,
                                     this->floatingThresholdLow, this->floatingThresholdUp);
            }
        }
    }
    else{ // NO PYRAMID
        // Reference image is copied and converted to type T
        this->referencePyramid[0]=nifti_copy_nim_info(this->inputReference);
        this->referencePyramid[0]->data = (T *)calloc(this->referencePyramid[0]->nvox,
                                                      this->referencePyramid[0]->nbyper);
        memcpy(this->referencePyramid[0]->data, this->inputReference->data,
               this->referencePyramid[0]->nvox* this->referencePyramid[0]->nbyper);
        reg_changeDatatype<T>(this->referencePyramid[0]);

        // Floating image is copied and converted to type T
        this->floatingPyramid[0]=nifti_copy_nim_info(this->inputFloating);
        this->floatingPyramid[0]->data = (T *)calloc(this->floatingPyramid[0]->nvox,
                                                     this->floatingPyramid[0]->nbyper);
        memcpy(this->floatingPyramid[0]->data, this->inputFloating->data,
               this->floatingPyramid[0]->nvox* this->floatingPyramid[0]->nbyper);
        reg_changeDatatype<T>(this->floatingPyramid[0]);

        // Mask image is copied and converted to type unsigned char image
        if(this->maskImage!=NULL){
            tempMaskImagePyramid[0]=nifti_copy_nim_info(this->maskImage);
            tempMaskImagePyramid[0]->data = (T *)calloc(tempMaskImagePyramid[0]->nvox,
                                                        tempMaskImagePyramid[0]->nbyper);
            memcpy(tempMaskImagePyramid[0]->data, this->maskImage->data,
                   tempMaskImagePyramid[0]->nvox* tempMaskImagePyramid[0]->nbyper);
            reg_tool_binarise_image(tempMaskImagePyramid[0]);
            reg_changeDatatype<unsigned char>(tempMaskImagePyramid[0]);
        }
        else tempMaskImagePyramid[0]=NULL;

        // Create a target mask here with the same dimension
        this->activeVoxelNumber[0]=this->referencePyramid[0]->nx *
                this->referencePyramid[0]->ny *
                this->referencePyramid[0]->nz;
        this->maskPyramid[0]=(int *)calloc(this->activeVoxelNumber[0], sizeof(int));
        if(tempMaskImagePyramid[0]!=NULL){
            reg_tool_binaryImage2int(tempMaskImagePyramid[0],
                                     this->maskPyramid[0],
                                     this->activeVoxelNumber[0]);
        }
        free(tempMaskImagePyramid[0]);free(tempMaskImagePyramid);

        // SMOOTH THE INPUT IMAGES IF REQUIRED
        if(this->referenceSmoothingSigma!=0.0){
            bool smoothAxis[8]={false,true,true,true,false,false,false,false};
            reg_gaussianSmoothing<T>(this->referencePyramid[0], this->referenceSmoothingSigma, smoothAxis);
        }
        if(this->floatingSmoothingSigma!=0.0){
            bool smoothAxis[8]={false,true,true,true,false,false,false,false};
            reg_gaussianSmoothing<T>(this->floatingPyramid[0], this->floatingSmoothingSigma, smoothAxis);
        }

        if(this->useSSD){
            this->maxSSD=new T[1];
            // THRESHOLD THE INPUT IMAGES IF REQUIRED
            reg_thresholdImage<T>(referencePyramid[0],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
            reg_thresholdImage<T>(floatingPyramid[0],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
            // The maximal difference image is extracted for normalisation
            T tempMaxSSD1 = (referencePyramid[0]->cal_min - floatingPyramid[0]->cal_max) *
                    (referencePyramid[0]->cal_min - floatingPyramid[0]->cal_max);
            T tempMaxSSD2 = (referencePyramid[0]->cal_max - floatingPyramid[0]->cal_min) *
                    (referencePyramid[0]->cal_max - floatingPyramid[0]->cal_min);
            this->maxSSD[0]=tempMaxSSD1>tempMaxSSD2?tempMaxSSD1:tempMaxSSD2;
        }
        else{
            // RESCALE THE INPUT IMAGE INTENSITY
            /* the target and source are resampled between 2 and bin-3
             * The images are then shifted by two which is the suport of the spline used
             * by the parzen window filling of the joint histogram */

            float referenceRescalingArrayDown[10];
            float referenceRescalingArrayUp[10];
            float floatingRescalingArrayDown[10];
            float floatingRescalingArrayUp[10];
            for(int t=0;t<this->referencePyramid[0]->nt;t++){
                // INCREASE THE BIN SIZES
                this->referenceBinNumber[t] += 4;
                referenceRescalingArrayDown[t] = 2.f;
                referenceRescalingArrayUp[t] = this->referenceBinNumber[t]-3;
            }
            for(int t=0;t<this->floatingPyramid[0]->nt;t++){
                // INCREASE THE BIN SIZES
                this->floatingBinNumber[t] += 4;
                floatingRescalingArrayDown[t] = 2.f;
                floatingRescalingArrayUp[t] = this->floatingBinNumber[t]-3;
            }
            reg_intensityRescale(this->referencePyramid[0],referenceRescalingArrayDown,referenceRescalingArrayUp,
                                 this->referenceThresholdLow, this->referenceThresholdUp);
            reg_intensityRescale(this->floatingPyramid[0],floatingRescalingArrayDown,floatingRescalingArrayUp,
                                 this->floatingThresholdLow, this->floatingThresholdUp);
        }
    } // END NO PYRAMID

    // DETERMINE THE GRID SPACING AND CREATE THE GRID
    if(this->inputControlPointGrid==NULL){
        if(this->spacing[1]!=this->spacing[1]) this->spacing[1]=this->spacing[0];
        if(this->spacing[2]!=this->spacing[2]) this->spacing[2]=this->spacing[0];
        /* Convert the spacing from voxel to mm if necessary */
        if(this->usePyramid){
            if(this->spacing[0]<0) this->spacing[0] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dx;
            if(this->spacing[1]<0) this->spacing[1] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dy;
            if(this->spacing[2]<0) this->spacing[2] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dz;
        }
        else{
            if(this->spacing[0]<0) this->spacing[0] *= -1.0f * this->referencePyramid[0]->dx;
            if(this->spacing[1]<0) this->spacing[1] *= -1.0f * this->referencePyramid[0]->dy;
            if(this->spacing[2]<0) this->spacing[2] *= -1.0f * this->referencePyramid[0]->dz;
        }

        float gridSpacing[3];
        gridSpacing[0] = this->spacing[0] * powf(2.0f, (float)(this->levelToPerform-1));
        gridSpacing[1] = this->spacing[1] * powf(2.0f, (float)(this->levelToPerform-1));
        gridSpacing[2] = 1.0f;

        /* allocate the control point image */
        int dim_cpp[8];
        dim_cpp[0]=5;
        dim_cpp[1]=(int)floor(this->referencePyramid[0]->nx*this->referencePyramid[0]->dx/gridSpacing[0])+5;
        dim_cpp[2]=(int)floor(this->referencePyramid[0]->ny*this->referencePyramid[0]->dy/gridSpacing[1])+5;
        dim_cpp[3]=1;
        dim_cpp[5]=2;
        if(this->referencePyramid[0]->nz>1){
            gridSpacing[2] = this->spacing[2] * powf(2.0f, (float)(this->levelToPerform-1));
            dim_cpp[3]=(int)floor(this->referencePyramid[0]->nz*this->referencePyramid[0]->dz/gridSpacing[2])+5;
            dim_cpp[5]=3;
        }
        dim_cpp[4]=dim_cpp[6]=dim_cpp[7]=1;
        if(sizeof(T)==4) this->controlPointGrid = nifti_make_new_nim(dim_cpp, NIFTI_TYPE_FLOAT32, true);
        else this->controlPointGrid = nifti_make_new_nim(dim_cpp, NIFTI_TYPE_FLOAT64, true);
        this->controlPointGrid->cal_min=0;
        this->controlPointGrid->cal_max=0;
        this->controlPointGrid->pixdim[0]=1.0f;
        this->controlPointGrid->pixdim[1]=this->controlPointGrid->dx=gridSpacing[0];
        this->controlPointGrid->pixdim[2]=this->controlPointGrid->dy=gridSpacing[1];
        if(this->referencePyramid[0]->nz==1){
            this->controlPointGrid->pixdim[3]=this->controlPointGrid->dz=1.0f;
        }
        else this->controlPointGrid->pixdim[3]=this->controlPointGrid->dz=gridSpacing[2];
        this->controlPointGrid->pixdim[4]=this->controlPointGrid->dt=1.0f;
        this->controlPointGrid->pixdim[5]=this->controlPointGrid->du=1.0f;
        this->controlPointGrid->pixdim[6]=this->controlPointGrid->dv=1.0f;
        this->controlPointGrid->pixdim[7]=this->controlPointGrid->dw=1.0f;
        this->controlPointGrid->qform_code=this->referencePyramid[0]->qform_code;
        this->controlPointGrid->sform_code=this->referencePyramid[0]->sform_code;

        // The qform (and sform) are set for the control point position image
        float qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac;
        nifti_mat44_to_quatern( this->referencePyramid[0]->qto_xyz, &qb, &qc, &qd, &qx, &qy, &qz, &dx, &dy, &dz, &qfac);
        this->controlPointGrid->quatern_b=qb;
        this->controlPointGrid->quatern_c=qc;
        this->controlPointGrid->quatern_d=qd;
        this->controlPointGrid->qfac=qfac;

        this->controlPointGrid->qto_xyz = nifti_quatern_to_mat44(qb, qc, qd, qx, qy, qz,
                                                                 this->controlPointGrid->dx, this->controlPointGrid->dy, this->controlPointGrid->dz, qfac);

        // Origin is shifted from 1 control point in the qform
        float originIndex[3];
        float originReal[3];
        originIndex[0] = -1.0f;
        originIndex[1] = -1.0f;
        originIndex[2] = 0.0f;
        if(referencePyramid[0]->nz>1) originIndex[2] = -1.0f;
        reg_mat44_mul(&(this->controlPointGrid->qto_xyz), originIndex, originReal);
        if(this->controlPointGrid->qform_code==0) this->controlPointGrid->qform_code=1;
        this->controlPointGrid->qto_xyz.m[0][3] = this->controlPointGrid->qoffset_x = originReal[0];
        this->controlPointGrid->qto_xyz.m[1][3] = this->controlPointGrid->qoffset_y = originReal[1];
        this->controlPointGrid->qto_xyz.m[2][3] = this->controlPointGrid->qoffset_z = originReal[2];

        this->controlPointGrid->qto_ijk = nifti_mat44_inverse(this->controlPointGrid->qto_xyz);

        if(this->controlPointGrid->sform_code>0){
            nifti_mat44_to_quatern(this->referencePyramid[0]->sto_xyz, &qb, &qc, &qd, &qx,
                                   &qy, &qz, &dx, &dy, &dz, &qfac);

            this->controlPointGrid->sto_xyz = nifti_quatern_to_mat44(qb, qc, qd, qx, qy, qz,
                                                                     this->controlPointGrid->dx, this->controlPointGrid->dy, this->controlPointGrid->dz, qfac);

            // Origin is shifted from 1 control point in the sform
            originIndex[0] = -1.0f;
            originIndex[1] = -1.0f;
            originIndex[2] = 0.0f;
            if(this->referencePyramid[0]->nz>1) originIndex[2] = -1.0f;
            reg_mat44_mul(&(this->controlPointGrid->sto_xyz), originIndex, originReal);
            this->controlPointGrid->sto_xyz.m[0][3] = originReal[0];
            this->controlPointGrid->sto_xyz.m[1][3] = originReal[1];
            this->controlPointGrid->sto_xyz.m[2][3] = originReal[2];

            this->controlPointGrid->sto_ijk = nifti_mat44_inverse(this->controlPointGrid->sto_xyz);
        }
        // The control point position image is initialised with the affine transformation
        if(this->affineTransformation==NULL){
            mat44 identityAffine;
            identityAffine.m[0][0]=1.f;
            identityAffine.m[0][1]=0.f;
            identityAffine.m[0][2]=0.f;
            identityAffine.m[0][3]=0.f;
            identityAffine.m[1][0]=0.f;
            identityAffine.m[1][1]=1.f;
            identityAffine.m[1][2]=0.f;
            identityAffine.m[1][3]=0.f;
            identityAffine.m[2][0]=0.f;
            identityAffine.m[2][1]=0.f;
            identityAffine.m[2][2]=1.f;
            identityAffine.m[2][3]=0.f;
            identityAffine.m[3][0]=0.f;
            identityAffine.m[3][1]=0.f;
            identityAffine.m[3][2]=0.f;
            identityAffine.m[3][3]=1.f;
            if(reg_bspline_initialiseControlPointGridWithAffine(&identityAffine, this->controlPointGrid))
                return 1;
        }
        else if(reg_bspline_initialiseControlPointGridWithAffine(this->affineTransformation, this->controlPointGrid))
            return 1;
    }
    else{
        // The control point grid image is initialised with the provided grid
        this->controlPointGrid = nifti_copy_nim_info(this->inputControlPointGrid);
        this->controlPointGrid->data = (void *)malloc( this->controlPointGrid->nvox *
                                                      this->controlPointGrid->nbyper);
        memcpy( this->controlPointGrid->data, this->inputControlPointGrid->data,
               this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
        // The final grid spacing is computed
        this->spacing[0] = this->controlPointGrid->dx / powf(2.0f, (float)(this->levelToPerform-1));
        this->spacing[1] = this->controlPointGrid->dy / powf(2.0f, (float)(this->levelToPerform-1));
        if(this->controlPointGrid->nz>1)
            this->spacing[2] = this->controlPointGrid->dz / powf(2.0f, (float)(this->levelToPerform-1));
    }

#ifdef NDEBUG
    if(this->verbose){
#endif
        printf("[%s] **************************************************\n", this->executableName);
        printf("[%s] INPUT PARAMETERS\n", this->executableName);
        printf("[%s] **************************************************\n", this->executableName);
        printf("[%s] Reference image:\n", this->executableName);
        printf("[%s] \t* name: %s\n", this->executableName, this->inputReference->fname);
        printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
               this->inputReference->nx, this->inputReference->ny,
               this->inputReference->nz, this->inputReference->nt);
        printf("[%s] \t* image spacing: %g x %g x %g mm\n",
               this->executableName, this->inputReference->dx,
               this->inputReference->dy, this->inputReference->dz);
        for(int i=0;i<this->inputReference->nt;i++){
            printf("[%s] \t* intensity threshold for timepoint %i/%i: [%.2g %.2g]\n", this->executableName,
                   i+1, this->inputReference->nt, this->referenceThresholdLow[i],this->referenceThresholdUp[i]);
            if(!this->useSSD)
                printf("[%s] \t* binnining size for timepoint %i/%i: %i\n", this->executableName,
                       i+1, this->inputReference->nt, this->referenceBinNumber[i]-4);
        }
        printf("[%s] \t* gaussian smoothing sigma: %g\n", this->executableName, this->referenceSmoothingSigma);
        printf("[%s]\n", this->executableName);
        printf("[%s] Floating image:\n", this->executableName);
        printf("[%s] \t* name: %s\n", this->executableName, this->inputFloating->fname);
        printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
               this->inputFloating->nx, this->inputFloating->ny,
               this->inputFloating->nz, this->inputFloating->nt);
        printf("[%s] \t* image spacing: %g x %g x %g mm\n",
               this->executableName, this->inputFloating->dx,
               this->inputFloating->dy, this->inputFloating->dz);
        for(int i=0;i<this->inputFloating->nt;i++){
            printf("[%s] \t* intensity threshold for timepoint %i/%i: [%.2g %.2g]\n", this->executableName,
                   i+1, this->inputFloating->nt, this->floatingThresholdLow[i],this->floatingThresholdUp[i]);
            if(!this->useSSD)
                printf("[%s] \t* binnining size for timepoint %i/%i: %i\n", this->executableName,
                       i+1, this->inputFloating->nt, this->floatingBinNumber[i]-4);
        }
        printf("[%s] \t* gaussian smoothing sigma: %g\n",
               this->executableName, this->floatingSmoothingSigma);
        printf("[%s]\n", this->executableName);
        printf("[%s] Warped image padding value: %g\n", this->executableName, this->warpedPaddingValue);
        printf("[%s]\n", this->executableName);
        printf("[%s] Level number: %i\n", this->executableName, this->levelNumber);
        if(this->levelNumber!=this->levelToPerform)
            printf("[%s] \t* Level to perform: %i\n", this->executableName, this->levelToPerform);
        printf("[%s]\n", this->executableName);
        printf("[%s] Maximum iteration number per level: %i\n", this->executableName, this->maxiterationNumber);
        printf("[%s]\n", this->executableName);
        printf("[%s] Final spacing in mm: %g %g %g\n", this->executableName,
               this->spacing[0], this->spacing[1], this->spacing[2]);
        printf("[%s]\n", this->executableName);
        if(this->useSSD)
            printf("[%s] The SSD is used as a similarity measure.\n", this->executableName);
        else printf("[%s] The NMI is used as a similarity measure.\n", this->executableName);
        printf("[%s] Similarity measure term weight: %g\n", this->executableName, this->similarityWeight);
        printf("[%s]\n", this->executableName);
        printf("[%s] Bending energy penalty term weight: %g\n", this->executableName, this->bendingEnergyWeight);
        printf("[%s]\n", this->executableName);
        printf("[%s] Linear energy penalty term weights: %g %g %g\n", this->executableName,
               this->linearEnergyWeight0, this->linearEnergyWeight1, this->linearEnergyWeight2);
        printf("[%s]\n", this->executableName);
        printf("[%s] Jacobian-based penalty term weight: %g\n", this->executableName, this->jacobianLogWeight);
        if(this->jacobianLogWeight>0){
            if(this->jacobianLogApproximation) printf("[%s] \t* Jacobian-based penalty term is approximated\n",
                                                      this->executableName);
            else printf("[%s] \t* Jacobian-based penalty term is not approximated\n", this->executableName);
        }
        printf("[%s] --------------------------------------------------\n", this->executableName);
#ifdef NDEBUG
    }
#endif

    this->initialised=true;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::Initialise_f3d() done\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::GetDeformationField()
{
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   false, //composition
                                   true // bspline
                                   );
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::WarpFloatingImage(int inter)
{
    // Compute the deformation field
    this->GetDeformationField();
    // Resample the floating image
    reg_resampleSourceImage(this->currentReference,
                            this->currentFloating,
                            this->warped,
                            this->deformationFieldImage,
                            this->currentMask,
                            inter,
                            this->warpedPaddingValue);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeSimilarityMeasure()
{
    double measure=0.;
    if(this->useSSD){
        measure = -reg_getSSD(this->currentReference,
                              this->warped,
                              this->currentMask);
        if(this->usePyramid)
            measure /= this->maxSSD[this->currentLevel];
        else measure /= this->maxSSD[0];
    }
    else{
        reg_getEntropies(this->currentReference,
                         this->warped,
                         //2,
                         this->referenceBinNumber,
                         this->floatingBinNumber,
                         this->probaJointHistogram,
                         this->logJointHistogram,
                         this->entropies,
                         this->currentMask);
        measure = (this->entropies[0]+this->entropies[1])/this->entropies[2];
    }
    return double(this->similarityWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
    double value=0.;

    if(type==2){
        value = reg_bspline_jacobian(this->controlPointGrid,
                                     this->currentReference,
                                     false);
    }
    else{
        value = reg_bspline_jacobian(this->controlPointGrid,
                                     this->currentReference,
                                     this->jacobianLogApproximation);
    }
    unsigned int maxit=5;
    if(type>0) maxit=20;
    unsigned int it=0;
    while(value!=value && it<maxit){
        if(type==2){
            value = reg_bspline_correctFolding(this->controlPointGrid,
                                               this->currentReference,
                                               false);
        }
        else{
            value = reg_bspline_correctFolding(this->controlPointGrid,
                                               this->currentReference,
                                               this->jacobianLogApproximation);
        }
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Folding correction\n");
#endif
        it++;
    }
    if(type>0){
        if(value!=value){
            this->RestoreCurrentControlPoint();
            fprintf(stderr, "[NiftyReg ERROR] The folding correction scheme failed\n");
        }
        else{
#ifdef NDEBUG
            if(this->verbose){
#endif
                printf("[%s] Folding correction, %i step(s)\n", this->executableName, it);
#ifdef NDEBUG
            }
#endif
        }
    }
    return (double)this->jacobianLogWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeBendingEnergyPenaltyTerm()
{
    double value = reg_bspline_bendingEnergy(this->controlPointGrid);
    return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm()
{
    double values[3];
    reg_bspline_linearEnergy(this->controlPointGrid, values);
    return this->linearEnergyWeight0*values[0] +
            this->linearEnergyWeight1*values[1] +
            this->linearEnergyWeight2*values[2];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::GetVoxelBasedGradient()
{
    // The intensity gradient is first computed
    reg_getSourceImageGradient(this->currentReference,
                               this->currentFloating,
                               this->warpedGradientImage,
                               this->deformationFieldImage,
                               this->currentMask,
                               this->interpolation);

    if(this->useSSD){
        // Compute the voxel based SSD gradient
        T localMaxSSD=this->maxSSD[0];
        if(this->usePyramid) localMaxSSD=this->maxSSD[this->currentLevel];
        reg_getVoxelBasedSSDGradient(this->currentReference,
                                     this->warped,
                                     this->warpedGradientImage,
                                     this->voxelBasedMeasureGradientImage,
                                     localMaxSSD,
                                     this->currentMask
                                     );
    }
    else{
        // Compute the voxel based NMI gradient
        reg_getVoxelBasedNMIGradientUsingPW(this->currentReference,
                                            this->warped,
                                            //2,
                                            this->warpedGradientImage,
                                            this->referenceBinNumber,
                                            this->floatingBinNumber,
                                            this->logJointHistogram,
                                            this->entropies,
                                            this->voxelBasedMeasureGradientImage,
                                            this->currentMask);
    }
    return 0;
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::GetSimilarityMeasureGradient()
{
    this->GetVoxelBasedGradient();

    // The voxel based NMI gradient is convolved with a spline kernel
    int smoothingRadius[3];
    smoothingRadius[0] = (int)( 2.0*this->controlPointGrid->dx/this->currentReference->dx );
    smoothingRadius[1] = (int)( 2.0*this->controlPointGrid->dy/this->currentReference->dy );
    smoothingRadius[2] = (int)( 2.0*this->controlPointGrid->dz/this->currentReference->dz );
    reg_smoothImageForCubicSpline<T>(this->voxelBasedMeasureGradientImage,
                                     smoothingRadius);

    // The node based NMI gradient is extracted
    reg_voxelCentric2NodeCentric(this->nodeBasedMeasureGradientImage,
                                 this->voxelBasedMeasureGradientImage,
                                 this->similarityWeight,
                                 false);

    /* The gradient is converted from voxel space to real space */
    mat44 *floatingMatrix_xyz=NULL;
    int controlPointNumber=this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz;
    int i;
    if(this->currentFloating->sform_code>0)
        floatingMatrix_xyz = &(this->currentFloating->sto_xyz);
    else floatingMatrix_xyz = &(this->currentFloating->qto_xyz);
    if(this->currentReference->nz==1){
        T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
        T *gradientValuesY = &gradientValuesX[controlPointNumber];
        T newGradientValueX, newGradientValueY;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gradientValuesX, gradientValuesY, floatingMatrix_xyz, controlPointNumber) \
    private(newGradientValueX, newGradientValueY, i)
#endif
        for(i=0; i<controlPointNumber; i++){
            newGradientValueX = gradientValuesX[i] * floatingMatrix_xyz->m[0][0] +
                    gradientValuesY[i] * floatingMatrix_xyz->m[0][1];
            newGradientValueY = gradientValuesX[i] * floatingMatrix_xyz->m[1][0] +
                    gradientValuesY[i] * floatingMatrix_xyz->m[1][1];
            gradientValuesX[i] = newGradientValueX;
            gradientValuesY[i] = newGradientValueY;
        }
    }
    else{
        T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
        T *gradientValuesY = &gradientValuesX[controlPointNumber];
        T *gradientValuesZ = &gradientValuesY[controlPointNumber];
        T newGradientValueX, newGradientValueY, newGradientValueZ;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gradientValuesX, gradientValuesY, gradientValuesZ, floatingMatrix_xyz, controlPointNumber) \
    private(newGradientValueX, newGradientValueY, newGradientValueZ, i)
#endif
        for(i=0; i<controlPointNumber; i++){

            newGradientValueX = gradientValuesX[i] * floatingMatrix_xyz->m[0][0] +
                    gradientValuesY[i] * floatingMatrix_xyz->m[0][1] +
                    gradientValuesZ[i] * floatingMatrix_xyz->m[0][2];
            newGradientValueY = gradientValuesX[i] * floatingMatrix_xyz->m[1][0] +
                    gradientValuesY[i] * floatingMatrix_xyz->m[1][1] +
                    gradientValuesZ[i] * floatingMatrix_xyz->m[1][2];
            newGradientValueZ = gradientValuesX[i] * floatingMatrix_xyz->m[2][0] +
                    gradientValuesY[i] * floatingMatrix_xyz->m[2][1] +
                    gradientValuesZ[i] * floatingMatrix_xyz->m[2][2];
            gradientValuesX[i] = newGradientValueX;
            gradientValuesY[i] = newGradientValueY;
            gradientValuesZ[i] = newGradientValueZ;
        }
    }

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::GetBendingEnergyGradient()
{
    reg_bspline_bendingEnergyGradient(this->controlPointGrid,
                                      this->currentReference,
                                      this->nodeBasedMeasureGradientImage,
                                      this->bendingEnergyWeight);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::GetLinearEnergyGradient()
{
    reg_bspline_linearEnergyGradient(this->controlPointGrid,
                                     this->currentReference,
                                     this->nodeBasedMeasureGradientImage,
                                     this->linearEnergyWeight0,
                                     this->linearEnergyWeight1,
                                     this->linearEnergyWeight2);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::GetJacobianBasedGradient()
{
    reg_bspline_jacobianDeterminantGradient(this->controlPointGrid,
                                            this->currentReference,
                                            this->nodeBasedMeasureGradientImage,
                                            this->jacobianLogWeight,
                                            this->jacobianLogApproximation);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::ComputeConjugateGradient()
{
    int nodeNumber = this->nodeBasedMeasureGradientImage->nx *
            this->nodeBasedMeasureGradientImage->ny *
            this->nodeBasedMeasureGradientImage->nz;
    int i;
    if(this->currentIteration==1){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Conjugate gradient initialisation\n");
#endif
        // first conjugate gradient iteration
        if(this->currentReference->nz==1){
            T *conjGPtrX = &conjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjHPtrX = &conjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjGPtrX, conjGPtrY, \
    gradientValuesX, gradientValuesY, nodeNumber) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjHPtrX[i] = conjGPtrX[i] = - gradientValuesX[i];
                conjHPtrY[i] = conjGPtrY[i] = - gradientValuesY[i];
            }
        }else{
            T *conjGPtrX = &conjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjGPtrZ = &conjGPtrY[nodeNumber];
            T *conjHPtrX = &conjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *conjHPtrZ = &conjHPtrY[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
            T *gradientValuesZ = &gradientValuesY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjHPtrZ, conjGPtrX, conjGPtrY, conjGPtrZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjHPtrX[i] = conjGPtrX[i] = - gradientValuesX[i];
                conjHPtrY[i] = conjGPtrY[i] = - gradientValuesY[i];
                conjHPtrZ[i] = conjGPtrZ[i] = - gradientValuesZ[i];
            }
        }
    }
    else{
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Conjugate gradient update\n");
#endif
        double dgg=0.0, gg=0.0;
        if(this->currentReference->nz==1){
            T *conjGPtrX = &conjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjHPtrX = &conjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjGPtrX, conjGPtrY, \
    gradientValuesX, gradientValuesY, nodeNumber) \
    private(i) \
    reduction(+:gg) \
    reduction(+:dgg)
#endif
            for(i=0; i<nodeNumber;i++){
                gg += conjHPtrX[i] * conjGPtrX[i];
                gg += conjHPtrY[i] * conjGPtrY[i];
                dgg += (gradientValuesX[i] + conjGPtrX[i]) * gradientValuesX[i];
                dgg += (gradientValuesY[i] + conjGPtrY[i]) * gradientValuesY[i];
            }
            double gam = dgg/gg;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjGPtrX, conjGPtrY, \
    gradientValuesX, gradientValuesY, nodeNumber, gam) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjGPtrX[i] = - gradientValuesX[i];
                conjGPtrY[i] = - gradientValuesY[i];
                conjHPtrX[i] = (float)(conjGPtrX[i] + gam * conjHPtrX[i]);
                conjHPtrY[i] = (float)(conjGPtrY[i] + gam * conjHPtrY[i]);
                gradientValuesX[i] = - conjHPtrX[i];
                gradientValuesY[i] = - conjHPtrY[i];
            }
        }
        else{
            T *conjGPtrX = &conjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjGPtrZ = &conjGPtrY[nodeNumber];
            T *conjHPtrX = &conjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *conjHPtrZ = &conjHPtrY[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
            T *gradientValuesZ = &gradientValuesY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjHPtrZ, conjGPtrX, conjGPtrY, conjGPtrZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber) \
    private(i) \
    reduction(+:gg) \
    reduction(+:dgg)
#endif
            for(i=0; i<nodeNumber;i++){
                gg += conjHPtrX[i] * conjGPtrX[i];
                gg += conjHPtrY[i] * conjGPtrY[i];
                gg += conjHPtrZ[i] * conjGPtrZ[i];
                dgg += (gradientValuesX[i] + conjGPtrX[i]) * gradientValuesX[i];
                dgg += (gradientValuesY[i] + conjGPtrY[i]) * gradientValuesY[i];
                dgg += (gradientValuesZ[i] + conjGPtrZ[i]) * gradientValuesZ[i];
            }
            double gam = dgg/gg;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjHPtrZ, conjGPtrX, conjGPtrY, conjGPtrZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber, gam) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjGPtrX[i] = - gradientValuesX[i];
                conjGPtrY[i] = - gradientValuesY[i];
                conjGPtrZ[i] = - gradientValuesZ[i];
                conjHPtrX[i] = (float)(conjGPtrX[i] + gam * conjHPtrX[i]);
                conjHPtrY[i] = (float)(conjGPtrY[i] + gam * conjHPtrY[i]);
                conjHPtrZ[i] = (float)(conjGPtrZ[i] + gam * conjHPtrZ[i]);
                gradientValuesX[i] = - conjHPtrX[i];
                gradientValuesY[i] = - conjHPtrY[i];
                gradientValuesZ[i] = - conjHPtrZ[i];
            }
        }
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d<T>::GetMaximalGradientLength()
{
    return reg_getMaximalLength<T>(this->nodeBasedMeasureGradientImage);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::UpdateControlPointPosition(T scale)
{
    int nodeNumber = this->controlPointGrid->nx *
            this->controlPointGrid->ny *
            this->controlPointGrid->nz;
    int i;
    if(this->currentReference->nz==1){
        T *controlPointValuesX = static_cast<T *>(this->controlPointGrid->data);
        T *controlPointValuesY = &controlPointValuesX[nodeNumber];
        T *bestControlPointValuesX = &this->bestControlPointPosition[0];
        T *bestControlPointValuesY = &bestControlPointValuesX[nodeNumber];
        T *gradientValuesX = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
        T *gradientValuesY = &gradientValuesX[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(controlPointValuesX, controlPointValuesY, bestControlPointValuesX, \
    bestControlPointValuesY, gradientValuesX, gradientValuesY, nodeNumber, scale) \
    private(i)
#endif
        for(i=0; i<nodeNumber;i++){
            controlPointValuesX[i] = bestControlPointValuesX[i] + scale * gradientValuesX[i];
            controlPointValuesY[i] = bestControlPointValuesY[i] + scale * gradientValuesY[i];
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
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(controlPointValuesX, controlPointValuesY, controlPointValuesZ, \
    bestControlPointValuesX, bestControlPointValuesY, bestControlPointValuesZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber, scale) \
    private(i)
#endif
        for(i=0; i<nodeNumber;i++){
            controlPointValuesX[i] = bestControlPointValuesX[i] + scale * gradientValuesX[i];
            controlPointValuesY[i] = bestControlPointValuesY[i] + scale * gradientValuesY[i];
            controlPointValuesZ[i] = bestControlPointValuesZ[i] + scale * gradientValuesZ[i];
        }
    }

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d<T>::Run_f3d()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::Run_f3d() called\n");
#endif
    if(!this->initialised){
        if( this->Initisalise_f3d() )
            return 1;
    }

    for(unsigned int level=0; level<this->levelToPerform; level++){

        this->currentLevel = level;

        if(this->usePyramid){
            this->currentReference = this->referencePyramid[this->currentLevel];
            this->currentFloating = this->floatingPyramid[this->currentLevel];
            this->currentMask = this->maskPyramid[this->currentLevel];
        }
        else{
            this->currentReference = this->referencePyramid[0];
            this->currentFloating = this->floatingPyramid[0];
            this->currentMask = this->maskPyramid[0];
        }

        // ALLOCATE IMAGES THAT DEPENDS ON THE TARGET IMAGE
        this->AllocateWarped();
        this->AllocateDeformationField();
        this->AllocateWarpedGradient();
        this->AllocateVoxelBasedMeasureGradient();
        this->AllocateJointHistogram();

        // The grid is refined if necessary
        this->AllocateCurrentInputImage(level);

        // ALLOCATE IMAGES THAT DEPENDS ON THE CONTROL POINT IMAGE
        this->AllocateNodeBasedMeasureGradient();
        this->AllocateBestControlPointArray();
        this->SaveCurrentControlPoint();
        if(this->useConjGradient){
            this->AllocateConjugateGradientVariables();
        }

#ifdef NDEBUG
        if(this->verbose){
#endif
            printf("[%s] **************************************************\n", this->executableName);
            printf("[%s] Current level: %i / %i\n", this->executableName, level+1, this->levelNumber);
            printf("[%s] Current reference image\n", this->executableName);
            printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
                   this->currentReference->nx, this->currentReference->ny,
                   this->currentReference->nz,this->currentReference->nt);
            printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
                   this->currentReference->dx, this->currentReference->dy,
                   this->currentReference->dz);
            printf("[%s] Current floating image\n", this->executableName);
            printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
                   this->currentFloating->nx, this->currentFloating->ny,
                   this->currentFloating->nz,this->currentFloating->nt);
            printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
                   this->currentFloating->dx, this->currentFloating->dy,
                   this->currentFloating->dz);
            printf("[%s] Current control point image\n", this->executableName);
            printf("[%s] \t* image dimension: %i x %i x %i\n", this->executableName,
                   this->controlPointGrid->nx, this->controlPointGrid->ny,
                   this->controlPointGrid->nz);
            printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
                   this->controlPointGrid->dx, this->controlPointGrid->dy,
                   this->controlPointGrid->dz);
#ifdef NDEBUG
        }
#endif

#ifndef NDEBUG
        if(this->currentReference->sform_code>0)
            reg_mat44_disp(&(this->currentReference->sto_xyz), (char *)"[NiftyReg DEBUG] Reference sform");
        else reg_mat44_disp(&(this->currentReference->qto_xyz), (char *)"[NiftyReg DEBUG] Reference qform");

        if(this->currentFloating->sform_code>0)
            reg_mat44_disp(&(this->currentFloating->sto_xyz), (char *)"[NiftyReg DEBUG] Floating sform");
        else reg_mat44_disp(&(this->currentFloating->qto_xyz), (char *)"[NiftyReg DEBUG] Floating qform");

        if(this->controlPointGrid->sform_code>0)
            reg_mat44_disp(&(this->controlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] CPP sform");
        else reg_mat44_disp(&(this->controlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] CPP qform");
#endif

        T maxStepSize = (this->currentReference->dx>this->currentReference->dy)?this->currentReference->dx:this->currentReference->dy;
        maxStepSize = (this->currentReference->dz>maxStepSize)?this->currentReference->dz:maxStepSize;
        T currentSize = maxStepSize;
        T smallestSize = maxStepSize / 100.0f;

        // Compute initial penalty terms
        double bestWJac = 0.0;
        if(this->jacobianLogWeight>0)
            bestWJac = this->ComputeJacobianBasedPenaltyTerm(1); // 20 iterations

        double bestWBE = 0.0;
        if(this->bendingEnergyWeight>0)
            bestWBE = this->ComputeBendingEnergyPenaltyTerm();

        double bestWLE = 0.0;
        if(this->linearEnergyWeight0>0 || this->linearEnergyWeight1>0 || this->linearEnergyWeight2>0)
            bestWLE = this->ComputeLinearEnergyPenaltyTerm();

        // Compute initial similarity measure
        double bestWMeasure = 0.0;
        if(this->similarityWeight>0){
            this->WarpFloatingImage(this->interpolation);
            bestWMeasure = this->ComputeSimilarityMeasure();
        }

        // Evalulate the objective function value
        double bestValue = bestWMeasure - bestWBE - bestWLE - bestWJac;

#ifdef NDEBUG
        if(this->verbose){
#endif
            if(this->useSSD)
                printf("[%s] Initial objective function: %g = (wSSD)%g - (wBE)%g - (wLE)%g - (wJAC)%g\n",
                       this->executableName, bestValue, bestWMeasure, bestWBE, bestWLE, bestWJac);
            else printf("[%s] Initial objective function: %g = (wNMI)%g - (wBE)%g - (wLE)%g - (wJAC)%g\n",
                        this->executableName, bestValue, bestWMeasure, bestWBE, bestWLE, bestWJac);
#ifdef NDEBUG
        }
#endif
        // The initial objective function values are kept

        this->currentIteration = 0;
        while(this->currentIteration<this->maxiterationNumber){

            if(currentSize<=smallestSize)
                // The algorithm converged and will stop in f3d.
                // It will stop approximation in f3d2
                if(this->CheckStoppingCriteria(true)) break;

            // Compute the gradient of the similarity measure
            if(this->similarityWeight>0){
                this->WarpFloatingImage(this->interpolation);
                this->ComputeSimilarityMeasure();
                this->GetSimilarityMeasureGradient();
            }
            else{
                T* nodeGradPtr = static_cast<T *>(this->nodeBasedMeasureGradientImage->data);
                for(unsigned int i=0; i<this->nodeBasedMeasureGradientImage->nvox; ++i)
                    *nodeGradPtr++=0;
            }this->currentIteration++;

            // The gradient is smoothed using a Gaussian kernel if it is required
            if(this->gradientSmoothingSigma!=0){
                reg_gaussianSmoothing<T>(this->nodeBasedMeasureGradientImage,
                                         fabs(this->gradientSmoothingSigma),
                                         NULL);
            }

            // The conjugate gradient is computed, only on the similarity measure gradient
            if(this->useConjGradient && this->similarityWeight>0) this->ComputeConjugateGradient();

            // Compute the penalty term gradients if required
            if(this->bendingEnergyWeight>0)
                this->GetBendingEnergyGradient();
            if(this->jacobianLogWeight>0)
                this->GetJacobianBasedGradient();
            if(this->linearEnergyWeight0>0 || this->linearEnergyWeight1>0 || this->linearEnergyWeight2>0)
                this->GetLinearEnergyGradient();

            T maxLength = this->GetMaximalGradientLength();
#ifndef NDEBUG
            printf("[NiftyReg DEBUG] Objective function gradient maximal length: %g\n",maxLength);
#endif
            if(maxLength==0){
                printf("No Gradient ... exit\n");
                break;
            }

            // A line ascent is performed
            int lineIteration = 0;
            currentSize=maxStepSize;
            T addedStep=0.0f;
            while(currentSize>smallestSize && lineIteration<12){
                T currentLength = -currentSize/maxLength;
#ifndef NDEBUG
                printf("[NiftyReg DEBUG] Current added max step: %g\n", currentSize);
#endif
                this->UpdateControlPointPosition(currentLength);

                // The new objective function value is computed
                double currentWJac=0;
                if(this->jacobianLogWeight>0){
                    currentWJac = this->ComputeJacobianBasedPenaltyTerm(0); // 5 iterations
                }

                double currentWBE=0;
                if(this->bendingEnergyWeight>0)
                    currentWBE = this->ComputeBendingEnergyPenaltyTerm();

                double currentWLE = 0.0;
                if(this->linearEnergyWeight0>0 || this->linearEnergyWeight1>0 || this->linearEnergyWeight2>0)
                    currentWLE = this->ComputeLinearEnergyPenaltyTerm();

                double currentWMeasure = 0.0;
                if(this->similarityWeight>0){
                    this->WarpFloatingImage(this->interpolation);
                    currentWMeasure = this->ComputeSimilarityMeasure();
                } this->currentIteration++;
                double currentValue = currentWMeasure - currentWBE - currentWLE - currentWJac;

                if(currentValue>bestValue){
                    bestValue = currentValue;
                    bestWMeasure = currentWMeasure;
                    bestWBE = currentWBE;
                    bestWLE = currentWLE;
                    bestWJac = currentWJac;
                    addedStep += currentSize;
                    currentSize*=1.1f;
                    currentSize = (currentSize<maxStepSize)?currentSize:maxStepSize;
                    this->SaveCurrentControlPoint();
#ifndef NDEBUG
                    printf("[NiftyReg DEBUG] [%i] objective function: %g = %g - %g - %g | KEPT\n",
                           this->currentIteration, currentValue, currentWMeasure, currentWBE, currentWJac);
#endif
                }
                else{
                    currentSize*=0.5;
#ifndef NDEBUG
                    printf("[NiftyReg DEBUG] [%i] objective function: %g = %g - %g - %g | REJECTED\n",
                           this->currentIteration, currentValue, currentWMeasure, currentWBE, currentWJac);
#endif
                }
                lineIteration++;
                if(this->CheckStoppingCriteria(false)) break;
            }
            this->RestoreCurrentControlPoint();
            currentSize=addedStep;
#ifdef NDEBUG
            if(this->verbose){
#endif
                printf("[%s] [%i] Current objective function: %g",
                       this->executableName, this->currentIteration, bestValue);
                if(this->useSSD)
                    printf(" = (wSSD)%g", bestWMeasure);
                else printf(" = (wNMI)%g", bestWMeasure);
                if(this->bendingEnergyWeight>0)
                    printf(" - (wBE)%.2e", bestWBE);
                if(this->linearEnergyWeight0>0 || this->linearEnergyWeight1>0 || this->linearEnergyWeight2>0)
                    printf(" - (wLE)%.2e", bestWLE);
                if(this->jacobianLogWeight>0)
                    printf(" - (wJAC)%.2e", bestWJac);
                printf(" [+ %g mm]\n", addedStep);
#ifdef NDEBUG
            }
#endif
        }

        // FINAL FOLDING CORRECTION
        if(this->jacobianLogWeight>0 && this->jacobianLogApproximation==true)
            this->ComputeJacobianBasedPenaltyTerm(2); // 20 iterations without approximation

        // SOME CLEANING IS PERFORMED
        this->ClearWarped();
        this->ClearDeformationField();
        this->ClearWarpedGradient();
        this->ClearVoxelBasedMeasureGradient();
        this->ClearNodeBasedMeasureGradient();
        this->ClearConjugateGradientVariables();
        this->ClearBestControlPointArray();
        this->ClearJointHistogram();
        if(this->usePyramid){
            nifti_image_free(this->referencePyramid[level]);this->referencePyramid[level]=NULL;
            nifti_image_free(this->floatingPyramid[level]);this->floatingPyramid[level]=NULL;
            free(this->maskPyramid[level]);this->maskPyramid[level]=NULL;
        }
        else if(level==this->levelToPerform-1){
            nifti_image_free(this->referencePyramid[0]);this->referencePyramid[0]=NULL;
            nifti_image_free(this->floatingPyramid[0]);this->floatingPyramid[0]=NULL;
            free(this->maskPyramid[0]);this->maskPyramid[0]=NULL;
        }

        this->ClearCurrentInputImage();

#ifdef NDEBUG
        if(this->verbose){
#endif
            printf("[%s] Current registration level done\n", this->executableName);
            printf("[%s] --------------------------------------------------\n", this->executableName);
#ifdef NDEBUG
        }
#endif

    } // level this->levelToPerform

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::Run_f3d() done\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image *reg_f3d<T>::GetWarpedImage()
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
    this->currentMask=NULL;

    reg_f3d<T>::AllocateWarped();
    reg_f3d<T>::AllocateDeformationField();

    reg_f3d<T>::WarpFloatingImage(3); // cubic spline interpolation

    reg_f3d<T>::ClearDeformationField();

    nifti_image *resultImage = nifti_copy_nim_info(this->warped);
    resultImage->data=(void *)malloc(resultImage->nvox*resultImage->nbyper);
    memcpy(resultImage->data, this->warped->data, resultImage->nvox*resultImage->nbyper);

    reg_f3d<T>::ClearWarped();
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image * reg_f3d<T>::GetControlPointPositionImage()
{
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->controlPointGrid);
    returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
    memcpy(returnedControlPointGrid->data, this->controlPointGrid->data,
           returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
    return returnedControlPointGrid;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d<T>::CheckStoppingCriteria(bool convergence)
{
    if(convergence) return 1;
    if(this->currentIteration>=this->maxiterationNumber) return 1;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
