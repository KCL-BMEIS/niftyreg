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
    this->optimiser=NULL;
#ifdef _USE_CUDA
    this->optimiser_gpu=NULL;
#endif
    this->maxiterationNumber=300;
    this->optimiseX=true;
    this->optimiseY=true;
    this->optimiseZ=true;
    this->perturbationNumber=0;

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
    this->bendingEnergyWeight=0.005;
    this->linearEnergyWeight0=0.;
    this->linearEnergyWeight1=0.;
    this->L2NormWeight=0.;
    this->jacobianLogWeight=0.;
    this->jacobianLogApproximation=true;
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
    this->useKLD=false;
    this->additive_mc_nmi = false;
    this->useConjGradient=true;
    this->maxSSD=NULL;
    this->entropies[0]=this->entropies[1]=this->entropies[2]=this->entropies[3]=0.;
    this->approxParzenWindow=true;
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
    this->nodeBasedGradientImage=NULL;
    this->probaJointHistogram=NULL;
    this->logJointHistogram=NULL;

    this->interpolation=1;

    this->gridRefinement=true;

    this->funcProgressCallback=NULL;
    this->paramsProgressCallback=NULL;

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
    this->ClearJointHistogram();
    this->ClearNodeBasedGradient();
    this->ClearVoxelBasedMeasureGradient();
    if(this->controlPointGrid!=NULL){
        nifti_image_free(this->controlPointGrid);
        this->controlPointGrid=NULL;
    }
    if(this->referencePyramid!=NULL){
        if(this->usePyramid){
            for(unsigned int i=0;i<this->levelToPerform;i++){
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
            for(unsigned int i=0;i<this->levelToPerform;i++){
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
            for(unsigned int i=0;i<this->levelToPerform;i++){
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
    if(this->optimiser!=NULL){delete this->optimiser;this->optimiser=NULL;}
#ifdef _USE_CUDA
    if(this->optimiser_gpu!=NULL){delete this->optimiser_gpu;this->optimiser_gpu=NULL;}
#endif
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetReferenceImage(nifti_image *r)
{
    this->inputReference = r;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetFloatingImage(nifti_image *f)
{
    this->inputFloating = f;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetMaximalIterationNumber(unsigned int dance)
{
    this->maxiterationNumber=dance;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetReferenceBinNumber(int l, unsigned int v)
{
    this->referenceBinNumber[l] = v;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetFloatingBinNumber(int l, unsigned int v)
{
    this->floatingBinNumber[l] = v;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetControlPointGridImage(nifti_image *cp)
{
    this->inputControlPointGrid = cp;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetReferenceMask(nifti_image *m)
{
    this->maskImage = m;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetAffineTransformation(mat44 *a)
{
    this->affineTransformation=a;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetBendingEnergyWeight(T be)
{
    this->bendingEnergyWeight = be;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetLinearEnergyWeights(T w0, T w1)
{
    this->linearEnergyWeight0=w0;
    this->linearEnergyWeight1=w1;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetL2NormDisplacementWeight(T w)
{
    this->L2NormWeight=w;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetJacobianLogWeight(T j)
{
    this->jacobianLogWeight = j;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::ApproximateParzenWindow()
{
    this->approxParzenWindow = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotApproximateParzenWindow()
{
    this->approxParzenWindow = false;
    return;
}/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::ApproximateJacobianLog()
{
    this->jacobianLogApproximation = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotApproximateJacobianLog()
{
    this->jacobianLogApproximation = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetReferenceSmoothingSigma(T s)
{
    this->referenceSmoothingSigma = s;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetFloatingSmoothingSigma(T s)
{
    this->floatingSmoothingSigma = s;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetReferenceThresholdUp(unsigned int i, T t)
{
    this->referenceThresholdUp[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetReferenceThresholdLow(unsigned int i, T t)
{
    this->referenceThresholdLow[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetFloatingThresholdUp(unsigned int i, T t)
{
    this->floatingThresholdUp[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetFloatingThresholdLow(unsigned int i, T t)
{
    this->floatingThresholdLow[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetWarpedPaddingValue(T p)
{
    this->warpedPaddingValue = p;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetSpacing(unsigned int i, T s)
{
    this->spacing[i] = s;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetLevelNumber(unsigned int l)
{
    this->levelNumber = l;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetLevelToPerform(unsigned int l)
{
    this->levelToPerform = l;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetGradientSmoothingSigma(T g)
{
    this->gradientSmoothingSigma = g;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UseSSD()
{
    this->useSSD = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotUseSSD()
{
    this->useSSD = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UseKLDivergence()
{
    this->useKLD = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotUseKLDivergence()
{
    this->useKLD = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UseConjugateGradient()
{
    this->useConjGradient = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotUseConjugateGradient()
{
    this->useConjGradient = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::PrintOutInformation()
{
    this->verbose = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotPrintOutInformation()
{
    this->verbose = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
//template<class T>
//void reg_f3d<T>::SetThreadNumber(int t)
//{
//	this->threadNumber = t;
//	return 0;
//}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotUsePyramidalApproach()
{
    this->usePyramid=false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UseNeareatNeighborInterpolation()
{
    this->interpolation=0;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UseLinearInterpolation()
{
    this->interpolation=1;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UseCubicSplineInterpolation()
{
    this->interpolation=3;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateCurrentInputImage()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d<T>::AllocateCurrentInputImage called.\n");
#endif
    if(this->currentLevel!=0 && this->gridRefinement==true)
        reg_bspline_refineControlPointGrid(this->currentReference, this->controlPointGrid);

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d<T>::AllocateCurrentInputImage done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearCurrentInputImage()
{
    this->currentReference=NULL;
    this->currentMask=NULL;
    this->currentFloating=NULL;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateWarped()
{
    if(this->currentReference==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The reference image is not defined\n");
        exit(1);
    }
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
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearWarped()
{
    if(this->warped!=NULL){
        nifti_image_free(this->warped);
        this->warped=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateDeformationField()
{
    if(this->currentReference==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The reference image is not defined\n");
        exit(1);
    }
    if(this->controlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The control point image is not defined\n");
        exit(1);
    }
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
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearDeformationField()
{
    if(this->deformationFieldImage!=NULL){
        nifti_image_free(this->deformationFieldImage);
        this->deformationFieldImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateWarpedGradient()
{
    if(this->deformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The deformation field image is not defined\n");
        exit(1);
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
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearWarpedGradient()
{
    if(this->warpedGradientImage!=NULL){
        nifti_image_free(this->warpedGradientImage);
        this->warpedGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateVoxelBasedMeasureGradient()
{
    if(this->deformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The deformation field image is not defined\n");
        exit(1);
    }
    reg_f3d<T>::ClearVoxelBasedMeasureGradient();
    this->voxelBasedMeasureGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
    this->voxelBasedMeasureGradientImage->data = (void *)calloc(this->voxelBasedMeasureGradientImage->nvox,
                                                                this->voxelBasedMeasureGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearVoxelBasedMeasureGradient()
{
    if(this->voxelBasedMeasureGradientImage!=NULL){
        nifti_image_free(this->voxelBasedMeasureGradientImage);
        this->voxelBasedMeasureGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateNodeBasedGradient()
{
    if(this->controlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The control point image is not defined\n");
        exit(1);
    }
    reg_f3d<T>::ClearNodeBasedGradient();
    this->nodeBasedGradientImage = nifti_copy_nim_info(this->controlPointGrid);
    this->nodeBasedGradientImage->data = (void *)calloc(this->nodeBasedGradientImage->nvox,
                                                        this->nodeBasedGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearNodeBasedGradient()
{
    if(this->nodeBasedGradientImage!=NULL){
        nifti_image_free(this->nodeBasedGradientImage);
        this->nodeBasedGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateJointHistogram()
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
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearJointHistogram()
{
    if(this->probaJointHistogram!=NULL){
        free(this->probaJointHistogram);
        this->probaJointHistogram=NULL;
    }
    if(this->logJointHistogram!=NULL){
        free(this->logJointHistogram);
        this->logJointHistogram=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::CheckParameters_f3d()
{
    // CHECK THAT BOTH INPUT IMAGES ARE DEFINED
    if(this->inputReference==NULL){
        fprintf(stderr,"[NiftyReg ERROR] No reference image has been defined.\n");
        exit(1);
    }
    if(this->inputFloating==NULL){
        fprintf(stderr,"[NiftyReg ERROR] No floating image has been defined.\n");
        exit(1);
    }

    if(this->useSSD){
        if(inputReference->nt!=inputFloating->nt){
            fprintf(stderr,"[NiftyReg ERROR] SSD is only available with reference and floating images with same dimension along the t-axis.\n");
            exit(1);
        }
    }

    // CHECK THE MASK DIMENSION IF IT IS DEFINED
    if(this->maskImage!=NULL){
        if(this->inputReference->nx != maskImage->nx ||
                this->inputReference->ny != maskImage->ny ||
                this->inputReference->nz != maskImage->nz){
            fprintf(stderr,"* The mask image has different x, y or z dimension than the reference image.\n");
            exit(1);
        }
    }

    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if(this->levelToPerform>0){
        this->levelToPerform=this->levelToPerform<this->levelNumber?this->levelToPerform:this->levelNumber;
    }
    else this->levelToPerform=this->levelNumber;

    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
    if(strcmp(this->executableName,"NiftyReg F3D")==0){
        T penaltySum=this->bendingEnergyWeight +
                this->linearEnergyWeight0 +
                this->linearEnergyWeight1 +
                this->L2NormWeight +
                this->jacobianLogWeight;
        if(penaltySum>=1.0){
            this->similarityWeight=0;
            this->similarityWeight /= penaltySum;
            this->bendingEnergyWeight /= penaltySum;
            this->linearEnergyWeight0 /= penaltySum;
            this->linearEnergyWeight1 /= penaltySum;
            this->L2NormWeight /= penaltySum;
            this->jacobianLogWeight /= penaltySum;
        }
        else this->similarityWeight=1.0 - penaltySum;
    }
    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if(this->levelToPerform==0 || this->levelToPerform>this->levelNumber)
        this->levelToPerform=this->levelNumber;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::CheckParameters_f3d() done\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::Initisalise_f3d()
{
    if(this->initialised) return;

    this->CheckParameters_f3d();

    // CREATE THE PYRAMIDE IMAGES
    if(this->usePyramid){
        this->referencePyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
        this->floatingPyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
        this->maskPyramid = (int **)malloc(this->levelToPerform*sizeof(int *));
        this->activeVoxelNumber= (int *)malloc(this->levelToPerform*sizeof(int));
    }
    else{
        this->referencePyramid = (nifti_image **)malloc(sizeof(nifti_image *));
        this->floatingPyramid = (nifti_image **)malloc(sizeof(nifti_image *));
        this->maskPyramid = (int **)malloc(sizeof(int *));
        this->activeVoxelNumber= (int *)malloc(sizeof(int));
    }

    // FINEST LEVEL OF REGISTRATION
    if(this->usePyramid){
        reg_createImagePyramid<T>(this->inputReference, this->referencePyramid, this->levelNumber, this->levelToPerform);
        reg_createImagePyramid<T>(this->inputFloating, this->floatingPyramid, this->levelNumber, this->levelToPerform);
        if (this->maskImage!=NULL)
            reg_createMaskPyramid<T>(this->maskImage, this->maskPyramid, this->levelNumber, this->levelToPerform, this->activeVoxelNumber);
        else{
            for(unsigned int l=0;l<this->levelToPerform;++l){
                this->activeVoxelNumber[l]=this->referencePyramid[l]->nx*this->referencePyramid[l]->ny*this->referencePyramid[l]->nz;
                this->maskPyramid[l]=(int *)calloc(activeVoxelNumber[l],sizeof(int));
            }
        }
    }
    else{
        reg_createImagePyramid<T>(this->inputReference, this->referencePyramid, 1, 1);
        reg_createImagePyramid<T>(this->inputFloating, this->floatingPyramid, 1, 1);
        if (this->maskImage!=NULL)
            reg_createMaskPyramid<T>(this->maskImage, this->maskPyramid, 1, 1, this->activeVoxelNumber);
        else{
            this->activeVoxelNumber[0]=this->referencePyramid[0]->nx*this->referencePyramid[0]->ny*this->referencePyramid[0]->nz;
            this->maskPyramid[0]=(int *)calloc(activeVoxelNumber[0],sizeof(int));
        }
    }

    // SMOOTH THE INPUT IMAGES IF REQUIRED
    unsigned int pyramidalLevelNumber=1;
    if(this->usePyramid) pyramidalLevelNumber=this->levelToPerform;
    for(unsigned int l=0; l<pyramidalLevelNumber; l++){
        if(this->referenceSmoothingSigma!=0.0){
            bool smoothAxis[8]={false,true,true,true,false,false,false,false};
            reg_gaussianSmoothing<T>(this->referencePyramid[l], this->referenceSmoothingSigma, smoothAxis);
        }
        if(this->floatingSmoothingSigma!=0.0){
            bool smoothAxis[8]={false,true,true,true,false,false,false,false};
            reg_gaussianSmoothing<T>(this->floatingPyramid[l], this->floatingSmoothingSigma, smoothAxis);
        }
    }

    if(this->useSSD || this->useKLD){
        // THRESHOLD THE INPUT IMAGES IF REQUIRED
        this->maxSSD=new T[pyramidalLevelNumber];
        for(unsigned int l=0; l<pyramidalLevelNumber; l++){
            reg_thresholdImage<T>(referencePyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
            reg_thresholdImage<T>(floatingPyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
        }
        // The maximal difference image is extracted for normalisation of the SSD
        if(this->useSSD){
            this->maxSSD=new T[pyramidalLevelNumber];
            for(unsigned int l=0; l<pyramidalLevelNumber; l++){
                T tempMaxSSD1 = (referencePyramid[l]->cal_min - floatingPyramid[l]->cal_max) *
                        (referencePyramid[l]->cal_min - floatingPyramid[l]->cal_max);
                T tempMaxSSD2 = (referencePyramid[l]->cal_max - floatingPyramid[l]->cal_min) *
                        (referencePyramid[l]->cal_max - floatingPyramid[l]->cal_min);
                this->maxSSD[l]=tempMaxSSD1>tempMaxSSD2?tempMaxSSD1:tempMaxSSD2;
            }
        }
    }
    else{
        // NMI is used here
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
        for(unsigned int l=0; l<pyramidalLevelNumber; l++){
            reg_intensityRescale(this->referencePyramid[l],
                                 referenceRescalingArrayDown,
                                 referenceRescalingArrayUp,
                                 this->referenceThresholdLow,
                                 this->referenceThresholdUp);
            reg_intensityRescale(this->floatingPyramid[l],
                                 floatingRescalingArrayDown,
                                 floatingRescalingArrayUp,
                                 this->floatingThresholdLow,
                                 this->floatingThresholdUp);
        }
    }

    // DETERMINE THE GRID SPACING AND CREATE THE GRID
    if(this->inputControlPointGrid==NULL){

        // Set the spacing along y and z if undefined. Their values are set to match
        // the spacing along the x axis
        if(this->spacing[1]!=this->spacing[1]) this->spacing[1]=this->spacing[0];
        if(this->spacing[2]!=this->spacing[2]) this->spacing[2]=this->spacing[0];

        /* Convert the spacing from voxel to mm if necessary */
        float spacingInMillimeter[3]={this->spacing[0],this->spacing[1],this->spacing[2]};
        if(this->usePyramid){
            if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dx;
            if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dy;
            if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dz;
        }
        else{
            if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->referencePyramid[0]->dx;
            if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->referencePyramid[0]->dy;
            if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->referencePyramid[0]->dz;
        }

        // Define the spacing for the first level
        float gridSpacing[3];
        gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(this->levelToPerform-1));
        gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(this->levelToPerform-1));
        gridSpacing[2] = 1.0f;
        if(this->referencePyramid[0]->nz>1)
            gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(this->levelToPerform-1));

        // Create and allocate the control point image
        reg_createControlPointGrid<T>(&this->controlPointGrid,
                                      this->referencePyramid[0],
                                      gridSpacing);

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
                exit(1);
        }
        else if(reg_bspline_initialiseControlPointGridWithAffine(this->affineTransformation, this->controlPointGrid))
            exit(1);
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
        // Print out some global information about the registration
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
        printf("[%s] Maximum iteration number per level: %i\n", this->executableName, (int)this->maxiterationNumber);
        printf("[%s]\n", this->executableName);
        printf("[%s] Final spacing in mm: %g %g %g\n", this->executableName,
               this->spacing[0], this->spacing[1], this->spacing[2]);
        printf("[%s]\n", this->executableName);
        if(this->useSSD)
            printf("[%s] The SSD is used as a similarity measure.\n", this->executableName);
        if(this->useKLD)
            printf("[%s] The KL divergence is used as a similarity measure.\n", this->executableName);
        else{
            printf("[%s] The NMI is used as a similarity measure.\n", this->executableName);
            if(this->approxParzenWindow || this->inputReference->nt>1 || this->inputFloating->nt>1)
                printf("[%s] The Parzen window joint histogram filling is approximated\n", this->executableName);
            else printf("[%s] The Parzen window joint histogram filling is not approximated\n", this->executableName);
        }
        printf("[%s] Similarity measure term weight: %g\n", this->executableName, this->similarityWeight);
        printf("[%s]\n", this->executableName);
        printf("[%s] Bending energy penalty term weight: %g\n", this->executableName, this->bendingEnergyWeight);
        printf("[%s]\n", this->executableName);
        printf("[%s] Linear energy penalty term weights: %g %g\n", this->executableName,
               this->linearEnergyWeight0, this->linearEnergyWeight1);
        printf("[%s]\n", this->executableName);
        printf("[%s] L2 norm of the displacement penalty term weights: %g\n", this->executableName,
               this->L2NormWeight);
        printf("[%s]\n", this->executableName);
        printf("[%s] Jacobian-based penalty term weight: %g\n", this->executableName, this->jacobianLogWeight);
        if(this->jacobianLogWeight>0){
            if(this->jacobianLogApproximation) printf("[%s] \t* Jacobian-based penalty term is approximated\n",
                                                      this->executableName);
            else printf("[%s] \t* Jacobian-based penalty term is not approximated\n", this->executableName);
        }
#ifdef NDEBUG
    }
#endif

    this->initialised=true;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::Initialise_f3d() done\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetDeformationField()
{
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   false, //composition
                                   true // bspline
                                   );
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::WarpFloatingImage(int inter)
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
    return;
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
                              NULL,
                              this->currentMask);
        if(this->usePyramid)
            measure /= this->maxSSD[this->currentLevel];
        else measure /= this->maxSSD[0];
    }
    else if(this->useKLD){
        measure = -reg_getKLDivergence(this->currentReference,
                                       this->warped,
                                       NULL,
                                       this->currentMask);
    }
    else{
        // Use additive NMI when the flag is set and we have multi channel input
        if(this->currentReference->nt>1 &&
                this->currentReference->nt == this->warped->nt && additive_mc_nmi){

            fprintf(stderr, "WARNING: Modification for Jorge - reg_f3d<T>::ComputeSimilarityMeasure()\n");

            T *referencePtr=static_cast<T *>(this->currentReference->data);
            T *warpedPtr=static_cast<T *>(this->warped->data);

            measure=0.;
            for(int t=0;t<this->currentReference->nt;++t){

                nifti_image *temp_referenceImage = nifti_copy_nim_info(this->currentReference);
                temp_referenceImage->dim[0]=temp_referenceImage->ndim=3;
                temp_referenceImage->dim[4]=temp_referenceImage->nt=1;
                temp_referenceImage->nvox=
                        temp_referenceImage->nx*
                        temp_referenceImage->ny*
                        temp_referenceImage->nz;
                temp_referenceImage->data=(void *)malloc(temp_referenceImage->nvox*temp_referenceImage->nbyper);
                T *tempRefPtr=static_cast<T *>(temp_referenceImage->data);
                memcpy(tempRefPtr, &referencePtr[t*temp_referenceImage->nvox],
                       temp_referenceImage->nvox*temp_referenceImage->nbyper);

                nifti_image *temp_warpedImage = nifti_copy_nim_info(this->warped);
                temp_warpedImage->dim[0]=temp_warpedImage->ndim=3;
                temp_warpedImage->dim[4]=temp_warpedImage->nt=1;
                temp_warpedImage->nvox=
                        temp_warpedImage->nx*
                        temp_warpedImage->ny*
                        temp_warpedImage->nz;
                temp_warpedImage->data=(void *)malloc(temp_warpedImage->nvox*temp_warpedImage->nbyper);
                T *tempWarPtr=static_cast<T *>(temp_warpedImage->data);
                memcpy(tempWarPtr, &warpedPtr[t*temp_warpedImage->nvox],
                       temp_warpedImage->nvox*temp_warpedImage->nbyper);

                reg_getEntropies(temp_referenceImage,
                                 temp_warpedImage,
                                 this->referenceBinNumber,
                                 this->floatingBinNumber,
                                 this->probaJointHistogram,
                                 this->logJointHistogram,
                                 this->entropies,
                                 this->currentMask,
                                 this->approxParzenWindow);
                measure += (this->entropies[0]+this->entropies[1])/this->entropies[2];

                nifti_image_free(temp_referenceImage);
                nifti_image_free(temp_warpedImage);
            }
            measure /= (double)(this->currentReference->nt);
        }
        else {
            reg_getEntropies(this->currentReference,
                             this->warped,
                             this->referenceBinNumber,
                             this->floatingBinNumber,
                             this->probaJointHistogram,
                             this->logJointHistogram,
                             this->entropies,
                             this->currentMask,
                             this->approxParzenWindow);
            measure = (this->entropies[0]+this->entropies[1])/this->entropies[2];
        }
    }
    return double(this->similarityWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
    if(this->jacobianLogWeight<=0) return 0.;

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
            this->optimiser->RestoreBestDOF();
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
    if(this->bendingEnergyWeight<=0) return 0.;

    double value = reg_bspline_bendingEnergy(this->controlPointGrid);
    return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm()
{
    if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0)
        return 0.;

    double values_le[2]={0.,0.};
    reg_bspline_linearEnergy(this->controlPointGrid, values_le);

    return this->linearEnergyWeight0*values_le[0] +
            this->linearEnergyWeight1*values_le[1];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeL2NormDispPenaltyTerm()
{
    if(this->L2NormWeight<=0)
        return 0.;

    double values_l2=reg_bspline_L2norm_displacement(this->controlPointGrid);

    return (double)this->L2NormWeight*values_l2;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetVoxelBasedGradient()
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
                                     NULL,
                                     localMaxSSD,
                                     this->currentMask
                                     );
    }
    else if(this->useKLD){
        // Compute the voxel based KL divergence gradient
        reg_getKLDivergenceVoxelBasedGradient(this->currentReference,
                                              this->warped,
                                              this->warpedGradientImage,
                                              this->voxelBasedMeasureGradientImage,
                                              NULL,
                                              this->currentMask
                                              );
    }
    else{
        // Use additive NMI when the flag is set and we have multi channel input
        if(this->currentReference->nt>1 &&
                this->currentReference->nt == this->warped->nt && additive_mc_nmi){

            fprintf(stderr, "WARNING: Modification for Jorge - reg_f3d<T>::GetVoxelBasedGradient()\n");

            T *referencePtr=static_cast<T *>(this->currentReference->data);
            T *warpedPtr=static_cast<T *>(this->currentFloating->data);
            T *gradientPtr=static_cast<T *>(this->warpedGradientImage->data);

            reg_tools_addSubMulDivValue(this->voxelBasedMeasureGradientImage,
                                        this->voxelBasedMeasureGradientImage,
                                        0.f,2);

            for(int t=0;t<this->currentReference->nt;++t){

                nifti_image *temp_referenceImage = nifti_copy_nim_info(this->currentReference);
                temp_referenceImage->dim[0]=temp_referenceImage->ndim=3;
                temp_referenceImage->dim[4]=temp_referenceImage->nt=1;
                temp_referenceImage->nvox=
                        temp_referenceImage->nx*
                        temp_referenceImage->ny*
                        temp_referenceImage->nz;
                temp_referenceImage->data=(void *)malloc(temp_referenceImage->nvox*temp_referenceImage->nbyper);
                T *tempRefPtr=static_cast<T *>(temp_referenceImage->data);
                memcpy(tempRefPtr, &referencePtr[t*temp_referenceImage->nvox],
                       temp_referenceImage->nvox*temp_referenceImage->nbyper);

                nifti_image *temp_warpedImage = nifti_copy_nim_info(this->warped);
                temp_warpedImage->dim[0]=temp_warpedImage->ndim=3;
                temp_warpedImage->dim[4]=temp_warpedImage->nt=1;
                temp_warpedImage->nvox=
                        temp_warpedImage->nx*
                        temp_warpedImage->ny*
                        temp_warpedImage->nz;
                temp_warpedImage->data=(void *)malloc(temp_warpedImage->nvox*temp_warpedImage->nbyper);
                T *tempWarPtr=static_cast<T *>(temp_warpedImage->data);
                memcpy(tempWarPtr, &warpedPtr[t*temp_warpedImage->nvox],
                       temp_warpedImage->nvox*temp_warpedImage->nbyper);

                nifti_image *temp_gradientImage = nifti_copy_nim_info(this->warpedGradientImage);
                temp_gradientImage->dim[4]=temp_gradientImage->nt=1;
                temp_gradientImage->nvox=
                        temp_gradientImage->nx*
                        temp_gradientImage->ny*
                        temp_gradientImage->nz*
                        temp_gradientImage->nt*
                        temp_gradientImage->nu;
                temp_gradientImage->data=(void *)malloc(temp_gradientImage->nvox*temp_gradientImage->nbyper);
                T *tempGraPtr=static_cast<T *>(temp_gradientImage->data);
                for(int u=0;u<temp_gradientImage->nu;++u){
                    size_t index=(u*this->warpedGradientImage->nt+t)*temp_referenceImage->nvox;
                    memcpy(&tempGraPtr[u*temp_referenceImage->nvox],
                           &gradientPtr[index],
                           temp_referenceImage->nvox*temp_referenceImage->nbyper);
                }

                reg_getEntropies(temp_referenceImage,
                                 temp_warpedImage,
                                 this->referenceBinNumber,
                                 this->floatingBinNumber,
                                 this->probaJointHistogram,
                                 this->logJointHistogram,
                                 this->entropies,
                                 this->currentMask,
                                 this->approxParzenWindow);

                nifti_image *temp_nmiGradientImage = nifti_copy_nim_info(this->voxelBasedMeasureGradientImage);
                temp_nmiGradientImage->data=(void *)malloc(temp_nmiGradientImage->nvox*temp_nmiGradientImage->nbyper);

                reg_getVoxelBasedNMIGradientUsingPW(temp_referenceImage,
                                                    temp_warpedImage,
                                                    temp_gradientImage,
                                                    this->referenceBinNumber,
                                                    this->floatingBinNumber,
                                                    this->logJointHistogram,
                                                    this->entropies,
                                                    temp_nmiGradientImage,
                                                    this->currentMask,
                                                    this->approxParzenWindow);

                reg_tools_addSubMulDivImages(temp_nmiGradientImage,
                                             this->voxelBasedMeasureGradientImage,
                                             this->voxelBasedMeasureGradientImage,0);

                nifti_image_free(temp_referenceImage);
                nifti_image_free(temp_warpedImage);
                nifti_image_free(temp_gradientImage);
                nifti_image_free(temp_nmiGradientImage);
            }
            reg_tools_addSubMulDivValue(this->voxelBasedMeasureGradientImage,
                                        this->voxelBasedMeasureGradientImage,
                                        (float)(this->currentReference->nt),3);
        }
        else{
            // Compute the voxel based NMI gradient
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
        }
    }
    return;
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetSimilarityMeasureGradient()
{
    this->GetVoxelBasedGradient();

    // The voxel based NMI gradient is convolved with a spline kernel
    int smoothingRadius[3];
    smoothingRadius[0] = (int)( 2.0*this->controlPointGrid->dx/this->currentReference->dx );
    smoothingRadius[1] = (int)( 2.0*this->controlPointGrid->dy/this->currentReference->dy );
    smoothingRadius[2] = (int)( 2.0*this->controlPointGrid->dz/this->currentReference->dz );
    reg_tools_CubicSplineKernelConvolution<T>(this->voxelBasedMeasureGradientImage,
                                              smoothingRadius);

    // The node based NMI gradient is extracted
    reg_voxelCentric2NodeCentric(this->nodeBasedGradientImage,
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
        T *gradientValuesX = static_cast<T *>(this->nodeBasedGradientImage->data);
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
        T *gradientValuesX = static_cast<T *>(this->nodeBasedGradientImage->data);
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

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetBendingEnergyGradient()
{
    if(this->bendingEnergyWeight<=0) return;

    reg_bspline_bendingEnergyGradient(this->controlPointGrid,
                                      this->currentReference,
                                      this->nodeBasedGradientImage,
                                      this->bendingEnergyWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetLinearEnergyGradient()
{
    if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return;

    reg_bspline_linearEnergyGradient(this->controlPointGrid,
                                     this->currentReference,
                                     this->nodeBasedGradientImage,
                                     this->linearEnergyWeight0,
                                     this->linearEnergyWeight1);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetL2NormDispGradient()
{
    if(this->L2NormWeight<=0) return;

    reg_bspline_L2norm_dispGradient(this->controlPointGrid,
                                    this->currentReference,
                                    this->nodeBasedGradientImage,
                                    this->L2NormWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetJacobianBasedGradient()
{
    if(this->jacobianLogWeight<=0) return;

    reg_bspline_jacobianDeterminantGradient(this->controlPointGrid,
                                            this->currentReference,
                                            this->nodeBasedGradientImage,
                                            this->jacobianLogWeight,
                                            this->jacobianLogApproximation);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::SetGradientImageToZero()
{
    T* nodeGradPtr = static_cast<T *>(this->nodeBasedGradientImage->data);
    for(unsigned int i=0; i<this->nodeBasedGradientImage->nvox; ++i)
        *nodeGradPtr++=0;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::DisplayCurrentLevelParameters()
{
#ifdef NDEBUG
    if(this->verbose){
#endif
        printf("[%s] **************************************************\n", this->executableName);
        printf("[%s] Current level: %i / %i\n", this->executableName, this->currentLevel+1, this->levelNumber);
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
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::GetObjectiveFunctionValue()
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

    // Store the global objective function value
    return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWL2 - this->currentWJac;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::UpdateParameters(T scale)
{
    T *currentDOF=this->optimiser->GetCurrentDOF();
    T *bestDOF=this->optimiser->GetBestDOF();
    T *gradient=this->optimiser->GetGradient();

    // Update the control point position
    if(this->optimiser->GetOptimiseX()==true &&
       this->optimiser->GetOptimiseY()==true &&
       this->optimiser->GetOptimiseZ()==true)
    {
        // Update the values for all axis displacement
        for(size_t i=0;i<this->optimiser->GetDOFNumber();++i){
            currentDOF[i] = bestDOF[i] + scale * gradient[i];
        }
    }
    else
    {
        size_t voxNumber = this->optimiser->GetVoxNumber();
        // Update the values for the x-axis displacement
        if(this->optimiser->GetOptimiseX()==true){
            for(size_t i=0;i<voxNumber;++i){
                currentDOF[i] = bestDOF[i] + scale * gradient[i];
            }
        }
        // Update the values for the y-axis displacement
        if(this->optimiser->GetOptimiseY()==true){
            T *currentDOFY=&currentDOF[voxNumber];
            T *bestDOFY=&bestDOF[voxNumber];
            T *gradientY=&gradient[voxNumber];
            for(size_t i=0;i<voxNumber;++i){
                currentDOFY[i] = bestDOFY[i] + scale * gradientY[i];
            }
        }
        // Update the values for the z-axis displacement
        if(this->optimiser->GetOptimiseZ()==true && this->optimiser->GetNDim()>2){
            T *currentDOFZ=&currentDOF[2*voxNumber];
            T *bestDOFZ=&bestDOF[2*voxNumber];
            T *gradientZ=&gradient[2*voxNumber];
            for(size_t i=0;i<voxNumber;++i){
                currentDOFZ[i] = bestDOFZ[i] + scale * gradientZ[i];
            }
        }
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::SetOptimiser()
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
                                0, // currentIterationNumber,
                                this,
                                static_cast<T *>(this->controlPointGrid->data),
                                static_cast<T *>(this->nodeBasedGradientImage->data)
                                );
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::Run_f3d()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] %s::Run_f3d() called\n", this->executableName);
#endif

    if(!this->initialised) this->Initisalise_f3d();

    // Compute the resolution of the progress bar
    float iProgressStep=1, nProgressSteps;
    nProgressSteps = this->levelToPerform*this->maxiterationNumber;

    for(this->currentLevel=0;
        this->currentLevel<this->levelToPerform;
        this->currentLevel++){

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
        this->AllocateJointHistogram();

        // The grid is refined if necessary
        this->AllocateCurrentInputImage();

        this->DisplayCurrentLevelParameters();

        T maxStepSize = (this->currentReference->dx>this->currentReference->dy)?this->currentReference->dx:this->currentReference->dy;
        maxStepSize = (this->currentReference->dz>maxStepSize)?this->currentReference->dz:maxStepSize;
        T currentSize = maxStepSize;
        T smallestSize = maxStepSize / 100.0f;

        // ALLOCATE IMAGES THAT ARE REQUIRED TO COMPUTE THE GRADIENT
        this->AllocateVoxelBasedMeasureGradient();
        this->AllocateNodeBasedGradient();

        // initialise the optimiser
        this->SetOptimiser();

        // Loop over the number of perturbation to do
        for(size_t perturbation=0;perturbation<=this->perturbationNumber;++perturbation){

            // Evalulate the objective function value
            this->bestWJac=this->currentWJac;
            this->bestWBE=this->currentWBE;
            this->bestWLE=this->currentWLE;
            this->bestWL2=this->currentWL2;
            this->bestWMeasure=this->currentWMeasure;
            this->bestIC=this->currentIC;

            double bestValue;
#ifdef _USE_CUDA
                if(this->optimiser_gpu!=NULL)
                    bestValue=this->optimiser_gpu->GetBestObjFunctionValue();
                else
#endif
                    bestValue=this->optimiser->GetBestObjFunctionValue();

#ifdef NDEBUG
            if(this->verbose){
#endif
                if(this->useSSD)
                    printf("[%s] Initial objective function: %g = (wSSD)%g - (wBE)%g - (wLE)%g - (wL2)%g - (wJAC)%g\n",
                           this->executableName, bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWL2, this->bestWJac);
                else if(this->useKLD)
                    printf("[%s] Initial objective function: %g = (wKLD)%g - (wBE)%g - (wLE)%g - (wL2)%g - (wJAC)%g\n",
                           this->executableName, bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWL2, this->bestWJac);
                else printf("[%s] Initial objective function: %g = (wNMI)%g - (wBE)%g - (wLE)%g - (wL2)%g - (wJAC)%g\n",
                            this->executableName, bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWL2, this->bestWJac);
                if(this->bestIC!=0)
                    printf("[%s] Initial Inverse consistency value: %g\n", this->executableName, this->bestIC);
#ifdef NDEBUG
            }
#endif

            while(true){

                if(currentSize==0)
                    break;

#ifdef _USE_CUDA
                if(this->optimiser_gpu!=NULL){
                    if(this->optimiser_gpu->GetCurrentIterationNumber()>=this->optimiser_gpu->GetMaxIterationNumber())
                        break;
                }
                else
#endif
                {
                    if(this->optimiser->GetCurrentIterationNumber()>=this->optimiser->GetMaxIterationNumber())
                        break;
                }

                // Compute the gradient of the similarity measure
                if(this->similarityWeight>0){
                    this->WarpFloatingImage(this->interpolation);
                    this->ComputeSimilarityMeasure();
                    this->GetSimilarityMeasureGradient();
                }
                else{
                    this->SetGradientImageToZero();
                }
#ifdef _USE_CUDA
                if(this->optimiser_gpu!=NULL)
                    this->optimiser_gpu->IncrementCurrentIterationNumber();
                else
#endif
                this->optimiser->IncrementCurrentIterationNumber();

                // The gradient is smoothed using a Gaussian kernel if it is required
                if(this->gradientSmoothingSigma!=0){
                    reg_gaussianSmoothing<T>(this->nodeBasedGradientImage,
                                             fabs(this->gradientSmoothingSigma),
                                             NULL);
                }

                // Compute the penalty term gradients if required
                this->GetBendingEnergyGradient();
                this->GetJacobianBasedGradient();
                this->GetLinearEnergyGradient();
                this->GetL2NormDispGradient();
                this->GetInverseConsistencyGradient();

                // Initialise the line search initial step size
                currentSize=currentSize>maxStepSize?maxStepSize:currentSize;
//                currentSize=maxStepSize;

                // A line search is performed
#ifdef _USE_CUDA
                if(this->optimiser_gpu!=NULL)
                    this->optimiser_gpu->Optimise(maxStepSize,smallestSize,currentSize);
                else
#endif
                this->optimiser->Optimise(maxStepSize,smallestSize,currentSize);

#ifdef NDEBUG
                if(this->verbose){
#endif
                    this->bestWMeasure=this->currentWMeasure;
                    this->bestWBE=this->currentWBE;
                    this->bestWLE=this->currentWLE;
                    this->bestWL2=this->currentWL2;
                    this->bestWJac=this->currentWJac;
                    this->bestIC=this->currentIC;
#ifdef _USE_CUDA
                    if(this->optimiser_gpu!=NULL)
                        printf("[%s] [%i] Current objective function: %g",
                               this->executableName,
                               (int)this->optimiser_gpu->GetCurrentIterationNumber(),
                               this->optimiser_gpu->GetBestObjFunctionValue());
                    else
#endif
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
#ifdef NDEBUG
                }
#endif

                // Monitoring progression when f3d is ran as a library
                if(currentSize==0.f){
#ifdef _USE_CUDA
                    if(this->optimiser_gpu!=NULL)
                        iProgressStep += this->optimiser_gpu->GetMaxIterationNumber() - 1 - this->optimiser_gpu->GetCurrentIterationNumber();
                    else
#endif
                        iProgressStep += this->optimiser->GetMaxIterationNumber() - 1 - this->optimiser->GetCurrentIterationNumber();
                    if(funcProgressCallback && paramsProgressCallback)
                    {
                        (*funcProgressCallback)(100.*iProgressStep/nProgressSteps,
                                                paramsProgressCallback);
                    }
                    break;
                }
                else{
                    iProgressStep++;
                    if(funcProgressCallback && paramsProgressCallback){
                        (*funcProgressCallback)(100.*iProgressStep/nProgressSteps,
                                                paramsProgressCallback);
                    }
                }
            } // while
            if(perturbation<this->perturbationNumber){
#ifdef _USE_CUDA
                if(this->optimiser_gpu!=NULL)
                    this->optimiser_gpu->Perturbation(smallestSize);
                else
#endif
                this->optimiser->Perturbation(smallestSize);
                currentSize=maxStepSize;
#ifdef NDEBUG
                if(this->verbose){
#endif
                    printf("[%s] Perturbation Step - The number of iteration is reset to 0\n",
                           this->executableName);
                    printf("[%s] Perturbation Step - Every control point positions is altered by [-%g %g]\n",
                           this->executableName,
                           smallestSize,
                           smallestSize);

#ifdef NDEBUG
                }
#endif
            }
        } // perturbation loop

        // FINAL FOLDING CORRECTION
        if(this->jacobianLogWeight>0 && this->jacobianLogApproximation==true)
            this->ComputeJacobianBasedPenaltyTerm(2); // 20 iterations without approximation

        // SOME CLEANING IS PERFORMED
#ifdef _USE_CUDA
        if(this->optimiser_gpu!=NULL){
            delete this->optimiser_gpu;
            this->optimiser_gpu=NULL;
        }
        else
#endif
        {
            delete this->optimiser;
            this->optimiser=NULL;
        }
        this->ClearWarped();
        this->ClearDeformationField();
        this->ClearWarpedGradient();
        this->ClearVoxelBasedMeasureGradient();
        this->ClearNodeBasedGradient();
        this->ClearJointHistogram();
        if(this->usePyramid){
            nifti_image_free(this->referencePyramid[this->currentLevel]);this->referencePyramid[this->currentLevel]=NULL;
            nifti_image_free(this->floatingPyramid[this->currentLevel]);this->floatingPyramid[this->currentLevel]=NULL;
            free(this->maskPyramid[this->currentLevel]);this->maskPyramid[this->currentLevel]=NULL;
        }
        else if(this->currentLevel==this->levelToPerform-1){
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

    if ( funcProgressCallback && paramsProgressCallback ) 
    {
        (*funcProgressCallback)( 100., paramsProgressCallback);
    }

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d::Run_f3d() done\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image **reg_f3d<T>::GetWarpedImage()
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

    nifti_image **resultImage= (nifti_image **)malloc(2*sizeof(nifti_image *));
    resultImage[0]=nifti_copy_nim_info(this->warped);
    resultImage[0]->cal_min=this->inputFloating->cal_min;
    resultImage[0]->cal_max=this->inputFloating->cal_max;
    resultImage[0]->scl_slope=this->inputFloating->scl_slope;
    resultImage[0]->scl_inter=this->inputFloating->scl_inter;
    resultImage[0]->data=(void *)malloc(resultImage[0]->nvox*resultImage[0]->nbyper);
    memcpy(resultImage[0]->data, this->warped->data, resultImage[0]->nvox*resultImage[0]->nbyper);

    resultImage[1]=NULL;

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

#endif
