/**
 * @file _reg_base.cpp
 * @author Marc Modat
 * @date 15/11/2012
 *
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BASE_CPP
#define _REG_BASE_CPP

#include "_reg_base.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_base<T>::reg_base(int refTimePoint,int floTimePoint)
{
    this->optimiser=NULL;
    this->maxiterationNumber=300;
    this->optimiseX=true;
    this->optimiseY=true;
    this->optimiseZ=true;
    this->perturbationNumber=0;

    this->executableName=(char *)"NiftyReg BASE";
    this->referenceTimePoint=refTimePoint;
    this->floatingTimePoint=floTimePoint;
    this->inputReference=NULL; // pointer to external
    this->inputFloating=NULL; // pointer to external
    this->maskImage=NULL; // pointer to external
    this->affineTransformation=NULL;  // pointer to external
    this->referenceMask=NULL;
    this->referenceSmoothingSigma=0.;
    this->floatingSmoothingSigma=0.;
    this->referenceThresholdUp=new float[this->referenceTimePoint];
    this->referenceThresholdLow=new float[this->referenceTimePoint];
    this->floatingThresholdUp=new float[this->floatingTimePoint];
    this->floatingThresholdLow=new float[this->floatingTimePoint];
    for(int i=0; i<this->referenceTimePoint; i++){
        this->referenceThresholdUp[i]=std::numeric_limits<T>::max();
        this->referenceThresholdLow[i]=-std::numeric_limits<T>::max();
    }
    for(int i=0; i<this->floatingTimePoint; i++){
        this->floatingThresholdUp[i]=std::numeric_limits<T>::max();
        this->floatingThresholdLow[i]=-std::numeric_limits<T>::max();
    }
    this->warpedPaddingValue=std::numeric_limits<T>::quiet_NaN();
    this->levelNumber=3;
    this->levelToPerform=0;
    this->gradientSmoothingSigma=0;
    this->verbose=true;
    this->useConjGradient=true;
    this->useApproxGradient=false;
    this->usePyramid=true;

    this->useSSD=false;
    this->useKLD=false;
    this->useLNCC = std::numeric_limits<T>::quiet_NaN();

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

    this->interpolation=1;

    this->funcProgressCallback=NULL;
    this->paramsProgressCallback=NULL;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_base constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_base<T>::~reg_base()
{
    this->ClearWarped();
    this->ClearWarpedGradient();
    this->ClearDeformationField();
    this->ClearVoxelBasedMeasureGradient();
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
    if(this->floatingThresholdUp!=NULL){delete []this->floatingThresholdUp;this->floatingThresholdUp=NULL;}
    if(this->floatingThresholdLow!=NULL){delete []this->floatingThresholdLow;this->floatingThresholdLow=NULL;}
    if(this->activeVoxelNumber!=NULL){delete []this->activeVoxelNumber;this->activeVoxelNumber=NULL;}
    if(this->optimiser!=NULL){delete this->optimiser;this->optimiser=NULL;}
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_base destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceImage(nifti_image *r)
{
    this->inputReference = r;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingImage(nifti_image *f)
{
    this->inputFloating = f;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetMaximalIterationNumber(unsigned int dance)
{
    this->maxiterationNumber=dance;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceMask(nifti_image *m)
{
    this->maskImage = m;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetAffineTransformation(mat44 *a)
{
    this->affineTransformation=a;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceSmoothingSigma(T s)
{
    this->referenceSmoothingSigma = s;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingSmoothingSigma(T s)
{
    this->floatingSmoothingSigma = s;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceThresholdUp(unsigned int i, T t)
{
    this->referenceThresholdUp[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceThresholdLow(unsigned int i, T t)
{
    this->referenceThresholdLow[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingThresholdUp(unsigned int i, T t)
{
    this->floatingThresholdUp[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingThresholdLow(unsigned int i, T t)
{
    this->floatingThresholdLow[i] = t;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetWarpedPaddingValue(T p)
{
    this->warpedPaddingValue = p;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetLevelNumber(unsigned int l)
{
    this->levelNumber = l;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetLevelToPerform(unsigned int l)
{
    this->levelToPerform = l;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetGradientSmoothingSigma(T g)
{
    this->gradientSmoothingSigma = g;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseConjugateGradient()
{
    this->useConjGradient = true;
    this->useApproxGradient = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseConjugateGradient()
{
    this->useConjGradient = false;
    this->useApproxGradient = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseApproximatedGradient()
{
    this->useConjGradient = false;
    this->useApproxGradient = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseApproximatedGradient()
{
    this->useConjGradient = true;
    this->useApproxGradient = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::PrintOutInformation()
{
    this->verbose = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotPrintOutInformation()
{
    this->verbose = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUsePyramidalApproach()
{
    this->usePyramid=false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseNeareatNeighborInterpolation()
{
    this->interpolation=0;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseLinearInterpolation()
{
    this->interpolation=1;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseCubicSplineInterpolation()
{
    this->interpolation=3;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearCurrentInputImage()
{
    this->currentReference=NULL;
    this->currentMask=NULL;
    this->currentFloating=NULL;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateWarped()
{
    if(this->currentReference==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The reference image is not defined\n");
        exit(1);
    }
    reg_base<T>::ClearWarped();
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
void reg_base<T>::ClearWarped()
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
void reg_base<T>::AllocateDeformationField()
{
    if(this->currentReference==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The reference image is not defined\n");
        exit(1);
    }
    reg_base<T>::ClearDeformationField();
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
    this->deformationFieldImage->nbyper = sizeof(T);
    if(sizeof(T)==sizeof(float))
        this->deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
    else this->deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
    this->deformationFieldImage->data = (void *)calloc(this->deformationFieldImage->nvox,
                                                       this->deformationFieldImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearDeformationField()
{
    if(this->deformationFieldImage!=NULL){
        nifti_image_free(this->deformationFieldImage);
        this->deformationFieldImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateWarpedGradient()
{
    if(this->deformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The deformation field image is not defined\n");
        exit(1);
    }
    reg_base<T>::ClearWarpedGradient();
    this->warpedGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
    this->warpedGradientImage->dim[0]=this->warpedGradientImage->ndim=5;
    this->warpedGradientImage->nt = this->warpedGradientImage->dim[4] = this->currentFloating->nt;
    this->warpedGradientImage->nvox =	this->warpedGradientImage->nx *
            this->warpedGradientImage->ny *
            this->warpedGradientImage->nz *
            this->warpedGradientImage->nt *
            this->warpedGradientImage->nu;
    this->warpedGradientImage->data = (void *)calloc(this->warpedGradientImage->nvox,
                                                     this->warpedGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearWarpedGradient()
{
    if(this->warpedGradientImage!=NULL){
        nifti_image_free(this->warpedGradientImage);
        this->warpedGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateVoxelBasedMeasureGradient()
{
    if(this->deformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The deformation field image is not defined\n");
        exit(1);
    }
    reg_base<T>::ClearVoxelBasedMeasureGradient();
    this->voxelBasedMeasureGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
    this->voxelBasedMeasureGradientImage->data = (void *)calloc(this->voxelBasedMeasureGradientImage->nvox,
                                                                this->voxelBasedMeasureGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearVoxelBasedMeasureGradient()
{
    if(this->voxelBasedMeasureGradientImage!=NULL){
        nifti_image_free(this->voxelBasedMeasureGradientImage);
        this->voxelBasedMeasureGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::CheckParameters()
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

    // CHECK THE MASK DIMENSION IF IT IS DEFINED
    if(this->maskImage!=NULL){
        if(this->inputReference->nx != maskImage->nx ||
                this->inputReference->ny != maskImage->ny ||
                this->inputReference->nz != maskImage->nz){
            fprintf(stderr,"[NiftyReg ERROR] The mask image has different x, y or z dimension than the reference image.\n");
            exit(1);
        }
    }

    // CHECK THE NUMBER OF LEVEL TO PERFORM
    if(this->levelToPerform>0){
        this->levelToPerform=this->levelToPerform<this->levelNumber?this->levelToPerform:this->levelNumber;
    }
    else this->levelToPerform=this->levelNumber;
    if(this->levelToPerform==0 || this->levelToPerform>this->levelNumber)
        this->levelToPerform=this->levelNumber;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_base::CheckParameters() done\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::Initisalise()
{
    if(this->initialised) return;

    this->CheckParameters();

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

    unsigned int pyramidalLevelNumber=1;
    if(this->usePyramid) pyramidalLevelNumber=this->levelToPerform;

    // SMOOTH THE INPUT IMAGES IF REQUIRED
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

    if(this->useSSD || this->useKLD || this->useLNCC==this->useLNCC){
        // THRESHOLD THE INPUT IMAGES IF REQUIRED
        this->maxSSD=new T[pyramidalLevelNumber];
        for(unsigned int l=0; l<pyramidalLevelNumber; l++){
            reg_thresholdImage<T>(this->referencePyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
            reg_thresholdImage<T>(this->floatingPyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
        }
        // The maximal difference image is extracted for normalisation of the SSD
        if(this->useSSD){
            this->maxSSD=new T[pyramidalLevelNumber];
            for(unsigned int l=0; l<pyramidalLevelNumber; l++){
                T tempMaxSSD1 = (this->referencePyramid[l]->cal_min - this->floatingPyramid[l]->cal_max) *
                        (this->referencePyramid[l]->cal_min - this->floatingPyramid[l]->cal_max);
                T tempMaxSSD2 = (this->referencePyramid[l]->cal_max - this->floatingPyramid[l]->cal_min) *
                        (this->referencePyramid[l]->cal_max - this->floatingPyramid[l]->cal_min);
                this->maxSSD[l]=tempMaxSSD1>tempMaxSSD2?tempMaxSSD1:tempMaxSSD2;
            }
        }
    }
    else{
        // RESCALE THE INPUT IMAGE INTENSITY TO USE WITH NMI
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

    this->initialised=true;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_base::Initialise() done\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::SetOptimiser()
{
    if(this->useConjGradient)
        this->optimiser=new reg_conjugateGradient<T>();
    else this->optimiser=new reg_optimiser<T>();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_base<T>::ComputeSimilarityMeasure()
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
    else if(this->useLNCC==this->useLNCC){
        measure = reg_getLNCC(this->currentReference,
                              this->warped,
                              this->useLNCC,
                              this->currentMask);
    }
    else{
        // Use additive NMI when the flag is set and we have multi channel input
        if(this->currentReference->nt>1 &&
                this->currentReference->nt == this->warped->nt && additive_mc_nmi){

            fprintf(stderr, "WARNING: Modification for Jorge - reg_base<T>::ComputeSimilarityMeasure()\n");

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
void reg_base<T>::GetVoxelBasedGradient()
{
    // The intensity gradient is first computed
    reg_getImageGradient(this->currentFloating,
                         this->warpedGradientImage,
                         this->deformationFieldImage,
                         this->currentMask,
                         this->interpolation,
                         this->warpedPaddingValue);

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
    else if(this->useLNCC==this->useLNCC){
        reg_getVoxelBasedLNCCGradient(this->currentReference,
                                      this->warped,
                                      this->warpedGradientImage,
                                      this->voxelBasedMeasureGradientImage,
                                      this->useLNCC,
                                      this->currentMask
                                      );
    }
    else{
        // Use additive NMI when the flag is set and we have multi channel input
        if(this->currentReference->nt>1 &&
                this->currentReference->nt == this->warped->nt &&
                additive_mc_nmi){

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
template<class T>
void reg_base<T>::SetReferenceBinNumber(int l, unsigned int v)
{
    this->referenceBinNumber[l] = v;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingBinNumber(int l, unsigned int v)
{
    this->floatingBinNumber[l] = v;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::ApproximateParzenWindow()
{
    this->approxParzenWindow = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotApproximateParzenWindow()
{
    this->approxParzenWindow = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseSSD()
{
    this->useSSD = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseSSD()
{
    this->useSSD = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseKLDivergence()
{
    this->useKLD = true;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseKLDivergence()
{
    this->useKLD = false;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseLNCC(T stdev)
{
    this->useLNCC = stdev;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseLNCC()
{
    this->useLNCC = std::numeric_limits<T>::quiet_NaN();
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::WarpFloatingImage(int inter)
{
    // Compute the deformation field
    this->GetDeformationField();
    // Resample the floating image
    reg_resampleImage(this->currentFloating,
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
void reg_base<T>::AllocateJointHistogram()
{
    reg_base<T>::ClearJointHistogram();
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
void reg_base<T>::ClearJointHistogram()
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
template <class T>
void reg_base<T>::Run()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] %s::Run() called\n", this->executableName);
#endif

    if(!this->initialised) this->Initisalise();

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
        if(!this->useSSD && !this->useKLD && this->useLNCC!=this->useLNCC)
			this->AllocateJointHistogram();

        // The grid is refined if necessary
        T maxStepSize=this->InitialiseCurrentLevel();
        T currentSize = maxStepSize;
        T smallestSize = maxStepSize / (T)100.0;

        this->DisplayCurrentLevelParameters();

        // ALLOCATE IMAGES THAT ARE REQUIRED TO COMPUTE THE GRADIENT
        this->AllocateVoxelBasedMeasureGradient();
        this->AllocateTransformationGradient();

        // initialise the optimiser
        this->SetOptimiser();

        // Loop over the number of perturbation to do
        for(size_t perturbation=0;perturbation<=this->perturbationNumber;++perturbation){

            // Evalulate the objective function value
            this->UpdateBestObjFunctionValue();
            this->PrintInitialObjFunctionValue();

            while(true){

                if(currentSize==0)
                    break;

                if(this->optimiser->GetCurrentIterationNumber()>=this->optimiser->GetMaxIterationNumber())
                    break;

                // Compute the objective function gradient
                this->GetObjectiveFunctionGradient();

                // Normalise the gradient
                this->NormaliseGradient();

                // Initialise the line search initial step size
                currentSize=currentSize>maxStepSize?maxStepSize:currentSize;

                // A line search is performed
                this->optimiser->Optimise(maxStepSize,smallestSize,currentSize);

				// Update the obecjtive function variables and print some information
                this->PrintCurrentObjFunctionValue(currentSize);

                // Monitoring progression when f3d is ran as a library
                if(currentSize==0.f){
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
        this->CorrectTransformation();

        // SOME CLEANING IS PERFORMED
        delete this->optimiser;
        this->optimiser=NULL;

        this->ClearWarped();
        this->ClearDeformationField();
        this->ClearWarpedGradient();
        this->ClearVoxelBasedMeasureGradient();
        this->ClearTransformationGradient();
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
    printf("[NiftyReg DEBUG] %s::Run() done\n", this->executableName);
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif // _REG_BASE_CPP
