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
#ifdef _NR_DEV

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
    this->intermediateDeformationField=NULL;
    this->jacobianMatrices=NULL;
    this->approxComp=false;
    this->stepNumber=6;
    this->useSymmetry=false;

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
int reg_f3d2<T>::SetCompositionStepNumber(int s)
{
    this->stepNumber = s;
    return 0;

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::UseSimilaritySymmetry()
{
    this->useSymmetry=true;
    return 0;

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::ApproximateComposition()
{
    this->approxComp=true;
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
    else this->stepNumber=(int)this->controlPointGrid->du;

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

    // Allocate the intermediate deformation field images
    this->intermediateDeformationField = (nifti_image **)malloc(this->stepNumber*sizeof(nifti_image *));
    for(int i=0; i<this->stepNumber; ++i){
        this->intermediateDeformationField[i]=nifti_copy_nim_info(this->deformationFieldImage);
        this->intermediateDeformationField[i]->data=(void *)malloc(this->intermediateDeformationField[i]->nvox *
                                                                          this->intermediateDeformationField[i]->nbyper);
    }

    // Allocate the intermediate Jacobian image
    this->jacobianMatrices=nifti_copy_nim_info(this->deformationFieldImage);
    if(this->jacobianMatrices->nz>1)
        this->jacobianMatrices->dim[5]=this->jacobianMatrices->nu=9;
    else this->jacobianMatrices->dim[5]=this->jacobianMatrices->nu=4;
   this->jacobianMatrices->nvox= this->jacobianMatrices->nx *
                                 this->jacobianMatrices->ny *
                                 this->jacobianMatrices->nz *
                                 this->jacobianMatrices->nt *
                                 this->jacobianMatrices->nu;
   this->jacobianMatrices->data=(void *)malloc(this->jacobianMatrices->nvox *
                                               this->jacobianMatrices->nbyper);

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::ClearCurrentInputImage()
{
    if(this->intermediateDeformationField!=NULL){
        for(int i=0;i<this->stepNumber;++i){
            if(this->intermediateDeformationField[i]!=NULL)
                nifti_image_free(this->intermediateDeformationField[i]);
            this->intermediateDeformationField[i]=NULL;
        }
        free(this->intermediateDeformationField);
    }
    this->intermediateDeformationField=NULL;

    if(this->jacobianMatrices!=NULL){
        nifti_image_free(this->jacobianMatrices);
    }
    this->jacobianMatrices=NULL;

    reg_f3d<T>::ClearCurrentInputImage();

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d2<T>::GetDeformationField()
{
#ifndef NDEBUG
    if(this->approxComp==true)
        printf("[NiftyReg DEBUG] Velocity integration performed with approximation\n");
    else printf("[NiftyReg DEBUG] Velocity integration performed without approximation\n");
#endif

    reg_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                            this->deformationFieldImage,
                                            NULL, // intermediate
                                            this->approxComp
                                            );
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
int reg_f3d2<T>::GetVoxelBasedGradient()
{
    /* COMPUTE THE GRADIENT BETWEEN THE REFERENCE AND THE WARPED FLOATING */
    reg_f3d<T>::GetVoxelBasedGradient();    

//{
//// modulate gradient by the jacobian determiant
//nifti_image *jacobianMap = nifti_copy_nim_info(this->currentReference);
//jacobianMap->data=(void *)malloc(jacobianMap->nvox*jacobianMap->nbyper);
//reg_bspline_GetJacobianMapFromVelocityField(this->controlPointGrid,
//                                            jacobianMap);
//T *jacPtr=static_cast<T *>(jacobianMap->data);
//T *gxPtr=static_cast<T *>(this->voxelBasedMeasureGradientImage->data);
//T *gyPtr=&(gxPtr[jacobianMap->nvox]);
//for(unsigned int i=0; i<jacobianMap->nvox; ++i){
//    gxPtr[i] *= jacPtr[i];
//    gyPtr[i] *= jacPtr[i];
//}
//nifti_set_filenames(jacobianMap, "forJacDet.nii", 0, 0);
//nifti_image_write(jacobianMap);
//nifti_image_free(jacobianMap);
//}
    // All the inverse intermediate deformation fields are computed
    reg_getInverseDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                   this->deformationFieldImage,
                                                   this->intermediateDeformationField,
                                                   false // No approximation here
                                                   );

    nifti_image *tempGradientImage=NULL;
    if(this->useSymmetry){
        /* COMPUTE THE GRADIENT BETWEEN THE FLOATING AND THE WARPED REFERENCE */
        tempGradientImage=nifti_copy_nim_info(this->warpedGradientImage);
        tempGradientImage->data = (void *)malloc(tempGradientImage->nvox*tempGradientImage->nbyper);

        // reference and floating are swapped and the inverse deformation field is used
        reg_resampleSourceImage(this->currentFloating,
                                this->currentReference,
                                this->warped,
                                this->deformationFieldImage,
                                this->currentMask,
                                USE_LINEAR_INTERPOLATION,
                                this->warpedPaddingValue);

        // The intensity gradient is first computed
        // reference and floating are swapped and the inverse deformation field is used
        reg_getSourceImageGradient(this->currentFloating,
                                   this->currentReference,
                                   this->warpedGradientImage,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   USE_LINEAR_INTERPOLATION);

        if(this->useSSD){
            // Compute the voxel based SSD gradient
            T localMaxSSD=this->maxSSD[0];
            if(this->usePyramid) localMaxSSD=this->maxSSD[this->currentLevel];
            // reference and floating are swapped
            reg_getVoxelBasedSSDGradient(this->currentFloating,
                                         this->warped,
                                         this->warpedGradientImage,
                                         tempGradientImage,
                                         localMaxSSD,
                                         this->currentMask
                                         );
        }
        else{
            // reference and floating are swapped
            // Fill the joint histogram reversed
            reg_getEntropies(this->currentFloating,
                             this->warped,
                             //2,
                             this->floatingBinNumber,
                             this->referenceBinNumber,
                             this->probaJointHistogram,
                             this->logJointHistogram,
                             this->entropies,
                             this->currentMask);
            // Compute the voxel based NMI gradient
            // reference and floating are swapped
            reg_getVoxelBasedNMIGradientUsingPW(this->currentFloating,
                                                this->warped,
                                                //2,
                                                this->warpedGradientImage,
                                                this->floatingBinNumber,
                                                this->referenceBinNumber,
                                                this->logJointHistogram,
                                                this->entropies,
                                                tempGradientImage,
                                                this->currentMask);
        }
        reg_tools_addSubMulDivValue(tempGradientImage,tempGradientImage,-1,2);
//{
//// modulate gradient by the jacobian determiant
//nifti_image *negatedVel = nifti_copy_nim_info(this->controlPointGrid);
//negatedVel->data=(void *)malloc(negatedVel->nvox*negatedVel->nbyper);
//memcpy(negatedVel->data,this->controlPointGrid->data,negatedVel->nvox*negatedVel->nbyper);
//reg_getDisplacementFromDeformation(negatedVel);
//reg_tools_addSubMulDivValue(negatedVel,negatedVel,-1,2);
//reg_getDeformationFromDisplacement(negatedVel);
//nifti_image *jacobianMap = nifti_copy_nim_info(this->currentReference);
//jacobianMap->data=(void *)malloc(jacobianMap->nvox*jacobianMap->nbyper);
//reg_bspline_GetJacobianMapFromVelocityField(negatedVel,
//                                            jacobianMap);
//T *jacPtr=static_cast<T *>(jacobianMap->data);
//T *gxPtr=static_cast<T *>(tempGradientImage->data);
//T *gyPtr=&(gxPtr[jacobianMap->nvox]);
//for(unsigned int i=0; i<jacobianMap->nvox; ++i){
//    gxPtr[i] *= jacPtr[i];
//    gyPtr[i] *= jacPtr[i];
//}
//nifti_set_filenames(jacobianMap, "bckJacDet.nii", 0, 0);
//nifti_image_write(jacobianMap);
//nifti_image_free(negatedVel);
//nifti_image_free(jacobianMap);
//}
    }


    /* EXPONENTIATE THE R/F(T) GRADIENT */
    for(int i=0; i<this->stepNumber-1; ++i){

        // The jacobian matrices are computed from the deformation field
        reg_defField_getJacobianMatrix(this->intermediateDeformationField[i],
                                       this->jacobianMatrices);

        // The gradient is re-oriented
        reg_resampleImageGradient(this->voxelBasedMeasureGradientImage,
                                  this->warpedGradientImage,
                                  this->intermediateDeformationField[i],
                                  this->jacobianMatrices,
                                  this->currentMask,
                                  USE_LINEAR_INTERPOLATION);

        // The reoriented gradient is added to the previous gradient
        reg_tools_addSubMulDivImages(this->voxelBasedMeasureGradientImage, this->warpedGradientImage, this->voxelBasedMeasureGradientImage, 0);
    }

    if(this->useSymmetry){
        /* EXPONENTIATE THE R(T-1)/F GRADIENT */
        reg_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                       this->deformationFieldImage,
                                                       this->intermediateDeformationField,
                                                       false // No approximation here
                                                       );

        for(int i=0; i<this->stepNumber-1; ++i){

            // The jacobian matrices are computed from the deformation field
            reg_defField_getJacobianMap(this->intermediateDeformationField[i],
                                        this->jacobianMatrices);

            // The gradient is re-oriented
            reg_resampleImageGradient(tempGradientImage,
                                      this->warpedGradientImage,
                                      this->intermediateDeformationField[i],
                                      this->jacobianMatrices,
                                      this->currentMask,
                                      USE_LINEAR_INTERPOLATION);

            // The reoriented gradient is added to the previous gradient
            reg_tools_addSubMulDivImages(tempGradientImage, this->warpedGradientImage, tempGradientImage, 0);
        }

        // Sum up both gradient
        reg_tools_addSubMulDivImages(this->voxelBasedMeasureGradientImage, tempGradientImage, this->voxelBasedMeasureGradientImage, 0); // addition
        nifti_image_free(tempGradientImage);
    }

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
int reg_f3d2<T>::CheckStoppingCriteria(bool convergence)
{
    if(convergence){
        if(this->approxComp==true){
            this->approxComp=false;
#ifdef NDEBUG
            if(this->verbose)
#endif
                printf("[%s] Squaring is now performed without approximation\n",
                       this->executableName);
        }
        else return 1;
    }
    else{
        if(this->approxComp==true){
            if( this->currentIteration>=(this->maxiterationNumber-(float)this->maxiterationNumber*0.1f) ){
                this->approxComp=false;
#ifdef NDEBUG
                if(this->verbose)
#endif
                printf("[%s] Squaring is now performed without approximation\n",
                       this->executableName);
            }
        }
        else{
            if(this->currentIteration>=this->maxiterationNumber) return 1;
        }
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
#endif //_NR_DEV
