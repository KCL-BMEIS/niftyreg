/*
 *  _reg_f3d_gpu.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_GPU_CPP
#define _REG_F3D_GPU_CPP

#include "_reg_f3d_gpu.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_gpu<T>::reg_f3d_gpu(int refTimePoint,int floTimePoint)
    :reg_f3d<T>::reg_f3d(refTimePoint,floTimePoint)
{
    this->currentReference_gpu=NULL;
    this->currentFloating_gpu=NULL;
    this->currentMask_gpu=NULL;
    this->warped_gpu=NULL;
    this->controlPointGrid_gpu=NULL;
    this->deformationFieldImage_gpu=NULL;
    this->warpedGradientImage_gpu=NULL;
    this->voxelBasedMeasureGradientImage_gpu=NULL;
    this->nodeBasedMeasureGradientImage_gpu=NULL;
    this->conjugateG_gpu=NULL;
    this->conjugateH_gpu=NULL;
    this->bestControlPointPosition_gpu=NULL;
    this->logJointHistogram_gpu=NULL;

    this->currentReference2_gpu=NULL;
    this->currentFloating2_gpu=NULL;
    this->warped2_gpu=NULL;
    this->warpedGradientImage2_gpu=NULL;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu constructor called\n");
#endif
                                                                            }
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_gpu<T>::~reg_f3d_gpu()
{
    if(this->currentReference_gpu!=NULL)
        cudaCommon_free<float>(&this->currentReference_gpu);
    if(this->currentFloating_gpu!=NULL)
        cudaCommon_free(&this->currentFloating_gpu);
    if(this->currentMask_gpu!=NULL)
        cudaCommon_free<int>(&this->currentMask_gpu);
    if(this->warped_gpu!=NULL)
        cudaCommon_free<float>(&this->warped_gpu);
    if(this->controlPointGrid_gpu!=NULL)
        cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    if(this->deformationFieldImage_gpu!=NULL)
        cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
    if(this->warpedGradientImage_gpu!=NULL)
        cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
    if(this->voxelBasedMeasureGradientImage_gpu!=NULL)
        cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
    if(this->nodeBasedMeasureGradientImage_gpu!=NULL)
        cudaCommon_free<float4>(&this->nodeBasedMeasureGradientImage_gpu);
    if(this->conjugateG_gpu!=NULL)
        cudaCommon_free<float4>(&this->conjugateG_gpu);
    if(this->conjugateH_gpu!=NULL)
        cudaCommon_free<float4>(&this->conjugateH_gpu);
    if(this->bestControlPointPosition_gpu!=NULL)
        cudaCommon_free<float4>(&this->bestControlPointPosition_gpu);
    if(this->logJointHistogram_gpu!=NULL)
        cudaCommon_free<float>(&this->logJointHistogram_gpu);

    if(this->currentReference2_gpu!=NULL)
        cudaCommon_free<float>(&this->currentReference2_gpu);
    if(this->currentFloating2_gpu!=NULL)
        cudaCommon_free(&this->currentFloating2_gpu);
    if(this->warped2_gpu!=NULL)
        cudaCommon_free<float>(&this->warped2_gpu);
    if(this->warpedGradientImage2_gpu!=NULL)
        cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateWarped()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarped called.\n");
#endif
    if(this->currentReference==NULL)
        return 1;
    this->ClearWarped();
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
    CUDA_SAFE_CALL(cudaMallocHost(&(this->warped->data), this->warped->nvox*this->warped->nbyper));
    if(this->warped->nt==1){
        if(cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, this->warped->dim)) return 1;
    }
    else if(this->warped->nt==2){
        if(cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, &this->warped2_gpu, this->warped->dim)) return 1;
    }
    else{
        printf("[NiftyReg ERROR] reg_f3d_gpu does not handle more than 2 time points in the floating image.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarped done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearWarped()
{
    if(this->warped!=NULL){
        CUDA_SAFE_CALL(cudaFreeHost(this->warped->data));
        this->warped->data = NULL;
        nifti_image_free(this->warped);
        this->warped=NULL;
    }
    if(this->warped_gpu!=NULL){
        cudaCommon_free<float>(&this->warped_gpu);
        this->warped_gpu=NULL;
    }
    if(this->warped2_gpu!=NULL){
        cudaCommon_free<float>(&this->warped2_gpu);
        this->warped2_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateDeformationField()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateDeformationField called.\n");
#endif
    this->ClearDeformationField();
    CUDA_SAFE_CALL(cudaMalloc(&this->deformationFieldImage_gpu, this->activeVoxelNumber[this->currentLevel]*sizeof(float4)));

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateDeformationField done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearDeformationField()
{
    if(this->deformationFieldImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
        this->deformationFieldImage_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateWarpedGradient()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarpedGradient called.\n");
#endif
    this->ClearWarpedGradient();
    if(this->inputFloating->nt==1){
        CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu, this->activeVoxelNumber[this->currentLevel]*sizeof(float4)));
    }
    else if(this->inputFloating->nt==2){
        CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu, this->activeVoxelNumber[this->currentLevel]*sizeof(float4)));
        CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage2_gpu, this->activeVoxelNumber[this->currentLevel]*sizeof(float4)));
    }
    else{
        printf("[NiftyReg ERROR] reg_f3d_gpu does not handle more than 2 time points in the floating image.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarpedGradient done.\n");
#endif

    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearWarpedGradient()
{
    if(this->warpedGradientImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
        this->warpedGradientImage_gpu=NULL;
    }
    if(this->warpedGradientImage2_gpu!=NULL){
        cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);
        this->warpedGradientImage2_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateVoxelBasedMeasureGradient()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateVoxelBasedMeasureGradient called.\n");
#endif
    this->ClearVoxelBasedMeasureGradient();
    if(cudaCommon_allocateArrayToDevice(&this->voxelBasedMeasureGradientImage_gpu, this->currentReference->dim)) return 1;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateVoxelBasedMeasureGradient done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearVoxelBasedMeasureGradient()
{
    if(this->voxelBasedMeasureGradientImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
        this->voxelBasedMeasureGradientImage_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateNodeBasedMeasureGradient()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateNodeBasedMeasureGradient called.\n");
#endif
    this->ClearNodeBasedMeasureGradient();
    if(cudaCommon_allocateArrayToDevice(&this->nodeBasedMeasureGradientImage_gpu, this->controlPointGrid->dim)) return 1;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateNodeBasedMeasureGradient done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearNodeBasedMeasureGradient()
{
    if(this->nodeBasedMeasureGradientImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->nodeBasedMeasureGradientImage_gpu);
        this->nodeBasedMeasureGradientImage_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateConjugateGradientVariables()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateConjugateGradientVariables called.\n");
#endif
    if(this->controlPointGrid==NULL)
        return 1;
    this->ClearConjugateGradientVariables();
    if(cudaCommon_allocateArrayToDevice(&this->conjugateG_gpu, this->controlPointGrid->dim)) return 1;
    if(cudaCommon_allocateArrayToDevice(&this->conjugateH_gpu, this->controlPointGrid->dim)) return 1;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateConjugateGradientVariables done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearConjugateGradientVariables()
{
    if(this->conjugateG_gpu!=NULL){
        cudaCommon_free<float4>(&this->conjugateG_gpu);
        this->conjugateG_gpu=NULL;
    }
    if(this->conjugateH_gpu!=NULL){
        cudaCommon_free<float4>(&this->conjugateH_gpu);
        this->conjugateH_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateBestControlPointArray()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateBestControlPointArray called.\n");
#endif
    if(this->controlPointGrid==NULL)
        return 1;
    this->ClearBestControlPointArray();
    if(cudaCommon_allocateArrayToDevice(&this->bestControlPointPosition_gpu, this->controlPointGrid->dim)) return 1;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateBestControlPointArray done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearBestControlPointArray()
{
    cudaCommon_free<float4>(&this->bestControlPointPosition_gpu);
    this->bestControlPointPosition_gpu=NULL;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateJointHistogram()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateJointHistogram called.\n");
#endif
    this->ClearJointHistogram();
    reg_f3d<T>::AllocateJointHistogram();
    CUDA_SAFE_CALL(cudaMalloc(&this->logJointHistogram_gpu, this->totalBinNumber*sizeof(float)));
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateJointHistogram done.\n");
#endif
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearJointHistogram()
{
    reg_f3d<T>::ClearJointHistogram();
    if(this->logJointHistogram_gpu!=NULL){
        cudaCommon_free<float>(&this->logJointHistogram_gpu);
        this->logJointHistogram_gpu=NULL;
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::SaveCurrentControlPoint()
{
    CUDA_SAFE_CALL(cudaMemcpy(this->bestControlPointPosition_gpu, this->controlPointGrid_gpu,
                    this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz*sizeof(float4),
                    cudaMemcpyDeviceToDevice));
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::RestoreCurrentControlPoint()
{
    CUDA_SAFE_CALL(cudaMemcpy(this->controlPointGrid_gpu, this->bestControlPointPosition_gpu,
                    this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz*sizeof(float4),
                    cudaMemcpyDeviceToDevice));
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_gpu<T>::ComputeJacobianBasedPenaltyTerm(int type)
{

    double value;
    if(type==2){
        value = reg_bspline_ComputeJacobianPenaltyTerm_gpu(this->currentReference,
                                                           this->controlPointGrid,
                                                           &this->controlPointGrid_gpu,
                                                           false);
    }
    else{
        value = reg_bspline_ComputeJacobianPenaltyTerm_gpu(this->currentReference,
                                                           this->controlPointGrid,
                                                           &this->controlPointGrid_gpu,
                                                           this->jacobianLogApproximation);
    }
    unsigned int maxit=5;
    if(type>0) maxit=20;
    unsigned int it=0;
    while(value!=value && it<maxit){
        if(type==2){
            value = reg_bspline_correctFolding_gpu(this->currentReference,
                                                   this->controlPointGrid,
                                                   &this->controlPointGrid_gpu,
                                                   false);
        }
        else{
            value = reg_bspline_correctFolding_gpu(this->currentReference,
                                                   this->controlPointGrid,
                                                   &this->controlPointGrid_gpu,
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
                printf("[NiftyReg F3D] Folding correction, %i step(s)\n", it);
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
double reg_f3d_gpu<T>::ComputeBendingEnergyPenaltyTerm()
{
    double value = reg_bspline_ApproxBendingEnergy_gpu(this->controlPointGrid,
                                                       &this->controlPointGrid_gpu);
    return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::WarpFloatingImage(int inter)
{

    // Compute the deformation field
    reg_bspline_gpu(this->controlPointGrid,
                    this->currentReference,
                    &this->controlPointGrid_gpu,
                    &this->deformationFieldImage_gpu,
                    &this->currentMask_gpu,
                    this->activeVoxelNumber[this->currentLevel]);

    // Resample the floating image
    reg_resampleSourceImage_gpu(this->currentReference,
                                this->currentFloating,
                                &this->warped_gpu,
                                &this->currentFloating_gpu,
                                &this->deformationFieldImage_gpu,
                                &this->currentMask_gpu,
                                this->activeVoxelNumber[this->currentLevel],
                                this->warpedPaddingValue);
    if(this->currentFloating->nt==2){
        reg_resampleSourceImage_gpu(this->currentReference,
                                    this->currentFloating,
                                    &this->warped2_gpu,
                                    &this->currentFloating2_gpu,
                                    &this->deformationFieldImage_gpu,
                                    &this->currentMask_gpu,
                                    this->activeVoxelNumber[this->currentLevel],
                                    this->warpedPaddingValue);
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_gpu<T>::ComputeSimilarityMeasure()
{
    if(this->currentFloating->nt==1){
        if(cudaCommon_transferFromDeviceToNifti<float>(this->warped, &this->warped_gpu)) return 1;
    }
    else if(this->currentFloating->nt==2){
        if(cudaCommon_transferFromDeviceToNifti<float>(this->warped, &this->warped_gpu, &this->warped2_gpu)) return 1;
    }

    double measure=0.;
    if(this->currentFloating->nt==1){
        reg_getEntropies<double>(   this->currentReference,
                                    this->warped,
                                    2,
                                    this->referenceBinNumber,
                                    this->floatingBinNumber,
                                    this->probaJointHistogram,
                                    this->logJointHistogram,
                                    this->entropies,
                                    this->currentMask);
    }
    else if(this->currentFloating->nt==2){
        reg_getEntropies2x2_gpu(this->currentReference,
                                 this->warped,
                                 2,
                                 this->referenceBinNumber,
                                 this->floatingBinNumber,
                                 this->probaJointHistogram,
                                 this->logJointHistogram,
                                 &this->logJointHistogram_gpu,
                                 this->entropies,
                                 this->currentMask);
    }


    measure = double(this->entropies[0]+this->entropies[1])/double(this->entropies[2]);

    return double(1.0-this->bendingEnergyWeight-this->jacobianLogWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::GetSimilarityMeasureGradient()
{
    // The log joint jistogram is first transfered to the GPU
    float *tempB=NULL;
    CUDA_SAFE_CALL(cudaMallocHost(&tempB, this->totalBinNumber*sizeof(float)));
    for(unsigned int i=0; i<this->totalBinNumber;i++){
        tempB[i]=(float)this->logJointHistogram[i];
    }
    CUDA_SAFE_CALL(cudaMemcpy(this->logJointHistogram_gpu, tempB, this->totalBinNumber*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(tempB));

    // The intensity gradient is first computed
    reg_getSourceImageGradient_gpu(	this->currentReference,
                                    this->currentFloating,
                                    &this->currentFloating_gpu,
                                    &this->deformationFieldImage_gpu,
                                    &this->warpedGradientImage_gpu,
                                    this->activeVoxelNumber[this->currentLevel]);

    if(this->currentFloating->nt==2){
        reg_getSourceImageGradient_gpu(	this->currentReference,
                                        this->currentFloating,
                                        &this->currentFloating2_gpu,
                                        &this->deformationFieldImage_gpu,
                                        &this->warpedGradientImage2_gpu,
                                        this->activeVoxelNumber[this->currentLevel]);
    }

    // The voxel based NMI gradient
    if(this->currentFloating->nt==1){
        reg_getVoxelBasedNMIGradientUsingPW_gpu(this->currentReference,
                                                this->warped,
                                                &this->currentReference_gpu,
                                                &this->warped_gpu,
                                                &this->warpedGradientImage_gpu,
                                                &this->logJointHistogram_gpu,
                                                &this->voxelBasedMeasureGradientImage_gpu,
                                                &this->currentMask_gpu,
                                                this->activeVoxelNumber[this->currentLevel],
                                                this->entropies,
                                                this->referenceBinNumber[0],
                                                this->floatingBinNumber[0]);
    }
    else if(this->currentFloating->nt==2){
        reg_getVoxelBasedNMIGradientUsingPW2x2_gpu( this->currentReference,
                                                    this->warped,
                                                    &this->currentReference_gpu,
                                                    &this->currentReference2_gpu,
                                                    &this->warped_gpu,
                                                    &this->warped2_gpu,
                                                    &this->warpedGradientImage_gpu,
                                                    &this->warpedGradientImage2_gpu,
                                                    &this->logJointHistogram_gpu,
                                                    &this->voxelBasedMeasureGradientImage_gpu,
                                                    &this->currentMask_gpu,
                                                    this->activeVoxelNumber[this->currentLevel],
                                                    this->entropies,
                                                    this->referenceBinNumber,
                                                    this->floatingBinNumber);
    }

    // The voxel based gradient is smoothed
    int smoothingRadius[3];
    smoothingRadius[0] = (int)( 2.0*this->controlPointGrid->dx/this->currentReference->dx );
    smoothingRadius[1] = (int)( 2.0*this->controlPointGrid->dy/this->currentReference->dy );
    smoothingRadius[2] = (int)( 2.0*this->controlPointGrid->dz/this->currentReference->dz );
    reg_smoothImageForCubicSpline_gpu(  this->warped,
                                        &this->voxelBasedMeasureGradientImage_gpu,
                                        smoothingRadius);
    // The node gradient is extracted
    reg_voxelCentric2NodeCentric_gpu(   this->warped,
                                        this->controlPointGrid,
                                        &this->voxelBasedMeasureGradientImage_gpu,
                                        &this->nodeBasedMeasureGradientImage_gpu,
                                        1.0-this->bendingEnergyWeight-this->jacobianLogWeight);
    /* The NMI gradient is converted from voxel space to real space */
    mat44 *floatingMatrix_xyz=NULL;
    if(this->currentFloating->sform_code>0)
        floatingMatrix_xyz = &(this->currentFloating->sto_xyz);
    else floatingMatrix_xyz = &(this->currentFloating->qto_xyz);
    reg_convertNMIGradientFromVoxelToRealSpace_gpu( floatingMatrix_xyz,
                                                    this->controlPointGrid,
                                                    &this->nodeBasedMeasureGradientImage_gpu);
    // The gradient is smoothed using a Gaussian kernel if it is required
    if(this->gradientSmoothingSigma!=0){
        reg_gaussianSmoothing_gpu(this->controlPointGrid,
                                  &this->nodeBasedMeasureGradientImage_gpu,
                                  this->gradientSmoothingSigma,
                                  NULL);
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::GetBendingEnergyGradient()
{
    reg_bspline_ApproxBendingEnergyGradient_gpu(this->currentReference,
                                                 this->controlPointGrid,
                                                 &this->controlPointGrid_gpu,
                                                 &this->nodeBasedMeasureGradientImage_gpu,
                                                 this->bendingEnergyWeight);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::GetJacobianBasedGradient()
{
    reg_bspline_ComputeJacobianGradient_gpu(this->currentReference,
                                            this->controlPointGrid,
                                            &this->controlPointGrid_gpu,
                                            &this->nodeBasedMeasureGradientImage_gpu,
                                            this->jacobianLogWeight,
                                            this->jacobianLogApproximation);
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ComputeConjugateGradient(unsigned int iteration)
{
    if(iteration==1){
        // first conjugate gradient iteration
        reg_initialiseConjugateGradient(&this->nodeBasedMeasureGradientImage_gpu,
                                        &this->conjugateG_gpu,
                                        &this->conjugateH_gpu,
                                        this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz);
    }
    else{
        // conjugate gradient computation if iteration != 1
        reg_GetConjugateGradient(&this->nodeBasedMeasureGradientImage_gpu,
                                 &this->conjugateG_gpu,
                                 &this->conjugateH_gpu,
                                 this->controlPointGrid->nx*
                                 this->controlPointGrid->ny*
                                 this->controlPointGrid->nz);
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_gpu<T>::GetMaximalGradientLength()
{
    T maxLength = reg_getMaximalLength_gpu(&this->nodeBasedMeasureGradientImage_gpu,
                                           this->controlPointGrid->nx*
                                           this->controlPointGrid->ny*
                                           this->controlPointGrid->nz);
    return maxLength;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::UpdateControlPointPosition(T scale)
{
    if(this->useComposition){ // the control point positions are updated using composition
        this->RestoreCurrentControlPoint();
        reg_spline_cppComposition_gpu(  this->controlPointGrid,
                                        this->controlPointGrid,
                                        &this->controlPointGrid_gpu,
                                        &this->nodeBasedMeasureGradientImage_gpu,
                                        scale,
                                        1);
    }
    else{ // the control point positions are updated using addition
        reg_updateControlPointPosition_gpu(this->controlPointGrid,
                                           &this->controlPointGrid_gpu,
                                           &this->bestControlPointPosition_gpu,
                                           &this->nodeBasedMeasureGradientImage_gpu,
                                           scale);
    }
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::AllocateCurrentInputImage()
{
    if(this->currentReference_gpu!=NULL) cudaCommon_free<float>(&this->currentReference_gpu);
    if(this->currentReference2_gpu!=NULL) cudaCommon_free<float>(&this->currentReference2_gpu);
    if(this->currentReference->nt==1){
        if(cudaCommon_allocateArrayToDevice<float>(&this->currentReference_gpu, this->currentReference->dim)) return 1;
        if(cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentReference_gpu, this->currentReference)) return 1;
    }
    else if(this->currentReference->nt==2){
        if(cudaCommon_allocateArrayToDevice<float>(&this->currentReference_gpu,&this->currentReference2_gpu, this->currentReference->dim)) return 1;
        if(cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentReference_gpu, &this->currentReference2_gpu, this->currentReference)) return 1;
    }

    if(this->currentFloating_gpu!=NULL) cudaCommon_free(&this->currentFloating_gpu);
    if(this->currentFloating2_gpu!=NULL) cudaCommon_free(&this->currentFloating2_gpu);
    if(this->currentReference->nt==1){
        if(cudaCommon_allocateArrayToDevice<float>(&this->currentFloating_gpu, this->currentFloating->dim)) return 1;
        if(cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentFloating_gpu, this->currentFloating)) return 1;
    }
    else if(this->currentReference->nt==2){
        if(cudaCommon_allocateArrayToDevice<float>(&this->currentFloating_gpu, &this->currentFloating2_gpu, this->currentFloating->dim)) return 1;
        if(cudaCommon_transferNiftiToArrayOnDevice<float>(&this->currentFloating_gpu, &this->currentFloating2_gpu, this->currentFloating)) return 1;
    }
    if(this->controlPointGrid_gpu!=NULL) cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    if(cudaCommon_allocateArrayToDevice<float4>(&this->controlPointGrid_gpu, this->controlPointGrid->dim)) return 1;
    if(cudaCommon_transferNiftiToArrayOnDevice<float4>(&this->controlPointGrid_gpu, this->controlPointGrid)) return 1;

    int *targetMask_h;
    CUDA_SAFE_CALL(cudaMallocHost(&targetMask_h,this->activeVoxelNumber[this->currentLevel]*sizeof(int)));
    int *targetMask_h_ptr = &targetMask_h[0];
    for(int i=0;i<this->currentReference->nx*this->currentReference->ny*this->currentReference->nz;i++){
        if( this->currentMask[i]!=-1) *targetMask_h_ptr++=i;
    }
    CUDA_SAFE_CALL(cudaMalloc(&this->currentMask_gpu, this->activeVoxelNumber[this->currentLevel]*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(this->currentMask_gpu, targetMask_h, this->activeVoxelNumber[this->currentLevel]*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(targetMask_h));
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::ClearCurrentInputImage()
{
    if(cudaCommon_transferFromDeviceToNifti<float4>(this->controlPointGrid, &this->controlPointGrid_gpu)) return 1;
    cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    this->controlPointGrid_gpu=NULL;
    cudaCommon_free(&this->currentReference_gpu);
    this->currentReference_gpu=NULL;
    cudaCommon_free(&this->currentFloating_gpu);
    this->currentFloating_gpu=NULL;
    CUDA_SAFE_CALL(cudaFree(this->currentMask_gpu));
    this->currentMask_gpu=NULL;

    if(this->currentReference->nt==2){
        cudaCommon_free(&this->currentReference2_gpu);
        this->currentReference2_gpu=NULL;
        cudaCommon_free(&this->currentFloating2_gpu);
        this->currentFloating2_gpu=NULL;
    }
    this->currentReference=NULL;
    this->currentMask=NULL;
    this->currentFloating=NULL;
    return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::CheckMemoryMB_f3d()
{
    if(!this->initialised){
        if( reg_f3d<T>::Initisalise_f3d() )
            return 1;
    }

    unsigned int totalMemoryRequiered=0;
    // reference image
    totalMemoryRequiered += this->referencePyramid[this->levelToPerform-1]->nvox * sizeof(float);

    // floating image
    totalMemoryRequiered += this->floatingPyramid[this->levelToPerform-1]->nvox * sizeof(float);

    // warped image
    totalMemoryRequiered += this->referencePyramid[this->levelToPerform-1]->nvox * sizeof(float);

    // mask image
    totalMemoryRequiered += this->activeVoxelNumber[this->levelToPerform-1] * sizeof(int);

    // deformation field
    totalMemoryRequiered += this->activeVoxelNumber[this->levelToPerform-1] * sizeof(float4);

    // voxel based intensity gradient
    totalMemoryRequiered += this->referencePyramid[this->levelToPerform-1]->nvox * sizeof(float4);

    // voxel based NMI gradient + smoothing
    totalMemoryRequiered += 2 * this->referencePyramid[this->levelToPerform-1]->nvox * sizeof(float4);

    // control point grid
    unsigned int cp=1;
    cp *= (int)floor(this->referencePyramid[this->levelToPerform-1]->nx*this->referencePyramid[this->levelToPerform-1]->dx/this->spacing[0])+5;
    cp *= (int)floor(this->referencePyramid[this->levelToPerform-1]->ny*this->referencePyramid[this->levelToPerform-1]->dy/this->spacing[1])+5;
    if(this->referencePyramid[this->levelToPerform-1]->nz>1)
        cp *= (int)floor(this->referencePyramid[this->levelToPerform-1]->nz*this->referencePyramid[this->levelToPerform-1]->dz/this->spacing[2])+5;
    totalMemoryRequiered += cp * sizeof(float4);

    // node based NMI gradient
    totalMemoryRequiered += cp * sizeof(float4);

    // conjugate gradient
    totalMemoryRequiered += 2 * cp * sizeof(float4);

    // joint histogram
    unsigned int histogramSize[3]={1,1,1};
    for(int i=0;i<this->referenceTimePoint;i++){
        histogramSize[0] *= this->referenceBinNumber[i];
        histogramSize[1] *= this->referenceBinNumber[i];
    }
    for(int i=0;i<this->floatingTimePoint;i++){
        histogramSize[0] *= this->floatingBinNumber[i];
        histogramSize[2] *= this->floatingBinNumber[i];
    }
    histogramSize[0] += histogramSize[1] + histogramSize[2];
    totalMemoryRequiered += histogramSize[0] * sizeof(float);

    // jacobian array
    if(this->jacobianLogWeight>0)
        totalMemoryRequiered += 10 * this->referencePyramid[this->levelToPerform-1]->nvox * sizeof(float);

    return (int)(ceil(totalMemoryRequiered/1000000));

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
