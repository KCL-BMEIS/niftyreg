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
	this->inverseConsistencyWeight=0;
	this->BCHUpdate=false;
    this->useGradientCumulativeExp=false;
    this->BCHUpdateValue=0;

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
void reg_f3d2<T>::UseBCHUpdate(int v)
{
    this->BCHUpdate = true;
    this->useGradientCumulativeExp = false;
    this->BCHUpdateValue=v;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::UseGradientCumulativeExp()
{
	this->BCHUpdate = false;
    this->useGradientCumulativeExp = true;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d2<T>::Initisalise()
{
    reg_f3d_sym<T>::Initisalise();

    // Convert the deformation field into velocity field
    this->controlPointGrid->intent_code=NIFTI_INTENT_VECTOR;
    this->backwardControlPointGrid->intent_code=NIFTI_INTENT_VECTOR;
    memset(this->controlPointGrid->intent_name, 0, 16);
    strcpy(this->controlPointGrid->intent_name,"NREG_TRANS");
    memset(this->backwardControlPointGrid->intent_name, 0, 16);
    strcpy(this->backwardControlPointGrid->intent_name,"NREG_TRANS");
    this->controlPointGrid->intent_p1=SPLINE_VEL_GRID;
    this->backwardControlPointGrid->intent_p1=SPLINE_VEL_GRID;

    // Set the default step number for scaling and squaring
    this->controlPointGrid->intent_p2=6;
    this->backwardControlPointGrid->intent_p2=6;

    // Set the affine parametrisation
    if(this->affineTransformation!=NULL){
        // Set the reorientation matrices using the provided affine transformation
        mat44 inverseAffineTransformation = nifti_mat44_inverse(*this->affineTransformation);
        // Create extensions to store the affine parametrisations
        if(this->controlPointGrid->ext_list!=NULL)
            free(this->controlPointGrid->ext_list);
        this->controlPointGrid->num_ext=1;
        this->controlPointGrid->ext_list=(nifti1_extension *)malloc(sizeof(nifti1_extension));
        this->controlPointGrid->ext_list->esize=16*sizeof(float)+16;
        this->controlPointGrid->ext_list->ecode=NIFTI_ECODE_IGNORE;
        this->controlPointGrid->ext_list->edata=(char *)malloc((this->controlPointGrid->ext_list->esize-8)*sizeof(float));
        mat44 *forExtPtr = reinterpret_cast<mat44 *>(this->controlPointGrid->ext_list->edata);
        for(unsigned int i=0;i<4;++i)
            for(unsigned int j=0;j<4;++j)
                forExtPtr->m[i][j] = this->affineTransformation->m[i][j];

        if(this->backwardControlPointGrid->ext_list!=NULL)
            free(this->backwardControlPointGrid->ext_list);
        this->backwardControlPointGrid->num_ext=1;
        this->backwardControlPointGrid->ext_list=(nifti1_extension *)malloc(sizeof(nifti1_extension));
        this->backwardControlPointGrid->ext_list->esize=16*sizeof(float)+16;
        this->backwardControlPointGrid->ext_list->ecode=NIFTI_ECODE_IGNORE;
        this->backwardControlPointGrid->ext_list->edata=(char *)malloc((this->backwardControlPointGrid->ext_list->esize-8)*sizeof(float));
        forExtPtr = reinterpret_cast<mat44 *>(this->backwardControlPointGrid->ext_list->edata);
        for(unsigned int i=0;i<4;++i)
            for(unsigned int j=0;j<4;++j)
                forExtPtr->m[i][j] = inverseAffineTransformation.m[i][j];
    }

    // Set the velocity field parametrisations to 0 displacement
    reg_tools_multiplyValueToImage(this->controlPointGrid,this->controlPointGrid,0.f);
    reg_tools_multiplyValueToImage(this->backwardControlPointGrid,this->backwardControlPointGrid,0.f);
    // Convert the parametrisations into deformation fields
    reg_getDeformationFromDisplacement(this->controlPointGrid);
    reg_getDeformationFromDisplacement(this->backwardControlPointGrid);

#ifdef NDEBUG
    if(this->verbose){
#endif
        printf("[%s]\n", this->executableName);
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2::Initialise_f3d() done\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetDeformationField()
{
    // By default the number of steps is automatically updated
    bool updateStepNumber=true;
    // The provided step number is used for the final resampling
    if(this->optimiser==false)
         updateStepNumber=false;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration forward. Step number update=%i\n",updateStepNumber);
#endif
    // The forward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                   this->deformationFieldImage,
                                                   updateStepNumber
                                                   );
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration backward. Step number update=%i\n",updateStepNumber);
#endif
    // The number of step number is copied over from the forward transformation
    this->backwardControlPointGrid->intent_p2=this->controlPointGrid->intent_p2;
    // The backward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDeformationFieldFromVelocityGrid(this->backwardControlPointGrid,
                                                   this->backwardDeformationFieldImage,
                                                   false
                                                   );
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyErrorField()
{
    if(this->inverseConsistencyWeight<=0) return;

    fprintf(stderr, "NR ERROR - reg_f3d2<T>::GetInverseConsistencyErrorField() has to be implemented");
    reg_exit(1);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyGradient()
{
    if(this->inverseConsistencyWeight<=0) return;

    fprintf(stderr, "NR ERROR - reg_f3d2<T>::GetInverseConsistencyGradient() has to be implemented");
    reg_exit(1);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetVoxelBasedGradient()
{
    reg_f3d_sym<T>::GetVoxelBasedGradient();

    // Exponentiate the gradients if required
    if(this->useGradientCumulativeExp)
        this->ExponentiateGradient();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::ExponentiateGradient()
{
    if(!this->useGradientCumulativeExp) return;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
	// Exponentiate the forward gradient using the backward transformation
#ifndef NDEBUG
	printf("[NiftyReg f3d2] Update the forward measure gradient using a Dartel like approach\n");
#endif
	// Create all deformation field images needed for resampling
    nifti_image **tempDef=(nifti_image **)malloc(
                (unsigned int)(fabs(this->backwardControlPointGrid->intent_p2)+1) *
                sizeof(nifti_image *));
    for(unsigned int i=0; i<=(unsigned int)fabs(this->backwardControlPointGrid->intent_p2);++i){
        tempDef[i]=nifti_copy_nim_info(this->deformationFieldImage);
		tempDef[i]->data=(void *)malloc(tempDef[i]->nvox*tempDef[i]->nbyper);
	}
	// Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(this->backwardControlPointGrid,
												   tempDef);

	/* Allocate a temporary gradient image to store the backward gradient */
    nifti_image *tempGrad=nifti_copy_nim_info(this->voxelBasedMeasureGradientImage);

	tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
    for(unsigned int i=0; i<(int)fabs(this->backwardControlPointGrid->intent_p2);++i){

		reg_resampleGradient(this->voxelBasedMeasureGradientImage, // floating
							 tempGrad, // warped - out
							 tempDef[i], // deformation field
							 1, // interpolation type - linear
							 0.f); // padding value
        reg_tools_addImageToImage(tempGrad, // in1
                                     this->voxelBasedMeasureGradientImage, // in2
                                     this->voxelBasedMeasureGradientImage); // out
    }

	// Free the temporary deformation field
    for(unsigned int i=0; i<=(int)fabs(this->backwardControlPointGrid->intent_p2);++i){
		nifti_image_free(tempDef[i]);
		tempDef[i]=NULL;
	}
	free(tempDef);tempDef=NULL;
	// Free the temporary gradient image
	nifti_image_free(tempGrad); tempGrad=NULL;
	// Normalise the forward gradient
    reg_tools_divideValueToImage(this->voxelBasedMeasureGradientImage, // in
                                 this->voxelBasedMeasureGradientImage, // out
                                 powf(2.f,fabsf(this->backwardControlPointGrid->intent_p2))); // value

	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
	/* Exponentiate the backward gradient using the forward transformation */
#ifndef NDEBUG
	printf("[NiftyReg f3d2] Update the backward measure gradient using a Dartel like approach\n");
#endif
	// Allocate a temporary gradient image to store the backward gradient
	tempGrad=nifti_copy_nim_info(this->backwardVoxelBasedMeasureGradientImage);
	tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
	// Create all deformation field images needed for resampling
    tempDef=(nifti_image **)malloc((unsigned int)(fabs(this->controlPointGrid->intent_p2)+1) * sizeof(nifti_image *));
    for(unsigned int i=0; i<=(unsigned int)fabs(this->controlPointGrid->intent_p2);++i){
        tempDef[i]=nifti_copy_nim_info(this->backwardDeformationFieldImage);
		tempDef[i]->data=(void *)malloc(tempDef[i]->nvox*tempDef[i]->nbyper);
	}
	// Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(this->controlPointGrid,
                                                   tempDef);

    for(unsigned int i=0; i<(int)fabs(this->controlPointGrid->intent_p2);++i){

		reg_resampleGradient(this->backwardVoxelBasedMeasureGradientImage, // floating
							 tempGrad, // warped - out
							 tempDef[i], // deformation field
							 1, // interpolation type - linear
                             0.f); // padding value
        reg_tools_addImageToImage(tempGrad, // in1
                                  this->backwardVoxelBasedMeasureGradientImage, // in2
                                  this->backwardVoxelBasedMeasureGradientImage); // out
	}

	// Free the temporary deformation field
    for(unsigned int i=0; i<=(int)fabs(this->controlPointGrid->intent_p2);++i){
		nifti_image_free(tempDef[i]);
		tempDef[i]=NULL;
	}
	free(tempDef);tempDef=NULL;
	// Free the temporary gradient image
	nifti_image_free(tempGrad); tempGrad=NULL;
	// Normalise the backward gradient
    reg_tools_divideValueToImage(this->backwardVoxelBasedMeasureGradientImage, // in
                                 this->backwardVoxelBasedMeasureGradientImage, // out
                                 powf(2.f,fabsf(this->controlPointGrid->intent_p2))); // value

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::UpdateParameters(float scale)
{
    // Restore the last successfull control point grids
    this->optimiser->RestoreBestDOF();

    /************************/
    /**** Forward update ****/
    /************************/
    // Scale the gradient image
    nifti_image *forwardScaledGradient=nifti_copy_nim_info(this->transformationGradient);
    forwardScaledGradient->data=(void *)malloc(forwardScaledGradient->nvox*forwardScaledGradient->nbyper);
    reg_tools_multiplyValueToImage(this->transformationGradient,
                                   forwardScaledGradient,
                                   scale); // *(scale)
    // The scaled gradient image is added to the current estimate of the transformation using
    // a simple addition or by computing the BCH update
    // Note that the gradient has been integrated over the path of transformation previously
    if(this->BCHUpdate){
        // Compute the BCH update
		printf("USING BCH FORWARD - TESTING ONLY\n");
#ifndef NDEBUG
        printf("[NiftyReg f3d2] Update the forward control point grid using BCH approximation\n");
#endif
        compute_BCH_update(this->controlPointGrid,
                           forwardScaledGradient,
                           this->BCHUpdateValue);
    }
    else{
        // Update the velocity field
        reg_tools_addImageToImage(this->controlPointGrid, // in1
                                  forwardScaledGradient, // in2
                                  this->controlPointGrid); // out
    }
    // Clean the temporary nifti_images
    nifti_image_free(forwardScaledGradient);forwardScaledGradient=NULL;

    /************************/
    /**** Backward update ***/
    /************************/
    // Scale the gradient image
    nifti_image *backwardScaledGradient=nifti_copy_nim_info(this->backwardTransformationGradient);
    backwardScaledGradient->data=(void *)malloc(backwardScaledGradient->nvox*backwardScaledGradient->nbyper);
    reg_tools_multiplyValueToImage(this->backwardTransformationGradient,
                                   backwardScaledGradient,
                                   scale); // *(scale)
    // The scaled gradient image is added to the current estimate of the transformation using
    // a simple addition or by computing the BCH update
    // Note that the gradient has been integrated over the path of transformation previously
    if(this->BCHUpdate){
        // Compute the BCH update
		printf("USING BCH BACKWARD - TESTING ONLY\n");
#ifndef NDEBUG
        printf("[NiftyReg f3d2] Update the backward control point grid using BCH approximation\n");
#endif
        compute_BCH_update(this->backwardControlPointGrid,
                           backwardScaledGradient,
                           this->BCHUpdateValue);
    }
    else{
        // Update the velocity field
        reg_tools_addImageToImage(this->backwardControlPointGrid, // in1
                                  backwardScaledGradient, // in2
                                  this->backwardControlPointGrid); // out
    }
    // Clean the temporary nifti_images
    nifti_image_free(backwardScaledGradient);backwardScaledGradient=NULL;

    /****************************/
    /******** Symmetrise ********/
    /****************************/

    // In order to ensure symmetry the forward and backward velocity fields
    // are averaged in both image spaces: reference and floating
    /****************************/
    nifti_image *warpedForwardTrans = nifti_copy_nim_info(this->backwardControlPointGrid);
    warpedForwardTrans->data=(void *)malloc(warpedForwardTrans->nvox*warpedForwardTrans->nbyper);
    nifti_image *warpedBackwardTrans = nifti_copy_nim_info(this->controlPointGrid);
    warpedBackwardTrans->data=(void *)malloc(warpedBackwardTrans->nvox*warpedBackwardTrans->nbyper);

    // Both parametrisations are converted into displacement
    reg_getDisplacementFromDeformation(this->controlPointGrid);
    reg_getDisplacementFromDeformation(this->backwardControlPointGrid);

    // The parametrisation have first to be resampled and reoriented
    if(this->controlPointGrid->sto_xyz==this->backwardControlPointGrid->sto_xyz &&
       this->controlPointGrid->nx==this->backwardControlPointGrid->nx &&
       this->controlPointGrid->ny==this->backwardControlPointGrid->ny &&
       this->controlPointGrid->nz==this->backwardControlPointGrid->nz &&
       this->controlPointGrid->dx==this->backwardControlPointGrid->dx &&
       this->controlPointGrid->dy==this->backwardControlPointGrid->dy &&
       this->controlPointGrid->dz==this->backwardControlPointGrid->dz){
#ifndef NDEBUG
        printf("[NiftyReg f3d2] Addition based symmetrisation\n");
#endif
        // Both parametrisations are copied over
        memcpy(warpedBackwardTrans->data,this->backwardControlPointGrid,warpedBackwardTrans->nvox*warpedBackwardTrans->nbyper);
        memcpy(warpedForwardTrans->data,this->controlPointGrid,warpedForwardTrans->nvox*warpedForwardTrans->nbyper);
    }
    /****************************/
    else{
#ifndef NDEBUG
        printf("[NiftyReg f3d2] Interpolation based symmetrisation\n");
#endif
        // Allocate some deformation field images
        nifti_image *backward2forwardDEF = nifti_copy_nim_info(this->backwardControlPointGrid);
        backward2forwardDEF->data=(void *)calloc(backward2forwardDEF->nvox,backward2forwardDEF->nbyper);
        if(this->backwardControlPointGrid->ext_list!=NULL){
            // Read the specified affine transformation
            reg_affine_deformationField(reinterpret_cast<mat44 *>(this->backwardControlPointGrid->ext_list->edata),
                                        backward2forwardDEF);
        }
        else{
            // Use an identity transformation
            reg_tools_multiplyValueToImage(backward2forwardDEF,backward2forwardDEF,0.f);
            reg_getDeformationFromDisplacement(backward2forwardDEF);
        }
        // Resample the forward grid in the space of the backward grid
        // The forward deformation grid has been set to displacement grid in order to
        // enable zero padding
        reg_resampleGradient(this->controlPointGrid, // floating displacement field
                             warpedForwardTrans, // warped displacement field
                             backward2forwardDEF, // deformation field
                             1, // linear interpolation
                             0.f // padding
                             );
        // Clean the temporary deformation fields
        nifti_image_free(backward2forwardDEF);backward2forwardDEF=NULL;
        // Allocate some deformation field images
        nifti_image *forward2backwardDEF = nifti_copy_nim_info(this->controlPointGrid);
        forward2backwardDEF->data=(void *)calloc(forward2backwardDEF->nvox,forward2backwardDEF->nbyper);
        if(this->controlPointGrid->ext_list!=NULL){
            // Read the specified affine transformation
            reg_affine_deformationField(reinterpret_cast<mat44 *>(this->controlPointGrid->ext_list->edata),
                                        forward2backwardDEF);
        }
        else{
            // Use an identity transformation
            reg_tools_multiplyValueToImage(forward2backwardDEF,forward2backwardDEF,0.f);
            reg_getDeformationFromDisplacement(forward2backwardDEF);
        }
        // Resample the backward grid in the space of the forward grid
        // The backward deformation grid has been set to displacement grid in order to
        // enable zero padding
        reg_resampleGradient(this->backwardControlPointGrid, // floating displacement field
                             warpedBackwardTrans, // warped displacement field
                             forward2backwardDEF, // deformation field
                             1, // linear interpolation
                             0.f // padding
                             );
        // Clean the temporary deformation fields
        nifti_image_free(forward2backwardDEF);forward2backwardDEF=NULL;
    }
    /* Average velocity fields into forward and backward space */
    // Substraction as the propagated has to be negated
    reg_tools_substractImageToImage(this->backwardControlPointGrid, // displacement
                                    warpedForwardTrans, // displacement
                                    this->backwardControlPointGrid); // displacement output
    reg_tools_substractImageToImage(this->controlPointGrid, // displacement
                                    warpedBackwardTrans, // displacement
                                    this->controlPointGrid); // displacement output
    // Division by 2
    reg_tools_multiplyValueToImage(this->backwardControlPointGrid, // displacement
                                   this->backwardControlPointGrid, // displacement
                                   0.5f); // *(0.5)
    reg_tools_multiplyValueToImage(this->controlPointGrid, // displacement
                                   this->controlPointGrid, // displacement
                                   0.5f); // *(0.5)
    // Clean the temporary allocated velocity fields
    nifti_image_free(warpedForwardTrans);warpedForwardTrans=NULL;
    nifti_image_free(warpedBackwardTrans);warpedBackwardTrans=NULL;

    // Convert the velocity field from displacement to deformation
    reg_getDeformationFromDisplacement(this->controlPointGrid);
    reg_getDeformationFromDisplacement(this->backwardControlPointGrid);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image **reg_f3d2<T>::GetWarpedImage()
{
    // The initial images are used
    if(this->inputReference==NULL ||
            this->inputFloating==NULL ||
            this->controlPointGrid==NULL ||
            this->backwardControlPointGrid==NULL){
        fprintf(stderr,"[NiftyReg ERROR] reg_f3d_sym::GetWarpedImage()\n");
        fprintf(stderr," * The reference, floating and both control point grid images have to be defined\n");
    }

    // Set the input images
    reg_f3d2<T>::currentReference = this->inputReference;
    reg_f3d2<T>::currentFloating = this->inputFloating;
    // No mask is used to perform the final resampling
    reg_f3d2<T>::currentMask = NULL;
    reg_f3d2<T>::currentFloatingMask = NULL;

    // Allocate the forward and backward warped images
    reg_f3d2<T>::AllocateWarped();
    // Allocate the forward and backward dense deformation field
    reg_f3d2<T>::AllocateDeformationField();

    // Warp the floating images into the reference spaces using a cubic spline interpolation
    reg_f3d2<T>::WarpFloatingImage(3); // cubic spline interpolation

    // Clear the deformation field
    reg_f3d2<T>::ClearDeformationField();

    // Allocate and save the forward transformation warped image
    nifti_image **resultImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
    resultImage[0] = nifti_copy_nim_info(this->warped);
    resultImage[0]->cal_min=this->inputFloating->cal_min;
    resultImage[0]->cal_max=this->inputFloating->cal_max;
    resultImage[0]->scl_slope=this->inputFloating->scl_slope;
    resultImage[0]->scl_inter=this->inputFloating->scl_inter;
    resultImage[0]->data=(void *)malloc(resultImage[0]->nvox*resultImage[0]->nbyper);
    memcpy(resultImage[0]->data, this->warped->data, resultImage[0]->nvox*resultImage[0]->nbyper);

    // Allocate and save the backward transformation warped image
    resultImage[1] = nifti_copy_nim_info(this->backwardWarped);
    resultImage[1]->cal_min=this->inputReference->cal_min;
    resultImage[1]->cal_max=this->inputReference->cal_max;
    resultImage[1]->scl_slope=this->inputReference->scl_slope;
    resultImage[1]->scl_inter=this->inputReference->scl_inter;
    resultImage[1]->data=(void *)malloc(resultImage[1]->nvox*resultImage[1]->nbyper);
    memcpy(resultImage[1]->data, this->backwardWarped->data, resultImage[1]->nvox*resultImage[1]->nbyper);

    // Clear the warped images
    reg_f3d2<T>::ClearWarped();

    // Return the two final warped images
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
