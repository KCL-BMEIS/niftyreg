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
	this->BCHUpdate=false;
	this->ISS=false;
    this->BCHUpdateValue=0;

    this->forward2backward_reorient=NULL;
    this->backward2forward_reorient=NULL;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d2 constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d2<T>::~reg_f3d2()
{
    if(this->forward2backward_reorient!=NULL)
        delete []this->forward2backward_reorient;
    this->forward2backward_reorient=NULL;
    if(this->backward2forward_reorient!=NULL)
        delete []this->backward2forward_reorient;
    this->backward2forward_reorient=NULL;
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
template <class T>
void reg_f3d2<T>::UseBCHUpdate(int v)
{
    this->BCHUpdate = true;
	this->ISS = false;
    this->BCHUpdateValue=v;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::UseInverseSclalingSquaring()
{
	this->BCHUpdate = false;
	this->ISS = true;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::DefineReorientationMatrices()
{

    mat33 vox2real_for, vox2real_bck;
//    mat33 real2vox_bck, real2vox_bck;
    int orient_for_i, orient_for_j, orient_for_k;
    int orient_bck_i, orient_bck_j, orient_bck_k;

    // Extract the ijk->xyz matrices
    if(this->controlPointGrid->sform_code>0){
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
//                real2vox_for.m[i][j]=this->controlPointGrid->sto_ijk.m[i][j];
                vox2real_for.m[i][j]=this->controlPointGrid->sto_xyz.m[i][j];
            }
        }
        // Extract the orientation coefficients to define reorientation matrices
        nifti_mat44_to_orientation(this->controlPointGrid->sto_xyz, &orient_for_i, &orient_for_j, &orient_for_k);
    }
    else{
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
//                real2vox_for.m[i][j]=this->controlPointGrid->qto_ijk.m[i][j];
                vox2real_for.m[i][j]=this->controlPointGrid->qto_xyz.m[i][j];
            }
        }
        // Extract the orientation coefficients to define reorientation matrices
        nifti_mat44_to_orientation(this->controlPointGrid->qto_xyz, &orient_for_i, &orient_for_j, &orient_for_k);
    }
    if(this->backwardControlPointGrid->sform_code>0){
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
//                real2vox_bck.m[i][j]=this->backwardControlPointGrid->sto_ijk.m[i][j];
                vox2real_bck.m[i][j]=this->backwardControlPointGrid->sto_xyz.m[i][j];
            }
        }
        // Extract the orientation coefficients to define reorientation matrices
        nifti_mat44_to_orientation(this->backwardControlPointGrid->sto_xyz, &orient_bck_i, &orient_bck_j, &orient_bck_k);
    }
    else{
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
//                real2vox_bck.m[i][j]=this->backwardControlPointGrid->qto_ijk.m[i][j];
                vox2real_bck.m[i][j]=this->backwardControlPointGrid->qto_xyz.m[i][j];
            }
        }
        // Extract the orientation coefficients to define reorientation matrices
        nifti_mat44_to_orientation(this->backwardControlPointGrid->qto_xyz, &orient_bck_i, &orient_bck_j, &orient_bck_k);
    }

    // Create two mat33 full of zero
    mat33 for_no_orient;
    mat33 bck_no_orient;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            for_no_orient.m[i][j]=0.f;
            bck_no_orient.m[i][j]=0.f;
        }
    }
    // Generate a matrix that maps the forward grid into a diagonal space
    switch(orient_for_i){
    case NIFTI_L2R:for_no_orient.m[0][0]=1.f;break;
    case NIFTI_R2L:for_no_orient.m[0][0]=-1.f;break;
    case NIFTI_P2A:for_no_orient.m[0][1]=1.f;break;
    case NIFTI_A2P:for_no_orient.m[0][1]=-1.f;break;
    case NIFTI_I2S:for_no_orient.m[0][2]=1.f;break;
    case NIFTI_S2I:for_no_orient.m[0][2]=-1.f;break;
    }
    switch(orient_for_j){
    case NIFTI_L2R:for_no_orient.m[1][0]=1.f;break;
    case NIFTI_R2L:for_no_orient.m[1][0]=-1.f;break;
    case NIFTI_P2A:for_no_orient.m[1][1]=1.f;break;
    case NIFTI_A2P:for_no_orient.m[1][1]=-1.f;break;
    case NIFTI_I2S:for_no_orient.m[1][2]=1.f;break;
    case NIFTI_S2I:for_no_orient.m[1][2]=-1.f;break;
    }
    switch(orient_for_k){
    case NIFTI_L2R:for_no_orient.m[2][0]=1.f;break;
    case NIFTI_R2L:for_no_orient.m[2][0]=-1.f;break;
    case NIFTI_P2A:for_no_orient.m[2][1]=1.f;break;
    case NIFTI_A2P:for_no_orient.m[2][1]=-1.f;break;
    case NIFTI_I2S:for_no_orient.m[2][2]=1.f;break;
    case NIFTI_S2I:for_no_orient.m[2][2]=-1.f;break;
    }

    // Generate a matrix that maps the backward grid into a diagonal space
    switch(orient_bck_i){
    case NIFTI_L2R:bck_no_orient.m[0][0]=1.f;break;
    case NIFTI_R2L:bck_no_orient.m[0][0]=-1.f;break;
    case NIFTI_P2A:bck_no_orient.m[0][1]=1.f;break;
    case NIFTI_A2P:bck_no_orient.m[0][1]=-1.f;break;
    case NIFTI_I2S:bck_no_orient.m[0][2]=1.f;break;
    case NIFTI_S2I:bck_no_orient.m[0][2]=-1.f;break;
    }
    switch(orient_bck_j){
    case NIFTI_L2R:bck_no_orient.m[1][0]=1.f;break;
    case NIFTI_R2L:bck_no_orient.m[1][0]=-1.f;break;
    case NIFTI_P2A:bck_no_orient.m[1][1]=1.f;break;
    case NIFTI_A2P:bck_no_orient.m[1][1]=-1.f;break;
    case NIFTI_I2S:bck_no_orient.m[1][2]=1.f;break;
    case NIFTI_S2I:bck_no_orient.m[1][2]=-1.f;break;
    }
    switch(orient_bck_k){
    case NIFTI_L2R:bck_no_orient.m[2][0]=1.f;break;
    case NIFTI_R2L:bck_no_orient.m[2][0]=-1.f;break;
    case NIFTI_P2A:bck_no_orient.m[2][1]=1.f;break;
    case NIFTI_A2P:bck_no_orient.m[2][1]=-1.f;break;
    case NIFTI_I2S:bck_no_orient.m[2][2]=1.f;break;
    case NIFTI_S2I:bck_no_orient.m[2][2]=-1.f;break;
    }

    // Compute the matrices to use to warps one vector field into the space of another
    mat33 forward2backward = nifti_mat33_mul(for_no_orient,vox2real_for);
    mat33 backward2forward = nifti_mat33_mul(bck_no_orient,vox2real_bck);
    mat33 tempA=nifti_mat33_inverse(forward2backward);
    mat33 tempB=nifti_mat33_inverse(backward2forward);
    forward2backward=nifti_mat33_mul(tempB,forward2backward);
    backward2forward=nifti_mat33_mul(tempA,backward2forward);
//    reg_mat33_disp(&forward2backward, (char *) "forward2backward");
//    reg_mat33_disp(&backward2forward, (char *) "backward2forward");

    // Check if the matrices are different from identity
    bool identicalMatrices=true;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            if(forward2backward.m[i][j]!=backward2forward.m[i][j])
                identicalMatrices=false;
        }
    }
    // Save the computed matrices
    if(identicalMatrices==false){
        // Clean the matrices if necessary
        if(this->forward2backward_reorient!=NULL)
            delete []this->forward2backward_reorient;
        this->forward2backward_reorient=NULL;
        if(this->backward2forward_reorient!=NULL)
            delete []this->backward2forward_reorient;
        this->backward2forward_reorient=NULL;
        // Allocate the matrices
        this->forward2backward_reorient=new mat33[1];
        this->backward2forward_reorient=new mat33[1];
        // Compute the rotation matrices for reorientation
        forward2backward=nifti_mat33_polar(forward2backward);
        backward2forward=nifti_mat33_polar(backward2forward);
        // Save the matrices
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                this->forward2backward_reorient->m[i][j]=forward2backward.m[i][j];
                this->backward2forward_reorient->m[i][j]=backward2forward.m[i][j];
            }
        }
    }

    return;
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

    this->controlPointGrid->intent_p1=this->stepNumber;
    this->backwardControlPointGrid->intent_p1=this->stepNumber;

    memset(this->controlPointGrid->intent_name, 0, 16);
    memset(this->backwardControlPointGrid->intent_name, 0, 16);
    strcpy(this->controlPointGrid->intent_name,"NREG_VEL_STEP");
    strcpy(this->backwardControlPointGrid->intent_name,"NREG_VEL_STEP");

    // Define the reorientation matrices if needed
    this->DefineReorientationMatrices();

#ifdef NDEBUG
    if(this->verbose){
#endif
        printf("[%s]\n", this->executableName);
        printf("[%s] Exponentiation of the velocity field is performed using %i steps\n",
               this->executableName, this->stepNumber);
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
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration forward\n");
#endif
    // The forward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                    this->deformationFieldImage);
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Velocity integration backward\n");
#endif
    // The backward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDeformationFieldFromVelocityGrid(this->backwardControlPointGrid,
                                                    this->backwardDeformationFieldImage);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyErrorField()
{
    if(this->inverseConsistencyWeight<=0) return;

    if(this->similarityWeight<=0){
        reg_spline_getDeformationFieldFromVelocityGrid(this->controlPointGrid,
                                                        this->deformationFieldImage);
        reg_spline_getDeformationFieldFromVelocityGrid(this->backwardControlPointGrid,
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
    if(this->inverseConsistencyWeight<=0) return;

    fprintf(stderr, "NR ERROR - reg_f3d2<T>::GetInverseConsistencyGradient() has to be implemented");
    exit(1);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::GetVoxelBasedGradient()
{
    reg_f3d_sym<T>::GetVoxelBasedGradient();

    // Exponentiate the gradients if required
    if(this->ISS)
        this->ExponentiateGradient();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d2<T>::ExponentiateGradient()
{
    if(!this->ISS) return;

    /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
	// Exponentiate the forward gradient using the backward transformation
#ifndef NDEBUG
	printf("[NiftyReg f3d2] Update the forward measure gradient using a Dartel like approach\n");
#endif
	// Create all deformation field images needed for resampling
    nifti_image **tempDef=(nifti_image **)malloc(
                (unsigned int)(fabs(this->backwardControlPointGrid->intent_p1)+1) *
                sizeof(nifti_image *));
	for(unsigned int i=0; i<=(unsigned int)fabs(this->backwardControlPointGrid->intent_p1);++i){
        tempDef[i]=nifti_copy_nim_info(this->deformationFieldImage);
		tempDef[i]->data=(void *)malloc(tempDef[i]->nvox*tempDef[i]->nbyper);
	}
	// Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(this->backwardControlPointGrid,
												   tempDef);

	/* Allocate a temporary gradient image to store the backward gradient */
    nifti_image *tempGrad=nifti_copy_nim_info(this->voxelBasedMeasureGradientImage);

	tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
	for(unsigned int i=0; i<(int)fabs(this->backwardControlPointGrid->intent_p1);++i){

		reg_resampleGradient(this->voxelBasedMeasureGradientImage, // floating
							 tempGrad, // warped - out
							 tempDef[i], // deformation field
							 1, // interpolation type - linear
							 0.f); // padding value
        reg_tools_addSubMulDivImages(tempGrad, // in1
                                     this->voxelBasedMeasureGradientImage, // in2
                                     this->voxelBasedMeasureGradientImage, // out
                                     0); // addition
    }

	// Free the temporary deformation field
	for(unsigned int i=0; i<=(int)fabs(this->backwardControlPointGrid->intent_p1);++i){
		nifti_image_free(tempDef[i]);
		tempDef[i]=NULL;
	}
	free(tempDef);tempDef=NULL;
	// Free the temporary gradient image
	nifti_image_free(tempGrad); tempGrad=NULL;
	// Normalise the forward gradient
	reg_tools_addSubMulDivValue(this->voxelBasedMeasureGradientImage, // in
								this->voxelBasedMeasureGradientImage, // out
								powf(2.f,fabsf(this->backwardControlPointGrid->intent_p1)), // value
								3); // division

	/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
	/* Exponentiate the backward gradient using the forward transformation */
#ifndef NDEBUG
	printf("[NiftyReg f3d2] Update the backward measure gradient using a Dartel like approach\n");
#endif
	// Allocate a temporary gradient image to store the backward gradient
	tempGrad=nifti_copy_nim_info(this->backwardVoxelBasedMeasureGradientImage);
	tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
	// Create all deformation field images needed for resampling
	tempDef=(nifti_image **)malloc((unsigned int)(fabs(this->controlPointGrid->intent_p1)+1) * sizeof(nifti_image *));
	for(unsigned int i=0; i<=(unsigned int)fabs(this->controlPointGrid->intent_p1);++i){
        tempDef[i]=nifti_copy_nim_info(this->backwardDeformationFieldImage);
		tempDef[i]->data=(void *)malloc(tempDef[i]->nvox*tempDef[i]->nbyper);
	}
	// Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(this->controlPointGrid,
                                                   tempDef);

	for(unsigned int i=0; i<(int)fabs(this->controlPointGrid->intent_p1);++i){

		reg_resampleGradient(this->backwardVoxelBasedMeasureGradientImage, // floating
							 tempGrad, // warped - out
							 tempDef[i], // deformation field
							 1, // interpolation type - linear
                             0.f); // padding value
        reg_tools_addSubMulDivImages(tempGrad, // in1
                                     this->backwardVoxelBasedMeasureGradientImage, // in2
                                     this->backwardVoxelBasedMeasureGradientImage, // out
                                     0); // addition
	}

	// Free the temporary deformation field
	for(unsigned int i=0; i<=(int)fabs(this->controlPointGrid->intent_p1);++i){
		nifti_image_free(tempDef[i]);
		tempDef[i]=NULL;
	}
	free(tempDef);tempDef=NULL;
	// Free the temporary gradient image
	nifti_image_free(tempGrad); tempGrad=NULL;
	// Normalise the backward gradient
	reg_tools_addSubMulDivValue(this->backwardVoxelBasedMeasureGradientImage, // in
								this->backwardVoxelBasedMeasureGradientImage, // out
								powf(2.f,fabsf(this->controlPointGrid->intent_p1)), // value
                                3); // division

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
    reg_tools_addSubMulDivValue(this->transformationGradient,
                                forwardScaledGradient,
                                scale,
                                2); // *(scale)

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
        reg_tools_addSubMulDivImages(this->controlPointGrid, // in1
                                     forwardScaledGradient, // in2
                                     this->controlPointGrid, // out
                                     0); // addition
    }
    // Clean the temporary nifti_images
    nifti_image_free(forwardScaledGradient);forwardScaledGradient=NULL;

    /************************/
    /**** Backward update ***/
    /************************/
    // Scale the gradient image
    nifti_image *backwardScaledGradient=nifti_copy_nim_info(this->backwardTransformationGradient);
    backwardScaledGradient->data=(void *)malloc(backwardScaledGradient->nvox*backwardScaledGradient->nbyper);
    reg_tools_addSubMulDivValue(this->backwardTransformationGradient,
                                backwardScaledGradient,
                                scale,
                                2); // *(scale)


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
        reg_tools_addSubMulDivImages(this->backwardControlPointGrid, // in1
                                     backwardScaledGradient, // in2
                                     this->backwardControlPointGrid, // out
                                     0); // addition
    }
    // Clean the temporary nifti_images
    nifti_image_free(backwardScaledGradient);backwardScaledGradient=NULL;

    /****************************/
    /******** Symmetrise ********/
    /****************************/

    // In order to ensure symmetry the forward and backward velocity fields
    // are averaged in both image spaces: reference and floating

    /****************************/
    /* Propagate the forward transformation within the backward space */
    nifti_image *forward2backward = nifti_copy_nim_info(this->backwardControlPointGrid);
    nifti_image *forward2backwardDEF = nifti_copy_nim_info(this->backwardControlPointGrid);
    forward2backward->data=(void *)malloc(forward2backward->nvox*forward2backward->nbyper);
    forward2backwardDEF->data=(void *)calloc(forward2backwardDEF->nvox,forward2backwardDEF->nbyper);

    // Set the deformation field to identity
    reg_tools_addSubMulDivValue(forward2backwardDEF,forward2backwardDEF,0.f,2); // (*0)
    reg_getDeformationFromDisplacement(forward2backwardDEF);

    // Resample the forward grid in the space of the backward grid
    // Set the forward deformation grid to displacement grid in order to
    // enable zero padding
    reg_getDisplacementFromDeformation(this->controlPointGrid);
    reg_resampleImage(this->controlPointGrid, // floating displacement field
                      forward2backward, // warped displacement field
                      forward2backwardDEF, // deformation field
                      NULL, // no mask
                      1, // linear interpolation
                      0.f // padding
                      );
    // Clean the temporary deformation field
    nifti_image_free(forward2backwardDEF);forward2backwardDEF=NULL;

    /****************************/
    /* Propagate the backward transformation within the forward space */
    nifti_image *backward2forward = nifti_copy_nim_info(this->controlPointGrid);
    nifti_image *backward2forwardDEF = nifti_copy_nim_info(this->controlPointGrid);
    backward2forward->data=(void *)malloc(backward2forward->nvox*backward2forward->nbyper);
    backward2forwardDEF->data=(void *)calloc(backward2forwardDEF->nvox,backward2forwardDEF->nbyper);

    // Set the deformation field to identity
    reg_tools_addSubMulDivValue(backward2forwardDEF,backward2forwardDEF,0.f,2); // (*0)
    reg_getDeformationFromDisplacement(backward2forwardDEF);

    // Resample the backward grid in the space of the forward grid
    // Set the backward deformation grid to displacement grid in order to
    // enable zero padding
    reg_getDisplacementFromDeformation(this->backwardControlPointGrid); // in order to use a zero padding
    reg_resampleImage(this->backwardControlPointGrid, // floating displacement field
                      backward2forward, // warped displacement field
                      backward2forwardDEF, // deformation field
                      NULL, // no mask
                      1, // linear interpolation
                      0.f // padding
                      );
    // Clean the temporary deformation field
    nifti_image_free(backward2forwardDEF);backward2forwardDEF=NULL;

    /****************************/
    if(this->forward2backward_reorient!=NULL && this->backward2forward_reorient!=NULL){

        // Reorient the warped grids if necessary
#ifdef _WIN32
        long node, nodeNumber;
#else
        size_t node, nodeNumber;
#endif
        // Reorient the forward grid in the backward space
        nodeNumber=this->backwardControlPointGrid->nx*this->backwardControlPointGrid->ny*this->backwardControlPointGrid->nz;
        T *propVelFieldPtrX=static_cast<T *>(forward2backward->data);
        T *propVelFieldPtrY=&propVelFieldPtrX[nodeNumber];

        mat33 *forward2backward_matrix=this->forward2backward_reorient;

        if(this->backwardControlPointGrid->nz>1){
            T velValues[3];
            T reoriented[3];
            T *propVelFieldPtrZ=&propVelFieldPtrY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(node, reoriented, velValues) \
    shared(nodeNumber, propVelFieldPtrX,propVelFieldPtrY, \
    propVelFieldPtrZ, forward2backward_matrix)
#endif // _OPENMP
            for(node=0;node<nodeNumber;++node){
                velValues[0]=propVelFieldPtrX[node];
                velValues[1]=propVelFieldPtrY[node];
                velValues[2]=propVelFieldPtrZ[node];
                reoriented[0] =
                         forward2backward_matrix->m[0][0] * velValues[0] +
                         forward2backward_matrix->m[0][1] * velValues[1] +
                         forward2backward_matrix->m[0][2] * velValues[2] ;
                reoriented[1] =
                         forward2backward_matrix->m[1][0] * velValues[0] +
                         forward2backward_matrix->m[1][1] * velValues[1] +
                         forward2backward_matrix->m[1][2] * velValues[2] ;
                reoriented[2] =
                         forward2backward_matrix->m[2][0] * velValues[0] +
                         forward2backward_matrix->m[2][1] * velValues[1] +
                         forward2backward_matrix->m[2][2] * velValues[2] ;

                propVelFieldPtrX[node] = reoriented[0];
                propVelFieldPtrY[node] = reoriented[1];
                propVelFieldPtrZ[node] = reoriented[2];
            }
        }
        else{
            T velValues[2];
            T reoriented[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(node, reoriented, velValues) \
    shared(nodeNumber, propVelFieldPtrX,propVelFieldPtrY, \
           forward2backward_matrix)
#endif // _OPENMP
            for(node=0;node<nodeNumber;++node)            {
                velValues[0]=propVelFieldPtrX[node];
                velValues[1]=propVelFieldPtrY[node];
                reoriented[0] =
                         forward2backward_matrix->m[0][0] * velValues[0] +
                         forward2backward_matrix->m[0][1] * velValues[1] ;
                reoriented[1] =
                         forward2backward_matrix->m[1][0] * velValues[0] +
                         forward2backward_matrix->m[1][1] * velValues[1] ;

                propVelFieldPtrX[node] = reoriented[0];
                propVelFieldPtrY[node] = reoriented[1];
            }
        }

        // Reorient the backward grid in the forward space
        nodeNumber=this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz;
        propVelFieldPtrX=static_cast<T *>(backward2forward->data);
        propVelFieldPtrY=&propVelFieldPtrX[nodeNumber];

        mat33 *backward2forward_matrix=this->backward2forward_reorient;

        if(this->controlPointGrid->nz>1){
            T velValues[3];
            T reoriented[3];
            T *propVelFieldPtrZ=&propVelFieldPtrY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(node, reoriented, velValues) \
    shared(nodeNumber, propVelFieldPtrX,propVelFieldPtrY, \
    propVelFieldPtrZ, backward2forward_matrix)
#endif // _OPENMP
            for(node=0;node<nodeNumber;++node){
                velValues[0]=propVelFieldPtrX[node];
                velValues[1]=propVelFieldPtrY[node];
                velValues[2]=propVelFieldPtrZ[node];
                reoriented[0] =
                         backward2forward_matrix->m[0][0] * velValues[0] +
                         backward2forward_matrix->m[0][1] * velValues[1] +
                         backward2forward_matrix->m[0][2] * velValues[2] ;
                reoriented[1] =
                         backward2forward_matrix->m[1][0] * velValues[0] +
                         backward2forward_matrix->m[1][1] * velValues[1] +
                         backward2forward_matrix->m[1][2] * velValues[2] ;
                reoriented[2] =
                         backward2forward_matrix->m[2][0] * velValues[0] +
                         backward2forward_matrix->m[2][1] * velValues[1] +
                         backward2forward_matrix->m[2][2] * velValues[2] ;

                propVelFieldPtrX[node] = reoriented[0];
                propVelFieldPtrY[node] = reoriented[1];
                propVelFieldPtrZ[node] = reoriented[2];
            }
        }
        else{
            T velValues[2];
            T reoriented[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(node, reoriented, velValues) \
    shared(nodeNumber, propVelFieldPtrX,propVelFieldPtrY, \
    backward2forward_matrix)
#endif // _OPENMP
            for(node=0;node<nodeNumber;++node){
                velValues[0]=propVelFieldPtrX[node];
                velValues[1]=propVelFieldPtrY[node];
                reoriented[0] =
                         backward2forward_matrix->m[0][0] * velValues[0] +
                         backward2forward_matrix->m[0][1] * velValues[1] ;
                reoriented[1] =
                         backward2forward_matrix->m[1][0] * velValues[0] +
                         backward2forward_matrix->m[1][1] * velValues[1] ;

                propVelFieldPtrX[node] = reoriented[0];
                propVelFieldPtrY[node] = reoriented[1];
            }
        }
    } // End - reorient the warped grids if necessary
    /****************************/

    /* Average velocity fields into forward and backward space */
    // Substraction as the propagated has to be negated
    reg_tools_addSubMulDivImages(this->backwardControlPointGrid, // displacement
                                 forward2backward, // displacement
                                 this->backwardControlPointGrid, // displacement output
                                 1); // substraction
    reg_tools_addSubMulDivImages(this->controlPointGrid, // displacement
                                 backward2forward, // displacement
                                 this->controlPointGrid, // displacement output
                                 1); // substraction
    // Division by 2
    reg_tools_addSubMulDivValue(this->backwardControlPointGrid, // displacement
                                this->backwardControlPointGrid, // displacement
                                0.5f,
                                2); // *(0.5)
    reg_tools_addSubMulDivValue(this->controlPointGrid, // displacement
                                this->controlPointGrid, // displacement
                                0.5f,
                                2); // *(0.5)
    // Clean the temporary allocated velocity field
    nifti_image_free(forward2backward);forward2backward=NULL;
    nifti_image_free(backward2forward);backward2forward=NULL;
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
