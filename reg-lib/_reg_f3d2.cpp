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

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_f3d2<T>::reg_f3d2(int refTimePoint,int floTimePoint)
   :reg_f3d_sym<T>::reg_f3d_sym(refTimePoint,floTimePoint)
{
   this->executableName=(char *)"NiftyReg F3D2";
   this->inverseConsistencyWeight=0;
   this->BCHUpdate=false;
   this->useGradientCumulativeExp=true;
   this->BCHUpdateValue=0;

#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2 constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_f3d2<T>::~reg_f3d2()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2 destructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UseBCHUpdate(int v)
{
   this->BCHUpdate = true;
   this->useGradientCumulativeExp = false;
   this->BCHUpdateValue=v;
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::UseGradientCumulativeExp()
{
   this->BCHUpdate = false;
   this->useGradientCumulativeExp = true;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::DoNotUseGradientCumulativeExp()
{
   this->useGradientCumulativeExp = false;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d2<T>::Initialise()
{
   reg_f3d_sym<T>::Initialise();

   // Convert the control point grid into velocity field parametrisation
   this->controlPointGrid->intent_p1=SPLINE_VEL_GRID;
   this->backwardControlPointGrid->intent_p1=SPLINE_VEL_GRID;
   // Set the number of composition to 6 by default
   this->controlPointGrid->intent_p2=6;
   this->backwardControlPointGrid->intent_p2=6;

#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2::Initialise_f3d() done");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetDeformationField()
{
   // By default the number of steps is automatically updated
   bool updateStepNumber=true;
   // The provided step number is used for the final resampling
   if(this->optimiser==NULL)
      updateStepNumber=false;
#ifndef NDEBUG
   char text[255];
   sprintf(text, "Velocity integration forward. Step number update=%i",updateStepNumber);
   reg_print_msg_debug(text);
#endif
   // The forward transformation is computed using the scaling-and-squaring approach
   reg_spline_getDefFieldFromVelocityGrid(this->controlPointGrid,
                                          this->deformationFieldImage,
                                          updateStepNumber
                                          );
#ifndef NDEBUG
   sprintf(text, "Velocity integration backward. Step number update=%i",updateStepNumber);
   reg_print_msg_debug(text);
#endif
   // The number of step number is copied over from the forward transformation
   this->backwardControlPointGrid->intent_p2=this->controlPointGrid->intent_p2;
   // The backward transformation is computed using the scaling-and-squaring approach
   reg_spline_getDefFieldFromVelocityGrid(this->backwardControlPointGrid,
                                          this->backwardDeformationFieldImage,
                                          false
                                          );
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyErrorField(bool forceAll)
{
   if(this->inverseConsistencyWeight<=0) return;

   if(forceAll)
   {
      reg_print_fct_error("reg_f3d2<T>::GetInverseConsistencyErrorField()");
      reg_print_msg_error("Option not supported in F3D2");
   }
   else
   {
      reg_print_fct_error("reg_f3d2<T>::GetInverseConsistencyErrorField()");
      reg_print_msg_error("Option not supported in F3D2");
   }
   reg_exit(1);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyGradient()
{
   if(this->inverseConsistencyWeight<=0) return;

   reg_print_fct_error("reg_f3d2<T>::GetInverseConsistencyGradient()");
   reg_print_msg_error("Option not supported in F3D2");
   reg_exit(1);

   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetVoxelBasedGradient()
{
   reg_f3d_sym<T>::GetVoxelBasedGradient();

   // Exponentiate the gradients if required
   this->ExponentiateGradient();
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::ExponentiateGradient()
{
   if(!this->useGradientCumulativeExp) return;

   /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
   // Exponentiate the forward gradient using the backward transformation
#ifndef NDEBUG
   reg_print_msg_debug("Update the forward measure gradient using a Dartel like approach");
#endif
   // Create all deformation field images needed for resampling
   nifti_image **tempDef=(nifti_image **)malloc(
                            (unsigned int)(fabs(this->backwardControlPointGrid->intent_p2)+1) *
                            sizeof(nifti_image *));
   for(unsigned int i=0; i<=(unsigned int)fabs(this->backwardControlPointGrid->intent_p2); ++i)
   {
      tempDef[i]=nifti_copy_nim_info(this->deformationFieldImage);
      tempDef[i]->data=(void *)malloc(tempDef[i]->nvox*tempDef[i]->nbyper);
   }
   // Generate all intermediate deformation fields
   reg_spline_getIntermediateDefFieldFromVelGrid(this->backwardControlPointGrid,
         tempDef);

   // Remove the affine component
   nifti_image *affine_disp=NULL;
   if(this->affineTransformation!=NULL){
      affine_disp=nifti_copy_nim_info(this->deformationFieldImage);
      affine_disp->data=(void *)malloc(affine_disp->nvox*affine_disp->nbyper);
      mat44 backwardAffineTransformation=nifti_mat44_inverse(*this->affineTransformation);
      reg_affine_getDeformationField(&backwardAffineTransformation,
                                     affine_disp);
      reg_getDisplacementFromDeformation(affine_disp);
   }

   /* Allocate a temporary gradient image to store the backward gradient */
   nifti_image *tempGrad=nifti_copy_nim_info(this->voxelBasedMeasureGradientImage);

   tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
   for(int i=0; i<(int)fabsf(this->backwardControlPointGrid->intent_p2); ++i)
   {
      reg_tools_substractImageToImage(tempDef[i],
                                      affine_disp,
                                      tempDef[i]);
      reg_resampleGradient(this->voxelBasedMeasureGradientImage, // floating
                           tempGrad, // warped - out
                           tempDef[i], // deformation field
                           1, // interpolation type - linear
                           0.f); // padding value
      reg_tools_addImageToImage(tempGrad, // in1
                                this->voxelBasedMeasureGradientImage, // in2
                                this->voxelBasedMeasureGradientImage); // out
   }

   // Free the temporary deformation fields
   for(int i=0; i<=(int)fabsf(this->backwardControlPointGrid->intent_p2); ++i)
   {
      nifti_image_free(tempDef[i]);
      tempDef[i]=NULL;
   }
   free(tempDef);
   tempDef=NULL;
   // Free the temporary gradient image
   nifti_image_free(tempGrad);
   tempGrad=NULL;
   // Free the temporary affine displacement field
   if(affine_disp!=NULL)
      nifti_image_free(affine_disp);
   affine_disp=NULL;
   // Normalise the forward gradient
   reg_tools_divideValueToImage(this->voxelBasedMeasureGradientImage, // in
                                this->voxelBasedMeasureGradientImage, // out
                                powf(2.f,fabsf(this->backwardControlPointGrid->intent_p2))); // value

   /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
   /* Exponentiate the backward gradient using the forward transformation */
#ifndef NDEBUG
   reg_print_msg_debug("Update the backward measure gradient using a Dartel like approach");
#endif
   // Allocate a temporary gradient image to store the backward gradient
   tempGrad=nifti_copy_nim_info(this->backwardVoxelBasedMeasureGradientImage);
   tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
   // Create all deformation field images needed for resampling
   tempDef=(nifti_image **)malloc((unsigned int)(fabs(this->controlPointGrid->intent_p2)+1) * sizeof(nifti_image *));
   for(unsigned int i=0; i<=(unsigned int)fabs(this->controlPointGrid->intent_p2); ++i)
   {
      tempDef[i]=nifti_copy_nim_info(this->backwardDeformationFieldImage);
      tempDef[i]->data=(void *)malloc(tempDef[i]->nvox*tempDef[i]->nbyper);
   }
   // Generate all intermediate deformation fields
   reg_spline_getIntermediateDefFieldFromVelGrid(this->controlPointGrid,
         tempDef);

   // Remove the affine component
   if(this->affineTransformation!=NULL){
      affine_disp=nifti_copy_nim_info(this->backwardDeformationFieldImage);
      affine_disp->data=(void *)malloc(affine_disp->nvox*affine_disp->nbyper);
      reg_affine_getDeformationField(this->affineTransformation,
                                     affine_disp);
      reg_getDisplacementFromDeformation(affine_disp);
   }

   for(int i=0; i<(int)fabsf(this->controlPointGrid->intent_p2); ++i)
   {
      reg_tools_substractImageToImage(tempDef[i],
                                      affine_disp,
                                      tempDef[i]);
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
   for(int i=0; i<=(int)fabsf(this->controlPointGrid->intent_p2); ++i)
   {
      nifti_image_free(tempDef[i]);
      tempDef[i]=NULL;
   }
   free(tempDef);
   tempDef=NULL;
   // Free the temporary gradient image
   nifti_image_free(tempGrad);
   tempGrad=NULL;
   // Free the temporary affine displacement field
   if(affine_disp!=NULL)
      nifti_image_free(affine_disp);
   affine_disp=NULL;
   // Normalise the backward gradient
   reg_tools_divideValueToImage(this->backwardVoxelBasedMeasureGradientImage, // in
                                this->backwardVoxelBasedMeasureGradientImage, // out
                                powf(2.f,fabsf(this->controlPointGrid->intent_p2))); // value

   return;
}
/* *************************************************************** */
/* *************************************************************** */
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
   if(this->BCHUpdate)
   {
      // Compute the BCH update
      reg_print_msg_warn("USING BCH FORWARD - TESTING ONLY");
#ifndef NDEBUG
      reg_print_msg_debug("Update the forward control point grid using BCH approximation");
#endif
      compute_BCH_update(this->controlPointGrid,
                         forwardScaledGradient,
                         this->BCHUpdateValue);
   }
   else
   {
      // Update the velocity field
      reg_tools_addImageToImage(this->controlPointGrid, // in1
                                forwardScaledGradient, // in2
                                this->controlPointGrid); // out
   }
   // Clean the temporary nifti_images
   nifti_image_free(forwardScaledGradient);
   forwardScaledGradient=NULL;

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
   if(this->BCHUpdate)
   {
      // Compute the BCH update
      reg_print_msg_warn("USING BCH BACKWARD - TESTING ONLY");
#ifndef NDEBUG
      reg_print_msg_debug("Update the backward control point grid using BCH approximation");
#endif
      compute_BCH_update(this->backwardControlPointGrid,
                         backwardScaledGradient,
                         this->BCHUpdateValue);
   }
   else
   {
      // Update the velocity field
      reg_tools_addImageToImage(this->backwardControlPointGrid, // in1
                                backwardScaledGradient, // in2
                                this->backwardControlPointGrid); // out
   }
   // Clean the temporary nifti_images
   nifti_image_free(backwardScaledGradient);
   backwardScaledGradient=NULL;

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

   // Both parametrisations are copied over
   memcpy(warpedBackwardTrans->data,this->backwardControlPointGrid->data,warpedBackwardTrans->nvox*warpedBackwardTrans->nbyper);
   memcpy(warpedForwardTrans->data,this->controlPointGrid->data,warpedForwardTrans->nvox*warpedForwardTrans->nbyper);

   // and substracted (sum and negation)
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
   nifti_image_free(warpedForwardTrans);
   warpedForwardTrans=NULL;
   nifti_image_free(warpedBackwardTrans);
   warpedBackwardTrans=NULL;

   // Convert the velocity field from displacement to deformation
   reg_getDeformationFromDisplacement(this->controlPointGrid);
   reg_getDeformationFromDisplacement(this->backwardControlPointGrid);

   return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
nifti_image **reg_f3d2<T>::GetWarpedImage()
{
   // The initial images are used
   if(this->inputReference==NULL ||
         this->inputFloating==NULL ||
         this->controlPointGrid==NULL ||
         this->backwardControlPointGrid==NULL)
   {
      reg_print_fct_error("reg_f3d2<T>::GetWarpedImage()");
      reg_print_msg_error("The reference, floating and control point grid images have to be defined");
      reg_exit(1);
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
/* *************************************************************** */
/* *************************************************************** */
template class reg_f3d2<float>;
#endif
