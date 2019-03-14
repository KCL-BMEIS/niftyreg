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
#include <algorithm>  // for std::copy
#include <cassert>

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
   reg_spline_getDefFieldFromVelocityGrid(this->controlPointGrid,  // in
                                          this->deformationFieldImage,  // out
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
   reg_exit();
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetInverseConsistencyGradient()
{
   if(this->inverseConsistencyWeight<=0) return;

   reg_print_fct_error("reg_f3d2<T>::GetInverseConsistencyGradient()");
   reg_print_msg_error("Option not supported in F3D2");
   reg_exit();

   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::GetVoxelBasedGradient()
{
   reg_f3d_sym<T>::GetVoxelBasedGradient();

   // Exponentiate the gradients of the data term if required
   // (gradients of the rgularisation are not exponentiated)
   this->ExponentiateGradient();
   // Compute the gradient of the exponential of the velocity field
   this->LucasExpGradient();
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d2<T>::ExponentiateGradient()
/*
 * Exponentiation of the gradient inspired by DARTEL algorithm [1].
 *
 * [1] A fast diffeomorphic image registration algorithm, J. Ashburner, NeuroImage 2007.
 *
 */
{
   if(!this->useGradientCumulativeExp) return;

   /* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ */
   // Exponentiate the forward gradient using the backward transformation
#ifndef NDEBUG
   reg_print_msg_debug("Update the forward measure gradient using a Dartel like approach");
#endif
   // Create all deformation field images needed for resampling
   // see [1] eq (46)
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

   /* Allocate a temporary gradient image to store the forward gradient */
   nifti_image *tempGrad=nifti_copy_nim_info(this->voxelBasedMeasureGradient);

   tempGrad->data=(void *)malloc(tempGrad->nvox*tempGrad->nbyper);
   for(int i=0; i<(int)fabsf(this->backwardControlPointGrid->intent_p2); ++i)
   // voxelBasedMeasureGradient first corresponds to 2^K * b^(0) in [1] eq (39)
   {
      if(affine_disp!=NULL)
         reg_tools_substractImageToImage(tempDef[i],
                                         affine_disp,
                                         tempDef[i]);
      // warp the current gradient
      // second term of the right hand-side of eq (44) in [1]
      // except the jacobian seems to be missing...
      reg_resampleGradient(this->voxelBasedMeasureGradient, // floating
                           tempGrad, // warped - out
                           tempDef[i], // deformation field
                           1, // interpolation type - linear
                           0.f); // padding value
      // addition in [1] eq (44)
      reg_tools_addImageToImage(tempGrad, // in1
                                this->voxelBasedMeasureGradient, // in2
                                this->voxelBasedMeasureGradient); // out
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
   // 1/2^K term in [1] eq (39)
   reg_tools_divideValueToImage(this->voxelBasedMeasureGradient, // in
                                this->voxelBasedMeasureGradient, // out
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
      if(affine_disp!=NULL)
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
void reg_f3d2<T>::LucasExpGradient() {
    /*
     * Gradient of the exponential as computed in the scaling and squaring algorithm
     * using a linear interpolation [1].
     *
     * [1] The Phase Flow Method, Ying L. and Candes E.J., Journal of Computational Physics, 2006.
     *
    */
    if(!this->useLucasExpGradient) return;

    size_t voxelNumber = (size_t)this->currentReference->nx*this->currentReference->ny*this->currentReference->nz;

    // Create all forward and backward (dense) deformation field needed for resampling:
    // Initialise the forward deformation fields: phi^(0), phi^(1), ..., phi^(K)
    // TOOPTIMISE: In fact we don't use the last one... could be moved to GetObjectiveFunctionGradient
    // to avoid to do the scaling and squaring twice (in WarpFloatingImage)
    nifti_image **tempForDef=(nifti_image **)malloc(
            (unsigned int)(fabs(this->controlPointGrid->intent_p2)+1) * sizeof(nifti_image *));
    for(unsigned int i=0; i<=(unsigned int)fabs(this->controlPointGrid->intent_p2); ++i) {
        tempForDef[i]=nifti_copy_nim_info(this->deformationFieldImage);
        tempForDef[i]->data=(void *)malloc(tempForDef[i]->nvox*tempForDef[i]->nbyper);
    }
    // Initialise the backward deformation fields
    nifti_image **tempBackDef=(nifti_image **)malloc(
            (unsigned int)(fabs(this->backwardControlPointGrid->intent_p2)+1) * sizeof(nifti_image *));
    for(unsigned int i=0; i<=(unsigned int)fabs(this->backwardControlPointGrid->intent_p2); ++i) {
        tempBackDef[i]=nifti_copy_nim_info(this->deformationFieldImage);
        tempBackDef[i]->data=(void *)malloc(tempBackDef[i]->nvox*tempBackDef[i]->nbyper);
    }
    // Generate all intermediate deformation fields
    reg_spline_getIntermediateDefFieldFromVelGrid(this->controlPointGrid,
                                                  tempForDef);
    reg_spline_getIntermediateDefFieldFromVelGrid(this->backwardControlPointGrid,
                                                  tempBackDef);

    // Assume there is no affine transformation.
    // Need to test if it would work with an affine transformation component
    assert(this->affineTransformation==NULL);
    nifti_image *affine_back_disp=NULL;
    nifti_image *affine_for_disp=NULL;
    if (this->affineTransformation!=NULL) {
        // Allocate the affine component that need to be removed to the backward deformations
        affine_back_disp = nifti_copy_nim_info(this->deformationFieldImage);
        affine_back_disp->data = (void *) malloc(affine_back_disp->nvox * affine_back_disp->nbyper);
        mat44 backwardAffineTransformation = nifti_mat44_inverse(*this->affineTransformation);
        reg_affine_getDeformationField(&backwardAffineTransformation,
                                       affine_back_disp);
        reg_getDisplacementFromDeformation(affine_back_disp);
        // Allocate the affine component that need to be removed to the forward deformations
        affine_for_disp = nifti_copy_nim_info(this->backwardDeformationFieldImage);
        affine_for_disp->data = (void *) malloc(affine_for_disp->nvox * affine_for_disp->nbyper);
        reg_affine_getDeformationField(this->affineTransformation,
                                       affine_for_disp);
        reg_getDisplacementFromDeformation(affine_for_disp);
    }
    // remove affine components of necessary
    if (affine_for_disp!=NULL) {
        for(unsigned int i=0; i<=(unsigned int)fabs(this->controlPointGrid->intent_p2); ++i) {
            reg_tools_substractImageToImage(tempForDef[i],
                                            affine_for_disp,
                                            tempForDef[i]);
        }
    }
    if (affine_back_disp!=NULL) {
        for(unsigned int i=0; i<=(unsigned int)fabs(this->backwardControlPointGrid->intent_p2); ++i) {
            reg_tools_substractImageToImage(tempBackDef[i],
                                            affine_back_disp,
                                            tempBackDef[i]);
        }
    }

#ifndef NDEBUG
    reg_print_msg_debug("Backpropagate the gradient throw the scaling and squaring operations of the forward transformation.");
#endif
    /* Allocate temporary gradient images to store the forward gradient terms */
    nifti_image *tempPrevGrad=nifti_copy_nim_info(this->voxelBasedMeasureGradient);
    tempPrevGrad->data=(void *)malloc(tempPrevGrad->nvox*tempPrevGrad->nbyper);

    nifti_image *tempGrad1=nifti_copy_nim_info(this->voxelBasedMeasureGradient);
//    nifti_image *tempGrad2=nifti_copy_nim_info(this->voxelBasedMeasureGradient);
//    nifti_image *tempGrad3=nifti_copy_nim_info(this->voxelBasedMeasureGradient);
    tempGrad1->data=(void *)malloc(tempGrad1->nvox*tempGrad1->nbyper);
//    tempGrad2->data=(void *)malloc(tempGrad2->nvox*tempGrad2->nbyper);
//    tempGrad3->data=(void *)malloc(tempGrad3->nvox*tempGrad3->nbyper);

    // Create two temporary images to store gradient images projected on a single channel
    nifti_image *tempGradChannel = nifti_copy_nim_info(this->currentReference);
    nifti_image *tempDefChannel = nifti_copy_nim_info(this->currentReference);
    tempGradChannel->data=(void *)malloc(tempGradChannel->nvox * tempGradChannel->nbyper);
    tempDefChannel->data=(void *)malloc(tempDefChannel->nvox * tempDefChannel->nbyper);

    ////////////////////////////////////////////////
    // backpropagate the forward gradient
    for(int k=(int)fabsf(this->controlPointGrid->intent_p2)-1; k>=0; --k) {
        // Set temporary gradient to previous value of the gradient
        reg_tools_multiplyValueToImage(tempPrevGrad, tempPrevGrad, 0.f);
        reg_tools_addImageToImage(tempPrevGrad, // in1
                                  this->voxelBasedMeasureGradient, // in2
                                  tempPrevGrad); // out

        // First term of the propagated forward gradient (similar to exponentiateGradient)
        // warp the current gradient to get the first term of the propagated gradient
        reg_resampleGradient(tempPrevGrad, // floating
                             this->voxelBasedMeasureGradient, // out
                             tempBackDef[k], // deformation field
                             1, // interpolation type -> linear
                             0.f); // padding value
//        std::cout << "gradient term 1 success" << std::endl;

        //// PRINTS TO REMOVE
//        std::cout.precision(17);
//        T maxGrad = 0;
//        T *warpImgGradient = static_cast<T *>(this->warImgGradient->data);
//    T *backWarpImgGradient = static_cast<T *>(this->backwardWarpedGradientImage->data);
//        T *warpImgSimGradient = static_cast<T *>(this->voxelBasedMeasureGradient->data);
//    T *warpBackImgSimGradient = static_cast<T *>(this->backwardVoxelBasedMeasureGradientImage->data);
//        for (int i = 0; i < this->voxelBasedMeasureGradient->nvox; i++) {
//            if (fabs(warpImgSimGradient[i]) > maxGrad) {
//                maxGrad = fabs(warpImgSimGradient[i]);
//            }
//            std::cout << "warp_img_grad[" << i << "] = " << std::fixed << warpImgGradient[i] << std::endl;
//            std::cout << "exp_voxel_based_measure_grad[" << i << "] = " << std::fixed << warpImgSimGradient[i] << std::endl;
//        }
//        std::cout << "max ExpVoxelBasedMeasureGrad = " << maxGrad << std::endl;





        // Second term of the propagated forward gradient
        // sum over the channels of the gradient
        // X channel
//        tempGradChannel->data = &tempPrevGrad->data;
//        tempDefChannel->data = &tempForDef[k]->data;
        std::copy(static_cast<T *>(tempPrevGrad->data),  // start array to copy
                  static_cast<T *>(tempPrevGrad->data) + voxelNumber,  // end array to copy
                  static_cast<T *>(tempGradChannel->data));  // start array out
        std::copy(static_cast<T *>(tempForDef[k]->data),  // start array to copy
                  static_cast<T *>(tempForDef[k]->data) + voxelNumber,  // end array to copy
                  static_cast<T *>(tempDefChannel->data));  // start array out

        reg_getImageGradient(tempDefChannel, // image input
                             tempGrad1, // out
                             tempForDef[k], // deformation - phi^(k)
                             NULL,  // this->currentMask,
                             1,  // interp - linear
                             0,  // this->warpedPaddingValue,
                             0); // time point
        reg_tools_multiplyImageToGradient(tempGradChannel,  // (scalar) image in
                                          tempGrad1,  // gradient image in
                                          tempGrad1);  // gradient image out
        reg_io_WriteImageFile(tempGrad1, "tempGrad1_exp.nii");
//        reg_tools_addImageToImage(tempGrad1,  // in
//                                  this->voxelBasedMeasureGradient,  // in
//                                  this->voxelBasedMeasureGradient);  // out


        //// PRINTS TO REMOVE
//        std::cout.precision(17);
//        T maxGrad = 0;
//        T *warpImgGradient = static_cast<T *>(this->warImgGradient->data);
//        T *prevGrad = static_cast<T *>(tempPrevGrad->data);
//        T *backWarpImgGradient = static_cast<T *>(this->backwardWarpedGradientImage->data);
//        T *warpImgSimGradient = static_cast<T *>(this->voxelBasedMeasureGradient->data);
//        T *warpBackImgSimGradient = static_cast<T *>(this->backwardVoxelBasedMeasureGradientImage->data);
//        for (int i = 0; i < this->voxelBasedMeasureGradient->nvox; i++) {
//            if (fabs(warpImgSimGradient[i]) > maxGrad) {
//                maxGrad = fabs(warpImgSimGradient[i]);
//            }
//            std::cout << "warp_img_grad[" << i << "] = " << std::fixed << warpImgGradient[i] << std::endl;
//            std::cout << "prev_grad[" << i << "] = " << std::fixed << prevGrad[i] << std::endl;
//            std::cout << "exp_voxel_based_measure_grad[" << i << "] = " << std::fixed << warpImgSimGradient[i] << std::endl;
//        }
//        std::cout << "max ExpVoxelBasedMeasureGrad = " << maxGrad << std::endl;


//        std::cout << "x axis success" << std::endl;
        // Y channel
//        tempGradChannel->data = (void *) &static_cast<T *>(tempGradChannel->data)[voxelNumber];
//        tempDefChannel->data = (void *) &static_cast<T *>(tempDefChannel->data)[voxelNumber];
        std::copy(static_cast<T *>(tempPrevGrad->data) + voxelNumber,  // start array to copy
                  static_cast<T *>(tempPrevGrad->data) + 2*voxelNumber,  // end array to copy
                  static_cast<T *>(tempGradChannel->data));  // start array out
        std::copy(static_cast<T *>(tempForDef[k]->data) + voxelNumber,  // start array to copy
                  static_cast<T *>(tempForDef[k]->data) + 2*voxelNumber,  // end array to copy
                  static_cast<T *>(tempDefChannel->data));  // start array out
        reg_getImageGradient(tempDefChannel, // image input
                             tempGrad1, // out
                             tempForDef[k], // deformation - phi^(k)
                             this->currentMask,
                             1,  // interp - linear
                             0, // this->warpedPaddingValue,
                             0); // time point
        reg_tools_multiplyImageToGradient(tempGradChannel,  // (scalar) image in
                                          tempGrad1,  // gradient image in
                                          tempGrad1);  // gradient image out
//        reg_tools_addImageToImage(tempGrad1,  // in
//                                  this->voxelBasedMeasureGradient,  // in
//                                  this->voxelBasedMeasureGradient);  // out
        // Z axis
        if (this->currentReference->nz>1) {
//            tempGradChannel->data = (void *) &static_cast<T *>(tempGradChannel->data)[voxelNumber];
//            tempDefChannel->data = (void *) &static_cast<T *>(tempDefChannel->data)[voxelNumber];
            std::copy(static_cast<T *>(tempPrevGrad->data) + 2*voxelNumber,  // start array to copy
                      static_cast<T *>(tempPrevGrad->data) + 3*voxelNumber,  // end array to copy
                      static_cast<T *>(tempGradChannel->data));  // start array out
            std::copy(static_cast<T *>(tempForDef[k]->data) + 2*voxelNumber,  // start array to copy
                      static_cast<T *>(tempForDef[k]->data) + 3*voxelNumber,  // end array to copy
                      static_cast<T *>(tempDefChannel->data));  // start array out
            reg_getImageGradient(tempDefChannel, // image input
                                 tempGrad1, // out
                                 tempForDef[k], // deformation - phi^(k)
                                 this->currentMask,
                                 1,  // interp - linear
                                 0,  // this->warpedPaddingValue,
                                 0); // time point
            reg_tools_multiplyImageToGradient(tempGradChannel,  // (scalar) image in
                                              tempGrad1,  // gradient image in
                                              tempGrad1);  // gradient image out
//            reg_tools_addImageToImage(tempGrad1,  // in
//                                      this->voxelBasedMeasureGradient,  // in
//                                      this->voxelBasedMeasureGradient);  // out
        }
    }
    // Normalise the forward gradient (divide by 2^K)
    reg_tools_divideValueToImage(this->voxelBasedMeasureGradient, // in
                                 this->voxelBasedMeasureGradient, // out
                                 powf(2.f,fabsf(this->controlPointGrid->intent_p2))); // value

    //// PRINTS TO REMOVE
//    std::cout.precision(17);
//    T maxGrad = 0;
//    T *warpImgGradient = static_cast<T *>(this->warImgGradient->data);
//    T *backWarpImgGradient = static_cast<T *>(this->backwardWarpedGradientImage->data);
//    T *warpImgSimGradient = static_cast<T *>(this->voxelBasedMeasureGradient->data);
//    T *warpBackImgSimGradient = static_cast<T *>(this->backwardVoxelBasedMeasureGradientImage->data);
//    for (int i = 0; i < this->voxelBasedMeasureGradient->nvox; i++) {
//        if (fabs(warpImgSimGradient[i]) > maxGrad) {
//            maxGrad = fabs(warpImgSimGradient[i]);
//        }
//        std::cout << "warp_img_grad[" << i << "] = " << std::fixed << warpImgGradient[i] << std::endl;
//        std::cout << "exp_voxel_based_measure_grad[" << i << "] = " << std::fixed << warpImgSimGradient[i] << std::endl;
//    }
//    std::cout << "max ExpVoxelBasedMeasureGrad = " << maxGrad << std::endl;


    /////////////////////////////////
#ifndef NDEBUG
    reg_print_msg_debug("Backpropagate the gradient throw the scaling and squaring operations of the backward transformation.");
#endif
    /////////////////////////////////////////////
    // backpropagate the backward gradient
    for(int k=(int)fabsf(this->backwardControlPointGrid->intent_p2)-1; k>=0; --k) {
        // Set temporary gradient to previous value of the gradient
        reg_tools_multiplyValueToImage(tempPrevGrad, tempPrevGrad, 0.f);
        reg_tools_addImageToImage(tempPrevGrad, // in1
                                  this->backwardVoxelBasedMeasureGradientImage, // in2
                                  tempPrevGrad); // out

        // First term of the propagated forward gradient
        // warp the current gradient to get the first term of the propagated gradient
        reg_resampleGradient(tempPrevGrad, // floating
                             this->backwardVoxelBasedMeasureGradientImage, // out
                             tempForDef[k], // deformation field
                             1, // interpolation type -> linear
                             0.f); // padding value

        // Second term of the propagated forward gradient
        // sum over the channels of the gradient
        // X channel
//        tempGradChannel->data = &tempPrevGrad->data;
//        tempDefChannel->data = &tempBackDef[k]->data;
        std::copy(static_cast<T *>(tempPrevGrad->data),  // start array to copy
                  static_cast<T *>(tempPrevGrad->data) + voxelNumber,  // end array to copy
                  static_cast<T *>(tempGradChannel->data));  // start array out
        std::copy(static_cast<T *>(tempBackDef[k]->data),  // start array to copy
                  static_cast<T *>(tempBackDef[k]->data) + voxelNumber,  // end array to copy
                  static_cast<T *>(tempDefChannel->data));  // start array out
        reg_getImageGradient(tempDefChannel, // image input
                             tempGrad1, // out
                             tempBackDef[k], // deformation - phi^(-k)
                             this->currentFloatingMask,
                             1,  // interp - linear
                             this->warpedPaddingValue,
                             0); // time point
        reg_tools_multiplyImageToGradient(tempGradChannel,  // (scalar) image in
                                          tempGrad1,  // gradient image in
                                          tempGrad1);  // gradient image out
//        reg_tools_addImageToImage(tempGrad1,  // in
//                                  this->backwardVoxelBasedMeasureGradientImage,  // in
//                                  this->backwardVoxelBasedMeasureGradientImage);  // out
        // Y channel
//        tempGradChannel->data = (void *) &static_cast<T *>(tempPrevGrad->data)[voxelNumber];
//        tempDefChannel->data = (void *) &static_cast<T *>(tempBackdef[k]->data)[voxelNumber];
        std::copy(static_cast<T *>(tempPrevGrad->data) + voxelNumber,  // start array to copy
                  static_cast<T *>(tempPrevGrad->data) + 2*voxelNumber,  // end array to copy
                  static_cast<T *>(tempGradChannel->data));  // start array out
        std::copy(static_cast<T *>(tempBackDef[k]->data) + voxelNumber,  // start array to copy
                  static_cast<T *>(tempBackDef[k]->data) + 2*voxelNumber,  // end array to copy
                  static_cast<T *>(tempDefChannel->data));  // start array out
        reg_getImageGradient(tempDefChannel, // image input
                             tempGrad1, // out
                             tempBackDef[k], // deformation - phi^(-k)
                             this->currentFloatingMask,
                             1,  // interp - linear
                             this->warpedPaddingValue,
                             0); // time point
        reg_tools_multiplyImageToGradient(tempGradChannel,  // (scalar) image in
                                          tempGrad1,  // gradient image in
                                          tempGrad1);  // gradient image out
//        reg_tools_addImageToImage(tempGrad1,  // in
//                                  this->backwardVoxelBasedMeasureGradientImage,  // in
//                                  this->backwardVoxelBasedMeasureGradientImage);  // out
        // Z axis
        if (this->currentReference->nz>1) {
//            tempGradChannel->data = (void *) &static_cast<T *>(tempGradChannel->data)[voxelNumber];
//            tempDefChannel->data = (void *) &static_cast<T *>(tempDefChannel->data)[voxelNumber];
            std::copy(static_cast<T *>(tempPrevGrad->data) + 2*voxelNumber,  // start array to copy
                      static_cast<T *>(tempPrevGrad->data) + 3*voxelNumber,  // end array to copy
                      static_cast<T *>(tempGradChannel->data));  // start array out
            std::copy(static_cast<T *>(tempBackDef[k]->data) + 2*voxelNumber,  // start array to copy
                      static_cast<T *>(tempBackDef[k]->data) + 3*voxelNumber,  // end array to copy
                      static_cast<T *>(tempDefChannel->data));  // start array out
            reg_getImageGradient(tempDefChannel, // image input
                                 tempGrad1, // out
                                 tempBackDef[k], // deformation - phi^(-k)
                                 this->currentFloatingMask,
                                 1,  // interp - linear
                                 this->warpedPaddingValue,
                                 0); // time point
            reg_tools_multiplyImageToGradient(tempGradChannel,  // (scalar) image in
                                              tempGrad1,  // gradient image in
                                              tempGrad1);  // gradient image out
//            reg_tools_addImageToImage(tempGrad1,  // in
//                                      this->backwardVoxelBasedMeasureGradientImage,  // in
//                                      this->backwardVoxelBasedMeasureGradientImage);  // out
        }
    }
    // Normalise the forward gradient (divide by 2^K)
    reg_tools_divideValueToImage(this->backwardVoxelBasedMeasureGradientImage, // in
                                 this->backwardVoxelBasedMeasureGradientImage, // out
                                 powf(2.f,fabsf(this->backwardControlPointGrid->intent_p2))); // value



    // Free the temporary backward deformation fields
    for(int i=0; i<=(int)fabsf(this->backwardControlPointGrid->intent_p2); ++i) {
        nifti_image_free(tempBackDef[i]);
        tempBackDef[i]=NULL;
    }
    free(tempBackDef);
    tempBackDef=NULL;
    // Free the temporary affine backward displacement field
    if(affine_back_disp!=NULL)
        nifti_image_free(affine_back_disp);
    affine_back_disp=NULL;
    // Free the temporary forward deformation field
    for(int i=0; i<=(int)fabsf(this->controlPointGrid->intent_p2); ++i) {
        nifti_image_free(tempForDef[i]);
        tempForDef[i]=NULL;
    }
    free(tempForDef);
    tempForDef=NULL;
    // Free the temporary affine displacement field
    if(affine_for_disp!=NULL)
        nifti_image_free(affine_for_disp);
    affine_for_disp=NULL;
    // Free the temporary gradient images
    nifti_image_free(tempGrad1);
    tempGrad1=NULL;
    nifti_image_free(tempPrevGrad);
    tempPrevGrad=NULL;

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
   // Please notice that the scale can be negative so as to maximise the objective function
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
      // Reset the gradient along the axes if appropriate
      reg_setGradientToZero(forwardScaledGradient,
                            !this->optimiser->GetOptimiseX(),
                            !this->optimiser->GetOptimiseY(),
                            !this->optimiser->GetOptimiseZ());
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
      // Reset the gradient along the axes if appropriate
      reg_setGradientToZero(backwardScaledGradient,
                            !this->optimiser->GetOptimiseX(),
                            !this->optimiser->GetOptimiseY(),
                            !this->optimiser->GetOptimiseZ());
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
      reg_exit();
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
   nifti_image **warpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
   warpedImage[0] = nifti_copy_nim_info(this->warped);
   warpedImage[0]->cal_min=this->inputFloating->cal_min;
   warpedImage[0]->cal_max=this->inputFloating->cal_max;
   warpedImage[0]->scl_slope=this->inputFloating->scl_slope;
   warpedImage[0]->scl_inter=this->inputFloating->scl_inter;
   warpedImage[0]->data=(void *)malloc(warpedImage[0]->nvox*warpedImage[0]->nbyper);
   memcpy(warpedImage[0]->data, this->warped->data, warpedImage[0]->nvox*warpedImage[0]->nbyper);

   // Allocate and save the backward transformation warped image
   warpedImage[1] = nifti_copy_nim_info(this->backwardWarped);
   warpedImage[1]->cal_min=this->inputReference->cal_min;
   warpedImage[1]->cal_max=this->inputReference->cal_max;
   warpedImage[1]->scl_slope=this->inputReference->scl_slope;
   warpedImage[1]->scl_inter=this->inputReference->scl_inter;
   warpedImage[1]->data=(void *)malloc(warpedImage[1]->nvox*warpedImage[1]->nbyper);
   memcpy(warpedImage[1]->data, this->backwardWarped->data, warpedImage[1]->nvox*warpedImage[1]->nbyper);

   // Clear the warped images
   reg_f3d2<T>::ClearWarped();

   // Return the two final warped images
   return warpedImage;
}
/* *************************************************************** */
/* *************************************************************** */
template class reg_f3d2<float>;
template class reg_f3d2<double>;
#endif
