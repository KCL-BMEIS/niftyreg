/**
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

#include <nifti/nifti1_io.h>
#include "_reg_f3d.h"
#include "_reg_aladin.h"

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_f3d<T>::reg_f3d(int refTimePoint,int floTimePoint)
   : reg_base<T>::reg_base(refTimePoint,floTimePoint)
{

   this->executableName=(char *)"NiftyReg F3D";
   this->inputControlPointGrid=NULL; // pointer to external
   this->controlPointGrid=NULL;
   this->bendingEnergyWeight=0.001;
   this->linearEnergyWeight=0.01;
   this->jacobianLogWeight=0.;
   this->jacobianLogApproximation=true;
   this->spacing[0]=-5;
   this->spacing[1]=std::numeric_limits<T>::quiet_NaN();
   this->spacing[2]=std::numeric_limits<T>::quiet_NaN();
   this->useConjGradient=true;
   this->useApproxGradient=false;

   //    this->approxParzenWindow=true;

   this->transformationGradient=NULL;

   this->gridRefinement=true;

#ifdef BUILD_DEV
   pairwiseEnergyWeight=0;
   linearSpline=false;
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::reg_f3d");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_f3d<T>::~reg_f3d()
{
   this->ClearTransformationGradient();
   if(this->controlPointGrid!=NULL)
   {
      nifti_image_free(this->controlPointGrid);
      this->controlPointGrid=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::~reg_f3d");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetControlPointGridImage(nifti_image *cp)
{
   this->inputControlPointGrid = cp;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetControlPointGridImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetBendingEnergyWeight(T be)
{
   this->bendingEnergyWeight = be;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetBendingEnergyWeight");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetLinearEnergyWeight(T le)
{
   this->linearEnergyWeight=le;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetLinearEnergyWeight");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetJacobianLogWeight(T j)
{
   this->jacobianLogWeight = j;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetJacobianLogWeight");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::ApproximateJacobianLog()
{
   this->jacobianLogApproximation = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ApproximateJacobianLog");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DoNotApproximateJacobianLog()
{
   this->jacobianLogApproximation = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::DoNotApproximateJacobianLog");
#endif
}
/* *************************************************************** */
#ifdef BUILD_DEV
template<class T>
void reg_f3d<T>::UseLinearSpline()
{
   this->linearSpline=true;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DoNotLinearSpline()
{
   this->linearSpline=false;
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetPairwiseEnergyWeight(T pw)
{
   this->pairwiseEnergyWeight=pw;
}
#endif
/* *************************************************************** */
template<class T>
void reg_f3d<T>::SetSpacing(unsigned int i, T s)
{
   this->spacing[i] = s;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetSpacing");
#endif
}
/* *************************************************************** */
template <class T>
T reg_f3d<T>::InitialiseCurrentLevel()
{
   // Set the initial step size for the gradient ascent
   T maxStepSize = this->currentReference->dx>this->currentReference->dy?this->currentReference->dx:this->currentReference->dy;
   if(this->currentReference->ndim>2)
      maxStepSize = (this->currentReference->dz>maxStepSize)?this->currentReference->dz:maxStepSize;

   // Refine the control point grid if required
   if(this->gridRefinement==true)
   {
      if(this->currentLevel==0){
         this->bendingEnergyWeight = this->bendingEnergyWeight / static_cast<T>(powf(16.0f, this->levelNumber-1));
      }
      else
      {
         reg_spline_refineControlPointGrid(this->controlPointGrid,this->currentReference);
         this->bendingEnergyWeight = this->bendingEnergyWeight * static_cast<T>(16);
      }
   }

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::InitialiseCurrentLevel");
#endif
   return maxStepSize;
}
/* *************************************************************** */
template <class T>
void reg_f3d<T>::AllocateTransformationGradient()
{
   if(this->controlPointGrid==NULL)
   {
      reg_print_fct_error("reg_f3d<T>::AllocateTransformationGradient()");
      reg_print_msg_error("The control point image is not defined");
      reg_exit();
   }
   reg_f3d<T>::ClearTransformationGradient();
   this->transformationGradient = nifti_copy_nim_info(this->controlPointGrid);
   this->transformationGradient->data = (void *)calloc(this->transformationGradient->nvox,
                                                       this->transformationGradient->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::AllocateTransformationGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d<T>::ClearTransformationGradient()
{
   if(this->transformationGradient!=NULL)
   {
      // MARTA ADDENDUM
      // Save the gradients for each level
      if(this->pathOutGradient != NULL){
          std::string b(this->pathOutGradient);
          char gradient_text[255];
          sprintf(gradient_text, "/transformationGradient_level%d_final.nii.gz", this->currentLevel);
          b.append(gradient_text);
          reg_io_WriteImageFile(this->transformationGradient,b.c_str());
      } // END ADDENDUM
      nifti_image_free(this->transformationGradient);
      this->transformationGradient=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ClearTransformationGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d<T>::AllocateRigidConstraintMask()
{
   if(this->use_rigidConstraint==true && this->inputRigidMask!=NULL){
      if(this->transformationGradient==NULL)
      {
         reg_print_fct_error("reg_f3d<T>::AllocateRigidConstraintMask()");
         reg_print_msg_error("The transformation gradient image is not defined");
         reg_exit();
      }
      // Allocate the current mask image based on the size of the gradient along the
      // x, y and z axes and the input rigid mask image for the rest
      this->currentRigidMask = nifti_copy_nim_info(this->transformationGradient);
      this->currentRigidMask->dim[0]=this->currentRigidMask->ndim=this->inputRigidMask->ndim;
      this->currentRigidMask->dim[4]=this->currentRigidMask->nt=this->inputRigidMask->nt;
      this->currentRigidMask->dim[5]=this->currentRigidMask->nu=this->inputRigidMask->nu;
      this->currentRigidMask->nvox = (size_t)this->currentRigidMask->nx *
            this->currentRigidMask->ny * this->currentRigidMask->nz *
            this->currentRigidMask->nt * this->currentRigidMask->nu;
      this->currentRigidMask->datatype = this->inputRigidMask->datatype;
      this->currentRigidMask->nbyper = this->inputRigidMask->nbyper;
      this->currentRigidMask->data = (void *)malloc(this->currentRigidMask->nvox *
                                                    this->currentRigidMask->nbyper);
      // Compute an identity deformation field and use the gradient image to store it
      reg_tools_multiplyValueToImage(this->transformationGradient,
                                     this->transformationGradient,
                                     0);
      reg_getDeformationFromDisplacement(this->transformationGradient);
      reg_resampleImage(this->inputRigidMask,
                        this->currentRigidMask,
                        this->transformationGradient,
                        NULL,
                        0, // nearest-neighboor interpolation
                        0 // padding with a value of 0
                        );
      reg_tools_changeDatatype<T>(this->currentRigidMask);
      reg_tools_binarise_image(this->currentRigidMask);
      float *radius=new float[this->currentRigidMask->nt];
      for(int i=0; i<this->currentRigidMask->nt; ++i){
         radius[i]=-1.f;
      }
      reg_tools_kernelConvolution(this->currentRigidMask,
                                  radius,
                                  CUBIC_SPLINE_KERNEL);
      reg_tools_binarise_image(this->currentRigidMask);
      reg_tools_changeDatatype<unsigned char>(this->currentRigidMask);
      delete []radius;

//      if(this->currentLevel == 0){

//          int mask_to_consider = this->inputRigidMask->nt -1;
////          size_t mask_to_consider = 1;

//          //unsigned char *maskPtr = static_cast<unsigned char *>(maskImage->data);
//          size_t voxelNumber = (size_t)this->inputRigidMask->nx *
//                               this->inputRigidMask->ny * this->inputRigidMask->nz;

//          // Create an array to store the rigid transformation
//          mat44 **mask_transformation = (mat44 **)malloc(mask_to_consider*sizeof(mat44 *));
//          for(int t=0; t<mask_to_consider; ++t) {
//              mask_transformation[t] = (mat44 *)malloc(sizeof(mat44));
//          }

//          // loop over the rigid masks to extract affines for bones (no loop on implant)
//          unsigned char *maskPointer = static_cast<unsigned char *>(this->inputRigidMask->data);
//          for(int t=0; t<mask_to_consider; ++t){

//              reg_aladin<float> *REG;
//              REG = new reg_aladin<float>; //TODO: symmetric version?

//              REG->SetInputReference(this->inputReference);
//              REG->SetInputFloating(this->inputFloating);

//              nifti_image *currentMask = nifti_copy_nim_info(this->inputRigidMask);
//              currentMask->nt=1;
//              currentMask->nvox = voxelNumber;
//              currentMask->data = malloc(voxelNumber*currentMask->nbyper);
//              memcpy(currentMask->data,
//                     &maskPointer[t*voxelNumber],
//                     voxelNumber*currentMask->nbyper);
//              REG->SetInputMask(currentMask);
//              REG->SetPerformRigid(1);
//              REG->SetPerformAffine(0);

//              REG->Run();

//              memcpy(mask_transformation[t],
//                     REG->GetTransformationMatrix(),
//                     sizeof(mat44));
//              delete REG;
//              nifti_image_free(currentMask);
//          }
//          maskPointer = static_cast<unsigned char *>(this->currentRigidMask->data);
//          size_t keypoint_number = 0;
//          voxelNumber = (size_t)this->currentRigidMask->nx *
//                        this->currentRigidMask->ny * this->currentRigidMask->nz;

//          for(int i=0; i<voxelNumber*mask_to_consider; ++i){
//              keypoint_number += maskPointer[i];
//          }
//          T *refKeyPoint = (T *)malloc(keypoint_number*3*sizeof(T));
//          T *floKeyPoint = (T *)malloc(keypoint_number*3*sizeof(T));
//          size_t k = 0;

//          T *cppPtrX = static_cast<T *>(this->controlPointGrid->data);
//          T *cppPtrY = &cppPtrX[voxelNumber];
//          T *cppPtrZ = &cppPtrY[voxelNumber];
//          for(int t=0; t<mask_to_consider; ++t) {
//              bool boum = false;
//              for (int i = 0; i < voxelNumber; ++i) {
//                  if (maskPointer[i+t*voxelNumber] > 0) {
//                      T temp_ref[3], temp_flo[3];
//                      temp_ref[0]=refKeyPoint[k] = cppPtrX[i];
//                      temp_ref[1]=refKeyPoint[k+keypoint_number] = cppPtrY[i];
//                      temp_ref[2]=refKeyPoint[k+2*keypoint_number] = cppPtrZ[i];
//                      reg_mat44_mul(mask_transformation[t], temp_ref, temp_flo);
//                      floKeyPoint[k]=temp_flo[0]-temp_ref[0];
//                      floKeyPoint[k+keypoint_number]=temp_flo[1]-temp_ref[1];
//                      floKeyPoint[k+2*keypoint_number]=temp_flo[2]-temp_ref[2];
//                      ++k;
//                      if (!boum){
//                          std::cout << temp_ref[0] << " " << temp_ref[1] << " " << temp_ref[2] << std::endl;
//                          std::cout << temp_flo[0] << " " << temp_flo[1] << " " << temp_flo[2] << std::endl;
//                          boum = true;
//                      }
//                  }
//              }
//          }

//          reg_io_WriteImageFile(this->currentRigidMask, "current_rigid_masks.nii.gz");
//          reg_exit();

//          // Create the TPS object
//          reg_tps<T> *TPS = new reg_tps<T>(3, keypoint_number);
//          TPS->SetAproxInter(1);
//          TPS->SetPosition(&refKeyPoint[0],
//                           &refKeyPoint[keypoint_number],
//                           &refKeyPoint[2*keypoint_number],
//                           &floKeyPoint[0],
//                           &floKeyPoint[keypoint_number],
//                           &floKeyPoint[2*keypoint_number]);
//          TPS->FillDeformationField(this->controlPointGrid);
//          delete TPS;

//          for(int t=0; t<mask_to_consider; ++t) {
//              free(mask_transformation[t]);
//          }
//          free(mask_transformation);
//      }
   }

}
/* *************************************************************** */
template <class T>
void reg_f3d<T>::ClearRigidConstraintMask()
{
   if(this->currentRigidMask!=NULL)
   {
      nifti_image_free(this->currentRigidMask);
      this->currentRigidMask=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ClearRigidConstraintMask");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CheckParameters()
{
   reg_base<T>::CheckParameters();
   // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
   if(strcmp(this->executableName,"NiftyReg F3D")==0 ||
         strcmp(this->executableName,"NiftyReg F3D GPU")==0)
   {
#ifdef BUILD_DEV
   if(this->linearSpline==true){
      if(this->bendingEnergyWeight>0){
         this->bendingEnergyWeight=0;
         reg_print_msg_warn("The weight of the bending energy term is set to 0 when using linear spline");
      }
      if(this->linearEnergyWeight>0){
         this->linearEnergyWeight=0;
         reg_print_msg_warn("The weight of the lienar energy term is set to 0 when using linear spline");
      }
      if(this->jacobianLogWeight>0){
         this->jacobianLogWeight=0;
         reg_print_msg_warn("The weight of the Jacobian based regularisation term is set to 0 when using linear spline");
      }
   }
   T penaltySum=this->pairwiseEnergyWeight;
#else
      T penaltySum=this->bendingEnergyWeight +
            this->linearEnergyWeight +
            this->jacobianLogWeight +
            this->landmarkRegWeight;
#endif
      if(penaltySum>=1.0)
      {
         this->similarityWeight=0;
         this->similarityWeight /= penaltySum;
         this->bendingEnergyWeight /= penaltySum;
         this->linearEnergyWeight /= penaltySum;
         this->jacobianLogWeight /= penaltySum;
         this->landmarkRegWeight /= penaltySum;
#ifdef BUILD_DEV
         this->pairwiseEnergyWeight /= penaltySum;
#endif
      }
      else this->similarityWeight=1.0 - penaltySum;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::CheckParameters");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::Initialise()
{
   if(this->initialised) return;

   reg_base<T>::Initialise();

   // DETERMINE THE GRID SPACING AND CREATE THE GRID
   if(this->inputControlPointGrid==NULL)
   {
      // Set the spacing along y and z if undefined. Their values are set to match
      // the spacing along the x axis
      if(this->spacing[1]!=this->spacing[1]) this->spacing[1]=this->spacing[0];
      if(this->spacing[2]!=this->spacing[2]) this->spacing[2]=this->spacing[0];

      /* Convert the spacing from voxel to mm if necessary */
      float spacingInMillimeter[3]= {this->spacing[0],this->spacing[1],this->spacing[2]};
      if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->inputReference->dx;
      if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->inputReference->dy;
      if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->inputReference->dz;

      // Define the spacing for the first level
      float gridSpacing[3];
      gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(this->levelNumber-1));
      gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(this->levelNumber-1));
      gridSpacing[2] = 1.0f;
      if(this->referencePyramid[0]->nz>1)
         gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(this->levelNumber-1));

      // Create and allocate the control point image
      reg_createControlPointGrid<T>(&this->controlPointGrid,
                                    this->referencePyramid[0],
            gridSpacing);

      // The control point position image is initialised with the affine transformation
      if(this->affineTransformation==NULL)
      {
         memset(this->controlPointGrid->data,0,
                this->controlPointGrid->nvox*this->controlPointGrid->nbyper);
         reg_tools_multiplyValueToImage(this->controlPointGrid,this->controlPointGrid,0.f);
         reg_getDeformationFromDisplacement(this->controlPointGrid);
      }
      else reg_affine_getDeformationField(this->affineTransformation, this->controlPointGrid);
   }
   else
   {
      // The control point grid image is initialised with the provided grid
      this->controlPointGrid = nifti_copy_nim_info(this->inputControlPointGrid);
      this->controlPointGrid->data = (void *)malloc( this->controlPointGrid->nvox *
                                                     this->controlPointGrid->nbyper);
      memcpy( this->controlPointGrid->data, this->inputControlPointGrid->data,
              this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
      // The final grid spacing is computed
      this->spacing[0] = this->controlPointGrid->dx / powf(2.0f, (float)(this->levelNumber-1));
      this->spacing[1] = this->controlPointGrid->dy / powf(2.0f, (float)(this->levelNumber-1));
      if(this->controlPointGrid->nz>1)
         this->spacing[2] = this->controlPointGrid->dz / powf(2.0f, (float)(this->levelNumber-1));
   }
#ifdef BUILD_DEV
   if(this->linearSpline)
      this->controlPointGrid->intent_p1=LIN_SPLINE_GRID;
#endif
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      char text[255];
      // Print out some global information about the registration
      reg_print_info(this->executableName, "***********************************************************");
      reg_print_info(this->executableName, "INPUT PARAMETERS");
      reg_print_info(this->executableName, "***********************************************************");
      reg_print_info(this->executableName, "Reference image:");
      sprintf(text, "\t* name: %s", this->inputReference->fname);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->inputReference->nx, this->inputReference->ny,
              this->inputReference->nz, this->inputReference->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->inputReference->dx,
              this->inputReference->dy, this->inputReference->dz);
      reg_print_info(this->executableName, text);
      for(int i=0; i<this->inputReference->nt; i++)
      {
         sprintf(text, "\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                 i, this->inputReference->nt-1, this->referenceThresholdLow[i],this->referenceThresholdUp[i]);
         reg_print_info(this->executableName, text);
         if(this->measure_nmi!=NULL){
            if(this->measure_nmi->GetTimepointsWeights()[i]>0.0){
               sprintf(text, "\t* binnining size for timepoint %i/%i: %i",
                       i, this->inputFloating->nt-1, this->measure_nmi->GetReferenceBinNumber()[i]-4);
               reg_print_info(this->executableName, text);
            }
         }
      }
      sprintf(text, "\t* gaussian smoothing sigma: %g", this->referenceSmoothingSigma);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      reg_print_info(this->executableName, "Floating image:");
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* name: %s", this->inputFloating->fname);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->inputFloating->nx, this->inputFloating->ny,
              this->inputFloating->nz, this->inputFloating->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->inputFloating->dx,
              this->inputFloating->dy, this->inputFloating->dz);
      reg_print_info(this->executableName, text);
      for(int i=0; i<this->inputFloating->nt; i++)
      {
         sprintf(text, "\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                 i, this->inputFloating->nt-1, this->floatingThresholdLow[i],this->floatingThresholdUp[i]);
         reg_print_info(this->executableName, text);
         if(this->measure_nmi!=NULL){
          if (this->measure_nmi->GetTimepointsWeights()[i]>0.0){
               sprintf(text, "\t* binnining size for timepoint %i/%i: %i",
                       i, this->inputFloating->nt-1, this->measure_nmi->GetFloatingBinNumber()[i]-4);
               reg_print_info(this->executableName, text);
            }
         }
      }
      sprintf(text, "\t* gaussian smoothing sigma: %g", this->floatingSmoothingSigma);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      sprintf(text, "Warped image padding value: %g", this->warpedPaddingValue);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      sprintf(text, "Level number: %i", this->levelNumber);
      reg_print_info(this->executableName, text);
      if(this->levelNumber!=this->levelToPerform){
         sprintf(text, "\t* Level to perform: %i", this->levelToPerform);
         reg_print_info(this->executableName, text);
      }
      reg_print_info(this->executableName, "");
      sprintf(text, "Maximum iteration number during the last level: %i", (int)this->maxiterationNumber);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");

#ifdef BUILD_DEV
      if(this->linearSpline){
         sprintf(text, "Linear interpolation is used for the parametrisation");
         reg_print_info(this->executableName, text);
      }
      else{
#endif
         sprintf(text, "Cubic B-Spline is used for the parametrisation");
         reg_print_info(this->executableName, text);
#ifdef BUILD_DEV
      }
#endif
      sprintf(text, "Final spacing in mm: %g %g %g",
              this->spacing[0], this->spacing[1], this->spacing[2]);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      if(this->measure_ssd!=NULL)
         reg_print_info(this->executableName, "The SSD is used as a similarity measure.");
      if(this->measure_kld!=NULL)
         reg_print_info(this->executableName, "The KL divergence is used as a similarity measure.");
      if(this->measure_lncc!=NULL)
         reg_print_info(this->executableName, "The LNCC is used as a similarity measure.");
      if(this->measure_dti!=NULL)
         reg_print_info(this->executableName, "A DTI based measure is used as a similarity measure.");
      if(this->measure_mind!=NULL)
         reg_print_info(this->executableName, "MIND is used as a similarity measure.");
      if(this->measure_mindssc!=NULL)
         reg_print_info(this->executableName, "MINDSSC is used as a similarity measure.");
      if(this->measure_nmi!=NULL || (this->measure_dti==NULL && this->measure_kld==NULL &&
                                     this->measure_lncc==NULL &&  this->measure_nmi==NULL &&
                                     this->measure_ssd==NULL && this->measure_mind==NULL  &&
                                     this->measure_mindssc==NULL) )
         reg_print_info(this->executableName, "The NMI is used as a similarity measure.");
      sprintf(text, "Similarity measure term weight: %g", this->similarityWeight);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      if(this->bendingEnergyWeight>0){
         sprintf(text, "Bending energy penalty term weight: %g", this->bendingEnergyWeight);
         reg_print_info(this->executableName, text);
         reg_print_info(this->executableName, "");
      }
      if((this->linearEnergyWeight)>0){
         sprintf(text, "Linear energy penalty term weight: %g",
                 this->linearEnergyWeight);
         reg_print_info(this->executableName, text);
         reg_print_info(this->executableName, "");
      }
      if(this->jacobianLogWeight>0){
         sprintf(text, "Jacobian-based penalty term weight: %g", this->jacobianLogWeight);
         reg_print_info(this->executableName, text);
         if(this->jacobianLogApproximation){
            reg_print_info(this->executableName, "\t* Jacobian-based penalty term is approximated");
         }
         else reg_print_info(this->executableName, "\t* Jacobian-based penalty term is not approximated");
         reg_print_info(this->executableName, "");
      }
      if((this->landmarkRegWeight)>0){
         sprintf(text, "Landmark distance regularisation term weight: %g",
                 this->landmarkRegWeight);
         reg_print_info(this->executableName, text);
         reg_print_info(this->executableName, "");
      }
#ifdef BUILD_DEV
      if((this->pairwiseEnergyWeight)>0){
         sprintf(text, "Pairwise energy penalty term weight: %g",
                 this->pairwiseEnergyWeight);
         reg_print_info(this->executableName, text);
         reg_print_info(this->executableName, "");
      }
#endif
#ifdef NDEBUG
   }
#endif

   this->initialised=true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::Initialise");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetDeformationField()
{
   reg_spline_getDeformationField(this->controlPointGrid,
                                  this->deformationFieldImage,
                                  this->currentMask,
                                  false, //composition
                                  true // bspline
                                  );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetDeformationField");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
   if(this->jacobianLogWeight<=0) return 0.;

   double value=0.;

   if(type==2)
   {
      value = reg_spline_getJacobianPenaltyTerm(this->controlPointGrid,
                                                this->currentReference,
                                                false);
   }
   else
   {
      value = reg_spline_getJacobianPenaltyTerm(this->controlPointGrid,
                                                this->currentReference,
                                                this->jacobianLogApproximation);
   }
   unsigned int maxit=5;
   if(type>0) maxit=20;
   unsigned int it=0;
   while(value!=value && it<maxit)
   {
      if(type==2)
      {
         value = reg_spline_correctFolding(this->controlPointGrid,
                                           this->currentReference,
                                           false);
      }
      else
      {
         value = reg_spline_correctFolding(this->controlPointGrid,
                                           this->currentReference,
                                           this->jacobianLogApproximation);
      }
#ifndef NDEBUG
      reg_print_msg_debug("Folding correction");
#endif
      it++;
   }
   if(type>0)
   {
      if(value!=value)
      {
         this->optimiser->RestoreBestDOF();
         reg_print_fct_warn("reg_f3d<T>::ComputeJacobianBasedPenaltyTerm()");
         reg_print_msg_warn("The folding correction scheme failed");
      }
      else
      {
#ifndef NDEBUG
         if(it>0){
            char text[255];
            sprintf(text, "Folding correction, %i step(s)", it);
            reg_print_msg_debug(text);
         }
#endif
      }
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeJacobianBasedPenaltyTerm");
#endif
   return (double)this->jacobianLogWeight * value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeBendingEnergyPenaltyTerm()
{
   if(this->bendingEnergyWeight<=0) return 0.;

   double value = reg_spline_approxBendingEnergy(this->controlPointGrid);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeBendingEnergyPenaltyTerm");
#endif
   return this->bendingEnergyWeight * value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm()
{
   if(this->linearEnergyWeight<=0)
      return 0.;

   double value = reg_spline_approxLinearEnergy(this->controlPointGrid);

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeLinearEnergyPenaltyTerm");
#endif
   return this->linearEnergyWeight*value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm()
{
   if(this->landmarkRegWeight<=0)
      return 0.;

   double value = reg_spline_getLandmarkDistance(this->controlPointGrid,
                                                 this->landmarkRegNumber,
                                                 this->landmarkReference,
                                                 this->landmarkFloating);

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeLandmarkDistancePenaltyTerm");
#endif
   return this->landmarkRegWeight*value;
}
/* *************************************************************** */
/* *************************************************************** */
#ifdef BUILD_DEV
template <class T>
double reg_f3d<T>::ComputePairwiseEnergyPenaltyTerm()
{
   if(this->pairwiseEnergyWeight<=0)
      return 0.;

   double value = reg_spline_approxLinearPairwise(this->controlPointGrid);

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputePairwiseEnergyPenaltyTerm");
#endif
   return this->pairwiseEnergyWeight*value;
}
#endif
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetSimilarityMeasureGradient()
{
   this->GetVoxelBasedGradient();

   int kernel_type=CUBIC_SPLINE_KERNEL;
#ifdef BUILD_DEV
   if(this->linearSpline)
      kernel_type=LINEAR_KERNEL;
#endif
   // The voxel based NMI gradient is convolved with a spline kernel
   // Convolution along the x axis
   float currentNodeSpacing[3];
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dx;
   bool activeAxis[3]= {1,0,0};
   reg_tools_kernelConvolution(this->voxelBasedMeasureGradient,
                               currentNodeSpacing,
                               kernel_type,
                               NULL, // mask
                               NULL, // all volumes are considered as active
                               activeAxis
                               );
   // Convolution along the y axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dy;
   activeAxis[0]=0;
   activeAxis[1]=1;
   reg_tools_kernelConvolution(this->voxelBasedMeasureGradient,
                               currentNodeSpacing,
                               kernel_type,
                               NULL, // mask
                               NULL, // all volumes are considered as active
                               activeAxis
                               );
   // Convolution along the z axis if required
   if(this->voxelBasedMeasureGradient->nz>1)
   {
      currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dz;
      activeAxis[1]=0;
      activeAxis[2]=1;
      reg_tools_kernelConvolution(this->voxelBasedMeasureGradient,
                                  currentNodeSpacing,
                                  kernel_type,
                                  NULL, // mask
                                  NULL, // all volumes are considered as active
                                  activeAxis
                                  );
   }

   // The node based NMI gradient is extracted
   mat44 reorientation;
   if(this->currentFloating->sform_code>0)
      reorientation = this->currentFloating->sto_ijk;
   else reorientation = this->currentFloating->qto_ijk;
   reg_voxelCentric2NodeCentric(this->transformationGradient,
                                this->voxelBasedMeasureGradient,
                                this->similarityWeight,
                                false, // no update
                                &reorientation
                                );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetSimilarityMeasureGradient");
#endif
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetBendingEnergyGradient()
{
   if(this->bendingEnergyWeight<=0) return;

   reg_spline_approxBendingEnergyGradient(this->controlPointGrid,
                                          this->transformationGradient,
                                          this->bendingEnergyWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetBendingEnergyGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetLinearEnergyGradient()
{
   if(this->linearEnergyWeight<=0) return;

   reg_spline_approxLinearEnergyGradient(this->controlPointGrid,
                                         this->transformationGradient,
                                         this->linearEnergyWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetLinearEnergyGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetJacobianBasedGradient()
{
   if(this->jacobianLogWeight<=0) return;

   reg_spline_getJacobianPenaltyTermGradient(this->controlPointGrid,
                                             this->currentReference,
                                             this->transformationGradient,
                                             this->jacobianLogWeight,
                                             this->jacobianLogApproximation);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetJacobianBasedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetLandmarkDistanceGradient()
{
   if(this->landmarkRegWeight<=0) return;

   reg_spline_getLandmarkDistanceGradient(this->controlPointGrid,
                                          this->transformationGradient,
                                          this->landmarkRegNumber,
                                          this->landmarkReference,
                                          this->landmarkFloating,
                                          this->landmarkRegWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetLandmarkDistanceGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
#ifdef BUILD_DEV
template <class T>
void reg_f3d<T>::GetPairwiseEnergyGradient()
{
   if(this->pairwiseEnergyWeight<=0) return;

   reg_spline_approxLinearPairwiseGradient(this->controlPointGrid,
                                           this->transformationGradient,
                                           this->pairwiseEnergyWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetPairwiseEnergyGradient");
#endif
}
#endif
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::SetGradientImageToZero()
{
   T* nodeGradPtr = static_cast<T *>(this->transformationGradient->data);
   for(size_t i=0; i<this->transformationGradient->nvox; ++i)
      *nodeGradPtr++=0;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetGradientImageToZero");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_f3d<T>::NormaliseGradient()
{
   // First compute the gradient max length for normalisation purpose
   //	T maxGradValue=0;
   size_t voxNumber = this->transformationGradient->nx *
         this->transformationGradient->ny *
         this->transformationGradient->nz;
   T *ptrX = static_cast<T *>(this->transformationGradient->data);
   T *ptrY = &ptrX[voxNumber];
   T *ptrZ = NULL;
   T maxGradValue=0;
   //	float *length=(float *)calloc(voxNumber,sizeof(float));
   if(this->transformationGradient->nz>1)
   {
      ptrZ = &ptrY[voxNumber];
      for(size_t i=0; i<voxNumber; i++)
      {
         T valX=0,valY=0,valZ=0;
         if(this->optimiseX==true)
            valX = *ptrX++;
         if(this->optimiseY==true)
            valY = *ptrY++;
         if(this->optimiseZ==true)
            valZ = *ptrZ++;
         //			length[i] = (float)(sqrt(valX*valX + valY*valY + valZ*valZ));
         T length = (T)(sqrt(valX*valX + valY*valY + valZ*valZ));
         maxGradValue = (length>maxGradValue)?length:maxGradValue;
      }
   }
   else
   {
      for(size_t i=0; i<voxNumber; i++)
      {
         T valX=0,valY=0;
         if(this->optimiseX==true)
            valX = *ptrX++;
         if(this->optimiseY==true)
            valY = *ptrY++;
         //			length[i] = (float)(sqrt(valX*valX + valY*valY));
         T length = (T)(sqrt(valX*valX + valY*valY));
         maxGradValue = (length>maxGradValue)?length:maxGradValue;
      }
   }


   if(strcmp(this->executableName,"NiftyReg F3D")==0)
   {
      // The gradient is normalised if we are running f3d
      // It will be normalised later when running f3d_sym or f3d2
#ifndef NDEBUG
      char text[255];
      sprintf(text, "Objective function gradient maximal length: %g",maxGradValue);
      reg_print_msg_debug(text);
#endif
      ptrX = static_cast<T *>(this->transformationGradient->data);
      if(this->transformationGradient->nz>1)
      {
         ptrX = static_cast<T *>(this->transformationGradient->data);
         ptrY = &ptrX[voxNumber];
         ptrZ = &ptrY[voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            T valX=0,valY=0,valZ=0;
            if(this->optimiseX==true)
               valX = *ptrX;
            if(this->optimiseY==true)
               valY = *ptrY;
            if(this->optimiseZ==true)
               valZ = *ptrZ;
            *ptrX++ = valX / maxGradValue;
            *ptrY++ = valY / maxGradValue;
            *ptrZ++ = valZ / maxGradValue;
         }
      }
      else
      {
         ptrX = static_cast<T *>(this->transformationGradient->data);
         ptrY = &ptrX[voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            T valX=0,valY=0;
            if(this->optimiseX==true)
               valX = *ptrX;
            if(this->optimiseY==true)
               valY = *ptrY;
            *ptrX++ = valX / maxGradValue;
            *ptrY++ = valY / maxGradValue;
         }
      }
   }
   // Returns the largest gradient distance
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::NormaliseGradient");
#endif

   return maxGradValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::DisplayCurrentLevelParameters()
{
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      char text[255];
      sprintf(text, "Current level: %i / %i", this->currentLevel+1, this->levelNumber);
      reg_print_info(this->executableName, text);
      sprintf(text, "Maximum iteration number: %i", (int)this->maxiterationNumber);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "Current reference image");
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->currentReference->nx, this->currentReference->ny,
              this->currentReference->nz,this->currentReference->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->currentReference->dx, this->currentReference->dy,
              this->currentReference->dz);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "Current floating image");
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->currentFloating->nx, this->currentFloating->ny,
              this->currentFloating->nz,this->currentFloating->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->currentFloating->dx, this->currentFloating->dy,
              this->currentFloating->dz);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "Current control point image");
      sprintf(text, "\t* image dimension: %i x %i x %i",
              this->controlPointGrid->nx, this->controlPointGrid->ny,
              this->controlPointGrid->nz);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->controlPointGrid->dx, this->controlPointGrid->dy,
              this->controlPointGrid->dz);
      reg_print_info(this->executableName, text);
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
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::DisplayCurrentLevelParameters");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d<T>::GetObjectiveFunctionValue()
{
   this->currentWJac = this->ComputeJacobianBasedPenaltyTerm(1); // 20 iterations

   this->currentWBE = this->ComputeBendingEnergyPenaltyTerm();

   this->currentWLE = this->ComputeLinearEnergyPenaltyTerm();

   this->currentWLand = this->ComputeLandmarkDistancePenaltyTerm();

#ifdef BUILD_DEV
   this->currentWPE = this->ComputePairwiseEnergyPenaltyTerm();
#endif

   // Compute initial similarity measure
   this->currentWMeasure = 0.0;
   if(this->similarityWeight>0)
   {
      this->WarpFloatingImage(this->interpolation);
      this->currentWMeasure = this->ComputeSimilarityMeasure();
   }
#ifndef NDEBUG
   char text[255];
   sprintf(text, "(wMeasure) %g | (wBE) %g | (wLE) %g | (wJac) %g | (wLan) %g",
           this->currentWMeasure, this->currentWBE, this->currentWLE, this->currentWJac, this->currentWLand);
   reg_print_msg_debug(text);
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionValue");
#endif
   // Store the global objective function value

#ifdef BUILD_DEV
   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac - this->currentWLandmarkReg - this->currentWPE;
#else
   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac - this->currentWLand;
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::UpdateParameters(float scale)
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
      for(size_t i=0; i<this->optimiser->GetDOFNumber(); ++i)
      {
         currentDOF[i] = bestDOF[i] + scale * gradient[i];
      }
   }
   else
   {
      size_t voxNumber = this->optimiser->GetVoxNumber();
      // Update the values for the x-axis displacement
      if(this->optimiser->GetOptimiseX()==true)
      {
         for(size_t i=0; i<voxNumber; ++i)
         {
            currentDOF[i] = bestDOF[i] + scale * gradient[i];
         }
      }
      // Update the values for the y-axis displacement
      if(this->optimiser->GetOptimiseY()==true)
      {
         T *currentDOFY=&currentDOF[voxNumber];
         T *bestDOFY=&bestDOF[voxNumber];
         T *gradientY=&gradient[voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            currentDOFY[i] = bestDOFY[i] + scale * gradientY[i];
         }
      }
      // Update the values for the z-axis displacement
      if(this->optimiser->GetOptimiseZ()==true && this->optimiser->GetNDim()>2)
      {
         T *currentDOFZ=&currentDOF[2*voxNumber];
         T *bestDOFZ=&bestDOF[2*voxNumber];
         T *gradientZ=&gradient[2*voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            currentDOFZ[i] = bestDOFZ[i] + scale * gradientZ[i];
         }
      }
   }
   // Apply the rigid constraint if required
//   if(this->use_rigidConstraint){
//       // MARTA ADDENDUM
//       // Save the transformation for each level before application of gradient
//       if(this->pathOutGradient != NULL){
//           std::string b(this->pathOutGradient);
//           char gradient_text[255];
//           sprintf(gradient_text, "/transformation_level%d_before.nii.gz", this->currentLevel);
//           b.append(gradient_text);
//           reg_io_WriteImageFile(this->controlPointGrid,b.c_str());
//       } // END ADDENDUM

//       // MARTA ADDENDUM ARGUMENT FOR NR of ITERATIONS
//      regulariseNonLinearGradientWithRigidConstraint(this->controlPointGrid,
//                                                     this->currentRigidMask,
//                                                     false,
//                                                     this->nrIterationsRigid);
//       // MARTA ADDENDUM ARGUMENT FOR NR of ITERATIONS

//       // MARTA ADDENDUM
//       // Save the transformation for each level after application of gradient
//       if(this->pathOutGradient != NULL){
//           std::string b_2(this->pathOutGradient);
//           char gradient_text[255];
//           sprintf(gradient_text, "/transformation_level%d_after.nii.gz", this->currentLevel);
//           b_2.append(gradient_text);
//           reg_io_WriteImageFile(this->controlPointGrid,b_2.c_str());
//       } // END ADDENDUM
//   }

   if(this->use_rigidConstraint){
      // Create a temporary image
      nifti_image *temp_scaled = nifti_copy_nim_info(this->controlPointGrid);
      temp_scaled->data=(void *)malloc
            (temp_scaled->nvox*temp_scaled->nbyper);
      memcpy(temp_scaled->data,
             this->controlPointGrid->data,
             temp_scaled->nvox*temp_scaled->nbyper);
      // Remove the identify
      reg_getDisplacementFromDeformation(temp_scaled);
      // Scale down the displacement field
      if(this->controlPointGrid->intent_p1==SPLINE_VEL_GRID){
         reg_tools_divideValueToImage(temp_scaled,
                                      temp_scaled,
                                      pow(2.0f,fabs((float)this->controlPointGrid->intent_p2)));
      }
      // apply the contraint
      regulariseNonLinearGradientWithRigidConstraint(temp_scaled,
                                                     this->currentRigidMask,
                                                     true,
                                                     this->nrIterationsRigid);
      // Scale up the displacement field
      if(this->controlPointGrid->intent_p1==SPLINE_VEL_GRID){
         reg_tools_multiplyValueToImage(temp_scaled,
                                        temp_scaled,
                                        pow(2.0f,fabs((float)this->controlPointGrid->intent_p2)));
      }
      // Add the identity
      reg_getDeformationFromDisplacement(temp_scaled);
      // Restore in the original velovity grid
      memcpy(this->controlPointGrid->data,
             temp_scaled->data,
             temp_scaled->nvox*temp_scaled->nbyper);
      // free temporary image
      nifti_image_free(temp_scaled);
   } // END ADDENDUM
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::UpdateParameters");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::SetOptimiser()
{
   reg_base<T>::SetOptimiser();
   this->optimiser->Initialise(this->controlPointGrid->nvox,
                               this->controlPointGrid->nz>1?3:2,
                               this->optimiseX,
                               this->optimiseY,
                               this->optimiseZ,
                               this->maxiterationNumber,
                               0, // currentIterationNumber,
                               this,
                               static_cast<T *>(this->controlPointGrid->data),
                               static_cast<T *>(this->transformationGradient->data)
                               );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::SmoothGradient()
{
   // The gradient is smoothed using a Gaussian kernel if it is required
   if(this->gradientSmoothingSigma!=0)
   {
      float kernel = fabs(this->gradientSmoothingSigma);
      reg_tools_kernelConvolution(this->transformationGradient,
                                  &kernel,
                                  GAUSSIAN_KERNEL);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SmoothGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d<T>::GetApproximatedGradient()
{
   // Loop over every control point
   T *gridPtr = static_cast<T *>(this->controlPointGrid->data);
   T *gradPtr = static_cast<T *>(this->transformationGradient->data);
   T eps = this->controlPointGrid->dx / 100.f;
   for(size_t i=0; i<this->controlPointGrid->nvox; ++i)
   {
      T currentValue = this->optimiser->GetBestDOF()[i];
      gridPtr[i] = currentValue + eps;
      double valPlus = this->GetObjectiveFunctionValue();
      gridPtr[i] = currentValue - eps;
      double valMinus = this->GetObjectiveFunctionValue();
      gridPtr[i] = currentValue;
      gradPtr[i] = -(T)((valPlus - valMinus ) / (2.0*eps));
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetApproximatedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
nifti_image **reg_f3d<T>::GetWarpedImage()
{
   // The initial images are used
   if(this->inputReference==NULL ||
         this->inputFloating==NULL ||
         this->controlPointGrid==NULL)
   {
      reg_print_fct_error("reg_f3d<T>::GetWarpedImage()");
      reg_print_msg_error("The reference, floating and control point grid images have to be defined");
      reg_exit();
   }

   this->currentReference = this->inputReference;
   this->currentFloating = this->inputFloating;
   this->currentMask=NULL;

   reg_base<T>::AllocateWarped();
   reg_base<T>::AllocateDeformationField();
   reg_base<T>::WarpFloatingImage(3); // cubic spline interpolation
   reg_base<T>::ClearDeformationField();

   nifti_image **warpedImage= (nifti_image **)malloc(2*sizeof(nifti_image *));
   warpedImage[0]=nifti_copy_nim_info(this->warped);
   warpedImage[0]->cal_min=this->inputFloating->cal_min;
   warpedImage[0]->cal_max=this->inputFloating->cal_max;
   warpedImage[0]->scl_slope=this->inputFloating->scl_slope;
   warpedImage[0]->scl_inter=this->inputFloating->scl_inter;
   warpedImage[0]->data=(void *)malloc(warpedImage[0]->nvox*warpedImage[0]->nbyper);
   memcpy(warpedImage[0]->data, this->warped->data, warpedImage[0]->nvox*warpedImage[0]->nbyper);

   warpedImage[1]=NULL;

   reg_f3d<T>::ClearWarped();
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetWarpedImage");
#endif
   return warpedImage;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
nifti_image * reg_f3d<T>::GetControlPointPositionImage()
{
   nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->controlPointGrid);
   returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   memcpy(returnedControlPointGrid->data, this->controlPointGrid->data,
          returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   return returnedControlPointGrid;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetControlPointPositionImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::UpdateBestObjFunctionValue()
{
   this->bestWMeasure=this->currentWMeasure;
   this->bestWBE=this->currentWBE;
   this->bestWLE=this->currentWLE;
   this->bestWJac=this->currentWJac;
   this->bestWLand=this->currentWLand;
#ifdef BUILD_DEV
   this->bestWPE=this->currentWPE;
#endif
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::UpdateBestObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintInitialObjFunctionValue()
{
   if(!this->verbose) return;

   double bestValue=this->optimiser->GetBestObjFunctionValue();

   char text[255];
#ifdef BUILD_DEV
   sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g - (wLAN)%g - (wPW)%g",
           bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWJac, this->bestWLandmarkReg, this->bestWPE);
#else
   sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g - (wLAN)%g",
           bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWJac, this->bestWLand);
#endif
   reg_print_info(this->executableName, text);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::PrintInitialObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::PrintCurrentObjFunctionValue(T currentSize)
{
   if(!this->verbose) return;

   char text[255];
   sprintf(text, "[%i] Current objective function: %g",
           (int)this->optimiser->GetCurrentIterationNumber(),
           this->optimiser->GetBestObjFunctionValue());
   sprintf(text+strlen(text), " = (wSIM)%g", this->bestWMeasure);
   if(this->bendingEnergyWeight>0)
      sprintf(text+strlen(text), " - (wBE)%.2e", this->bestWBE);
   if(this->linearEnergyWeight>0)
      sprintf(text+strlen(text), " - (wLE)%.2e", this->bestWLE);
   if(this->jacobianLogWeight>0)
      sprintf(text+strlen(text), " - (wJAC)%.2e", this->bestWJac);
   if(this->landmarkRegWeight>0)
      sprintf(text+strlen(text), " - (wLAN)%.2e", this->bestWLand);
#ifdef BUILD_DEV
   if(this->pairwiseEnergyWeight>0)
      sprintf(text+strlen(text), " - (wPW)%.2e", this->bestWPE);
#endif
   sprintf(text+strlen(text), " [+ %g mm]", currentSize);
   reg_print_info(this->executableName, text);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::PrintCurrentObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::GetObjectiveFunctionGradient()
{

   if(!this->useApproxGradient)
   {
      // Compute the gradient of the similarity measure
      if(this->similarityWeight>0)
      {
         this->WarpFloatingImage(this->interpolation);
         this->GetSimilarityMeasureGradient();
      }
      else
      {
         this->SetGradientImageToZero();
      }
      // Compute the penalty term gradients if required
      this->GetBendingEnergyGradient();
      this->GetJacobianBasedGradient();
      this->GetLinearEnergyGradient();
      this->GetLandmarkDistanceGradient();
#ifdef BUILD_DEV
      this->GetPairwiseEnergyGradient();
#endif
   }
   else
   {
      this->GetApproximatedGradient();
   }

   this->optimiser->IncrementCurrentIterationNumber();

   // Smooth the gradient if require
   this->SmoothGradient();
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d<T>::CorrectTransformation()
{
   if(this->jacobianLogWeight>0 && this->jacobianLogApproximation==true)
      this->ComputeJacobianBasedPenaltyTerm(2); // 20 iterations without approximation
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::CorrectTransformation");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
#ifdef BUILD_DEV
template<class T>
void reg_f3d<T>::DiscreteInitialisation()
{
   // Check if the discrete initialisation can be performed
   if(this->measure_mind!=NULL || this->measure_mindssc!=NULL || this->measure_ssd!=NULL || sizeof(float)!=sizeof(T))
   {
      if(this->currentReference->nt>1){
         reg_print_fct_error("reg_f3d<T>::DiscreteInitialisation()");
         reg_print_msg_error("This function does not support 4D for now");
         reg_exit();
      }
      // Warp the floating image
      this->WarpFloatingImage(3);

      // Define the descriptor images
      nifti_image *MIND_refImg = NULL;
      nifti_image *MIND_warImg = NULL;

      // Set the length of the descriptor
      int desc_length = 1;
      if(this->measure_mind!=NULL)
         desc_length = 6;
      else if(this->measure_mindssc!=NULL)
         desc_length = 12;

      // Initialise the measure of similarity use to compute the distance between the blocks
      reg_ssd *ssdMeasure = new reg_ssd();
      for(int i=0;i<desc_length;++i)
         ssdMeasure->SetActiveTimepoint(i);

      if((this->measure_mind!=NULL || this->measure_mindssc!=NULL) && this->measure_ssd==NULL){
         // Allocate MIND descriptor of the reference image
         MIND_refImg = nifti_copy_nim_info(this->currentReference);
         MIND_refImg->ndim = MIND_refImg->dim[0] = 4;
         MIND_refImg->nt = MIND_refImg->dim[4] = desc_length;
         MIND_refImg->nvox = MIND_refImg->nvox*desc_length;
         MIND_refImg->data=(void *)calloc(MIND_refImg->nvox,
                                          MIND_refImg->nbyper);
         // Allocate MIND descriptor of the warped image
         MIND_warImg = nifti_copy_nim_info(this->warped);
         MIND_warImg->ndim = MIND_warImg->dim[0] = 4;
         MIND_warImg->nt = MIND_warImg->dim[4] = desc_length;
         MIND_warImg->nvox = MIND_warImg->nvox*desc_length;
         MIND_warImg->data=(void *)calloc(MIND_warImg->nvox,
                                          MIND_warImg->nbyper);

         // Allocate a mask embedding all voxel for the warped image
         int *temp_mask = (int *)calloc(this->warped->nx*this->warped->ny*this->warped->nz,
                                        sizeof(int));

         int offset = 1;
         if(this->measure_mindssc!=NULL)
            offset = this->measure_mindssc->GetDescriptorOffset();
         else offset = this->measure_mind->GetDescriptorOffset();

         // Compute the descriptors
         if(this->measure_mindssc!=NULL){
            // Compute the MINDSSC descriptor of the reference image
            GetMINDSSCImageDesciptor(this->currentReference,
                                     MIND_refImg,
                                     this->currentMask,
                                     offset,
                                     0);
            // Compute the MINDSSC descriptor of the warped image
            GetMINDSSCImageDesciptor(this->warped,
                                     MIND_warImg,
                                     temp_mask,
                                     offset,
                                     0);

         }
         else{
            // Compute the MIND descriptor of the reference image
            GetMINDImageDesciptor(this->currentReference,
                                  MIND_refImg,
                                  this->currentMask,
                                  offset,
                                  0);
            // Compute the MIND descriptor of the warped image
            GetMINDImageDesciptor(this->warped,
                                  MIND_warImg,
                                  temp_mask,
                                  offset,
                                  0);
         }
         free(temp_mask);
         // Initialise the measure with the descriptors
         ssdMeasure->InitialiseMeasure(MIND_refImg,
                                       MIND_warImg,
                                       this->currentMask,
                                       MIND_warImg,
                                       NULL,
                                       NULL);
      }
      else{
         // Initialise the measure with the input images
         ssdMeasure->InitialiseMeasure(this->currentReference,
                                       this->warped,
                                       this->currentMask,
                                       this->warped,
                                       NULL,
                                       NULL);
      }
      //
      // Create and initialise the discretisation initialisation object
      //
      //int discrete_increment=3;
      //int discretisation_radius=discrete_increment*reg_ceil(this->controlPointGrid->dx/this->currentReference->dx);
      int discrete_increment=1;
      //DEBUG
      //std::cout<<"(this->levelNumber-this->currentLevel-1)="<<(this->levelNumber-this->currentLevel-1)<<std::endl;
      //DEBUG
      int discretisation_radius=
              reg_ceil(discrete_increment*(this->controlPointGrid->dx/this->currentReference->dx)/pow(2.0,(this->levelNumber-this->currentLevel-1)));
      //
//      reg_discrete_init *discrete_init_object = new reg_discrete_init(ssdMeasure,
//                                                                      this->currentReference,
//                                                                      this->controlPointGrid,
//                                                                      discretisation_radius,
//                                                                      discrete_increment,
//                                                                      100,
//                                                                      this->bendingEnergyWeight*pow(16.f,(this->levelNumber-this->currentLevel-1)) +
//                                                                      this->linearEnergyWeight);
      reg_mrf *discrete_init_object = new reg_mrf(ssdMeasure,
                                                  this->currentReference,
                                                  this->controlPointGrid,
                                                  discretisation_radius,
                                                  discrete_increment,
                                                  this->bendingEnergyWeight*pow(16.f,(this->levelNumber-this->currentLevel-1)) +
                                                  this->linearEnergyWeight);

      // Run the discrete initialisation
      discrete_init_object->Run();

      // Free all the allocate objects
      if(MIND_refImg!=NULL)
         nifti_image_free(MIND_refImg);
      if(MIND_warImg!=NULL)
         nifti_image_free(MIND_warImg);
      delete ssdMeasure;
      delete discrete_init_object;
      char text[255];

      sprintf(text, "Discrete initialisation done");
      reg_print_info(this->executableName, text);
   }
   else{
      reg_print_msg_error("The discrete initialisation can only be performed when using SSD, MIND or MIND-SSC");
      reg_print_msg_error("when single precision is used.");
      reg_print_msg_error("No discrete initialisation has been performed");
   }
}
#endif
/* *************************************************************** */
/* *************************************************************** */

template class reg_f3d<float>;
#endif
