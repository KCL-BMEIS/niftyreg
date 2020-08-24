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

#include "_reg_f3d.h"

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
         this->linearEnergyWeight = this->linearEnergyWeight / static_cast<T>(powf(3.0f, this->levelNumber-1));
      }
      else
      {
         reg_spline_refineControlPointGrid(this->controlPointGrid,this->currentReference);
         this->bendingEnergyWeight = this->bendingEnergyWeight * static_cast<T>(16);
         this->linearEnergyWeight = this->linearEnergyWeight * static_cast<T>(3);
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
      nifti_image_free(this->transformationGradient);
      this->transformationGradient=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ClearTransformationGradient");
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
      T penaltySum=this->bendingEnergyWeight +
            this->linearEnergyWeight +
            this->jacobianLogWeight +
            this->landmarkRegWeight;
      if(penaltySum>=1.0)
      {
         this->similarityWeight=0;
         this->similarityWeight /= penaltySum;
         this->bendingEnergyWeight /= penaltySum;
         this->linearEnergyWeight /= penaltySum;
         this->jacobianLogWeight /= penaltySum;
         this->landmarkRegWeight /= penaltySum;
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
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
     std::string text;
      // Print out some global information about the registration
      reg_print_info(this->executableName, "***********************************************************");
      reg_print_info(this->executableName, "INPUT PARAMETERS");
      reg_print_info(this->executableName, "***********************************************************");
      reg_print_info(this->executableName, "Reference image:");
      text = stringFormat("\t* name: %s", this->inputReference->fname);
      reg_print_info(this->executableName, text.c_str());
      text = stringFormat("\t* image dimension: %i x %i x %i x %i",
              this->inputReference->nx, this->inputReference->ny,
              this->inputReference->nz, this->inputReference->nt);
      reg_print_info(this->executableName, text.c_str());
      text = stringFormat("\t* image spacing: %g x %g x %g mm",
              this->inputReference->dx,
              this->inputReference->dy, this->inputReference->dz);
      reg_print_info(this->executableName, text.c_str());
      for(int i=0; i<this->inputReference->nt; i++)
      {
         text = stringFormat("\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                 i, this->inputReference->nt-1, this->referenceThresholdLow[i],this->referenceThresholdUp[i]);
         reg_print_info(this->executableName, text.c_str());
         if(this->measure_nmi!=NULL){
            if(this->measure_nmi->GetTimepointsWeights()[i]>0.0){
               text = stringFormat("\t* binnining size for timepoint %i/%i: %i",
                       i, this->inputFloating->nt-1, this->measure_nmi->GetReferenceBinNumber()[i]-4);
               reg_print_info(this->executableName, text.c_str());
            }
         }
      }
      text = stringFormat("\t* gaussian smoothing sigma: %g", this->referenceSmoothingSigma);
      reg_print_info(this->executableName, text.c_str());
      reg_print_info(this->executableName, "");
      reg_print_info(this->executableName, "Floating image:");
      reg_print_info(this->executableName, text.c_str());
      text = stringFormat("\t* name: %s", this->inputFloating->fname);
      reg_print_info(this->executableName, text.c_str());
      text = stringFormat("\t* image dimension: %i x %i x %i x %i",
              this->inputFloating->nx, this->inputFloating->ny,
              this->inputFloating->nz, this->inputFloating->nt);
      reg_print_info(this->executableName, text.c_str());
      text = stringFormat("\t* image spacing: %g x %g x %g mm",
              this->inputFloating->dx,
              this->inputFloating->dy, this->inputFloating->dz);
      reg_print_info(this->executableName, text.c_str());
      for(int i=0; i<this->inputFloating->nt; i++)
      {
         text = stringFormat("\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                 i, this->inputFloating->nt-1, this->floatingThresholdLow[i],this->floatingThresholdUp[i]);
         reg_print_info(this->executableName, text.c_str());
         if(this->measure_nmi!=NULL){
          if (this->measure_nmi->GetTimepointsWeights()[i]>0.0){
               text = stringFormat("\t* binnining size for timepoint %i/%i: %i",
                       i, this->inputFloating->nt-1, this->measure_nmi->GetFloatingBinNumber()[i]-4);
               reg_print_info(this->executableName, text.c_str());
            }
         }
      }
      text = stringFormat("\t* gaussian smoothing sigma: %g", this->floatingSmoothingSigma);
      reg_print_info(this->executableName, text.c_str());
      reg_print_info(this->executableName, "");
      text = stringFormat("Warped image padding value: %g", this->warpedPaddingValue);
      reg_print_info(this->executableName, text.c_str());
      reg_print_info(this->executableName, "");
      text = stringFormat("Level number: %i", this->levelNumber);
      reg_print_info(this->executableName, text.c_str());
      if(this->levelNumber!=this->levelToPerform){
         text = stringFormat("\t* Level to perform: %i", this->levelToPerform);
         reg_print_info(this->executableName, text.c_str());
      }
      reg_print_info(this->executableName, "");
      text = stringFormat("Maximum iteration number during the last level: %i", (int)this->maxiterationNumber);
      reg_print_info(this->executableName, text.c_str());
      reg_print_info(this->executableName, "");

      text = stringFormat("Final spacing in mm: %g %g %g",
              this->spacing[0], this->spacing[1], this->spacing[2]);
      reg_print_info(this->executableName, text.c_str());
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
      text = stringFormat("Similarity measure term weight: %g", this->similarityWeight);
      reg_print_info(this->executableName, text.c_str());
      reg_print_info(this->executableName, "");
      if(this->bendingEnergyWeight>0){
         text = stringFormat("Bending energy penalty term weight: %g", this->bendingEnergyWeight);
         reg_print_info(this->executableName, text.c_str());
         reg_print_info(this->executableName, "");
      }
      if((this->linearEnergyWeight)>0){
         text = stringFormat("Linear energy penalty term weight: %g",
                 this->linearEnergyWeight);
         reg_print_info(this->executableName, text.c_str());
         reg_print_info(this->executableName, "");
      }
      if(this->jacobianLogWeight>0){
         text = stringFormat("Jacobian-based penalty term weight: %g", this->jacobianLogWeight);
         reg_print_info(this->executableName, text.c_str());
         if(this->jacobianLogApproximation){
            reg_print_info(this->executableName, "\t* Jacobian-based penalty term is approximated");
         }
         else reg_print_info(this->executableName, "\t* Jacobian-based penalty term is not approximated");
         reg_print_info(this->executableName, "");
      }
      if((this->landmarkRegWeight)>0){
         text = stringFormat("Landmark distance regularisation term weight: %g",
                 this->landmarkRegWeight);
         reg_print_info(this->executableName, text.c_str());
         reg_print_info(this->executableName, "");
      }
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
template <class T>
void reg_f3d<T>::GetSimilarityMeasureGradient()
{
   this->GetVoxelBasedGradient();


   /*char im_name[100];
   sprintf(im_name, "debug_vox_grad_lev_%i_iter_%i.nii", this->currentLevel, this->optimiser->GetCurrentIterationNumber());
   reg_io_WriteImageFile(this->voxelBasedMeasureGradient, im_name);*/



   int kernel_type=CUBIC_SPLINE_KERNEL;
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


   /*sprintf(im_name, "debug_cp_grad_lev_%i_iter_%i.nii", this->currentLevel, this->optimiser->GetCurrentIterationNumber());
   reg_io_WriteImageFile(this->transformationGradient, im_name);
   reg_exit(0);*/

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
   //	reg_heapSort(length,voxNumber);
   //	T maxGradValue = (T)(length[90*voxNumber/100 - 1]);
   //	free(length);


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
            //				T tempLength = (float)(sqrt(valX*valX + valY*valY + valZ*valZ));
            //				if(tempLength>maxGradValue){
            //					*ptrX *= maxGradValue / tempLength;
            //					*ptrY *= maxGradValue / tempLength;
            //					*ptrZ *= maxGradValue / tempLength;
            //				}
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
            //				T tempLength = (float)(sqrt(valX*valX + valY*valY));
            //				if(tempLength>maxGradValue){
            //					*ptrX *= maxGradValue / tempLength;
            //					*ptrY *= maxGradValue / tempLength;
            //				}
            *ptrX++ = valX / maxGradValue;
            *ptrY++ = valY / maxGradValue;
         }
      }
   }
   // Returns the largest gradient distance
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::NormaliseGradient");
#endif

   //   reg_io_WriteImageFile(transformationGradient,
   //                         "gradient.nii");
   //   reg_exit();

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

   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac - this->currentWLand;
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

   this->AllocateWarped();
   this->AllocateDeformationField();
   this->WarpFloatingImage(3); // cubic spline interpolation
   this->ClearDeformationField();

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
   sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g - (wLAN)%g",
           bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWJac, this->bestWLand);
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

template class reg_f3d<float>;
#endif
