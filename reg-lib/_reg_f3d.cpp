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
reg_f3d<T>::reg_f3d(unsigned platformFlag, int refTimePoint,int floTimePoint)
   : reg_base<T>::reg_base(platformFlag, refTimePoint,floTimePoint)
{
   this->executableName=(char *)"NiftyReg F3D";
   //this->inputControlPointGrid=NULL; // pointer to external
   //this->controlPointGrid=NULL;
   this->bendingEnergyWeight=0.001;
   this->linearEnergyWeight=0.01;
   this->jacobianLogWeight=0.;
   this->jacobianLogApproximation=true;
   //this->spacing[0]=-5;
   //this->spacing[1]=std::numeric_limits<T>::quiet_NaN();
   //this->spacing[2]=std::numeric_limits<T>::quiet_NaN();
   this->useConjGradient=true;
   this->useApproxGradient=false;

   //    this->approxParzenWindow=true;

   this->transformationGradient=NULL;

   this->gridRefinement=true;

#ifdef BUILD_DEV
   pairwiseEnergyWeight=0;
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
   if(this->con->getCurrentControlPointGrid()!=NULL)
   {
      this->con->ClearControlPointGrid();
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
   this->con->setInputControlPointGrid(cp);
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
   this->con->setLinearSpline(true);
}
/* *************************************************************** */
template<class T>
void reg_f3d<T>::DoNotLinearSpline()
{
   this->con->setLinearSpline(false);
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
   this->con->setSpacing(i,s);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetSpacing");
#endif
}
/* *************************************************************** */
template <class T>
T reg_f3d<T>::InitialiseCurrentLevel()
{
   // Set the initial step size for the gradient ascent
   T maxStepSize = this->con->getCurrentReference()->dx>this->con->getCurrentReference()->dy?this->con->getCurrentReference()->dx:this->con->getCurrentReference()->dy;
   if(this->con->getCurrentReference()->ndim>2)
      maxStepSize = (this->con->getCurrentReference()->dz>maxStepSize)?this->con->getCurrentReference()->dz:maxStepSize;

   // Refine the control point grid if required
   if(this->gridRefinement==true)
   {
      if(this->currentLevel==0){
         this->bendingEnergyWeight = this->bendingEnergyWeight / static_cast<T>(powf(16.0f, this->con->getLevelNumber()-1));
      }
      else
      {
         Kernel* refineKernel = this->con->getPlatform()->createKernel(RefineControlPointGridKernel::getName(), this->con);
         refineKernel->castTo<RefineControlPointGridKernel>()->calculate();
         delete refineKernel;
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
   if(this->con->getCurrentControlPointGrid()==NULL)
   {
      reg_print_fct_error("reg_f3d<T>::AllocateTransformationGradient()");
      reg_print_msg_error("The control point image is not defined");
      reg_exit();
   }
   reg_f3d<T>::ClearTransformationGradient();
   this->transformationGradient = nifti_copy_nim_info(this->con->getCurrentControlPointGrid());
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
#ifdef BUILD_DEV
   if(this->con->getLinearSpline()==true){
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
   T penaltySum=this->bendingEnergyWeight +
           this->linearEnergyWeight +
           this->jacobianLogWeight+this->pairwiseEnergyWeight;
#else
      T penaltySum=this->bendingEnergyWeight +
            this->linearEnergyWeight +
            this->jacobianLogWeight;
#endif
      if(penaltySum>=1.0)
      {
         this->similarityWeight=0;
         this->similarityWeight /= penaltySum;
         this->bendingEnergyWeight /= penaltySum;
         this->linearEnergyWeight /= penaltySum;
         this->jacobianLogWeight /= penaltySum;
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
   //CPU init
   reg_base<T>::Initialise();

   //CPU init for the control point grid
   // DETERMINE THE GRID SPACING AND CREATE THE GRID
   if(this->con->getInputControlPointGrid()==NULL)
   {
      // Set the spacing along y and z if undefined. Their values are set to match
      // the spacing along the x axis
      if(this->con->getSpacing()[1]!=this->con->getSpacing()[1]) this->con->setSpacing(1,this->con->getSpacing()[0]);
      if(this->con->getSpacing()[2]!=this->con->getSpacing()[2]) this->con->setSpacing(2,this->con->getSpacing()[0]);

      /* Convert the spacing from voxel to mm if necessary */
      float spacingInMillimeter[3]= {this->con->getSpacing()[0],this->con->getSpacing()[1],this->con->getSpacing()[2]};
      if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->con->getInputReference()->dx;
      if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->con->getInputReference()->dy;
      if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->con->getInputReference()->dz;

      // Define the spacing for the first level
      float gridSpacing[3];
      gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(this->con->getLevelNumber()-1));
      gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(this->con->getLevelNumber()-1));
      gridSpacing[2] = 1.0f;
      if(this->con->getReferencePyramid()[0]->nz>1)
         gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(this->con->getLevelNumber()-1));

      // Create and allocate the control point image
      this->con->AllocateControlPointGrid(gridSpacing);

      // The control point position image is initialised with the affine transformation
      if(this->con->getAffineTransformation()==NULL)
      {
         memset(this->con->getCurrentControlPointGrid()->data,0,
                this->con->getCurrentControlPointGrid()->nvox*this->con->getCurrentControlPointGrid()->nbyper);
         reg_tools_multiplyValueToImage(this->con->getCurrentControlPointGrid(),this->con->getCurrentControlPointGrid(),0.f);
         reg_getDeformationFromDisplacement(this->con->getCurrentControlPointGrid());
         //4 the moment - to update the gpu
         this->con->setCurrentControlPointGrid(this->con->getCurrentControlPointGrid());
      }
      else
      {
          reg_affine_getDeformationField(this->con->getAffineTransformation(), this->con->getCurrentControlPointGrid());
          //4 the moment - to update the gpu
          this->con->setCurrentControlPointGrid(this->con->getCurrentControlPointGrid());
      }
   }
   else
   {
      // The control point grid image is initialised with the provided grid
      this->con->AllocateControlPointGrid();
   }
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
      sprintf(text, "\t* name: %s", this->con->getInputReference()->fname);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->con->getInputReference()->nx, this->con->getInputReference()->ny,
              this->con->getInputReference()->nz, this->con->getInputReference()->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->con->getInputReference()->dx,
              this->con->getInputReference()->dy, this->con->getInputReference()->dz);
      reg_print_info(this->executableName, text);
      for(int i=0; i<this->con->getInputReference()->nt; i++)
      {
         sprintf(text, "\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                 i, this->con->getInputReference()->nt-1, this->con->getReferenceThresholdLow()[i],this->con->getReferenceThresholdUp()[i]);
         reg_print_info(this->executableName, text);
         if(this->measure_nmi!=NULL){
            if(this->measure_nmi->GetActiveTimepoints()[i]){
               sprintf(text, "\t* binnining size for timepoint %i/%i: %i",
                       i, this->con->getInputFloating()->nt-1, this->measure_nmi->GetReferenceBinNumber()[i]-4);
               reg_print_info(this->executableName, text);
            }
         }
      }
      sprintf(text, "\t* gaussian smoothing sigma: %g", this->con->getReferenceSmoothingSigma());
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      reg_print_info(this->executableName, "Floating image:");
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* name: %s", this->con->getInputFloating()->fname);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->con->getInputFloating()->nx, this->con->getInputFloating()->ny,
              this->con->getInputFloating()->nz, this->con->getInputFloating()->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->con->getInputFloating()->dx,
              this->con->getInputFloating()->dy, this->con->getInputFloating()->dz);
      reg_print_info(this->executableName, text);
      for(int i=0; i<this->con->getInputFloating()->nt; i++)
      {
         sprintf(text, "\t* intensity threshold for timepoint %i/%i: [%.2g %.2g]",
                 i, this->con->getInputFloating()->nt-1, this->con->getFloatingThresholdLow()[i],this->con->getFloatingThresholdUp()[i]);
         reg_print_info(this->executableName, text);
         if(this->measure_nmi!=NULL){
            if(this->measure_nmi->GetActiveTimepoints()[i]){
               sprintf(text, "\t* binnining size for timepoint %i/%i: %i",
                       i, this->con->getInputFloating()->nt-1, this->measure_nmi->GetFloatingBinNumber()[i]-4);
               reg_print_info(this->executableName, text);
            }
         }
      }
      sprintf(text, "\t* gaussian smoothing sigma: %g", this->con->getFloatingSmoothingSigma());
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      sprintf(text, "Warped image padding value: %g", this->con->getWarpedPaddingValue());
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");
      sprintf(text, "Level number: %i", this->con->getLevelNumber());
      reg_print_info(this->executableName, text);
      if(this->con->getLevelNumber()!=this->con->getLevelToPerform()){
         sprintf(text, "\t* Level to perform: %i", this->con->getLevelToPerform());
         reg_print_info(this->executableName, text);
      }
      reg_print_info(this->executableName, "");
      sprintf(text, "Maximum iteration number during the last level: %i", (int)this->maxiterationNumber);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "");

#ifdef BUILD_DEV
      if(this->con->getLinearSpline()){
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
              this->con->getSpacing()[0], this->con->getSpacing()[1], this->con->getSpacing()[2]);
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
   Kernel* splineKernel = this->con->getPlatform()->createKernel(SplineDeformationFieldKernel::getName(), this->con);
   splineKernel->castTo<SplineDeformationFieldKernel>()->calculate(false);
   delete splineKernel;
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
      value = reg_spline_getJacobianPenaltyTerm(this->con->getCurrentControlPointGrid(),
                                                this->con->getCurrentReference(),
                                                false);
   }
   else
   {
      value = reg_spline_getJacobianPenaltyTerm(this->con->getCurrentControlPointGrid(),
                                                this->con->getCurrentReference(),
                                                this->jacobianLogApproximation);
   }
   unsigned int maxit=5;
   if(type>0) maxit=20;
   unsigned int it=0;
   while(value!=value && it<maxit)
   {
      if(type==2)
      {
         value = reg_spline_correctFolding(this->con->getCurrentControlPointGrid(),
                                           this->con->getCurrentReference(),
                                           false);
      }
      else
      {
         value = reg_spline_correctFolding(this->con->getCurrentControlPointGrid(),
                                           this->con->getCurrentReference(),
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

   double value = reg_spline_approxBendingEnergy(this->con->getCurrentControlPointGrid());
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

   double value = reg_spline_approxLinearEnergy(this->con->getCurrentControlPointGrid());

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeLinearEnergyPenaltyTerm");
#endif
   return this->linearEnergyWeight*value;
}
/* *************************************************************** */
/* *************************************************************** */
#ifdef BUILD_DEV
template <class T>
double reg_f3d<T>::ComputePairwiseEnergyPenaltyTerm()
{
   if(this->pairwiseEnergyWeight<=0)
      return 0.;

   double value = reg_spline_approxLinearPairwise(this->con->getCurrentControlPointGrid());

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
   if(this->con->getLinearSpline())
      kernel_type=LINEAR_KERNEL;
#endif
   // The voxel based NMI gradient is convolved with a spline kernel
   // Convolution along the x axis
   float currentNodeSpacing[3];
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->con->getCurrentControlPointGrid()->dx;
   bool activeAxis[3]= {1,0,0};
   reg_tools_kernelConvolution(this->voxelBasedMeasureGradient,
                               currentNodeSpacing,
                               kernel_type,
                               NULL, // mask
                               NULL, // all volumes are considered as active
                               activeAxis
                               );
   // Convolution along the y axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->con->getCurrentControlPointGrid()->dy;
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
      currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->con->getCurrentControlPointGrid()->dz;
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
   if(this->con->getCurrentFloating()->sform_code>0)
      reorientation = this->con->getCurrentFloating()->sto_ijk;
   else reorientation = this->con->getCurrentFloating()->qto_ijk;
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

   reg_spline_approxBendingEnergyGradient(this->con->getCurrentControlPointGrid(),
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

   reg_spline_approxLinearEnergyGradient(this->con->getCurrentControlPointGrid(),
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

   reg_spline_getJacobianPenaltyTermGradient(this->con->getCurrentControlPointGrid(),
                                             this->con->getCurrentReference(),
                                             this->transformationGradient,
                                             this->jacobianLogWeight,
                                             this->jacobianLogApproximation);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetJacobianBasedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
#ifdef BUILD_DEV
template <class T>
void reg_f3d<T>::GetPairwiseEnergyGradient()
{
   if(this->pairwiseEnergyWeight<=0) return;

   reg_spline_approxLinearPairwiseGradient(this->con->getCurrentControlPointGrid(),
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
      sprintf(text, "Current level: %i / %i", this->currentLevel+1, this->con->getLevelNumber());
      reg_print_info(this->executableName, text);
      sprintf(text, "Maximum iteration number: %i", (int)this->maxiterationNumber);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "Current reference image");
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->con->getCurrentReference()->nx, this->con->getCurrentReference()->ny,
              this->con->getCurrentReference()->nz,this->con->getCurrentReference()->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->con->getCurrentReference()->dx, this->con->getCurrentReference()->dy,
              this->con->getCurrentReference()->dz);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "Current floating image");
      sprintf(text, "\t* image dimension: %i x %i x %i x %i",
              this->con->getCurrentFloating()->nx, this->con->getCurrentFloating()->ny,
              this->con->getCurrentFloating()->nz,this->con->getCurrentFloating()->nt);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->con->getCurrentFloating()->dx, this->con->getCurrentFloating()->dy,
              this->con->getCurrentFloating()->dz);
      reg_print_info(this->executableName, text);
      reg_print_info(this->executableName, "Current control point image");
      sprintf(text, "\t* image dimension: %i x %i x %i",
              this->con->getCurrentControlPointGrid()->nx, this->con->getCurrentControlPointGrid()->ny,
              this->con->getCurrentControlPointGrid()->nz);
      reg_print_info(this->executableName, text);
      sprintf(text, "\t* image spacing: %g x %g x %g mm",
              this->con->getCurrentControlPointGrid()->dx, this->con->getCurrentControlPointGrid()->dy,
              this->con->getCurrentControlPointGrid()->dz);
      reg_print_info(this->executableName, text);
#ifdef NDEBUG
   }
#endif

#ifndef NDEBUG
   if(this->con->getCurrentReference()->sform_code>0)
      reg_mat44_disp(&(this->con->getCurrentReference()->sto_xyz), (char *)"[NiftyReg DEBUG] Reference sform");
   else reg_mat44_disp(&(this->con->getCurrentReference()->qto_xyz), (char *)"[NiftyReg DEBUG] Reference qform");

   if(this->con->getCurrentFloating()->sform_code>0)
      reg_mat44_disp(&(this->con->getCurrentFloating()->sto_xyz), (char *)"[NiftyReg DEBUG] Floating sform");
   else reg_mat44_disp(&(this->con->getCurrentFloating()->qto_xyz), (char *)"[NiftyReg DEBUG] Floating qform");

   if(this->con->getCurrentControlPointGrid()->sform_code>0)
      reg_mat44_disp(&(this->con->getCurrentControlPointGrid()->sto_xyz), (char *)"[NiftyReg DEBUG] CPP sform");
   else reg_mat44_disp(&(this->con->getCurrentControlPointGrid()->qto_xyz), (char *)"[NiftyReg DEBUG] CPP qform");
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
   sprintf(text, "(wMeasure) %g | (wBE) %g | (wLE) %g | (wJac) %g",
           this->currentWMeasure, this->currentWBE, this->currentWLE, this->currentWJac);
   reg_print_msg_debug(text);
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionValue");
#endif
   // Store the global objective function value

#ifdef BUILD_DEV
   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac - this->currentWPE;
#else
   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWJac;
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
   this->optimiser->Initialise(this->con->getCurrentControlPointGrid()->nvox,
                               this->con->getCurrentControlPointGrid()->nz>1?3:2,
                               this->optimiseX,
                               this->optimiseY,
                               this->optimiseZ,
                               this->maxiterationNumber,
                               0, // currentIterationNumber,
                               this,
                               static_cast<T *>(this->con->getCurrentControlPointGrid()->data),
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
   T *gridPtr = static_cast<T *>(this->con->getCurrentControlPointGrid()->data);
   T *gradPtr = static_cast<T *>(this->transformationGradient->data);
   T eps = this->con->getCurrentControlPointGrid()->dx / 100.f;
   for(size_t i=0; i<this->con->getCurrentControlPointGrid()->nvox; ++i)
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
   if(this->con->getInputReference()==NULL ||
         this->con->getInputFloating()==NULL ||
         this->con->getCurrentControlPointGrid()==NULL)
   {
      reg_print_fct_error("reg_f3d<T>::GetWarpedImage()");
      reg_print_msg_error("The reference, floating and control point grid images have to be defined");
      reg_exit();
   }

   int *mask = (int *)calloc(this->con->getInputReference()->nx*
                             this->con->getInputReference()->ny*
                             this->con->getInputReference()->nz,
                             sizeof(int));

   this->con->setCurrentReference(this->con->getInputReference());
   this->con->setCurrentFloating(this->con->getInputFloating());
   this->con->setCurrentReferenceMask(mask,
                                      this->con->getInputReference()->nx*
                                      this->con->getInputReference()->ny*
                                      this->con->getInputReference()->nz);

   reg_base<T>::AllocateWarped();
   reg_base<T>::AllocateDeformationField();
   reg_base<T>::WarpFloatingImage(3); // cubic spline interpolation
   reg_base<T>::ClearDeformationField();
   free(mask);

   nifti_image **warpedImage= (nifti_image **)malloc(2*sizeof(nifti_image *));
   warpedImage[0]=nifti_copy_nim_info(this->con->getCurrentWarped());
   warpedImage[0]->cal_min=this->con->getInputFloating()->cal_min;
   warpedImage[0]->cal_max=this->con->getInputFloating()->cal_max;
   warpedImage[0]->scl_slope=this->con->getInputFloating()->scl_slope;
   warpedImage[0]->scl_inter=this->con->getInputFloating()->scl_inter;
   warpedImage[0]->data=(void *)malloc(warpedImage[0]->nvox*warpedImage[0]->nbyper);
   memcpy(warpedImage[0]->data, this->con->getCurrentWarped()->data, warpedImage[0]->nvox*warpedImage[0]->nbyper);

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
   nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->con->getCurrentControlPointGrid());
   returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   memcpy(returnedControlPointGrid->data, this->con->getCurrentControlPointGrid()->data,
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
   sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g - (wPW)%g",
           bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWJac, this->bestWPE);
#else
   sprintf(text, "Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wJAC)%g",
           bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWJac);
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
      sprintf(text+strlen(text), "- (wJAC)%.2e", this->bestWJac);
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
      if(this->con->getCurrentReference()->nt>1){
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
         MIND_refImg = nifti_copy_nim_info(this->con->getCurrentReference());
         MIND_refImg->ndim = MIND_refImg->dim[0] = 4;
         MIND_refImg->nt = MIND_refImg->dim[4] = desc_length;
         MIND_refImg->nvox = MIND_refImg->nvox*desc_length;
         MIND_refImg->data=(void *)calloc(MIND_refImg->nvox,
                                          MIND_refImg->nbyper);
         // Allocate MIND descriptor of the warped image
         MIND_warImg = nifti_copy_nim_info(this->con->getCurrentWarped());
         MIND_warImg->ndim = MIND_warImg->dim[0] = 4;
         MIND_warImg->nt = MIND_warImg->dim[4] = desc_length;
         MIND_warImg->nvox = MIND_warImg->nvox*desc_length;
         MIND_warImg->data=(void *)calloc(MIND_warImg->nvox,
                                          MIND_warImg->nbyper);

         // Allocate a mask embedding all voxel for the warped image
         int *temp_mask = (int *)calloc(this->con->getCurrentWarped()->nx*this->con->getCurrentWarped()->ny*this->con->getCurrentWarped()->nz,
                                        sizeof(int));

         int offset = 1;
         if(this->measure_mindssc!=NULL)
            offset = this->measure_mindssc->GetDescriptorOffset();
         else offset = this->measure_mind->GetDescriptorOffset();

         // Compute the descriptors
         if(this->measure_mindssc!=NULL){
            // Compute the MINDSSC descriptor of the reference image
            GetMINDSSCImageDesciptor(this->con->getCurrentReference(),
                                     MIND_refImg,
                                     this->con->getCurrentReferenceMask(),
                                     offset,
                                     0);
            // Compute the MINDSSC descriptor of the warped image
            GetMINDSSCImageDesciptor(this->con->getCurrentWarped(),
                                     MIND_warImg,
                                     temp_mask,
                                     offset,
                                     0);

         }
         else{
            // Compute the MIND descriptor of the reference image
            GetMINDImageDesciptor(this->con->getCurrentReference(),
                                  MIND_refImg,
                                  this->con->getCurrentReferenceMask(),
                                  offset,
                                  0);
            // Compute the MIND descriptor of the warped image
            GetMINDImageDesciptor(this->con->getCurrentWarped(),
                                  MIND_warImg,
                                  temp_mask,
                                  offset,
                                  0);
         }
         free(temp_mask);
         // Initialise the measure with the descriptors
         ssdMeasure->InitialiseMeasure(MIND_refImg,
                                       MIND_warImg,
                                       this->con->getCurrentReferenceMask(),
                                       MIND_warImg,
                                       NULL,
                                       NULL);
      }
      else{
         // Initialise the measure with the input images
         ssdMeasure->InitialiseMeasure(this->con->getCurrentReference(),
                                       this->con->getCurrentWarped(),
                                       this->con->getCurrentReferenceMask(),
                                       this->con->getCurrentWarped(),
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
              reg_ceil(discrete_increment*(this->con->getCurrentControlPointGrid()->dx/this->con->getCurrentReference()->dx)/pow(2.0,(this->con->getLevelNumber()-this->currentLevel-1)));
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
                                                  this->con->getCurrentReference(),
                                                  this->con->getCurrentControlPointGrid(),
                                                  discretisation_radius,
                                                  discrete_increment,
                                                  this->bendingEnergyWeight*pow(16.f,(this->con->getLevelNumber()-this->currentLevel-1)) +
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
