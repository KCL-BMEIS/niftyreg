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

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_base<T>::reg_base(unsigned platformFlag, int refTimePoint, int floTimePoint)
{
   if(platformFlag == NR_PLATFORM_CPU)
        this->forwardGlobalContent = new GlobalContent(platformFlag, refTimePoint, floTimePoint);
#ifdef _USE_CUDA
   else if(platformFlag == NR_PLATFORM_CUDA)
        this->forwardGlobalContent = new CudaGlobalContent(refTimePoint, floTimePoint);
#endif
#ifdef _USE_OPENCL
   else if(platformFlag == NR_PLATFORM_CL)
        this->forwardGlobalContent = new ClGlobalContent(refTimePoint, floTimePoint);
#endif
   this->optimiser=NULL;
   this->maxiterationNumber=150;
   this->optimiseX=true;
   this->optimiseY=true;
   this->optimiseZ=true;
   this->perturbationNumber=0;
   this->useConjGradient=true;
   this->useApproxGradient=false;

   this->measure_ssd=NULL;
   this->measure_kld=NULL;
   this->measure_dti=NULL;
   this->measure_lncc=NULL;
   this->measure_nmi=NULL;
   this->measure_mind=NULL;
   this->measure_mindssc=NULL;

   this->similarityWeight=0.; // is automatically set depending of the penalty term weights

   this->executableName=(char *)"NiftyReg BASE";
   this->gradientSmoothingSigma=0;
   this->verbose=true;
   //this->usePyramid=true;
   this->forwardJacobianMatrix=NULL;

   this->initialised=false;

   this->warImgGradient=NULL;
   this->voxelBasedMeasureGradient=NULL;

   this->interpolation=1;

#ifdef BUILD_DEV
   this->discrete_init=false;
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::reg_base");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_base<T>::~reg_base()
{
   this->ClearWarpedGradient();
   this->ClearVoxelBasedMeasureGradient();

   if(this->optimiser!=NULL)
   {
      delete this->optimiser;
      this->optimiser=NULL;
   }

   if(this->measure_nmi!=NULL)
      delete this->measure_nmi;
   if(this->measure_ssd!=NULL)
      delete this->measure_ssd;
   if(this->measure_kld!=NULL)
      delete this->measure_kld;
   if(this->measure_dti!=NULL)
      delete this->measure_dti;
   if(this->measure_lncc!=NULL)
      delete this->measure_lncc;
   if(this->measure_mind!=NULL)
      delete this->measure_mind;
   if(this->measure_mindssc!=NULL)
      delete this->measure_mindssc;

   delete this->forwardGlobalContent;
   //Platform
//   delete this->platform;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::~reg_base");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
//template<class T>
//void reg_base<T>::setPlaform(Platform* inputPlatform)
//{
//    this->platform = inputPlatform;
//}
/* *************************************************************** */
//template<class T>
//Platform* reg_base<T>::getPlaform()
//{
//    return this->platform;
//}
/* *************************************************************** */
//template<class T>
//void reg_base<T>::setPlatformCode(int inputPlatformCode) {
//    this->platformCode = inputPlatformCode;
//}
/* *************************************************************** */
//template<class T>
//void reg_base<T>::setGpuIdx(unsigned inputGPUIdx) {
//    this->gpuIdx = inputGPUIdx;
//}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::SetInputReference(nifti_image *r)
{
   this->forwardGlobalContent->setInputReference(r);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetInputReference");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetInputFloating(nifti_image *f)
{
   this->forwardGlobalContent->setInputFloating(f);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetInputFloating");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetInputReferenceMask(nifti_image *m)
{
   this->forwardGlobalContent->setInputReferenceMask(m);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetInputReferenceMask");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetMaximalIterationNumber(unsigned int iter)
{
   this->maxiterationNumber=iter;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetMaximalIterationNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetAffineTransformation(mat44 *a)
{
   this->forwardGlobalContent->setAffineTransformation(a);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetAffineTransformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceSmoothingSigma(T s)
{
   this->forwardGlobalContent->setReferenceSmoothingSigma(s);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingSmoothingSigma(T s)
{
   this->forwardGlobalContent->setFloatingSmoothingSigma(s);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdUp(unsigned int i, T t)
{
   this->forwardGlobalContent->setReferenceThresholdUp(i,t);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceThresholdUp");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdLow(unsigned int i, T t)
{
   this->forwardGlobalContent->setReferenceThresholdLow(i,t);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceThresholdLow");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdUp(unsigned int i, T t)
{
   this->forwardGlobalContent->setFloatingThresholdUp(i,t);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingThresholdUp");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdLow(unsigned int i, T t)
{
   this->forwardGlobalContent->setFloatingThresholdLow(i,t);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingThresholdLow");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::UseRobustRange()
{
   this->forwardGlobalContent->useRobustRange();
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseRobustRange");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::DoNotUseRobustRange()
{
   this->forwardGlobalContent->doNotUseRobustRange();
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseRobustRange");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetWarpedPaddingValue(T p)
{
   this->forwardGlobalContent->setWarpedPaddingValue(p);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetWarpedPaddingValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelNumber(unsigned int l)
{
   this->forwardGlobalContent->setLevelNumber(l);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetLevelNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelToPerform(unsigned int l)
{
   this->forwardGlobalContent->setLevelToPerform(l);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetLevelToPerform");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetGradientSmoothingSigma(T g)
{
   this->gradientSmoothingSigma = g;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetGradientSmoothingSigma");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseConjugateGradient()
{
   this->useConjGradient = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseConjugateGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseConjugateGradient()
{
   this->useConjGradient = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotUseConjugateGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseApproximatedGradient()
{
   this->useApproxGradient = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseApproximatedGradient()
{
   this->useApproxGradient = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotUseApproximatedGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::PrintOutInformation()
{
   this->verbose = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::PrintOutInformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotPrintOutInformation()
{
   this->verbose = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotPrintOutInformation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUsePyramidalApproach()
{
   this->forwardGlobalContent->doNotUsePyramidalApproach();
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotUsePyramidalApproach");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNeareatNeighborInterpolation()
{
   this->interpolation=0;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseNeareatNeighborInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLinearInterpolation()
{
   this->interpolation=1;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseLinearInterpolation");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseCubicSplineInterpolation()
{
   this->interpolation=3;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseCubicSplineInterpolation");
#endif
}
/* *************************************************************** */
#ifdef BUILD_DEV
/* *************************************************************** */
template <class T>
void reg_base<T>::UseDiscreteInit()
{
   this->discrete_init=true;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::DoNotUseDiscreteInit()
{
   this->discrete_init=false;
}
/* *************************************************************** */
#endif
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearCurrentInputImage()
{
   this->forwardGlobalContent->ClearCurrentInputImages();
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearCurrentInputImage");
#endif
}
template <class T>
void reg_base<T>::AllocateWarpedGradient()
{
   if(this->forwardGlobalContent->getCurrentDeformationField()==NULL)
   {
      reg_print_fct_error("reg_base::AllocateWarpedGradient()");
      reg_print_msg_error("The deformation field image is not defined");
      reg_exit();
   }
   reg_base<T>::ClearWarpedGradient();
   this->warImgGradient = nifti_copy_nim_info(this->forwardGlobalContent->getCurrentDeformationField());
   this->warImgGradient->data = (void *)calloc(this->warImgGradient->nvox,
                                     this->warImgGradient->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::AllocateWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearWarpedGradient()
{
   if(this->warImgGradient!=NULL)
   {
      nifti_image_free(this->warImgGradient);
      this->warImgGradient=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::AllocateVoxelBasedMeasureGradient()
{
   if(this->forwardGlobalContent->getCurrentDeformationField()==NULL)
   {
      reg_print_fct_error("reg_base::AllocateVoxelBasedMeasureGradient()");
      reg_print_msg_error("The deformation field image is not defined");
      reg_exit();
   }
   reg_base<T>::ClearVoxelBasedMeasureGradient();
   this->voxelBasedMeasureGradient = nifti_copy_nim_info(this->forwardGlobalContent->getCurrentDeformationField());
   this->voxelBasedMeasureGradient->data = (void *)calloc(this->voxelBasedMeasureGradient->nvox,
         this->voxelBasedMeasureGradient->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::AllocateVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_base<T>::ClearVoxelBasedMeasureGradient()
{
   if(this->voxelBasedMeasureGradient!=NULL)
   {
      nifti_image_free(this->voxelBasedMeasureGradient);
      this->voxelBasedMeasureGradient=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::CheckParameters()
{
    this->forwardGlobalContent->CheckParameters();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::InitialiseSimilarity()
{
   // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
   if(this->measure_nmi==NULL &&
         this->measure_ssd==NULL &&
         this->measure_dti==NULL &&
         this->measure_lncc==NULL &&
         this->measure_lncc==NULL &&
         this->measure_kld==NULL &&
         this->measure_mind==NULL &&
         this->measure_mindssc==NULL)
   {
      this->measure_nmi=new reg_nmi;
      for(int i=0; i<this->forwardGlobalContent->getNbRefTimePoint(); ++i)
         this->measure_nmi->SetActiveTimepoint(i);
   }
   if(this->measure_nmi!=NULL)
      this->measure_nmi->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                           this->forwardGlobalContent->getCurrentFloating(),
                                           this->forwardGlobalContent->getCurrentReferenceMask(),
                                           this->forwardGlobalContent->getCurrentWarped(),
                                           this->warImgGradient,
                                           this->voxelBasedMeasureGradient
                                          );

   if(this->measure_ssd!=NULL)
      this->measure_ssd->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                           this->forwardGlobalContent->getCurrentFloating(),
                                           this->forwardGlobalContent->getCurrentReferenceMask(),
                                           this->forwardGlobalContent->getCurrentWarped(),
                                           this->warImgGradient,
                                           this->voxelBasedMeasureGradient
                                          );

   if(this->measure_kld!=NULL)
      this->measure_kld->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                           this->forwardGlobalContent->getCurrentFloating(),
                                           this->forwardGlobalContent->getCurrentReferenceMask(),
                                           this->forwardGlobalContent->getCurrentWarped(),
                                           this->warImgGradient,
                                           this->voxelBasedMeasureGradient
                                          );

   if(this->measure_lncc!=NULL)
      this->measure_lncc->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                            this->forwardGlobalContent->getCurrentFloating(),
                                            this->forwardGlobalContent->getCurrentReferenceMask(),
                                            this->forwardGlobalContent->getCurrentWarped(),
                                            this->warImgGradient,
                                            this->voxelBasedMeasureGradient
                                           );

   if(this->measure_dti!=NULL)
      this->measure_dti->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                           this->forwardGlobalContent->getCurrentFloating(),
                                           this->forwardGlobalContent->getCurrentReferenceMask(),
                                           this->forwardGlobalContent->getCurrentWarped(),
                                           this->warImgGradient,
                                           this->voxelBasedMeasureGradient
                                          );

   if(this->measure_mind!=NULL)
      this->measure_mind->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                            this->forwardGlobalContent->getCurrentFloating(),
                                            this->forwardGlobalContent->getCurrentReferenceMask(),
                                            this->forwardGlobalContent->getCurrentWarped(),
                                            this->warImgGradient,
                                            this->voxelBasedMeasureGradient
                                            );

   if(this->measure_mindssc!=NULL)
      this->measure_mindssc->InitialiseMeasure(this->forwardGlobalContent->getCurrentReference(),
                                               this->forwardGlobalContent->getCurrentFloating(),
                                               this->forwardGlobalContent->getCurrentReferenceMask(),
                                               this->forwardGlobalContent->getCurrentWarped(),
                                               this->warImgGradient,
                                               this->voxelBasedMeasureGradient
                                               );

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::InitialiseSimilarity");
#endif
   return;
}
/* *************************************************************** */
template<class T>
void reg_base<T>::Initialise()
{
   if(this->initialised) return;

   this->CheckParameters();

   this->forwardGlobalContent->InitialiseGlobalContent();

   this->initialised=true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::Initialise");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::SetOptimiser()
{
   if(this->useConjGradient)
      this->optimiser=new reg_conjugateGradient<T>();
   else this->optimiser=new reg_optimiser<T>();
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_base<T>::ComputeSimilarityMeasure()
{
   double measure=0.;
   if(this->measure_nmi!=NULL)
      measure += this->measure_nmi->GetSimilarityMeasureValue();

   if(this->measure_ssd!=NULL)
      measure += this->measure_ssd->GetSimilarityMeasureValue();

   if(this->measure_kld!=NULL)
      measure += this->measure_kld->GetSimilarityMeasureValue();

   if(this->measure_lncc!=NULL)
      measure += this->measure_lncc->GetSimilarityMeasureValue();

   if(this->measure_dti!=NULL)
      measure += this->measure_dti->GetSimilarityMeasureValue();

   if(this->measure_mind!=NULL)
      measure += this->measure_mind->GetSimilarityMeasureValue();

   if(this->measure_mindssc!=NULL)
      measure += this->measure_mindssc->GetSimilarityMeasureValue();

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ComputeSimilarityMeasure");
#endif
   return double(this->similarityWeight) * measure;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::GetVoxelBasedGradient()
{
   // The voxel based gradient image is filled with zeros
   reg_tools_multiplyValueToImage(this->voxelBasedMeasureGradient,
                                  this->voxelBasedMeasureGradient,
                                  0.f);

   // The intensity gradient is first computed
   //   if(this->measure_nmi!=NULL || this->measure_ssd!=NULL ||
   //         this->measure_kld!=NULL || this->measure_lncc!=NULL ||
   //         this->measure_dti!=NULL)
   //   {
   //    if(this->measure_dti!=NULL){
   //        reg_getImageGradient(this->currentFloating,
   //                             this->warImgGradient,
   //                             this->deformationFieldImage,
   //                             this->currentMask,
   //                             this->interpolation,
   //                             this->warpedPaddingValue,
   //                             this->measure_dti->GetActiveTimepoints(),
   //		 					   this->forwardJacobianMatrix,
   //							   this->warped);
   //    }
   //    else{
   //    }
   //   }

   //   if(this->measure_dti!=NULL)
   //      this->measure_dti->GetVoxelBasedSimilarityMeasureGradient();

   for(int t=0; t<this->forwardGlobalContent->getNbRefTimePoint(); ++t){
      reg_getImageGradient(this->forwardGlobalContent->getCurrentFloating(),
                           this->warImgGradient,
                           this->forwardGlobalContent->getCurrentDeformationField(),
                           this->forwardGlobalContent->getCurrentReferenceMask(),
                           this->interpolation,
                           this->forwardGlobalContent->getWarpedPaddingValue(),
                           t);

      // The gradient of the various measures of similarity are computed
      if(this->measure_nmi!=NULL)
         this->measure_nmi->GetVoxelBasedSimilarityMeasureGradient(t);

      if(this->measure_ssd!=NULL)
         this->measure_ssd->GetVoxelBasedSimilarityMeasureGradient(t);

      if(this->measure_kld!=NULL)
         this->measure_kld->GetVoxelBasedSimilarityMeasureGradient(t);

      if(this->measure_lncc!=NULL)
         this->measure_lncc->GetVoxelBasedSimilarityMeasureGradient(t);

      if(this->measure_mind!=NULL)
         this->measure_mind->GetVoxelBasedSimilarityMeasureGradient(t);

      if(this->measure_mindssc!=NULL)
         this->measure_mindssc->GetVoxelBasedSimilarityMeasureGradient(t);
   }

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::GetVoxelBasedGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
//template<class T>
//void reg_base<T>::ApproximateParzenWindow()
//{
//    if(this->measure_nmi==NULL)
//        this->measure_nmi=new reg_nmi;
//    this->measure_nmi=approxParzenWindow = true;
//    return;
//}
///* *************************************************************** */
//template<class T>
//void reg_base<T>::DoNotApproximateParzenWindow()
//{
//    if(this->measure_nmi==NULL)
//        this->measure_nmi=new reg_nmi;
//    this->measure_nmi=approxParzenWindow = false;
//    return;
//}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetReferenceBinNumber(int timepoint, int refBinNumber)
{
   if(this->measure_nmi==NULL)
      this->measure_nmi=new reg_nmi;
   this->measure_nmi->SetActiveTimepoint(timepoint);
   // I am here adding 4 to the specified bin number to accomodate for
   // the spline support
   this->measure_nmi->SetReferenceBinNumber(refBinNumber+4, timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseNMISetReferenceBinNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetFloatingBinNumber(int timepoint, int floBinNumber)
{
   if(this->measure_nmi==NULL)
      this->measure_nmi=new reg_nmi;
   this->measure_nmi->SetActiveTimepoint(timepoint);
   // I am here adding 4 to the specified bin number to accomodate for
   // the spline support
   this->measure_nmi->SetFloatingBinNumber(floBinNumber+4, timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseNMISetFloatingBinNumber");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseSSD(int timepoint)
{
   if(this->measure_ssd==NULL)
      this->measure_ssd=new reg_ssd;
   this->measure_ssd->SetActiveTimepoint(timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseSSD");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMIND(int timepoint, int offset)
{
   if(this->measure_mind==NULL)
      this->measure_mind=new reg_mind;
   this->measure_mind->SetActiveTimepoint(timepoint);
   this->measure_mind->SetDescriptorOffset(offset);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseMIND");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMINDSSC(int timepoint, int offset)
{
   if(this->measure_mindssc==NULL)
      this->measure_mindssc=new reg_mindssc;
   this->measure_mindssc->SetActiveTimepoint(timepoint);
   this->measure_mindssc->SetDescriptorOffset(offset);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseMINDSSC");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseKLDivergence(int timepoint)
{
   if(this->measure_kld==NULL)
      this->measure_kld=new reg_kld;
   this->measure_kld->SetActiveTimepoint(timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseKLDivergence");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLNCC(int timepoint, float stddev)
{
   if(this->measure_lncc==NULL)
      this->measure_lncc=new reg_lncc;
   this->measure_lncc->SetKernelStandardDeviation(timepoint,
         stddev);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseLNCC");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCKernelType(int type)
{
   if(this->measure_lncc==NULL)
   {
      reg_print_fct_error("reg_base<T>::SetLNCCKernelType");
      reg_print_msg_error("The LNCC object has to be created first");
      reg_exit();
   }
   this->measure_lncc->SetKernelType(type);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetLNCCKernelType");
#endif
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseDTI(bool *timepoint)
{
   reg_print_msg_error("The use of DTI has been deactivated as it requires some refactoring");
   reg_exit();

   if(this->measure_dti==NULL)
      this->measure_dti=new reg_dti;
   for(int i=0; i<this->forwardGlobalContent->getCurrentReference()->nt; ++i)
   {
      if(timepoint[i]==true)
         this->measure_dti->SetActiveTimepoint(i);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseDTI");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::WarpFloatingImage(int inter)
{
   // Compute the deformation field
   this->GetDeformationField();

   if(this->measure_dti==NULL)
   {
      // Resample the floating image
      reg_resampleImage(this->forwardGlobalContent->getCurrentFloating(),
                               this->forwardGlobalContent->getCurrentWarped(),
                               this->forwardGlobalContent->getCurrentDeformationField(),
                               this->forwardGlobalContent->getCurrentReferenceMask(),
                               inter,
                               this->forwardGlobalContent->getWarpedPaddingValue());
      //4 the moment - gpu kernel not implemented
      this->forwardGlobalContent->setCurrentWarped(this->forwardGlobalContent->GlobalContent::getCurrentWarped());
      //this->forwardGlobalContent->WarpFloatingImage(inter);
      //this->forwardGlobalContent->GlobalContent::setCurrentWarped(this->forwardGlobalContent->getCurrentWarped());
   }
   else
   {
      reg_defField_getJacobianMatrix(this->forwardGlobalContent->getCurrentDeformationField(),
                                     this->forwardJacobianMatrix);
      reg_resampleImage(this->forwardGlobalContent->getCurrentFloating(),
                        this->forwardGlobalContent->getCurrentWarped(),
                        this->forwardGlobalContent->getCurrentDeformationField(),
                        this->forwardGlobalContent->getCurrentReferenceMask(),
                        inter,
                        this->forwardGlobalContent->getWarpedPaddingValue(),
                        this->measure_dti->GetActiveTimepoints(),
                        this->forwardJacobianMatrix);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::WarpFloatingImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_base<T>::Run()
{
#ifndef NDEBUG
   char text[255];
   sprintf(text, "%s::Run() called", this->executableName);
   reg_print_msg_debug(text);
#endif
   //CPU init
   if(!this->initialised) this->Initialise();
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      reg_print_info(this->executableName, "***********************************************************");
#ifdef NDEBUG
   }
#endif

   // Update the maximal number of iteration to perform per level
   this->maxiterationNumber = this->maxiterationNumber * pow(2, this->forwardGlobalContent->getLevelToPerform()-1);

   // Loop over the different resolution level to perform
   for(this->currentLevel=0;
         this->currentLevel<this->forwardGlobalContent->getLevelToPerform();
         this->currentLevel++)
   {

      // Set the current input images
      if(this->forwardGlobalContent->isPyramidUsed()) {
        this->forwardGlobalContent->setCurrentReference(this->forwardGlobalContent->getReferencePyramid()[this->currentLevel]);
        this->forwardGlobalContent->setCurrentFloating(this->forwardGlobalContent->getFloatingPyramid()[this->currentLevel]);
        this->forwardGlobalContent->setCurrentReferenceMask(this->forwardGlobalContent->getMaskPyramid()[this->currentLevel], this->forwardGlobalContent->getActiveVoxelNumber()[this->currentLevel]);
      } else {
        this->forwardGlobalContent->setCurrentReference(this->forwardGlobalContent->getReferencePyramid()[0]);
        this->forwardGlobalContent->setCurrentFloating(this->forwardGlobalContent->getFloatingPyramid()[0]);
        this->forwardGlobalContent->setCurrentReferenceMask(this->forwardGlobalContent->getMaskPyramid()[0], this->forwardGlobalContent->getActiveVoxelNumber()[this->currentLevel]);
      }

      // Allocate image that depends on the reference image
      this->forwardGlobalContent->AllocateWarped();
      this->forwardGlobalContent->AllocateDeformationField();
      this->AllocateWarpedGradient();

      // The grid is refined if necessary
      T maxStepSize=this->InitialiseCurrentLevel();
      T currentSize = maxStepSize;
      T smallestSize = maxStepSize / (T)100.0;

      this->DisplayCurrentLevelParameters();

#ifdef BUILD_DEV
      // Perform the discrete initialisation if required
      if(this->discrete_init==true)
         this->DiscreteInitialisation();
#endif

      // Allocate image that are required to compute the gradient
      this->AllocateVoxelBasedMeasureGradient();
      this->AllocateTransformationGradient();

      // Initialise the measures of similarity
      this->InitialiseSimilarity();

      // initialise the optimiser
      this->SetOptimiser();

      // Loop over the number of perturbation to do
      for(size_t perturbation=0;
            perturbation<=this->perturbationNumber;
            ++perturbation)
      {

         // Evalulate the objective function value
         this->UpdateBestObjFunctionValue();
         this->PrintInitialObjFunctionValue();

         // Iterate until convergence or until the max number of iteration is reach
         while(true)
         {

            if(currentSize==0)
               break;

            if(this->optimiser->GetCurrentIterationNumber()>=this->optimiser->GetMaxIterationNumber()){
               reg_print_msg_warn("The current level reached the maximum number of iteration");
               break;
            }

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

         } // while
         if(perturbation<this->perturbationNumber)
         {

            this->optimiser->Perturbation(smallestSize);
            currentSize=maxStepSize;
#ifdef NDEBUG
            if(this->verbose)
            {
#endif
               char text[255];
               reg_print_info(this->executableName, "Perturbation Step - The number of iteration is reset to 0");
               sprintf(text, "Perturbation Step - Every control point positions is altered by [-%g %g]",
                      smallestSize, smallestSize);
               reg_print_info(this->executableName, text);

#ifdef NDEBUG
            }
#endif
         }
      } // perturbation loop

      // Final folding correction
      this->CorrectTransformation();

      // Some cleaning is performed
      delete this->optimiser;
      this->optimiser=NULL;
      this->forwardGlobalContent->ClearWarped();
      this->forwardGlobalContent->ClearDeformationField();
      this->ClearWarpedGradient();
      this->ClearVoxelBasedMeasureGradient();
      this->ClearTransformationGradient();
      if(this->forwardGlobalContent->isPyramidUsed()) {
         this->forwardGlobalContent->ClearCurrentImagePyramid(this->currentLevel);
      } else if(this->currentLevel==this->forwardGlobalContent->getLevelToPerform()-1) {
         this->forwardGlobalContent->ClearCurrentImagePyramid(0);
      }

      this->ClearCurrentInputImage();

#ifdef NDEBUG
      if(this->verbose)
      {
#endif
         reg_print_info(this->executableName, "Current registration level done");
         reg_print_info(this->executableName, "***********************************************************");
#ifdef NDEBUG
      }
#endif
      // Update the number of level for the next level
      this->maxiterationNumber /= 2;
   } // level this->levelToPerform

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::Run");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_base<T>::SetPlatformCode(const int platformCodeIn) {
    this->forwardGlobalContent->getPlatform()->setPlatformCode(platformCodeIn);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetGpuIdx(unsigned gpuIdxIn){
   this->forwardGlobalContent->getPlatform()->setGpuIdx(gpuIdxIn);
}
/* *************************************************************** */
/* *************************************************************** */
template class reg_base<float>;
#endif // _REG_BASE_CPP
