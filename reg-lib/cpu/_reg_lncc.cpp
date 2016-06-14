/**
 * @file  _reg_lncc.cpp
 * @author Aileen Corder
 * @author Marc Modat
 * @date 10/11/2012.
 * @brief CPP file for the LNCC related class and functions
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_LNCC_CPP
#define _REG_LNCC_CPP

#include "_reg_lncc.h"

/* *************************************************************** */
/* *************************************************************** */
reg_lncc::reg_lncc()
   : reg_measure()
{
   this->forwardCorrelationImage=NULL;
   this->referenceMeanImage=NULL;
   this->referenceSdevImage=NULL;
   this->warpedFloatingMeanImage=NULL;
   this->warpedFloatingSdevImage=NULL;
   this->forwardMask = NULL;

   this->backwardCorrelationImage=NULL;
   this->floatingMeanImage=NULL;
   this->floatingSdevImage=NULL;
   this->warpedReferenceMeanImage=NULL;
   this->warpedReferenceSdevImage=NULL;
   this->backwardMask = NULL;

   // Gaussian kernel is used by default
   this->kernelType=GAUSSIAN_KERNEL;

   for(int i=0; i<255; ++i)
      kernelStandardDeviation[i]=-5.f;
#ifndef NDEBUG
   reg_print_msg_debug("reg_lncc constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
reg_lncc::~reg_lncc()
{
   if(this->forwardCorrelationImage!=NULL)
      nifti_image_free(this->forwardCorrelationImage);
   this->forwardCorrelationImage=NULL;
   if(this->referenceMeanImage!=NULL)
      nifti_image_free(this->referenceMeanImage);
   this->referenceMeanImage=NULL;
   if(this->referenceSdevImage!=NULL)
      nifti_image_free(this->referenceSdevImage);
   this->referenceSdevImage=NULL;
   if(this->warpedFloatingMeanImage!=NULL)
      nifti_image_free(this->warpedFloatingMeanImage);
   this->warpedFloatingMeanImage=NULL;
   if(this->warpedFloatingSdevImage!=NULL)
      nifti_image_free(this->warpedFloatingSdevImage);
   this->warpedFloatingSdevImage=NULL;
   if(this->forwardMask!=NULL)
      free(this->forwardMask);
   this->forwardMask=NULL;

   if(this->backwardCorrelationImage!=NULL)
      nifti_image_free(this->backwardCorrelationImage);
   this->backwardCorrelationImage=NULL;
   if(this->floatingMeanImage!=NULL)
      nifti_image_free(this->floatingMeanImage);
   this->floatingMeanImage=NULL;
   if(this->floatingSdevImage!=NULL)
      nifti_image_free(this->floatingSdevImage);
   this->floatingSdevImage=NULL;
   if(this->warpedReferenceMeanImage!=NULL)
      nifti_image_free(this->warpedReferenceMeanImage);
   this->warpedReferenceMeanImage=NULL;
   if(this->warpedReferenceSdevImage!=NULL)
      nifti_image_free(this->warpedReferenceSdevImage);
   this->warpedReferenceSdevImage=NULL;
   if(this->backwardMask!=NULL)
      free(this->backwardMask);
   this->backwardMask=NULL;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_lncc::UpdateLocalStatImages(nifti_image *refImage,
                                     nifti_image *warImage,
                                     nifti_image *meanRefImage,
                                     nifti_image *meanWarImage,
                                     nifti_image *stdDevRefImage,
                                     nifti_image *stdDevWarImage,
                                     int *refMask,
                                     int *combinedMask,
                                     int current_timepoint)
{
   // Generate the foward mask to ignore all NaN values
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)refImage->nx*refImage->ny*refImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)refImage->nx*refImage->ny*refImage->nz;
#endif
   memcpy(combinedMask, refMask, voxelNumber*sizeof(int));
   reg_tools_removeNanFromMask(refImage, combinedMask);
   reg_tools_removeNanFromMask(warImage, combinedMask);

   DTYPE *origRefPtr = static_cast<DTYPE *>(refImage->data);
   DTYPE *meanRefPtr = static_cast<DTYPE *>(meanRefImage->data);
   DTYPE *sdevRefPtr = static_cast<DTYPE *>(stdDevRefImage->data);
   memcpy(meanRefPtr, &origRefPtr[current_timepoint*voxelNumber],
         voxelNumber*refImage->nbyper);
   memcpy(sdevRefPtr, &origRefPtr[current_timepoint*voxelNumber],
         voxelNumber*refImage->nbyper);

   reg_tools_multiplyImageToImage(stdDevRefImage, stdDevRefImage, stdDevRefImage);
   reg_tools_kernelConvolution(meanRefImage, this->kernelStandardDeviation,
                               this->kernelType, combinedMask);
   reg_tools_kernelConvolution(stdDevRefImage, this->kernelStandardDeviation,
                               this->kernelType, combinedMask);

   DTYPE *origWarPtr = static_cast<DTYPE *>(warImage->data);
   DTYPE *meanWarPtr = static_cast<DTYPE *>(meanWarImage->data);
   DTYPE *sdevWarPtr = static_cast<DTYPE *>(stdDevWarImage->data);
   memcpy(meanWarPtr, &origWarPtr[current_timepoint*voxelNumber],
         voxelNumber*warImage->nbyper);
   memcpy(sdevWarPtr, &origWarPtr[current_timepoint*voxelNumber],
         voxelNumber*warImage->nbyper);

   reg_tools_multiplyImageToImage(stdDevWarImage, stdDevWarImage, stdDevWarImage);
   reg_tools_kernelConvolution(meanWarImage, this->kernelStandardDeviation,
                               this->kernelType, combinedMask);
   reg_tools_kernelConvolution(stdDevWarImage, this->kernelStandardDeviation,
                               this->kernelType, combinedMask);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber, sdevRefPtr, meanRefPtr, sdevWarPtr, meanWarPtr) \
   private(voxel)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // G*(I^2) - (G*I)^2
      sdevRefPtr[voxel] = sqrt(sdevRefPtr[voxel] - reg_pow2(meanRefPtr[voxel]));
      sdevWarPtr[voxel] = sqrt(sdevWarPtr[voxel] - reg_pow2(meanWarPtr[voxel]));
      // Stabilise the computation
      if(sdevRefPtr[voxel]<1.e-06) sdevRefPtr[voxel]=static_cast<DTYPE>(0);
      if(sdevWarPtr[voxel]<1.e-06) sdevWarPtr[voxel]=static_cast<DTYPE>(0);
   }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_lncc::InitialiseMeasure(nifti_image *refImgPtr,
                                 nifti_image *floImgPtr,
                                 int *maskRefPtr,
                                 nifti_image *warFloImgPtr,
                                 nifti_image *warFloGraPtr,
                                 nifti_image *forVoxBasedGraPtr,
                                 int *maskFloPtr,
                                 nifti_image *warRefImgPtr,
                                 nifti_image *warRefGraPtr,
                                 nifti_image *bckVoxBasedGraPtr)
{
   reg_measure::InitialiseMeasure(refImgPtr,
                                  floImgPtr,
                                  maskRefPtr,
                                  warFloImgPtr,
                                  warFloGraPtr,
                                  forVoxBasedGraPtr,
                                  maskFloPtr,
                                  warRefImgPtr,
                                  warRefGraPtr,
                                  bckVoxBasedGraPtr);

   for(int i=0; i<this->referenceImagePointer->nt; ++i)
   {
      if(this->activeTimePoint[i])
      {
         reg_intensityRescale(this->referenceImagePointer,
                              i,
                              0.f,
                              1.f);
         reg_intensityRescale(this->floatingImagePointer,
                              i,
                              0.f,
                              1.f);
      }
   }

   // Check that no images are already allocated
   if(this->forwardCorrelationImage!=NULL)
      nifti_image_free(this->forwardCorrelationImage);
   this->forwardCorrelationImage=NULL;
   if(this->referenceMeanImage!=NULL)
      nifti_image_free(this->referenceMeanImage);
   this->referenceMeanImage=NULL;
   if(this->referenceSdevImage!=NULL)
      nifti_image_free(this->referenceSdevImage);
   this->referenceSdevImage=NULL;
   if(this->warpedFloatingMeanImage!=NULL)
      nifti_image_free(this->warpedFloatingMeanImage);
   this->warpedFloatingMeanImage=NULL;
   if(this->warpedFloatingSdevImage!=NULL)
      nifti_image_free(this->warpedFloatingSdevImage);
   this->warpedFloatingSdevImage=NULL;
   if(this->backwardCorrelationImage!=NULL)
      nifti_image_free(this->backwardCorrelationImage);
   this->backwardCorrelationImage=NULL;
   if(this->floatingMeanImage!=NULL)
      nifti_image_free(this->floatingMeanImage);
   this->floatingMeanImage=NULL;
   if(this->floatingSdevImage!=NULL)
      nifti_image_free(this->floatingSdevImage);
   this->floatingSdevImage=NULL;
   if(this->warpedReferenceMeanImage!=NULL)
      nifti_image_free(this->warpedReferenceMeanImage);
   this->warpedReferenceMeanImage=NULL;
   if(this->warpedReferenceSdevImage!=NULL)
      nifti_image_free(this->warpedReferenceSdevImage);
   this->warpedReferenceSdevImage=NULL;
   if(this->forwardMask!=NULL)
      free(this->forwardMask);
   this->forwardMask=NULL;
   if(this->backwardMask!=NULL)
      free(this->backwardMask);
   this->backwardMask=NULL;

   //
   size_t voxelNumber = (size_t)this->referenceImagePointer->nx *
         this->referenceImagePointer->ny * this->referenceImagePointer->nz;

   // Allocate the required image to store the correlation of the forward transformation
   this->forwardCorrelationImage=nifti_copy_nim_info(this->referenceImagePointer);
   this->forwardCorrelationImage->ndim=this->forwardCorrelationImage->dim[0]=this->referenceImagePointer->nz>1?3:2;
   this->forwardCorrelationImage->nt=this->forwardCorrelationImage->dim[4]=1;
   this->forwardCorrelationImage->nvox=voxelNumber;
   this->forwardCorrelationImage->data=(void *)malloc(voxelNumber *
                                                      this->forwardCorrelationImage->nbyper);

   // Allocate the required images to store mean and stdev of the reference image
   this->referenceMeanImage=nifti_copy_nim_info(this->forwardCorrelationImage);
   this->referenceMeanImage->data=(void *)malloc(this->referenceMeanImage->nvox *
                                                 this->referenceMeanImage->nbyper);

   this->referenceSdevImage=nifti_copy_nim_info(this->forwardCorrelationImage);
   this->referenceSdevImage->data=(void *)malloc(this->referenceSdevImage->nvox *
                                                 this->referenceSdevImage->nbyper);

   // Allocate the required images to store mean and stdev of the warped floating image
   this->warpedFloatingMeanImage=nifti_copy_nim_info(this->forwardCorrelationImage);
   this->warpedFloatingMeanImage->data=(void *)malloc(this->warpedFloatingMeanImage->nvox *
                                                      this->warpedFloatingMeanImage->nbyper);

   this->warpedFloatingSdevImage=nifti_copy_nim_info(this->forwardCorrelationImage);
   this->warpedFloatingSdevImage->data=(void *)malloc(this->warpedFloatingSdevImage->nvox *
                                                      this->warpedFloatingSdevImage->nbyper);

   // Allocate the array to store the mask of the forward image
   this->forwardMask=(int *)malloc(voxelNumber*sizeof(int));
   if(this->isSymmetric)
   {
      voxelNumber = (size_t)floatingImagePointer->nx *
            floatingImagePointer->ny * floatingImagePointer->nz;
      // Allocate the required image to store the correlation of the backward transformation
      this->backwardCorrelationImage=nifti_copy_nim_info(this->floatingImagePointer);
      this->backwardCorrelationImage->ndim=this->backwardCorrelationImage->dim[0]=this->floatingImagePointer->nz>1?3:2;
      this->backwardCorrelationImage->nt=this->backwardCorrelationImage->dim[4]=1;
      this->backwardCorrelationImage->nvox=voxelNumber;
      this->backwardCorrelationImage->data=(void *)malloc(voxelNumber *
                                                          this->backwardCorrelationImage->nbyper);

      // Allocate the required images to store mean and stdev of the floating image
      this->floatingMeanImage=nifti_copy_nim_info(this->backwardCorrelationImage);
      this->floatingMeanImage->data=(void *)malloc(this->floatingMeanImage->nvox *
                                                   this->floatingMeanImage->nbyper);

      this->floatingSdevImage=nifti_copy_nim_info(this->backwardCorrelationImage);
      this->floatingSdevImage->data=(void *)malloc(this->floatingSdevImage->nvox *
                                                   this->floatingSdevImage->nbyper);

      // Allocate the required images to store mean and stdev of the warped reference image
      this->warpedReferenceMeanImage=nifti_copy_nim_info(this->backwardCorrelationImage);
      this->warpedReferenceMeanImage->data=(void *)malloc(this->warpedReferenceMeanImage->nvox *
                                                          this->warpedReferenceMeanImage->nbyper);

      this->warpedReferenceSdevImage=nifti_copy_nim_info(this->backwardCorrelationImage);
      this->warpedReferenceSdevImage->data=(void *)malloc(this->warpedReferenceSdevImage->nvox *
                                                          this->warpedReferenceSdevImage->nbyper);

      // Allocate the array to store the mask of the backward image
      this->backwardMask=(int *)malloc(voxelNumber*sizeof(int));
   }
#ifndef NDEBUG
   char text[255];
   reg_print_msg_debug("reg_lncc::InitialiseMeasure().");
   sprintf(text, "Active time point:");
   for(int i=0; i<this->referenceImagePointer->nt; ++i)
      if(this->activeTimePoint[i])
         sprintf(text, "%s %i", text, i);
   reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getLNCCValue(nifti_image *referenceImage,
                        nifti_image *referenceMeanImage,
                        nifti_image *referenceSdevImage,
                        nifti_image *warpedImage,
                        nifti_image *warpedMeanImage,
                        nifti_image *warpedSdevImage,
                        int *combinedMask,
                        float *kernelStandardDeviation,
                        nifti_image *correlationImage,
                        int kernelType,
                        int current_timepoint)
{
#ifdef _WIN32
   long voxel;
   long voxelNumber=(long)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber=(size_t)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#endif

   // Compute the local correlation
   DTYPE *refImagePtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *currentRefPtr = &refImagePtr[current_timepoint*voxelNumber];

   DTYPE *warImagePtr=static_cast<DTYPE *>(warpedImage->data);
   DTYPE *currentWarPtr = &warImagePtr[current_timepoint*voxelNumber];

   DTYPE *refMeanPtr=static_cast<DTYPE *>(referenceMeanImage->data);
   DTYPE *warMeanPtr=static_cast<DTYPE *>(warpedMeanImage->data);
   DTYPE *refSdevPtr=static_cast<DTYPE *>(referenceSdevImage->data);
   DTYPE *warSdevPtr=static_cast<DTYPE *>(warpedSdevImage->data);
   DTYPE *correlaPtr=static_cast<DTYPE *>(correlationImage->data);

   for(size_t i=0; i<voxelNumber; ++i)
      correlaPtr[i] = currentRefPtr[i] * currentWarPtr[i];

   reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);

   double lncc_value_sum  = 0., lncc_value;
   double activeVoxel_num = 0.;

   // Iteration over all voxels
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,combinedMask,refMeanPtr,warMeanPtr, \
   refSdevPtr,warSdevPtr,correlaPtr) \
   private(voxel,lncc_value) \
   reduction(+:lncc_value_sum) \
   reduction(+:activeVoxel_num)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // Check if the current voxel belongs to the mask
      if(combinedMask[voxel]>-1)
      {
         lncc_value = (
                  correlaPtr[voxel] -
                  (refMeanPtr[voxel]*warMeanPtr[voxel])
                  ) /
               (refSdevPtr[voxel]*warSdevPtr[voxel]);

         if(lncc_value==lncc_value && isinf(lncc_value)==0)
         {
            lncc_value_sum += fabs(lncc_value);
            ++activeVoxel_num;
         }
      }
   }
   return lncc_value_sum/activeVoxel_num;
}
/* *************************************************************** */
/* *************************************************************** */
double reg_lncc::GetSimilarityMeasureValue()
{
   double lncc_value=0.f;
   int number_activeTimePoint = 0;

   for(int current_timepoint=0; current_timepoint<this->referenceImagePointer->nt; ++current_timepoint)
   {
      if(this->activeTimePoint[current_timepoint]==true)
      {
         // Compute the mean and variance of the reference and warped floating
         switch(this->referenceImagePointer->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            this->UpdateLocalStatImages<float>(this->referenceImagePointer,
                                               this->warpedFloatingImagePointer,
                                               this->referenceMeanImage,
                                               this->warpedFloatingMeanImage,
                                               this->referenceSdevImage,
                                               this->warpedFloatingSdevImage,
                                               this->referenceMaskPointer,
                                               this->forwardMask,
                                               current_timepoint);
            break;
         case NIFTI_TYPE_FLOAT64:
            this->UpdateLocalStatImages<double>(this->referenceImagePointer,
                                                this->warpedFloatingImagePointer,
                                                this->referenceMeanImage,
                                                this->warpedFloatingMeanImage,
                                                this->referenceSdevImage,
                                                this->warpedFloatingSdevImage,
                                                this->referenceMaskPointer,
                                                this->forwardMask,
                                                current_timepoint);
            break;
         }

         // Compute the LNCC - Forward
         switch(this->referenceImagePointer->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            lncc_value += reg_getLNCCValue<float>(this->referenceImagePointer,
                                                  this->referenceMeanImage,
                                                  this->referenceSdevImage,
                                                  this->warpedFloatingImagePointer,
                                                  this->warpedFloatingMeanImage,
                                                  this->warpedFloatingSdevImage,
                                                  this->forwardMask,
                                                  this->kernelStandardDeviation,
                                                  this->forwardCorrelationImage,
                                                  this->kernelType,
                                                  current_timepoint);
            break;
         case NIFTI_TYPE_FLOAT64:
            lncc_value += reg_getLNCCValue<double>(this->referenceImagePointer,
                                                   this->referenceMeanImage,
                                                   this->referenceSdevImage,
                                                   this->warpedFloatingImagePointer,
                                                   this->warpedFloatingMeanImage,
                                                   this->warpedFloatingSdevImage,
                                                   this->forwardMask,
                                                   this->kernelStandardDeviation,
                                                   this->forwardCorrelationImage,
                                                   this->kernelType,
                                                   current_timepoint);
            break;
         }
         if(this->isSymmetric)
         {
            // Compute the mean and variance of the floating and warped reference
            switch(this->floatingImagePointer->datatype)
            {
            case NIFTI_TYPE_FLOAT32:
               this->UpdateLocalStatImages<float>(this->floatingImagePointer,
                                                  this->warpedReferenceImagePointer,
                                                  this->floatingMeanImage,
                                                  this->warpedReferenceMeanImage,
                                                  this->floatingSdevImage,
                                                  this->warpedReferenceSdevImage,
                                                  this->floatingMaskPointer,
                                                  this->backwardMask,
                                                  current_timepoint);
               break;
            case NIFTI_TYPE_FLOAT64:
               this->UpdateLocalStatImages<double>(this->floatingImagePointer,
                                                   this->warpedReferenceImagePointer,
                                                   this->floatingMeanImage,
                                                   this->warpedReferenceMeanImage,
                                                   this->floatingSdevImage,
                                                   this->warpedReferenceSdevImage,
                                                   this->floatingMaskPointer,
                                                   this->backwardMask,
                                                   current_timepoint);
               break;
            }
            // Compute the LNCC - Backward
            switch(this->floatingImagePointer->datatype)
            {
            case NIFTI_TYPE_FLOAT32:
               lncc_value += reg_getLNCCValue<float>(this->floatingImagePointer,
                                                     this->floatingMeanImage,
                                                     this->floatingSdevImage,
                                                     this->warpedReferenceImagePointer,
                                                     this->warpedReferenceMeanImage,
                                                     this->warpedReferenceSdevImage,
                                                     this->backwardMask,
                                                     this->kernelStandardDeviation,
                                                     this->backwardCorrelationImage,
                                                     this->kernelType,
                                                     current_timepoint);
               break;
            case NIFTI_TYPE_FLOAT64:
               lncc_value += reg_getLNCCValue<double>(this->floatingImagePointer,
                                                      this->floatingMeanImage,
                                                      this->floatingSdevImage,
                                                      this->warpedReferenceImagePointer,
                                                      this->warpedReferenceMeanImage,
                                                      this->warpedReferenceSdevImage,
                                                      this->backwardMask,
                                                      this->kernelStandardDeviation,
                                                      this->backwardCorrelationImage,
                                                      this->kernelType,
                                                      current_timepoint);
               break;
            }
         }
      number_activeTimePoint++;
      }
   }
   return lncc_value/static_cast<double>(number_activeTimePoint);
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *referenceMeanImage,
                                   nifti_image *referenceSdevImage,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedMeanImage,
                                   nifti_image *warpedSdevImage,
                                   int *combinedMask,
                                   float *kernelStandardDeviation,
                                   nifti_image *correlationImage,
                                   nifti_image *warImgGradient,
                                   nifti_image *measureGradientImage,
                                   int kernelType,
                                   int current_timepoint)
{
#ifdef _WIN32
   long voxel;
   long voxelNumber=(long)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber=(size_t)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#endif

   // Compute the local correlation
   DTYPE *refImagePtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *currentRefPtr = &refImagePtr[current_timepoint*voxelNumber];

   DTYPE *warImagePtr=static_cast<DTYPE *>(warpedImage->data);
   DTYPE *currentWarPtr = &warImagePtr[current_timepoint*voxelNumber];

   DTYPE *refMeanPtr=static_cast<DTYPE *>(referenceMeanImage->data);
   DTYPE *warMeanPtr=static_cast<DTYPE *>(warpedMeanImage->data);
   DTYPE *refSdevPtr=static_cast<DTYPE *>(referenceSdevImage->data);
   DTYPE *warSdevPtr=static_cast<DTYPE *>(warpedSdevImage->data);
   DTYPE *correlaPtr=static_cast<DTYPE *>(correlationImage->data);

   for(size_t i=0; i<voxelNumber; ++i)
      correlaPtr[i] = currentRefPtr[i] * currentWarPtr[i];

   reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);

   double refMeanValue, warMeanValue, refSdevValue,
         warSdevValue, correlaValue;
   double temp1, temp2, temp3;

   // Iteration over all voxels
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,combinedMask,refMeanPtr,warMeanPtr, \
   refSdevPtr,warSdevPtr,correlaPtr) \
   private(voxel,refMeanValue,warMeanValue,refSdevValue, \
   warSdevValue, correlaValue, temp1, temp2, temp3)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // Check if the current voxel belongs to the mask
      if(combinedMask[voxel]>-1)
      {

         refMeanValue = refMeanPtr[voxel];
         warMeanValue = warMeanPtr[voxel];
         refSdevValue = refSdevPtr[voxel];
         warSdevValue = warSdevPtr[voxel];
         correlaValue = correlaPtr[voxel] - (refMeanValue*warMeanValue);

         temp1 = 1.0 / (refSdevValue * warSdevValue);
         temp2 = correlaValue /
               (refSdevValue*warSdevValue*warSdevValue*warSdevValue);
         temp3 = (correlaValue * warMeanValue) /
               (refSdevValue*warSdevValue*warSdevValue*warSdevValue)
               -
               refMeanValue / (refSdevValue * warSdevValue);
         if(temp1==temp1 && isinf(temp1)==0 &&
               temp2==temp2 && isinf(temp2)==0 &&
               temp3==temp3 && isinf(temp3)==0)
         {
            // Derivative of the absolute function
            if(correlaValue<0)
            {
               temp1 *= -1.;
               temp2 *= -1.;
               temp3 *= -1.;
            }
            warMeanPtr[voxel]=temp1;
            warSdevPtr[voxel]=temp2;
            correlaPtr[voxel]=temp3;
         }
         else warMeanPtr[voxel]=warSdevPtr[voxel]=correlaPtr[voxel]=0.;
      }
      else warMeanPtr[voxel]=warSdevPtr[voxel]=correlaPtr[voxel]=0.;
   }

   // Smooth the newly computed values
   reg_tools_kernelConvolution(warpedMeanImage, kernelStandardDeviation, kernelType, combinedMask);
   reg_tools_kernelConvolution(warpedSdevImage, kernelStandardDeviation, kernelType, combinedMask);
   reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);
   DTYPE *measureGradPtrX = static_cast<DTYPE *>(measureGradientImage->data);
   DTYPE *measureGradPtrY = &measureGradPtrX[voxelNumber];
   DTYPE *measureGradPtrZ = NULL;
   if(referenceImage->nz>1)
      measureGradPtrZ = &measureGradPtrY[voxelNumber];

   // Create pointers to the spatial gradient of the warped image
   DTYPE *warpGradPtrX = static_cast<DTYPE *>(warImgGradient->data);
   DTYPE *warpGradPtrY = &warpGradPtrX[voxelNumber];
   DTYPE *warpGradPtrZ = NULL;
   if(referenceImage->nz>1)
      warpGradPtrZ=&warpGradPtrY[voxelNumber];

   double common;
   // Iteration over all voxels
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,combinedMask,currentRefPtr,currentWarPtr, \
   warMeanPtr,warSdevPtr,correlaPtr,measureGradPtrX,measureGradPtrY, \
   measureGradPtrZ, warpGradPtrX, warpGradPtrY, warpGradPtrZ) \
   private(voxel, common)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // Check if the current voxel belongs to the mask
      if(combinedMask[voxel]>-1)
      {
         common = warMeanPtr[voxel] * currentRefPtr[voxel] -
               warSdevPtr[voxel] * currentWarPtr[voxel] +
               correlaPtr[voxel];
         measureGradPtrX[voxel] -= warpGradPtrX[voxel] * common;
         measureGradPtrY[voxel] -= warpGradPtrY[voxel] * common;
         if(warpGradPtrZ!=NULL)
            measureGradPtrZ[voxel] -= warpGradPtrZ[voxel] * common;
      }
   }
   // Check for NaN
   DTYPE val;
#ifdef _WIN32
   voxelNumber = (long)measureGradientImage->nvox;
#else
   voxelNumber=measureGradientImage->nvox;
#endif
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,measureGradPtrX) \
   private(voxel, val)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      val=measureGradPtrX[voxel];
      if(val!=val || isinf(val)!=0)
         measureGradPtrX[voxel]=static_cast<DTYPE>(0);
   }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_lncc::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint)
{
   // Check if the specified time point exists and is active
   reg_measure::GetVoxelBasedSimilarityMeasureGradient(current_timepoint);
   if(this->activeTimePoint[current_timepoint]==false)
      return;

   // Compute the mean and variance of the reference and warped floating
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      this->UpdateLocalStatImages<float>(this->referenceImagePointer,
                                         this->warpedFloatingImagePointer,
                                         this->referenceMeanImage,
                                         this->warpedFloatingMeanImage,
                                         this->referenceSdevImage,
                                         this->warpedFloatingSdevImage,
                                         this->referenceMaskPointer,
                                         this->forwardMask,
                                         current_timepoint);
      break;
   case NIFTI_TYPE_FLOAT64:
      this->UpdateLocalStatImages<double>(this->referenceImagePointer,
                                          this->warpedFloatingImagePointer,
                                          this->referenceMeanImage,
                                          this->warpedFloatingMeanImage,
                                          this->referenceSdevImage,
                                          this->warpedFloatingSdevImage,
                                          this->referenceMaskPointer,
                                          this->forwardMask,
                                          current_timepoint);
      break;
   }

   // Compute the LNCC gradient - Forward
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getVoxelBasedLNCCGradient<float>(this->referenceImagePointer,
                                           this->referenceMeanImage,
                                           this->referenceSdevImage,
                                           this->warpedFloatingImagePointer,
                                           this->warpedFloatingMeanImage,
                                           this->warpedFloatingSdevImage,
                                           this->forwardMask,
                                           this->kernelStandardDeviation,
                                           this->forwardCorrelationImage,
                                           this->warpedFloatingGradientImagePointer,
                                           this->forwardVoxelBasedGradientImagePointer,
                                           this->kernelType,
                                           current_timepoint);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getVoxelBasedLNCCGradient<double>(this->referenceImagePointer,
                                            this->referenceMeanImage,
                                            this->referenceSdevImage,
                                            this->warpedFloatingImagePointer,
                                            this->warpedFloatingMeanImage,
                                            this->warpedFloatingSdevImage,
                                            this->forwardMask,
                                            this->kernelStandardDeviation,
                                            this->forwardCorrelationImage,
                                            this->warpedFloatingGradientImagePointer,
                                            this->forwardVoxelBasedGradientImagePointer,
                                            this->kernelType,
                                            current_timepoint);
      break;
   }
   if(this->isSymmetric)
   {
      // Compute the mean and variance of the floating and warped reference
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         this->UpdateLocalStatImages<float>(this->floatingImagePointer,
                                            this->warpedReferenceImagePointer,
                                            this->floatingMeanImage,
                                            this->warpedReferenceMeanImage,
                                            this->floatingSdevImage,
                                            this->warpedReferenceSdevImage,
                                            this->floatingMaskPointer,
                                            this->backwardMask,
                                            current_timepoint);
         break;
      case NIFTI_TYPE_FLOAT64:
         this->UpdateLocalStatImages<double>(this->floatingImagePointer,
                                             this->warpedReferenceImagePointer,
                                             this->floatingMeanImage,
                                             this->warpedReferenceMeanImage,
                                             this->floatingSdevImage,
                                             this->warpedReferenceSdevImage,
                                             this->floatingMaskPointer,
                                             this->backwardMask,
                                             current_timepoint);
         break;
      }
      // Compute the LNCC gradient - Backward
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedLNCCGradient<float>(this->floatingImagePointer,
                                              this->floatingMeanImage,
                                              this->floatingSdevImage,
                                              this->warpedReferenceImagePointer,
                                              this->warpedReferenceMeanImage,
                                              this->warpedReferenceSdevImage,
                                              this->backwardMask,
                                              this->kernelStandardDeviation,
                                              this->backwardCorrelationImage,
                                              this->warpedReferenceGradientImagePointer,
                                              this->backwardVoxelBasedGradientImagePointer,
                                              this->kernelType,
                                              current_timepoint);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedLNCCGradient<double>(this->floatingImagePointer,
                                               this->floatingMeanImage,
                                               this->floatingSdevImage,
                                               this->warpedReferenceImagePointer,
                                               this->warpedReferenceMeanImage,
                                               this->warpedReferenceSdevImage,
                                               this->backwardMask,
                                               this->kernelStandardDeviation,
                                               this->backwardCorrelationImage,
                                               this->warpedReferenceGradientImagePointer,
                                               this->backwardVoxelBasedGradientImagePointer,
                                               this->kernelType,
                                               current_timepoint);
         break;
      }
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
#endif

