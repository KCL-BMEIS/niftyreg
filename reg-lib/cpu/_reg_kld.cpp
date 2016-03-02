/*
 *  _reg_kld.cpp
 *
 *
 *  Created by Marc Modat on 14/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_kld.h"

/* *************************************************************** */
/* *************************************************************** */
reg_kld::reg_kld()
   : reg_measure()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_kld constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_kld::InitialiseMeasure(nifti_image *refImgPtr,
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
   // Set the pointers using the parent class function
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

   // Check that the input images have the same number of time point
   if(this->referenceImagePointer->nt != this->floatingImagePointer->nt)
   {
      reg_print_fct_error("reg_kld::InitialiseMeasure");
      reg_print_msg_error("This number of time point should be the same for both input images");
      reg_exit();
   }
   // Input images are expected to be bounded between 0 and 1 as they
   // are meant to be probabilities
   for(int t=0; t<this->referenceImagePointer->nt; ++t){
      if(this->activeTimePoint[t]==true){
         float min_ref = reg_tools_getMinValue(this->referenceImagePointer, t);
         float max_ref = reg_tools_getMaxValue(this->referenceImagePointer, t);
         float min_flo = reg_tools_getMinValue(this->floatingImagePointer, t);
         float max_flo = reg_tools_getMaxValue(this->floatingImagePointer, t);
         if(min_ref<0.f || min_flo<0.f || max_ref>1.f || max_flo>1.f){
            reg_print_msg_error("The input images are expected to be probabilities to use the kld measure");
            reg_exit();
         }
      }
   }
#ifndef NDEBUG
   char text[255];
   reg_print_msg_debug("reg_kld::InitialiseMeasure().");
   sprintf(text, "Active time point:");
   for(int i=0; i<this->referenceImagePointer->nt; ++i)
      if(this->activeTimePoint[i])
         sprintf(text, "%s %i", text, i);
   reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_getKLDivergence(nifti_image *referenceImage,
                           nifti_image *warpedImage,
                           bool *activeTimePoint,
                           nifti_image *jacobianDetImg,
                           int *mask)
{
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif

   DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);
   int *maskPtr=NULL;
   bool MrClean=false;
   if(mask==NULL)
   {
      maskPtr=(int *)calloc(voxelNumber,sizeof(int));
      MrClean=true;
   }
   else maskPtr = &mask[0];

   DTYPE *jacPtr=NULL;
   if(jacobianDetImg!=NULL)
      jacPtr=static_cast<DTYPE *>(jacobianDetImg->data);
   double measure=0., num=0., tempRefValue, tempWarValue, tempValue;

   for(int time=0; time<referenceImage->nt; ++time)
   {
      if(activeTimePoint[time])
      {
         DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
         DTYPE *currentWarPtr=&warPtr[time*voxelNumber];
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,currentRefPtr, currentWarPtr, \
   maskPtr, jacobianDetImg, jacPtr) \
   private(voxel, tempRefValue, tempWarValue, tempValue) \
   reduction(+:measure) \
   reduction(+:num)
#endif
         for(voxel=0; voxel<voxelNumber; ++voxel)
         {
            if(maskPtr[voxel]>-1)
            {
               tempRefValue = currentRefPtr[voxel]+1e-16;
               tempWarValue = currentWarPtr[voxel]+1e-16;
               tempValue=tempRefValue*fabs(log(tempRefValue/tempWarValue));
               if(tempValue==tempValue &&
                     tempValue!=std::numeric_limits<double>::infinity())
               {
                  if(jacobianDetImg==NULL)
                  {
                     measure -= tempValue;
                     num++;
                  }
                  else
                  {
                     measure -= tempValue * jacPtr[voxel];
                     num+=jacPtr[voxel];
                  }
               }
            }
         }
      }
   }
   if(MrClean==true) free(maskPtr);
   return measure/num;
}
template double reg_getKLDivergence<float>
(nifti_image *,nifti_image *,bool *,nifti_image *,int *);
template double reg_getKLDivergence<double>
(nifti_image *,nifti_image *,bool *,nifti_image *,int *);
/* *************************************************************** */
double reg_kld::GetSimilarityMeasureValue()
{
   // Check that all the specified image are of the same datatype
   if(this->warpedFloatingImagePointer->datatype != this->referenceImagePointer->datatype)
   {
      reg_print_fct_error("reg_kld::GetSimilarityMeasureValue");
      reg_print_msg_error("Both input images are exepected to have the same type");
      reg_exit();
   }
   double KLDValue;
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      KLDValue = reg_getKLDivergence<float>
            (this->referenceImagePointer,
             this->warpedFloatingImagePointer,
             this->activeTimePoint,
             NULL, // HERE TODO this->forwardJacDetImagePointer,
             this->referenceMaskPointer
             );
      break;
   case NIFTI_TYPE_FLOAT64:
      KLDValue = reg_getKLDivergence<double>
            (this->referenceImagePointer,
             this->warpedFloatingImagePointer,
             this->activeTimePoint,
             NULL, // HERE TODO this->forwardJacDetImagePointer,
             this->referenceMaskPointer
             );
      break;
   default:
      reg_print_fct_error("reg_kld::GetSimilarityMeasureValue");
      reg_print_msg_error("Warped pixel type unsupported");
      reg_exit();
   }

   // Backward computation
   if(this->isSymmetric)
   {
      // Check that all the specified image are of the same datatype
      if(this->warpedReferenceImagePointer->datatype != this->floatingImagePointer->datatype)
      {
         reg_print_fct_error("reg_kld::GetSimilarityMeasureValue");
         reg_print_msg_error("Both input images are exepected to have the same type");
         reg_exit();
      }
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         KLDValue += reg_getKLDivergence<float>
               (this->floatingImagePointer,
                this->warpedReferenceImagePointer,
                this->activeTimePoint,
                NULL, // HERE TODO this->backwardJacDetImagePointer,
                this->floatingMaskPointer
                );
         break;
      case NIFTI_TYPE_FLOAT64:
         KLDValue += reg_getKLDivergence<double>
               (this->floatingImagePointer,
                this->warpedReferenceImagePointer,
                this->activeTimePoint,
                NULL, // HERE TODO this->backwardJacDetImagePointer,
                this->floatingMaskPointer
                );
         break;
      default:
         reg_print_fct_error("reg_kld::GetSimilarityMeasureValue");
         reg_print_msg_error("Warped pixel type unsupported");
         reg_exit();
      }
   }
   return KLDValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getKLDivergenceVoxelBasedGradient(nifti_image *referenceImage,
                                           nifti_image *warpedImage,
                                           nifti_image *warpedImageGradient,
                                           nifti_image *measureGradient,
                                           nifti_image *jacobianDetImg,
                                           int *mask,
                                           int current_timepoint)
{
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t  voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif

   DTYPE *refImagePtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warImagePtr=static_cast<DTYPE *>(warpedImage->data);
   DTYPE *currentRefPtr = &refImagePtr[current_timepoint*voxelNumber];
   DTYPE *currentWarPtr = &warImagePtr[current_timepoint*voxelNumber];
   int *maskPtr=NULL;
   bool MrClean=false;
   if(mask==NULL)
   {
      maskPtr=(int *)calloc(voxelNumber,sizeof(int));
      MrClean=true;
   }
   else maskPtr = &mask[0];

   DTYPE *jacPtr=NULL;
   if(jacobianDetImg!=NULL)
      jacPtr=static_cast<DTYPE *>(jacobianDetImg->data);
   double tempValue, tempGradX, tempGradY, tempGradZ, tempRefValue, tempWarValue;

   // Create pointers to the spatial gradient of the current warped volume
   DTYPE *currentGradPtrX=static_cast<DTYPE *>(warpedImageGradient->data);
   DTYPE *currentGradPtrY=&currentGradPtrX[voxelNumber];
   DTYPE *currentGradPtrZ=NULL;
   if(referenceImage->nz>1)
      currentGradPtrZ=&currentGradPtrY[voxelNumber];

   // Create pointers to the kld gradient image
   DTYPE *measureGradPtrX = static_cast<DTYPE *>(measureGradient->data);
   DTYPE *measureGradPtrY = &measureGradPtrX[voxelNumber];
   DTYPE *measureGradPtrZ = NULL;
   if(referenceImage->nz>1)
      measureGradPtrZ = &measureGradPtrY[voxelNumber];

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,currentRefPtr, currentWarPtr, \
   maskPtr, jacobianDetImg, jacPtr, referenceImage, \
   measureGradPtrX, measureGradPtrY, measureGradPtrZ, \
   currentGradPtrX, currentGradPtrY, currentGradPtrZ) \
   private(voxel, tempValue, tempGradX, tempGradY, tempGradZ, \
   tempRefValue, tempWarValue)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // Check if the current voxel is in the mask
      if(maskPtr[voxel]>-1)
      {
         // Read referenceImage and warpedImage probabilities and compute the ratio
         tempRefValue = currentRefPtr[voxel]+1e-16;
         tempWarValue = currentWarPtr[voxel]+1e-16;
         tempValue=(currentRefPtr[voxel]+1e-16)/(currentWarPtr[voxel]+1e-16);
         // Check if the intensity ratio is defined and different from zero
         if(tempValue==tempValue &&
               tempValue!=std::numeric_limits<double>::infinity() &&
               tempValue>0)
         {
            tempValue = tempRefValue * (tempValue>1?1.:-1.) / tempWarValue;

            // Jacobian modulation if the Jacobian determinant image is defined
            if(jacobianDetImg!=NULL)
               tempValue *= jacPtr[voxel];

            // Ensure that gradient of the warpedImage image along x-axis is not NaN
            tempGradX=currentGradPtrX[voxel];
            if(tempGradX==tempGradX)
               // Update the gradient along the x-axis
               measureGradPtrX[voxel] -= (DTYPE)(tempValue * tempGradX);

            // Ensure that gradient of the warpedImage image along y-axis is not NaN
            tempGradY=currentGradPtrY[voxel];
            if(tempGradY==tempGradY)
               // Update the gradient along the y-axis
               measureGradPtrY[voxel] -= (DTYPE)(tempValue * tempGradY);

            // Check if the current images are 3D
            if(referenceImage->nz>1)
            {
               // Ensure that gradient of the warpedImage image along z-axis is not NaN
               tempGradZ=currentGradPtrZ[voxel];
               if(tempGradZ==tempGradZ)
                  // Update the gradient along the z-axis
                  measureGradPtrZ[voxel] -= (DTYPE)(tempValue * tempGradZ);
            }
         }
      }
   }
   if(MrClean==true) free(maskPtr);
}
template void reg_getKLDivergenceVoxelBasedGradient<float>
(nifti_image *,nifti_image *,nifti_image *,nifti_image *,nifti_image *, int *, int);
template void reg_getKLDivergenceVoxelBasedGradient<double>
(nifti_image *,nifti_image *,nifti_image *,nifti_image *,nifti_image *, int *, int);
/* *************************************************************** */
void reg_kld::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint)
{
   // Check if the specified time point exists and is active
   reg_measure::GetVoxelBasedSimilarityMeasureGradient(current_timepoint);
   if(this->activeTimePoint[current_timepoint]==false)
      return;

   // Check if all required input images are of the same data type
   int dtype = this->referenceImagePointer->datatype;
   if(this->warpedFloatingImagePointer->datatype != dtype ||
         this->warpedFloatingGradientImagePointer->datatype != dtype ||
         this->forwardVoxelBasedGradientImagePointer->datatype != dtype
         )
   {
      reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
      reg_print_msg_error("Input images are exepected to be of the same type");
      reg_exit();
   }
   // Compute the gradient of the kld for the forward transformation
   switch(dtype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getKLDivergenceVoxelBasedGradient<float>
            (this->referenceImagePointer,
             this->warpedFloatingImagePointer,
             this->warpedFloatingGradientImagePointer,
             this->forwardVoxelBasedGradientImagePointer,
             NULL, // HERE TODO this->forwardJacDetImagePointer,
             this->referenceMaskPointer,
             current_timepoint
             );
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getKLDivergenceVoxelBasedGradient<double>
            (this->referenceImagePointer,
             this->warpedFloatingImagePointer,
             this->warpedFloatingGradientImagePointer,
             this->forwardVoxelBasedGradientImagePointer,
             NULL, // HERE TODO this->forwardJacDetImagePointer,
             this->referenceMaskPointer,
             current_timepoint
             );
      break;
   default:
      reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
   // Compute the gradient of the kld for the backward transformation
   if(this->isSymmetric)
   {
      dtype = this->floatingImagePointer->datatype;
      if(this->warpedReferenceImagePointer->datatype != dtype ||
            this->warpedReferenceGradientImagePointer->datatype != dtype ||
            this->backwardVoxelBasedGradientImagePointer->datatype != dtype
            )
      {
         reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("Input images are exepected to be of the same type");
         reg_exit();
      }
      // Compute the gradient of the nmi for the backward transformation
      switch(dtype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getKLDivergenceVoxelBasedGradient<float>
               (this->floatingImagePointer,
                this->warpedReferenceImagePointer,
                this->warpedReferenceGradientImagePointer,
                this->backwardVoxelBasedGradientImagePointer,
                NULL, // HERE TODO this->backwardJacDetImagePointer,
                this->floatingMaskPointer,
                current_timepoint
                );
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getKLDivergenceVoxelBasedGradient<double>
               (this->floatingImagePointer,
                this->warpedReferenceImagePointer,
                this->warpedReferenceGradientImagePointer,
                this->backwardVoxelBasedGradientImagePointer,
                NULL, // HERE TODO this->backwardJacDetImagePointer,
                this->floatingMaskPointer,
                current_timepoint
                );
         break;
      default:
         reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
