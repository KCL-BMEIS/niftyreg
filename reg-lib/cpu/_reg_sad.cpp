#include "_reg_sad.h"

//#define USE_LOG_SAD

/* *************************************************************** */
/* *************************************************************** */
reg_sad::reg_sad()
   : reg_measure()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_sad constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_sad::InitialiseMeasure(nifti_image *refImgPtr,
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
      reg_print_fct_error("reg_sad::InitialiseMeasure");
      reg_print_msg_error("This number of time point should be the same for both input images");
      reg_exit();
   }
   // Input images are normalised between 0 and 1
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
#ifndef NDEBUG
   char text[255];
   reg_print_msg_debug("reg_sad::InitialiseMeasure().");
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
double reg_getSADValue(nifti_image *referenceImage,
                       nifti_image *warpedImage,
                       bool *activeTimePoint,
                       nifti_image *jacobianDetImage,
                       int *mask,
                       float *currentValue
                      )
{

#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif
   // Create pointers to the reference and warped image data
   DTYPE *referencePtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warpedPtr=static_cast<DTYPE *>(warpedImage->data);
   // Create a pointer to the Jacobian determinant image if defined
   DTYPE *jacDetPtr=NULL;
   if(jacobianDetImage!=NULL)
      jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);


   double SAD_global=0.0, n=0.0;
   double refValue, warValue, diff;

   // Loop over the different time points
   for(int time=0; time<referenceImage->nt; ++time)
   {
      if(activeTimePoint[time])
      {
         // Create pointers to the current time point of the reference and warped images
         DTYPE *currentRefPtr=&referencePtr[time*voxelNumber];
         DTYPE *currentWarPtr=&warpedPtr[time*voxelNumber];

         double SAD_local=0.;
         n=0.;
#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(referenceImage, currentRefPtr, currentWarPtr, mask, \
                jacobianDetImage, jacDetPtr, voxelNumber) \
         private(voxel, refValue, warValue, diff) \
reduction(+:SAD_local) \
reduction(+:n)
#endif
         for(voxel=0; voxel<voxelNumber; ++voxel)
         {
            // Check if the current voxel belongs to the mask
            if(mask[voxel]>-1)
            {
               // Ensure that both ref and warped values are defined
               refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                      referenceImage->scl_inter);
               warValue = (double)(currentWarPtr[voxel] * referenceImage->scl_slope +
                                      referenceImage->scl_inter);
               if(refValue==refValue && warValue==warValue)
               {
                  diff = std::abs(refValue-warValue);
//						if(diff>0) diff=log(diff);
                  // Jacobian determinant modulation of the sad if required
                  if(jacDetPtr!=NULL)
                  {
                     SAD_local += diff * jacDetPtr[voxel];
                     n += jacDetPtr[voxel];
                  }
                  else
                  {
                     SAD_local += diff;
                     n += 1.0;
                  }
               }
            }
         }
         currentValue[time]=-SAD_local;
         SAD_global -= SAD_local/n;
      }
   }
   return SAD_global;
}
template double reg_getSADValue<float>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
template double reg_getSADValue<double>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
/* *************************************************************** */
double reg_sad::GetSimilarityMeasureValue()
{
   // Check that all the specified image are of the same datatype
   if(this->warpedFloatingImagePointer->datatype != this->referenceImagePointer->datatype)
   {
      reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
      reg_print_msg_error("Both input images are exepected to have the same type");
      reg_exit();
   }
   double SADValue;
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      SADValue = reg_getSADValue<float>
                 (this->referenceImagePointer,
                  this->warpedFloatingImagePointer,
                  this->activeTimePoint,
                  NULL, // HERE TODO this->forwardJacDetImagePointer,
                  this->referenceMaskPointer,
                  this->currentValue
                 );
      break;
   case NIFTI_TYPE_FLOAT64:
      SADValue = reg_getSADValue<double>
                 (this->referenceImagePointer,
                  this->warpedFloatingImagePointer,
                  this->activeTimePoint,
                  NULL, // HERE TODO this->forwardJacDetImagePointer,
                  this->referenceMaskPointer,
                  this->currentValue
                 );
      break;
   default:
      reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
      reg_print_msg_error("Warped pixel type unsupported");
      reg_exit();
   }

   // Backward computation
   if(this->isSymmetric)
   {
      // Check that all the specified image are of the same datatype
      if(this->warpedReferenceImagePointer->datatype != this->floatingImagePointer->datatype)
      {
         reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
         reg_print_msg_error("Both input images are exepected to have the same type");
         reg_exit();
      }
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         SADValue += reg_getSADValue<float>
                     (this->floatingImagePointer,
                      this->warpedReferenceImagePointer,
                      this->activeTimePoint,
                      NULL, // HERE TODO this->backwardJacDetImagePointer,
                      this->floatingMaskPointer,
                      this->currentValue
                     );
         break;
      case NIFTI_TYPE_FLOAT64:
         SADValue += reg_getSADValue<double>
                     (this->floatingImagePointer,
                      this->warpedReferenceImagePointer,
                      this->activeTimePoint,
                      NULL, // HERE TODO this->backwardJacDetImagePointer,
                      this->floatingMaskPointer,
                      this->currentValue
                     );
         break;
      default:
         reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
         reg_print_msg_error("Warped pixel type unsupported");
         reg_exit();
      }
   }
   return SADValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedSADGradient(nifti_image *referenceImage,
                                  nifti_image *warpedImage,
                                  bool *activeTimePoint,
                                  nifti_image *warImgGradient,
                                  nifti_image *sadGradientImage,
                                  nifti_image *jacobianDetImage,
                                  int *mask)
{
   // Create pointers to the reference and warped images
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif

   DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);

   // Pointer to the warped image gradient
   DTYPE *spatialGradPtr=static_cast<DTYPE *>(warImgGradient->data);

   // Create a pointer to the Jacobian determinant values if defined
   DTYPE *jacDetPtr=NULL;
   if(jacobianDetImage!=NULL)
      jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

   // Create an array to store the computed gradient per time point
   DTYPE *sadGradPtrX=static_cast<DTYPE *>(sadGradientImage->data);
   DTYPE *sadGradPtrY = &sadGradPtrX[voxelNumber];
   DTYPE *sadGradPtrZ = NULL;
   if(referenceImage->nz>1)
      sadGradPtrZ = &sadGradPtrY[voxelNumber];

   double refValue, warValue, common;
   // Loop over the different time points
   for(int time=0; time<referenceImage->nt; ++time)
   {
      if(activeTimePoint[time])
      {
         // Create some pointers to the current time point image to be accessed
         DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
         DTYPE *currentWarPtr=&warPtr[time*voxelNumber];

         // Create some pointers to the spatial gradient of the current warped volume
         DTYPE *spatialGradPtrX = &spatialGradPtr[time*voxelNumber];
         DTYPE *spatialGradPtrY = &spatialGradPtr[(referenceImage->nt+time)*voxelNumber];
         DTYPE *spatialGradPtrZ = NULL;
         if(referenceImage->nz>1)
            spatialGradPtrZ=&spatialGradPtr[(2*referenceImage->nt+time)*voxelNumber];


#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, time, \
                mask, jacDetPtr, spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, \
                sadGradPtrX, sadGradPtrY, sadGradPtrZ, voxelNumber) \
         private(voxel, refValue, warValue, common)
#endif
         for(voxel=0; voxel<voxelNumber; voxel++)
         {
            if(mask[voxel]>-1)
            {
               refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                      referenceImage->scl_inter);
               warValue = (double)(currentWarPtr[voxel] * warpedImage->scl_slope +
                                      warpedImage->scl_inter);
               if(refValue==refValue && warValue==warValue)
               {
                  common = -1.0 * std::abs(refValue - warValue);
                  if(jacDetPtr!=NULL)
                     common *= jacDetPtr[voxel];

                  if(spatialGradPtrX[voxel]==spatialGradPtrX[voxel])
                     sadGradPtrX[voxel] += (DTYPE)(common * spatialGradPtrX[voxel]);
                  if(spatialGradPtrY[voxel]==spatialGradPtrY[voxel])
                     sadGradPtrY[voxel] += (DTYPE)(common * spatialGradPtrY[voxel]);

                  if(sadGradPtrZ!=NULL)
                  {
                     if(spatialGradPtrZ[voxel]==spatialGradPtrZ[voxel])
                        sadGradPtrZ[voxel] += (DTYPE)(common * spatialGradPtrZ[voxel]);
                  }
               }
            }
         }
      }
   }// loop over time points
}
/* *************************************************************** */
template void reg_getVoxelBasedSADGradient<float>
(nifti_image *,nifti_image *,bool *,nifti_image *,nifti_image *,nifti_image *, int *);
template void reg_getVoxelBasedSADGradient<double>
(nifti_image *,nifti_image *,bool *,nifti_image *,nifti_image *,nifti_image *, int *);
/* *************************************************************** */
void reg_sad::GetVoxelBasedSimilarityMeasureGradient()
{
   // Check if all required input images are of the same data type
   int dtype = this->referenceImagePointer->datatype;
   if(this->warpedFloatingImagePointer->datatype != dtype ||
         this->warpedFloatingGradientImagePointer->datatype != dtype ||
         this->forwardVoxelBasedGradientImagePointer->datatype != dtype
     )
   {
      reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
      reg_print_msg_error("Input images are exepected to be of the same type");
      reg_exit();
   }
   // Compute the gradient of the sad for the forward transformation
   switch(dtype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getVoxelBasedSADGradient<float>
      (this->referenceImagePointer,
       this->warpedFloatingImagePointer,
       this->activeTimePoint,
       this->warpedFloatingGradientImagePointer,
       this->forwardVoxelBasedGradientImagePointer,
       NULL, // HERE TODO this->forwardJacDetImagePointer,
       this->referenceMaskPointer
      );
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getVoxelBasedSADGradient<double>
      (this->referenceImagePointer,
       this->warpedFloatingImagePointer,
       this->activeTimePoint,
       this->warpedFloatingGradientImagePointer,
       this->forwardVoxelBasedGradientImagePointer,
       NULL, // HERE TODO this->forwardJacDetImagePointer,
       this->referenceMaskPointer
      );
      break;
   default:
      reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
   // Compute the gradient of the sad for the backward transformation
   if(this->isSymmetric)
   {
      dtype = this->floatingImagePointer->datatype;
      if(this->warpedReferenceImagePointer->datatype != dtype ||
            this->warpedReferenceGradientImagePointer->datatype != dtype ||
            this->backwardVoxelBasedGradientImagePointer->datatype != dtype
        )
      {
         reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("Input images are exepected to be of the same type");
         reg_exit();
      }
      // Compute the gradient of the nmi for the backward transformation
      switch(dtype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedSADGradient<float>
         (this->floatingImagePointer,
          this->warpedReferenceImagePointer,
          this->activeTimePoint,
          this->warpedReferenceGradientImagePointer,
          this->backwardVoxelBasedGradientImagePointer,
          NULL, // HERE TODO this->backwardJacDetImagePointer,
          this->floatingMaskPointer
         );
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedSADGradient<double>
         (this->floatingImagePointer,
          this->warpedReferenceImagePointer,
          this->activeTimePoint,
          this->warpedReferenceGradientImagePointer,
          this->backwardVoxelBasedGradientImagePointer,
          NULL, // HERE TODO this->backwardJacDetImagePointer,
          this->floatingMaskPointer
         );
         break;
      default:
         reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
