/*
 *  _reg_dti.cpp
 *
 *
 *  Created by Ivor Simpson on 22/10/2013.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_dti.h"

/* *************************************************************** */
/* *************************************************************** */
reg_dti::reg_dti()
   : reg_measure()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_dti constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
// This function is directly the same as that used for reg_ssd
void reg_dti::InitialiseMeasure(nifti_image *refImgPtr,
                                nifti_image *floImgPtr,
                                int *maskRefPtr,
                                nifti_image *warFloImgPtr,
                                nifti_image *warFloGraPtr,
                                nifti_image *forVoxBasedGraPtr,
                                nifti_image *forwardLocalWeightPtr,
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
                                  forwardLocalWeightPtr,
                                  maskFloPtr,
                                  warRefImgPtr,
                                  warRefGraPtr,
                                  bckVoxBasedGraPtr);

   // Check that the input images have the same number of time point
   if(this->referenceImagePointer->nt != this->floatingImagePointer->nt)
   {
      reg_print_fct_error("reg_dti::InitialiseMeasure");
      reg_print_msg_error("This number of time point should be the same for both input images");
      reg_exit();
   }

   int j=0;
   for(int i=0; i<refImgPtr->nt; ++i)
   {
      //JM - note, the specific value of timePointWeight is not used for DTI images
      //any value > 0 indicates the 'time point' is active
      if(this->timePointWeight[i]>0)
      {
         this->dtIndicies[j++]=i;
#ifndef NDEBUG
         reg_print_msg_debug("reg_dti::InitialiseMeasure().");
         char text[255];
         sprintf(text, "Active time point: %i", i);
         reg_print_msg_debug(text);
#endif
      }
   }
   if((refImgPtr->nz>1 && j!=6) && (refImgPtr->nz==1 && j!=3))
   {
      reg_print_fct_error("reg_dti::InitialiseMeasure");
      reg_print_msg_error("Unexpected number of DTI components");
      reg_exit();
   }
}
/* *************************************************************** */
template<class DTYPE>
double reg_getDTIMeasureValue(nifti_image *referenceImage,
                              nifti_image *warpedImage,
                              int *mask,
                              unsigned int * dtIndicies
                             )
{
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*
                        referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*
                        referenceImage->ny*referenceImage->nz;
#endif

   /* As the tensor has 6 unique components that we need to worry about, read them out
   for the floating and reference images. */
   DTYPE *firstWarpedVox = static_cast<DTYPE *>(warpedImage->data);
   DTYPE *warpedIntensityXX = &firstWarpedVox[voxelNumber*dtIndicies[0]];
   DTYPE *warpedIntensityXY = &firstWarpedVox[voxelNumber*dtIndicies[1]];
   DTYPE *warpedIntensityYY = &firstWarpedVox[voxelNumber*dtIndicies[2]];
   DTYPE *warpedIntensityXZ = &firstWarpedVox[voxelNumber*dtIndicies[3]];
   DTYPE *warpedIntensityYZ = &firstWarpedVox[voxelNumber*dtIndicies[4]];
   DTYPE *warpedIntensityZZ = &firstWarpedVox[voxelNumber*dtIndicies[5]];

   DTYPE *firstRefVox = static_cast<DTYPE *>(referenceImage->data);
   DTYPE *referenceIntensityXX = &firstRefVox[voxelNumber*dtIndicies[0]];
   DTYPE *referenceIntensityXY = &firstRefVox[voxelNumber*dtIndicies[1]];
   DTYPE *referenceIntensityYY = &firstRefVox[voxelNumber*dtIndicies[2]];
   DTYPE *referenceIntensityXZ = &firstRefVox[voxelNumber*dtIndicies[3]];
   DTYPE *referenceIntensityYZ = &firstRefVox[voxelNumber*dtIndicies[4]];
   DTYPE *referenceIntensityZZ = &firstRefVox[voxelNumber*dtIndicies[5]];

   double DTI_cost=0.0, n=0.0;
   const double twoThirds = (2.0/3.0);
   DTYPE rXX, rXY, rYY, rXZ, rYZ, rZZ;
#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(referenceImage, referenceIntensityXX, referenceIntensityXY, referenceIntensityXZ, \
          referenceIntensityYY, referenceIntensityYZ, referenceIntensityZZ, \
          warpedIntensityXX,warpedIntensityXY,warpedIntensityXZ, \
          warpedIntensityYY,warpedIntensityYZ, warpedIntensityZZ, mask,voxelNumber) \
   private(voxel, rXX, rXY, rYY, rXZ, rYZ, rZZ) \
reduction(+:DTI_cost) \
reduction(+:n)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // Check if the current voxel belongs to the mask and the intensities are not nans
      if(mask[voxel]>-1 )
      {
         if(referenceIntensityXX[voxel]==referenceIntensityXX[voxel] &&
               warpedIntensityXX[voxel]==warpedIntensityXX[voxel])
         {
            // Calculate the elementwise residual of the diffusion tensor components
            rXX = referenceIntensityXX[voxel] - warpedIntensityXX[voxel];
            rXY = referenceIntensityXY[voxel] - warpedIntensityXY[voxel];
            rYY = referenceIntensityYY[voxel] - warpedIntensityYY[voxel];
            rXZ = referenceIntensityXZ[voxel] - warpedIntensityXZ[voxel];
            rYZ = referenceIntensityYZ[voxel] - warpedIntensityYZ[voxel];
            rZZ = referenceIntensityZZ[voxel] - warpedIntensityZZ[voxel];
            DTI_cost -= twoThirds * (reg_pow2(rXX) + reg_pow2(rYY) + reg_pow2(rZZ))
                        + 2.0 * (reg_pow2(rXY) + reg_pow2(rXZ) + reg_pow2(rYZ))
                        - twoThirds * (rXX*rYY+rXX*rZZ+rYY*rZZ);
            n++;
         } // check if values are defined
      } // check if voxel belongs mask
   } // loop over voxels
   return DTI_cost/n;
}
template double reg_getDTIMeasureValue<float>(nifti_image *,nifti_image *,int *, unsigned int *);
template double reg_getDTIMeasureValue<double>(nifti_image *,nifti_image *,int *, unsigned int *);
/* *************************************************************** */
double reg_dti::GetSimilarityMeasureValue()
{
   // Check that all the specified image are of the same datatype
   if(this->warpedFloatingImagePointer->datatype != this->referenceImagePointer->datatype)
   {
      reg_print_fct_error("reg_dti::GetSimilarityMeasureValue");
      reg_print_msg_error("Both input images are exepected to have the same type");
      reg_exit();
   }
   double DTIMeasureValue;
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      DTIMeasureValue = reg_getDTIMeasureValue<float>
                        (this->referenceImagePointer,
                         this->warpedFloatingImagePointer,
                         this->referenceMaskPointer,
                         this->dtIndicies
                        );
      break;
   case NIFTI_TYPE_FLOAT64:
      DTIMeasureValue = reg_getDTIMeasureValue<double>
                        (this->referenceImagePointer,
                         this->warpedFloatingImagePointer,
                         this->referenceMaskPointer,
                         this->dtIndicies
                        );
      break;
   default:
      reg_print_fct_error("reg_dti::GetSimilarityMeasureValue");
      reg_print_msg_error("Result pixel type unsupported in the DTI computation function");
      reg_exit();
   }

   // Backward computation
   if(this->isSymmetric)
   {
      // Check that all the specified image are of the same datatype
      if(this->warpedReferenceImagePointer->datatype != this->floatingImagePointer->datatype)
      {
         reg_print_fct_error("reg_dti::GetSimilarityMeasureValue");
         reg_print_msg_error("Both input images are exepected to have the same type");
         reg_exit();
      }
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         DTIMeasureValue += reg_getDTIMeasureValue<float>
                            (this->floatingImagePointer,
                             this->warpedReferenceImagePointer,
                             this->floatingMaskPointer,
                             this->dtIndicies
                            );
         break;
      case NIFTI_TYPE_FLOAT64:
         DTIMeasureValue += reg_getDTIMeasureValue<double>
                            (this->floatingImagePointer,
                             this->warpedReferenceImagePointer,
                             this->floatingMaskPointer,
                             this->dtIndicies
                            );
         break;
      default:
         reg_print_fct_error("reg_dti::GetSimilarityMeasureValue");
         reg_print_msg_error("Warped pixel type unsupported in the DTI computation function");
         reg_exit();
      }
   }
   return DTIMeasureValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedDTIMeasureGradient(nifti_image *referenceImage,
      nifti_image *warpedImage,
      nifti_image *warImgGradient,
      nifti_image *dtiMeasureGradientImage,
      int *mask,
      unsigned int * dtIndicies)
{
   // Create pointers to the reference and warped images
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif

   /* As the tensor has 6 unique components that we need to worry about, read them out
   for the floating and reference images. */
   DTYPE *firstWarpedVox = static_cast<DTYPE *>(warpedImage->data);
   DTYPE *warpedIntensityXX = &firstWarpedVox[voxelNumber*dtIndicies[0]];
   DTYPE *warpedIntensityXY = &firstWarpedVox[voxelNumber*dtIndicies[1]];
   DTYPE *warpedIntensityYY = &firstWarpedVox[voxelNumber*dtIndicies[2]];
   DTYPE *warpedIntensityXZ = &firstWarpedVox[voxelNumber*dtIndicies[3]];
   DTYPE *warpedIntensityYZ = &firstWarpedVox[voxelNumber*dtIndicies[4]];
   DTYPE *warpedIntensityZZ = &firstWarpedVox[voxelNumber*dtIndicies[5]];

   DTYPE *firstRefVox = static_cast<DTYPE *>(referenceImage->data);
   DTYPE *referenceIntensityXX = &firstRefVox[voxelNumber*dtIndicies[0]];
   DTYPE *referenceIntensityXY = &firstRefVox[voxelNumber*dtIndicies[1]];
   DTYPE *referenceIntensityYY = &firstRefVox[voxelNumber*dtIndicies[2]];
   DTYPE *referenceIntensityXZ = &firstRefVox[voxelNumber*dtIndicies[3]];
   DTYPE *referenceIntensityYZ = &firstRefVox[voxelNumber*dtIndicies[4]];
   DTYPE *referenceIntensityZZ = &firstRefVox[voxelNumber*dtIndicies[5]];

   // THE FOLLOWING IS WRONG
   reg_print_msg_error("ERROR IN THE DTI GRADIENT COMPUTATION - TO FIX");
   reg_exit();
   unsigned int gradientVoxels = warImgGradient->nu*voxelNumber;
   DTYPE *firstGradVox = static_cast<DTYPE *>(warImgGradient->data);
   DTYPE *spatialGradXX = &firstGradVox[gradientVoxels*dtIndicies[0]];
   DTYPE *spatialGradXY = &firstGradVox[gradientVoxels*dtIndicies[1]];
   DTYPE *spatialGradYY = &firstGradVox[gradientVoxels*dtIndicies[2]];
   DTYPE *spatialGradXZ = &firstGradVox[gradientVoxels*dtIndicies[3]];
   DTYPE *spatialGradYZ = &firstGradVox[gradientVoxels*dtIndicies[4]];
   DTYPE *spatialGradZZ = &firstGradVox[gradientVoxels*dtIndicies[5]];

   // Create an array to store the computed gradient per time point
   DTYPE *dtiMeasureGradPtrX=static_cast<DTYPE *>(dtiMeasureGradientImage->data);
   DTYPE *dtiMeasureGradPtrY = &dtiMeasureGradPtrX[voxelNumber];
   DTYPE *dtiMeasureGradPtrZ = &dtiMeasureGradPtrY[voxelNumber];

   const double twoThirds = 2.0/3.0;
   const double fourThirds = 4.0/3.0;

   DTYPE rXX, rXY, rYY, rXZ, rYZ, rZZ, xxGrad, yyGrad, zzGrad, xyGrad, xzGrad, yzGrad;
#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(referenceIntensityXX, referenceIntensityXY, referenceIntensityXZ, \
          referenceIntensityYY, referenceIntensityYZ, referenceIntensityZZ,warpedIntensityXX, \
          warpedIntensityXY,warpedIntensityXZ ,warpedIntensityYY,warpedIntensityYZ, warpedIntensityZZ, \
          mask, spatialGradXX, spatialGradXY, spatialGradXZ, spatialGradYY, spatialGradYZ, spatialGradZZ, \
          dtiMeasureGradPtrX, dtiMeasureGradPtrY, dtiMeasureGradPtrZ, voxelNumber) \
   private(voxel, rXX, rXY, rYY, rXZ, rYZ, rZZ, xxGrad, yyGrad, zzGrad, xyGrad, xzGrad, yzGrad)
#endif
   for(voxel=0; voxel<voxelNumber; voxel++)
   {
      if(mask[voxel]>-1 )
      {
         if(referenceIntensityXX[voxel]==referenceIntensityXX[voxel] &&
               warpedIntensityXX[voxel]==warpedIntensityXX[voxel])
         {
            rXX = referenceIntensityXX[voxel] - warpedIntensityXX[voxel];
            rXY = referenceIntensityXY[voxel] - warpedIntensityXY[voxel];
            rYY = referenceIntensityYY[voxel] - warpedIntensityYY[voxel];
            rXZ = referenceIntensityXZ[voxel] - warpedIntensityXZ[voxel];
            rYZ = referenceIntensityYZ[voxel] - warpedIntensityYZ[voxel];
            rZZ = referenceIntensityZZ[voxel] - warpedIntensityZZ[voxel];

            xxGrad = fourThirds*rXX-twoThirds*(rYY+rZZ);
            yyGrad = fourThirds*rYY-twoThirds*(rXX+rZZ);
            zzGrad = fourThirds*rZZ-twoThirds*(rYY+rXX);
            xyGrad = 4.0*rXY;
            xzGrad = 4.0*rXZ;
            yzGrad = 4.0*rYZ;

            dtiMeasureGradPtrX[voxel] -= (spatialGradXX[voxel]*xxGrad+spatialGradYY[voxel]*yyGrad+spatialGradZZ[voxel]*zzGrad \
                                          + spatialGradXY[voxel]*xyGrad + spatialGradXZ[voxel]*xzGrad + spatialGradYZ[voxel]*yzGrad);

            dtiMeasureGradPtrY[voxel] -= (spatialGradXX[voxel+voxelNumber]*xxGrad+spatialGradYY[voxel+voxelNumber]*yyGrad+spatialGradZZ[voxel+voxelNumber]*zzGrad \
                                          + spatialGradXY[voxel+voxelNumber]*xyGrad + spatialGradXZ[voxel+voxelNumber]*xzGrad + spatialGradYZ[voxel+voxelNumber]*yzGrad);

            dtiMeasureGradPtrZ[voxel] -= (spatialGradXX[voxel+2*voxelNumber]*xxGrad+spatialGradYY[voxel+2*voxelNumber]*yyGrad \
                                          + spatialGradZZ[voxel+2*voxelNumber]*zzGrad + spatialGradXY[voxel+2*voxelNumber]*xyGrad  \
                                          + spatialGradXZ[voxel+2*voxelNumber]*xzGrad + spatialGradYZ[voxel+2*voxelNumber]*yzGrad);
         }
      }
   }
}
/* *************************************************************** */
template void reg_getVoxelBasedDTIMeasureGradient<float>
(nifti_image *,nifti_image *,nifti_image *,nifti_image *, int *, unsigned int *);
template void reg_getVoxelBasedDTIMeasureGradient<double>
(nifti_image *,nifti_image *,nifti_image *,nifti_image *, int *, unsigned int *);
/* *************************************************************** */
void reg_dti::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint)
{
   // Check if the specified time point exists and is active
   reg_measure::GetVoxelBasedSimilarityMeasureGradient(current_timepoint);
   if(this->timePointWeight[current_timepoint]==0.0)
      return;

   // Check if all required input images are of the same data type
   int dtype = this->referenceImagePointer->datatype;
   if(this->warpedFloatingImagePointer->datatype != dtype ||
         this->warpedFloatingGradientImagePointer->datatype != dtype ||
         this->forwardVoxelBasedGradientImagePointer->datatype != dtype
     )
   {
      reg_print_fct_error("reg_dti::GetVoxelBasedSimilarityMeasureGradient");
      reg_print_msg_error("Input images are exepected to be of the same type");
      reg_exit();
   }
   // Compute the gradient of the ssd for the forward transformation
   switch(dtype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getVoxelBasedDTIMeasureGradient<float>
      (this->referenceImagePointer,
       this->warpedFloatingImagePointer,
       this->warpedFloatingGradientImagePointer,
       this->forwardVoxelBasedGradientImagePointer,
       this->referenceMaskPointer,
       this->dtIndicies
      );
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getVoxelBasedDTIMeasureGradient<double>
      (this->referenceImagePointer,
       this->warpedFloatingImagePointer,
       this->warpedFloatingGradientImagePointer,
       this->forwardVoxelBasedGradientImagePointer,
       this->referenceMaskPointer,
       this->dtIndicies
      );
      break;
   default:
      reg_print_fct_error("reg_dti::GetVoxelBasedSimilarityMeasureGradient");
      reg_print_msg_error("The input image data type is not supported");
      reg_exit();
   }
   // Compute the gradient of the ssd for the backward transformation
   if(this->isSymmetric)
   {
      dtype = this->floatingImagePointer->datatype;
      if(this->warpedReferenceImagePointer->datatype != dtype ||
            this->warpedReferenceGradientImagePointer->datatype != dtype ||
            this->backwardVoxelBasedGradientImagePointer->datatype != dtype
        )
      {
         reg_print_fct_error("reg_dti::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("Input images are exepected to be of the same type");
         reg_exit();
      }
      // Compute the gradient of the nmi for the backward transformation
      switch(dtype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedDTIMeasureGradient<float>
         (this->floatingImagePointer,
          this->warpedReferenceImagePointer,
          this->warpedReferenceGradientImagePointer,
          this->backwardVoxelBasedGradientImagePointer,
          this->floatingMaskPointer,
          this->dtIndicies
         );
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedDTIMeasureGradient<double>
         (this->floatingImagePointer,
          this->warpedReferenceImagePointer,
          this->warpedReferenceGradientImagePointer,
          this->backwardVoxelBasedGradientImagePointer,
          this->floatingMaskPointer,
          this->dtIndicies
         );
         break;
      default:
         reg_print_fct_error("reg_dti::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("The input image data type is not supported");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
