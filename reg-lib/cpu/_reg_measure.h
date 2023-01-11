/** @file _reg_measure.h
 * @author Marc Modat
 * @date 25/06/2013
 * @brief Contains a measure class to embbed all measures of similarity classes
 * Also contains an interface class between reg_base and the measure class
 */

#pragma once

#include "_reg_tools.h"
#include <time.h>

/// @brief Class common to all measure of similarity classes
class reg_measure {
public:
   /// @brief Set the pointers to be ussed by the measure object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          nifti_image *localWeightSimPtr = nullptr,
                          int *maskFloPtr = nullptr,
                          nifti_image *warRefImgPtr = nullptr,
                          nifti_image *warRefGraPtr = nullptr,
                          nifti_image *bckVoxBasedGraPtr = nullptr) {
      this->isSymmetric = false;
      this->referenceImagePointer = refImgPtr;
      this->referenceTimePoint = this->referenceImagePointer->nt;
      this->floatingImagePointer = floImgPtr;
      this->referenceMaskPointer = maskRefPtr;
      this->warpedFloatingImagePointer = warFloImgPtr;
      this->warpedFloatingGradientImagePointer = warFloGraPtr;
      this->forwardVoxelBasedGradientImagePointer = forVoxBasedGraPtr;
      this->forwardLocalWeightSimImagePointer = localWeightSimPtr;
      if (maskFloPtr != nullptr && warRefImgPtr != nullptr && warRefGraPtr != nullptr && bckVoxBasedGraPtr != nullptr) {
         this->isSymmetric = true;
         this->floatingMaskPointer = maskFloPtr;
         this->warpedReferenceImagePointer = warRefImgPtr;
         this->warpedReferenceGradientImagePointer = warRefGraPtr;
         this->backwardVoxelBasedGradientImagePointer = bckVoxBasedGraPtr;
      } else {
         this->floatingMaskPointer = nullptr;
         this->warpedReferenceImagePointer = nullptr;
         this->warpedReferenceGradientImagePointer = nullptr;
         this->backwardVoxelBasedGradientImagePointer = nullptr;
      }
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] reg_measure::InitialiseMeasure()\n");
#endif
   }

   /// @brief Returns the registration measure of similarity value
   virtual double GetSimilarityMeasureValue() = 0;

   /// @brief Compute the voxel based measure of similarity gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) {
      if (current_timepoint < 0 || current_timepoint >= this->referenceImagePointer->nt) {
         reg_print_fct_error("reg_measure::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("The specified active timepoint is not defined in the ref/war images");
         reg_exit();
      }
   }

   /// @brief Here
   virtual void GetDiscretisedValue(nifti_image *, float *, int, int) {}

   virtual void SetTimepointWeight(int timepoint, double weight) {
      this->timePointWeight[timepoint] = weight;
   }

   virtual double* GetTimepointsWeights(void) {
      return this->timePointWeight;
   }

   virtual nifti_image* GetReferenceImage(void) {
      return this->referenceImagePointer;
   }

   virtual int* GetReferenceMask(void) {
      return this->referenceMaskPointer;
   }

protected:
   nifti_image *referenceImagePointer;
   int *referenceMaskPointer;
   nifti_image *warpedFloatingImagePointer;
   nifti_image *warpedFloatingGradientImagePointer;
   nifti_image *forwardVoxelBasedGradientImagePointer;
   nifti_image *forwardLocalWeightSimImagePointer;

   bool isSymmetric;
   nifti_image *floatingImagePointer;
   int *floatingMaskPointer;
   nifti_image *warpedReferenceImagePointer;
   nifti_image *warpedReferenceGradientImagePointer;
   nifti_image *backwardVoxelBasedGradientImagePointer;

   double timePointWeight[255] = {0};
   int referenceTimePoint;

   /// @brief Measure class constructor
   reg_measure() {
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] reg_measure constructor called\n");
#endif
   }

   /// @brief Measure class destructor
   virtual ~reg_measure() {}
};
