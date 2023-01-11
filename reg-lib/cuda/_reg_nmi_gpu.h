/*
 *  _reg_mutualinformation_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_nmi.h"
#include "_reg_measure_gpu.h"
#include "_reg_blocksize_gpu.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief NMI measure of similarity class - GPU based
class reg_nmi_gpu: public reg_nmi, public reg_measure_gpu {
public:
   /// @brief reg_nmi class constructor
   reg_nmi_gpu();
   /// @brief reg_nmi class destructor
   virtual ~reg_nmi_gpu();

   /// @brief Initialise the reg_nmi_gpu object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          int activeVoxNum,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          cudaArray *refDevicePtr,
                          cudaArray *floDevicePtr,
                          int *refMskDevicePtr,
                          float *warFloDevicePtr,
                          float4 *warFloGradDevicePtr,
                          float4 *forVoxBasedGraDevicePtr);
   /// @brief Returns the nmi value
   virtual double GetSimilarityMeasureValue() override;
   /// @brief Compute the voxel based nmi gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override;

protected:
   float *forwardJointHistogramLog_device;
	// float **backwardJointHistogramLog_device;
   void DeallocateHistogram();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief NMI measure of similarity class
class reg_multichannel_nmi_gpu: public reg_multichannel_nmi, public reg_measure_gpu {
public:
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          int activeVoxNum,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          cudaArray *refDevicePtr,
                          cudaArray *floDevicePtr,
                          int *refMskDevicePtr,
                          float *warFloDevicePtr,
                          float4 *warFloGradDevicePtr,
                          float4 *forVoxBasedGraDevicePtr) {}
   /// @brief reg_nmi class constructor
   reg_multichannel_nmi_gpu() {}
   /// @brief reg_nmi class destructor
   virtual ~reg_multichannel_nmi_gpu() {}
   /// @brief Returns the nmi value
   virtual double GetSimilarityMeasureValue() override { return 0; }
   /// @brief Compute the voxel based nmi gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
extern "C++"
void reg_getVoxelBasedNMIGradient_gpu(nifti_image *referenceImage,
                                      cudaArray *referenceImageArray_d,
                                      float *warpedImageArray_d,
                                      float4 *resultGradientArray_d,
                                      float *logJointHistogram_d,
                                      float4 *voxelNMIGradientArray_d,
                                      int *targetMask_d,
                                      int activeVoxelNumber,
                                      double *entropies,
                                      int refBinning,
                                      int floBinning);
