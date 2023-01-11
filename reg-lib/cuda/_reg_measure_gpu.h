/** @file _reg_measure_gpu.h
 * @author Marc Modat
 * @date 25/06/2013
 * @brief Contains a measure class to embbed all gpu measures of similarity classes
 * Also contains an interface class between reg_base and the measure class
 */

#pragma once

#include "_reg_lncc.h"
#include "_reg_dti.h"

#include "_reg_common_cuda.h"
#include "_reg_kld.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief Class that contains the GPU device pointers
class reg_measure_gpu
{
protected:
   /// @brief Measure class constructor
   reg_measure_gpu() {}
   /// @brief Measure class destructor
   virtual ~reg_measure_gpu() {}

   cudaArray *referenceDevicePointer;
   cudaArray *floatingDevicePointer;
   int *referenceMaskDevicePointer;
   int activeVoxeNumber;
   float *warpedFloatingDevicePointer;
   float4 *warpedFloatingGradientDevicePointer;
   float4 *forwardVoxelBasedGradientDevicePointer;
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
class reg_lncc_gpu : public reg_lncc , public reg_measure_gpu
{
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
                          float4 *forVoxBasedGraDevicePtr)
   {
      ;
   }
   /// @brief reg_lncc class constructor
   reg_lncc_gpu()
   {
      fprintf(stderr,"[ERROR] CUDA CANNOT BE USED WITH LNCC YET\n");
      reg_exit();
   }
   /// @brief reg_lncc class destructor
   virtual ~reg_lncc_gpu() {}
   /// @brief Returns the lncc value
   virtual double GetSimilarityMeasureValue() override { return 0; }
   /// @brief Compute the voxel based lncc gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
class reg_kld_gpu : public reg_kld , public reg_measure_gpu
{
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
                          float4 *forVoxBasedGraDevicePtr)
   {
      ;
   }
   /// @brief reg_kld_gpu class constructor
   reg_kld_gpu()
   {
      fprintf(stderr,"[ERROR] CUDA CANNOT BE USED WITH KLD YET\n");
      reg_exit();
   }
   /// @brief reg_kld_gpu class destructor
   virtual ~reg_kld_gpu() {}
   /// @brief Returns the kld value
   virtual double GetSimilarityMeasureValue() override { return 0; }
   /// @brief Compute the voxel based kld gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
class reg_dti_gpu : public reg_dti , public reg_measure_gpu
{
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
                          float4 *forVoxBasedGraDevicePtr)
   {
      ;
   }
   /// @brief reg_dti_gpu class constructor
   reg_dti_gpu()
   {
      fprintf(stderr,"[ERROR] CUDA CANNOT BE USED WITH DTI YET\n");
      reg_exit();
   }
   /// @brief reg_dti_gpu class destructor
   virtual ~reg_dti_gpu() {}
   /// @brief Returns the dti value
   virtual double GetSimilarityMeasureValue() override { return 0; }
   /// @brief Compute the voxel based dti gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient(int current_timepoint) override {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
