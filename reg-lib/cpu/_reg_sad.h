#ifndef _REG_SAD_H
#define _REG_SAD_H

#include "_reg_measure.h"
#include <cmath>

/* *************************************************************** */
/* *************************************************************** */
/// @brief SAD measure of similarity classe
class reg_sad : public reg_measure
{
public:
   /// @brief reg_sad class constructor
   reg_sad();
   /// @brief Initialise the reg_sad object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          int *maskFloPtr = NULL,
                          nifti_image *warRefImgPtr = NULL,
                          nifti_image *warRefGraPtr = NULL,
                          nifti_image *bckVoxBasedGraPtr = NULL);
   /// @brief Returns the sad value
   virtual double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based sad gradient
   virtual void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief Here
   virtual void GetDiscretisedValue(nifti_image *controlPointGridImage,
                                    float *discretisedValue,
                                    int discretise_radius,
                                    int discretise_step);
   /// @brief Measure class desstructor
   ~reg_sad() {}
protected:
   float currentValue[255];
};
/* *************************************************************** */

/** @brief Copmutes and returns the SAD between two input images
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param activeTimePoint Specified which time point volumes have to be considered
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the SAD. The argument is ignored if the
 * pointer is set to NULL
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 * @return Returns the computed sum squared difference
 */
extern "C++" template <class DTYPE>
double reg_getSADValue(nifti_image *referenceImage,
                       nifti_image *warpedImage,
                       bool *activeTimePoint,
                       nifti_image *jacobianDeterminantImage,
                       int *mask,
                       float *currentValue
                      );

/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param activeTimePoint Specified which time point volumes have to be considered
 * @param warpedImageGradient Spatial gradient of the input warped image
 * @param sadGradientImage Output image htat will be updated with the
 * value of the SAD gradient
 * @param jacobianDeterminantImage Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the SAD. The argument is ignored if the
 * pointer is set to NULL
 * @param maxSD Input scalar that contain the difference value between
 * the highest and the lowest intensity.
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 */
extern "C++" template <class DTYPE>
void reg_getVoxelBasedSADGradient(nifti_image *referenceImage,
                                  nifti_image *warpedImage,
                                  bool *activeTimePoint,
                                  nifti_image *warpedImageGradient,
                                  nifti_image *sadGradientImage,
                                  nifti_image *jacobianDeterminantImage,
                                  int *mask
                                 );

#endif // _REG_SAD_H
