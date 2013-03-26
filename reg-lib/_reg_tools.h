/**
 * @file _reg_tools.h
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_H
#define _REG_TOOLS_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>

#include "_reg_maths.h"


/** @brief This function check some header parameters and correct them in
 * case of error. For example no dimension is lower than one. The scl_sclope
 * can not be equal to zero. The qto_xyz and qto_ijk are populated if
 * both qform_code and sform_code are set to zero.
 * @param image Input image to check and correct if necessary
 */
extern "C++"
void reg_checkAndCorrectDimension(nifti_image *image);

/** @brief Rescale an input image between two user-defined values.
 * Some threshold can also be applied concurrenlty
 * @param image Image to be rescaled
 * @param newMin Intensity lower bound after rescaling
 * @param newMax Intensity higher bound after rescaling
 * @param lowThr Intensity to use as lower threshold
 * @param upThr Intensity to use as higher threshold
 */
extern "C++"
void reg_intensityRescale(nifti_image *image,
                          float *newMin,
                          float *newMax,
                          float *lowThr,
                          float *upThr
                          );


/** @brief reg_getRealImageSpacing
 * @param image image
 * @param spacingValues spacingValues
 */
extern "C++" template <class DTYPE>
void reg_getRealImageSpacing(nifti_image *image,
                             DTYPE *spacingValues);

/** @brief Convolve a cubic spline kernel with the provided image
 * @param image Image to be convolved with the kernel
 * @param radius Radius of the cubic spline kernel. The array
 * contains the radius along each axis
 */
void reg_tools_CubicSplineKernelConvolution(nifti_image *image,
                                            float spacingVoxel[3]
                                            );

/** @brief Smooth an image using a Gaussian kernel
 * @param image Image to be smoothed
 * @param sigma Standard deviation of the Gaussian kernel
 * to use. The kernel is bounded between +/- 3 sigma.
 * @param axis Boolean array to specify which axis have to be
 * smoothed. The array follow the dim array of the nifti header.
 */
extern "C++" template <class PrecisionTYPE>
void reg_gaussianSmoothing(nifti_image *image,
                           PrecisionTYPE sigma,
                           bool *axis
                           );
// void reg_gaussianSmoothing<float>(nifti_image *, float, bool[8]);
// void reg_gaussianSmoothing<double>(nifti_image *, double, bool[8]);

/** @brief Downsample an image by a ratio of two
 * @param image Image to be downsampled
 * @param type The image is first smoothed  using a Gaussian
 * kernel of 0.7 voxel standard deviation before being downsample
 * if type is set to true.
 * @param axis Boolean array to specify which axis have to be
 * downsampled. The array follow the dim array of the nifti header.
 */
extern "C++" template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image,
                         int type,
                         bool *axis
                         );

/** @brief Returns the maximal euclidean distance from a
 * deformation field image
 * @param image Vector image to be considered
 * @return Scalar value that corresponds to the longest
 * euclidean distance
 */
extern "C++" template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image);

/** @brief Change the datatype of a nifti image
 * @param image Image to be updated.
 */
extern "C++" template <class NewTYPE>
void reg_tools_changeDatatype(nifti_image *image);

/** @brief Perform some basic arithmetic operation between
 * two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 * @param type Type of operation to be performed between 0 and 3
 * with 0, 1, 2 and 3 corresponding to addition, substraction,
 * multiplication and division respectively.
 */
extern "C++"
void reg_tools_addSubMulDivImages(nifti_image *img1,
                                  nifti_image *img2,
                                  nifti_image *out,
                                  int type);

/** @brief Perform some basic arithmetic operation to an image
 * @param img1 Input image to consider
 * @param out Result image that contains the result of the operation.
 * @param val Value to be added/substracted/multiplied/divided to
 * input image
 * @param type Type of operation to be performed between 0 and 3
 * with 0, 1, 2 and 3 corresponding to addition, substraction,
 * multiplication and division respectively.
 */
extern "C++"
void reg_tools_addSubMulDivValue(nifti_image *img1,
                                 nifti_image *out,
                                 float val,
                                 int type);

/** @brief Binarise an input image. All values different
 * from 0 are set to 1, 0 otherwise.
 * @param img Image that will be binarise inline
 */
extern "C++"
void reg_tools_binarise_image(nifti_image *img);

/** @brief Binarise an input image. The binarisation is
 * performed according to a threshold value that is
 * user-defined.
 * @param img Image that will be binarise inline
 * @param thr Threshold value used for binarisation.
 * All values bellow thr are set to 0. All values equal
 * or bellow thr are set to 1
 */
extern "C++"
void reg_tools_binarise_image(nifti_image *img,
                              float thr);

/** @brief Convert a binary image into an array of int.
 * This is used to define a mask within the registration
 * function.
 * @param img Input image
 * @param array The data array from the input nifti image
 * is binarised and stored in this array.
 * @param activeVoxelNumber This reference is updated
 * with the number of voxel that are included into the
 * mask
 */
extern "C++"
void reg_tools_binaryImage2int(nifti_image *img,
                               int *array,
                               int &activeVoxelNumber);

/** @brief Compute the mean root mean squared error between
 * two vector images
 * @param imgA Input vector image
 * @param imgB Input vector image
 * @return Mean rsoot mean squared error valueis returned
 */
extern "C++"
double reg_tools_getMeanRMS(nifti_image *imgA,
                            nifti_image *imgB);

/** @brief Set all voxels from an image to NaN if the voxel
 * bellong to the mask
 * @param img Input image to be masked with NaN value
 * @param mask Input mask that defines which voxels
 * have to be set to NaN
 * @param res Output image
 */
extern "C++"
int reg_tools_nanMask_image(nifti_image *img,
                            nifti_image *mask,
                            nifti_image *res);

/** @brief Get the minimal value of an image
 * @param img Input image
 * @return min value
 */
extern "C++"
float reg_tools_getMinValue(nifti_image *img);

/** @brief Get the maximal value of an image
 * @param img Input image
 * @return max value
 */
extern "C++"
float reg_tools_getMaxValue(nifti_image *img);

/** @brief Generate a pyramid from an input image.
 * @param input Input image to be downsampled to create the pyramid
 * @param pyramid Output array of images that will contains the
 * different resolution images of the pyramid
 * @param levelNumber Number of level to use to create the pyramid.
 * 1 level corresponds to the original image resolution.
 * @param levelToPerform Number to level that will be perform during
 * the registration.
 */
extern "C++" template<class DTYPE>
int reg_createImagePyramid(nifti_image * input,
                           nifti_image **pyramid,
                           unsigned int levelNumber,
                           unsigned int levelToPerform);
/** @brief Generate a pyramid from an input mask image.
 * @param input Input image to be downsampled to create the pyramid
 * @param pyramid Output array of mask images that will contains the
 * different resolution images of the pyramid
 * @param levelNumber Number of level to use to create the pyramid.
 * 1 level corresponds to the original image resolution.
 * @param levelToPerform Number to level that will be perform during
 * the registration.
 * @param activeVoxelNumber Array that contains the number of active
 * voxel for each level of the pyramid
 */
extern "C++" template<class DTYPE>
int reg_createMaskPyramid(nifti_image *input,
                          int **pyramid,
                          unsigned int levelNumber,
                          unsigned int levelToPerform,
                          int *activeVoxelNumber);

/** @brief this function will threshold an image to the values provided,
 * set the scl_slope and sct_inter of the image to 1 and 0
 * (SSD uses actual image data values),
 * and sets cal_min and cal_max to have the min/max image data values.
 * @param image Input image to be thresholded.
 * @param lowThr Lower threshold value. All Value bellow the threshold
 * are set to the threshold value.
 * @param upThr Upper threshold value. All Value above the threshold
 * are set to the threshold value.
 */
extern "C++" template<class T>
void reg_thresholdImage(nifti_image *image,
                        T lowThr,
                        T upThr
                        );

/** @brief This function flipp the specified axis
 * @param image Input image to be flipped
 * @param array Array that will contain the flipped
 * input image->data array
 * @param cmd String that contains the letter(s) of the axis
 * to flip (xyztuvw)
 */
extern "C++"
void reg_flippAxis(nifti_image *image,
                   void *array,
                   std::string cmd
                   );

/* *************************************************************** */
/** @brief This function converts an image containing deformation
 * field into a displacement field
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
extern "C++"
int reg_getDisplacementFromDeformation(nifti_image *image);
/* *************************************************************** */
/** @brief This function converts an image containing a displacement field
 * into a displacement field.
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
extern "C++"
int reg_getDeformationFromDisplacement(nifti_image *image);
/* *************************************************************** */


#endif
