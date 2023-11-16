/**
 * @file _reg_tools.h
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <functional>
#include "_reg_maths.h"
#include "Debug.hpp"

using namespace NiftyReg;
using namespace std::string_literals;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using RNifti::NiftiImage;
using RNifti::NiftiImageData;
using NiftiDim = NiftiImage::Dim;

enum class ConvKernelType { Mean, Linear, Gaussian, Cubic };

/* *************************************************************** */
/** @brief This function check some header parameters and correct them in
 * case of error. For example no dimension is lower than one. The scl_slope
 * can not be equal to zero. The qto_xyz and qto_ijk are populated if
 * both qform_code and sform_code are set to zero.
 * @param image Input image to check and correct if necessary
 */
void reg_checkAndCorrectDimension(nifti_image *image);
/* *************************************************************** */
/** @brief Check if the specified filename corresponds to an image.
 * @param name Input filename
 * @return True is the specified filename corresponds to an image,
 * false otherwise.
 */
bool reg_isAnImageFileName(const char *name);
/* *************************************************************** */
/** @brief Rescale an input image between two user-defined values.
 * Some threshold can also be applied concurrently
 * @param image Image to be rescaled
 * @param newMin Intensity lower bound after rescaling
 * @param newMax Intensity higher bound after rescaling
 * @param lowThr Intensity to use as lower threshold
 * @param upThr Intensity to use as higher threshold
 */
void reg_intensityRescale(nifti_image *image,
                          int timepoint,
                          float newMin,
                          float newMax);
/* *************************************************************** */
/** @brief Set the scl_slope to 1 and the scl_inter to 0 and rescale
 * the intensity values
 * @param image Image to be updated
 */
void reg_tools_removeSCLInfo(nifti_image *img);
/* *************************************************************** */
/** @brief reg_getRealImageSpacing
 * @param image image
 * @param spacingValues spacingValues
 */
void reg_getRealImageSpacing(nifti_image *image,
                             float *spacingValues);
/* *************************************************************** */
/** @brief Smooth an image using a specified kernel
 * @param image Image to be smoothed
 * @param sigma Standard deviation of the kernel to use.
 * The kernel is bounded between +/- 3 sigma.
 * @param kernelType Type of kernel to use.
 * @param mask An integer mask over which the smoothing should occur.
 * @param timePoints Boolean array to specify which time points have to be
 * smoothed. The array follow the dim array of the nifti header.
 * @param axes Boolean array to specify which axes have to be
 * smoothed. The array follow the dim array of the nifti header.
 */
void reg_tools_kernelConvolution(nifti_image *image,
                                 const float *sigma,
                                 const ConvKernelType kernelType,
                                 const int *mask = nullptr,
                                 const bool *timePoints = nullptr,
                                 const bool *axes = nullptr);
/* *************************************************************** */
/** @brief Smooth a label image using a Gaussian kernel
 * @param image Image to be smoothed
 * @param varianceX The variance of the Gaussian kernel in X
 * @param varianceY The variance of the Gaussian kernel in Y
 * @param varianceZ The variance of the Gaussian kernel in Z
 * @param mask An integer mask over which the Gaussian smoothing should occur.
 * @param timePoints Boolean array to specify which time points have to be
 * smoothed.
 */
void reg_tools_labelKernelConvolution(nifti_image *image,
                                      float varianceX,
                                      float varianceY,
                                      float varianceZ,
                                      int *mask = nullptr,
                                      bool *timePoints = nullptr);
/* *************************************************************** */
/** @brief Downsample an image by a ratio of two
 * @param image Image to be downsampled
 * @param type The image is first smoothed  using a Gaussian
 * kernel of 0.7 voxel standard deviation before being downsample
 * if type is set to true.
 * @param axes Boolean array to specify which axes have to be
 * downsampled. The array follow the dim array of the nifti header.
 */
template <class PrecisionType>
void reg_downsampleImage(nifti_image *image,
                         int type,
                         bool *axes);
/* *************************************************************** */
/** @brief Returns the maximal euclidean distance from a
 * deformation field image
 * @param image Vector image to be considered
 * @return Scalar value that corresponds to the longest
 * euclidean distance
 */
template <class PrecisionType>
PrecisionType reg_getMaximalLength(const nifti_image *image,
                                   const bool optimiseX,
                                   const bool optimiseY,
                                   const bool optimiseZ);
/* *************************************************************** */
/** @brief Change the datatype of a nifti image
 * @param image Image to be updated.
 */
template <class NewType>
void reg_tools_changeDatatype(nifti_image *image,
                              int type = -1);
/* *************************************************************** */
/** @brief Add two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
void reg_tools_addImageToImage(const nifti_image *img1,
                               const nifti_image *img2,
                               nifti_image *out);
/* *************************************************************** */
/** @brief Subtract two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
void reg_tools_subtractImageFromImage(const nifti_image *img1,
                                      const nifti_image *img2,
                                      nifti_image *out);
/* *************************************************************** */
/** @brief Multiply two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
void reg_tools_multiplyImageToImage(const nifti_image *img1,
                                    const nifti_image *img2,
                                    nifti_image *out);
/* *************************************************************** */
/** @brief Divide two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
void reg_tools_divideImageToImage(const nifti_image *img1,
                                  const nifti_image *img2,
                                  nifti_image *out);
/* *************************************************************** */
/** @brief Add a scalar to all image intensity
 * @param img Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be added to input image
 */
void reg_tools_addValueToImage(const nifti_image *img,
                               nifti_image *out,
                               const double val);
/* *************************************************************** */
/** @brief Subtract a scalar from all image intensity
 * @param img Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be subtracted from input image
 */
void reg_tools_subtractValueFromImage(const nifti_image *img,
                                      nifti_image *out,
                                      const double val);
/* *************************************************************** */
/** @brief Multiply a scalar to all image intensity
 * @param img Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be multiplied to input image
 */
void reg_tools_multiplyValueToImage(const nifti_image *img,
                                    nifti_image *out,
                                    const double val);
/* *************************************************************** */
/** @brief Divide a scalar to all image intensity
 * @param img Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be divided to input image
 */
void reg_tools_divideValueToImage(const nifti_image *img,
                                  nifti_image *out,
                                  const double val);
/* *************************************************************** */
/** @brief Binarise an input image. All values different
 * from 0 are set to 1, 0 otherwise.
 * @param img Image that will be binarise inline
 */
void reg_tools_binarise_image(nifti_image *img);
/* *************************************************************** */
/** @brief Binarise an input image. The binarisation is
 * performed according to a threshold value that is
 * user-defined.
 * @param img Image that will be binarise inline
 * @param thr Threshold value used for binarisation.
 * All values bellow thr are set to 0. All values equal
 * or bellow thr are set to 1
 */
void reg_tools_binarise_image(nifti_image *img,
                              float thr);
/* *************************************************************** */
/** @brief Convert a binary image into an array of int.
 * This is used to define a mask within the registration
 * function.
 * @param img Input image
 * @param array The data array from the input nifti image
 * is binarised and stored in this array.
 */
void reg_tools_binaryImage2int(const nifti_image *img,
                               int *array);
/* *************************************************************** */
/** @brief Compute the mean root mean squared error between
 * two vector images
 * @param imgA Input vector image
 * @param imgB Input vector image
 * @return Mean root mean squared error values returned
 */
double reg_tools_getMeanRMS(const nifti_image *imgA,
                            const nifti_image *imgB);
/* *************************************************************** */
/** @brief Set all voxels from an image to NaN if the voxel
 * belong to the mask
 * @param img Input image to be masked with NaN value
 * @param mask Input mask that defines which voxels
 * have to be set to NaN
 * @param res Output image
 */
int reg_tools_nanMask_image(const nifti_image *img,
                            const nifti_image *mask,
                            nifti_image *res);
/* *************************************************************** */
/** @brief Set all the voxel with NaN value in the input image to
 * background in the input mask
 * @param img Input image
 * @param mask Input mask which is updated in place
 */
int reg_tools_removeNanFromMask(const nifti_image *image, int *mask);
/* *************************************************************** */
/** @brief Get the minimal value of an image
 * @param img Input image
 * @param timepoint active time point. All time points are used if set to -1
 * @return min value
 */
float reg_tools_getMinValue(const nifti_image *img, int timepoint);
/* *************************************************************** */
/** @brief Get the maximal value of an image
 * @param img Input image
 * @param timepoint active time point. All time points are used if set to -1
 * @return max value
 */
float reg_tools_getMaxValue(const nifti_image *img, int timepoint);
/* *************************************************************** */
/** @brief Get the mean value of an image
 * @param img Input image
 * @return mean value
 */
float reg_tools_getMeanValue(const nifti_image *img);
/* *************************************************************** */
/** @brief Get the std value of an image
 * @param img Input image
 * @return std value
 */
float reg_tools_getSTDValue(const nifti_image *img);
/* *************************************************************** */
/** @brief Generate a pyramid from an input image.
 * @param input Input image to be downsampled to create the pyramid
 * @param pyramid Output array of images that will contains the
 * different resolution images of the pyramid
 * @param levelNumber Number of level to use to create the pyramid.
 * 1 level corresponds to the original image resolution.
 * @param levelToPerform Number to level that will be perform during
 * the registration.
 */
template<class DataType>
void reg_createImagePyramid(const NiftiImage& input,
                            vector<NiftiImage>& pyramid,
                            unsigned levelNumber,
                            unsigned levelToPerform);
/* *************************************************************** */
/** @brief Generate a pyramid from an input mask image.
 * @param input Input image to be downsampled to create the pyramid
 * @param pyramid Output array of mask images that will contains the
 * different resolution images of the pyramid
 * @param levelNumber Number of level to use to create the pyramid.
 * 1 level corresponds to the original image resolution.
 * @param levelToPerform Number to level that will be perform during
 * the registration.
 */
template<class DataType>
void reg_createMaskPyramid(const NiftiImage& input,
                           vector<unique_ptr<int[]>>& pyramid,
                           unsigned levelNumber,
                           unsigned levelToPerform);
/* *************************************************************** */
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
template<class T>
void reg_thresholdImage(nifti_image *image,
                        T lowThr,
                        T upThr);
/* *************************************************************** */
/** @brief This function flip the specified axis
 * @param image Input image to be flipped
 * @param array Array that will contain the flipped
 * input image->data array
 * @param cmd String that contains the letter(s) of the axis
 * to flip (xyztuvw)
 */
void reg_flipAxis(const nifti_image *image,
                  void **outputArray,
                  const std::string& cmd);
/* *************************************************************** */
/** @brief This function converts an image containing deformation
 * field into a displacement field
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
int reg_getDisplacementFromDeformation(nifti_image *image);
/* *************************************************************** */
/** @brief This function converts an image containing a displacement field
 * into a displacement field.
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
int reg_getDeformationFromDisplacement(nifti_image *image);
/* *************************************************************** */
/** @brief Set the gradient value along specified direction to zero
 * @param image Input Image that will be modified
 * @param xAxis Boolean to specified if the x-axis has to be zeroed
 * @param yAxis Boolean to specified if the y-axis has to be zeroed
 * @param zAxis Boolean to specified if the z-axis has to be zeroed
 */
void reg_setGradientToZero(nifti_image *image,
                           bool xAxis,
                           bool yAxis,
                           bool zAxis);
/* *************************************************************** */
/* *************************************************************** */
/** @brief The functions returns the largest ratio between two arrays
 * The returned value is the largest value computed as ((A/B)-1)
 * If A or B are zeros then the (A-B) value is returned.
 */
template<class DataType>
double reg_test_compare_arrays(const DataType *ptrA,
                               const DataType *ptrB,
                               size_t nvox);
/* *************************************************************** */
/** @brief The functions returns the largest ratio between input image intensities
 * The returned value is the largest value computed as ((A/B)-1)
 * If A or B are zeros then the (A-B) value is returned.
 */
double reg_test_compare_images(const nifti_image *imgA,
                               const nifti_image *imgB);
/* *************************************************************** */
/** @brief The absolute operator is applied to the input image
 */
void reg_tools_abs_image(nifti_image *img);
/* *************************************************************** */
void mat44ToCptr(const mat44& mat, float *cMat);
/* *************************************************************** */
void cPtrToMat44(mat44 *mat, const float *cMat);
/* *************************************************************** */
void mat33ToCptr(const mat33 *mat, float *cMat, const unsigned numMats);
/* *************************************************************** */
void cPtrToMat33(mat33 *mat, const float *cMat);
/* *************************************************************** */
template<typename T>
void matmnToCptr(const T **mat, T *cMat, unsigned m, unsigned n);
/* *************************************************************** */
template<typename T>
void cPtrToMatmn(T **mat, const T *cMat, unsigned m, unsigned n);
/* *************************************************************** */
void coordinateFromLinearIndex(int index, int maxValue_x, int maxValue_y, int& x, int& y, int& z);
/* *************************************************************** */
/** @brief Duplicates the nifti image
 * @param image Input image
 * @param copyData Boolean to specify if the image data should be copied
 * @return The duplicated image
 */
nifti_image* nifti_dup(const nifti_image& image, const bool copyData = true);
/* *************************************************************** */
/// @brief Prints the command line
void PrintCmdLine(const int argc, const char *const *argv, const bool verbose);
/* *************************************************************** */
