#pragma once
#include "nifti1_io.h"
#include "_reg_blockMatching.h"

void launchConvolution(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis);
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField,float** def_d, int** mask_d, float** trans_d, bool compose = false);
void launchBlockMatching(nifti_image * target, _reg_blockMatchingParam *params, float **targetImageArray_d, float **resultImageArray_d, float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d, float** targetMat_d);
void launchResample(nifti_image *floatingImage, nifti_image *warpedImage,  int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d, int** mask_d, float** floMat_d);
void runKernel2(nifti_image *floatingImage, nifti_image *warpedImage,  int interp, float paddingValue, int *dtiIndeces, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d, int** mask_d, float** floIJKMat_d);
