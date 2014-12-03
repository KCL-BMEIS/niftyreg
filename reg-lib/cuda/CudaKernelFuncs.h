#pragma once
#include "nifti1_io.h"

void launchConvolution(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis);
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField,float** def_d, int** mask_d, float** trans_d, bool compose = false);
void launchResample(nifti_image *floatingImage, nifti_image *warpedImage,  int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d, int** mask_d, float** floMat_d);
