#pragma once
#include "nifti1_io.h"

void launchConvolution(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis);
void launchResample(nifti_image *floatingImage, nifti_image *warpedImage,  int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d, int** mask_d, float** floMat_d);
void launchOptimizer();//TODO

double sortAndReduce(float* lengths_d, float* target_d, float* result_d, float* newResult_d, const unsigned int numBlocks, const unsigned int numToKeep, const unsigned int m);
