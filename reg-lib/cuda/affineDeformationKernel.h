#pragma once

#include "RNifti.h"

void launchAffine(mat44 *affineTransformation, nifti_image *deformationField, float* def_d, const int* mask_d, float* trans_d, bool compose = false);