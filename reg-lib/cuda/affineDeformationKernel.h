#pragma once
#include "nifti1_io.h"
//
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField, float** def_d, int** mask_d, float** trans_d, bool compose = false);