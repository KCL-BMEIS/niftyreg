#pragma once
#include "nifti1_io.h"
#include "_reg_blockMatching.h"

typedef struct nifti_params {

	int ndim; /*!< last dimension greater than 1 (1..7) */
	int nx; /*!< dimensions of grid array             */
	int ny; /*!< dimensions of grid array             */
	int nz; /*!< dimensions of grid array             */
	int nt; /*!< dimensions of grid array             */
	int nu; /*!< dimensions of grid array             */
	int nv; /*!< dimensions of grid array             */
	int nw; /*!< dimensions of grid array             */
	unsigned long nvox; /*!< number of voxels = nx*ny*nz*...*nw   */
	int nbyper; /*!< bytes per voxel, matches datatype    */
	int datatype; /*!< type of data in voxels: DT_* code    */

	float dx; /*!< grid spacings      */
	float dy; /*!< grid spacings      */
	float dz; /*!< grid spacings      */
	float dt; /*!< grid spacings      */
	float du; /*!< grid spacings      */
	float dv; /*!< grid spacings      */
	float dw; /*!< grid spacings      */
	unsigned int nxyz;           //xyz image size

} nifti_params_t;

void launchConvolution(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis);
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField,float** def_d, int** mask_d, bool compose = false);
void launchBlockMatching(nifti_image * target, _reg_blockMatchingParam *params, float **targetImageArray_d, float **resultImageArray_d, float **targetPosition_d, float **resultPosition_d, int **activeBlock_d, int **mask_d);
void launchResample(nifti_image *floatingImage, nifti_image *warpedImage, int *mask, int interp, float paddingValue, bool *dti_timepoint, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d, int** mask_d);
void runKernel2(nifti_image *floatingImage, nifti_image *warpedImage, int *mask, int interp, float paddingValue, int *dtiIndeces, mat33 * jacMat, float** floatingImage_d, float** warpedImage_d, float** deformationFieldImage_d, int** mask_d);
