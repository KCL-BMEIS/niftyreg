#pragma once
#include "nifti1_io.h"
#include "_reg_blockMatching.h"

typedef struct nifti_params {

	int ndim;                    /*!< last dimension greater than 1 (1..7) */
	int nx;                      /*!< dimensions of grid array             */
	int ny;                      /*!< dimensions of grid array             */
	int nz;                      /*!< dimensions of grid array             */
	int nt;                      /*!< dimensions of grid array             */
	int nu;                      /*!< dimensions of grid array             */
	int nv;                      /*!< dimensions of grid array             */
	int nw;                      /*!< dimensions of grid array             */
	unsigned long nvox;          /*!< number of voxels = nx*ny*nz*...*nw   */
	int nbyper;                  /*!< bytes per voxel, matches datatype    */
	int datatype;                /*!< type of data in voxels: DT_* code    */

	float dx;                    /*!< grid spacings      */
	float dy;                    /*!< grid spacings      */
	float dz;                    /*!< grid spacings      */
	float dt;                    /*!< grid spacings      */
	float du;                    /*!< grid spacings      */
	float dv;                    /*!< grid spacings      */
	float dw;                    /*!< grid spacings      */
	unsigned int nxyz;           //xyz image size

}nifti_params_t;

void launch(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoint, bool *axis);
void launchAffine(mat44 *affineTransformation, nifti_image *deformationField, bool compose = false, int *mask = NULL);
void launchResample(nifti_image *floatingImage, nifti_image *warpedImage, nifti_image *deformationField, int *mask, int interp, float paddingValue, bool *dti_timepoint=NULL, mat33 * jacMat=NULL);
void launchBlockMatching(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask);


void launchOptimize(_reg_blockMatchingParam *params, mat44 *transformation_matrix, bool affine);
void launchOptimizeAffine(_reg_blockMatchingParam* params, mat44* transformation_matrix, bool affine);
void launchOptimizeRigid(_reg_blockMatchingParam* params, mat44* transformation_matrix, bool affine);