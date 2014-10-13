/** @file _reg_common_gpu.h
 * @author Marc Modat
 * @date 25/03/2009.
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_COMMON_GPU_H
#define _REG_COMMON_GPU_H

#include "_reg_blocksize_gpu.h"

/* ******************************** */
/* ******************************** */
int cudaCommon_setCUDACard(CUcontext *ctx,
                           bool verbose);
/* ******************************** */
void cudaCommon_unsetCUDACard(CUcontext *ctx);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **, cudaArray **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, int);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, DTYPE **, int *);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **, cudaArray **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **, DTYPE **, nifti_image *);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE, class DTYPE2>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *, DTYPE **);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *, DTYPE **);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *, DTYPE **, DTYPE **);
/* ******************************** */
/* ******************************** */
extern "C++"
void cudaCommon_free(cudaArray **);
/* ******************************** */
extern "C++" template <class DTYPE>
void cudaCommon_free(DTYPE **);
/* ******************************** */
/* ******************************** */
extern "C++" template <class DTYPE>
int cudaCommon_allocateNiftiToDevice(nifti_image **image_d, int *dim);

template <class DTYPE>
int cudaCommon_transferNiftiToNiftiOnDevice1(nifti_image **image_d, nifti_image *img);


extern "C++"
template <class DTYPE> 
int cudaCommon_transferFromDeviceToNiftiSimple(DTYPE **, nifti_image * );

extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple1(DTYPE **array_d, DTYPE *img, const unsigned  nvox);

#endif
