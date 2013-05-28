/*
 * @file _reg_ssd_gpu.h
 * @author Marc Modat
 * @date 14/11/2012
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_GPU_H
#define _REG_SSD_GPU_H

#include "_reg_tools_gpu.h"

extern "C++"
float reg_getSSD_gpu(nifti_image *referenceImage,
                     cudaArray **reference_d,
                     float **warped_d,
                     int **mask_d,
                     int activeVoxelNumber
                     );


extern "C++"
void reg_getVoxelBasedSSDGradient_gpu(nifti_image *referenceImage,
									  cudaArray **reference_d,
									  float **warped_d,
									  float4 **spaGradient_d,
									  float4 **ssdGradient_d,
									  float maxSD,
                                      int **mask_d,
                                      int activeVoxelNumber
									  );
#endif
