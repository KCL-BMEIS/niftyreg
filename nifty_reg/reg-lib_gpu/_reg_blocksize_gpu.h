/*
 *  _reg_blocksize_gpu.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifdef _USE_CUDA

#ifndef _REG_BLOCKSIZE_GPU_H
#define _REG_BLOCKSIZE_GPU_H

#include "nifti1_io.h"
#include "cuda_runtime.h"
#include <cutil.h>

#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
        struct __attribute__(aligned(4)) float4{
                float x,y,z,w;
        };
#endif

#ifdef _HIGH_CAPA
	#define Block_reg_affine_deformationField 512                       // 16 regs - 100% occupancy
	#define Block_reg_resampleSourceImage 512                           // 16 regs - 100% occupancy
	#define Block_reg_spline_cppComposition 320                         // 25 regs - 063% occupancy
	#define Block_reg_getSourceImageGradient 320                        // 23 regs - 063% occupancy
	#define Block_reg_getVoxelBasedNMIGradientUsingPW 320               // 24 regs - 063% occupancy
	#define Block_reg_ApplyConvolutionWindowAlongX 512                  // 11 regs - 100% occupancy
	#define Block_reg_ApplyConvolutionWindowAlongY 512                  // 11 regs - 100% occupancy
	#define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 12 regs - 100% occupancy
	#define Block_reg_voxelCentric2NodeCentric 512                      // 11 regs - 083% occupancy
	#define Block_reg_convertNMIGradientFromVoxelToRealSpace 448        // 18 regs - 088% occupancy
	#define Block_reg_GetConjugateGradient1 512                         // 12 regs - 100% occupancy
	#define Block_reg_bspline_ApproxBendingEnergy 384                   // 39 regs - 038% occupancy
	#define Block_reg_bspline_storeApproxBendingEnergy 384              // 39 regs - 038% occupancy
	#define Block_reg_bspline_getApproxBendingEnergyGradient 384        // 19 regs - 075% occupancy
    #define Block_reg_bspline_ApproxJacobian 512                        // 25 regs - 050% occupancy
    #define Block_reg_bspline_JacobianGradient 512                      // 27 regs - 050% occupancy
    #define Block_reg_bspline_ApproxJacobianGradient 512                // 32 regs - 050% occupancy
    #define Block_reg_bspline_CorrectFolding 512                        // 27 regs - 050% occupancy
    #define Block_reg_bspline_ApproxCorrectFolding 512                  // 31 regs - 050% occupancy
#else
	#define Block_reg_affine_deformationField 256                       // 16 regs - 067% occupancy
	#define Block_reg_resampleSourceImage 256                           // 16 regs - 067% occupancy
	#define Block_reg_spline_cppComposition 320                         // 25 regs - 042% occupancy
	#define Block_reg_getSourceImageGradient 320                        // 23 regs - 042% occupancy
	#define Block_reg_getVoxelBasedNMIGradientUsingPW 320               // 24 regs - 042% occupancy
	#define Block_reg_ApplyConvolutionWindowAlongX 320                  // 11 regs - 083% occupancy
	#define Block_reg_ApplyConvolutionWindowAlongY 320                  // 11 regs - 083% occupancy
	#define Block_reg_ApplyConvolutionWindowAlongZ 320                  // 12 regs - 083% occupancy
	#define Block_reg_voxelCentric2NodeCentric 320                      // 11 regs - 083% occupancy
	#define Block_reg_convertNMIGradientFromVoxelToRealSpace 448        // 18 regs - 058% occupancy
	#define Block_reg_GetConjugateGradient1 320                         // 12 regs - 083% occupancy
	#define Block_reg_bspline_ApproxBendingEnergy 192                   // 39 regs - 025% occupancy
	#define Block_reg_bspline_storeApproxBendingEnergy 192              // 39 regs - 025% occupancy
	#define Block_reg_bspline_getApproxBendingEnergyGradient 384        // 19 regs - 050% occupancy
    #define Block_reg_bspline_ApproxJacobian 320                        // 25 regs - 042% occupancy
    #define Block_reg_bspline_JacobianGradient 256                      // 27 regs - 033% occupancy
    #define Block_reg_bspline_ApproxJacobianGradient 256                // 32 regs - 030% occupancy
    #define Block_reg_bspline_CorrectFolding 256                        // 27 regs - 033% occupancy
    #define Block_reg_bspline_ApproxCorrectFolding 256                  // 31 regs - 033% occupancy
#endif

#define Block_reg_bspline_Jacobian 192                                  // 41 regs - 025% occupancy
#define Block_reg_bspline_JacobianFromVelocityField 128                 // 49 regs - 025% occupancy
#define Block_reg_bspline_ApproxJacobianFromVelocityField 128           // 49 regs - 025% occupancy

#define Block_reg_freeForm_interpolatePosition 256                      // 30 regs - 025% occupancy
#define Block_reg_updateControlPointPosition 384                        // 08 regs - 100% occupancy
#define Block_reg_spline_cppDeconvolve 384                              // 19 regs - 075% occupancy (050% in 1.0/1.1)
#define Block_reg_spline_deformationFromDisplacement 256                // 19 regs - 075% occupancy (050% in 1.0/1.1)
#define Block_reg_initialiseConjugateGradient 384                       // 09 regs - 100% occupancy
#define Block_reg_GetConjugateGradient2 384                             // 10 regs - 100% occupancy
#define Block_reg_getMaximalLength 384                                  // 07 regs - 100% occupancy
#define Block_reg_FillConvolutionWindows 384                            // 04 regs - 100% occupancy
#define Block_target_block 512                                          // 26 regs - 100% occupancy
#define Block_result_block 343                                          // 21 regs - 034% occupancy



#endif
#endif
