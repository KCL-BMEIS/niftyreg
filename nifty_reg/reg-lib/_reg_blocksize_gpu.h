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
    #ifdef _20_CAPA
        #define Block_reg_affine_deformationField 512                       // 16 regs

        #define Block_result_block 480                                      // 21 regs
        #define Block_target_block 512                                      // 15 regs

        #define Block_reg_bspline_JacobianMatrixFromVel 192                 // 42 regs
        #define Block_reg_bspline_PositionToIndices 512                     // 17 regs
        #define Block_reg_bspline_GetSquaredLogJacDet 512                   // 07 regs
        #define Block_reg_bspline_ApproxJacobianMatrix 416                  // 25 regs
        #define Block_reg_bspline_ApproxJacDetFromVelField 128              // 49 regs
        #define Block_reg_bspline_ApproxJacobianGradient 512                // 32 regs
        #define Block_reg_bspline_JacobianMatrix 192                        // 41 regs
        #define Block_reg_bspline_JacDetFromVelField 128                    // 49 regs
        #define Block_reg_spline_getDeformationFromDisplacement 512         // 19 regs
        #define Block_reg_bspline_JacobianGradient 384                      // 27 regs
        #define Block_reg_bspline_ApproxCorrectFolding 512                  // 31 regs
        #define Block_reg_freeForm_deformationField 352                     // 30 regs
        #define Block_reg_bspline_CorrectFolding 384                        // 27 regs
        #define Block_reg_spline_cppDeconvolve 512                          // 18 regs
        #define Block_reg_bspline_storeApproxBendingEnergy 384              // 39 regs
        #define Block_reg_bspline_SetJacDetToOne 512                        // 02 regs
        #define Block_reg_bspline_getApproxBendingEnergyGradient 512        // 19 regs
        #define Block_reg_bspline_ApproxJacDet 448                          // 24 regs
        #define Block_reg_spline_cppComposition 416                         // 25 regs
        #define Block_reg_bspline_JacDet 192                                // 41 regs
        #define Block_reg_bspline_JacobianGradFromVel 512                   // 20 regs
        #define Block_reg_bspline_ApproxBendingEnergy 384                   // 39 regs

        #define Block_reg_marginaliseResultX 512                            // 07 regs
        #define Block_reg_getVoxelBasedNMIGradientUsingPW2x2 384            // 42 regs
        #define Block_reg_getVoxelBasedNMIGradientUsingPW 416               // 25 regs
        #define Block_reg_smoothJointHistogramW 512                         // 08 regs
        #define Block_reg_smoothJointHistogramX 512                         // 07 regs
        #define Block_reg_smoothJointHistogramY 512                         // 11 regs
        #define Block_reg_marginaliseResultXY 512                           // 07 regs
        #define Block_reg_smoothJointHistogramZ 512                         // 11 regs
        #define Block_reg_marginaliseTargetXY 512                           // 06 regs
        #define Block_reg_marginaliseTargetX 512                            // 06 regs

        #define Block_reg_getSourceImageGradient 448                        // 23 regs
        #define Block_reg_resampleSourceImage 512                           // 16 regs

        #define Block_reg_GetConjugateGradient2 512                         // 10 regs
        #define Block_reg_GetConjugateGradient1 512                         // 12 regs
        #define Block_reg_convertNMIGradientFromVoxelToRealSpace 512        // 16 regs
        #define Block_reg_updateControlPointPosition 512                    // 08 regs
        #define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 15 regs
        #define Block_reg_ApplyConvolutionWindowAlongY 512                  // 14 regs
        #define Block_reg_ApplyConvolutionWindowAlongX 512                  // 14 regs
        #define Block_reg_getMaximalLength 512                              // 07 regs
        #define Block_reg_initialiseConjugateGradient 512                   // 09 regs
        #define Block_reg_voxelCentric2NodeCentric 512                      // 11 regs
    #else
        #define Block_reg_affine_deformationField 512                       // 16 regs

        #define Block_result_block 384                                      // 21 regs
        #define Block_target_block 512                                      // 15 regs

        #define Block_reg_bspline_JacobianMatrixFromVel 192                 // 42 regs - shared 12328
        #define Block_reg_bspline_PositionToIndices 448                     // 17 regs
        #define Block_reg_bspline_GetSquaredLogJacDet 512                   // 07 regs
        #define Block_reg_bspline_ApproxJacobianMatrix 320                  // 25 regs
        #define Block_reg_bspline_ApproxJacDetFromVelField 128              // 49 regs
        #define Block_reg_bspline_ApproxJacobianGradient 512                // 32 regs
        #define Block_reg_bspline_JacobianMatrix 192                        // 41 regs
        #define Block_reg_bspline_JacDetFromVelField 128                    // 49 regs
        #define Block_reg_spline_getDeformationFromDisplacement 384         // 19 regs
        #define Block_reg_bspline_JacobianGradient 512                      // 27 regs
        #define Block_reg_bspline_ApproxCorrectFolding 512                  // 31 regs
        #define Block_reg_freeForm_deformationField 480                     // 30 regs
        #define Block_reg_bspline_CorrectFolding 512                        // 27 regs
        #define Block_reg_spline_cppDeconvolve 448                          // 18 regs
        #define Block_reg_bspline_storeApproxBendingEnergy 384              // 39 regs
        #define Block_reg_bspline_SetJacDetToOne 512                        // 02 regs
        #define Block_reg_bspline_getApproxBendingEnergyGradient 384        // 19 regs
        #define Block_reg_bspline_ApproxJacDet 320                          // 24 regs
        #define Block_reg_spline_cppComposition 320                         // 25 regs
        #define Block_reg_bspline_JacDet 192                                // 41 regs
        #define Block_reg_bspline_JacobianGradFromVel 384                   // 20 regs
        #define Block_reg_bspline_ApproxBendingEnergy 384                   // 39 regs

        #define Block_reg_marginaliseResultX 512                            // 07 regs
        #define Block_reg_getVoxelBasedNMIGradientUsingPW2x2 384            // 42 regs
        #define Block_reg_getVoxelBasedNMIGradientUsingPW 320               // 25 regs
        #define Block_reg_smoothJointHistogramW 512                         // 08 regs
        #define Block_reg_smoothJointHistogramX 512                         // 07 regs
        #define Block_reg_smoothJointHistogramY 512                         // 11 regs
        #define Block_reg_marginaliseResultXY 512                           // 07 regs
        #define Block_reg_smoothJointHistogramZ 512                         // 11 regs
        #define Block_reg_marginaliseTargetXY 512                           // 06 regs
        #define Block_reg_marginaliseTargetX 512                            // 06 regs

        #define Block_reg_getSourceImageGradient 320                        // 23 regs
        #define Block_reg_resampleSourceImage 512                           // 16 regs

        #define Block_reg_GetConjugateGradient2 512                         // 10 regs
        #define Block_reg_GetConjugateGradient1 512                         // 12 regs
        #define Block_reg_convertNMIGradientFromVoxelToRealSpace 512        // 16 regs
        #define Block_reg_updateControlPointPosition 512                    // 08 regs
        #define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 15 regs
        #define Block_reg_ApplyConvolutionWindowAlongY 512                  // 14 regs
        #define Block_reg_ApplyConvolutionWindowAlongX 512                  // 14 regs
        #define Block_reg_getMaximalLength 512                              // 07 regs
        #define Block_reg_initialiseConjugateGradient 512                   // 09 regs
        #define Block_reg_voxelCentric2NodeCentric 512                      // 11 regs
    #endif
#else
    #define Block_reg_affine_deformationField 512                       // 16 regs

    #define Block_result_block 343                                      // 21 regs
    #define Block_target_block 512                                      // 15 regs

    #define Block_reg_bspline_JacobianMatrixFromVel 192                 // 42 regs - shared 12328
    #define Block_reg_bspline_PositionToIndices 448                     // 17 regs
    #define Block_reg_bspline_GetSquaredLogJacDet 384                   // 07 regs
    #define Block_reg_bspline_ApproxJacobianMatrix 320                  // 25 regs
    #define Block_reg_bspline_ApproxJacDetFromVelField 128              // 49 regs
    #define Block_reg_bspline_ApproxJacobianGradient 256                // 32 regs
    #define Block_reg_bspline_JacobianMatrix 192                        // 41 regs
    #define Block_reg_bspline_JacDetFromVelField 128                    // 49 regs
    #define Block_reg_spline_getDeformationFromDisplacement 384         // 19 regs
    #define Block_reg_bspline_JacobianGradient 256                      // 27 regs
    #define Block_reg_bspline_ApproxCorrectFolding 256                  // 31 regs
    #define Block_reg_freeForm_deformationField 256                     // 30 regs
    #define Block_reg_bspline_CorrectFolding 256                        // 27 regs
    #define Block_reg_spline_cppDeconvolve 448                          // 18 regs
    #define Block_reg_bspline_storeApproxBendingEnergy 192              // 39 regs
    #define Block_reg_bspline_SetJacDetToOne 384                        // 02 regs
    #define Block_reg_bspline_getApproxBendingEnergyGradient 384        // 19 regs
    #define Block_reg_bspline_ApproxJacDet 320                          // 24 regs
    #define Block_reg_spline_cppComposition 320                         // 25 regs
    #define Block_reg_bspline_JacDet 192                                // 41 regs
    #define Block_reg_bspline_JacobianGradFromVel 384                   // 20 regs
    #define Block_reg_bspline_ApproxBendingEnergy 192                   // 39 regs

    #define Block_reg_marginaliseResultX 384                            // 07 regs
    #define Block_reg_getVoxelBasedNMIGradientUsingPW2x2 192            // 42 regs
    #define Block_reg_getVoxelBasedNMIGradientUsingPW 320               // 25 regs
    #define Block_reg_smoothJointHistogramW 384                         // 08 regs
    #define Block_reg_smoothJointHistogramX 384                         // 07 regs
    #define Block_reg_smoothJointHistogramY 320                         // 11 regs
    #define Block_reg_marginaliseResultXY 384                           // 07 regs
    #define Block_reg_smoothJointHistogramZ 320                         // 11 regs
    #define Block_reg_marginaliseTargetXY 384                           // 06 regs
    #define Block_reg_marginaliseTargetX 384                            // 06 regs

    #define Block_reg_getSourceImageGradient 320                        // 23 regs
    #define Block_reg_resampleSourceImage 512                           // 16 regs

    #define Block_reg_GetConjugateGradient2 384                         // 10 regs
    #define Block_reg_GetConjugateGradient1 320                         // 12 regs
    #define Block_reg_convertNMIGradientFromVoxelToRealSpace 512        // 16 regs
    #define Block_reg_updateControlPointPosition 384                    // 08 regs
    #define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 15 regs
    #define Block_reg_ApplyConvolutionWindowAlongY 512                  // 14 regs
    #define Block_reg_ApplyConvolutionWindowAlongX 512                  // 14 regs
    #define Block_reg_getMaximalLength 384                              // 07 regs
    #define Block_reg_initialiseConjugateGradient 384                   // 09 regs
    #define Block_reg_voxelCentric2NodeCentric 320                      // 11 regs
#endif




#endif
