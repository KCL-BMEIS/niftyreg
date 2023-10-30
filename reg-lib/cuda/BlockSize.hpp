/** @file BlockSize.hpp
 * @author Marc Modat
 * @date 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

#include <memory>

namespace NiftyReg {
/* *************************************************************** */
struct BlockSize {
    /* _reg_blockMatching_gpu */
    unsigned target_block;
    unsigned result_block;
    /* _reg_mutualinformation_gpu */
    unsigned reg_smoothJointHistogramX;
    unsigned reg_smoothJointHistogramY;
    unsigned reg_smoothJointHistogramZ;
    unsigned reg_smoothJointHistogramW;
    unsigned reg_marginaliseTargetX;
    unsigned reg_marginaliseTargetXY;
    unsigned reg_marginaliseResultX;
    unsigned reg_marginaliseResultXY;
    unsigned reg_getVoxelBasedNMIGradientUsingPW2D;
    unsigned reg_getVoxelBasedNMIGradientUsingPW3D;
    unsigned reg_getVoxelBasedNMIGradientUsingPW2x2;
    /* _reg_globalTransformation_gpu */
    unsigned reg_affine_getDeformationField;
    /* _reg_localTransformation_gpu */
    unsigned reg_spline_getDeformationField2D;
    unsigned reg_spline_getDeformationField3D;
    unsigned reg_spline_getApproxSecondDerivatives2D;
    unsigned reg_spline_getApproxSecondDerivatives3D;
    unsigned reg_spline_getApproxBendingEnergy2D;
    unsigned reg_spline_getApproxBendingEnergy3D;
    unsigned reg_spline_getApproxBendingEnergyGradient2D;
    unsigned reg_spline_getApproxBendingEnergyGradient3D;
    unsigned reg_spline_getApproxJacobianValues2D;
    unsigned reg_spline_getApproxJacobianValues3D;
    unsigned reg_spline_approxLinearEnergyGradient;
    unsigned reg_spline_getJacobianValues2D;
    unsigned reg_spline_getJacobianValues3D;
    unsigned reg_spline_logSquaredValues;
    unsigned reg_spline_computeApproxJacGradient2D;
    unsigned reg_spline_computeApproxJacGradient3D;
    unsigned reg_spline_computeJacGradient2D;
    unsigned reg_spline_computeJacGradient3D;
    unsigned reg_spline_approxCorrectFolding3D;
    unsigned reg_spline_correctFolding3D;
    unsigned reg_getDeformationFromDisplacement;
    unsigned reg_defField_compose2D;
    unsigned reg_defField_compose3D;
    unsigned reg_defField_getJacobianMatrix;
    /* _reg_optimiser_gpu */
    unsigned reg_initialiseConjugateGradient;
    unsigned reg_getConjugateGradient1;
    unsigned reg_getConjugateGradient2;
    unsigned GetMaximalLength;
    unsigned reg_updateControlPointPosition;
    /* _reg_ssd_gpu */
    unsigned GetSsdValue;
    unsigned GetSsdGradient;
    /* _reg_tools_gpu */
    unsigned reg_voxelCentricToNodeCentric;
    unsigned reg_convertNMIGradientFromVoxelToRealSpace;
    unsigned reg_ApplyConvolutionWindowAlongX;
    unsigned reg_ApplyConvolutionWindowAlongY;
    unsigned reg_ApplyConvolutionWindowAlongZ;
    unsigned Arithmetic;
    /* _reg_resampling_gpu */
    unsigned reg_resampleImage2D;
    unsigned reg_resampleImage3D;
    unsigned reg_getImageGradient2D;
    unsigned reg_getImageGradient3D;
};
/* *************************************************************** */
struct BlockSize100: public BlockSize {
    BlockSize100() {
        target_block = 512; // 15 reg - 32 smem - 24 cmem
        result_block = 384; // 21 reg - 11048 smem - 24 cmem
        /* _reg_mutualinformation_gpu */
        reg_smoothJointHistogramX = 384; // 07 reg - 24 smem - 20 cmem
        reg_smoothJointHistogramY = 320; // 11 reg - 24 smem - 20 cmem
        reg_smoothJointHistogramZ = 320; // 11 reg - 24 smem - 20 cmem
        reg_smoothJointHistogramW = 384; // 08 reg - 24 smem - 20 cmem
        reg_marginaliseTargetX = 384; // 06 reg - 24 smem
        reg_marginaliseTargetXY = 384; // 07 reg - 24 smem
        reg_marginaliseResultX = 384; // 06 reg - 24 smem
        reg_marginaliseResultXY = 384; // 07 reg - 24 smem
        reg_getVoxelBasedNMIGradientUsingPW2D = 384; // 21 reg - 24 smem - 32 cmem
        reg_getVoxelBasedNMIGradientUsingPW3D = 320; // 25 reg - 24 smem - 32 cmem
        reg_getVoxelBasedNMIGradientUsingPW2x2 = 192; // 42 reg - 24 smem - 36 cmem
        /* _reg_globalTransformation_gpu */
        reg_affine_getDeformationField = 512; // 16 reg - 24 smem
        /* _reg_localTransformation_gpu */
        reg_spline_getDeformationField2D = 384; // 20 reg - 6168 smem - 28 cmem
        reg_spline_getDeformationField3D = 192; // 37 reg - 6168 smem - 28 cmem
        reg_spline_getApproxSecondDerivatives2D = 512; // 15 reg - 132 smem - 32 cmem
        reg_spline_getApproxSecondDerivatives3D = 192; // 38 reg - 672 smem - 104 cmem
        reg_spline_getApproxBendingEnergy2D = 384; // 07 reg - 24 smem
        reg_spline_getApproxBendingEnergy3D = 320; // 12 reg - 24 smem
        reg_spline_getApproxBendingEnergyGradient2D = 512; // 15 reg - 132 smem - 36 cmem
        reg_spline_getApproxBendingEnergyGradient3D = 256; // 27 reg - 672 smem - 108 cmem
        reg_spline_getApproxJacobianValues2D = 384; // 17 reg - 104 smem - 36 cmem
        reg_spline_getApproxJacobianValues3D = 256; // 27 reg - 356 smem - 108 cmem
        reg_spline_approxLinearEnergyGradient = 384; // 40 reg
        reg_spline_getJacobianValues2D = 256; // 29 reg - 32 smem - 16 cmem - 32 lmem
        reg_spline_getJacobianValues3D = 192; // 41 reg - 6176 smem - 20 cmem - 32 lmem
        reg_spline_logSquaredValues = 384; // 07 reg - 24 smem - 36 cmem
        reg_spline_computeApproxJacGradient2D = 320; // 23 reg - 96 smem - 72 cmem
        reg_spline_computeApproxJacGradient3D = 256; // 32 reg - 384 smem - 144 cmem
        reg_spline_computeJacGradient2D = 384; // 21 reg - 24 smem - 64 cmem
        reg_spline_computeJacGradient3D = 256; // 32 reg - 24 smem - 64 cmem
        reg_spline_approxCorrectFolding3D = 256; // 32 reg - 24 smem - 24 cmem
        reg_spline_correctFolding3D = 256; // 31 reg - 24 smem - 32 cmem
        reg_getDeformationFromDisplacement = 384; // 09 reg - 24 smem
        reg_defField_compose2D = 512; // 15 reg - 24 smem - 08 cmem - 16 lmem
        reg_defField_compose3D = 384; // 21 reg - 24 smem - 08 cmem - 24 lmem
        reg_defField_getJacobianMatrix = 512; // 16 reg - 24 smem - 04 cmem
        /* _reg_optimiser_gpu */
        reg_initialiseConjugateGradient = 384; // 09 reg - 24 smem
        reg_getConjugateGradient1 = 320; // 12 reg - 24 smem
        reg_getConjugateGradient2 = 384; // 10 reg - 40 smem
        GetMaximalLength = 384; // 04 reg - 24 smem
        reg_updateControlPointPosition = 384; // 08 reg - 24 smem
        /* _reg_ssd_gpu */
        GetSsdValue = 320; // 12 reg - 24 smem - 08 cmem
        GetSsdGradient = 320; // 12 reg - 24 smem - 08 cmem
        /* _reg_tools_gpu */
        reg_voxelCentricToNodeCentric = 320; // 11 reg - 24 smem - 16 cmem
        reg_convertNMIGradientFromVoxelToRealSpace = 512; // 16 reg - 24 smem
        reg_ApplyConvolutionWindowAlongX = 512; // 14 reg - 28 smem - 08 cmem
        reg_ApplyConvolutionWindowAlongY = 512; // 14 reg - 28 smem - 08 cmem
        reg_ApplyConvolutionWindowAlongZ = 512; // 15 reg - 28 smem - 08 cmem
        Arithmetic = 384; // 5 reg - 24 smem
        /* _reg_resampling_gpu */
        reg_resampleImage2D = 320; // 10 reg - 24 smem - 12 cmem
        reg_resampleImage3D = 512; // 16 reg - 24 smem - 12 cmem
        reg_getImageGradient2D = 512; // 16 reg - 24 smem - 20 cmem - 24 lmem
        reg_getImageGradient3D = 320; // 24 reg - 24 smem - 16 cmem - 32 lmem
        NR_FUNC_CALLED();
    }
};
/* *************************************************************** */
struct BlockSize300: public BlockSize {
    BlockSize300() {
        target_block = 640; // 45 reg
        result_block = 640; // 47 reg - ????? smem
        /* _reg_mutualinformation_gpu */
        reg_smoothJointHistogramX = 768; // 34 reg
        reg_smoothJointHistogramY = 768; // 34 reg
        reg_smoothJointHistogramZ = 768; // 34 reg
        reg_smoothJointHistogramW = 768; // 34 reg
        reg_marginaliseTargetX = 1024; // 24 reg
        reg_marginaliseTargetXY = 1024; // 24 reg
        reg_marginaliseResultX = 1024; // 24 reg
        reg_marginaliseResultXY = 1024; // 24 reg
        reg_getVoxelBasedNMIGradientUsingPW2D = 768; // 38 reg
        reg_getVoxelBasedNMIGradientUsingPW3D = 640; // 45 reg
        reg_getVoxelBasedNMIGradientUsingPW2x2 = 576; // 55 reg
        /* _reg_globalTransformation_gpu */
        reg_affine_getDeformationField = 1024; // 23 reg
        /* _reg_localTransformation_gpu */
        reg_spline_getDeformationField2D = 1024; // 34 reg
        reg_spline_getDeformationField3D = 1024; // 34 reg
        reg_spline_getApproxSecondDerivatives2D = 1024; // 25 reg
        reg_spline_getApproxSecondDerivatives3D = 768; // 34 reg
        reg_spline_getApproxBendingEnergy2D = 1024; // 23 reg
        reg_spline_getApproxBendingEnergy3D = 1024; // 23 reg
        reg_spline_getApproxBendingEnergyGradient2D = 1024; // 28 reg
        reg_spline_getApproxBendingEnergyGradient3D = 768; // 33 reg
        reg_spline_getApproxJacobianValues2D = 768; // 34 reg
        reg_spline_getApproxJacobianValues3D = 640; // 46 reg
        reg_spline_approxLinearEnergyGradient = 768; // 40 reg
        reg_spline_getJacobianValues2D = 768; // 34 reg
        reg_spline_getJacobianValues3D = 768; // 34 reg
        reg_spline_logSquaredValues = 1024; // 23 reg
        reg_spline_computeApproxJacGradient2D = 768; // 34 reg
        reg_spline_computeApproxJacGradient3D = 768; // 38 reg
        reg_spline_computeJacGradient2D = 768; // 34 reg
        reg_spline_computeJacGradient3D = 768; // 37 reg
        reg_spline_approxCorrectFolding3D = 768; // 34 reg
        reg_spline_correctFolding3D = 768; // 34 reg
        reg_getDeformationFromDisplacement = 1024; // 18 reg
        reg_defField_compose2D = 1024; // 23 reg
        reg_defField_compose3D = 1024; // 24 reg
        reg_defField_getJacobianMatrix = 768; // 34 reg
        /* _reg_optimiser_gpu */
        reg_initialiseConjugateGradient = 1024; // 20 reg
        reg_getConjugateGradient1 = 1024; // 22 reg
        reg_getConjugateGradient2 = 1024; // 25 reg
        GetMaximalLength = 1024; // 20 reg
        reg_updateControlPointPosition = 1024; // 22 reg
        /* _reg_ssd_gpu */
        GetSsdValue = 768; // 34 reg
        GetSsdGradient = 768; // 34 reg
        /* _reg_tools_gpu */
        reg_voxelCentricToNodeCentric = 1024; // 23 reg
        reg_convertNMIGradientFromVoxelToRealSpace = 1024; // 23 reg
        reg_ApplyConvolutionWindowAlongX = 1024; // 25 reg
        reg_ApplyConvolutionWindowAlongY = 1024; // 25 reg
        reg_ApplyConvolutionWindowAlongZ = 1024; // 25 reg
        Arithmetic = 1024; //
        /* _reg_resampling_gpu */
        reg_resampleImage2D = 1024; // 23 reg
        reg_resampleImage3D = 1024; // 24 reg
        reg_getImageGradient2D = 1024; // 34 reg
        reg_getImageGradient3D = 1024; // 34 reg
        NR_FUNC_CALLED();
    }
};
/* *************************************************************** */
} // namespace NiftyReg
