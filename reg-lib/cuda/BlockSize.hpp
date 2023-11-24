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
    unsigned reg_affine_getDeformationField;
    unsigned reg_spline_getDeformationField2D;
    unsigned reg_spline_getDeformationField3D;
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
    unsigned reg_defField_compose2D;
    unsigned reg_defField_compose3D;
    unsigned reg_defField_getJacobianMatrix;
    unsigned reg_voxelCentricToNodeCentric;
    unsigned reg_convertNmiGradientFromVoxelToRealSpace;
    unsigned reg_ApplyConvolutionWindowAlongX;
    unsigned reg_ApplyConvolutionWindowAlongY;
    unsigned reg_ApplyConvolutionWindowAlongZ;
    unsigned Arithmetic;
    unsigned reg_resampleImage2D;
    unsigned reg_resampleImage3D;
    unsigned reg_getImageGradient2D;
    unsigned reg_getImageGradient3D;
};
/* *************************************************************** */
struct BlockSize100: public BlockSize {
    BlockSize100() {
        reg_affine_getDeformationField = 512; // 16 reg - 24 smem
        reg_spline_getDeformationField2D = 384; // 20 reg - 6168 smem - 28 cmem
        reg_spline_getDeformationField3D = 192; // 37 reg - 6168 smem - 28 cmem
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
        reg_defField_compose2D = 512; // 15 reg - 24 smem - 08 cmem - 16 lmem
        reg_defField_compose3D = 384; // 21 reg - 24 smem - 08 cmem - 24 lmem
        reg_defField_getJacobianMatrix = 512; // 16 reg - 24 smem - 04 cmem
        reg_voxelCentricToNodeCentric = 320; // 11 reg - 24 smem - 16 cmem
        reg_convertNmiGradientFromVoxelToRealSpace = 512; // 16 reg - 24 smem
        reg_ApplyConvolutionWindowAlongX = 512; // 14 reg - 28 smem - 08 cmem
        reg_ApplyConvolutionWindowAlongY = 512; // 14 reg - 28 smem - 08 cmem
        reg_ApplyConvolutionWindowAlongZ = 512; // 15 reg - 28 smem - 08 cmem
        Arithmetic = 384; // 5 reg - 24 smem
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
        reg_affine_getDeformationField = 1024; // 23 reg
        reg_spline_getDeformationField2D = 1024; // 34 reg
        reg_spline_getDeformationField3D = 1024; // 34 reg
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
        reg_defField_compose2D = 1024; // 23 reg
        reg_defField_compose3D = 1024; // 24 reg
        reg_defField_getJacobianMatrix = 768; // 34 reg
        reg_voxelCentricToNodeCentric = 1024; // 23 reg
        reg_convertNmiGradientFromVoxelToRealSpace = 1024; // 23 reg
        reg_ApplyConvolutionWindowAlongX = 1024; // 25 reg
        reg_ApplyConvolutionWindowAlongY = 1024; // 25 reg
        reg_ApplyConvolutionWindowAlongZ = 1024; // 25 reg
        Arithmetic = 1024; //
        reg_resampleImage2D = 1024; // 23 reg
        reg_resampleImage3D = 1024; // 24 reg
        reg_getImageGradient2D = 1024; // 34 reg
        reg_getImageGradient3D = 1024; // 34 reg
        NR_FUNC_CALLED();
    }
};
/* *************************************************************** */
} // namespace NiftyReg
