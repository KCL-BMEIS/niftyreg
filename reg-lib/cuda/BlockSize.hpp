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
    unsigned GetDeformationField2d;
    unsigned GetDeformationField3d;
    unsigned GetApproxJacobianValues2d;
    unsigned GetApproxJacobianValues3d;
    unsigned ApproxLinearEnergyGradient;
    unsigned GetJacobianValues2d;
    unsigned GetJacobianValues3d;
    unsigned LogSquaredValues;
    unsigned ComputeApproxJacGradient2d;
    unsigned ComputeApproxJacGradient3d;
    unsigned ComputeJacGradient2d;
    unsigned ComputeJacGradient3d;
    unsigned ApproxCorrectFolding3d;
    unsigned CorrectFolding3d;
    unsigned DefFieldCompose2d;
    unsigned DefFieldCompose3d;
    unsigned GetJacobianMatrix;
    unsigned reg_voxelCentricToNodeCentric;
    unsigned reg_convertNmiGradientFromVoxelToRealSpace;
    unsigned reg_ApplyConvolutionWindowAlongX;
    unsigned reg_ApplyConvolutionWindowAlongY;
    unsigned reg_ApplyConvolutionWindowAlongZ;
    unsigned Arithmetic;
};
/* *************************************************************** */
struct BlockSize100: public BlockSize {
    BlockSize100() {
        reg_affine_getDeformationField = 512; // 16 reg - 24 smem
        GetDeformationField2d = 384; // 20 reg - 6168 smem - 28 cmem
        GetDeformationField3d = 192; // 37 reg - 6168 smem - 28 cmem
        GetApproxJacobianValues2d = 384; // 17 reg - 104 smem - 36 cmem
        GetApproxJacobianValues3d = 256; // 27 reg - 356 smem - 108 cmem
        ApproxLinearEnergyGradient = 384; // 40 reg
        GetJacobianValues2d = 256; // 29 reg - 32 smem - 16 cmem - 32 lmem
        GetJacobianValues3d = 192; // 41 reg - 6176 smem - 20 cmem - 32 lmem
        LogSquaredValues = 384; // 07 reg - 24 smem - 36 cmem
        ComputeApproxJacGradient2d = 320; // 23 reg - 96 smem - 72 cmem
        ComputeApproxJacGradient3d = 256; // 32 reg - 384 smem - 144 cmem
        ComputeJacGradient2d = 384; // 21 reg - 24 smem - 64 cmem
        ComputeJacGradient3d = 256; // 32 reg - 24 smem - 64 cmem
        ApproxCorrectFolding3d = 256; // 32 reg - 24 smem - 24 cmem
        CorrectFolding3d = 256; // 31 reg - 24 smem - 32 cmem
        DefFieldCompose2d = 512; // 15 reg - 24 smem - 08 cmem - 16 lmem
        DefFieldCompose3d = 384; // 21 reg - 24 smem - 08 cmem - 24 lmem
        GetJacobianMatrix = 512; // 16 reg - 24 smem - 04 cmem
        reg_voxelCentricToNodeCentric = 320; // 11 reg - 24 smem - 16 cmem
        reg_convertNmiGradientFromVoxelToRealSpace = 512; // 16 reg - 24 smem
        reg_ApplyConvolutionWindowAlongX = 512; // 14 reg - 28 smem - 08 cmem
        reg_ApplyConvolutionWindowAlongY = 512; // 14 reg - 28 smem - 08 cmem
        reg_ApplyConvolutionWindowAlongZ = 512; // 15 reg - 28 smem - 08 cmem
        Arithmetic = 384; // 5 reg - 24 smem
        NR_FUNC_CALLED();
    }
};
/* *************************************************************** */
struct BlockSize300: public BlockSize {
    BlockSize300() {
        reg_affine_getDeformationField = 1024; // 23 reg
        GetDeformationField2d = 1024; // 34 reg
        GetDeformationField3d = 1024; // 34 reg
        GetApproxJacobianValues2d = 768; // 34 reg
        GetApproxJacobianValues3d = 640; // 46 reg
        ApproxLinearEnergyGradient = 768; // 40 reg
        GetJacobianValues2d = 768; // 34 reg
        GetJacobianValues3d = 768; // 34 reg
        LogSquaredValues = 1024; // 23 reg
        ComputeApproxJacGradient2d = 768; // 34 reg
        ComputeApproxJacGradient3d = 768; // 38 reg
        ComputeJacGradient2d = 768; // 34 reg
        ComputeJacGradient3d = 768; // 37 reg
        ApproxCorrectFolding3d = 768; // 34 reg
        CorrectFolding3d = 768; // 34 reg
        DefFieldCompose2d = 1024; // 23 reg
        DefFieldCompose3d = 1024; // 24 reg
        GetJacobianMatrix = 768; // 34 reg
        reg_voxelCentricToNodeCentric = 1024; // 23 reg
        reg_convertNmiGradientFromVoxelToRealSpace = 1024; // 23 reg
        reg_ApplyConvolutionWindowAlongX = 1024; // 25 reg
        reg_ApplyConvolutionWindowAlongY = 1024; // 25 reg
        reg_ApplyConvolutionWindowAlongZ = 1024; // 25 reg
        Arithmetic = 1024;
        NR_FUNC_CALLED();
    }
};
/* *************************************************************** */
} // namespace NiftyReg
