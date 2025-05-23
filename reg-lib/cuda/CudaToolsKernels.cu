/*
 *  CudaToolsKernels.cu
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 */

#include "CudaCommon.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<bool is3d>
__device__ void VoxelCentricToNodeCentricKernel(float4 *nodeImageCuda,
                                                cudaTextureObject_t voxelImageTexture,
                                                const int3 nodeImageDims,
                                                const int3 voxelImageDims,
                                                const float weight,
                                                const mat44 transformation,
                                                const mat33 reorientation,
                                                const int index) {
    // Calculate the node coordinates
    const auto [x, y, z] = IndexToDims<is3d>(index, nodeImageDims);
    // Transform into voxel coordinates
    float voxelCoord[3], nodeCoord[3] = { static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };
    Mat44Mul<float, is3d>(transformation, nodeCoord, voxelCoord);

    // Linear interpolation
    float basisX[2], basisY[2], basisZ[2], interpolatedValue[3]{};
    const int pre[3] = { Floor<int>(voxelCoord[0]), Floor<int>(voxelCoord[1]), Floor<int>(voxelCoord[2]) };
    basisX[1] = voxelCoord[0] - static_cast<float>(pre[0]);
    basisX[0] = 1.f - basisX[1];
    basisY[1] = voxelCoord[1] - static_cast<float>(pre[1]);
    basisY[0] = 1.f - basisY[1];
    if constexpr (is3d) {
        basisZ[1] = voxelCoord[2] - static_cast<float>(pre[2]);
        basisZ[0] = 1.f - basisZ[1];
    }
    for (char c = 0; c < 2; c++) {
        const int indexZ = pre[2] + c;
        if (-1 < indexZ && indexZ < voxelImageDims.z) {
            for (char b = 0; b < 2; b++) {
                const int indexY = pre[1] + b;
                if (-1 < indexY && indexY < voxelImageDims.y) {
                    for (char a = 0; a < 2; a++) {
                        const int indexX = pre[0] + a;
                        if (-1 < indexX && indexX < voxelImageDims.x) {
                            const int index = (indexZ * voxelImageDims.y + indexY) * voxelImageDims.x + indexX;
                            float linearWeight = basisX[a] * basisY[b];
                            if constexpr (is3d) linearWeight *= basisZ[c];
                            const float4 voxelValue = tex1Dfetch<float4>(voxelImageTexture, index);
                            interpolatedValue[0] += linearWeight * voxelValue.x;
                            interpolatedValue[1] += linearWeight * voxelValue.y;
                            if constexpr (is3d)
                                interpolatedValue[2] += linearWeight * voxelValue.z;
                        }
                    }
                }
            }
        }
    }

    float reorientedValue[3];
    Mat33Mul<is3d>(reorientation, interpolatedValue, weight, reorientedValue);
    nodeImageCuda[index] = { reorientedValue[0], reorientedValue[1], reorientedValue[2], 0 };
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
