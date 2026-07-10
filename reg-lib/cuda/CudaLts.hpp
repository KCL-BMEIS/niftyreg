#pragma once

#include "CudaCommon.hpp"
#include "_reg_blockMatching.h"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
/**
 * @brief GPU implementation of the Least Trimmed Squares affine/rigid estimation.
 *
 * Consumes the block-matching correspondences that already live on the device
 * (referencePositionCuda / warpedPositionCuda, interleaved x,y[,z] in world-mm, warped may hold
 * NaN for unmatched blocks) and updates @p transformationMatrix. Mirrors the CPU `optimize()`
 * (reg-lib/cpu/_reg_globalTrans.cpp) but keeps the iterate-trim-reestimate loop on the device so
 * the block-match -> LTS loop does not round-trip through the host every iteration.
 *
 * @param params Block-matching parameters (host struct; provides activeBlockNumber, dim,
 *               definedActiveBlockNumber, percent_to_keep, blockNumber[2] for 2D/3D).
 * @param transformationMatrix In/out affine (host mat44); used as the initial estimate and
 *               overwritten with the fitted transform.
 * @param referencePositionCuda Device pointer to the reference block positions (activeBlockNumber*dim).
 * @param warpedPositionCuda Device pointer to the matched/warped positions (activeBlockNumber*dim).
 * @param affine True for a 12-DoF affine fit, false for a 6-DoF rigid fit.
 */
void OptimizeLts(_reg_blockMatchingParam *params,
                 mat44 *transformationMatrix,
                 const float *referencePositionCuda,
                 const float *warpedPositionCuda,
                 bool affine);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
