#pragma once

#include "CudaCommon.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
/**
 * @brief Get maximal value of the gradient image
 * @param imageCuda Cuda device pointer to the gradient image
 * @param nVoxels Number of voxels in the image
 * @param optimiseX Flag to indicate if the x component of the gradient is optimised
 * @param optimiseY Flag to indicate if the y component of the gradient is optimised
 * @param optimiseZ Flag to indicate if the z component of the gradient is optimised
 * @return The maximal value of the gradient image
*/
float GetMaximalLength(const float4 *imageCuda,
                       const size_t nVoxels,
                       const bool optimiseX,
                       const bool optimiseY,
                       const bool optimiseZ);
/* *************************************************************** */
/**
 * @brief Normalise the gradient image
 * @param imageCuda Cuda device pointer to the gradient image
 * @param nVoxels Number of voxels in the image
 * @param maxGradLength The maximal value of the gradient image
 * @param optimiseX Flag to indicate if the x component of the gradient is optimised
 * @param optimiseY Flag to indicate if the y component of the gradient is optimised
 * @param optimiseZ Flag to indicate if the z component of the gradient is optimised
*/
void NormaliseGradient(float4 *imageCuda,
                       const size_t nVoxels,
                       const double maxGradLength,
                       const bool optimiseX,
                       const bool optimiseY,
                       const bool optimiseZ);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
