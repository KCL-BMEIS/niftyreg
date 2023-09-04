#pragma once

#include "CudaCommon.hpp"
#include "_reg_optimiser.h"
#include "_reg_tools_gpu.h"

/* *************************************************************** */
/** @class reg_optimiser_gpu
 * @brief Standard gradient ascent optimisation for GPU
 */
class reg_optimiser_gpu: public reg_optimiser<float> {
protected:
    float4 *currentDofCuda, *currentDofBwCuda;
    float4 *bestDofCuda, *bestDofBwCuda;
    float4 *gradientCuda, *gradientBwCuda;

public:
    reg_optimiser_gpu();
    virtual ~reg_optimiser_gpu();
    virtual void StoreCurrentDof() override;
    virtual void RestoreBestDof() override;

    // float4s are casted to floats for compatibility with the CPU class
    virtual float* GetCurrentDof() override {
        return reinterpret_cast<float*>(this->currentDofCuda);
    }
    virtual float* GetCurrentDofBw() override {
        return reinterpret_cast<float*>(this->currentDofBwCuda);
    }
    virtual float* GetBestDof() override {
        return reinterpret_cast<float*>(this->bestDofCuda);
    }
    virtual float* GetBestDofBw() override {
        return reinterpret_cast<float*>(this->bestDofBwCuda);
    }
    virtual float* GetGradient() override {
        return reinterpret_cast<float*>(this->gradientCuda);
    }
    virtual float* GetGradientBw() override {
        return reinterpret_cast<float*>(this->gradientBwCuda);
    }

    virtual void Initialise(size_t nvox,
                            int ndim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxIt,
                            size_t startIt,
                            InterfaceOptimiser *intOpt,
                            float *cppData,
                            float *gradData,
                            size_t nvoxBw,
                            float *cppDataBw,
                            float *gradDataBw) override;
    virtual void Perturbation(float length) override;
};
/* *************************************************************** */
/** @class reg_conjugateGradient_gpu
 * @brief Conjugate gradient ascent optimisation for GPU
 */
class reg_conjugateGradient_gpu: public reg_optimiser_gpu {
protected:
    float4 *array1, *array1Bw;
    float4 *array2, *array2Bw;
    bool firstCall;

#ifdef NR_TESTING
public:
#endif
    virtual void UpdateGradientValues() override;

public:
    reg_conjugateGradient_gpu();
    virtual ~reg_conjugateGradient_gpu();

    virtual void Initialise(size_t nvox,
                            int ndim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxIt,
                            size_t startIt,
                            InterfaceOptimiser *intOpt,
                            float *cppData,
                            float *gradData,
                            size_t nvoxBw,
                            float *cppDataBw,
                            float *gradDataBw) override;
    virtual void Optimise(float maxLength,
                          float smallLength,
                          float& startLength) override;
    virtual void Perturbation(float length) override;
};
/* *************************************************************** */
void reg_initialiseConjugateGradient_gpu(float4 *gradientImageCuda,
                                         float4 *conjugateGCuda,
                                         float4 *conjugateHCuda,
                                         const size_t nVoxels);
/* *************************************************************** */
void reg_getConjugateGradient_gpu(float4 *gradientImageCuda,
                                  float4 *conjugateGCuda,
                                  float4 *conjugateHCuda,
                                  const size_t nVoxels,
                                  const bool isSymmetric,
                                  float4 *gradientImageBwCuda,
                                  float4 *conjugateGBwCuda,
                                  float4 *conjugateHBwCuda,
                                  const size_t nVoxelsBw);
/* *************************************************************** */
void reg_updateControlPointPosition_gpu(const size_t nVoxels,
                                        float4 *controlPointImageCuda,
                                        const float4 *bestControlPointCuda,
                                        const float4 *gradientImageCuda,
                                        const float scale,
                                        const bool optimiseX,
                                        const bool optimiseY,
                                        const bool optimiseZ);
/* *************************************************************** */
