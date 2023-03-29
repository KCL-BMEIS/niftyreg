#pragma once

#include "_reg_common_cuda.h"
#include "_reg_optimiser.h"
#include "_reg_tools_gpu.h"

/* *************************************************************** */
/** @class reg_optimiser_gpu
 * @brief Standard gradient ascent optimisation for GPU
 */
class reg_optimiser_gpu: public reg_optimiser<float> {
protected:
    float4 *currentDofCuda; // pointers
    float4 *gradientCuda; // pointers
    float4 *bestDofCuda; // allocated here

public:
    reg_optimiser_gpu();
    virtual ~reg_optimiser_gpu();

    // Float4 are casted to float for compatibility with the cpu class
    virtual float* GetCurrentDof() override {
        return reinterpret_cast<float*>(this->currentDofCuda);
    }
    virtual float* GetBestDof() override {
        return reinterpret_cast<float*>(this->bestDofCuda);
    }
    virtual float* GetGradient() override {
        return reinterpret_cast<float*>(this->gradientCuda);
    }

    virtual void RestoreBestDof() override;
    virtual void StoreCurrentDof() override;

    virtual void Initialise(size_t nvox,
                            int ndim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxIt,
                            size_t start,
                            InterfaceOptimiser *intOpt,
                            float *cppData,
                            float *gradData = nullptr,
                            size_t nvoxBw = 0,
                            float *cppDataBw = nullptr,
                            float *gradDataBw = nullptr) override;
    virtual void Perturbation(float length) override;
};
/* *************************************************************** */
/** @class reg_conjugateGradient_gpu
 * @brief Conjugate gradient ascent optimisation for GPU
 */
class reg_conjugateGradient_gpu: public reg_optimiser_gpu {
protected:
    float4 *array1;
    float4 *array2;
    bool firstCall;
    void UpdateGradientValues(); /// @brief Update the gradient array

public:
    reg_conjugateGradient_gpu();
    virtual ~reg_conjugateGradient_gpu();

    virtual void Initialise(size_t nvox,
                            int ndim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxIt,
                            size_t start,
                            InterfaceOptimiser *intOpt,
                            float *cppData,
                            float *gradData = nullptr,
                            size_t nvoxBw = 0,
                            float *cppDataBw = nullptr,
                            float *gradDataBw = nullptr) override;
    virtual void Optimise(float maxLength,
                          float smallLength,
                          float &startLength) override;
    virtual void Perturbation(float length) override;
};
/* *************************************************************** */
/** @brief
 */
extern "C++"
void reg_initialiseConjugateGradient_gpu(float4 *gradientImageCuda,
                                         float4 *conjugateGCuda,
                                         float4 *conjugateHCuda,
                                         const size_t& nVoxels);
/* *************************************************************** */
/** @brief
 */
extern "C++"
void reg_GetConjugateGradient_gpu(float4 *gradientImageCuda,
                                  float4 *conjugateGCuda,
                                  float4 *conjugateHCuda,
                                  const size_t& nVoxels);
/* *************************************************************** */
/** @brief
 */
extern "C++"
void reg_updateControlPointPosition_gpu(const size_t& nVoxels,
                                        float4 *controlPointImageCuda,
                                        const float4 *bestControlPointCuda,
                                        const float4 *gradientImageCuda,
                                        const float& scale,
                                        const bool& optimiseX,
                                        const bool& optimiseY,
                                        const bool& optimiseZ);
/* *************************************************************** */
