#pragma once

#include "F3dContent.h"
#include "KernelFactory.h"
#include "CpuKernelFactory.h"
#include "ComputeFactory.h"
#include "_reg_optimiser.h"
#ifdef _USE_CUDA
#include "CudaF3dContent.h"
#include "CudaKernelFactory.h"
#include "CudaComputeFactory.h"
#include "CudaContextSingleton.h"
#include "_reg_optimiser_gpu.h"
#endif
#ifdef _USE_OPENCL
#include "ClKernelFactory.h"
#include "ClComputeFactory.h"
#include "ClContextSingleton.h"
#endif

#define NR_PLATFORM_CPU  0
#define NR_PLATFORM_CUDA 1
#define NR_PLATFORM_CL   2

class Platform {
public:
    Platform(int platformCodeIn);
    virtual ~Platform();

    Compute* CreateCompute(Content *con) const;
    Kernel* CreateKernel(const std::string& name, Content *con) const;
    template<typename Type>
    reg_optimiser<Type>* CreateOptimiser(F3dContent *con,
                                         InterfaceOptimiser *opt,
                                         size_t maxIterationNumber,
                                         bool useConjGradient,
                                         bool optimiseX,
                                         bool optimiseY,
                                         bool optimiseZ);

    std::string GetName();

    int GetPlatformCode();
    //void SetPlatformCode(const int platformCodeIn);
    void SetGpuIdx(unsigned gpuIdxIn);
    unsigned GetGpuIdx();

private:
    KernelFactory *kernelFactory;
    ComputeFactory *computeFactory;
    std::string platformName;
    int platformCode;
    unsigned gpuIdx;
};
