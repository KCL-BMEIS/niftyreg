#pragma once

#include "F3dContent.h"
#include "KernelFactory.h"
#include "ComputeFactory.h"
#include "_reg_optimiser.h"

enum class PlatformType { Cpu, Cuda, OpenCl };

class Platform {
public:
    Platform(const PlatformType& platformTypeIn);
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

    PlatformType GetPlatformType();
    //void SetPlatformType(const PlatformType& platformTypeIn);
    void SetGpuIdx(unsigned gpuIdxIn);
    unsigned GetGpuIdx();

private:
    KernelFactory *kernelFactory;
    ComputeFactory *computeFactory;
    std::string platformName;
    PlatformType platformType;
    unsigned gpuIdx;
};
