#pragma once

#include "F3dContent.h"
#include "KernelFactory.h"
#include "ComputeFactory.h"
#include "MeasureFactory.h"
#include "_reg_optimiser.h"

enum class PlatformType { Cpu, Cuda, OpenCl };

class Platform {
public:
    Platform(const PlatformType& platformTypeIn);
    virtual ~Platform();

    Compute* CreateCompute(Content& con) const;
    Kernel* CreateKernel(const std::string& name, Content *con) const;
    template<typename Type>
    reg_optimiser<Type>* CreateOptimiser(F3dContent& con,
                                         InterfaceOptimiser& opt,
                                         size_t maxIterationNumber,
                                         bool useConjGradient,
                                         bool optimiseX,
                                         bool optimiseY,
                                         bool optimiseZ) const;
    Measure* CreateMeasure() const;

    std::string GetName() const;
    PlatformType GetPlatformType() const;
    void SetGpuIdx(unsigned gpuIdxIn);
    unsigned int GetGpuIdx() const;

private:
    KernelFactory *kernelFactory = nullptr;
    ComputeFactory *computeFactory = nullptr;
    MeasureFactory *measureFactory = nullptr;
    std::string platformName;
    PlatformType platformType;
    unsigned gpuIdx;
};
