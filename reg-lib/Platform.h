#pragma once

#include "F3dContent.h"
#include "ComputeFactory.h"
#include "ContentCreatorFactory.h"
#include "KernelFactory.h"
#include "MeasureFactory.h"
#include "_reg_optimiser.h"

enum class PlatformType { Cpu, Cuda, OpenCl };

class Platform {
public:
    Platform(const PlatformType& platformTypeIn);
    ~Platform();

    std::string GetName() const;
    PlatformType GetPlatformType() const;
    unsigned int GetGpuIdx() const;
    void SetGpuIdx(unsigned gpuIdxIn);

    Compute* CreateCompute(Content& con) const;
    ContentCreator* CreateContentCreator(const ContentType& conType = ContentType::Base) const;
    Kernel* CreateKernel(const std::string& name, Content *con) const;
    Measure* CreateMeasure() const;
    template<typename Type>
    reg_optimiser<Type>* CreateOptimiser(F3dContent& con,
                                         InterfaceOptimiser& opt,
                                         size_t maxIterationNumber,
                                         bool useConjGradient,
                                         bool optimiseX,
                                         bool optimiseY,
                                         bool optimiseZ,
                                         F3dContent *conBw = nullptr) const;

private:
    ComputeFactory *computeFactory = nullptr;
    ContentCreatorFactory *contentCreatorFactory = nullptr;
    KernelFactory *kernelFactory = nullptr;
    MeasureFactory *measureFactory = nullptr;
    std::string platformName;
    PlatformType platformType;
    unsigned gpuIdx;
};
