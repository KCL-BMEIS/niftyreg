#pragma once

#include "F3dContent.h"
#include "ComputeFactory.h"
#include "ContentCreatorFactory.h"
#include "KernelFactory.h"
#include "MeasureFactory.h"
#include "Optimiser.hpp"

enum class PlatformType { Cpu, Cuda, OpenCl };
constexpr PlatformType PlatformTypes[] = {
    PlatformType::Cpu,
#ifdef USE_CUDA
    PlatformType::Cuda,
#endif
#ifdef USE_OPENCL
    PlatformType::OpenCl
#endif
};

class Platform {
public:
    Platform(const PlatformType platformTypeIn);
    ~Platform();

    std::string GetName() const;
    PlatformType GetPlatformType() const;
    unsigned GetGpuIdx() const;
    void SetGpuIdx(unsigned gpuIdxIn);

    Compute* CreateCompute(Content& con) const;
    ContentCreator* CreateContentCreator(const ContentType conType = ContentType::Base) const;
    Kernel* CreateKernel(const std::string& name, Content *con) const;
    Measure* CreateMeasure() const;
    template<typename Type>
    Optimiser<Type>* CreateOptimiser(F3dContent& con,
                                     InterfaceOptimiser& opt,
                                     size_t maxIterationNumber,
                                     bool useConjGradient,
                                     bool optimiseX,
                                     bool optimiseY,
                                     bool optimiseZ,
                                     F3dContent *conBw = nullptr) const;

    static constexpr bool IsCudaEnabled() {
#ifdef USE_CUDA
        return true;
#endif
        return false;
    }
    static constexpr bool IsOpenClEnabled() {
#ifdef USE_OPENCL
        return true;
#endif
        return false;
    }

private:
    ComputeFactory *computeFactory = nullptr;
    ContentCreatorFactory *contentCreatorFactory = nullptr;
    KernelFactory *kernelFactory = nullptr;
    MeasureFactory *measureFactory = nullptr;
    std::string platformName;
    PlatformType platformType;
    unsigned gpuIdx;
};
