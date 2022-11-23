#pragma once

#include <map>
#include <string>
#include <vector>

#define NR_PLATFORM_CPU  0
#define NR_PLATFORM_CUDA 1
#define NR_PLATFORM_CL   2

class Kernel;
class KernelFactory;
class AladinContent;

class Platform {
public:
    Platform(int platformCode);
    virtual ~Platform();

    Kernel* CreateKernel(const std::string& name, AladinContent *con) const;
    std::string GetName();

    int GetPlatformCode();
    //void SetPlatformCode(const int platformCodeIn);
    void SetGpuIdx(unsigned gpuIdxIn);
    unsigned GetGpuIdx();

private:
    KernelFactory *factory;
    std::string platformName;
    int platformCode;
    unsigned gpuIdx;
};
