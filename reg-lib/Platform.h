#ifndef PLATFORM_H_
#define PLATFORM_H_

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

    Kernel *createKernel(const std::string& name, AladinContent *con) const;
    std::string getName();

    int getPlatformCode();
    //void setPlatformCode(const int platformCodeIn);
    void setGpuIdx(unsigned gpuIdxIn);
    unsigned getGpuIdx();

private:
    KernelFactory* factory;
    std::string platformName;
    int platformCode;
    unsigned gpuIdx;
};



#endif //PLATFORM_H_
