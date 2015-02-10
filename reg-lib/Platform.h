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
class Content;

class  Platform {
public:
	Platform(int platformCode);
	virtual ~Platform();

	Kernel *createKernel(const std::string& name, Content *con) const;
	std::string getName();
	void setClIdx(int clIdxIn);


private:

	KernelFactory* factory;
	std::string platformName;

};



#endif //PLATFORM_H_
