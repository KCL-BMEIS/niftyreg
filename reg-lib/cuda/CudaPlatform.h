#ifndef CudaPLATFORM_H_
#define CudaPLATFORM_H_

#include "Context.h"
#include "Platform.h"

class CudaPlatform : public Platform
{
public:
	CudaPlatform();

	//set platform specific data to context
	void setPlatformData(Context &ctx);
	std::string getName(){ return "cuda_platform"; }
private:
	unsigned int getRank();
};
#endif //CudaPLATFORM_H_