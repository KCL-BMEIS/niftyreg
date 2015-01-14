#ifndef CudaPLATFORM_H_
#define CudaPLATFORM_H_

#include "Content.h"
#include "Platform.h"

class CudaPlatform : public Platform
{
public:
	CudaPlatform();

	std::string getName(){ return "cuda_platform"; }

};
#endif //CudaPLATFORM_H_
