#pragma once

#include "AladinContent.h"
#include "CUDAGlobalContent.h"

#include "_reg_tools.h"

class CudaAladinContent: public AladinContent, public CudaGlobalContent {

public:
	CudaAladinContent();
    virtual ~CudaAladinContent();

    void InitBlockMatchingParams();
    virtual void ClearBlockMatchingParams();

	//device getters
	float* getReferencePosition_d();
	float* getWarpedPosition_d();
    int *getTotalBlock_d();
    float* getTransformationMatrix_d();

	//	float* getAR_d(); // Removed until CUDA SVD is added back
	//	float* getU_d(); // Removed until CUDA SVD is added back
	//	float* getVT_d(); // Removed until CUDA SVD is added back
	//	float* getSigma_d(); // Removed until CUDA SVD is added back
	//	float* getLengths_d(); // Removed until CUDA SVD is added back
	//	float* getNewWarpedPos_d(); // Removed until CUDA SVD is added back

    //setters
    virtual void setTransformationMatrix(mat44 *transformationMatrixIn);
    virtual void setTransformationMatrix(mat44 transformationMatrixIn);
    virtual void setBlockMatchingParams(_reg_blockMatchingParam* bmp);


    _reg_blockMatchingParam* getBlockMatchingParams();

protected:
    float *referencePosition_d;
    float *warpedPosition_d;
    int   *totalBlock_d;
    float* transformationMatrix_d;

    //svd
    //	float* AR_d;//A and then pseudoinverse  // Removed until CUDA SVD is added back
    //	float* U_d; // Removed until CUDA SVD is added back
    //	float* VT_d; // Removed until CUDA SVD is added back
    //	float* Sigma_d; // Removed until CUDA SVD is added back
    //	float* lengths_d; // Removed until CUDA SVD is added back
    //	float* newWarpedPos_d; // Removed until CUDA SVD is added back

private:
	//void uploadAladinContent();
    //void allocateCuPtrs();
	void freeCuPtrs();

};
