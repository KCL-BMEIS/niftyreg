#pragma once

#include <cuda.h>
#include "_reg_tools.h"
#include "BlockSize.hpp"

namespace NiftyReg {
/* *************************************************************** */
class CudaContext {
public:
    CudaContext(CudaContext const&) = delete;
    void operator=(CudaContext const&) = delete;

    static CudaContext& GetInstance() {
        // Instantiated on first use.
        static CudaContext instance; // Guaranteed to be destroyed.
        return instance;
    }

    static const BlockSize* GetBlockSize() {
        return GetInstance().blockSize.get();
    }

    void SetCudaIdx(unsigned cudaIdxIn);
    CUcontext GetContext();
    bool IsCardDoubleCapable();

private:
    CudaContext();
    ~CudaContext();

    bool isCardDoubleCapable;
    CUcontext cudaContext;
    unsigned numDevices;
    unsigned cudaIdx;
    unique_ptr<BlockSize> blockSize;

    void PickCard(unsigned deviceId);
    void SetBlockSize(int major);
};
/* *************************************************************** */
} // namespace NiftyReg
