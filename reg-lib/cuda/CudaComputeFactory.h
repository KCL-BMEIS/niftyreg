#pragma once

#include "ComputeFactory.h"
#include "CudaCompute.h"

class CudaComputeFactory: public ComputeFactory {
public:
    virtual Compute* Produce(Content& con) override { return new CudaCompute(con); }
};
