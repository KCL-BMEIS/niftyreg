#pragma once

#include "ComputeFactory.h"
#include "ClCompute.h"

class ClComputeFactory: public ComputeFactory {
public:
    virtual Compute* Produce(Content& con) override { return new ClCompute(con); }
};
