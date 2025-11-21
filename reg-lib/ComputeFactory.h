#pragma once

#include "Compute.h"

class ComputeFactory {
public:
    virtual ~ComputeFactory() = default;
    virtual Compute* Produce(Content& con) { return new Compute(con); }
};
