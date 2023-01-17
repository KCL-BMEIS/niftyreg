#pragma once

#include "Compute.h"

class ComputeFactory {
public:
    virtual Compute* Produce(Content& con) { return new Compute(con); }
};
