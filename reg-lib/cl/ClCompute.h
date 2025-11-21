#pragma once

#include "Compute.h"

class ClCompute: public Compute {
public:
    ClCompute(Content& con): Compute(con) {}

    virtual void ResampleImage(int inter, float paddingValue) override;
};
